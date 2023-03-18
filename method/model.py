import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict
from method.model_components import BertAttention, LinearLayer, BertSelfAttention, TrainablePositionalEncoding
from method.model_components import clip_nce, frame_nce

from gadgets.match import HungarianMatcher

class BGM_Net(nn.Module):
    def __init__(self, config):
        super(BGM_Net, self).__init__()
        self.config = config

        self.query_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_desc_l,
                                                           hidden_size=config.hidden_size, dropout=config.input_drop)
        self.clip_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                         hidden_size=config.hidden_size, dropout=config.input_drop)
        self.frame_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                          hidden_size=config.hidden_size, dropout=config.input_drop)
        #
        self.query_input_proj = LinearLayer(config.query_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        #
        self.query_encoder = BertAttention(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop))

        self.matcher = HungarianMatcher()
        self.clip_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        self.clip_encoder = copy.deepcopy(self.query_encoder)

        self.frame_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,
                                             dropout=config.input_drop, relu=True)
        self.frame_encoder = copy.deepcopy(self.query_encoder)

        self.modular_vector_mapping = nn.Linear(config.hidden_size, out_features=1, bias=False)

        
        self.pool_layers = nn.ModuleList([nn.Identity()]
                                         + [nn.AvgPool1d(i, stride=1) for i in range(2, config.map_size + 1)]
                                         )

        self.mapping_linear = nn.ModuleList([nn.Linear(config.hidden_size, out_features=config.hidden_size, bias=False)
                                             for i in range(2)])

        self.clip_nce_criterion = clip_nce(reduction='mean')
        self.video_nce_criterion = frame_nce(reduction='mean')


        self.reset_parameters()

    def reset_parameters(self):
        """ Initialize the weights."""

        def re_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv1d):
                module.reset_parameters()
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(re_init)

    def set_hard_negative(self, use_hard_negative, hard_pool_size):
        """use_hard_negative: bool; hard_pool_size: int, """
        self.config.use_hard_negative = use_hard_negative
        self.config.hard_pool_size = hard_pool_size
    
    def set_use_matcher(self, use_matcher):
        """use_matcher: bool"""
        self.config.use_matcher = use_matcher



    def forward(self, clip_video_feat, frame_video_feat, frame_video_mask, query_feat, query_mask, query_labels):

        # (N=128, L_frame=102, 384),  # (N=128, #clips=528, 384), note: 528=u*(u+1)/2 = 32 * 33 / 2
        encoded_frame_feat, vid_proposal_feat = self.encode_context(
            clip_video_feat, frame_video_feat, frame_video_mask)
        
        use_matcher = self.config.use_matcher
        
        # (#caption=640, B=128),  (640, 640),    (640, 128),          (640, 640)
        # first two normalized(cosine similarity, [-1, 1]), last two unnormalized. (q1*q2)
        # q2q_score, clip_v2v_score, frame_v2v_score: (640, 640)
        clip_scale_scores, frame_scale_scores, clip_scale_scores_, frame_scale_scores_, \
            q2q_score, clip_v2v_score, frame_v2v_score \
            = self.get_pred_from_raw_query(
            query_feat, query_mask, query_labels, vid_proposal_feat, encoded_frame_feat, frame_video_mask,
            is_train=True, use_matcher=use_matcher)

        label_dict = {}     # index_in_batch : [index in label from 0~#captions-1]
        for index, label in enumerate(query_labels):
            if label in label_dict:
                label_dict[label].append(index)
            else:
                label_dict[label] = []
                label_dict[label].append(index)


        # if use_matcher:        
        #     clip_nce_loss = 0.02 * self.video_nce_criterion(clip_scale_scores_)
        #     clip_trip_loss = self.get_frame_trip_loss(clip_scale_scores)


        #     frame_nce_loss = 0.04 * self.video_nce_criterion(frame_scale_scores_)
        #     frame_trip_loss = self.get_frame_trip_loss(frame_scale_scores)
        #else:
        clip_nce_loss = 0.02 * self.clip_nce_criterion(query_labels, label_dict, clip_scale_scores_)
        clip_trip_loss = self.get_clip_triplet_loss(clip_scale_scores, query_labels)


        frame_nce_loss = 0.04 * self.video_nce_criterion(frame_scale_scores_)
        frame_trip_loss = self.get_frame_trip_loss(frame_scale_scores)

        # add alignment loss to align v_clip and v_frame, note different scale of losses.
        clip_align_loss = torch.nn.functional.l1_loss(q2q_score, clip_v2v_score)
        #frame_align_loss = torch.nn.functional.l1_loss(q2q_score, frame_v2v_score)
        
        loss = clip_nce_loss + frame_nce_loss + clip_trip_loss + frame_trip_loss + \
               clip_align_loss #+ frame_align_loss
        #loss = clip_trip_loss + frame_trip_loss
        #loss = clip_nce_loss  + frame_nce_loss + clip_align_loss
        
        return loss, {"loss_overall": float(loss), 'clip_nce_loss': clip_nce_loss,
                      'clip_trip_loss': clip_trip_loss, 'frame_trip_loss': frame_trip_loss,
                      'frame_nce_loss': frame_nce_loss, 
                      'clip_align_loss': clip_align_loss, }#'frame_align_loss': frame_align_loss}



    def encode_query(self, query_feat, query_mask):
        encoded_query = self.encode_input(query_feat, query_mask, self.query_input_proj, self.query_encoder,
                                          self.query_pos_embed)  # (N, Lq, D)
        video_query = self.get_modularized_queries(encoded_query, query_mask)  # (N, D) * 1

        return video_query

    def encode_context(self, clip_video_feat, frame_video_feat, video_mask=None):
        #   corresponds to the first step of the two branch(clip & frame)
        #   FC + transformer embedding
        encoded_clip_feat = self.encode_input(clip_video_feat, None, self.clip_input_proj, self.clip_encoder,
                                               self.clip_pos_embed) # (N=128, L=32, D_hid=384)

        encoded_frame_feat = self.encode_input(frame_video_feat, video_mask, self.frame_input_proj,
                                                self.frame_encoder,
                                                self.frame_pos_embed)  # (N=128, L=102, D_hid=384)
        # corresponds to 'clip construction'
        vid_proposal_feat_map = self.encode_feat_map(encoded_clip_feat)


        return encoded_frame_feat, \
               vid_proposal_feat_map


    def encode_feat_map(self, x_feat):

        pool_in = x_feat.permute(0, 2, 1)       # (N, D, L)

        proposal_feat_map = []
        for idx, pool in enumerate(self.pool_layers):
            x = pool(pool_in).permute(0, 2, 1)
            proposal_feat_map.append(x)
        proposal_feat_map = torch.cat(proposal_feat_map, dim=1)


        return proposal_feat_map


    @staticmethod
    def encode_input(feat, mask, input_proj_layer, encoder_layer, pos_embed_layer):
        """
        Args:
            feat: (N, L, D_input), torch.float32
            mask: (N, L), torch.float32, with 1 indicates valid query, 0 indicates mask
            input_proj_layer: down project input
            encoder_layer: encoder layer
            pos_embed_layer: positional embedding layer
        """
        feat = input_proj_layer(feat)
        feat = pos_embed_layer(feat)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (N, 1, L), torch.FloatTensor    <<-- BertSelfAttention layer needs attention mask 
                                                                    #   of shape (N, Lq, L). So repeat on query dimension.
        return encoder_layer(feat, mask)  # (N, L, D_hidden)

    def get_modularized_queries(self, encoded_query, query_mask):
        """
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
            return_modular_att: bool
        """
        modular_attention_scores = self.modular_vector_mapping(encoded_query)  # (N, L, 2 or 1)
        modular_attention_scores = F.softmax(mask_logits(modular_attention_scores, query_mask.unsqueeze(2)), dim=1)
        modular_queries = torch.einsum("blm,bld->bmd", modular_attention_scores, encoded_query)  # (N, 2 or 1, D)
        return modular_queries.squeeze()

    @staticmethod
    def modify_clip_scores(clip_scores, pair_scores, query_labels):
        # clip_scores: (#query, bs), src:pair_scores: (#query), index:query_labels: (#query)
        index = torch.LongTensor(query_labels).unsqueeze(-1).to(clip_scores.device)    # (#query, 1)
        src = pair_scores.unsqueeze(-1) # (#query, 1)
        modify_scores = clip_scores.scatter_(1, index, src)
        
        return modify_scores
    @staticmethod
    def calc_clip_scale_scores(modularied_query, context_feat, norm=True):
        
        if norm:
            # input: (#caption, D), (N, #clips=L*(L+1)/2, D)
            modularied_query = F.normalize(modularied_query, dim=-1)
            context_feat = F.normalize(context_feat, dim=-1)
        # (bs, #clips, #query)    
        clip_level_query_context_scores = torch.matmul(context_feat, modularied_query.t())
        
        return clip_level_query_context_scores
    
    @staticmethod
    def get_clip_scale_scores(modularied_query, context_feat):
        # input: (#caption, D), (N, #clips=L*(L+1)/2, D)
        modularied_query = F.normalize(modularied_query, dim=-1)
        context_feat = F.normalize(context_feat, dim=-1)

        clip_level_query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0)

        query_context_scores, indices = torch.max(clip_level_query_context_scores,
                                                  dim=1)  # (N, N) diagonal positions are positive pairs
        return query_context_scores, indices

    

    @staticmethod
    def get_unnormalized_clip_scale_scores(modularied_query, context_feat):

        query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0)

        query_context_scores, _ = torch.max(query_context_scores, dim=1)

        return query_context_scores

    
    def matcher_key_clip_guided_attention(self, frame_feat, feat_mask, clip_feats, query_labels):
        #input: (128, 102, 384), (128, 102), (#cap=640, D=384), list:(640)
        # (640, 102, 384)
        expand_frame_feat = frame_feat[query_labels]

        key = self.mapping_linear[0](expand_frame_feat) # (640, 102, 384)
        query = clip_feats.unsqueeze(-1) # (640,384,1)
        value = self.mapping_linear[1](expand_frame_feat)   # (640, 102, 384)

        if feat_mask is not None:
            expand_feat_mask = feat_mask[query_labels]  # (640, 102)
            scores = torch.bmm(key, query).squeeze()    # (640, 102)
            masked_scores = scores.masked_fill(expand_feat_mask.eq(0), -1e9).unsqueeze(1)   # (640, 1, 102)
            masked_scores = nn.Softmax(dim=-1)(masked_scores)           # (640, 1, 102)
            attention_feat = torch.bmm(masked_scores, value).squeeze()  # (640, 384)
        else:
            scores = nn.Softmax(dim=-1)(torch.bmm(key, query).transpose(1, 2))
            attention_feat = torch.bmm(scores, value).squeeze()

        return attention_feat
    
    def key_clip_guided_attention(self, frame_feat, proposal_feat, feat_mask, max_index, query_labels):
        #input: (128, 102, 384), (128, 528, 384), (128, 102), (#cap=640, N=128), list:(640)
        # (640)
        selected_max_index = max_index[[i for i in range(max_index.shape[0])], query_labels]
        # (640, 102, 384)
        expand_frame_feat = frame_feat[query_labels]
        # (640, 528, 384)
        expand_proposal_feat = proposal_feat[query_labels]

        key = self.mapping_linear[0](expand_frame_feat) # (640, 102, 384)
        query = expand_proposal_feat[[i for i in range(key.shape[0])], selected_max_index, :].unsqueeze(-1) # (640,384,1)
        value = self.mapping_linear[1](expand_frame_feat)   # (640, 102, 384)

        if feat_mask is not None:
            expand_feat_mask = feat_mask[query_labels]  # (640, 102)
            scores = torch.bmm(key, query).squeeze()    # (640, 102)
            masked_scores = scores.masked_fill(expand_feat_mask.eq(0), -1e9).unsqueeze(1)   # (640, 1, 102)
            masked_scores = nn.Softmax(dim=-1)(masked_scores)           # (640, 1, 102)
            attention_feat = torch.bmm(masked_scores, value).squeeze()  # (640, 384)
        else:
            scores = nn.Softmax(dim=-1)(torch.bmm(key, query).transpose(1, 2))
            attention_feat = torch.bmm(scores, value).squeeze()

        return attention_feat

    def key_clip_guided_attention_in_inference(self, frame_feat, proposal_feat, feat_mask, max_index):
        key = self.mapping_linear[0](frame_feat)
        value = self.mapping_linear[1](frame_feat)
        num_vid = frame_feat.shape[0]

        index = torch.arange(num_vid).unsqueeze(1)
        query = proposal_feat[index, max_index.t()]
        if feat_mask is not None:
            scores = torch.bmm(key, query.transpose(2, 1))
            masked_scores = scores.masked_fill(feat_mask.unsqueeze(-1).eq(0), -1e9)
            masked_scores = nn.Softmax(dim=1)(masked_scores)
            attention_feat = torch.bmm(masked_scores.transpose(1, 2), value)
        else:
            scores = torch.bmm(key, query.transpose(2, 1))
            scores = nn.Softmax(dim=1)(scores)
            attention_feat = torch.bmm(scores.transpose(1, 2), value)

        return attention_feat

    @staticmethod
    def get_match_clips(matcher, video_proposal_feat, sim_scores, query_labels):
        '''
            Args:
                input: 
                    matcher: input logits:[num_captions, batch_size, num_moments] & query_labels: [num_cap]
                                output indices, len=bs, each tuple (query_indice, moment_indice), 
                                                        len(q_ind)=#query of that video in the batch
                    video_proposal_feat: (bs, #clips, D)
            Return:
                match_indices:     list of len(bs),   each a tuple(i, j), len(i)=len(j)=#cap of the video
        '''
        # (bs, #clips, #query)  => (#query, bs, #clips)
        sim_scores = sim_scores.permute(2, 0, 1)
        match_indices = matcher(sim_scores, query_labels)
        
        match_clip_feats = []
        for i in range(len(match_indices)):
            q_ind, clip_ind = match_indices[i]
            match_clip_feats.append(video_proposal_feat[i, clip_ind])    # (#cap of that video, D)
        match_clip_feats = torch.cat(match_clip_feats, dim=0)   # (#cap, D)
        return match_clip_feats
    

    def get_pred_from_raw_query(self, query_feat, query_mask, query_labels=None,
                                video_proposal_feat=None,
                                video_feat=None,
                                video_feat_mask=None,
                                is_train=False,
                                use_matcher=False):     
        '''
        Input:
            video_query: query_embedding, (#cap, D)
            query_labels: list:[len=#cap]
            video_proposal_feat: clip embedding, (bs, #clips, D)
            video_feat: frame embedding, (bs, max_clip_num in batch, D)
            video_feat_mask: frame mask, (bs, max_clip_num in batch)
            is_train: model in training or inference mode
            use_matcher: use Hungarian Algorithm to get mapping.
        Mid:
             clip_scale_scores: (#cap=640, bs=128)
             key_clip_indices:(#cap=640, bs=128)
        '''
        # get sentence-level embedding from word embedding
        video_query = self.encode_query(query_feat, query_mask)     # (#caption=640, D=384)
        
        if is_train:
            # Hungarian matching
            if use_matcher:
                # (bs, #clips, #query) 
                sim_scores = self.calc_clip_scale_scores(video_query, video_proposal_feat, norm=True)
                sim_scores_ = self.calc_clip_scale_scores(video_query, video_proposal_feat, norm=False)
                # (#query, D)
                clip_feats = self.get_match_clips(self.matcher, video_proposal_feat, sim_scores, query_labels)
                
                # (bs, #query)
                q2v_max_scores, _ = torch.max(sim_scores, dim=1)
                q2v_max_scores_, _ = torch.max(sim_scores_, dim=1)
                # (#query, bs)
                q2v_max_scores = q2v_max_scores.transpose(0, 1)
                q2v_max_scores_ = q2v_max_scores_.transpose(0, 1)
                
                # sim. between matching pairs (q, v),  (#query)
                pair_score = torch.mul(F.normalize(video_query, dim=-1), F.normalize(clip_feats, dim=-1)).sum(dim=-1)
                pair_score_ = torch.mul(video_query, clip_feats).sum(dim=-1)
                
                # (640, 128)
                clip_scale_scores = self.modify_clip_scores(q2v_max_scores, pair_score, query_labels)
                clip_scale_scores_ = self.modify_clip_scores(q2v_max_scores_, pair_score_, query_labels)
                
                # # (#cap, #cap). first dim for video, second for query. positive pairs on diagonal
                # # (640, 640)
                # clip_scale_scores = self.calc_clip_scale_scores(video_query, clip_feats, norm=True)
                # clip_scale_scores_ = self.calc_clip_scale_scores(video_query, clip_feats, norm=False)
                # (640, 384)    corresponds to key_clip & frame set attention, get embedding 'r' in the paper.(Fig.4) 
                frame_scale_feat = self.matcher_key_clip_guided_attention(video_feat, video_feat_mask,
                                                            clip_feats, query_labels)
                # (640, 640)
                frame_scale_scores = torch.matmul(F.normalize(video_query, dim=-1),
                                                F.normalize(frame_scale_feat, dim=-1).t())
                # (640, 640)    unnormalized scores
                frame_scale_scores_ = torch.matmul(video_query, frame_scale_feat.t())
                
                q2q_score = torch.matmul(F.normalize(video_query, dim=-1),
                                        F.normalize(video_query, dim=-1).t())
                clip_v2v_score = torch.matmul(F.normalize(clip_feats, dim=-1),
                                        F.normalize(clip_feats, dim=-1).t())
                frame_v2v_score = torch.matmul(F.normalize(frame_scale_feat, dim=-1),
                                        F.normalize(frame_scale_feat, dim=-1).t())
                
                return clip_scale_scores, frame_scale_scores, clip_scale_scores_,frame_scale_scores_, \
                    q2q_score, clip_v2v_score, frame_v2v_score
            # Maximum matching (Greedy Search)
            else:
                # get clip-level retrieval scores
                # score for the key clip: (#caption=640, N=128),      indices for key clip:(640, 128)
                clip_scale_scores, key_clip_indices = self.get_clip_scale_scores(
                    video_query, video_proposal_feat)   # (#caption, D_text),       # (N, #clips, D_vid)
                
                # (640, 384)    corresponds to key_clip & frame set attention, get embedding 'r' in the paper.(Fig.4) 
                frame_scale_feat = self.key_clip_guided_attention(video_feat, video_proposal_feat, video_feat_mask,
                                                            key_clip_indices, query_labels)
                # (640, 640)
                frame_scale_scores = torch.matmul(F.normalize(video_query, dim=-1),
                                                F.normalize(frame_scale_feat, dim=-1).t())
                # (640, 128)    unnormalized scores
                clip_scale_scores_ = self.get_unnormalized_clip_scale_scores(video_query, video_proposal_feat)
                # (640, 640)    unnormalized scores
                frame_scale_scores_ = torch.matmul(video_query, frame_scale_feat.t())

                # (640, 640)   q2q score
                q2q_score = torch.matmul(F.normalize(video_query, dim=-1),
                                    F.normalize(video_query, dim=-1).t())
                
                q_indices = torch.arange(len(video_query)).to(video_proposal_feat.device)
                max_clip_indices = key_clip_indices[q_indices, query_labels]    # 640
                _clip_feats = video_proposal_feat[query_labels, max_clip_indices]   # (640, D)
                
                clip_v2v_score = torch.matmul(F.normalize(_clip_feats, dim=-1),
                                    F.normalize(_clip_feats, dim=-1).t())
                
                frame_v2v_score = torch.matmul(F.normalize(frame_scale_feat, dim=-1),
                                            F.normalize(frame_scale_feat, dim=-1).t())
                
                return clip_scale_scores, frame_scale_scores, clip_scale_scores_,frame_scale_scores_, \
                    q2q_score, clip_v2v_score, frame_v2v_score
               
        # inference part
        else:
            # get clip-level retrieval scores
            # score for the key clip: (#caption=640, N=128),      indices for key clip:(640, 128)
            clip_scale_scores, key_clip_indices = self.get_clip_scale_scores(
                video_query, video_proposal_feat)   # (#caption, D_text),       # (N, #clips, D_vid)
            frame_scale_feat = self.key_clip_guided_attention_in_inference(video_feat, video_proposal_feat, video_feat_mask,
                                                                       key_clip_indices).to(video_query.device)
            frame_scales_cores_ = torch.mul(F.normalize(frame_scale_feat, dim=-1),
                                            F.normalize(video_query, dim=-1).unsqueeze(0))
            frame_scale_scores = torch.sum(frame_scales_cores_, dim=-1).transpose(1, 0)

            return clip_scale_scores, frame_scale_scores, key_clip_indices

    def get_clip_triplet_loss(self, query_context_scores, labels):
        v2t_scores = query_context_scores.t()       # (128, 640)
        t2v_scores = query_context_scores           # (640, 128)
        labels = np.array(labels)

        # cal_v2t_loss
        v2t_loss = 0
        for i in range(v2t_scores.shape[0]):
            pos_pair_scores = torch.mean(v2t_scores[i][np.where(labels == i)])      # one video may correspond to multiple caption
                                                                                    # take average of the scores

            neg_pair_scores, _ = torch.sort(v2t_scores[i][np.where(labels != i)[0]], descending=True)   # (#cap - #current_video_cap)
            if self.config.use_hard_negative:               # take the highest neg score
                sample_neg_pair_scores = neg_pair_scores[0]
            else:       # sample one neg score
                v2t_sample_max_idx = neg_pair_scores.shape[0]
                sample_neg_pair_scores = neg_pair_scores[
                    torch.randint(0, v2t_sample_max_idx, size=(1,)).to(v2t_scores.device)]


            v2t_loss += (self.config.margin + sample_neg_pair_scores - pos_pair_scores).clamp(min=0).sum()

        # cal_t2v_loss
        text_indices = torch.arange(t2v_scores.shape[0]).to(t2v_scores.device)  # (0,1,2,...,640-1)
        t2v_pos_scores = t2v_scores[text_indices, labels]       # (640)
        mask_score = copy.deepcopy(t2v_scores.data)
        mask_score[text_indices, labels] = 999
        _, sorted_scores_indices = torch.sort(mask_score, descending=True, dim=1)       # (640, 128)
        t2v_sample_max_idx = min(1 + self.config.hard_pool_size,
                                 t2v_scores.shape[1]) if self.config.use_hard_negative else t2v_scores.shape[1]
        sample_indices = sorted_scores_indices[     # sample one negative sample    (640)
            text_indices, torch.randint(1, t2v_sample_max_idx, size=(t2v_scores.shape[0],)).to(t2v_scores.device)]

        t2v_neg_scores = t2v_scores[text_indices, sample_indices]   # 640

        t2v_loss = (self.config.margin + t2v_neg_scores - t2v_pos_scores).clamp(min=0)
        # first term equals to t2v_loss.mean()
        return t2v_loss.sum() / len(t2v_scores) + v2t_loss / len(v2t_scores)

    def get_frame_trip_loss(self, query_context_scores):
        """ ranking loss between (pos. query + pos. video) and (pos. query + neg. video) or (neg. query + pos. video)
        Args:
            query_context_scores: (N, N), cosine similarity [-1, 1],
                Each row contains the scores between the query to each of the videos inside the batch.
        """
        # (#caps=640, 640)
        bsz = len(query_context_scores)

        diagonal_indices = torch.arange(bsz).to(query_context_scores.device)   # positive pairs
        pos_scores = query_context_scores[diagonal_indices, diagonal_indices]  # (N, )
        query_context_scores_masked = copy.deepcopy(query_context_scores.data)
        # impossibly large for cosine similarity, the copy is created as modifying the original will cause error
        query_context_scores_masked[diagonal_indices, diagonal_indices] = 999
        pos_query_neg_context_scores = self.get_neg_scores(query_context_scores, query_context_scores_masked)   # (640)
        neg_query_pos_context_scores = self.get_neg_scores(query_context_scores.transpose(0, 1),                # (640)
                                                           query_context_scores_masked.transpose(0, 1))
        loss_neg_ctx = self.get_ranking_loss(pos_scores, pos_query_neg_context_scores)
        loss_neg_q = self.get_ranking_loss(pos_scores, neg_query_pos_context_scores)
        return loss_neg_ctx + loss_neg_q

    def get_neg_scores(self, scores, scores_masked):
        """
        scores: (N, N), cosine similarity [-1, 1],
            Each row are scores: query --> all videos. Transposed version: video --> all queries.
        scores_masked: (N, N) the same as scores, except that the diagonal (positive) positions
            are masked with a large value.
        """

        bsz = len(scores)
        batch_indices = torch.arange(bsz).to(scores.device)

        _, sorted_scores_indices = torch.sort(scores_masked, descending=True, dim=1)

        sample_min_idx = 1  # skip the masked positive

        sample_max_idx = min(sample_min_idx + self.config.hard_pool_size, bsz) if self.config.use_hard_negative else bsz

        # sample_max_idx = 2

        # (N, )
        sampled_neg_score_indices = sorted_scores_indices[batch_indices, torch.randint(sample_min_idx, sample_max_idx,
                                                                                       size=(bsz,)).to(scores.device)]

        sampled_neg_scores = scores[batch_indices, sampled_neg_score_indices]  # (N, )
        return sampled_neg_scores

    def get_ranking_loss(self, pos_score, neg_score):
        """ Note here we encourage positive scores to be larger than negative scores.
        Args:
            pos_score: (N, ), torch.float32
            neg_score: (N, ), torch.float32
        """
        return torch.clamp(self.config.margin + neg_score - pos_score, min=0).sum() / len(pos_score)



def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)
