import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import logging
import torch.backends.cudnn as cudnn
import pickle
from method.model import BGM_Net
from torch.utils.data import DataLoader
from method.data_provider import BasicDataset,VisDataSet,\
    TxtDataSet,read_video_ids, collate_frame_val, collate_text_val
from tqdm import tqdm
from collections import defaultdict
import torch
from utils.basic_utils import AverageMeter, BigFile, read_dict
from method.config import TestOptions
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def ap_score(sorted_labels):
    nr_relevant = len([x for x in sorted_labels if x > 0])
    if nr_relevant == 0:
        return 0.0

    length = len(sorted_labels)
    ap = 0.0
    rel = 0

    for i in range(length):
        lab = sorted_labels[i]
        if lab >= 1:
            rel += 1
            ap += float(rel) / (i + 1.0)
    ap /= nr_relevant
    return ap

def get_gt(video_metas, query_metas):
    '''
    Param:
        video_metas: list of str. test video name. ['friends_s01e03_seg02_clip_19', ...]
        query_metas: list of str. e.x. 's02e05_seg02_clip_13#enc#4'
    Return:
        v2t_gt: [ [list of vid's queries' index] ,...]
        t2v_gt: dict. query's index: [vid's index]    
    '''
    v2t_gt = []
    for vid_id in video_metas:
        v2t_gt.append([])
        for i, query_id in enumerate(query_metas):
            if query_id.split('#', 1)[0] == vid_id:
                v2t_gt[-1].append(i)

    t2v_gt = {}
    for i, t_gts in enumerate(v2t_gt):
        for t_gt in t_gts:
            t2v_gt.setdefault(t_gt, [])
            t2v_gt[t_gt].append(i)

    return v2t_gt, t2v_gt

def eval_q2m(scores, q2m_gts):
    n_q, n_m = scores.shape

    gt_ranks = np.zeros((n_q,), np.int32)
    aps = np.zeros(n_q)
    for i in range(n_q):
        s = scores[i]   # (#video=2179)
        sorted_idxs = np.argsort(s) # (#video=2179,)
        rank = n_m + 1
        tmp_set = []
        for k in q2m_gts[i]:    # only a number here for q2m_gts[i]
            tmp = np.where(sorted_idxs == k)[0][0] + 1  # a number indicating the rank (descending)
            if tmp < rank:
                rank = tmp

        gt_ranks[i] = rank

    # compute metrics
    r1 = 100.0 * len(np.where(gt_ranks <= 1)[0]) / n_q
    r5 = 100.0 * len(np.where(gt_ranks <= 5)[0]) / n_q
    r10 = 100.0 * len(np.where(gt_ranks <= 10)[0]) / n_q
    r100 = 100.0 * len(np.where(gt_ranks <= 100)[0]) / n_q
    medr = np.median(gt_ranks)
    meanr = gt_ranks.mean()

    return (r1, r5, r10, r100, medr, meanr)

def t2v_map(c2i, t2v_gts):
    perf_list = []
    for i in range(c2i.shape[0]):
        d_i = c2i[i, :]         # (#video=2179) score of video w.r.t query
        labels = [0]*len(d_i)

        x = t2v_gts[i][0]
        labels[x] = 1

        sorted_labels = [labels[x] for x in np.argsort(d_i)]

        current_score = ap_score(sorted_labels)
        perf_list.append(current_score)
    return np.mean(perf_list)


def compute_context_info(model, eval_dataset, opt):
    model.eval()
    n_total_vid = len(eval_dataset)
    context_dataloader = DataLoader(eval_dataset, collate_fn=collate_frame_val, batch_size=opt.eval_context_bsz,
                                    num_workers=opt.num_workers, shuffle=False, pin_memory=opt.pin_memory)
    bsz = opt.eval_context_bsz
    metas = []  # list(dicts)
    frame_feat, frame_mask = [], []
    vid_proposal_feat = None
    for idx, batch in tqdm(enumerate(context_dataloader), desc="Computing query2video scores",
                           total=len(context_dataloader)):
        metas.extend(batch[-1])
        clip_video_feat_ = batch[0].to(opt.device)
        frame_video_feat_ = batch[1].to(opt.device)
        frame_mask_ = batch[2].to(opt.device)
        _frame_feat, _video_proposal_feat,  = model.encode_context(clip_video_feat_, frame_video_feat_, frame_mask_)
        _video_proposal_feat = _video_proposal_feat.cpu().numpy()
        frame_feat.append(_frame_feat)
        frame_mask.append(frame_mask_)
        if vid_proposal_feat is None:
            vid_proposal_feat = np.zeros((n_total_vid, _video_proposal_feat.shape[1], opt.hidden_size),
                                         dtype=np.float32)
            vid_proposal_feat[idx * bsz:(idx + 1) * bsz] = _video_proposal_feat
        else:
            vid_proposal_feat[idx * bsz:(idx + 1) * bsz] = _video_proposal_feat
    vid_proposal_feat = torch.from_numpy(vid_proposal_feat).to(opt.device)
    def cat_tensor(tensor_list):
        if len(tensor_list) == 0:
            return None
        else:
            seq_l = [e.shape[1] for e in tensor_list]
            b_sizes = [e.shape[0] for e in tensor_list]
            b_sizes_cumsum = np.cumsum([0] + b_sizes)
            if len(tensor_list[0].shape) == 3:
                hsz = tensor_list[0].shape[2]
                res_tensor = tensor_list[0].new_zeros(sum(b_sizes), max(seq_l), hsz)
            elif len(tensor_list[0].shape) == 2:
                res_tensor = tensor_list[0].new_zeros(sum(b_sizes), max(seq_l))
            else:
                raise ValueError("Only support 2/3 dimensional tensors")
            for i, e in enumerate(tensor_list):
                res_tensor[b_sizes_cumsum[i]:b_sizes_cumsum[i+1], :seq_l[i]] = e
            return res_tensor

    return dict(
        video_metas=metas,  # list(dict) (N_videos)
        video_proposal_feat=vid_proposal_feat,
        video_feat=cat_tensor(frame_feat),
        video_mask=cat_tensor(frame_mask)
        )

def compute_query2ctx_info(model, eval_dataset, opt, ctx_info):
    model.eval()


    query_eval_loader = DataLoader(eval_dataset, collate_fn=collate_text_val, batch_size=opt.eval_query_bsz,
                                   num_workers=opt.num_workers, shuffle=False, pin_memory=opt.pin_memory)

    query_metas = []
    clip_scale_scores = []
    frame_scale_scores = []
    score_sum = []
    
    max_indices = []
    for idx, batch in tqdm(enumerate(query_eval_loader), desc="Computing q embedding", total=len(query_eval_loader)):

        _query_metas = batch[-1]
        query_metas.extend(batch[-1])
        query_feat = batch[0].to(opt.device)
        query_mask = batch[1].to(opt.device) 
        # add key_clip_indices here
        _clip_scale_scores, _frame_scale_scores, key_clip_indices = model.get_pred_from_raw_query(
            query_feat, query_mask, None, ctx_info["video_proposal_feat"], ctx_info["video_feat"], ctx_info['video_mask'],
            is_train=False
            )
        _score_sum = opt.clip_scale_w*_clip_scale_scores + opt.frame_scale_w*_frame_scale_scores

        clip_scale_scores.append(_clip_scale_scores)
        frame_scale_scores.append(_frame_scale_scores)
        score_sum.append(_score_sum)
        
        max_indices.append(key_clip_indices)
        
    clip_scale_scores = torch.cat(clip_scale_scores, dim=0).cpu().numpy().copy()
    frame_scale_scores = torch.cat(frame_scale_scores, dim=0).cpu().numpy().copy()
    score_sum = torch.cat(score_sum, dim=0).cpu().numpy().copy()
    max_indices = torch.cat(max_indices, dim=0).cpu().numpy().copy()
    return clip_scale_scores, frame_scale_scores, score_sum, query_metas, max_indices



def cal_perf(t2v_all_errors, t2v_gt):

    # video retrieval
    (t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr) = eval_q2m(t2v_all_errors, t2v_gt)
    t2v_map_score = t2v_map(t2v_all_errors, t2v_gt)


    logging.info(" * Text to Video:")
    logging.info(" * r_1_5_10_100, medr, meanr: {}".format([round(t2v_r1, 1), round(t2v_r5, 1), round(t2v_r10, 1), round(t2v_r100, 1)]))
    logging.info(" * recall sum: {}".format(round(t2v_r1+t2v_r5+t2v_r10+t2v_r100, 1)))
    logging.info(" * mAP: {}".format(round(t2v_map_score, 4)))
    logging.info(" * "+'-'*10)

    return (t2v_r1, t2v_r5, t2v_r10,t2v_r100, t2v_medr, t2v_meanr, t2v_map_score)


def get_query_mv(query_metas, opt):
    import json
    ratios = []
    with open(opt.ratio_file, "r", encoding="utf-8") as f:
        ratio_dict = json.load(f)
    for q_name in query_metas:
        ratios.append(ratio_dict[q_name])
    return np.asarray(ratios)      

def eval_epoch(model, val_video_dataset, val_text_dataset, opt):
    '''
        video_metas: list: len=2179, test video name. ['friends_s01e03_seg02_clip_19', ...]
        video_proposal_feat:  tensor: (2179, 528=#moments=32*33/2, 384=Dim)
        
        query_context_scores: coarse branch tensor: (#queries=10895, #video=2179)
        global_query_context_scores: fine branch tensor: (#queries=10895, #video=2179)
        score_sum: combination of two branches tensor: (#queries=10895, #video=2179)
        query_metas: list of str: query's name, len=10895. 
    '''
    model.eval()
    logger.info("Computing scores")
    # dict: ['video_metas', 'video_proposal_feat', 'video_feat', 'video_mask']
    
    context_info = compute_context_info(model, val_video_dataset, opt)
    
    # return clip_scale_scores, frame_scale_scores, score_sum, query_metas
    # nparray, nparray, nparray, list
    # (10895, 2179), (10895, 2179), (10895, 2179), list of str. e.x. 's02e05_seg02_clip_13#enc#4', (10895, 2179)
    query_context_scores, global_query_context_scores, score_sum, query_metas, max_indices = compute_query2ctx_info(model,
                                                                                                        val_text_dataset,
                                                                                                        opt,
                                                                                                        context_info)
    video_metas = context_info['video_metas']
    
    # # filter according to group
    # ratios = get_query_mv(query_metas, opt)     # numpy array. each query's m/v ratio
    # valid_idx = (ratios > 0.5)
    # query_context_scores = query_context_scores[valid_idx]
    # global_query_context_scores = global_query_context_scores[valid_idx]
    # score_sum = score_sum[valid_idx]
    # query_metas = np.array(query_metas)[valid_idx].tolist()
    
    # list of list: video index to query index.      dict : q_index : [len=1, v_index]
    v2t_gt, t2v_gt = get_gt(video_metas, query_metas)
    
    # get query's top-k retrieval results
    #get_top(-1 * score_sum, t2v_gt, query_metas, video_metas, max_indices)
    
    # get st, ed from clip_index
    
    print('clip_scale_scores:')
    cal_perf(-1 * query_context_scores, t2v_gt)
    print('frame_scale_scores:')
    cal_perf(-1 * global_query_context_scores, t2v_gt)
    print('score_sum:')
    t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr, t2v_map_score = cal_perf(-1 * score_sum, t2v_gt)
    currscore = 0
    currscore += (t2v_r1 + t2v_r5 + t2v_r10 + t2v_r100)

    return currscore

def get_top(scores, t2v_gt, query_metas, video_metas, max_indices, k=5):
    import jsonlines
    rank_idx = np.argsort(scores, axis=1)   # (#query=10895, #vid=2179)
    query_videos = [t2v_gt[i][0] for i in range(len(query_metas))]  # each query's GT video index, len=10895
    query_videos_idx = [int(np.where(rank_idx[i] == query_videos[i])[0]) for i in range(len(rank_idx))] # GT's rank
    # only take top-k results
    rank_idx = rank_idx[:, :k]
    video_metas = np.asarray(video_metas)
    retrieve_videos = [video_metas[idx].tolist() for idx in rank_idx]    # top-k video's name for query, (#query, k)
    
    retrieve_clip_idx = max_indices[[i for i in range(len(query_metas))]*k, rank_idx.flatten()] # (#query, k)
    retrieve_clip_idx = np.reshape(retrieve_clip_idx, (len(query_metas), -1))
    for i in range(len(query_metas)):
        item = {
            "query": query_metas[i],
            "ret_videos": retrieve_videos[i],
            "GT_rank": query_videos_idx[i],
            "GT_video": video_metas[query_videos[i]],
            "clip_idx": retrieve_clip_idx[i].tolist()
        }
        with jsonlines.open('retrieval_with_idx.jsonl', mode = 'a') as json_writer:
            json_writer.write(item)

def get_ts(max_indices):
    # max_indices: (#query=10895, #vid=2179)
    pass
def setup_model(opt):
    """Load model from checkpoint and move to specified device"""
    ckpt_filepath = os.path.join(opt.ckpt_filepath)
    checkpoint = torch.load(ckpt_filepath)
    loaded_model_cfg = checkpoint["model_cfg"]
    NAME_TO_MODELS = {'BGM_Net':BGM_Net}
    model = NAME_TO_MODELS[opt.model_name](loaded_model_cfg)
    
    model.load_state_dict(checkpoint["model"])
    logger.info("Loaded model saved at epoch {} from checkpoint: {}".format(checkpoint["epoch"], opt.ckpt_filepath))

    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        if len(opt.device_ids) > 1:
            logger.info("Use multi GPU", opt.device_ids)
            model = torch.nn.DataParallel(model, device_ids=opt.device_ids)  # use multi GPU
    return model

def start_inference(opt=None):
    logger.info("Setup config, data and model...")
    if opt is None:
        opt = TestOptions().parse()
    cudnn.benchmark = False
    cudnn.deterministic = True

    rootpath = opt.root_path
    collection = opt.collection
    testCollection = '%stest' % collection
    # {'test': 'tvrtest.caption.txt'}
    cap_file = {'test': '%s.caption.txt' % testCollection}

    # caption   {'test': './data/netdisk/tvr/TextData/tvrtest.caption.txt'}
    caption_files = {x: os.path.join(rootpath, collection, 'TextData', cap_file[x])
                     for x in cap_file} 

    text_feat_path = os.path.join(rootpath, collection, 'TextData', 'roberta_%s_query_feat.hdf5' % collection)
    
    # m/v dict for queries : only available for TVR dataset
    opt.ratio_file = os.path.join(rootpath, collection, 'TextData', 'query_mv.json')
    
    # Load visual features
    visual_feat_path = os.path.join(rootpath, collection, 'FeatureData', opt.visual_feature)

    visual_feats = BigFile(visual_feat_path)
    # dict: video_name : list of frame_name
    # e.x.  'castle_s01e01_seg02_clip_00' : ['castle_s01e01_seg02_clip_00_0', 'castle_s01e01_seg02_clip_00_1', ...]
    video2frames =  read_dict(os.path.join(rootpath, collection, 'FeatureData', opt.visual_feature, 'video2frames.txt'))

    # ['friends_s01e03_seg02_clip_19', 'friends_s04e21_seg02_clip_18', ...]
    test_video_ids_list = read_video_ids(caption_files['test'])
    test_vid_dataset = VisDataSet(visual_feats, video2frames, opt,
                                               video_ids=test_video_ids_list)
    test_text_dataset = TxtDataSet(caption_files['test'], text_feat_path, opt)



    model = setup_model(opt)

    logger.info("Starting inference...")
    with torch.no_grad():
        score = eval_epoch(model, test_vid_dataset, test_text_dataset, opt)



if __name__ == '__main__':
    start_inference()