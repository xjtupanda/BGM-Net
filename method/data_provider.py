import json
import torch
import scipy.interpolate
import torch.utils.data as data
import numpy as np
import re
import h5py


def getVideoId(cap_id):
    vid_id = cap_id.split('#')[0]
    return vid_id

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    return string.strip().lower().split()

def read_video_ids(cap_file):
    video_ids_list = []
    with open(cap_file, 'r') as cap_reader:
        for line in cap_reader.readlines():
            cap_id, caption = line.strip().split(' ', 1)
            video_id = getVideoId(cap_id)
            if video_id not in video_ids_list:
                video_ids_list.append(video_id)
    return video_ids_list

# resize feature to fixed length using 1D(temporal) interpolation
def poolData(data,num_prop=32,num_bin=1,num_sample_bin=3,pool_type="mean"):
    T, C = data.shape
    if len(data)==1:
        video_feature=np.stack([data]*num_prop)
        return video_feature
    
    gap_len = num_prop / float(T)
    x = [gap_len / 2 + ii * gap_len for ii in range(len(data))] 
    f=scipy.interpolate.interp1d(x,data,axis=0) # interpolate features in Time Dimension
        
    video_feature=[]
    zero_sample=np.zeros(data.shape[1:])
    #zero_sample=np.zeros(num_bin*400)    # all zero feature for one point
    tmp_anchor_xmin=[1.0/num_prop*i for i in range(num_prop)]   # make normalized([0,1]) timestamps,
                                                                # left point of interval
    tmp_anchor_xmax=[1.0/num_prop*i for i in range(1,num_prop+1)] # right point of interval
    
    num_sample=num_bin*num_sample_bin
    for idx in range(num_prop):
        xmin=max(x[0]+0.0001, tmp_anchor_xmin[idx] * num_prop)
        xmax=min(x[-1]-0.0001, tmp_anchor_xmax[idx] * num_prop)
        if xmax<x[0]:
            video_feature.append(zero_sample)
            continue
        if xmin>x[-1]:
            video_feature.append(zero_sample)
            continue
            
        plen=(xmax-xmin)/(num_sample-1)
        x_new=[xmin+plen*ii for ii in range(num_sample)]
        y_new=f(x_new)
        y_new_pool=[]
        for b in range(num_bin):
            tmp_y_new=y_new[num_sample_bin*b:num_sample_bin*(b+1)]
            if pool_type=="mean":
                tmp_y_new=np.mean(y_new,axis=0)
            elif pool_type=="max":
                tmp_y_new=np.max(y_new,axis=0)
            y_new_pool.append(tmp_y_new)
        y_new_pool=np.stack(y_new_pool).squeeze()
        video_feature.append(y_new_pool)
    video_feature=np.stack(video_feature)
    return video_feature

def average_to_fixed_length(visual_input, map_size):
    visual_input = torch.from_numpy(visual_input)       # (vid_len, feat_dim)
    num_sample_clips = map_size
    num_clips = visual_input.shape[0]
    idxs = torch.arange(0, num_sample_clips + 1, 1.0) / num_sample_clips * num_clips

    idxs = torch.min(torch.round(idxs).long(), torch.tensor(num_clips - 1))

    new_visual_input = []

    for i in range(num_sample_clips):

        s_idx, e_idx = idxs[i].item(), idxs[i + 1].item()
        if s_idx < e_idx:
            new_visual_input.append(torch.mean(visual_input[s_idx:e_idx], dim=0))
        else:
            new_visual_input.append(visual_input[s_idx])
    new_visual_input = torch.stack(new_visual_input, dim=0).numpy()


    return new_visual_input

def uniform_feature_sampling(features, max_len):
    num_clips = features.shape[0]
    if max_len is None or num_clips <= max_len:
        return features
    idxs = np.arange(0, max_len + 1, 1.0) / max_len * num_clips
    idxs = np.round(idxs).astype(np.int32)
    idxs[idxs > num_clips - 1] = num_clips - 1
    new_features = []
    for i in range(max_len):
        s_idx, e_idx = idxs[i], idxs[i + 1]
        if s_idx < e_idx:
            new_features.append(np.mean(features[s_idx:e_idx], axis=0))
        else:
            new_features.append(features[s_idx])
    new_features = np.asarray(new_features)
    return new_features


def l2_normalize_np_array(np_array, eps=1e-5):
    """np_array: np.ndarray, (*, D), where the last dim will be normalized"""
    return np_array / (np.linalg.norm(np_array, axis=-1, keepdims=True) + eps)



def collate_train(data):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    dataset's get_item: return clip_video_feature, frame_video_feature, cap_tensors, index, cap_ids, video_id
        clip_video_feature: (L=32, D=3072)
        frame_video_feature: (L<=128, D=3072)
        cap_tensors: list, each contains a caption embedding of shape: (L<=30, D=768)
        index: video_index from __getitem__
        cap_ids: list of str, each is a caption's id corresponding to video_id. e.x. friends_s01e03_seg02_clip_19#enc#0
        video_id: video_name    e.x. friends_s01e03_seg02_clip_19
        
        type(data): list
        len(data): batch_size
    """
    # Sort a data list by vid_len(frame_nums)
    if data[0][1] is not None:
        data.sort(key=lambda x: len(x[1]), reverse=True)
    clip_video_features, frame_video_features, captions, idxs, cap_ids, video_ids = zip(*data)

    #videos
    clip_videos = torch.cat(clip_video_features, dim=0).float()

    video_lengths = [len(frame) for frame in frame_video_features]  # len of each vid
    frame_vec_len = len(frame_video_features[0][0])                 # feature dimension
    frame_videos = torch.zeros(len(frame_video_features), max(video_lengths), frame_vec_len)
    videos_mask = torch.zeros(len(frame_video_features), max(video_lengths))
    for i, frames in enumerate(frame_video_features):
        end = video_lengths[i]
        frame_videos[i, :end, :] = frames[:end, :]
        videos_mask[i, :end] = 1.0

    #captions
    feat_dim = captions[0][0].shape[-1]

    merge_captions = []
    all_lengths = []
    labels = []

    for index, caps in enumerate(captions):
        labels.extend(index for _ in range(len(caps)))
        all_lengths.extend(len(cap) for cap in caps)
        merge_captions.extend(cap for cap in caps)

    target = torch.zeros(len(all_lengths), max(all_lengths), feat_dim)
    words_mask = torch.zeros(len(all_lengths), max(all_lengths))

    for index, cap in enumerate(merge_captions):
        end = all_lengths[index]
        target[index, :end, :] = cap[:end, :]
        words_mask[index, :end] = 1.0



    return dict(clip_video_features=clip_videos,
                frame_video_features=frame_videos,
                videos_mask=videos_mask,
                text_feat=target,
                text_mask=words_mask,
                text_labels=labels
                )


def collate_frame_val(data):
    clip_video_features, frame_video_features, idxs, video_ids = zip(*data)

    # Merge videos (convert tuple of 1D tensor to 4D tensor)
    # videos
    clip_videos = torch.cat(clip_video_features, dim=0).float()

    video_lengths = [len(frame) for frame in frame_video_features]
    frame_vec_len = len(frame_video_features[0][0])
    frame_videos = torch.zeros(len(frame_video_features), max(video_lengths), frame_vec_len)
    videos_mask = torch.zeros(len(frame_video_features), max(video_lengths))
    for i, frames in enumerate(frame_video_features):
        end = video_lengths[i]
        frame_videos[i, :end, :] = frames[:end, :]
        videos_mask[i, :end] = 1.0

    return clip_videos, frame_videos, videos_mask, idxs, video_ids


def collate_text_val(data):
    if data[0][0] is not None:
        data.sort(key=lambda x: len(x[0]), reverse=True)
    captions,idxs, cap_ids = zip(*data)

    if captions[0] is not None:
        # Merge captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        target = torch.zeros(len(captions), max(lengths), captions[0].shape[-1])
        words_mask = torch.zeros(len(captions), max(lengths))
        for i, cap in enumerate(captions):
            end = lengths[i]
            target[i, :end] = cap[:end]
            words_mask[i, :end] = 1.0
    else:
        target = None
        lengths = None
        words_mask = None


    return target, words_mask, idxs, cap_ids




class BasicDataset(data.Dataset):
    """
    Load captions and video frame features by pre-trained CNN model.
    """

    def __init__(self, cap_file, visual_feat, text_feat_path, opt, video2frames=None):
        # Captions
        self.captions = {}          # map cap_id to caption
        self.cap_ids = []           # contain all the cap_id
        self.video_ids = []         # contain all the vid_id    (vid_id may correspond to multiple cap_ids)
        self.vid_caps = {}          # contain all the {vid_name:[cap_id]}
        self.video2frames = video2frames    # a dict, {vid_name:[frame_name]}
                                            # e.x. 'castle_s01e01_seg02_clip_00': ['castle_s01e01_seg02_clip_00_xx']
        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)    # e.x. cap_id: friends_s01e03_seg02_clip_19#enc#0 
                                                                #      caption: Phoebe puts one of her ponytails in her mouth.
                video_id = getVideoId(cap_id)                   # friends_s01e03_seg02_clip_19
                self.captions[cap_id] = caption     # dict: {cap_id: caption}
                self.cap_ids.append(cap_id)
                if video_id not in self.video_ids:
                    self.video_ids.append(video_id)
                if video_id in self.vid_caps:
                    self.vid_caps[video_id].append(cap_id)
                else:
                    self.vid_caps[video_id] = []
                    self.vid_caps[video_id].append(cap_id)
        self.visual_feat = visual_feat
        self.text_feat_path = text_feat_path

        self.map_size = opt.map_size            # clip_feature_len:  32
        self.max_ctx_len = opt.max_ctx_l        # frame_feature_len: 128
        self.max_desc_len = opt.max_desc_l      # max len of descriptions(captions). 30

        self.open_file = False
        self.length = len(self.vid_caps)



    def __getitem__(self, index):
        '''
            index: index of video
            return: 
                   feature vector of all the frames of the video
                   caption's embedding
        '''
        if self.open_file:
            self.open_file = True
        else:
            self.text_feat = h5py.File(self.text_feat_path, 'r')

            self.open_file = True
        video_id = self.video_ids[index]
        cap_ids = self.vid_caps[video_id]   # a list of cap_ids(str) corresponding to the video_id

        # video
        frame_list = self.video2frames[video_id]


        frame_vecs = []
        for frame_id in frame_list:
            frame_vecs.append(self.visual_feat.read_one(frame_id))      # frame-wise feature vector. each 3072 dim

        #clip_video_feature = average_to_fixed_length(np.array(frame_vecs), self.map_size)   # sample to len:32
        clip_video_feature = poolData(np.array(frame_vecs), self.map_size)
        clip_video_feature = l2_normalize_np_array(clip_video_feature)                      # l2 normalization on feature dim
        clip_video_feature = torch.from_numpy(clip_video_feature).unsqueeze(0)              # (1, 32, D)

        frame_video_feature = uniform_feature_sampling(np.array(frame_vecs), self.max_ctx_len)  # sample to len<=128
        frame_video_feature = l2_normalize_np_array(frame_video_feature)
        frame_video_feature = torch.from_numpy(frame_video_feature)

        # text
        cap_tensors = []
        for cap_id in cap_ids:

            cap_feat = self.text_feat[cap_id][...]  # (sentence_len, dim=768)

            cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat))[:self.max_desc_len]  # (L<=30, text_feat_dim)
            cap_tensors.append(cap_tensor)

        return clip_video_feature, frame_video_feature, cap_tensors, index, cap_ids, video_id

    def __len__(self):
        return self.length

class VisDataSet(data.Dataset):

    def __init__(self, visual_feat, video2frames, opt, video_ids=None):
        self.visual_feat = visual_feat
        self.video2frames = video2frames
        if video_ids is not None:
            self.video_ids = video_ids
        else:
            self.video_ids = video2frames.keys()
        self.length = len(self.video_ids)
        self.map_size = opt.map_size
        self.max_ctx_len = opt.max_ctx_l
    def __getitem__(self, index):
        video_id = self.video_ids[index]
        frame_list = self.video2frames[video_id]
        frame_vecs = []
        for frame_id in frame_list:
            frame_vecs.append(self.visual_feat.read_one(frame_id))
        #clip_video_feature = average_to_fixed_length(np.array(frame_vecs), self.map_size)
        clip_video_feature = poolData(np.array(frame_vecs), self.map_size)
        clip_video_feature = l2_normalize_np_array(clip_video_feature)
        clip_video_feature = torch.from_numpy(clip_video_feature).unsqueeze(0)

        frame_video_feature = uniform_feature_sampling(np.array(frame_vecs), self.max_ctx_len)
        frame_video_feature = l2_normalize_np_array(frame_video_feature)
        frame_video_feature = torch.from_numpy(frame_video_feature)

        return clip_video_feature, frame_video_feature, index, video_id

    def __len__(self):
        return self.length


class TxtDataSet(data.Dataset):
    """
    Load captions
    """

    def __init__(self, cap_file, text_feat_path, opt):
        # Captions
        self.captions = {}
        self.cap_ids = []
        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
        self.text_feat_path = text_feat_path
        self.max_desc_len = opt.max_desc_l
        self.open_file = False
        self.length = len(self.cap_ids)

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        if self.open_file:
            self.open_file = True
        else:
            self.text_feat = h5py.File(self.text_feat_path, 'r')

            self.open_file = True


        cap_feat = self.text_feat[cap_id][...]

        cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat))[:self.max_desc_len]

        return cap_tensor, index, cap_id

    def __len__(self):
        return self.length




if __name__ == '__main__':
    pass


