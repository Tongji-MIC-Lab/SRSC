import pickle
import h5py
import torch
import os
import json
import torch.utils.data as data
import numpy as np
from utils.opt import parse_opt
opt = parse_opt()



if opt.dataset == 'msvd':
    f = 'utils/Vid2Url_Full.txt'
    msvd_vid2id = eval(open(f, 'r').read())
    print(f'len of msvd_vid2id dict: {len(msvd_vid2id)}') 
    vid2id = msvd_vid2id
else:
    vid2id = None


class V2TDataset(data.Dataset):
    def __init__(self, cap_pkl, frame_feature_h5, region_feature_h5, ss_feature_dir, vid2id=vid2id):
        with open(cap_pkl, 'rb') as f:
            self.captions, self.pos_tags, self.lengths, self.video_ids = pickle.load(f)
        h5 = h5py.File(frame_feature_h5, 'r')
        self.video_feats = h5[opt.feature_h5_feats]
        h5 = h5py.File(region_feature_h5, 'r')
        self.region_feats = h5[opt.region_visual_feats]
        self.spatial_feats = h5[opt.region_spatial_feats]
        self.ss_feature_dir = ss_feature_dir
        self.vid2id = vid2id


    def __getitem__(self, index):
        caption = self.captions[index]
        pos_tag = self.pos_tags[index]
        length = self.lengths[index]
        video_id = self.video_ids[index]
        
        video_feat = torch.from_numpy(self.video_feats[video_id])
        region_feat = torch.from_numpy(self.region_feats[video_id])
        spatial_feat = torch.from_numpy(self.spatial_feats[video_id])
        # map video num to video url
        if self.vid2id:
            ss_feat = torch.load(os.path.join(self.ss_feature_dir, self.vid2id['vid'+str(video_id+1)]+'.pt'))
        else:
            ss_feat = torch.load(os.path.join(self.ss_feature_dir, 'video'+str(video_id)+'.pt'))
        ss_feats_len = len(ss_feat)
        ss_sampled_idxs = np.linspace(0, ss_feats_len-1, 26, dtype=int) # 26 == max_frames
        ss_feat = ss_feat[ss_sampled_idxs]
        assert len(ss_feat) == len(video_feat)
        return video_feat, region_feat, spatial_feat, ss_feat, caption, pos_tag, length, video_id

    def __len__(self):
        return len(self.captions)


class VideoDataset(data.Dataset):
    def __init__(self, eval_range, frame_feature_h5, region_feature_h5, ss_feature_dir, vid2id=vid2id):
        self.eval_list = tuple(range(*eval_range))
        h5 = h5py.File(frame_feature_h5, 'r')
        self.video_feats = h5[opt.feature_h5_feats]
        h5 = h5py.File(region_feature_h5, 'r')
        self.region_feats = h5[opt.region_visual_feats]
        self.spatial_feats = h5[opt.region_spatial_feats]
        self.ss_feature_dir = ss_feature_dir
        self.vid2id = vid2id

    def __getitem__(self, index):
        video_id = self.eval_list[index]
        video_feat = torch.from_numpy(self.video_feats[video_id])
        region_feat = torch.from_numpy(self.region_feats[video_id])
        spatial_feat = torch.from_numpy(self.spatial_feats[video_id])
        if self.vid2id:
            ss_feat = torch.load(os.path.join(self.ss_feature_dir, self.vid2id['vid'+str(video_id+1)]+'.pt'))
        else:
            ss_feat = torch.load(os.path.join(self.ss_feature_dir, 'video'+str(video_id)+'.pt'))
        ss_feats_len = len(ss_feat)
        ss_sampled_idxs = np.linspace(0, ss_feats_len-1, 26, dtype=int) # 26 == max_frames
        ss_feat = ss_feat[ss_sampled_idxs]
        assert len(ss_feat) == len(video_feat)
        return video_feat, region_feat, spatial_feat, ss_feat, video_id

    def __len__(self):
        return len(self.eval_list)


def train_collate_fn(data):
    data.sort(key=lambda x: x[-1], reverse=True)

    videos, regions, spatials, ss_feat, captions, pos_tags, lengths, video_ids = zip(*data)

    videos = torch.stack(videos, 0)
    regions = torch.stack(regions, 0)
    spatials =torch.stack(spatials, 0)
    ss_feat = torch.stack(ss_feat, 0)

    captions = torch.stack(captions, 0)
    pos_tags = torch.stack(pos_tags, 0)
    return videos, regions, spatials, ss_feat, captions, pos_tags, lengths, video_ids


def eval_collate_fn(data):
    data.sort(key=lambda x: x[-1], reverse=False)

    videos, regions, spatials, ss_feat, video_ids = zip(*data)

    videos = torch.stack(videos, 0)
    regions = torch.stack(regions, 0)
    spatials = torch.stack(spatials, 0)
    ss_feat = torch.stack(ss_feat, 0)

    return videos, regions, spatials, ss_feat, video_ids


def get_train_loader(cap_pkl, frame_feature_h5, region_feature_h5, ss_feat_dir, batch_size=100, shuffle=True, num_workers=3, pin_memory=True):
    v2t = V2TDataset(cap_pkl, frame_feature_h5, region_feature_h5, ss_feat_dir)
    data_loader = torch.utils.data.DataLoader(dataset=v2t,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=train_collate_fn,
                                              pin_memory=pin_memory)
    return data_loader


def get_eval_loader(cap_pkl, frame_feature_h5, region_feature_h5, ss_feat_dir, batch_size=100, shuffle=False, num_workers=1, pin_memory=False):
    vd = VideoDataset(cap_pkl, frame_feature_h5, region_feature_h5, ss_feat_dir)
    data_loader = torch.utils.data.DataLoader(dataset=vd,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=eval_collate_fn,
                                              pin_memory=pin_memory)
    return data_loader


if __name__ == '__main__':
    train_loader = get_train_loader(opt.train_caption_pkl_path, opt.feature_h5_path, opt.region_feature_h5_path)
    print(len(train_loader))
    d = next(iter(train_loader))
    print(d[0].size())
    print(d[1].size())
    print(d[2].size())
    print(d[3].size())
    print(len(d[4]))
    print(d[5])
