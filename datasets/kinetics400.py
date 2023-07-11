"""Dataset utils for NN."""
import os
import random
from glob import glob
from pprint import pprint
import uuid
import tempfile

import numpy as np
import ffmpeg
import skvideo.io
import pandas as pd
from skvideo.io import ffprobe
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


class Kinetics400Dataset(Dataset):
    """UCF101 dataset for recognition. The class index start from 0.
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
        test_sample_num： number of clips sampled from a video. 1 for clip accuracy.
    """
    def __init__(self, root_dir, clip_len, split='1', train=True, transforms_=None, test_sample_num=10):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.split = split
        self.train = train
        self.transforms_ = transforms_
        self.test_sample_num = test_sample_num
        self.toPIL = transforms.ToPILImage()
        class_idx_path = os.path.join(root_dir, 'split', 'classInd.txt')
        self.class_idx2label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1]
        self.class_label2idx = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(1)[0]

        if self.train:
            train_split_path = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
            # train_split_path = os.path.join(root_dir, 'split', 'testlist0' + self.split + '.txt')
            # self.train_split = pd.read_csv(train_split_path, header=None)[0]
        else:
            # test_split_path = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt')
            # self.test_split = pd.read_csv(test_split_path, header=None, sep=' ')[0]
            test_split_path = os.path.join(root_dir, 'split', 'testlist0' + self.split + '.txt')
            self.test_split = pd.read_csv(test_split_path, header=None)[0]

        print('Use split'+ self.split)

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index, [0-100]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            # videoname = self.train_split[idx]
            videoname = self.test_split[idx]
        class_idx = self.class_label2idx[videoname[:videoname.find('/')]]
        filename = os.path.join(self.root_dir, 'video', videoname)
        if 'v_HandstandPushups' in filename:
            filename = filename.replace('v_HandstandPushups', 'v_HandStandPushups')
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape
        
        # random select a clip for train
        if self.train:
            clip_start = random.randint(0, length - self.clip_len)
            clip = videodata[clip_start: clip_start + self.clip_len]

            if self.transforms_:
                trans_clip = []
                # fix seed, apply the sample `random transformation` for all frames in the clip 
                seed = random.random()
                for frame in clip:
                    random.seed(seed)
                    frame = self.toPIL(frame) # PIL image
                    frame = self.transforms_(frame) # tensor [C x H x W]
                    trans_clip.append(frame)
                # (T x C X H x W) to (C X T x H x W)
                clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
            else:
                clip = torch.tensor(clip)
        #NOTE: minus 1 because index ranges 0 to num-1
            return clip, torch.tensor(int(class_idx))-1
        # sample several clips for test
        else:
            clip_start = random.randint(0, length - self.clip_len)
            clip = videodata[clip_start: clip_start + self.clip_len]

            if self.transforms_:
                trans_clip = []
                # fix seed, apply the sample `random transformation` for all frames in the clip 
                seed = random.random()
                for frame in clip:
                    random.seed(seed)
                    frame = self.toPIL(frame) # PIL image
                    frame = self.transforms_(frame) # tensor [C x H x W]
                    trans_clip.append(frame)
                # (T x C X H x W) to (C X T x H x W)
                clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
            else:
                clip = torch.tensor(clip)
        #NOTE: minus 1 because index ranges 0 to num-1
            return clip, torch.tensor(int(class_idx))-1
        # sample several clips for test


class Kinetics400SRSCDataset(Dataset):
    """Kinetics dataset for serial restoration of shuffled clips. Generate clips and permutes them on-the-fly.
    Need the corresponding configuration file exists. 
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        interval (int): number of frames between clips, 16/32.
        tuple_len (int): number of clips in each tuple, 3/4/5.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """
    def __init__(self, root_dir, clip_len, interval, tuple_len, train=True, transforms_=None):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.interval = interval
        self.tuple_len = tuple_len
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        self.tuple_total_frames = clip_len * tuple_len + interval * (tuple_len - 1)

        if self.train:
            srsc_train_split_name = 'srsc_train_{}_{}_{}.txt'.format(clip_len, interval, tuple_len)
            srsc_train_split_path = os.path.join(root_dir, 'split', srsc_train_split_name)
            # self.train_split = pd.read_csv(srsc_train_split_path, header=None)[0][:1000]
            self.train_split = pd.read_csv(srsc_train_split_path, header=None)[0]

        else:
            srsc_test_split_name = 'srsc_test_{}_{}_{}.txt'.format(clip_len, interval, tuple_len)
            srsc_test_split_path = os.path.join(root_dir, 'split', srsc_test_split_name)
            self.test_split = pd.read_csv(srsc_test_split_path, header=None)[0]

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            tuple_clip (tensor): [tuple_len x channel x time x height x width]
            tuple_order (tensor): [tuple_len]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
        
        filename = os.path.join(self.root_dir, videoname)
        # correcting wrong spelling
        # if 'v_HandstandPushups' in filename:
        #     filename = filename.replace('v_HandstandPushups', 'v_HandStandPushups')
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape

        tuple_clip = []
        tuple_order = list(range(0, self.tuple_len))
        
        # random select tuple for train, deterministic random select for test
        if self.train:
            tuple_start = random.randint(0, length - self.tuple_total_frames)
        else:
            random.seed(idx)
            tuple_start = random.randint(0, length - self.tuple_total_frames)

        clip_start = tuple_start
        for _ in range(self.tuple_len):
            clip = videodata[clip_start: clip_start + self.clip_len]
            tuple_clip.append(clip)
            clip_start = clip_start + self.clip_len + self.interval

        clip_and_order = list(zip(tuple_clip, tuple_order))
        # random shuffle for train, the same shuffle for test
        if self.train:
            random.shuffle(clip_and_order)
        else:
            random.seed(idx)
            random.shuffle(clip_and_order)
        tuple_clip, tuple_order = zip(*clip_and_order)

        if self.transforms_:
            trans_tuple = []
            for clip in tuple_clip:
                trans_clip = []
                # fix seed, apply the sample `random transformation` for all frames in the clip 
                seed = random.random()
                for frame in clip:
                    random.seed(seed)
                    frame = self.toPIL(frame) # PIL image
                    frame = self.transforms_(frame) # tensor [C x H x W]
                    trans_clip.append(frame)
                # (T x C X H x W) to (C X T x H x W)
                trans_clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
                trans_tuple.append(trans_clip)
            tuple_clip = trans_tuple
        else:
            tuple_clip = [torch.tensor(clip) for clip in tuple_clip]

        return torch.stack(tuple_clip), torch.tensor(tuple_order)



def export_tuple(tuple_clip, tuple_order, dir):
    """export tuple_clip and set its name with correct order.
    
    Args:
        tuple_clip (tensor): [tuple_len x channel x time x height x width]
        tuple_order (tensor): [tuple_len]
    """
    tuple_len, channel, time, height, width = tuple_clip.shape
    for i in range(tuple_len):
        filename = os.path.join(dir, 'c{}.mp4'.format(tuple_order[i]))
        skvideo.io.vwrite(filename, tuple_clip[i])


def gen_kinetics400_srsc_splits(root_dir, clip_len, interval, tuple_len):
    """ Generate split files
    assume: root_dir==compress
    compress
      train_256
        c1
          v1.mp4
        c2 
          v2.mp4
      val_256
      split(to make)
    """

    """Generate split files for different configs."""
    srsc_train_split_name = 'srsc_train_{}_{}_{}.txt'.format(clip_len, interval, tuple_len)
    srsc_train_split_path = os.path.join(root_dir, 'split', srsc_train_split_name)
    srsc_val_split_name = 'srsc_val_{}_{}_{}.txt'.format(clip_len, interval, tuple_len)
    srsc_val_split_path = os.path.join(root_dir, 'split', srsc_val_split_name)
    # minimum length of video to extract one tuple
    min_video_len = clip_len * tuple_len + interval * (tuple_len - 1)

    def _video_longer_enough(filename):
        """Return true if video `filename` is longer than `min_video_len`"""
        path = os.path.join(root_dir, filename)
        # path = path.replace("v_HandstandPushups", "v_HandStandPushups")
        # correcting the name mismatch
        metadata = ffprobe(path)
        if 'video' in metadata.keys():
            metadata = ffprobe(path)['video']
        else:
            print(path)
            return False
        # print(metadata)
        return eval(metadata['@nb_frames']) >= min_video_len

    print('start generating train')
    train_split = pd.read_csv(os.path.join(root_dir, 'split', 'trainlist.txt'), header=None, sep=' ')[0]
    train_split = train_split[train_split.apply(_video_longer_enough)]
    train_split.to_csv(srsc_train_split_path, index=None)
    print('end generating train')


    print('start generating val')
    print(os.getcwd())
    val_split = pd.read_csv(os.path.join(root_dir, 'split', 'vallist.txt'), header=None, sep=' ')[0]
    # print(val_split)
    val_split = val_split[val_split.apply(_video_longer_enough)]
    val_split.to_csv(srsc_val_split_path, index=None)
    print('end generating val')


def gen_kinetics400_video_files(root_dir, out_file=None):
    """ Generate video lists
    assume: root_dir=compress
    compress
      train_256
        c1
          v1.mp4
        c2 
          v2.mp4
      val_256
      split(to make)
    """
    cur_path = os.getcwd()
    os.chdir(root_dir)

    trainlist = open('split/trainlist.txt', 'a')
    vallist = open('split/vallist.txt', 'a')

    for d in os.listdir('train_256'):
        for v in os.listdir('train_256/'+d):
            trainlist.write('/'.join(['train_256', d, v])+'\n')

    for d in os.listdir('val_256'):
        for v in os.listdir('val_256/'+d):
            vallist.write('/'.join(['val_256', d, v])+'\n')

    trainlist.close()
    vallist.close()


if __name__ == '__main__':
    seed = 632
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    gen_kinetics400_video_files('../data/compress')
    gen_kinetics400_srsc_splits('../data/compress', 16, 8, 3)
