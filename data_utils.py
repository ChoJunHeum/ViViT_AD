import os
import random
from functools import reduce

import glob
import torch
import numpy as np
import scipy.io as scio
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
import cv2

def img2tensor(raw_img, pil2tensor, resize_h, resize_w, symmetric=True):
    """
    Load and resize img, convert to tensor then map to device.
    Image.open: [W, H]
    ToTensor: [C, H, W]
    if symmetric==True, [-1., 1]
    else, [0., 1.]
    Return tensor, shape: CHW (RGB)
    """
    if isinstance(raw_img, str):
        img = Image.open(raw_img)
    else:
        img = raw_img
    img = img.resize([resize_w, resize_h])
    img_t = pil2tensor(img)
    if symmetric:
        img_t = img_t*2-1
    return img_t


class CropTrainDataset(Dataset):
    """
    cv2 imread: BGR, HWC (H, W, C)
    PIL open: RGB, HWC (H, W, C)
    ToTensor: [C, H, W]

    batches:
        x_t: [B, C_out * nframe, resize_h, resize_w]
        y_t: [B, C_out, resize_h, resize_w]
    """
    def __init__(self, cfg):
        self.resize_h = cfg.resize_h
        self.resize_w = cfg.resize_w
        self.symmetric = cfg.symmetric
        self.device = cfg.device
        self.pil2tensor = ToTensor()
        total_frames = cfg.nframe  # input_frame 

        self.all_path = [] # len(self.all_path) = nframe
        for folder_t in sorted(glob.glob(f'{cfg.cropped_data}/*')): # t0, t1, ...
            path_infolder = glob.glob(f'{folder_t}/*.jpg')
            path_infolder.sort()
            self.all_path.append(path_infolder) 
        lengths = [len(folder_t) for folder_t in self.all_path]
        total_length = reduce(lambda x, y: x+y, lengths)

        assert len(self.all_path) == total_frames, f'{len(self.all_path)} and {total_frames}'
        assert total_length == lengths[0] * total_frames, f'{total_length} and {lengths[0] * total_frames}'

    def __len__(self): 
        '''
        Return length of paired sets
        '''
        return len(self.all_path[0])

    def __getitem__(self, index):
        '''
        frames_all = [ img_from_t0, img_from_t1, ... ]
        len(frames_all) == nframe
        one_path[i]: shape [3, resize_h, resize_w]
        '''
        one_patch_path = [folder_t[index] for folder_t in self.all_path]

        one_patch = [img2tensor(path,
                        pil2tensor=self.pil2tensor,
                        resize_h=self.resize_h,
                        resize_w=self.resize_w,
                        symmetric=self.symmetric,
                        ) for path in one_patch_path]
        
        x_t = torch.cat(one_patch, dim=0).to(self.device)
        # y_t = torch.as_tensor(one_patch[-1], device=self.device)

        return x_t

class TotalTrainDataset(Dataset):
    """
    cv2 imread: BGR, HWC (H, W, C)
    PIL open: RGB, HWC (H, W, C)
    ToTensor: [C, H, W]

    batches:
        x_t: [B, C_out * nframe, resize_h, resize_w]
        y_t: [B, C_out, resize_h, resize_w]
    """
    def __init__(self, cfg):
        self.resize_h = cfg.resize_h
        self.resize_w = cfg.resize_w
        self.symmetric = cfg.symmetric
        self.device = cfg.device
        self.pil2tensor = ToTensor()
        total_frames = cfg.nframe  # input_frame 
        self.clip_length = 8

        self.all_path = [] # len(self.all_path) = nframe
        for folder_t in sorted(glob.glob(f'{cfg.cropped_data}/*')): # t0, t1, ...
            path_infolder = glob.glob(f'{folder_t}/*.jpg')
            path_infolder.sort()
            self.all_path.append(path_infolder) 

        # self.all_path: [[t0], [t1], ... , [t8]]
        # len(self.all_path) = 9

        self.videos = []
        self.all_seqs = []

        for folder in sorted(glob.glob(f'{cfg.train_data}/*')):
            all_imgs = glob.glob(f'{folder}/*.jpg')
            all_imgs.sort()
            self.videos.append(all_imgs)

            random_seq = list(range(len(all_imgs) - 8))
            self.all_seqs.append(random_seq)

        lengths = [len(folder_t) for folder_t in self.all_path]
        total_length = reduce(lambda x, y: x+y, lengths)

        assert len(self.all_path) == total_frames, f'{len(self.all_path)} and {total_frames}'
        assert total_length == lengths[0] * total_frames, f'{total_length} and {lengths[0] * total_frames}'

    def __len__(self): 
        '''
        Return length of paired sets
        '''
        return len(self.all_path[0])

    def __getitem__(self, index):
        '''
        frames_all = [ img_from_t0, img_from_t1, ... ]
        len(frames_all) == nframe
        one_path[i]: shape [3, resize_h, resize_w]
        '''
        one_patch_path = [folder_t[index] for folder_t in self.all_path]

        one_patch = [img2tensor(path,
                        pil2tensor=self.pil2tensor,
                        resize_h=self.resize_h,
                        resize_w=self.resize_w,
                        symmetric=self.symmetric,
                        ) for path in one_patch_path]
        
        x_t = torch.cat(one_patch, dim=0).to(self.device)

        one_folder = self.videos[index]

        video_clip = []

        start = self.all_seqs[index][-1]  # Always use the last index in self.all_seqs.

        for i in range(start, start + self.clip_length):
            img = cv2.imread(one_folder[i])
 
            video_clip.append(img)

        return [x_t, video_clip]


class train_target_dataset(Dataset):
    """
    No data augmentation.
    Normalized from [0, 255] to [-1, 1], the channels are BGR due to cv2 and liteFlownet.
    """

    def __init__(self, cfg):
        self.img_h = cfg.img_size[0] # 256
        self.img_w = cfg.img_size[1] # 256
        self.clip_length = 9

        self.videos = []
        self.all_seqs = []
        for folder in sorted(glob.glob(f'{cfg.train_data}/*')):
            all_imgs = glob.glob(f'{folder}/*.jpg')
            all_imgs.sort()
            self.videos.append(all_imgs)

            random_seq = list(range(len(all_imgs) - 8))
            self.all_seqs.append(random_seq)

    def __len__(self):  # This decide the indice range of the PyTorch Dataloader.
        return len(self.videos)

    def __getitem__(self, indice):  # Indice decide which video folder to be loaded.

        one_folder = self.videos[indice]

        video_clip = []

        start = self.all_seqs[indice][-1]  # Always use the last index in self.all_seqs.

        for i in range(start, start + self.clip_length):
            img = cv2.imread(one_folder[i])
 
            video_clip.append(img)

        return indice, video_clip

class FrameDataset(object):
    def __init__(self, cfg):
        self.nframe = cfg.nframe
        self.total_frames = self.nframe + 1

        self.all_pathes = []
        self.flat_seq = []
        self.all_seq = []
        
        tr_dir_names = os.listdir(cfg.train_data)
        tr_dir_names.sort()
        for folder_name in tr_dir_names:
            folder_path = os.path.join(cfg.train_data, folder_name)
            folder_idx = int(folder_name)
            img_pathes = glob.glob(f"{folder_path}/*.jpg")
            img_pathes.sort()
            self.all_pathes.append(img_pathes)

            img_seq = list(range(len(img_pathes) - self.nframe))
            folder_seq = [f'{folder_idx} {img_idx}' for img_idx in img_seq]
            self.flat_seq.extend(folder_seq)
            self.all_seq.append(img_seq)

    def __len__(self):
        # return len(self.mem) - (self.nframe)
        return len(self.flat_seq)

    def __getitem__(self, index):
        frames = []
        seq_str = self.flat_seq[index]
        folder_idx, img_idx = seq_str.split()
        folder_idx = int(folder_idx) - 1
        img_idx = int(img_idx)
        start = self.all_seq[folder_idx][img_idx]
        for frame_idx in range(start, start + self.total_frames):
            frames.append(Image.open(self.all_pathes[folder_idx][frame_idx]))
        return frames[:-1], frames[-1]
    
    def shuffle(self):
        random.shuffle(self.flat_seq)


class TestDataset:
    def __init__(self, cfg, folder):
        self.total_frames = cfg.nframe + 1 # input_frame + target_frame
        self.imgs = glob.glob(f'{folder}/*.jpg')
        self.imgs.sort()

    def __len__(self):
        return len(self.imgs) - (self.total_frames - 1)

    def __getitem__(self, indice):
        frames = []
        for i in range(indice, indice + self.total_frames):
            frames.append(Image.open(self.imgs[i]))

        return frames[:-1], frames[-1]


class Label_loader:
    def __init__(self, cfg, folders):
        assert cfg.te_name in ('ped2', 'avenue', 'shanghai'), f'Did not find the related gt for \'{cfg.te_name}\'.'
        self.te_name = cfg.te_name
        self.mat_path = f'{cfg.te_path}/{self.te_name}.mat'
        self.folders = folders

    def __call__(self):
        if self.te_name == 'shanghaitech':
            gt = self.load_shanghaitech()
        else:
            gt = self.load_ucsd_avenue()
        return gt

    def load_ucsd_avenue(self):
        abnormal_events = scio.loadmat(self.mat_path, squeeze_me=True)['gt']

        all_gt = []
        for i in range(abnormal_events.shape[0]):
            length = len(os.listdir(self.folders[i]))
            sub_video_gt = np.zeros((length,), dtype=np.int8)
            one_abnormal = abnormal_events[i]

            if one_abnormal.ndim == 1:
                one_abnormal = one_abnormal.reshape((one_abnormal.shape[0], -1))

            for j in range(one_abnormal.shape[1]):
                start = one_abnormal[0, j] - 1
                end = one_abnormal[1, j]
                sub_video_gt[start: end] = 1
            all_gt.append(sub_video_gt)

        return all_gt

    def load_shanghaitech(self):
        np_list = glob.glob(f'{self.mat_path}/frame_masks/')
        np_list.sort()

        gt = []
        for npy in np_list:
            gt.append(np.load(npy))

        return gt
