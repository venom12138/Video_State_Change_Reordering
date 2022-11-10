import os
from os import path, replace
import math
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
import sys
sys.path.append('/cluster/home2/yjw/venom/XMem')
from dataset.range_transform import im_normalization, im_mean
from dataset.reseed import reseed
import yaml
import matplotlib.pyplot as plt
from glob import glob

class EPICDataset(Dataset):
    """
    Works for EPIC training
    For each sequence:
    - Pick three frames
    - Pick two objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """
    def __init__(self, data_root, yaml_root, openword_test=False, num_frames=5, repr_type=None, finetune=False):
        print('We are using EPIC Dataset !!!!!')
        self.data_root = data_root
        self.num_frames = num_frames
        with open(os.path.join(yaml_root), 'r') as f:
            self.data_info = yaml.safe_load(f)
            
        with open(os.path.join(self.data_root, 'train_open_word.yaml'), 'r') as f:
            self.open_word_info = yaml.safe_load(f)
        f.close()
        
        self.vids = [] 
        for key in list(self.data_info.keys()):
            if openword_test:
                if self.open_word_info[key] != 'svsn':
                    continue
                
            PART = key.split('_')[0]
            VIDEO_ID = '_'.join(key.split('_')[:2])
            vid_rgb_path = os.path.join(self.data_root, PART, 'rgb_frames', VIDEO_ID, key)
            # print(vid_gt_path)
            # print(glob(vid_gt_path))
            if len(glob(f"{vid_rgb_path}/*.jpg")) >= 2:
                self.vids.append(key)
                
        self.repr_type = repr_type
        assert repr_type in ['SlowFast', 'VideoMae']
        assert num_frames == 5
        
        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.01, 0.01, 0.01, 0),
        ])
        # 仿射变换：平移旋转之类的
        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0 if finetune else 15, shear=0 if finetune else 10, interpolation=InterpolationMode.BILINEAR, fill=im_mean),
        ])

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0 if finetune else 15, shear=0 if finetune else 10, interpolation=InterpolationMode.NEAREST, fill=0),
        ])

        # These transform are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.03, 0.03, 0),
            transforms.RandomGrayscale(0.05),
        ])

        
        self.all_im_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((224, 224), scale=(0.36,1.00), interpolation=InterpolationMode.BILINEAR)
        ])

        self.all_gt_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((224, 224), scale=(0.36,1.00), interpolation=InterpolationMode.NEAREST)
        ])

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])

    def __getitem__(self, idx):
        video_value = self.data_info[self.vids[idx]] # video value
        
        info = {}
        info['name'] = self.vids[idx]

        vid_im_path = path.join(self.data_root, video_value['participant_id'], 'rgb_frames', video_value['video_id'], self.vids[idx])

        frames = list(range(video_value['start_frame'], video_value['stop_frame']))

        info['frames'] = [] # Appended with actual frames

        num_frames = self.num_frames
        
        frames_idx = np.random.choice(list(range(len(frames))), size=num_frames, replace=False)
        frames_idx = np.sort(frames_idx)
        
        all_images = []
        all_slow_images = []
        
        gt_order = np.array(list(range(len(frames_idx))))
        noise = np.random.rand(num_frames)  # noise in [0, 1]
        idx = np.argsort(noise)
        # 打乱顺序
        frames_idx = frames_idx[idx]
        gt_order = gt_order[idx]
        if self.repr_type == 'SlowFast':
            near_k = 8
            
        elif self.repr_type == 'VideoMae':
            near_k = 8
        
        for f_idx in frames_idx:
            # 每个f_idx 附近取k帧
            if f_idx > near_k//2 and f_idx < len(frames) - near_k//2:
                k_index_list = list(range(f_idx-near_k//2, f_idx+near_k//2))
            elif f_idx <= near_k//2:
                k_index_list = list(range(0, near_k))
            elif f_idx >= len(frames) - near_k//2:
                k_index_list = list(range(len(frames)-near_k, len(frames)))
            
            images = []
            slow_images = []
        
            for k_idx in k_index_list:
                jpg_name = 'frame_' + str(frames[k_idx]).zfill(10)+ '.jpg'
                info['frames'].append(jpg_name)
                this_im = Image.open(path.join(vid_im_path, jpg_name)).convert('RGB')
                this_im = self.all_im_dual_transform(this_im)
                this_im = self.all_im_lone_transform(this_im)
                this_im = self.pair_im_dual_transform(this_im)
                this_im = self.pair_im_lone_transform(this_im)
                this_im = self.final_im_transform(this_im)
                slow_images.append(this_im)
                this_im = this_im.unsqueeze(0)
                if self.repr_type == 'SlowFast':
                    # [4, 3, 384, 384]
                    this_im = torch.repeat_interleave(this_im, 4, dim=0)
                elif self.repr_type == 'VideoMae':
                    # [2, 3, 384, 384]
                    this_im = torch.repeat_interleave(this_im, 2, dim=0)
                images.append(this_im)

            # [8/2*near_k, 3, 384, 384]
            images = torch.cat(images, 0)
            slow_images = torch.stack(slow_images, 0)
            
            all_images.append(images)
            all_slow_images.append(slow_images)
        
        all_images = torch.stack(all_images)
        all_slow_images = torch.stack(all_slow_images)
        # print(f"all_images:{all_images.shape}")
        # print(f"all_slow_images:{all_slow_images.shape}")
        # dd
        
        data = {
            'rgb': all_images, # [5, 8/2*near_k, 3, H, W]
            'slow_images': all_slow_images, # [5, 8/2, 3, H, W]
            'gt_order': gt_order, # [num_frames]
            'text':video_value['narration'],
            'info': info,
        }

        return data

    def __len__(self):
        return len(self.vids)

if __name__ == '__main__':
    dataset = EPICDataset(data_root='/home/venom/projects/XMem/EPIC_train', yaml_root='/home/venom/projects/XMem/EPIC_train/EPIC100_state_positive_train.yaml', num_frames=5, finetune=False)
    print(len(dataset))
    data = dataset[2]
    print(data['info'])
    print(data['gt_order'])