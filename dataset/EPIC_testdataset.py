import os
from os import path, replace

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
from torch.utils.data import DataLoader
from glob import glob
import skimage.measure as measure

class EPICtestDataset(Dataset):
    def __init__(self, data_root, yaml_root, valset_yaml_root, num_frames=5, repr_type=None):
        print('We are using EPIC testDataset !!!!!')
        self.data_root = data_root
        self.num_frames = num_frames
        with open(os.path.join(yaml_root), 'r') as f:
            self.data_info = yaml.safe_load(f)
        f.close()
        
        with open(os.path.join(self.data_root, 'val_open_word.yaml'), 'r') as f:
            self.open_word_info = yaml.safe_load(f)
        f.close()
        
        self.vids = []
        # 将没有标注的都去掉
        for key in list(self.data_info.keys()):
            PART = key.split('_')[0]
            VIDEO_ID = '_'.join(key.split('_')[:2])
            vid_rgb_path = os.path.join(self.data_root, PART, 'rgb_frames', VIDEO_ID, key)
            # print(vid_gt_path)
            # print(glob(vid_gt_path))
            if len(glob(f"{vid_rgb_path}/*.jpg")) >= 2:
                self.vids.append(key)
        
        with open(os.path.join(valset_yaml_root), 'r') as f:
            self.valset_info = yaml.safe_load(f)
        self.repr_type = repr_type
        assert repr_type in ['SlowFast', 'VideoMae']
        assert num_frames == 5

        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        
        self.im_transform = transforms.Compose([
            transforms.Resize((224,224), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            im_normalization,
        ])
        

    def __getitem__(self, idx):
        video_value = self.data_info[self.vids[idx]] # video value
        selected_frames = np.array(self.valset_info[self.vids[idx]]) # sorted selected frames
        all_frames = list(range(video_value['start_frame'], video_value['stop_frame']))
        all_frames = ['frame_' + str(i).zfill(10)+ '.jpg' for i in all_frames]
        info = {}
        assert len(selected_frames) == self.num_frames
        vid_im_path = path.join(self.data_root, video_value['participant_id'], 'rgb_frames', video_value['video_id'], self.vids[idx])
        
        all_images = []
        all_slow_images = []
        # gt_order = np.array(list(range(self.num_frames)))
        gt_order = np.argsort(selected_frames)
        
        if self.repr_type == 'SlowFast':
            near_k = 8
        elif self.repr_type == 'VideoMae':
            near_k = 8
            
        for frame in selected_frames:
            f_idx = all_frames.index(frame)
            # 每个f_idx 附近取k帧
            if f_idx > near_k//2 and f_idx < len(all_frames) - near_k//2:
                k_index_list = list(range(f_idx-near_k//2, f_idx+near_k//2))
            elif f_idx <= near_k//2:
                k_index_list = list(range(0, near_k))
            elif f_idx >= len(all_frames) - near_k//2:
                k_index_list = list(range(len(all_frames)-near_k, len(all_frames)))
            
            images = []
            slow_images = []
            info.update({frame: []})
            
            for k_idx in k_index_list:
                jpg_name = all_frames[k_idx]
                frame_path = os.path.join(vid_im_path, jpg_name)
                info[frame].append(frame_path)
                this_im = Image.open(frame_path).convert('RGB')
                this_im = self.im_transform(this_im)
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
        
        data = {
            'rgb': all_images, # [5, 8/2*near_k, 3, H, W]
            'slow_images': all_slow_images, # [5, 8/2, 3, H, W]
            'gt_order': gt_order, # [num_frames]
            'text':video_value['narration'],
            'open_word_type':self.open_word_info[self.vids[idx]],
        }

        return data

    def __len__(self):
        return len(self.vids)

# old before 1015

if __name__ == '__main__':
    dataset = EPICtestDataset(data_root='../data', yaml_root='../data/EPIC55_cut_subset_200.yaml', max_num_obj=3, finetune=False)
    val_loader = DataLoader(dataset, 1,  shuffle=False, num_workers=4)
    for i, data in enumerate(val_loader):
        print(data['info']['name'][0])
        print(data['rgb'][0].shape)
        print(data['first_frame_gt'][0][0].shape)
        print(data['whether_save_mask'][0][1])
        # print(data[])
        dd
    
    # images = dataset[2]
    # print(f"name={images['first_frame_gt'].shape}")
    
    # for obj in range(images['first_frame_gt'].shape[1]):
    #     plt.imsave(f"../visuals/gt_{obj}.jpg", images['first_frame_gt'][0,obj],cmap='gray')