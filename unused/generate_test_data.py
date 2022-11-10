import yaml
import os
import numpy as np

yaml_root = '/home/venom/projects/XMem/val_data/EPIC100_state_positive_val.yaml'
data_root = '/home/venom/data/EPIC_val_split'
with open(os.path.join(yaml_root), 'r') as f:
    data_info = yaml.safe_load(f)
f.close()

valid_info = {}
# 将没有标注的都去掉
for key in list(data_info.keys()):
    PART = key.split('_')[0]
    VIDEO_ID = '_'.join(key.split('_')[:2])
    vid_rgb_path = os.path.join(data_root, PART, 'rgb_frames', VIDEO_ID, key)
    
    start_frame = data_info[key]['start_frame']
    end_frame = data_info[key]['stop_frame']
    all_frames = list(range(start_frame, end_frame))
    selected_frames = np.random.choice(all_frames, size=5, replace=False)
    selected_frames = np.array(list(sorted(selected_frames)))
    gt_order = np.array(list(range(5)))
    
    noise = np.random.rand(5)  # noise in [0, 1]
    idx = np.argsort(noise)
    # 打乱顺序
    selected_frames = selected_frames[idx]
    gt_order = gt_order[idx]
    print(gt_order)
    for frame in selected_frames:
        jpg_path = os.path.join(vid_rgb_path, 'frame_' + str(frame).zfill(10) + '.jpg')
        if not os.path.exists(jpg_path):
            raise NotImplementedError
    valid_info.update({key: ['frame_' + str(frame).zfill(10) + '.jpg' for frame in selected_frames]})

with open('./reordering_val.yaml', 'w') as f:
    yaml.dump(valid_info, f)