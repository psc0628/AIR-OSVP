import os
import torch
import numpy as np
import torch.utils.data as data
import numpy as np
from pointnet2_ops import pointnet2_utils
from utils.logger import *


class COSC_Dataset(data.Dataset):
    def __init__(self, root, obj_list, num_points, logger=None, do_sample=True):
        # self.file_list = ['/home/media/Documents/PoinTr/COSC/Asian_Dragon/rotate_0/cloud_0.txt']
        self.file_list = []
        self.num_points = num_points
        self.do_sample = do_sample

        print_log(f'[DATASET]: loading data from {root}', logger=logger)
        print_log(f'[DATASET]: load data from {obj_list}', logger=logger)
        print_log(f'[DATASET]: load {num_points} points per instance', logger=logger)
        print_log(f"[DATASET]: {'do sampling' if do_sample else 'no sampling'}", logger=logger)

        for obj in obj_list:
            obj_path = os.path.join(root, obj)
            for rot in os.listdir(obj_path):
                rot_path = os.path.join(obj_path, rot)
                for f in os.listdir(rot_path):
                    if f.startswith('cloud'):
                        self.file_list.append(os.path.join(rot_path, f))
        print_log(f'[DATASET]: {len(self.file_list)} instances were loaded')

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
    
    def fps(self, pc):
        fps_idx = pointnet2_utils.furthest_point_sample(pc, self.num_points) 
        return pc[fps_idx]
        
    def __getitem__(self, idx):
        pointcloud_path = self.file_list[idx]
        view_state_path = self.file_list[idx].replace('cloud', 'state')
        gt_path = self.file_list[idx].replace('cloud', 'ids')

        pc = np.loadtxt(pointcloud_path, dtype=np.float32)
        if self.do_sample:
            sampled_indices = np.random.choice(pc.shape[0], size=self.num_points, replace=False if pc.shape[0] > self.num_points else True)
            pc = pc[sampled_indices]
        pc = torch.from_numpy(pc).float()

        vs = np.loadtxt(view_state_path, dtype=np.float32)
        vs = torch.from_numpy(vs).float()

        idx = torch.tensor(np.loadtxt(gt_path, dtype=np.long))
        gt = torch.zeros(32)
        gt[idx] = 1
        return pc, vs, gt

    def __len__(self):
        return len(self.file_list)