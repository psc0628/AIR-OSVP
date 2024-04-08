from utils.config import *
from models import build_model_from_cfg
import os
import torch
import argparse
import numpy as np
import time

import open3d as o3d
import copy


def fixedNumDownSamplePCL(initPcd, desiredNumOfPoint, leftVoxelSize, rightVoxelSize):
    """ Use the method voxel_down_sample defined in open3d and do bisection iteratively
        to get the appropriate voxel_size which yields the points with the desired number.
        INPUT:
            initPcd: open3d.geometry.PointCloud
            desiredNumOfPoint: int, the desired number of points after down sampling
            leftVoxelSize: float, the initial bigger voxel size to do bisection
            rightVoxelSize: float, the initial smaller voxel size to do bisection
        OUTPUT:
            pcd: down sampled pointcloud

    """
    assert leftVoxelSize > rightVoxelSize, "leftVoxelSize should be larger than rightVoxelSize"
    assert len(
        initPcd.points) > desiredNumOfPoint, "desiredNumOfPoint should be less than or equal to the num of points in the given point cloud."
    if len(initPcd.points) == desiredNumOfPoint:
        return initPcd

    pcd = copy.deepcopy(initPcd)
    pcd = pcd.voxel_down_sample(leftVoxelSize)
    assert len(pcd.points) <= desiredNumOfPoint, "Please specify a larger leftVoxelSize."
    pcd = copy.deepcopy(initPcd)
    pcd = pcd.voxel_down_sample(rightVoxelSize)
    assert len(pcd.points) >= desiredNumOfPoint, "Please specify a smaller rightVoxelSize."

    pcd = copy.deepcopy(initPcd)
    midVoxelSize = (leftVoxelSize + rightVoxelSize) / 2.
    pcd = pcd.voxel_down_sample(midVoxelSize)

    midVoxelSizeGlobal = -1

    while len(pcd.points) != desiredNumOfPoint:
        if len(pcd.points) < desiredNumOfPoint:
            leftVoxelSize = copy.copy(midVoxelSize)
        else:
            rightVoxelSize = copy.copy(midVoxelSize)
        midVoxelSize = (leftVoxelSize + rightVoxelSize) / 2.
        pcd = copy.deepcopy(initPcd)
        pcd = pcd.voxel_down_sample(midVoxelSize)
        # 如果体素大小不变
        if midVoxelSizeGlobal == midVoxelSize:
            break
        midVoxelSizeGlobal = midVoxelSize

    return pcd

def parse_args():
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--experiment_path', default='./pointrVP/trained_network', help='experiment path')
    parser.add_argument('--ckpt_path', default='best_ckpt.pth', help='specify the checkpoint file to be loaded')
    parser.add_argument('--infer_file', required=True, help='file the objects to be infered')
    parser.add_argument('--num_of_test', type=int, default=500, help='num of test to choose the largest view set')
    parser.add_argument('--num_of_max_points_memery', type=int, default=20480, help='num of point can be stored in gpu')
    args = parser.parse_args()

    return args

def load_data(pc, view_state_path, gt_path=None, do_sample=True, num_points=2048):
    if pc.shape[0] > args.num_of_max_points_memery:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        # o3d.io.write_point_cloud("./data/test.ply", pcd)
        downsampled_pcd = fixedNumDownSamplePCL(pcd, args.num_of_max_points_memery, 0.012, 0.002)
        # o3d.io.write_point_cloud("./data/test_ds.ply", downsampled_pcd)
        print('downsample to num:' + str(len(downsampled_pcd.points)))
        pc = np.asarray(downsampled_pcd.points)

    pc = torch.from_numpy(pc).float()

    vs = np.loadtxt(view_state_path, dtype=np.float32)
    vs = torch.from_numpy(vs).float()

    if gt_path:
        idx = torch.tensor(np.loadtxt(gt_path, dtype=np.long))
        gt = torch.zeros(32)
        gt[idx] = 1
    else:
        gt = None
    return pc, vs, gt

def infer(args):
    config_path = os.path.join(args.experiment_path, 'config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(config_path):
        print(f'config file {config_path} does not exist')
        exit(1)

    checkpoint_path = os.path.join(args.experiment_path, args.ckpt_path)
    if not os.path.exists(checkpoint_path):
        print(f'checkpoint file {checkpoint_path} does not exist')
        exit(1)
    else:
        print(f"loading checkpoint from {checkpoint_path}")

    config = cfg_from_yaml_file(config_path)

    # print(config)
    base_model = build_model_from_cfg(config.model).to(device)
    base_model.load_state_dict(torch.load(checkpoint_path)['base_model'])

    base_model.eval()

    pointcloud_path = args.infer_file
    view_state_path = args.infer_file.replace('pc', 'vs')
    # gt_path = args.infer_file.replace('pc', 'ids')
    gt_path = None
    label_thresh = config.threshold_gamma

    run_time = 0.0

    pc_ori = np.loadtxt(pointcloud_path, dtype=np.float32)

    pc, vs, gt = load_data(pc_ori, view_state_path, gt_path=gt_path, do_sample=config.do_sample, num_points=config.model.num_points)
    pc = pc.unsqueeze(0).to(device)
    vs = vs.unsqueeze(0).to(device)

    with torch.no_grad():
        startTime = time.time()
        output = base_model(pc, vs)
        run_time += time.time() - startTime

        output = output.squeeze(0)
        pred_vs = torch.where(output >= label_thresh, torch.tensor(1).to(device), torch.tensor(0).to(device))
        pred_vs = torch.nonzero(pred_vs).squeeze().tolist()

        pred_vs = [pred_vs] if isinstance(pred_vs, int) else pred_vs
        print(pred_vs)

    return pred_vs, run_time

if __name__ == "__main__":
    args = parse_args()

    pred_vs, run_time = infer(args)

    gt_path = args.infer_file.replace('pc', 'ids')
    np.savetxt(gt_path, np.asarray([pred_vs]), fmt='%d')

    print('run time is ' + str(run_time))
    run_time_path = args.infer_file.replace('pc', 'time_pointrvp')
    run_time_path = run_time_path.replace('/data/', '/run_time/')
    np.savetxt(run_time_path, np.asarray([run_time]), fmt='%.6f')

# python infer_once.py --infer_file ../data/Dragon_r3_v0_pc.txt