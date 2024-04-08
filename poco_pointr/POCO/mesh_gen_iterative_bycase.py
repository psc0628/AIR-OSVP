import os
import numpy as np
import yaml
from tqdm import tqdm
import logging
import shutil

from sklearn.metrics import confusion_matrix
import torch_geometric.transforms as T

# torch imports
import torch
import torch.nn.functional as F

# lightconvpoint imports
from lightconvpoint.datasets.dataset import get_dataset
import lightconvpoint.utils.transforms as lcp_T
from lightconvpoint.utils.logs import logs_file
from lightconvpoint.utils.misc import dict_to_device

import utils.argparseFromFile as argparse
from utils.utils import wblue, wgreen
import utils.metrics as metrics
import datasets
import networks
import time
import math
from skimage import measure
import open3d as o3d

from torch.utils.tensorboard import SummaryWriter
from objprint import objprint as op


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_config_file(config, filename):
    with open(filename, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def export_pointcloud_region_growing_v2(
        network,
        latent,
        device=None,
        input_points=None,
):
    latent["pos_non_manifold"] = torch.from_numpy(input_points).to(device).unsqueeze(0).float()
    occ = network.from_latent(latent)
    occ = F.softmax(occ, dim=1)
    occ_np = occ.cpu().detach().numpy()
    pred_labels = np.argmax(occ_np, axis=1)
    return pred_labels


def export_mesh_and_refine_vertices_region_growing_v2(
        network,
        latent,
        padding=0,
        mc_value=0,
        device=None,
        num_pts=50000,
        refine_iter=10,
        simplification_target=None,
        input_points=None,
        refine_threshold=None,
        out_value=np.nan,
        step=None,
        dilation_size=2,
        return_volume=False
):
    bmin = np.min(input_points, axis=0)
    bmax = np.max(input_points, axis=0)
    print(bmin, bmax)

    resolutionX = math.ceil((bmax[0] - bmin[0]) / step)
    resolutionY = math.ceil((bmax[1] - bmin[1]) / step)
    resolutionZ = math.ceil((bmax[2] - bmin[2]) / step)

    bmin_pad = bmin - padding * step
    print(bmin_pad)

    pts_ids = (input_points - bmin) / step + padding
    pts_ids = pts_ids.astype(np.int32)

    # create the volume
    volume = np.full((resolutionX + 2 * padding, resolutionY + 2 * padding, resolutionZ + 2 * padding), np.nan,
                     dtype=np.float64)
    mask_to_see = np.full((resolutionX + 2 * padding, resolutionY + 2 * padding, resolutionZ + 2 * padding), True,
                          dtype=bool)
    while (pts_ids.shape[0] > 0):

        print("Pts", pts_ids.shape)

        # creat the mask
        mask = np.full((resolutionX + 2 * padding, resolutionY + 2 * padding, resolutionZ + 2 * padding), False,
                       dtype=bool)
        mask[pts_ids[:, 0], pts_ids[:, 1], pts_ids[:, 2]] = True

        # dilation
        for i in tqdm(range(pts_ids.shape[0]), ncols=100, disable=True):
            xc = int(pts_ids[i, 0])
            yc = int(pts_ids[i, 1])
            zc = int(pts_ids[i, 2])
            mask[max(0, xc - dilation_size):xc + dilation_size,
            max(0, yc - dilation_size):yc + dilation_size,
            max(0, zc - dilation_size):zc + dilation_size] = True

        # get the valid points
        valid_points_coord = np.argwhere(mask).astype(np.int32)
        valid_points = valid_points_coord * step + bmin_pad
        print(valid_points.shape)

        z = []
        near_surface_samples_torch = torch.tensor(valid_points, dtype=torch.float, device=device)
        for pnts in tqdm(torch.split(near_surface_samples_torch, num_pts, dim=0), ncols=100, disable=True):
            latent["pos_non_manifold"] = pnts.unsqueeze(0)
            occ_hat = network.from_latent(latent)
            class_dim = 1
            occ_hat = torch.stack([occ_hat[:, class_dim],
                                   occ_hat[:, [i for i in range(occ_hat.shape[1]) if i != class_dim]].max(dim=1)[0]],
                                  dim=1)
            occ_hat = F.softmax(occ_hat, dim=1)
            occ_hat[:, 0] = occ_hat[:, 0] * (-1)
            if class_dim == 0:
                occ_hat = occ_hat * (-1)

            occ_hat = occ_hat.sum(dim=1)
            outputs = occ_hat.squeeze(0)
            z.append(outputs.detach().cpu().numpy())

        z = np.concatenate(z, axis=0)
        z = z.astype(np.float64)

        # update the volume
        volume[mask] = z

        # create the masks
        mask_pos = np.full((resolutionX + 2 * padding, resolutionY + 2 * padding, resolutionZ + 2 * padding), False,
                           dtype=bool)
        mask_neg = np.full((resolutionX + 2 * padding, resolutionY + 2 * padding, resolutionZ + 2 * padding), False,
                           dtype=bool)

        # dilation
        for i in tqdm(range(pts_ids.shape[0]), ncols=100, disable=True):
            xc = int(pts_ids[i, 0])
            yc = int(pts_ids[i, 1])
            zc = int(pts_ids[i, 2])
            mask_to_see[xc, yc, zc] = False
            if volume[xc, yc, zc] <= 0:
                mask_neg[max(0, xc - dilation_size):xc + dilation_size,
                max(0, yc - dilation_size):yc + dilation_size,
                max(0, zc - dilation_size):zc + dilation_size] = True
            if volume[xc, yc, zc] >= 0:
                mask_pos[max(0, xc - dilation_size):xc + dilation_size,
                max(0, yc - dilation_size):yc + dilation_size,
                max(0, zc - dilation_size):zc + dilation_size] = True

        # get the new points
        new_mask = (mask_neg & (volume >= 0) & mask_to_see) | (mask_pos & (volume <= 0) & mask_to_see)
        pts_ids = np.argwhere(new_mask).astype(np.int32)

    volume[0:padding, :, :] = out_value
    volume[-padding:, :, :] = out_value
    volume[:, 0:padding, :] = out_value
    volume[:, -padding:, :] = out_value
    volume[:, :, 0:padding] = out_value
    volume[:, :, -padding:] = out_value

    # volume[np.isnan(volume)] = out_value
    maxi = volume[~np.isnan(volume)].max()
    mini = volume[~np.isnan(volume)].min()

    if not (maxi > mc_value and mini < mc_value):
        return None

    if return_volume:
        return volume

    # compute the marching cubes
    verts, faces, _, _ = measure.marching_cubes(
        volume=volume.copy(),
        level=mc_value,
    )
    print(f"verts.shape: {verts.shape}, faces.shape: {faces.shape}")
    # removing the nan values in the vertices
    values = verts.sum(axis=1)
    o3d_verts = o3d.utility.Vector3dVector(verts)
    o3d_faces = o3d.utility.Vector3iVector(faces)
    mesh = o3d.geometry.TriangleMesh(o3d_verts, o3d_faces)
    mesh.remove_vertices_by_mask(np.isnan(values))
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    if refine_iter > 0:

        dirs = verts - np.floor(verts)
        dirs = (dirs > 0).astype(dirs.dtype)

        mask = np.logical_and(dirs.sum(axis=1) > 0, dirs.sum(axis=1) < 2)
        v = verts[mask]
        dirs = dirs[mask]

        # initialize the two values (the two vertices for mc grid)
        v1 = np.floor(v)
        v2 = v1 + dirs

        # get the predicted values for both set of points
        v1 = v1.astype(int)
        v2 = v2.astype(int)
        preds1 = volume[v1[:, 0], v1[:, 1], v1[:, 2]]
        preds2 = volume[v2[:, 0], v2[:, 1], v2[:, 2]]

        # get the coordinates in the real coordinate system
        v1 = v1.astype(np.float32) * step + bmin_pad
        v2 = v2.astype(np.float32) * step + bmin_pad

        # tmp mask
        mask_tmp = np.logical_and(
            np.logical_not(np.isnan(preds1)),
            np.logical_not(np.isnan(preds2))
        )
        v = v[mask_tmp]
        dirs = dirs[mask_tmp]
        v1 = v1[mask_tmp]
        v2 = v2[mask_tmp]
        mask[mask] = mask_tmp

        # initialize the vertices
        verts = verts * step + bmin_pad
        v = v * step + bmin_pad

        # iterate for the refinement step
        for iter_id in tqdm(range(refine_iter), ncols=50, disable=True):

            print(f"iter {iter_id}")

            preds = []
            pnts_all = torch.tensor(v, dtype=torch.float, device=device)
            # print(pnts_all)
            for pnts in tqdm(torch.split(pnts_all, num_pts, dim=0), ncols=100, disable=True):

                latent["pos_non_manifold"] = pnts.unsqueeze(0)
                occ_hat = network.from_latent(latent)

                # get class and max non class
                class_dim = 1
                occ_hat = torch.stack([occ_hat[:, class_dim],
                                       occ_hat[:, [i for i in range(occ_hat.shape[1]) if i != class_dim]].max(dim=1)[
                                           0]], dim=1)
                occ_hat = F.softmax(occ_hat, dim=1)
                occ_hat[:, 0] = occ_hat[:, 0] * (-1)
                if class_dim == 0:
                    occ_hat = occ_hat * (-1)

                # occ_hat = -occ_hat.sum(dim=1)
                occ_hat = occ_hat.sum(dim=1)
                outputs = occ_hat.squeeze(0)

                # outputs = network.predict_from_latent(latent, pnts.unsqueeze(0), with_sigmoid=True)
                # outputs = outputs.squeeze(0)
                preds.append(outputs.detach().cpu().numpy())
            preds = np.concatenate(preds, axis=0)

            mask1 = (preds * preds1) > 0
            v1[mask1] = v[mask1]
            preds1[mask1] = preds[mask1]

            mask2 = (preds * preds2) > 0
            v2[mask2] = v[mask2]
            preds2[mask2] = preds[mask2]

            v = (v2 + v1) / 2

            verts[mask] = v

            # keep only the points that needs to be refined
            if refine_threshold is not None:
                mask_vertices = (np.linalg.norm(v2 - v1, axis=1) > refine_threshold)
                # print("V", mask_vertices.sum() , "/", v.shape[0])
                v = v[mask_vertices]
                preds1 = preds1[mask_vertices]
                preds2 = preds2[mask_vertices]
                v1 = v1[mask_vertices]
                v2 = v2[mask_vertices]
                mask[mask] = mask_vertices

                if v.shape[0] == 0:
                    break
                # print("V", v.shape[0])

    else:
        verts = verts * step + bmin_pad

    o3d_verts = o3d.utility.Vector3dVector(verts)
    o3d_faces = o3d.utility.Vector3iVector(faces)
    mesh = o3d.geometry.TriangleMesh(o3d_verts, o3d_faces)

    if simplification_target is not None and simplification_target > 0:
        mesh = o3d.geometry.TriangleMesh.simplify_quadric_decimation(mesh, simplification_target)

    return mesh


def export_mesh_and_refine_vertices_region_growing_v3(
        network, latent,
        resolution,
        padding=0,
        mc_value=0,
        device=None,
        num_pts=50000,
        refine_iter=10,
        simplification_target=None,
        input_points=None,
        refine_threshold=None,
        out_value=np.nan,
        step=None,
        dilation_size=2,
        whole_negative_component=False,
        return_volume=False
):
    bmin = input_points.min()
    bmax = input_points.max()
    # print(input_points.shape)
    if step is None:
        step = (bmax - bmin) / (resolution - 1)
        resolutionX = resolution
        resolutionY = resolution
        resolutionZ = resolution
    else:
        bmin = input_points.min(axis=0)
        bmax = input_points.max(axis=0)
        resolutionX = math.ceil((bmax[0] - bmin[0]) / step)
        resolutionY = math.ceil((bmax[1] - bmin[1]) / step)
        resolutionZ = math.ceil((bmax[2] - bmin[2]) / step)

    bmin_pad = bmin - padding * step
    bmax_pad = bmax + padding * step

    pts_ids = (input_points - bmin) / step + padding
    pts_ids = pts_ids.astype(np.int32)

    # create the volume
    volume = np.full((resolutionX + 2 * padding, resolutionY + 2 * padding, resolutionZ + 2 * padding), np.nan,
                     dtype=np.float64)
    mask_to_see = np.full((resolutionX + 2 * padding, resolutionY + 2 * padding, resolutionZ + 2 * padding), True,
                          dtype=bool)
    while (pts_ids.shape[0] > 0):

        # print("Pts", pts_ids.shape)

        # creat the mask
        mask = np.full((resolutionX + 2 * padding, resolutionY + 2 * padding, resolutionZ + 2 * padding), False,
                       dtype=bool)
        mask[pts_ids[:, 0], pts_ids[:, 1], pts_ids[:, 2]] = True

        # dilation
        for i in tqdm(range(pts_ids.shape[0]), ncols=100, disable=True):
            xc = int(pts_ids[i, 0])
            yc = int(pts_ids[i, 1])
            zc = int(pts_ids[i, 2])
            mask[max(0, xc - dilation_size):xc + dilation_size,
            max(0, yc - dilation_size):yc + dilation_size,
            max(0, zc - dilation_size):zc + dilation_size] = True

        # get the valid points
        valid_points_coord = np.argwhere(mask).astype(np.int32)
        valid_points = valid_points_coord * step + bmin_pad
        z = []
        near_surface_samples_torch = torch.tensor(valid_points, dtype=torch.float, device=device)
        for pnts in tqdm(torch.split(near_surface_samples_torch, num_pts, dim=0), ncols=100, disable=True):
            latent["pos_non_manifold"] = pnts.unsqueeze(0)
            occ_hat = network.from_latent(latent)
            class_dim = 1
            occ_hat = torch.stack([occ_hat[:, class_dim],
                                   occ_hat[:, [i for i in range(occ_hat.shape[1]) if i != class_dim]].max(dim=1)[0]],
                                  dim=1)
            occ_hat = F.softmax(occ_hat, dim=1)
            occ_hat[:, 0] = occ_hat[:, 0] * (-1)
            if class_dim == 0:
                occ_hat = occ_hat * (-1)

            # occ_hat = -occ_hat.sum(dim=1)
            occ_hat = occ_hat.sum(dim=1)
            outputs = occ_hat.squeeze(0)
            z.append(outputs.detach().cpu().numpy())

        z = np.concatenate(z, axis=0)
        z = z.astype(np.float64)

        # update the volume
        volume[mask] = z

        # create the masks
        mask_pos = np.full((resolutionX + 2 * padding, resolutionY + 2 * padding, resolutionZ + 2 * padding), False,
                           dtype=bool)
        mask_neg = np.full((resolutionX + 2 * padding, resolutionY + 2 * padding, resolutionZ + 2 * padding), False,
                           dtype=bool)

        # dilation
        for i in tqdm(range(pts_ids.shape[0]), ncols=100, disable=True):
            xc = int(pts_ids[i, 0])
            yc = int(pts_ids[i, 1])
            zc = int(pts_ids[i, 2])
            mask_to_see[xc, yc, zc] = False
            if volume[xc, yc, zc] <= 0:
                mask_neg[max(0, xc - dilation_size):xc + dilation_size,
                max(0, yc - dilation_size):yc + dilation_size,
                max(0, zc - dilation_size):zc + dilation_size] = True
            if volume[xc, yc, zc] >= 0:
                mask_pos[max(0, xc - dilation_size):xc + dilation_size,
                max(0, yc - dilation_size):yc + dilation_size,
                max(0, zc - dilation_size):zc + dilation_size] = True

        # get the new points

        new_mask = (mask_neg & (volume >= 0) & mask_to_see) | (mask_pos & (volume <= 0) & mask_to_see)
        pts_ids = np.argwhere(new_mask).astype(np.int32)

    volume[0:padding, :, :] = out_value
    volume[-padding:, :, :] = out_value
    volume[:, 0:padding, :] = out_value
    volume[:, -padding:, :] = out_value
    volume[:, :, 0:padding] = out_value
    volume[:, :, -padding:] = out_value

    # volume[np.isnan(volume)] = out_value
    maxi = volume[~np.isnan(volume)].max()
    mini = volume[~np.isnan(volume)].min()

    if not (maxi > mc_value and mini < mc_value):
        return None

    if return_volume:
        return volume

    # compute the marching cubes
    verts, faces, _, _ = measure.marching_cubes(
        volume=volume.copy(),
        level=mc_value,
    )

    # removing the nan values in the vertices
    values = verts.sum(axis=1)
    o3d_verts = o3d.utility.Vector3dVector(verts)
    o3d_faces = o3d.utility.Vector3iVector(faces)
    mesh = o3d.geometry.TriangleMesh(o3d_verts, o3d_faces)
    mesh.remove_vertices_by_mask(np.isnan(values))
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    if refine_iter > 0:

        dirs = verts - np.floor(verts)
        dirs = (dirs > 0).astype(dirs.dtype)

        mask = np.logical_and(dirs.sum(axis=1) > 0, dirs.sum(axis=1) < 2)
        v = verts[mask]
        dirs = dirs[mask]

        # initialize the two values (the two vertices for mc grid)
        v1 = np.floor(v)
        v2 = v1 + dirs

        # get the predicted values for both set of points
        v1 = v1.astype(int)
        v2 = v2.astype(int)
        preds1 = volume[v1[:, 0], v1[:, 1], v1[:, 2]]
        preds2 = volume[v2[:, 0], v2[:, 1], v2[:, 2]]

        # get the coordinates in the real coordinate system
        v1 = v1.astype(np.float32) * step + bmin_pad
        v2 = v2.astype(np.float32) * step + bmin_pad

        # tmp mask
        mask_tmp = np.logical_and(
            np.logical_not(np.isnan(preds1)),
            np.logical_not(np.isnan(preds2))
        )
        v = v[mask_tmp]
        dirs = dirs[mask_tmp]
        v1 = v1[mask_tmp]
        v2 = v2[mask_tmp]
        mask[mask] = mask_tmp

        # initialize the vertices
        verts = verts * step + bmin_pad
        v = v * step + bmin_pad

        # iterate for the refinement step
        for iter_id in tqdm(range(refine_iter), ncols=50, disable=True):

            # print(f"iter {iter_id}")

            preds = []
            pnts_all = torch.tensor(v, dtype=torch.float, device=device)
            # print(pnts_all)
            for pnts in tqdm(torch.split(pnts_all, num_pts, dim=0), ncols=100, disable=True):

                latent["pos_non_manifold"] = pnts.unsqueeze(0)
                occ_hat = network.from_latent(latent)

                # get class and max non class
                class_dim = 1
                occ_hat = torch.stack([occ_hat[:, class_dim],
                                       occ_hat[:, [i for i in range(occ_hat.shape[1]) if i != class_dim]].max(dim=1)[
                                           0]], dim=1)
                occ_hat = F.softmax(occ_hat, dim=1)
                occ_hat[:, 0] = occ_hat[:, 0] * (-1)
                if class_dim == 0:
                    occ_hat = occ_hat * (-1)

                # occ_hat = -occ_hat.sum(dim=1)
                occ_hat = occ_hat.sum(dim=1)
                outputs = occ_hat.squeeze(0)

                # outputs = network.predict_from_latent(latent, pnts.unsqueeze(0), with_sigmoid=True)
                # outputs = outputs.squeeze(0)
                preds.append(outputs.detach().cpu().numpy())
            preds = np.concatenate(preds, axis=0)

            mask1 = (preds * preds1) > 0
            v1[mask1] = v[mask1]
            preds1[mask1] = preds[mask1]

            mask2 = (preds * preds2) > 0
            v2[mask2] = v[mask2]
            preds2[mask2] = preds[mask2]

            v = (v2 + v1) / 2

            verts[mask] = v

            # keep only the points that needs to be refined
            if refine_threshold is not None:
                mask_vertices = (np.linalg.norm(v2 - v1, axis=1) > refine_threshold)
                # print("V", mask_vertices.sum() , "/", v.shape[0])
                v = v[mask_vertices]
                preds1 = preds1[mask_vertices]
                preds2 = preds2[mask_vertices]
                v1 = v1[mask_vertices]
                v2 = v2[mask_vertices]
                mask[mask] = mask_vertices

                if v.shape[0] == 0:
                    break
                # print("V", v.shape[0])

    else:
        verts = verts * step + bmin_pad

    o3d_verts = o3d.utility.Vector3dVector(verts)
    o3d_faces = o3d.utility.Vector3iVector(faces)
    mesh = o3d.geometry.TriangleMesh(o3d_verts, o3d_faces)

    if simplification_target is not None and simplification_target > 0:
        mesh = o3d.geometry.TriangleMesh.simplify_quadric_decimation(mesh, simplification_target)

    return mesh


def main(config, args):
    config = eval(str(config))
    device = torch.device(config['device'])
    if config["device"] == "cuda":
        torch.backends.cudnn.benchmark = True
    logging.getLogger().setLevel(config["logging"])

    # create the network
    N_LABELS = config["network_n_labels"]
    latent_size = config["network_latent_size"]
    backbone = config["network_backbone"]
    decoder = {'name': config["network_decoder"], 'k': config['network_decoder_k']}

    logging.info("Creating the network")

    def network_function():
        return networks.Network(3, latent_size, N_LABELS, backbone, decoder)

    net = network_function()
    net.to(device)

    logging.info("Getting the dataset")
    DatasetClass = get_dataset(eval("datasets." + config["dataset_name"]))
    train_transform = []
    test_transform = []

    # downsample 
    train_transform.append(
        lcp_T.FixedPoints(config["manifold_points"], item_list=["x", "pos", "normal", "y", "y_object"]))
    train_transform.append(lcp_T.FixedPoints(config["non_manifold_points"],
                                             item_list=["pos_non_manifold", "occupancies", "y_v", "y_v_object"]))

    random_rotation_x = config["training_random_rotation_x"]
    random_rotation_y = config["training_random_rotation_y"]
    random_rotation_z = config["training_random_rotation_z"]
    if random_rotation_x is not None and random_rotation_x > 0:
        train_transform += [
            lcp_T.RandomRotate(random_rotation_x, axis=0, item_list=["pos", "normal", "pos_non_manifold"]), ]
    if random_rotation_y is not None and random_rotation_y > 0:
        train_transform += [
            lcp_T.RandomRotate(random_rotation_y, axis=1, item_list=["pos", "normal", "pos_non_manifold"]), ]
    if random_rotation_z is not None and random_rotation_z > 0:
        train_transform += [
            lcp_T.RandomRotate(random_rotation_z, axis=2, item_list=["pos", "normal", "pos_non_manifold"]), ]

    # add noise to data
    if (config["random_noise"] is not None) and (config["random_noise"] > 0):
        train_transform.append(lcp_T.RandomNoiseNormal(sigma=config["random_noise"]))
        test_transform.append(lcp_T.RandomNoiseNormal(sigma=config["random_noise"]))

    if config["normals"]:
        logging.info("Normals as features")
        test_transform.append(lcp_T.FieldAsFeatures(["normal"]))

    # operate the permutations
    train_transform = train_transform + [lcp_T.Permutation("pos", [1, 0]),
                                         lcp_T.Permutation("pos_non_manifold", [1, 0]),
                                         lcp_T.Permutation("normal", [1, 0]),
                                         lcp_T.Permutation("x", [1, 0]),
                                         lcp_T.ToDict(), ]

    train_transform = T.Compose(train_transform)

    # 从这里开始修改路径

    cosc_root_dir = args.case_path
    save_root_dir = args.case_path.replace('Finals', 'Finals_results')
    bbx_dir = args.case_path.replace('Finals', 'bbx')
    bbx_dir = bbx_dir[0:bbx_dir.find('bbx') + 3]
    obj_name = args.obj_name

    print('cosc_root_dir:' + cosc_root_dir)
    print('save_root_dir:' + save_root_dir)
    print('bbx_dir:' + bbx_dir)
    print('obj_name:' + obj_name)

    if not os.path.exists(save_root_dir):
        os.makedirs(save_root_dir)

    scene_file_path = os.path.join(cosc_root_dir, "scene.txt")
    logging.info(f"loading data from {scene_file_path}")

    savedir_root = save_root_dir
    last_saved_file = os.path.join(savedir_root, f"out_view_best.ply")
    if os.path.exists(last_saved_file):
        print(f'{last_saved_file} is already exists')
        return
    logging.info(f"save file to: {savedir_root}")

    train_dataset = DatasetClass(root=config["dataset_root"],
                                 scene_file=scene_file_path,
                                 transform=train_transform,
                                 network_function=network_function,
                                 filter_name=config["filter_name"],
                                 num_non_manifold_points=config["non_manifold_points"],
                                 )

    # build the data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        # num_workers=config["threads"],
    )

    # create the optimizer
    logging.info("Creating the optimizer")
    optimizer = torch.optim.Adam(net.parameters(), config["training_lr_start"])

    # load shapenet checkpoint
    logging.info("loading from shapenet checkpoint")
    checkpoint = torch.load("./POCO/checkpoint.pth", map_location=device)
    net.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch_start = checkpoint["epoch"]
    train_iter_count = len(train_loader) * epoch_start

    os.makedirs(savedir_root, exist_ok=True)
    loss_layer = torch.nn.CrossEntropyLoss()

    # create the summary writer
    logging.info("Creating tensorboard summary writer")
    writer = SummaryWriter(log_dir=os.path.join(savedir_root, "logs_tb"))

    epoch = epoch_start
    best_train_iou = 0.0
    best_net_state_dict = None
    time_start = time.time()

    while epoch < 461:
        logging.info(f"Training epoch {epoch}")

        net.train()
        error = 0
        cm = np.zeros((N_LABELS, N_LABELS))

        t = tqdm(
            train_loader,
            leave=True
        )
        for data in t:
            data = dict_to_device(data, device)
            optimizer.zero_grad()

            outputs = net(data, spectral_only=True)
            occupancies = data["occupancies"]
            loss = loss_layer(outputs, occupancies)
            loss.backward()
            optimizer.step()

            # compute scores
            output_np = np.argmax(outputs.cpu().detach().numpy(), axis=1)
            target_np = occupancies.cpu().numpy()

            cm_ = confusion_matrix(
                target_np.ravel(), output_np.ravel(), labels=list(range(N_LABELS))
            )
            cm += cm_
            error += loss.item()

            # point wise scores on training
            train_oa = metrics.stats_overall_accuracy(cm)
            train_aa = metrics.stats_accuracy_per_class(cm)[0]
            train_iou = metrics.stats_iou_per_class(cm)[0]
            train_aloss = error / cm.sum()

            description = f"Epoch {epoch} | OA {train_oa * 100:.2f} | AA {train_aa * 100:.2f} | IoU {train_iou * 100:.2f} | Loss {train_aloss:.4e}"
            t.set_description_str(wblue(description))

        # save the logs
        train_log_data = {
            "OA_train": train_oa,
            "AA_train": train_aa,
            "IoU_train": train_iou,
            "Loss_train": train_aloss,
        }
        if best_train_iou < train_iou:
            best_train_iou = train_iou
            best_net_state_dict = net.state_dict()

        logs_file(os.path.join(savedir_root, f"logs_train.csv"), train_iter_count, train_log_data)

        # tensorboard logging
        writer.add_scalar('Loss/loss_train', train_aloss, train_iter_count)
        writer.add_scalar('Metrics/iou_train', train_iou, train_iter_count)

        epoch += 1

        # if epoch in [400, 550, 700, 850, 999]:
        # if epoch in [310, 360, 410, 460]:
        #     infer_bbx_path = os.path.join(bbx_dir, obj_name + '.txt')
        #     print(f"loading inp_pts from {infer_bbx_path}")
        #     inp_pts = np.loadtxt(infer_bbx_path)[:, :3]
        #     logging.info(f"loading bbx from {infer_bbx_path}")
        #
        #     logging.info("predicting all points")
        #     with torch.no_grad():
        #         for data in t:
        #             data = dict_to_device(data, device)
        #
        #             pts = data["pos"][0].transpose(1, 0).cpu().numpy()
        #             nls = data["x"][0].transpose(1, 0).cpu().numpy()
        #             pts = np.concatenate([pts, nls], axis=1)
        #             pts = pts.astype(np.float16)
        #
        #             latent = net.get_latent(data, with_correction=False)
        #
        #             print("predicting mesh")
        #             # mesh = export_mesh_and_refine_vertices_region_growing_v2(net, latent, device=device, input_points=inp_pts, padding=1, step=0.002, out_value=1,num_pts=25000, refine_iter=10)
        #             mesh = export_mesh_and_refine_vertices_region_growing_v3(
        #                 net, latent,
        #                 resolution=args.resolution,
        #                 padding=1,
        #                 mc_value=0,
        #                 device=device,
        #                 # input_points=data["pos"][0].cpu().numpy().transpose(1,0),
        #                 input_points=inp_pts,
        #                 refine_iter=10,
        #                 out_value=1,
        #                 step=None
        #             )
        #             time_pause = time.time()
        #             print(f"meshing time: {time_pause - time_start}")
        #             with open(os.path.join(savedir_root, 'time_cost.txt'), 'a+') as f:
        #                 f.write(f"meshing time: {time_pause - time_start}\n")
        #             save_path = os.path.join(savedir_root, f"out_view_{epoch}.ply")
        #             o3d.io.write_triangle_mesh(save_path, mesh)

    if best_net_state_dict is not None:
        print(f"saving pth to {os.path.join(savedir_root, 'checkpoint_best.pth')}")
        logging.info("loading best checkpoint")
        net.load_state_dict(best_net_state_dict)

        # infer_bbx_path = os.path.join(method_dir, obj, "full_bbx.txt")
        infer_bbx_path = os.path.join(bbx_dir, obj_name + '.txt')
        inp_pts = np.loadtxt(infer_bbx_path)[:, :3]
        logging.info(f"loading bbx from {infer_bbx_path}")

        logging.info("predicting all points")
        with torch.no_grad():
            for data in t:
                data = dict_to_device(data, device)

                pts = data["pos"][0].transpose(1, 0).cpu().numpy()
                nls = data["x"][0].transpose(1, 0).cpu().numpy()
                pts = np.concatenate([pts, nls], axis=1)
                pts = pts.astype(np.float16)

                latent = net.get_latent(data, with_correction=False)

                print("predicting mesh")
                mesh = export_mesh_and_refine_vertices_region_growing_v3(
                    net, latent,
                    resolution=args.resolution,
                    padding=1,
                    mc_value=0,
                    device=device,
                    # input_points=data["pos"][0].cpu().numpy().transpose(1,0),
                    input_points=inp_pts,
                    refine_iter=10,
                    out_value=1,
                    step=None
                )
                time_end = time.time()
                print(f"meshing time: {time_end - time_start}")
                with open(os.path.join(savedir_root, 'time_cost.txt'), 'a+') as f:
                    f.write(f"meshing time: {time_end - time_start}\n")
                save_path = os.path.join(savedir_root, f"out_view_best.ply")
                o3d.io.write_triangle_mesh(save_path, mesh)

if __name__ == "__main__":

    parser = argparse.ArgumentParserFromFile(description='Process some integers.')
    parser.add_argument('--config_default', type=str, default="configs/config_default.yaml")
    parser.add_argument('--config', '-c', type=str, default="configs/config_cosc.yaml")
    parser.add_argument('--case_path', type=str)
    parser.add_argument('--obj_name', type=str)
    parser.add_argument('--resolution', type=int, default=164)
    parser.update_file_arg_names(["config_default", "config"])

    config = parser.parse(use_unknown=True)

    args = parser.parse_args()

    logging.getLogger().setLevel(config["logging"])
    if config["logging"] == "DEBUG":
        config["threads"] = 0

    main(config, args)
