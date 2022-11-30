from typing import Tuple, Optional
import os
import pathlib
import copy
import time

# import igl
import numpy as np
import pandas as pd
import zarr
import tqdm
import pickle
import open3d as o3d
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation
import pytorch_lightning as pl
from torch.utils.data import Subset

from common.cache import file_attr_cache
from common.geometry_util import (
    barycentric_interpolation, mesh_sample_barycentric, get_matching_index_numpy)
from components.gridding import nocs_grid_sample

import MinkowskiEngine as ME


# helper functions
# ================
def _get_groups_df(samples_group):
    rows = dict()
    for key, group in samples_group.items():
        rows[key] = group.attrs.asdict()
    groups_df = pd.DataFrame(data=list(rows.values()), index=rows.keys())
    groups_df.drop_duplicates(inplace=True)
    groups_df['group_key'] = groups_df.index
    return groups_df


# data sets
# =========
class SparseUnet3DTrackingDataset2(Dataset):
    def __init__(self,
                 # zarr
                 zarr_path: str,
                 metadata_cache_dir: str,
                 video_info_path: str = None,
                 use_file_attr_cache: bool = True,
                 # sample size
                 num_pc_sample: int = 8000,
                 num_pc_sample_final: int = 4000,
                 num_volume_sample: int = 0,
                 num_surface_sample: int = 6000,
                 num_surface_sample_init: int = 12000,
                 use_gt_mesh_verts: bool = False,
                 # mixed sampling config
                 surface_sample_ratio: float = 0,
                 surface_sample_std: float = 0.05,
                 # feature config
                 use_rgb: bool = True,
                 use_nocs_as_feature: bool = False,
                 only_foreground_pc: bool = True,
                 # voxelization config
                 voxel_size: float = 0.0025,
                 # data augumentaiton
                 enable_augumentation: bool = True,
                 random_rot_range: Tuple[float, float] = (-90, 90),
                 enable_zero_center: bool = False,
                 num_views: int = 4,
                 pc_noise_std: float = 0,
                 use_pc_nocs_frame1_aug: bool = False,
                 use_fist_frame_pc_nocs_aug_in_test: bool = False,
                 pc_nocs_global_scale_aug_range: Tuple[float, float] = (0.8, 1.2),
                 pc_nocs_global_max_offset_aug: float = 0.05,
                 pc_nocs_gaussian_std: float = 0,
                 use_pc_nocs_ball_offset_aug: bool = False,
                 pc_nocs_ball_query_radius_range: Tuple[float, float] = (0.0, 0.2),
                 pc_nocs_ball_query_max_nn: int = 400,
                 use_mesh_nocs_aug: bool = False,
                 use_fist_frame_mesh_nocs_aug_in_test: bool = False,
                 mesh_nocs_global_scale_aug_range: Tuple[float, float] = (0.8, 1.2),
                 mesh_nocs_global_max_offset_aug: float = 0.05,
                 # volume config
                 volume_size: int = 128,
                 volume_group: str = 'nocs_winding_number_field',
                 tsdf_clip_value: Optional[float] = None,
                 volume_absolute_value: bool = False,
                 # random seed
                 static_epoch_seed: bool = False,
                 is_val: bool = False,
                 split_by_instance: bool = True,
                 # first-frame fitting config
                 alpha: float = 1000.0,
                 finetune_offset: Tuple[float, float, float] = (0., -0.03, 0.),
                 # catch all
                 **kwargs):
        """
        If static_point_sample is True, the points sampled for each index
        will be identical each time being called.
        """
        super().__init__()
        path = pathlib.Path(os.path.expanduser(zarr_path))
        assert(path.exists())
        self.zarr_path = str(path.absolute())
        if '.zip' in str(path.absolute()):
            store = zarr.ZipStore(str(path.absolute()), mode='r')
            root = zarr.group(store=store)
            self.use_zip = True
        else:
            root = zarr.open(str(path.absolute()), mode='r')
            self.use_zip = False
        samples_group = root['samples']
        instances_group = root['instances']

        # extract common info from sample group
        _, sample_group = next(samples_group.groups())
        print(sample_group.tree())

        # extract common info from instance group
        _, instance_group = next(instances_group.groups())
        if volume_size is not None:
            assert(str(volume_size) in
                instance_group['volume'][volume_group])

        # load group metadata
        if use_file_attr_cache:
            groups_df = file_attr_cache(zarr_path,
                cache_dir=metadata_cache_dir)(_get_groups_df)(samples_group)
        else:
            groups_df = _get_groups_df((samples_group))
        # check if index is sorted
        assert(groups_df.index.is_monotonic_increasing)
        groups_df['idx'] = np.arange(len(groups_df))

        # load instance data
        self.instance_data_dict = dict()
        if num_volume_sample > 0:
            for group_key, instance_group in instances_group.groups():
                self.instance_data_dict[group_key] = {
                    'volume':{volume_group:
                                  {str(volume_size):
                                       instance_group['volume'][volume_group][str(volume_size)][:]
                                   }}}

        # global state
        self.samples_group = samples_group
        self.groups_df = groups_df
        # sample size
        self.num_pc_sample = num_pc_sample
        self.num_pc_sample_final = num_pc_sample_final
        self.num_volume_sample = num_volume_sample
        self.num_surface_sample = num_surface_sample
        self.num_surface_sample_init = num_surface_sample_init
        self.use_gt_mesh_verts = use_gt_mesh_verts
        # mixed sampling config
        self.surface_sample_ratio = surface_sample_ratio
        self.surface_sample_std = surface_sample_std
        # feature config
        self.use_rgb = use_rgb
        self.use_nocs_as_feature = use_nocs_as_feature
        self.only_foreground_pc = only_foreground_pc
        # data augumentaiton
        self.enable_augumentation = enable_augumentation
        self.random_rot_range = random_rot_range
        self.num_views = num_views
        assert(num_views > 0)
        self.enable_zero_center = enable_zero_center
        self.pc_noise_std = pc_noise_std
        self.use_pc_nocs_frame1_aug = use_pc_nocs_frame1_aug
        self.use_fist_frame_pc_nocs_aug_in_test = use_fist_frame_pc_nocs_aug_in_test
        self.pc_nocs_global_scale_aug_range = pc_nocs_global_scale_aug_range
        self.pc_nocs_global_max_offset_aug = pc_nocs_global_max_offset_aug
        self.pc_nocs_gaussian_std = pc_nocs_gaussian_std
        self.use_pc_nocs_ball_offset_aug = use_pc_nocs_ball_offset_aug
        self.pc_nocs_ball_query_radius_range = pc_nocs_ball_query_radius_range
        self.pc_nocs_ball_query_max_nn = pc_nocs_ball_query_max_nn
        self.use_mesh_nocs_aug = use_mesh_nocs_aug
        self.use_fist_frame_mesh_nocs_aug_in_test = use_fist_frame_mesh_nocs_aug_in_test
        self.mesh_nocs_global_scale_aug_range = mesh_nocs_global_scale_aug_range
        self.mesh_nocs_global_max_offset_aug = mesh_nocs_global_max_offset_aug
        # volume config
        self.volume_size = volume_size
        self.volume_group = volume_group
        self.tsdf_clip_value = tsdf_clip_value
        self.volume_absolute_value = volume_absolute_value
        # random seed
        self.static_epoch_seed = static_epoch_seed
        self.is_val = is_val
        # split config
        self.split_by_instance = split_by_instance
        # voxelization config
        self.voxel_size = voxel_size
        # fisrt-frame fitting config
        self.alpha = alpha
        self.finetune_offset = np.array(finetune_offset)[np.newaxis, :].astype(np.float32)

        if video_info_path is None or video_info_path == 'None':
            if '.zip' in str(path.absolute()):
                video_info_path = str(path.absolute()).replace('.zip', '')
            else:
                video_info_path = self.zarr_path
        # find all video sequences
        self.find_video_idxs(video_info_path)
        # find all valid grip intervals
        self.find_valid_grip_intervals(video_info_path)
        # for testing only
        self.prev_surf_query_points1 = None
        self.prev_pc_nocs_points1 = None
        self.prev_frame_dataset_idx = None
        self.first_frame_mesh_nocs = None
        self.first_frame_mesh_sim = None

    def __len__(self):
        return len(self.groups_df)

    def set_prev_pose(self, surf_query_points1: np.ndarray, pc_nocs_points1: np.ndarray, prev_frame_dataset_idx: int,
                      first_frame_mesh_nocs: np.ndarray = None, first_frame_mesh_sim: np.ndarray = None):
        # saving nocs pose generated by prediction of previous frame
        self.prev_surf_query_points1 = surf_query_points1
        self.prev_pc_nocs_points1 = pc_nocs_points1
        self.prev_frame_dataset_idx = prev_frame_dataset_idx
        self.first_frame_mesh_nocs = first_frame_mesh_nocs
        self.first_frame_mesh_sim = first_frame_mesh_sim

    def find_video_idxs(self, video_seq_cache_dir: str):
        os.makedirs(video_seq_cache_dir, exist_ok=True)
        cache_path = os.path.join(video_seq_cache_dir, 'video_seq.pkl')
        if os.path.exists(cache_path):
            print('Loading video sequences cache in {}'.format(cache_path))
            with open(cache_path, 'rb') as f:
                self.video_to_idxs_dict, self.idx_to_video_list = pickle.load(f)
        else:
            data_length = self.__len__()
            self.video_to_idxs_dict = dict()
            self.idx_to_video_list = []
            print('Finding video sequences...')
            for idx in tqdm.tqdm(range(data_length), ncols=0):
                dataset_idx = idx
                row = self.groups_df.iloc[dataset_idx]
                group = self.samples_group[row.group_key]
                attrs = group.attrs.asdict()
                video_id = attrs['video_id']
                if video_id not in self.video_to_idxs_dict:
                    self.video_to_idxs_dict[video_id] = []
                self.video_to_idxs_dict[video_id].append(idx)
                self.idx_to_video_list.append(video_id)
            print('Finish finding video sequences!')
            with open(cache_path, 'wb') as f:
                pickle.dump((self.video_to_idxs_dict, self.idx_to_video_list), f)
            print('Saving video sequences cache to {}'.format(cache_path))

    def find_valid_grip_intervals(self, video_seq_cache_dir: str):
        os.makedirs(video_seq_cache_dir, exist_ok=True)
        def is_valid_grip(grip_vertex_ids):
            return grip_vertex_ids[0] != -1
        cache_path = os.path.join(video_seq_cache_dir, 'video_grip_interval_v2.pkl')
        if os.path.exists(cache_path):
            print('Loading video grip interval cache in {}'.format(cache_path))
            with open(cache_path, 'rb') as f:
                self.interval_to_idxs_dict, self.idx_to_interval_list = pickle.load(f)
        else:
            data_length = self.__len__()
            self.interval_to_idxs_dict = dict()
            self.idx_to_interval_list = []
            assert self.video_to_idxs_dict is not None
            print('Finding video valid grip intervals...')
            in_interval = False
            interval_count = 0
            for idx in tqdm.tqdm(range(data_length), ncols=0):
                dataset_idx = idx
                row = self.groups_df.iloc[dataset_idx]
                group = self.samples_group[row.group_key]
                attrs = group.attrs.asdict()
                video_id = attrs['video_id']
                grip_point_group = group['grip_vertex_id']
                left_grip_vertex_ids = grip_point_group['left_grip_vertex_id'][:]
                right_grip_vertex_ids = grip_point_group['right_grip_vertex_id'][:]
                if not in_interval and (is_valid_grip(left_grip_vertex_ids) or is_valid_grip(right_grip_vertex_ids)):
                    # interval start if any hand is grasped
                    self.interval_to_idxs_dict[interval_count] = []
                    in_interval = True

                if in_interval:
                    self.interval_to_idxs_dict[interval_count].append(idx)
                    self.idx_to_interval_list.append(interval_count)
                else:
                    self.idx_to_interval_list.append(-1)

                if in_interval and not is_valid_grip(left_grip_vertex_ids) and not is_valid_grip(right_grip_vertex_ids) \
                        or self.video_to_idxs_dict[video_id][-1] == idx:
                    # interval end (both hands are released) or video end
                    in_interval = False
                    interval_count += 1
            print('Finish finding {} valid grip intervals!'.format(interval_count))
            with open(cache_path, 'wb') as f:
                pickle.dump((self.interval_to_idxs_dict, self.idx_to_interval_list), f)
            print('Saving grip interval cache to {}'.format(cache_path))

    def data_io(self, idx: int) -> dict:
        dataset_idx = idx
        interval_id = self.idx_to_interval_list[idx]
        row = self.groups_df.iloc[dataset_idx]
        group = self.samples_group[row.group_key]
        volume_size = self.volume_size
        volume_group = self.volume_group
        tsdf_clip_value = self.tsdf_clip_value
        volume_absolute_value = self.volume_absolute_value
        num_volume_sample = self.num_volume_sample

        # io
        attrs = group.attrs.asdict()
        instance_id = attrs['instance_id']
        scale = attrs['scale']
        pc_group = group['point_cloud']
        mesh_group = group['mesh']
        grip_point_group = group['grip_vertex_id']
        if 'cls' in pc_group:
            pc_cls = pc_group['cls'][:]
            pc_cls[pc_cls > 0] = 1  # only two classes (foreground + background)
        else:
            pc_cls = np.zeros(pc_group['point'][:].shape[0]).astype(np.uint8)
        data = {
            'cloth_sim_verts': mesh_group['cloth_verts'][:],
            'cloth_nocs_verts': mesh_group['cloth_nocs_verts'][:],
            'cloth_faces_tri': mesh_group['cloth_faces_tri'][:],
            'pc_nocs': pc_group['nocs'][:],
            'pc_sim': pc_group['point'][:],
            'pc_sim_rgb': pc_group['rgb'][:],
            'pc_sizes': pc_group['sizes'][:],
            'pc_cls': pc_cls,
            'left_grip_vertex_ids': grip_point_group['left_grip_vertex_id'][:],
            'right_grip_vertex_ids': grip_point_group['right_grip_vertex_id'][:],
            'video_id': attrs['video_id'],
            'interval_id': interval_id,
            'scale': scale
        }

        # volume io
        if num_volume_sample > 0:
            volume_group = self.instance_data_dict[instance_id]['volume'][volume_group]
            raw_volume = volume_group[str(volume_size)]
            volume = np.expand_dims(raw_volume, (0, 1)).astype(np.float32)
            if tsdf_clip_value is not None:
                scaled_volume = volume / tsdf_clip_value
                volume = np.clip(scaled_volume, -1, 1)
            if volume_absolute_value:
                volume = np.abs(volume)
            data['volume'] = volume

        return data

    def get_base_data(self, idx:int, seed:int, data_in: dict) -> dict:
        """
        Get non-volumetric data as numpy arrays
        """
        num_pc_sample = self.num_pc_sample
        num_views = self.num_views
        # cloth_sim_aabb = self.cloth_sim_aabb

        if self.only_foreground_pc:
            foreground_idxs = data_in['pc_cls'] == 0
            if data_in['pc_cls'].shape[0] != data_in['pc_sim_rgb'].shape[0]:
                foreground_idxs = np.arange(data_in['pc_sim_rgb'].shape[0])
            data_in['pc_sim_rgb'] = data_in['pc_sim_rgb'][foreground_idxs]
            data_in['pc_sim'] = data_in['pc_sim'][foreground_idxs]
            data_in['pc_nocs'] = data_in['pc_nocs'][foreground_idxs]
            data_in['pc_cls'] = data_in['pc_cls'][foreground_idxs]

        rs = np.random.RandomState(seed=seed)
        all_idxs = np.arange(len(data_in['pc_sim']))
        all_num_views = len(data_in['pc_sizes'])
        if num_views < all_num_views:
            idxs_mask = np.zeros_like(all_idxs, dtype=np.bool)
            selected_view_idxs = np.sort(rs.choice(all_num_views, size=num_views, replace=False))
            view_idxs = np.concatenate([[0], np.cumsum(data_in['pc_sizes'])])
            for i in selected_view_idxs:
                idxs_mask[view_idxs[i]: view_idxs[i+1]] = True
            all_idxs = all_idxs[idxs_mask]

        if all_idxs.shape[0] >= num_pc_sample:
            selected_idxs = rs.choice(all_idxs, size=num_pc_sample, replace=False)
        else:
            np.random.seed(seed)
            np.random.shuffle(all_idxs)
            res_num = len(all_idxs) - num_pc_sample
            selected_idxs = np.concatenate([all_idxs, all_idxs[:res_num]], axis=0)

        pc_sim_rgb = data_in['pc_sim_rgb'][selected_idxs].astype(np.float32) / 255
        pc_sim = data_in['pc_sim'][selected_idxs].astype(np.float32)
        pc_nocs = data_in['pc_nocs'][selected_idxs].astype(np.float32)
        pc_cls = data_in['pc_cls'][selected_idxs].astype(np.int64)
        pc_nocs[pc_cls != 0, :] = -1.0

        dataset_idx = np.array([idx])
        video_id = np.array([int(data_in['video_id'])])
        interval_id = np.array([int(data_in['interval_id'])])
        scale = np.array([data_in['scale']])
        # cloth_sim_aabb = cloth_sim_aabb.reshape((1,)+cloth_sim_aabb.shape)

        cloth_sim_verts = data_in['cloth_sim_verts']
        cloth_nocs_verts = data_in['cloth_nocs_verts']
        left_grip_vertex_ids = data_in['left_grip_vertex_ids']
        right_grip_vertex_ids = data_in['right_grip_vertex_ids']
        left_grip_point_sim = np.array([-10., -10., -10.], dtype=np.float32)
        right_grip_point_sim = np.array([-10., -10., -10.], dtype=np.float32)
        left_grip_point_nocs = np.array([-2., -2., -2.], dtype=np.float32)
        right_grip_point_nocs = np.array([-2., -2., -2.], dtype=np.float32)
        for hand_id, grip_vertex_ids in enumerate((left_grip_vertex_ids, right_grip_vertex_ids)):
            if grip_vertex_ids[0] != -1:
                grip_vertices_sim = cloth_sim_verts[grip_vertex_ids, :]
                mean_grip_point_sim = np.mean(grip_vertices_sim, axis=0)
                grip_vertices_nocs = cloth_nocs_verts[grip_vertex_ids, :]
                mean_grip_point_nocs = np.mean(grip_vertices_nocs, axis=0)
                if hand_id == 0:
                    left_grip_point_sim = mean_grip_point_sim.astype(np.float32)
                    left_grip_point_nocs = mean_grip_point_nocs.astype(np.float32)
                else:
                    right_grip_point_sim = mean_grip_point_sim.astype(np.float32)
                    right_grip_point_nocs = mean_grip_point_nocs.astype(np.float32)

        data = {
            'x': pc_sim_rgb,
            'y': pc_nocs,
            'pos': pc_sim,
            'cls': pc_cls,
            'dataset_idx': dataset_idx,
            'video_id': video_id,
            'interval_id': interval_id,
            'left_grip_point_sim': left_grip_point_sim,
            'left_grip_point_nocs': left_grip_point_nocs,
            'right_grip_point_sim': right_grip_point_sim,
            'right_grip_point_nocs': right_grip_point_nocs,
            'scale': scale
        }
        return data

    def get_volume_sample(self, idx: int, data_in: dict) -> dict:
        """
        Sample points by interpolating the volume.
        """
        volume_group = self.volume_group
        num_volume_sample = self.num_volume_sample
        surface_sample_ratio = self.surface_sample_ratio
        surface_sample_std = self.surface_sample_std

        seed = idx
        rs = np.random.RandomState(seed=seed)
        if surface_sample_ratio == 0:
            query_points = rs.uniform(low=0, high=1, size=(num_volume_sample, 3)).astype(np.float32)
        else:
            # combine uniform and near-surface sample
            num_uniform_sample = int(num_volume_sample * surface_sample_ratio)
            num_surface_sample = num_volume_sample - num_uniform_sample
            uniform_query_points = rs.uniform(
                low=0, high=1, size=(num_uniform_sample, 3)).astype(np.float32)

            cloth_nocs_verts = data_in['cloth_nocs_verts']
            cloth_faces_tri = data_in['cloth_faces_tri']
            sampled_barycentric, sampled_face_idxs = mesh_sample_barycentric(
                verts=cloth_nocs_verts, faces=cloth_faces_tri,
                num_samples=num_surface_sample, seed=seed)
            sampled_faces = cloth_faces_tri[sampled_face_idxs]
            sampled_nocs_points = barycentric_interpolation(
                sampled_barycentric, cloth_nocs_verts, sampled_faces)
            surface_noise = rs.normal(loc=(0,) * 3, scale=(surface_sample_std,) * 3,
                                      size=(num_surface_sample, 3))
            surface_query_points = sampled_nocs_points + surface_noise
            mixed_query_points = np.concatenate(
                [uniform_query_points, surface_query_points], axis=0).astype(np.float32)
            query_points = np.clip(mixed_query_points, 0, 1)

        sample_values_torch = nocs_grid_sample(
            torch.from_numpy(data_in['volume']),
            torch.from_numpy(query_points))
        sample_values_numpy = sample_values_torch.view(
            sample_values_torch.shape[:-1]).numpy()
        if volume_group == 'nocs_occupancy_grid':
            # make sure number is either 0 or 1 for occupancy
            sample_values_numpy = (sample_values_numpy > 0.1).astype(np.float32)
        data = {
            'volume_query_points': query_points,
            'gt_volume_value': sample_values_numpy
        }
        data = self.reshape_for_batching(data)
        return data

    def get_pc_nocs_from_mesh(self, mesh_sim_points, pc_sim_points, mesh_nocs_points, is_train=False, return_dist=False):
        num_mesh_points = mesh_sim_points.shape[0]  # M
        num_pc_points = pc_sim_points.shape[0]  # N'
        mesh_sim_points_expand = torch.from_numpy(mesh_sim_points)\
            .unsqueeze(0).expand(num_pc_points, -1, -1).to('cpu' if is_train else 'cuda:0')  # (N', M, 3)
        pc_sim_points_expand = torch.from_numpy(pc_sim_points)\
            .unsqueeze(1).expand(-1, num_mesh_points, -1).to('cpu' if is_train else 'cuda:0')  # (N', M, 3)
        sim_dist = torch.norm(mesh_sim_points_expand - pc_sim_points_expand, p=2, dim=-1)  # (N', M)

        mesh_nocs_points_exapnd = torch.from_numpy(mesh_nocs_points)\
            .unsqueeze(0).expand(num_pc_points, -1, -1).to('cpu' if is_train else 'cuda:0')  # (N', M, 3)
        soft_argmax_weight = F.softmax(-sim_dist * self.alpha, dim=1).unsqueeze(2)  # (N', M, 1)
        pc_nocs_points = torch.sum(mesh_nocs_points_exapnd * soft_argmax_weight, dim=1, keepdim=False)  # (N', 3)
        if return_dist:
            return pc_nocs_points.cpu().numpy(), sim_dist.min(dim=1)[0].mean().cpu().numpy()
        else:
            return pc_nocs_points.cpu().numpy()

    def get_first_frame_surface_sample(self, data: dict, data_in_1: dict, data_in_2: dict, seed: int) -> dict:
        assert not self.enable_augumentation
        assert self.prev_surf_query_points1 is None and self.prev_pc_nocs_points1 is None
        first_frame_surface_sim_points = self.first_frame_mesh_sim + self.finetune_offset
        pc_sim_points1 = data['pos1']
        self.prev_pc_nocs_points1, mean_dist = self.get_pc_nocs_from_mesh(first_frame_surface_sim_points, pc_sim_points1,
                                                               self.first_frame_mesh_nocs, return_dist=True)
        assert not self.use_nocs_as_feature
        data['y1'] = self.prev_pc_nocs_points1.copy()
        self.prev_surf_query_points1 = self.first_frame_mesh_nocs

        out_data = dict(data)
        if self.use_pc_nocs_frame1_aug and self.use_fist_frame_pc_nocs_aug_in_test and self.static_epoch_seed:
            # only enable in test mode (first frame)
            out_data = self.pc_nocs_augmentation(seed, data=out_data)
        # get gt_sim_points with surf_query_points
        surface_sample_data1 = self.get_surface_sample_with_prev_pose(data_in_1, seed)
        surface_sample_data2 = self.get_surface_sample_with_prev_pose(data_in_2, seed)
        if self.enable_zero_center:
            surface_sample_data1['gt_sim_points'] -= data['input_center1']
            surface_sample_data2['gt_sim_points'] -= data['input_center2']
        out_data['surf_query_points1'] = surface_sample_data1['surf_query_points']
        out_data['gt_sim_points1'] = surface_sample_data1['gt_sim_points']
        out_data['surf_query_points2'] = surface_sample_data2['surf_query_points']
        out_data['gt_surf_query_points2'] = surface_sample_data2['gt_surf_query_points']
        out_data['gt_sim_points2'] = surface_sample_data2['gt_sim_points']
        return out_data

    def get_surface_sample_with_prev_pose(self, data_in: dict, seed: int) -> dict:
        # get surface sample given sampled nocs points in pose of previous frame
        pred_sampled_nocs_points = self.prev_surf_query_points1.copy()
        cloth_nocs_verts = data_in['cloth_nocs_verts']
        cloth_sim_verts = data_in['cloth_sim_verts']
        cloth_faces_tri = data_in['cloth_faces_tri']

        if self.use_gt_mesh_verts:
            assert not self.use_mesh_nocs_aug
            gt_nocs_points = cloth_nocs_verts
            gt_sim_points = cloth_sim_verts
        else:
            sampled_barycentric, sampled_face_idxs = mesh_sample_barycentric(
                verts=cloth_nocs_verts, faces=cloth_faces_tri,
                num_samples=self.num_surface_sample, seed=seed)
            sampled_faces = cloth_faces_tri[sampled_face_idxs]

            gt_sampled_nocs_points = barycentric_interpolation(
                sampled_barycentric, cloth_nocs_verts, sampled_faces)
            gt_sampled_sim_points = barycentric_interpolation(
                sampled_barycentric, cloth_sim_verts, sampled_faces)

            matching_inds = get_matching_index_numpy(pred_sampled_nocs_points, gt_sampled_nocs_points)
            pred_sampled_nocs_points = pred_sampled_nocs_points[matching_inds[:, 0], :].copy()
            gt_sim_points = gt_sampled_sim_points[matching_inds[:, 1], :]
            gt_nocs_points = gt_sampled_nocs_points[matching_inds[:, 1], :]
            assert pred_sampled_nocs_points.shape[0] == gt_sim_points.shape[0]
        data = {
            'surf_query_points': pred_sampled_nocs_points,
            'gt_surf_query_points': gt_nocs_points,
            'gt_sim_points': gt_sim_points,
        }
        return data

    def get_surface_sample(self, seed: int, noise_seed: int, data_in: dict, is_train=False) -> dict:
        num_surface_sample = self.num_surface_sample

        cloth_nocs_verts = data_in['cloth_nocs_verts']
        cloth_sim_verts = data_in['cloth_sim_verts']
        cloth_faces_tri = data_in['cloth_faces_tri']

        if self.use_gt_mesh_verts:
            assert not self.use_mesh_nocs_aug
            sampled_nocs_points = cloth_nocs_verts
            sampled_sim_points = cloth_sim_verts
        else:
            sampled_barycentric, sampled_face_idxs = mesh_sample_barycentric(
                verts=cloth_nocs_verts, faces=cloth_faces_tri,
                num_samples=num_surface_sample, seed=seed)
            sampled_faces = cloth_faces_tri[sampled_face_idxs]

            sampled_nocs_points = barycentric_interpolation(
                sampled_barycentric, cloth_nocs_verts, sampled_faces)
            sampled_sim_points = barycentric_interpolation(
                sampled_barycentric, cloth_sim_verts, sampled_faces)

        if self.use_mesh_nocs_aug:
            if is_train or \
                    (self.prev_surf_query_points1 is None and self.use_fist_frame_mesh_nocs_aug_in_test and not is_train) \
                    or self.is_val:
                # only enable in train mode or test mode (first frame)
                rs = np.random.RandomState(seed=seed)
                scale_factor = rs.uniform(self.mesh_nocs_global_scale_aug_range[0],
                                          self.mesh_nocs_global_scale_aug_range[1],
                                          (1, 3)).astype(np.float32)
                nocs_center = np.array([[0.5, 0.5, 0.5]]).astype(np.float32)
                aug_sampled_nocs_points = (sampled_nocs_points - nocs_center) * scale_factor + nocs_center
                # aug_sampled_nocs_points = sampled_nocs_points * scale_factor
                global_offset = rs.uniform(0., self.mesh_nocs_global_max_offset_aug, (1, 3)).astype(np.float32)
                aug_sampled_nocs_points = aug_sampled_nocs_points + global_offset
            else:
                aug_sampled_nocs_points = sampled_nocs_points
        else:
            aug_sampled_nocs_points = sampled_nocs_points

        data = {
            'surf_query_points': aug_sampled_nocs_points,
            'gt_surf_query_points': sampled_nocs_points,
            'gt_sim_points': sampled_sim_points,
        }

        return data

    def rotation_augumentation(self, seed: int, data: dict) -> dict:
        random_rot_range = self.random_rot_range
        assert(len(random_rot_range) == 2)
        assert(random_rot_range[0] <= random_rot_range[-1])

        rs = np.random.RandomState(seed=seed)
        rot_angle = rs.uniform(*random_rot_range)
        rot_mat = Rotation.from_euler(
            'y', rot_angle, degrees=True
            ).as_matrix().astype(np.float32)

        sim_point_keys = ['pos', 'gt_sim_points', 'left_grip_point_sim', 'right_grip_point_sim']
        out_data = dict(data)
        for key in sim_point_keys:
            if key in data:
                out_data[key] = (data[key] @ rot_mat.T).astype(np.float32)

        # record augumentation matrix for eval
        out_data['input_aug_rot_mat'] = rot_mat
        return out_data

    def point_cloud_normalize(self, data: dict) -> dict:
        pc_sim = data['pos']
        if 'input_center' not in data:
            center = np.zeros((1, 3)).astype(np.float32)
            center[0, 0] = (np.max(pc_sim[:, 0]) + np.min(pc_sim[:, 0])) / 2.0
            center[0, 2] = (np.max(pc_sim[:, 2]) + np.min(pc_sim[:, 2])) / 2.0
            center[0, 1] = np.min(pc_sim[:, 1])  # select the lowest point in Y axis as bottom center
        else:
            center = data['input_center']
        sim_point_keys = ['pos', 'gt_sim_points', 'left_grip_point_sim', 'right_grip_point_sim']
        # TODO: fix grip point in invalid cases (-10)
        out_data = dict(data)
        for key in sim_point_keys:
            if key in data:
                out_data[key] = (data[key] - center).astype(np.float32)

        # record center matrix for eval
        out_data['input_center'] = center
        return out_data

    def noise_augumentation(self, idx: int, data: dict) -> dict:
        pc_noise_std = self.pc_noise_std
        static_epoch_seed = self.static_epoch_seed

        pc_sim = data['pos']
        seed = idx if static_epoch_seed else None
        rs = np.random.RandomState(seed=seed)
        noise = rs.normal(
            loc=(0,)*3,
            scale=(pc_noise_std,)*3,
            size=pc_sim.shape).astype(np.float32)
        pc_sim_aug = pc_sim + noise
        out_data = dict(data)
        out_data['pos'] = pc_sim_aug
        return out_data

    def pc_nocs_augmentation(self, seed: int, data: dict) -> dict:
        pc_nocs = data['y'] if 'y' in data else data['y1']
        rs = np.random.RandomState(seed=seed)
        if self.use_pc_nocs_ball_offset_aug:
            assert 'gt_surf_query_points' in data
            pc_nocs_aug = pc_nocs.copy()
            pc_nocs = pc_nocs.copy()
            mesh_nocs = data['gt_surf_query_points'].copy()
            select_mesh_idx = rs.choice(np.arange(mesh_nocs.shape[0]), size=1)
            select_mesh_pts = mesh_nocs[select_mesh_idx, :]

            select_pc_idx = rs.choice(np.arange(pc_nocs.shape[0]), size=1)
            select_pc_loc = pc_nocs[select_pc_idx, :].T
            pc_pcd = o3d.geometry.PointCloud()
            pc_pcd.points = o3d.utility.Vector3dVector(pc_nocs)
            pc_pcd_tree = o3d.geometry.KDTreeFlann(pc_pcd)
            ball_radius = rs.uniform(self.pc_nocs_ball_query_radius_range[0],
                                     self.pc_nocs_ball_query_radius_range[1])
            k, select_pc_idxs, _ = pc_pcd_tree.search_hybrid_vector_3d(select_pc_loc, ball_radius,
                                                                       self.pc_nocs_ball_query_max_nn)
            select_pc_idxs = np.asarray(select_pc_idxs)
            select_pc = pc_nocs[select_pc_idxs, :].copy()
            select_pc_center = select_pc.mean(axis=0)[np.newaxis, :]
            local_offset = select_mesh_pts - select_pc_center
            aug_select_pc = select_pc + local_offset

            aug_select_pc_expand = aug_select_pc[np.newaxis, :, :].repeat(self.num_surface_sample, axis=0)  # (M, K, 3)
            mesh_nocs_expand = mesh_nocs[:, np.newaxis, :].repeat(k, axis=1)  # (M, K, 3)
            dist = aug_select_pc_expand - mesh_nocs_expand  # (M, K, 3)
            dist = (dist * dist).sum(axis=2) # (M, K)

            fit_mesh_idxs = np.argmin(dist, axis=0)  # (K)
            aug_select_pc = mesh_nocs[fit_mesh_idxs, :]

            pc_nocs_aug[select_pc_idxs, :] = aug_select_pc
        else:
            pc_nocs_aug = pc_nocs.copy()

        scale_factor = rs.uniform(self.pc_nocs_global_scale_aug_range[0],
                                  self.pc_nocs_global_scale_aug_range[1],
                                  (1, 3)).astype(np.float32)
        nocs_center = np.array([[0.5, 0.5, 0.5]]).astype(np.float32)
        pc_nocs_aug = (pc_nocs_aug - nocs_center) * scale_factor + nocs_center
        # pc_nocs_aug = pc_nocs * scale_factor
        global_offset = rs.uniform(0., self.pc_nocs_global_max_offset_aug, (1, 3)).astype(np.float32)
        if self.pc_nocs_gaussian_std > 0:
            noise = rs.normal(
                loc=(0,) * 3,
                scale=(self.pc_nocs_gaussian_std,) * 3,
                size=pc_nocs.shape).astype(np.float32)
            pc_nocs_aug = pc_nocs_aug + noise
        pc_nocs_aug = pc_nocs_aug + global_offset
        out_data = dict(data)
        if 'y' in out_data:
            out_data['y'] = pc_nocs_aug
        else:
            out_data['y1'] = pc_nocs_aug
        return out_data

    def reshape_for_batching(self, data: dict) -> dict:
        out_data = dict()
        for key, value in data.items():
            out_data[key] = value.reshape((1,) + value.shape)
        return out_data

    def voxelize_pc_points(self, data: dict, seed: int = 0) -> dict:
        out_data = dict(data)

        pc_sim_points1 = data['pos1']
        _, sel_pc_sim_points1 = ME.utils.sparse_quantize(pc_sim_points1 / self.voxel_size, return_index=True)
        if self.num_pc_sample_final > 0:
            origin_slim_pc_num = sel_pc_sim_points1.shape[0]
            assert origin_slim_pc_num >= self.num_pc_sample_final
            all_idxs = np.arange(origin_slim_pc_num)
            rs = np.random.RandomState(seed=seed)
            final_selected_idxs1 = rs.choice(all_idxs, size=self.num_pc_sample_final, replace=False)
            sel_pc_sim_points1 = sel_pc_sim_points1[final_selected_idxs1]
            assert sel_pc_sim_points1.shape[0] == self.num_pc_sample_final

        out_data['x1'] = out_data['x1'][sel_pc_sim_points1, :]
        if self.prev_pc_nocs_points1 is not None:
            out_data['y1'] = self.prev_pc_nocs_points1.copy()
            assert out_data['y1'].shape[0] == out_data['x1'].shape[0]
        else:
            out_data['y1'] = out_data['y1'][sel_pc_sim_points1, :]
        out_data['cls1'] = out_data['cls1'][sel_pc_sim_points1]
        out_data['pos1'] = out_data['pos1'][sel_pc_sim_points1, :]
        out_data['coords_pos1'] = np.floor(out_data['pos1'] / self.voxel_size)
        if self.use_nocs_as_feature:
            if self.use_rgb:
                out_data['feat1'] = np.concatenate([out_data['y1'], out_data['x1']], axis=-1)
            else:
                out_data['feat1'] = out_data['y1']
        else:
            if self.use_rgb:
                out_data['feat1'] = out_data['x1']
            else:
                out_data['feat1'] = np.zeros_like(out_data['x1'])

        pc_sim_points2 = data['pos2']
        _, sel_pc_sim_points2 = ME.utils.sparse_quantize(pc_sim_points2 / self.voxel_size, return_index=True)
        if self.num_pc_sample_final > 0:
            origin_slim_pc_num = sel_pc_sim_points2.shape[0]
            assert origin_slim_pc_num >= self.num_pc_sample_final
            all_idxs = np.arange(origin_slim_pc_num)
            rs = np.random.RandomState(seed=seed)
            final_selected_idxs2 = rs.choice(all_idxs, size=self.num_pc_sample_final, replace=False)
            sel_pc_sim_points2 = sel_pc_sim_points2[final_selected_idxs2]
            assert sel_pc_sim_points2.shape[0] == self.num_pc_sample_final
        out_data['x2'] = out_data['x2'][sel_pc_sim_points2, :]
        out_data['y2'] = out_data['y2'][sel_pc_sim_points2, :]
        out_data['cls2'] = out_data['cls2'][sel_pc_sim_points2]
        out_data['pos2'] = out_data['pos2'][sel_pc_sim_points2, :]
        out_data['coords_pos2'] = np.floor(out_data['pos2'] / self.voxel_size)
        # Use empty nocs (all -1) as feature
        empty_nocs = -1 * np.ones(out_data['y2'].shape, dtype=np.float32)
        if self.use_nocs_as_feature:
            if self.use_rgb:
                out_data['feat2'] = np.concatenate([empty_nocs, out_data['x2']], axis=-1)
            else:
                out_data['feat2'] = empty_nocs
        else:
            if self.use_rgb:
                out_data['feat2'] = out_data['x2']
            else:
                out_data['feat2'] = np.zeros_like(out_data['x2'])
        return out_data

    def __getitem__(self, idx: int) -> dict:
        try:
            num_surface_sample = self.num_surface_sample
            num_volume_sample = self.num_volume_sample
            enable_augumentation = self.enable_augumentation
            enable_zero_center = self.enable_zero_center
            pc_noise_std = self.pc_noise_std
            use_pc_nocs_frame1_aug = self.use_pc_nocs_frame1_aug

            video_id = self.idx_to_video_list[idx]
            video_start_idx = self.video_to_idxs_dict[video_id][0]
            # use the same data if it's the last frame in the video
            next_idx = min(idx + 1, self.video_to_idxs_dict[video_id][-1])

            prev_idx = self.prev_frame_dataset_idx if self.prev_frame_dataset_idx is not None else idx
            prev_video_id = self.idx_to_video_list[prev_idx]
            if prev_video_id != video_id:
                return dict()

            data_in_1 = self.data_io(prev_idx)  # the previous frame
            data_in_2 = self.data_io(next_idx)

            data_new_all = dict()
            data_torch_all = dict()
            share_seed = int(data_in_1['video_id']) if self.static_epoch_seed else int(time.time() * 1000000) % 21473647
            val_seed = idx if self.static_epoch_seed else int(time.time() * 1000000) % 21473647
            np.random.seed()
            # only use baseline_pred in train or test, not val
            use_baseline_rand = np.random.random_sample()
            for idx_offset, data_in in enumerate((data_in_1, data_in_2)):
                data = self.get_base_data(idx, share_seed, data_in=data_in)
                if num_volume_sample > 0:
                    volume_sample_data = self.get_volume_sample(share_seed+idx_offset, data_in=data_in)
                    data.update(volume_sample_data)
                if num_surface_sample > 0:
                    noise_seed = idx + idx_offset if self.static_epoch_seed else int(time.time() * 1000) % 21473647
                    if self.first_frame_mesh_nocs is None or self.first_frame_mesh_sim is None:
                        if self.prev_surf_query_points1 is None:
                            # must use the same query points for the two continous frames,
                            # different noise seed for validation
                            surface_sample_data = self.get_surface_sample(share_seed, noise_seed, data_in=data_in,
                                                                          is_train=not self.static_epoch_seed)
                        else:
                            # get gt_sim_points with surf_query_points1
                            surface_sample_data = self.get_surface_sample_with_prev_pose(data_in, share_seed)
                        data.update(surface_sample_data)
                    else:
                        # do it later using first-frame prediction
                        pass

                data['input_aug_rot_mat'] = np.eye(3, dtype=np.float32)
                if enable_augumentation:
                    # must use the same rotation augmentation for the two continous frames
                    data = self.rotation_augumentation(share_seed, data=data)
                if pc_noise_std > 0:
                    data = self.noise_augumentation(share_seed + idx_offset, data=data)
                if enable_zero_center:
                    assert self.is_val, 'Not implement zero-center in test mode!'
                    if idx_offset == 1:
                        # frame 2 use the input center as frame1
                        data['input_center'] = data_new_all['input_center1']
                    data = self.point_cloud_normalize(data=data)
                if use_pc_nocs_frame1_aug and idx_offset == 0:
                    if ((self.prev_surf_query_points1 is None and self.use_fist_frame_pc_nocs_aug_in_test and self.static_epoch_seed) \
                            or self.is_val):
                        # only enable in train mode or val mode or test mode (first frame)
                        data = self.pc_nocs_augmentation(val_seed, data=data)

                # add postfix in key names
                data_new = dict()
                for key, value in data.items():
                    data_new[key + str(idx_offset+1)] = value
                data_new_all.update(data_new)

            # voxelize pc points
            data_new_all = self.voxelize_pc_points(data_new_all, share_seed)

            if idx == self.prev_frame_dataset_idx and self.first_frame_mesh_nocs is not None and self.first_frame_mesh_sim is not None:
                # calculate first-frame pc nocs
                data_new_all = self.get_first_frame_surface_sample(data_new_all, data_in_1, data_in_2, share_seed)

            # change from numpy to torch tensors
            for key, value in data_new_all.items():
                data_torch_all[key] = torch.from_numpy(value)

            return data_torch_all
        except Exception as e:
            print(e)
            return dict()


# data modules
# ============
class SparseUnet3DTrackingDataModule2(pl.LightningDataModule):
    def __init__(self, **kwargs):
        """
        dataset_split: tuple of (train, val, test)
        """
        super().__init__()
        assert(len(kwargs['dataset_split']) == 3)
        self.kwargs = kwargs

        self.train_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        kwargs = self.kwargs
        split_seed = kwargs['split_seed']
        dataset_split = kwargs['dataset_split']

        train_args = dict(kwargs)
        train_args['static_epoch_seed'] = False
        train_dataset = SparseUnet3DTrackingDataset2(**train_args)
        val_dataset = copy.deepcopy(train_dataset)
        val_dataset.static_epoch_seed = True

        groups_df = train_dataset.groups_df
        group_keys = groups_df.index.to_numpy()
        groups_df['sample_instance_id'] = np.array([key.split('_')[0] for key in group_keys])
        instances_df = groups_df.groupby('sample_instance_id').agg({'idx': lambda x: sorted(x)})

        # split for train/val/test
        num_instances = len(instances_df)
        normalized_split = np.array(dataset_split)
        normalized_split = normalized_split / np.sum(normalized_split)
        instance_split = (normalized_split * num_instances).astype(np.int64)

        # add leftover instance to training set
        instance_split[0] += num_instances - np.sum(instance_split)

        # generate index for each
        all_idxs = np.arange(num_instances)
        rs = np.random.RandomState(seed=split_seed)
        perm_all_idxs = rs.permutation(all_idxs)

        split_instance_idx_list = list()
        prev_idx = 0
        for x in instance_split:
            next_idx = prev_idx + x
            split_instance_idx_list.append(perm_all_idxs[prev_idx: next_idx])
            prev_idx = next_idx
        assert(np.allclose([len(x) for x in split_instance_idx_list], instance_split))

        split_idx_list = list()
        for instance_idxs in split_instance_idx_list:
            idxs = np.sort(np.concatenate(instances_df.iloc[instance_idxs].idx))
            split_idx_list.append(idxs)
        assert(sum(len(x) for x in split_idx_list) == len(groups_df))

        # generate subsets
        if kwargs['remove_invalid_interval_in_train']:
            raw_train_idxs, val_idxs, test_idxs = split_idx_list
            train_idxs = []
            for idx in raw_train_idxs:
                if train_dataset.idx_to_interval_list[idx] != -1:
                    train_idxs.append(idx)
        else:
            train_idxs, val_idxs, test_idxs = split_idx_list
        train_subset = Subset(train_dataset, train_idxs)
        val_subset = Subset(val_dataset, val_idxs)
        test_subset = Subset(val_dataset, test_idxs)

        self.groups_df = groups_df
        self.train_idxs = train_idxs
        self.val_idxs = val_idxs
        self.test_idxs = test_idxs
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_subset = train_subset
        self.val_subset = val_subset
        self.test_subset = test_subset

    def train_dataloader(self):
        kwargs = self.kwargs
        batch_size = kwargs['batch_size']
        num_workers = kwargs['num_workers']
        train_subset = self.train_subset
        dataloader = DataLoader(train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=False,
            collate_fn=self.collate_pair_fn,
            drop_last=True)
        return dataloader

    def val_dataloader(self):
        kwargs = self.kwargs
        batch_size = kwargs['batch_size']
        num_workers = kwargs['num_workers']
        val_subset = self.val_subset
        dataloader = DataLoader(val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self.collate_pair_fn,
            drop_last=True)
        return dataloader

    def test_dataloader(self):
        kwargs = self.kwargs
        batch_size = kwargs['batch_size']
        num_workers = kwargs['num_workers']
        test_subset = self.test_subset
        dataloader = DataLoader(test_subset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_pair_fn,
            num_workers=num_workers,
            drop_last=True)
        return dataloader

    def collate_pair_fn(self, list_data: list) -> dict:
        try:
            if len(list_data[0]) == 0:
                return dict()
            out_dict = dict()
            batch_size = len(list_data)
            normal_keys = ('dataset_idx1', 'video_id1', 'scale1', 'input_aug_rot_mat1', 'interval_id1', 'input_center1',
                           'dataset_idx2', 'video_id2', 'scale2', 'input_aug_rot_mat2', 'interval_id2', 'input_center2',
                           'left_grip_point_sim1', 'left_grip_point_sim2',
                           'left_grip_point_nocs1', 'left_grip_point_nocs2',
                           'right_grip_point_sim1', 'right_grip_point_sim2',
                           'right_grip_point_nocs1', 'right_grip_point_nocs2'
                           )

            def create_batch(list_data, key, batch_size, stack=True):
                data_batch = []
                batch_idxs = []
                for batch_idx in range(batch_size):
                    data_batch.append(list_data[batch_idx][key])
                    data_len = list_data[batch_idx][key].shape[0]
                    batch_idxs.append(batch_idx * torch.ones(data_len, dtype=torch.long))
                batch_idxs = torch.cat(batch_idxs, dim=0)
                if stack:
                    data_batch = torch.stack(data_batch, dim=0)
                else:
                    data_batch = torch.cat(data_batch, dim=0)
                return data_batch, batch_idxs

            for key in normal_keys:
                if key in list_data[0]:
                    out_dict[key], _ = create_batch(list_data, key, batch_size, stack=True)

            normal_pc_keys = ('x1', 'y1', 'pos1', 'cls1', 'x2', 'y2', 'pos2', 'cls2')
            for key in normal_pc_keys:
                out_dict[key], batch_idxs = create_batch(list_data, key, batch_size, stack=False)
                if key == 'y1':
                    out_dict['pc_batch_idx1'] = batch_idxs
                elif key == 'y2':
                    out_dict['pc_batch_idx2'] = batch_idxs

            coords1_batch = []
            coords2_batch = []
            feat1_list = []
            feat2_list = []
            for batch_idx in range(batch_size):
                coords1_batch.append(list_data[batch_idx]['coords_pos1'])
                coords2_batch.append(list_data[batch_idx]['coords_pos2'])
                feat1_list.append(list_data[batch_idx]['feat1'])
                feat2_list.append(list_data[batch_idx]['feat2'])

            out_dict['coords1'], out_dict['feat1'] = ME.utils.sparse_collate(coords1_batch, feat1_list)
            out_dict['coords2'], out_dict['feat2'] = ME.utils.sparse_collate(coords2_batch, feat2_list)

            normal_mesh_keys = ('surf_query_points1', 'gt_surf_query_points1', 'gt_sim_points1',
                                'surf_query_points2', 'gt_surf_query_points2', 'gt_sim_points2',
                                'volume_query_points1', 'gt_volume_value1',
                                'volume_query_points2', 'gt_volume_value2')
            for key in normal_mesh_keys:
                if key not in list_data[0]:
                    continue
                out_dict[key], batch_idxs = create_batch(list_data, key, batch_size, stack=False)
                if key == 'surf_query_points1':
                    out_dict['mesh_batch_idx1'] = batch_idxs
                elif key == 'surf_query_points2':
                    out_dict['mesh_batch_idx2'] = batch_idxs

            return out_dict
        except Exception as e:
            print(e)
            return dict()
