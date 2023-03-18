# %%
# import
import os
import pathlib
import time

import hydra
import numpy as np
import pandas as pd
import torch
import wandb
import yaml
import zarr
from numcodecs import Blosc
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from common.torch_util import to_numpy
from common.geometry_util import barycentric_interpolation, mesh_sample_barycentric
from datasets.tracking_dataset import SparseUnet3DTrackingDataModule2
from networks.tracking_network import GarmentTrackingPipeline

# %%
# helper functions
def get_checkpoint_df(checkpoint_dir):
    all_checkpoint_paths = sorted(pathlib.Path(checkpoint_dir).glob('*.ckpt'))
    rows = list()
    for path in all_checkpoint_paths:
        fname = path.stem
        row = dict()
        for item in fname.split('-'):
            key, value = item.split('=')
            row[key] = float(value)
        row['path'] = str(path.absolute())
        rows.append(row)
    checkpoint_df = pd.DataFrame(rows)
    return checkpoint_df


def get_mc_surface(pred_samples_group, group_key,
                   init_num_points=12000,
                   final_num_points=6000,
                   value_threshold=0.13, seed=0,
                   value_key='marching_cubes_mesh/volume_gradient_magnitude'):
    sample_group = pred_samples_group[group_key]
    # io
    pred_mc_group = sample_group['marching_cubes_mesh']
    pred_mc_verts = pred_mc_group['verts'][:]
    pred_mc_faces = pred_mc_group['faces'][:]
    pred_mc_sim_verts = pred_mc_group['warp_field'][:]
    # point sample
    num_samples = int(init_num_points)
    pred_sample_bc, pred_sample_face_idx = mesh_sample_barycentric(
        pred_mc_verts, pred_mc_faces,
        num_samples=num_samples, seed=seed)
    pred_sample_nocs_points = barycentric_interpolation(
        pred_sample_bc,
        pred_mc_verts,
        pred_mc_faces[pred_sample_face_idx])
    pred_sample_sim_points = barycentric_interpolation(
        pred_sample_bc,
        pred_mc_sim_verts,
        pred_mc_faces[pred_sample_face_idx])
    # remove holes
    pred_value = sample_group[value_key][:]
    pred_sample_value = np.squeeze(barycentric_interpolation(
        pred_sample_bc,
        np.expand_dims(pred_value, axis=1),
        pred_mc_faces[pred_sample_face_idx]))
    is_valid_sample = pred_sample_value > value_threshold
    valid_pred_sample_nocs_points = pred_sample_nocs_points[is_valid_sample]
    valid_pred_sample_sim_points = pred_sample_sim_points[is_valid_sample]

    valid_num_samples = valid_pred_sample_nocs_points.shape[0]
    if valid_num_samples >= final_num_points:
        np.random.seed(seed)
        valid_idxs = np.random.choice(np.arange(valid_num_samples), size=final_num_points)
        valid_pred_sample_nocs_points = valid_pred_sample_nocs_points[valid_idxs, :]
        valid_pred_sample_sim_points = valid_pred_sample_sim_points[valid_idxs, :]
    else:
        np.random.seed(seed)
        shuffle_idxs = np.arange(valid_num_samples)
        np.random.shuffle(shuffle_idxs)
        valid_pred_sample_nocs_points = valid_pred_sample_nocs_points[shuffle_idxs, :]
        valid_pred_sample_sim_points = valid_pred_sample_sim_points[shuffle_idxs, :]
        res_num = final_num_points - valid_num_samples
        valid_pred_sample_nocs_points = np.concatenate([valid_pred_sample_nocs_points,
                                                        valid_pred_sample_nocs_points[:res_num, :]], axis=0)
        valid_pred_sample_sim_points = np.concatenate([valid_pred_sample_sim_points,
                                                       valid_pred_sample_sim_points[:res_num, :]], axis=0)
        assert valid_pred_sample_nocs_points.shape[0] == final_num_points

    return valid_pred_sample_nocs_points, valid_pred_sample_sim_points


# %%
# main script
@hydra.main(config_path="config",
            config_name="predict_tracking_gt")
def main(cfg: DictConfig) -> None:
    # hydra creates working directory automatically
    pred_output_dir = os.getcwd()
    print(pred_output_dir)

    # determine checkpoint
    checkpoint_path = os.path.expanduser(cfg.main.checkpoint_path)
    assert (pathlib.Path(checkpoint_path).exists())

    # load datamodule
    datamodule = SparseUnet3DTrackingDataModule2(**cfg.datamodule)
    datamodule.prepare_data()
    batch_size = datamodule.kwargs['batch_size']
    assert (batch_size == 1)
    # val and test dataloader both uses val_dataset
    val_dataset = datamodule.val_dataset
    # subset = getattr(datamodule, '{}_subset'.format(cfg.prediction.subset))
    dataloader = getattr(datamodule, '{}_dataloader'.format(cfg.prediction.subset))()
    num_samples = len(dataloader)

    # load input zarr
    input_zarr_path = os.path.expanduser(cfg.datamodule.zarr_path)
    input_root = zarr.open(input_zarr_path, 'r')
    input_samples_group = input_root['samples']

    if cfg.prediction.use_garmentnets_prediction:
        # garmentnets prediction zarr
        pred_zarr_path = os.path.join(cfg.main.garmentnets_prediction_output_dir, 'prediction.zarr')
        assert (pathlib.Path(pred_zarr_path).exists())
        assert cfg.prediction.use_valid_grip_interval, \
            'only support grip interval evluation for garmentnets prediction'
        pred_root = zarr.open(pred_zarr_path, 'r')
        pred_samples_group = pred_root['samples']

    # create output zarr
    output_zarr_path = os.path.join(pred_output_dir, 'prediction.zarr')
    store = zarr.DirectoryStore(output_zarr_path)
    compressor = Blosc(cname='zstd', clevel=6, shuffle=Blosc.BITSHUFFLE)
    output_root = zarr.group(store=store, overwrite=False)
    output_samples_group = output_root.require_group('samples', overwrite=False)

    root_attrs = {
        'subset': cfg.prediction.subset
    }
    output_root.attrs.put(root_attrs)

    # init wandb
    wandb_path = os.path.join(pred_output_dir, 'wandb')
    os.mkdir(wandb_path)
    wandb_run = wandb.init(
        project=os.path.basename(__file__),
        **cfg.logger)
    wandb_meta = {
        'run_name': wandb_run.name,
        'run_id': wandb_run.id
    }
    meta = {
        'script_path': __file__
    }

    # load module to gpu
    model_cpu = GarmentTrackingPipeline.load_from_checkpoint(checkpoint_path)
    device = torch.device('cuda:{}'.format(cfg.main.gpu_id))
    model = model_cpu.to(device)
    model.eval()
    model.requires_grad_(False)
    model.batch_size = batch_size
    assert model.batch_size == 1
    assert cfg.datamodule.num_workers == 0
    model.disable_mesh_nocs_refine_in_test = cfg.prediction.disable_mesh_nocs_refine_in_test
    model.disable_pc_nocs_refine_in_test = cfg.prediction.disable_pc_nocs_refine_in_test

    # dump final cfg
    all_config = {
        'config': OmegaConf.to_container(cfg, resolve=True),
        'output_dir': pred_output_dir,
        'wandb': wandb_meta,
        'meta': meta
    }
    yaml.dump(all_config, open('config.yaml', 'w'), default_flow_style=False)
    wandb.config.update(all_config)

    if cfg.prediction.use_garmentnets_prediction:
        # get pc_sim of the first frame
        start_idx = datamodule.test_idxs[0]
        test_group_row = val_dataset.groups_df.iloc[start_idx]
        group_key = test_group_row.group_key
        while group_key not in pred_samples_group:
            start_idx += 1
            test_group_row = val_dataset.groups_df.iloc[start_idx]
            group_key = test_group_row.group_key
        # calculate PC nocs of the first frame with garmentnets prediction
        next_sample_surface_nocs_points, next_sample_surface_sim_points = \
            get_mc_surface(pred_samples_group, group_key,
                           final_num_points=cfg.datamodule.num_surface_sample,
                           value_threshold=cfg.prediction.value_threshold, seed=start_idx)

        val_dataset.set_prev_pose(None, None, start_idx,
                                  next_sample_surface_nocs_points, next_sample_surface_sim_points)

    current_video_id = None
    current_interval_id = None
    current_video_frame_idx = 0
    current_valid_video_frame_idx = 0
    current_interval_frame_idx = 0
    use_grip_interval = cfg.prediction.use_valid_grip_interval
    current_mesh_nocs_points = None
    # loop
    for batch_idx, batch_cpu in enumerate(tqdm(dataloader)):
        if len(batch_cpu) == 0:
            continue
        # locate raw info
        dataset_idx = int(batch_cpu['dataset_idx1'][0])
        video_id = int(batch_cpu['video_id1'][0])
        assert batch_cpu['video_id1'][0] == batch_cpu['video_id2'][0]
        is_new_video = current_video_id is None or current_video_id != video_id
        if is_new_video:
            # move to a new video
            current_video_id = video_id
            current_video_frame_idx = 0
            current_valid_video_frame_idx = 0
            current_mesh_nocs_points = None

        if use_grip_interval:
            grip_interval_id = val_dataset.idx_to_interval_list[dataset_idx]
            if grip_interval_id == -1:
                # not in valid grip interval
                current_interval_id = None
                current_video_frame_idx += 1
                continue
            is_new_interval = current_interval_id is None or current_interval_id != grip_interval_id
            if is_new_interval:
                current_interval_id = grip_interval_id
                current_interval_frame_idx = 0
        else:
            grip_interval_id = video_id
            current_interval_id = current_video_id
            current_interval_frame_idx = current_video_frame_idx

        val_group_row = val_dataset.groups_df.iloc[dataset_idx]
        group_key = val_group_row.group_key
        attr_keys = ['scale', 'sample_id', 'garment_name']
        attrs = dict((x, val_group_row[x]) for x in attr_keys)
        attrs['batch_idx'] = batch_idx
        attrs['video_id'] = video_id
        attrs['video_frame_idx'] = current_video_frame_idx
        if use_grip_interval:
            attrs['interval_id'] = grip_interval_id
            attrs['interval_frame_idx'] = current_interval_frame_idx

        # load input zarr
        input_group = input_samples_group[group_key]

        # create zarr group
        output_group = output_samples_group.require_group(
            group_key, overwrite=False)
        output_group.attrs.put(attrs)

        batch = {key: value.to(device=device) for key, value in batch_cpu.items()}

        use_refine_mesh_for_query = current_valid_video_frame_idx < cfg.prediction.max_refine_mesh_step
        start_time = time.time()
        # stage 1
        result = model(batch, use_refine_mesh_for_query=use_refine_mesh_for_query)
        time_stage1 = time.time()
        print('Stage 1 used {} s....'.format(time_stage1 - start_time))

        # save nocs data
        nocs_data = result['encoder_result']
        if 'refined_pos_frame2' in nocs_data:
            pred_pc_nocs = nocs_data['refined_pos_frame2']
        else:
            pred_pc_nocs = nocs_data['pos_frame2']
        pc_data_torch = {
            'input_prev_nocs': batch['y1'],
            'pred_nocs': pred_pc_nocs,
            'input_points': batch['pos2'],
            'input_rgb': (batch['x2'] * 255).to(torch.uint8),
            'gt_nocs': batch['y2']
        }
        pc_data = dict((x[0], to_numpy(x[1])) for x in pc_data_torch.items())
        output_pc_group = output_group.require_group(
            'point_cloud', overwrite=False)
        for key, data in pc_data.items():
            output_pc_group.array(
                name=key, data=data, chunks=data.shape,
                compressor=compressor, overwrite=True)

        # save predicted mesh data
        rot_mat_torch = batch['input_aug_rot_mat1']
        gt_mesh_nocs_points = batch['gt_surf_query_points2']
        if 'refined_surf_query_points2' in nocs_data:
            if use_refine_mesh_for_query:
                pred_mesh_nocs_points = nocs_data['refined_surf_query_points2']
            else:
                pred_mesh_nocs_points = batch['surf_query_points2']
        else:
            if current_mesh_nocs_points is not None:
                pred_mesh_nocs_points = current_mesh_nocs_points.clone()
            else:
                pred_mesh_nocs_points = gt_mesh_nocs_points.clone()

        surface_decoder_result = result['surface_decoder_result']
        pred_warpfield = surface_decoder_result['out_features']
        pred_sim_points = pred_warpfield

        pred_sim_points = pred_sim_points @ rot_mat_torch.T
        pred_sim_points = pred_sim_points.squeeze(-1).transpose(0, 1)
        gt_sim_points = batch['gt_sim_points2']
        gt_sim_points = gt_sim_points @ rot_mat_torch.T
        gt_sim_points = gt_sim_points.squeeze(-1).transpose(0, 1)

        mesh_data_torch = {'pred_nocs_points': pred_mesh_nocs_points,
                           'pred_sim_points': pred_sim_points,
                           'gt_sim_points': gt_sim_points,
                           'gt_nocs_points': gt_mesh_nocs_points}
        l2_norm_error = torch.mean(torch.norm(pred_sim_points - gt_sim_points, dim=1))
        print('interval {}, interval frame {}, video {}, video frame {}, error: {}'.format(
            grip_interval_id, current_interval_frame_idx, video_id, current_video_frame_idx, l2_norm_error.item()))
        mesh_data = dict((x[0], to_numpy(x[1])) for x in mesh_data_torch.items())
        output_mesh_points_group = output_group.require_group(
            'mesh_points', overwrite=False)
        for key, data in mesh_data.items():
            output_mesh_points_group.array(
                name=key, data=data, chunks=data.shape,
                compressor=compressor, overwrite=True)

        # copy gt mesh data
        rot_mat = np.squeeze(to_numpy(batch_cpu['input_aug_rot_mat1']))
        aug_keys = ['cloth_verts']
        input_mesh_group = input_group['mesh']
        output_mesh_group = output_group.require_group('gt_mesh', overwrite=False)
        for key, value in input_mesh_group.arrays():
            data = value[:]
            if key in aug_keys:
                data = data @ rot_mat.T
            output_mesh_group.array(
                name=key, data=data, chunks=data.shape,
                compressor=compressor, overwrite=True)

        # logging
        log_data = {
            'prediction_batch_idx': batch_idx,
            'prediction_video_id': video_id,
            'prediction_video_frame_idx': current_video_frame_idx,
        }
        wandb.log(
            data=log_data,
            step=batch_idx)

        is_last_video_frame = len(val_dataset.video_to_idxs_dict[str(video_id).zfill(6)]) \
                              == current_video_frame_idx + 1
        if use_grip_interval:
            is_last_interval_frame = len(val_dataset.interval_to_idxs_dict[grip_interval_id]) \
                                     == current_interval_frame_idx + 1
            last_grip_interval_ids = [val_dataset.idx_to_interval_list[idx]
                                      for idx in range(dataset_idx + 1,
                                                       val_dataset.video_to_idxs_dict[str(video_id).zfill(6)][-1])]
            is_last_interval = True
            for grip_interval_id in last_grip_interval_ids:
                if grip_interval_id != -1:
                    is_last_interval = False
            if cfg.prediction.use_cross_interval_tracking:
                is_last_frame = is_last_video_frame or is_last_interval
            else:
                is_last_frame = is_last_interval_frame or is_last_video_frame
        else:
            is_last_frame = is_last_video_frame

        if is_last_frame:
            if cfg.prediction.use_garmentnets_prediction and dataset_idx + 1 < len(val_dataset):
                # get next frame pc_sim
                start_idx = dataset_idx + 1
                test_group_row = val_dataset.groups_df.iloc[start_idx]
                group_key = test_group_row.group_key
                while group_key not in pred_samples_group and start_idx + 1 < len(val_dataset):
                    start_idx += 1
                    test_group_row = val_dataset.groups_df.iloc[start_idx]
                    group_key = test_group_row.group_key
                # calculate PC nocs of the first frame with garmentnets prediction
                next_sample_surface_nocs_points, next_sample_surface_sim_points = \
                    get_mc_surface(pred_samples_group, group_key,
                                   final_num_points=cfg.datamodule.num_surface_sample,
                                   value_threshold=cfg.prediction.value_threshold, seed=start_idx)
                val_dataset.set_prev_pose(None, None, start_idx,
                                          next_sample_surface_nocs_points, next_sample_surface_sim_points)
            else:
                val_dataset.set_prev_pose(None, None, None, None, None)
        else:
            assert model.batch_size == 1
            if current_valid_video_frame_idx < cfg.prediction.max_refine_mesh_step:
                current_mesh_nocs_points = pred_mesh_nocs_points.clone().contiguous()
            else:
                if val_dataset.prev_surf_query_points1 is not None:
                    current_mesh_nocs_points = torch.from_numpy(val_dataset.prev_surf_query_points1.copy())
                else:
                    current_mesh_nocs_points = gt_mesh_nocs_points.clone().contiguous()
            current_pc_nocs = pred_pc_nocs.clone().contiguous()
            val_dataset.set_prev_pose(to_numpy(current_mesh_nocs_points), to_numpy(current_pc_nocs), dataset_idx + 1,
                                      None, None)

        current_video_frame_idx += 1
        current_valid_video_frame_idx += 1
        current_interval_frame_idx += 1


# %%
# driver
if __name__ == "__main__":
    main()
