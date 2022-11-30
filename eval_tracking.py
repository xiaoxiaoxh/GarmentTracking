# %%
# set numpy threads
import os
os.environ["OMP_NUM_THREADS"] = "20"
os.environ["OPENBLAS_NUM_THREADS"] = "20"
os.environ["MKL_NUM_THREADS"] = "20"
os.environ["VECLIB_MAXIMUM_THREADS"] = "20"
os.environ["NUMEXPR_NUM_THREADS"] = "20"
# currently requires custom-built igl-python binding
os.environ["IGL_PARALLEL_FOR_NUM_THREADS"] = "1"
import numpy as np
import igl

# %%
# import
import pathlib
from pprint import pprint
import json

import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import zarr
from numcodecs import Blosc
from tqdm import tqdm

import pandas as pd
import numpy as np
from scipy.spatial import ckdtree
import igl

from common.parallel_util import parallel_map
from common.geometry_util import (
    AABBNormalizer, AABBGripNormalizer)


# %%
# helper functions
def write_dict_to_group(data, group, compressor):
    for key, data in data.items():
        if isinstance(data, np.ndarray):
            group.array(
                name=key, data=data, chunks=data.shape, 
                compressor=compressor, overwrite=True)
        else:
            group[key] = data

def compute_pc_metrics(sample_key, samples_group, nocs_aabb, **kwargs):
    sample_group = samples_group[sample_key]
    # io
    pc_group = sample_group['point_cloud']
    gt_nocs = pc_group['gt_nocs'][:]
    pred_nocs = pc_group['pred_nocs'][:]

    # transform
    normalizer = AABBNormalizer(nocs_aabb)
    gt_nocs = normalizer.inverse(gt_nocs)
    pred_nocs = normalizer.inverse(pred_nocs)

    # compute
    nocs_diff = pred_nocs - gt_nocs
    nocs_error_mean_per_dim = np.mean(np.abs(nocs_diff), axis=0)
    nocs_diff_std_per_dim = np.std(nocs_diff, axis=0)

    mirror_gt_nocs = gt_nocs.copy()
    mirror_gt_nocs[:, 0] = -mirror_gt_nocs[:, 0] 
    mirror_nocs_error = pred_nocs - mirror_gt_nocs
    nocs_error_dist = np.linalg.norm(nocs_diff, axis=1)
    mirror_nocs_error_dist = np.linalg.norm(mirror_nocs_error, axis=1)
    mirror_min_nocs_error_dist = np.minimum(nocs_error_dist, mirror_nocs_error_dist)
    
    metrics = {
        'nocs_pc_error_distance': np.mean(nocs_error_dist),
        'nocs_pc_mirror_error_distance': np.mean(mirror_nocs_error_dist),
        'nocs_pc_min_agg_error_distance': np.mean(mirror_min_nocs_error_dist),
        'nocs_pc_agg_min_error_distance': np.minimum(np.mean(nocs_error_dist), np.mean(mirror_nocs_error_dist))
    }
    axis_order = ['x', 'y', 'z']
    per_dim_features = {
        'nocs_pc_diff_std': nocs_diff_std_per_dim,
        'nocs_pc_error': nocs_error_mean_per_dim,
    }
    for key, value in per_dim_features.items():
        for i in range(3):
            metrics['_'.join([key, axis_order[i]])] = value[i]
    return metrics


def compute_chamfer(sample_key, samples_group, nocs_aabb,
        **kwargs):
    sample_group = samples_group[sample_key]

    mesh_points_group = sample_group['mesh_points']
    pred_sim_points = mesh_points_group['pred_sim_points'][:]
    gt_sim_points = mesh_points_group['gt_sim_points'][:]

    # compute chamfer distance
    def get_chamfer(pred_points, gt_points):
        pred_tree = ckdtree.cKDTree(pred_points)
        gt_tree = ckdtree.cKDTree(gt_points)
        forward_distance, forward_nn_idx = gt_tree.query(pred_points, k=1)
        backward_distance, backward_nn_idx = pred_tree.query(gt_points, k=1)
        forward_chamfer = np.mean(forward_distance)
        backward_chamfer = np.mean(backward_distance)
        symmetrical_chamfer = np.mean([forward_chamfer, backward_chamfer])
        result = {
            # 'chamfer_forward': forward_chamfer,
            # 'chamfer_backward': backward_chamfer,
            'chamfer_symmetrical': symmetrical_chamfer
        }
        return result
    
    in_data = {
        'sim': {
            'pred_points': pred_sim_points,
            'gt_points': gt_sim_points
        },
    }

    key_order = ['sim']
    old_in_data = in_data
    in_data = dict([(x, old_in_data[x]) for x in key_order if x in old_in_data])
    
    result = dict()
    for category, kwargs in in_data.items():
        out_data = get_chamfer(**kwargs)
        for key, value in out_data.items():
            result['_'.join([key, category])] = value
    return result


def compute_euclidian(
        sample_key,
        samples_group,
        **kwargs):
    sample_group = samples_group[sample_key]

    mesh_points_group = sample_group['mesh_points']
    pred_sim_points = mesh_points_group['pred_sim_points'][:]
    gt_sim_points = mesh_points_group['gt_sim_points'][:]

    # compute chamfer distance
    def get_euclidian(pred_points, gt_points):
        euclidian = np.mean(np.linalg.norm(pred_sim_points - gt_sim_points, axis=1))
        result = {
            'euclidian': euclidian
        }
        return result

    in_data = {
        'sim': {
            'pred_points': pred_sim_points,
            'gt_points': gt_sim_points
        },
    }

    key_order = ['sim']
    old_in_data = in_data
    in_data = dict([(x, old_in_data[x]) for x in key_order if x in old_in_data])

    result = dict()
    for category, kwargs in in_data.items():
        out_data = get_euclidian(**kwargs)
        for key, value in out_data.items():
            result['_'.join([key, category])] = value
    return result


# %%
# visualization functions
def get_task_mesh_vis(
        sample_key, 
        samples_group,
        offset=(0.8,0,0),
        save_path=None,
        **kwargs):
    """
    Visualizes task space result as a point cloud
    Order:  GT sim mesh Pred sim mesh Sim point cloud
    """
    sample_group = samples_group[sample_key]
    # io
    mesh_points_group = sample_group['mesh_points']
    pred_sim_points = mesh_points_group['pred_sim_points'][:]
    pred_nocs_points = mesh_points_group['pred_nocs_points'][:]
    gt_nocs_points = mesh_points_group['gt_nocs_points'][:]
    gt_sim_points = mesh_points_group['gt_sim_points'][:]
    if 'attention_score' in mesh_points_group:
        attention_score = mesh_points_group['attention_score'][:].repeat(3, axis=1)
    else:
        attention_score = None

    pc_group = sample_group['point_cloud']
    gt_input_pc = pc_group['input_points'][:]
    gt_input_rgb = pc_group['input_rgb'][:].astype(np.float32)
    pred_input_nocs = pc_group['pred_nocs'][:]
    gt_nocs_pc = pc_group['gt_nocs'][:]

    mesh_group = sample_group['gt_mesh']
    cloth_faces_tri = mesh_group['cloth_faces_tri'][:]

    # vis
    offset_vec = np.array(offset)
    gt_sim_pc = np.concatenate([gt_sim_points - offset_vec, gt_nocs_points * 255], axis=1)
    pred_sim_pc = np.concatenate([pred_sim_points, pred_nocs_points * 255], axis=1)
    pred_nocs_pc = np.concatenate([gt_input_pc + 2 * offset_vec, pred_input_nocs * 255], axis=1)
    gt_rgb_pc = np.concatenate([gt_input_pc + offset_vec, gt_input_rgb], axis=1)
    gt_nocs_pc = np.concatenate([gt_input_pc + 3 * offset_vec, gt_nocs_pc * 255], axis=1)
    if attention_score is not None:
        pred_att_pc = np.concatenate([pred_sim_points - 2 * offset_vec, attention_score * 255], axis=1)
        all_pc = np.concatenate([pred_att_pc, gt_sim_pc, pred_sim_pc, gt_rgb_pc, pred_nocs_pc, gt_nocs_pc], axis=0).astype(np.float32)
    else:
        all_pc = np.concatenate([gt_sim_pc, pred_sim_pc, gt_rgb_pc, pred_nocs_pc, gt_nocs_pc], axis=0).astype(np.float32)
    if save_path is not None:
        num_mesh_points = pred_sim_pc.shape[0]
        num_pc_points = gt_rgb_pc.shape[0]
        padding = np.array([[num_mesh_points, num_mesh_points, num_mesh_points,
                             num_pc_points, num_pc_points, num_pc_points]]).astype(np.float32)
        all_pc = np.concatenate([all_pc, padding], axis=0).astype(np.float32)
        np.save(save_path, all_pc)
        print('Saving to {}!'.format(save_path))
        np.save(save_path.replace('vis', 'vis_faces'), cloth_faces_tri)
        print('Saving to {}!'.format(save_path.replace('vis', 'vis_faces')))
    vis_obj = wandb.Object3D(all_pc)
    return vis_obj


def get_nocs_pc_vis(
        sample_key, 
        samples_group, 
        offset=[1.0,0,0], **kwargs):
    """
    GT nocs pc Pred nocs pc (colored with gt nocs)
    """
    sample_group = samples_group[sample_key]
    # io
    pc_group = sample_group['point_cloud']
    gt_nocs_pc = pc_group['gt_nocs'][:]
    pred_nocs_pc = pc_group['pred_nocs'][:]
    input_prev_nocs_pc = pc_group['input_prev_nocs'][:]
    if 'pred_nocs_confidence' in pc_group:
        pred_nocs_confidence = pc_group['pred_nocs_confidence'][:]
    else:
        pred_nocs_confidence = None

    # vis
    offset_vec = np.array(offset)
    gt_nocs_vis = np.concatenate([gt_nocs_pc - offset_vec, gt_nocs_pc * 255], axis=1)
    pred_nocs_vis = np.concatenate([pred_nocs_pc, gt_nocs_pc * 255], axis=1)
    input_prev_nocs_vis = np.concatenate([input_prev_nocs_pc + offset_vec, input_prev_nocs_pc * 255], axis=1)
    if pred_nocs_confidence is not None:
        pred_confidence_vis = np.concatenate([pred_nocs_pc + 2 * offset_vec, pred_nocs_confidence * 255], axis=1)
        all_pc = np.concatenate([gt_nocs_vis, pred_nocs_vis, input_prev_nocs_vis, pred_confidence_vis])
    else:
        all_pc = np.concatenate([gt_nocs_vis, pred_nocs_vis, input_prev_nocs_vis])
    vis_obj = wandb.Object3D(all_pc)
    return vis_obj


def get_nocs_mesh_vis(sample_key,
                      samples_group,
                      offset=[1.0,0,0], **kwargs):
    """
    GT nocs pc Pred nocs pc (colored with gt nocs)
    """
    sample_group = samples_group[sample_key]
    # io
    mesh_group = sample_group['mesh_points']
    gt_nocs_mesh = mesh_group['gt_nocs_points'][:]
    pred_nocs_mesh = mesh_group['pred_nocs_points'][:]

    # vis
    offset_vec = np.array(offset)
    gt_nocs_vis = np.concatenate([gt_nocs_mesh - offset_vec, gt_nocs_mesh * 255], axis=1)
    pred_nocs_vis = np.concatenate([pred_nocs_mesh, gt_nocs_mesh * 255], axis=1)
    all_pc = np.concatenate([gt_nocs_vis, pred_nocs_vis])
    vis_obj = wandb.Object3D(all_pc)
    return vis_obj


# %%
# main script
@hydra.main(config_path="config", 
    config_name="eval_tracking_default.yaml")
def main(cfg: DictConfig) -> None:
    # load datase
    pred_output_dir = os.path.expanduser(cfg.main.prediction_output_dir)
    pred_config_path = os.path.join(pred_output_dir, 'config.yaml')
    pred_config_all = OmegaConf.load(pred_config_path)

    # setup wandb
    output_dir = os.getcwd()
    print(output_dir)

    wandb_path = os.path.join(output_dir, 'wandb')
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
    all_config = {
        'config': OmegaConf.to_container(cfg, resolve=True),
        'prediction_config': OmegaConf.to_container(pred_config_all, resolve=True),
        'output_dir': output_dir,
        'wandb': wandb_meta,
        'meta': meta
    }
    yaml.dump(all_config, open('config.yaml', 'w'), default_flow_style=False)
    wandb.config.update(all_config)

    # setup zarr
    pred_zarr_path = os.path.join(pred_output_dir, 'prediction.zarr')
    pred_root = zarr.open(pred_zarr_path, 'r+')
    samples_group = pred_root['samples']
    summary_group = pred_root.require_group('summary', overwrite=False)
    compressor = Blosc(cname='zstd', clevel=6, shuffle=Blosc.BITSHUFFLE)

    sample_key, sample_group = next(iter(samples_group.groups()))
    print(sample_group.tree())
    all_sample_keys = list()
    all_sample_groups = list()
    for sample_key, sample_group in samples_group.groups():
        all_sample_keys.append(sample_key)
        all_sample_groups.append(sample_group)

    global_metrics_group = summary_group.require_group('metrics', overwrite=False)
    global_per_sample_group = global_metrics_group.require_group('per_sample', overwrite=False)
    global_agg_group = global_metrics_group.require_group('aggregate', overwrite=False)

    # write instance order
    sample_keys_arr = np.array(all_sample_keys)
    global_per_sample_group.array('sample_keys', sample_keys_arr, 
        chunks=sample_keys_arr.shape, compressor=compressor, overwrite=True)

    # load aabb
    input_zarr_path = os.path.expanduser(
        pred_config_all.config.datamodule.zarr_path)
    input_root = zarr.open(input_zarr_path, 'r')
    input_samples_group = input_root['samples']
    input_summary_group = input_root['summary']
    nocs_aabb = input_summary_group['cloth_canonical_aabb_union'][:]
    sim_aabb = input_summary_group['cloth_aabb_union'][:]

    num_workers = cfg.main.num_workers
    sample_keys_series = pd.Series(all_sample_keys)
    result_df = parallel_map(
            lambda x: False,
            sample_keys_series,
            num_workers=num_workers,
            preserve_index=True)
    is_sample_null = result_df.result
    not_null_sample_keys_series = sample_keys_series.loc[~is_sample_null]

    # compute metrics
    metric_func_dict = {
        'compute_pc_metrics': compute_pc_metrics,
        'compute_chamfer': compute_chamfer,
        'compute_euclidian': compute_euclidian,
    }

    num_workers = cfg.main.num_workers
    all_metrics = dict()
    for func_key, func in metric_func_dict.items():
        print("Running {}".format(func_key))
        metric_args = OmegaConf.to_container(cfg.eval[func_key], resolve=True)
        if not metric_args['enabled']:
            print("Disabled, skipping")
            continue

        print("Config:")
        pprint(metric_args)
        result_df = parallel_map(
            lambda x: func(
                sample_key=x, 
                samples_group=samples_group, 
                input_samples_group=input_samples_group,
                nocs_aabb=nocs_aabb,
                sim_aabb=sim_aabb,
                **metric_args),
            not_null_sample_keys_series,
            num_workers=num_workers,
            preserve_index=True)
        # print error
        errors_series = result_df.loc[result_df.error.notnull()].error
        if len(errors_series) > 0:
            print("Errors:")
            print(errors_series)

        result_dict = dict()
        for key in sample_keys_series.index:
            data = dict()
            if key in result_df.index:
                value = result_df.result.loc[key]
                if value is not None:
                    data = value
            result_dict[key] = data
        this_metric_df = pd.DataFrame(
            list(result_dict.values()),
            index=sample_keys_series.index)

        for column in this_metric_df:
            all_metrics[column] = this_metric_df[column]
            value = np.array(this_metric_df[column])
            global_per_sample_group.array(
                name=column, data=value, chunks=value.shape, 
                compressor=compressor, overwrite=True)
            value_agg = np.nanmean(value)
            global_agg_group[column] = value_agg

    all_metrics_df = pd.DataFrame(
        all_metrics, 
        index=sample_keys_series.index)
    all_metrics_df['null_percentage'] = is_sample_null.astype(np.float32)

    all_metrics_agg = all_metrics_df.mean()
    for column in all_metrics_df:
        if 'euclidian' in column:
            all_metrics_agg[column + '@0.03'] = (all_metrics_df[column] <= 0.03).sum() \
                                                / len(all_metrics_df[column])
            all_metrics_agg[column + '@0.05'] = (all_metrics_df[column] <= 0.05).sum() \
                                                / len(all_metrics_df[column])
            all_metrics_agg[column + '@0.08'] = (all_metrics_df[column] <= 0.08).sum() \
                                                / len(all_metrics_df[column])
            all_metrics_agg[column + '@0.1'] = (all_metrics_df[column] <= 0.1).sum() \
                                                / len(all_metrics_df[column])
            all_metrics_agg[column + '@0.15'] = (all_metrics_df[column] <= 0.15).sum() \
                                               / len(all_metrics_df[column])

    print(all_metrics_agg)
    # save metric to disk
    all_metrics_path = os.path.join(output_dir, 'all_metrics.csv')
    agg_path = os.path.join(output_dir, 'all_metrics_agg.csv')
    summary_path = os.path.join(output_dir, 'summary.json')
    all_metrics_df.to_csv(all_metrics_path)
    all_metrics_df.describe().to_csv(agg_path)
    json.dump(dict(all_metrics_agg), open(summary_path, 'w'), indent=2)

    if cfg.vis.samples_per_instance <= 0:
        print("Done!")
        return

    # visualization
    # pick best and worst
    rank_column = all_metrics_df[cfg.vis.rank_metric]
    sorted_rank_column = rank_column.sort_values()
    best_idxs = sorted_rank_column.index[:cfg.vis.num_best]
    worst_idxs = sorted_rank_column.index[-cfg.vis.num_best:][::-1]
    if cfg.vis.random_sample_regular:
        num_samples = len(sorted_rank_column)
        vis_idxs = np.random.choice(num_samples, size=cfg.vis.num_normal)
    else:
        start_idx, end_idx = cfg.vis.vis_sample_idxs_range
        vis_idxs = np.arange(start_idx, end_idx+1)

    print('vis_idxs: {}'.format(vis_idxs.tolist()))
    vis_idx_dict = dict()
    for i, idx in enumerate(vis_idxs):
        vis_idx_dict[idx] = "regular_{0:02d}".format(i)
    for i, idx in enumerate(best_idxs):
        vis_idx_dict[idx] = "best_{0:02d}".format(i)
    for i, idx in enumerate(worst_idxs):
        vis_idx_dict[idx] = "worst_{0:02d}".format(i)

    vis_func_dict = {
        'task_mesh_vis': get_task_mesh_vis,
        'nocs_pc_vis': get_nocs_pc_vis,
        'nocs_mesh_vis': get_nocs_mesh_vis,
    }
    no_override_keys = list()
    # all_log_data = list()
    print("Logging visualization to wandb")
    for i in tqdm(range(len(all_metrics_df))):
        log_data = dict(all_metrics_df.loc[i])
        if i in vis_idx_dict:
            vis_key = vis_idx_dict[i]
            if cfg.vis.save_point_cloud:
                save_dir = os.path.join(output_dir, 'vis')
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                os.makedirs(save_dir.replace('vis', 'vis_faces'), exist_ok=True)
                save_path = os.path.join(save_dir, '{:0>4d}.npy'.format(i))
            else:
                save_path = None
            for func_key, func in vis_func_dict.items():
                metric_args = OmegaConf.to_container(cfg.vis[func_key], resolve=True)
                sample_key = sample_keys_series.loc[i]
                vis_obj = func(sample_key, samples_group,
                    nocs_aabb=nocs_aabb,
                    sim_aabb=sim_aabb,
                    save_path=save_path,
                    **metric_args)
                vis_name = '_'.join([func_key, vis_key])
                log_data[vis_name] = vis_obj
        # all_log_data.append(log_data)
        wandb_run.log(log_data, step=i)

    print("Logging summary to wandb")
    for key, value in tqdm(all_metrics_agg.items()):
        wandb_run.summary[key] = value
    print("Done!")

# %%
# driver
if __name__ == "__main__":
    main()
