# GarmentTracking

## Datasets
All the data are stored in [zarr](https://zarr.readthedocs.io/en/stable/) format. Please put all the data under `%PROJECT_DIR/data`.

- Folding
  - `vr_simulation_folding2_dataset.zarr/Tshirt`
  - `vr_simulation_folding2_dataset.zarr/Trousers`
  - `vr_simulation_folding2_dataset.zarr/Top`
  - `vr_simulation_folding2_dataset.zarr/Skirt`
- Flattening
  - `vr_simulation_flattening-grasp2_dataset.zarr/Tshirt`
  - `vr_simulation_flattening-grasp2_dataset.zarr/Trousers`
  - `vr_simulation_flattening-grasp2_dataset.zarr/Top`
  - `vr_simulation_flattening-grasp2_dataset.zarr/Skirt`

## Environment

### Requirements

- Python >= 3.7
- Pytorch >= 1.9.1
- CUDA >= 11.1

Please use the following commands to setup environments (we highly recommend installing Pytorch with pip for compatibility).

```
conda create -n garment_tracking python=3.9
conda activate garment_tracking
```

```bash
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```
```bash
conda install -y openblas-devel igl -c anaconda -c conda-forge
```

```bash
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
```
```bash
pip install torch-geometric torch-scatter torch_sparse torch_cluster torchmetrics==0.5.1 open3d pandas wandb pytorch-lightning==1.4.9 hydra-core scipy==1.7.0 scikit-image matplotlib zarr numcodecs tqdm dask numba
```



## Training

Here is the example for training ( `Tshirt`, `Folding` task):

```bash
python train_tracking.py datamodule.zarr_path=data/vr_simulation_folding2_dataset.zarr/Tshirt logger.offline=False  logger.name=Tshirt-folding2-tracking
```

Here `logger.offline=False` will enable online syncing (eg. losses, logs, visualization) for [wandb](wandb.ai). You can use offline syncing mode by setting`logger.offline=True`. 

Each running will create a new working directory (eg. `2022-11-03/12-33-00`) under `%PROJECT_DIR/outputs` which contains all the checkpoints and logs.

## Inference

Here are some examples for training ( `Tshirt`, `Folding` task):

- First-frame Initialization with GT:

```bash
python predict_tracking.py datamodule.zarr_path=data/vr_simulation_folding2_dataset.zarr/Tshirt datamodule.use_fist_frame_pc_nocs_aug_in_test=False datamodule.use_pc_nocs_frame1_aug=False datamodule.use_mesh_nocs_aug=False datamodule.use_fist_frame_mesh_nocs_aug_in_test=False main.checkpoint_path=/home/xuehan/GarmentTracking/outputs/2022-11-03/12-33-00/checkpoints/last.ckpt prediction.use_garmentnets_prediction=False logger.name=Tshirt-folding2-tracking_test-gt
```

- First-frame Initialization with noise:

```bash
python predict_tracking.py datamodule.zarr_path=data/vr_simulation_folding2_dataset.zarr/Tshirt datamodule.use_fist_frame_pc_nocs_aug_in_test=True datamodule.use_pc_nocs_frame1_aug=True datamodule.use_mesh_nocs_aug=True datamodule.use_fist_frame_mesh_nocs_aug_in_test=True datamodule.pc_nocs_global_scale_aug_range=[0.8,1.2] datamodule.pc_nocs_global_max_offset_aug=0.1 datamodule.pc_nocs_gaussian_std=0.05 datamodule.mesh_nocs_global_scale_aug_range=[0.8,1.2] prediction.max_refine_mesh_step=1 main.checkpoint_path=/home/xuehan/GarmentTracking/outputs/2022-11-03/12-33-00/checkpoints/last.ckpt  logger.name=Tshirt-folding2-tracking_test-noise
```

## Evaluation

Here is the example for evaluation ( `Tshirt`, `Folding` task):

```bash
python eval_tracking.py main.prediction_output_dir=/home/xuehan/GarmentTracking/outputs/2022-11-07/14-48-52  logger.name=Tshirt-folding2-tracking-base10_test-gt
```

The evaluation will also generate some visualization examples in the form of logs in [wandb](wandb.ai). You can set `logger.offline=False` if you want to enable automatic online syncing for [wandb](wandb.ai). You can also manually sync the logs later in offline mode by default.
