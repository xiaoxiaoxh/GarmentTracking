# %%
# import
import os
import pathlib
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

from datasets.tracking_dataset import SparseUnet3DTrackingDataModule2
from networks.tracking_network import GarmentTrackingPipeline

# %%
# main script
@hydra.main(config_path="config", config_name="train_tracking_default")
def main(cfg: DictConfig) -> None:
    # hydra creates working directory automatically
    print(os.getcwd())
    os.mkdir("checkpoints")

    datamodule = SparseUnet3DTrackingDataModule2(**cfg.datamodule)
    batch_size = datamodule.kwargs['batch_size']

    pipeline_model = GarmentTrackingPipeline(
        batch_size=batch_size, **cfg.garment_tracking_model)
    
    category = pathlib.Path(cfg.datamodule.zarr_path).stem
    cfg.logger.tags.append(category)
    logger = pl.loggers.WandbLogger(
        project=os.path.basename(__file__),
        **cfg.logger)
    # logger.watch(pipeline_model, **cfg.logger_watch)
    wandb_run = logger.experiment
    wandb_meta = {
        'run_name': wandb_run.name,
        'run_id': wandb_run.id
    }

    all_config = {
        'config': OmegaConf.to_container(cfg, resolve=True),
        'output_dir': os.getcwd(),
        'wandb': wandb_meta
    }
    yaml.dump(all_config, open('config.yaml', 'w'), default_flow_style=False)
    logger.log_hyperparams(all_config)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch}-{val_loss:.4f}",
        monitor='val_loss',
        save_last=True,
        save_top_k=5,
        mode='min', 
        save_weights_only=False, 
        every_n_epochs=1,
        save_on_train_epoch_end=True)
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        checkpoint_callback=True,
        logger=logger, 
        check_val_every_n_epoch=1,
        **cfg.trainer)
    trainer.fit(model=pipeline_model, datamodule=datamodule)

# %%
# driver
if __name__ == "__main__":
    main()
