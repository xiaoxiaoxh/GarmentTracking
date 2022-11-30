import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import wandb
import torch_scatter

from components.unet3d import Abstract3DUNet, DoubleConv
from components.mlp import MLP, MLP_V2
from networks.resunet import SparseResUNet2
from networks.transformer import TransformerSiamese
from networks.pointnet import MiniPointNetfeat
from common.torch_util import to_numpy
from common.visualization_util import (
    get_vis_idxs, render_nocs_pair)
from components.gridding import VirtualGrid

import MinkowskiEngine as ME


class VolumeTrackingFeatureAggregator(pl.LightningModule):
    def __init__(self,
                 nn_channels=(38, 64, 64),
                 batch_norm=True,
                 lower_corner=(0,0,0),
                 upper_corner=(1,1,1),
                 grid_shape=(32, 32, 32),
                 reduce_method='max',
                 include_point_feature=True,
                 use_gt_nocs_for_train=True,
                 use_mlp_v2=True,
                 ):
        super().__init__()
        self.save_hyperparameters()
        if use_mlp_v2:
            self.local_nn2 = MLP_V2(nn_channels, batch_norm=batch_norm, transpose_input=True)
        else:
            self.local_nn2 = MLP(nn_channels, batch_norm=batch_norm)
        self.lower_corner = tuple(lower_corner)
        self.upper_corner = tuple(upper_corner)
        self.grid_shape = tuple(grid_shape)
        self.reduce_method = reduce_method
        self.include_point_feature = include_point_feature
        self.use_gt_nocs_for_train = use_gt_nocs_for_train
    
    def forward(self, nocs_data, batch_size, is_train=False):
        lower_corner = self.lower_corner
        upper_corner = self.upper_corner
        grid_shape = self.grid_shape
        include_point_feature = self.include_point_feature
        reduce_method = self.reduce_method

        sim_points_frame2 = nocs_data['sim_points_frame2']
        if is_train and self.use_gt_nocs_for_train:
            points_frame2 = nocs_data['pos_gt_frame2']
        else:
            if 'refined_pos_frame2' in nocs_data:
                points_frame2 = nocs_data['refined_pos_frame2']
            else:
                points_frame2 = nocs_data['pos_frame2']
        nocs_features_frame2 = nocs_data['x_frame2']
        batch_idx_frame2 = nocs_data['batch_frame2']
        device = points_frame2.device
        float_dtype = points_frame2.dtype
        int_dtype = torch.int64

        vg = VirtualGrid(
            lower_corner=lower_corner, 
            upper_corner=upper_corner, 
            grid_shape=grid_shape, 
            batch_size=batch_size,
            device=device,
            int_dtype=int_dtype,
            float_dtype=float_dtype)
        
        # get aggregation target index
        points_grid_idxs_frame2 = vg.get_points_grid_idxs(points_frame2, batch_idx=batch_idx_frame2)
        flat_idxs_frame2 = vg.flatten_idxs(points_grid_idxs_frame2, keepdim=False)

        # get features
        features_list_frame2 = [nocs_features_frame2]
        if include_point_feature:
            points_grid_points_frame2 = vg.idxs_to_points(points_grid_idxs_frame2)
            local_offset_frame2 = points_frame2 - points_grid_points_frame2
            features_list_frame2.append(local_offset_frame2)
            features_list_frame2.append(sim_points_frame2)

        features_frame2 = torch.cat(features_list_frame2, axis=-1)
        
        # per-point transform
        if self.local_nn2 is not None:
            features_frame2 = self.local_nn2(features_frame2)

        # scatter
        volume_feature_flat_frame2 = torch_scatter.scatter(
            src=features_frame2.T, index=flat_idxs_frame2, dim=-1,
            dim_size=vg.num_grids, reduce=reduce_method)

        # reshape to volume
        feature_size = volume_feature_flat_frame2.shape[0]
        volume_feature_all = volume_feature_flat_frame2.reshape(
            (feature_size, batch_size) + grid_shape).permute((1,0,2,3,4))
        return volume_feature_all


class UNet3DTracking(pl.LightningModule):
    def __init__(self, in_channels, out_channels, f_maps=64, 
            layer_order='gcr', num_groups=8, num_levels=4):
        super().__init__()
        self.save_hyperparameters()
        self.abstract_3d_unet = Abstract3DUNet(
            in_channels=in_channels, out_channels=out_channels,
            final_sigmoid=False, basic_module=DoubleConv, f_maps=f_maps,
            layer_order=layer_order, num_groups=num_groups, 
            num_levels=num_levels, is_segmentation=False)
    
    def forward(self, data):
        result = self.abstract_3d_unet(data)
        return result


class ImplicitWNFDecoder(pl.LightningModule):
    def __init__(self,
            nn_channels=(128,256,256,3),
            batch_norm=True,
            last_layer_mlp=False,
            use_mlp_v2=False
            ):
        super().__init__()
        self.save_hyperparameters()
        if use_mlp_v2:
            self.mlp = MLP_V2(nn_channels, batch_norm=batch_norm, transpose_input=True)
        else:
            self.mlp = MLP(nn_channels, batch_norm=batch_norm, last_layer=last_layer_mlp)

    def forward(self, features_grid, query_points):
        """
        features_grid: (N,C,D,H,W)
        query_points: (N,M,3)
        """
        batch_size = features_grid.shape[0]
        if len(query_points.shape) == 2:
            query_points = query_points.view(batch_size, -1, 3)
        # normalize query points to (-1, 1), which is 
        # requried by grid_sample
        query_points_normalized = 2.0 * query_points - 1.0
        # shape (N,C,M,1,1)
        sampled_features = F.grid_sample(
            input=features_grid, 
            grid=query_points_normalized.view(
                *(query_points_normalized.shape[:2] + (1,1,3))), 
            mode='bilinear', padding_mode='border',
            align_corners=True)
        # shape (N,M,C)
        sampled_features = sampled_features.view(
            sampled_features.shape[:3]).permute(0,2,1)
        
        # shape (N,M,C)
        out_features = self.mlp(sampled_features)

        return out_features


class PointMeshNocsRefiner(pl.LightningModule):
    def __init__(self,
                 pc_pointnet_channels=(326, 256, 256, 1024),
                 mesh_pointnet_channels=(3, 64, 128, 1024),
                 pc_refine_mlp_channels=(2304, 1024, 512, 192),
                 mesh_refine_pointnet_channels=(2112, 512, 512, 1024),
                 mesh_refine_mlp_channels=(1024, 512, 256, 6),
                 detach_input_pc_feature=True,
                 detach_global_pc_feature=True,
                 detach_global_mesh_feature=True,
                 **kwargs):
        super(PointMeshNocsRefiner, self).__init__()
        self.pc_pointnet = MiniPointNetfeat(nn_channels=pc_pointnet_channels)
        self.mesh_pointnet = MiniPointNetfeat(nn_channels=mesh_pointnet_channels)
        self.mesh_refine_pointnet = MiniPointNetfeat(nn_channels=mesh_refine_pointnet_channels)
        self.pc_refine_mlp = MLP_V2(channels=pc_refine_mlp_channels, batch_norm=True)
        self.mesh_refine_mlp = MLP_V2(channels=mesh_refine_mlp_channels, batch_norm=True)
        self.detach_input_pc_feature = detach_input_pc_feature
        self.detach_global_pc_feature = detach_global_pc_feature
        self.detach_global_mesh_feature = detach_global_mesh_feature

    def forward(self, pc_nocs, pc_sim, pc_cls_logits, pc_feat, mesh_nocs, batch_size=16, **kwargs):
        """
        NOCS refiner to refine predicted PC NOCS
        :param pc_nocs: (B*N, 3)
        :param pc_sim: (B*N, 3)
        :param pc_cls_logits: (B*N, 64*3)
        :param pc_feat: (B*N, C)
        :param mesh_nocs: (B*M, 3)
        :param batch_size: int
        :return: refined PC NOCS logits (B*N, 64*3), refined mesh NOCS (B*M, 3)
        """
        pc_nocs = pc_nocs.view(batch_size, -1, 3)  # (B, N, 3)
        pc_sim = pc_sim.view(batch_size, -1, 3)  # (B, N, 3)
        mesh_nocs = mesh_nocs.view(batch_size, -1, 3)  # (B, M, 3)
        num_pc_points = pc_nocs.shape[1]  # N
        num_mesh_points = mesh_nocs.shape[1]  # M
        # detach logits
        pc_cls_logits = pc_cls_logits.view(batch_size, num_pc_points, -1).detach()  # (B, N, 64*3)
        pc_feat = pc_feat.view(batch_size, num_pc_points, -1)  # (B, N, C)
        if self.detach_input_pc_feature:
            pc_feat = pc_feat.detach()
        pc_input_all = torch.cat([pc_nocs, pc_sim, pc_cls_logits, pc_feat], dim=-1)  # (B, N, 3+3+64*3+C)
        pc_input_all = pc_input_all.transpose(1, 2)  # (B, 3+3+64*3+C, N)

        # pc pointnet
        pc_feat_dense, pc_feat_global = self.pc_pointnet(pc_input_all)  # (B, C', N), (B, C')

        # mesh pointnet
        mesh_input_all = mesh_nocs.transpose(1, 2)  # (B, 3, M)
        mesh_feat_dense, _ = self.mesh_pointnet(mesh_input_all)  # (B, C', M)
        pc_feat_global_expand = pc_feat_global.unsqueeze(2).expand(-1, -1, num_mesh_points)  # (B, C', M)
        if self.detach_global_pc_feature:
            pc_feat_global_expand = pc_feat_global_expand.detach()
        mesh_feat_dense_cat = torch.cat([mesh_feat_dense, pc_feat_global_expand], dim=1)  # (B, C'+C', M)
        _, mesh_refine_feat_global = self.mesh_refine_pointnet(mesh_feat_dense_cat)  # (B, C')

        # refine pc-nocs
        mesh_refine_feat_global_expand = mesh_refine_feat_global.unsqueeze(2).expand(-1,  -1, num_pc_points)  # (B, C', N)
        if self.detach_global_mesh_feature:
            mesh_refine_feat_global_expand = mesh_refine_feat_global_expand.detach()
        pc_feat_cat = torch.cat([pc_feat_dense, mesh_refine_feat_global_expand], dim=1)  # (B, C'+C', N)
        delta_pc_cls_logits = self.pc_refine_mlp(pc_feat_cat)  # (B, 64*3, N)
        delta_pc_cls_logits = delta_pc_cls_logits.transpose(1, 2)  # (B, N, 64*3)
        assert delta_pc_cls_logits.shape[-1] == pc_cls_logits.shape[-1]
        refine_pc_cls_logits = pc_cls_logits + delta_pc_cls_logits  # (B, N, 64*3)
        refine_pc_cls_logits = refine_pc_cls_logits.reshape(batch_size * num_pc_points, -1)  # (B*N, 64*3)

        # refine mesh-nocs
        mesh_refine_feat_global = mesh_refine_feat_global.unsqueeze(2)  # (B, C', 1)
        mesh_nocs_delta_logits = self.mesh_refine_mlp(mesh_refine_feat_global)  # (B, 6, 1)
        refine_offset, refine_scale = mesh_nocs_delta_logits[:, :3, 0].unsqueeze(1), \
                                      mesh_nocs_delta_logits[:, 3:, 0].unsqueeze(1)  # (B, 1, 3)
        nocs_center = torch.tensor([[[0.5, 0.5, 0.5]]]).to(mesh_nocs.get_device())  # (1, 1, 3)
        refined_mesh_nocs = (mesh_nocs - nocs_center) * refine_scale + nocs_center + refine_offset  # (B, M, 3)
        refined_mesh_nocs = refined_mesh_nocs.reshape(batch_size * num_mesh_points, -1)  # (B*M, 3)
        return refine_pc_cls_logits, refined_mesh_nocs


class GarmentTrackingPipeline(pl.LightningModule):
    """
    Use sparse ResUNet as backbone
    Use point-cloud pair as input(not mesh)
    Add transformer for self-attention and cross-attention
    """
    def __init__(self,
                 # sparse uned3d encoder params
                 sparse_unet3d_encoder_params,
                 # self-attention and cross-attention transformer params
                 transformer_params,
                 # pc nocs and mesh nocs refiner params
                 nocs_refiner_params,
                 # VolumeFeaturesAggregator params
                 volume_agg_params,
                 # unet3d params
                 unet3d_params,
                 # ImplicitWNFDecoder params
                 surface_decoder_params,
                 # training params
                 learning_rate=1e-4,
                 optimizer_type='Adam',
                 loss_type='l2',
                 volume_loss_weight=1.0,
                 warp_loss_weight=10.0,
                 nocs_loss_weight=1.0,
                 mesh_loss_weight=10.0,
                 use_nocs_refiner=True,
                 disable_pc_nocs_refine_in_test=False,
                 disable_mesh_nocs_refine_in_test=False,
                 # vis params
                 vis_per_items=0,
                 max_vis_per_epoch_train=0,
                 max_vis_per_epoch_val=0,
                 batch_size=None,
                 # debug params
                 debug=False
                 ):
        super().__init__()
        self.save_hyperparameters()

        criterion = None
        if loss_type == 'l2':
            criterion = nn.MSELoss(reduction='mean')
        elif loss_type == 'smooth_l1':
            criterion = nn.SmoothL1Loss(reduction='mean')
        else:
            raise RuntimeError("Invalid loss_type: {}".format(loss_type))

        self.sparse_unet3d_encoder = SparseResUNet2(**sparse_unet3d_encoder_params)
        self.transformer_siamese = TransformerSiamese(**transformer_params)
        self.nocs_refiner = PointMeshNocsRefiner(**nocs_refiner_params)
        self.volume_agg = VolumeTrackingFeatureAggregator(**volume_agg_params)
        self.unet_3d = UNet3DTracking(**unet3d_params)
        self.surface_decoder = ImplicitWNFDecoder(**surface_decoder_params)
        self.use_nocs_refiner = use_nocs_refiner
        self.disable_pc_nocs_refine_in_test = disable_pc_nocs_refine_in_test
        self.disable_mesh_nocs_refine_in_test = disable_mesh_nocs_refine_in_test
        self.mesh_loss_weight = mesh_loss_weight

        self.surface_criterion = criterion
        if self.transformer_siamese.nocs_bins is not None:
            self.nocs_criterion = nn.CrossEntropyLoss()
        else:
            self.nocs_criterion = criterion
        self.mesh_criterion = criterion

        self.volume_loss_weight = volume_loss_weight
        self.nocs_loss_weight = nocs_loss_weight
        self.warp_loss_weight = warp_loss_weight

        self.learning_rate = learning_rate
        assert optimizer_type in ('Adam', 'SGD')
        self.optimizer_type = optimizer_type
        self.vis_per_items = vis_per_items
        self.max_vis_per_epoch_train = max_vis_per_epoch_train
        self.max_vis_per_epoch_val = max_vis_per_epoch_val
        self.batch_size = batch_size
        self.debug = debug

    # forward function for each stage
    # ===============================
    def encoder_forward(self, data, is_train=False):
        input1 = ME.SparseTensor(data['feat1'], coordinates=data['coords1'])
        input2 = ME.SparseTensor(data['feat2'], coordinates=data['coords2'])

        features_frame1_sparse, _ = self.sparse_unet3d_encoder(input1)
        features_frame2_sparse, _ = self.sparse_unet3d_encoder(input2)
        features_frame1 = features_frame1_sparse.F
        features_frame2 = features_frame2_sparse.F

        pc_nocs_frame1 = data['y1']
        pos_gt_frame2 = data['y2']
        per_point_batch_idx_frame1 = data['pc_batch_idx1']
        per_point_batch_idx_frame2 = data['pc_batch_idx2']
        sim_points_frame1 = data['pos1']
        sim_points_frame2 = data['pos2']

        if (self.transformer_siamese.encoder_pos_embed_input_dim == 6 and
            not self.transformer_siamese.inverse_source_template)\
                or (self.transformer_siamese.decoder_pos_embed_input_dim == 6 and
                    self.transformer_siamese.inverse_source_template):
            frame1_coord = torch.cat([sim_points_frame1, pc_nocs_frame1], dim=-1)
        elif self.transformer_siamese.encoder_pos_embed_input_dim == 3:
            frame1_coord = sim_points_frame1
        else:
            raise NotImplementedError

        fusion_feature, pc_logits_frame2 = self.transformer_siamese(features_frame1, frame1_coord,
                                                                    features_frame2, sim_points_frame2)

        if self.transformer_siamese.nocs_bins is not None:
            # NOCS classification
            vg = self.transformer_siamese.get_virtual_grid(pc_logits_frame2.get_device())
            nocs_bins = self.transformer_siamese.nocs_bins
            pred_logits_bins = pc_logits_frame2.reshape(
                (pc_logits_frame2.shape[0], nocs_bins, 3))
            nocs_bin_idx_pred = torch.argmax(pred_logits_bins, dim=1)
            pc_nocs_frame2 = vg.idxs_to_points(nocs_bin_idx_pred)
        else:
            # NOCS regression
            pc_nocs_frame2 = pc_logits_frame2

        nocs_data = dict(
            x_frame2=fusion_feature,
            pos_frame1=pc_nocs_frame1,
            pos_frame2=pc_nocs_frame2,
            logits_frame2=pc_logits_frame2,
            pos_gt_frame2=pos_gt_frame2,
            batch_frame1=per_point_batch_idx_frame1,
            batch_frame2=per_point_batch_idx_frame2,
            sim_points_frame1=sim_points_frame1,
            sim_points_frame2=sim_points_frame2)

        if self.use_nocs_refiner:
            assert self.transformer_siamese.nocs_bins is not None
            mesh_nocs = data['surf_query_points2']
            batch_size = torch.max(per_point_batch_idx_frame2).item() + 1
            gt_mesh_nocs = data['gt_surf_query_points2'] if is_train else None
            refined_pc_logits_frame2, refined_surf_query_points2 = \
                self.nocs_refiner(pc_nocs_frame2, sim_points_frame2, pc_logits_frame2, fusion_feature, mesh_nocs,
                                  batch_size, is_train=is_train, gt_mesh_nocs=gt_mesh_nocs)
            if self.disable_mesh_nocs_refine_in_test:
                nocs_data['refined_surf_query_points2'] = mesh_nocs
            else:
                nocs_data['refined_surf_query_points2'] = refined_surf_query_points2
            if self.disable_pc_nocs_refine_in_test:
                refined_pc_logits_frame2 = pc_logits_frame2

            nocs_data['refined_logits_frame2'] = refined_pc_logits_frame2

            # get NOCS coordinates from logits
            vg = self.transformer_siamese.get_virtual_grid(pc_logits_frame2.get_device())
            nocs_bins = self.transformer_siamese.nocs_bins
            refined_pred_logits_bins = refined_pc_logits_frame2.reshape(
                (refined_pc_logits_frame2.shape[0], nocs_bins, 3))
            refined_nocs_bin_idx_pred = torch.argmax(refined_pred_logits_bins, dim=1)
            refined_pc_nocs_frame2 = vg.idxs_to_points(refined_nocs_bin_idx_pred)
            nocs_data['refined_pos_frame2'] = refined_pc_nocs_frame2

        return nocs_data
    
    def unet3d_forward(self, encoder_result, is_train=False):
        # volume agg
        in_feature_volume = self.volume_agg(encoder_result, self.batch_size, is_train)

        # unet3d
        out_feature_volume = self.unet_3d(in_feature_volume)
        unet3d_result = {
            'out_feature_volume': out_feature_volume
        }
        return unet3d_result

    def surface_decoder_forward(self, unet3d_result, query_points):
        out_feature_volume = unet3d_result['out_feature_volume']
        out_features = self.surface_decoder(out_feature_volume, query_points)
        decoder_result = {
            'out_features': out_features
        }
        return decoder_result

    # forward
    # =======
    def forward(self, data, is_train=False, use_refine_mesh_for_query=True):
        encoder_result = self.encoder_forward(data, is_train)
        if is_train:
            surface_query_points = data['gt_surf_query_points2']
        else:
            if use_refine_mesh_for_query:
                surface_query_points = encoder_result['refined_surf_query_points2']
            else:
                surface_query_points = data['surf_query_points2']
        unet3d_result = self.unet3d_forward(encoder_result, is_train)

        surface_decoder_result = self.surface_decoder_forward(
            unet3d_result, surface_query_points)

        result = {
            'encoder_result': encoder_result,
            'unet3d_result': unet3d_result,
            'surface_decoder_result': surface_decoder_result
        }
        return result

    # training
    # ========
    def configure_optimizers(self):
        if self.optimizer_type == 'Adam':
            return optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            return NotImplementedError

    def vis_batch(self, batch, batch_idx, result, is_train=False, img_size=256):
        nocs_data = result['encoder_result']
        pred_nocs = nocs_data['pos_frame2'].detach()
        if self.use_nocs_refiner:
            refined_pred_pc_nocs = nocs_data['refined_pos_frame2'].detach()
            refined_pred_mesh_nocs = nocs_data['refined_surf_query_points2'].detach()
        gt_nocs = nocs_data['pos_gt_frame2']
        batch_idxs = nocs_data['batch_frame2']
        this_batch_size = len(batch['dataset_idx1'])
        gt_mesh_nocs = batch['gt_surf_query_points2']

        vis_per_items = self.vis_per_items
        batch_size = self.batch_size
        if is_train:
            max_vis_per_epoch = self.max_vis_per_epoch_train
            prefix = 'train_'
        else:
            max_vis_per_epoch = self.max_vis_per_epoch_val
            prefix = 'val_'

        _, selected_idxs, vis_idxs = get_vis_idxs(batch_idx, 
            batch_size=batch_size, this_batch_size=this_batch_size, 
            vis_per_items=vis_per_items, max_vis_per_epoch=max_vis_per_epoch)
        
        log_data = dict()
        for i, vis_idx in zip(selected_idxs, vis_idxs):
            label = prefix + str(vis_idx)
            is_this_item = (batch_idxs == i)
            this_gt_nocs = to_numpy(gt_nocs[is_this_item])
            this_pred_nocs = to_numpy(pred_nocs[is_this_item])
            if self.use_nocs_refiner:
                refined_label = prefix + 'refine_pc_' + str(vis_idx)
                this_refined_pred_pc_nocs = to_numpy(refined_pred_pc_nocs[is_this_item])
                refined_pc_nocs_img = render_nocs_pair(this_gt_nocs, this_refined_pred_pc_nocs,
                                                    None, None, img_size=img_size)
                log_data[refined_label] = [wandb.Image(refined_pc_nocs_img, caption=refined_label)]

                refined_label = prefix + 'refine_mesh_' + str(vis_idx)
                this_refined_pred_mesh_nocs = to_numpy(refined_pred_mesh_nocs.reshape(this_batch_size, -1, 3)[i])
                this_gt_mesh_nocs = to_numpy(gt_mesh_nocs.reshape(this_batch_size, -1, 3)[i])
                refined_mesh_nocs_img = render_nocs_pair(this_gt_mesh_nocs, this_refined_pred_mesh_nocs,
                                                    None, None, img_size=img_size)
                log_data[refined_label] = [wandb.Image(refined_mesh_nocs_img, caption=refined_label)]

            nocs_img = render_nocs_pair(this_gt_nocs, this_pred_nocs, 
                None, None, img_size=img_size)
            img = nocs_img
            log_data[label] = [wandb.Image(img, caption=label)]

        return log_data

    def infer(self, batch, batch_idx, is_train=True):
        if len(batch) == 0:
            return dict(loss=torch.tensor(0., device='cuda:0', requires_grad=True))
        try:
            result = self(batch, is_train=is_train)
            encoder_result = result['encoder_result']

            # NOCS error distance
            pred_nocs = encoder_result['pos_frame2']
            gt_nocs = encoder_result['pos_gt_frame2']
            nocs_err_dist = torch.norm(pred_nocs - gt_nocs, dim=-1).mean()

            # NOCS loss
            if self.transformer_siamese.nocs_bins is not None:
                # classification
                nocs_bins = self.transformer_siamese.nocs_bins
                vg = self.transformer_siamese.get_virtual_grid(pred_nocs.get_device())
                pred_logits = encoder_result['logits_frame2']
                pred_logits_bins = pred_logits.reshape(
                    (pred_logits.shape[0], nocs_bins, 3))
                gt_nocs_idx = vg.get_points_grid_idxs(gt_nocs)
                nocs_loss = self.nocs_criterion(pred_logits_bins, gt_nocs_idx) * self.nocs_loss_weight
            else:
                # regression
                nocs_loss = self.nocs_criterion(pred_nocs, gt_nocs) * self.nocs_loss_weight

            if self.use_nocs_refiner:
                # only support classification
                assert self.transformer_siamese.nocs_bins is not None
                nocs_bins = self.transformer_siamese.nocs_bins
                refined_pred_nocs = encoder_result['refined_pos_frame2']
                refined_nocs_err_dist = torch.norm(refined_pred_nocs - gt_nocs, dim=-1).mean()

                vg = self.transformer_siamese.get_virtual_grid(refined_pred_nocs.get_device())
                refined_pred_logits = encoder_result['refined_logits_frame2']
                refined_pred_logits_bins = refined_pred_logits.reshape(
                    (refined_pred_logits.shape[0], nocs_bins, 3))
                gt_nocs_idx = vg.get_points_grid_idxs(gt_nocs)
                refined_nocs_loss = self.nocs_criterion(refined_pred_logits_bins, gt_nocs_idx) * self.nocs_loss_weight

                refined_mesh_nocs = encoder_result['refined_surf_query_points2']
                gt_mesh_nocs = batch['gt_surf_query_points2']
                mesh_loss = self.mesh_loss_weight * self.mesh_criterion(refined_mesh_nocs, gt_mesh_nocs)
                refined_mesh_err_dist = torch.norm(refined_mesh_nocs - gt_mesh_nocs, dim=-1).mean()

            # warp field loss (surface loss)
            surface_decoder_result = result['surface_decoder_result']
            surface_criterion = self.surface_criterion
            pred_warp_field = surface_decoder_result['out_features']
            gt_sim_points_frame2 = batch['gt_sim_points2']
            gt_warpfield = gt_sim_points_frame2.reshape(pred_warp_field.shape)
            warp_loss = surface_criterion(pred_warp_field, gt_warpfield)
            warp_loss = self.warp_loss_weight * warp_loss

            loss_dict = {
                'nocs_err_dist': nocs_err_dist,
                'nocs_loss': nocs_loss,
                'warp_loss': warp_loss,
            }
            if self.use_nocs_refiner:
                loss_dict['refined_nocs_loss'] = refined_nocs_loss
                loss_dict['refined_nocs_err_dist'] = refined_nocs_err_dist
                loss_dict['mesh_loss'] = mesh_loss
                loss_dict['refined_mesh_err_dist'] = refined_mesh_err_dist

            metrics = dict(loss_dict)
            metrics['loss'] = nocs_loss + warp_loss
            if self.use_nocs_refiner:
                metrics['loss'] += refined_nocs_loss
                metrics['loss'] += mesh_loss

            for key, value in metrics.items():
                log_key = ('train_' if is_train else 'val_') + key
                self.log(log_key, value)
            log_data = self.vis_batch(batch, batch_idx, result, is_train=is_train)
            self.logger.log_metrics(log_data, step=self.global_step)
        except Exception as e:
            raise e

        return metrics

    def training_step(self, batch, batch_idx):
        # torch.cuda.empty_cache()
        metrics = self.infer(batch, batch_idx, is_train=True)
        return metrics['loss']

    def validation_step(self, batch, batch_idx):
        # torch.cuda.empty_cache()
        metrics = self.infer(batch, batch_idx, is_train=False)
        return metrics['loss']
