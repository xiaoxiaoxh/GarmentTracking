import torch
import torch.nn.functional as F
from typing import Optional, List
from torch import nn, Tensor
from components.gridding import VirtualGrid
from components.mlp import MLP
from networks.multihead_attention import MultiheadAttention


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel=3, num_pos_feats=256):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        # xyz : BxNx3
        xyz = xyz.transpose(1, 2).contiguous()
        # Bx3xN
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


class TransformerSiamese(nn.Module):
    def __init__(self,
                 input_channels=3,
                 use_xyz=True,
                 input_size=4000,
                 d_model=64,
                 num_layers=1,
                 key_feature_dim=128,
                 with_pos_embed=True,
                 encoder_pos_embed_input_dim=3,
                 decoder_pos_embed_input_dim=3,
                 fea_channels=(128, 256, 256),
                 nocs_bins=64,
                 nocs_channels=(256, 256, 3),
                 feat_slim_last_layer=True,
                 nocs_slim_last_layer=True,
                 inverse_source_template=False,
                 ):
        super(TransformerSiamese, self).__init__()
        self.input_channels = input_channels
        self.use_xyz = use_xyz
        self.input_size = input_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.nocs_bins = nocs_bins
        self.encoder_pos_embed_input_dim = encoder_pos_embed_input_dim
        self.decoder_pos_embed_input_dim = decoder_pos_embed_input_dim
        assert encoder_pos_embed_input_dim in (3, 6)
        self.with_pos_embed = with_pos_embed
        self.inverse_source_template = inverse_source_template

        multihead_attn = MultiheadAttention(
            feature_dim=d_model, n_head=1, key_feature_dim=key_feature_dim)

        if self.with_pos_embed:
            encoder_pos_embed = PositionEmbeddingLearned(encoder_pos_embed_input_dim, d_model)
            decoder_pos_embed = PositionEmbeddingLearned(decoder_pos_embed_input_dim, d_model)
        else:
            encoder_pos_embed = None
            decoder_pos_embed = None

        self.fea_layer = MLP(fea_channels, batch_norm=True, last_layer=feat_slim_last_layer)

        output_dim = 3
        if nocs_bins is not None:
            output_dim = nocs_bins * 3
        assert nocs_channels[-1] == output_dim
        self.nocs_layer = MLP(nocs_channels, batch_norm=True, last_layer=nocs_slim_last_layer)

        self.encoder = TransformerEncoder(
            multihead_attn=multihead_attn, FFN=None,
            d_model=d_model, num_encoder_layers=num_layers,
            self_posembed=encoder_pos_embed)
        self.decoder = TransformerDecoder(
            multihead_attn=multihead_attn, FFN=None,
            d_model=d_model, num_decoder_layers=num_layers,
            key_feature_dim=key_feature_dim,
            self_posembed=decoder_pos_embed)

    def transform_fuse(self, search_feature, search_coord,
                       template_feature, template_coord):
        """Use transformer to fuse feature.

        template_feature : (B, C, N)
        template_coord : (B, N, 3) or (B, N, 6)
        """
        # BxCxN -> NxBxC
        search_feature = search_feature.permute(2, 0, 1)
        template_feature = template_feature.permute(2, 0, 1)

        ## encoder
        encoded_memory = self.encoder(template_feature,
                                      query_pos=template_coord if self.with_pos_embed else None)

        encoded_feat = self.decoder(search_feature,
                                    memory=encoded_memory,
                                    query_pos=search_coord)  # NxBxC

        # NxBxC -> BxNxC
        encoded_feat = encoded_feat.permute(1, 0, 2)
        encoded_feat = self.fea_layer(encoded_feat)  # BxNxC

        return encoded_feat

    def forward(self, template_feature, template_coord,
                search_feature, search_coord):
        """
            template_feature: (B*N, C)
            template_coord: (B*N, 3) or (B*N, 6)
            search_feature: (B*N, C)
            search_coord: (B*N, 3)
        """
        feature_size = template_feature.shape[-1]
        template_feature = template_feature.reshape(-1, self.input_size, feature_size).permute(0, 2, 1)  # (B, C, N)
        search_feature = search_feature.reshape(-1, self.input_size, feature_size).permute(0, 2, 1)  # (B, C, N)
        template_coord = template_coord.reshape(-1, self.input_size, template_coord.shape[-1])  # (B, N, 3 or 6)
        search_coord = search_coord.reshape(-1, self.input_size, 3)  # (B, N, 3)
        batch_size = template_feature.shape[0]

        if self.inverse_source_template:
            fusion_feature = self.transform_fuse(
                template_feature, template_coord, search_feature, search_coord)  # (B, N, C)
        else:
            fusion_feature = self.transform_fuse(
                search_feature, search_coord, template_feature, template_coord)  # (B, N, C)
        pred_nocs = self.nocs_layer(fusion_feature)  # (B, N, C'*3)
        if self.nocs_bins is None:
            # direct regression
            pred_nocs = torch.sigmoid(pred_nocs)  # (B, N, C'*3)

        fusion_feature = fusion_feature.reshape(batch_size * self.input_size, -1)  # (B*N, C)
        pred_nocs = pred_nocs.reshape(batch_size * self.input_size, -1)  # (B*N, C'*3)
        return fusion_feature, pred_nocs

    def logits_to_nocs(self, logits):
        nocs_bins = self.nocs_bins
        if nocs_bins is None:
            # directly regress from nn
            return logits

        # reshape
        logits_bins = None
        if len(logits.shape) == 2:
            logits_bins = logits.reshape((logits.shape[0], nocs_bins, 3))
        elif len(logits.shape) == 1:
            logits_bins = logits.reshape((nocs_bins, 3))

        bin_idx_pred = torch.argmax(logits_bins, dim=1, keepdim=False)

        # turn into per-channel classification problem
        vg = self.get_virtual_grid(logits.get_device())
        points_pred = vg.idxs_to_points(bin_idx_pred)
        return points_pred

    def get_virtual_grid(self, device):
        nocs_bins = self.nocs_bins
        vg = VirtualGrid(lower_corner=(0, 0, 0), upper_corner=(1, 1, 1),
                         grid_shape=(nocs_bins,) * 3, batch_size=1,
                         device=device, int_dtype=torch.int64,
                         float_dtype=torch.float32)
        return vg


class InstanceL2Norm(nn.Module):
    """Instance L2 normalization.
    """
    def __init__(self, size_average=True, eps=1e-5, scale=1.0):
        super().__init__()
        self.size_average = size_average
        self.eps = eps
        self.scale = scale

    def forward(self, input):
        if self.size_average:
            return input * (self.scale * ((input.shape[1] * input.shape[2] * input.shape[3]) / (
                        torch.sum((input * input).reshape(input.shape[0], 1, 1, -1), dim=3, keepdim=True) + self.eps)).sqrt())  # view
        else:
            return input * (self.scale / (torch.sum((input * input).reshape(input.shape[0], 1, 1, -1), dim=3, keepdim=True) + self.eps).sqrt())


class TransformerEncoderLayer(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model, self_posembed=None):
        super().__init__()
        self.self_attn = multihead_attn
        # Implementation of Feedforward model
        self.FFN = FFN
        self.norm = nn.InstanceNorm1d(d_model)
        self.self_posembed = self_posembed

        self.dropout = nn.Dropout(0.1)

    def with_pos_embed(self, tensor, pos_embed: Optional[Tensor]):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, query_pos=None):
        # BxNxC -> BxCxN -> NxBxC
        if self.self_posembed is not None and query_pos is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)
        else:
            query_pos_embed = None
        query = key = value = self.with_pos_embed(src, query_pos_embed)

        # self-attention
        # NxBxC
        src2 = self.self_attn(query=query, key=key, value=value)
        src = src + src2

        # NxBxC -> BxCxN -> NxBxC
        src = self.norm(src.permute(1, 2, 0)).permute(2, 0, 1)
        return F.relu(src)
        # return src


class TransformerEncoder(nn.Module):
    def __init__(self, multihead_attn, FFN,
                 d_model=512,
                 num_encoder_layers=6,
                 activation="relu",
                 self_posembed=None):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            multihead_attn, FFN, d_model, self_posembed=self_posembed)
        self.layers = _get_clones(encoder_layer, num_encoder_layers)

    def forward(self, src, query_pos=None):
        num_imgs, batch, dim = src.shape
        output = src

        for layer in self.layers:
            output = layer(output, query_pos=query_pos)

        # import pdb; pdb.set_trace()
        # [L,B,D] -> [B,D,L]
        # output_feat = output.reshape(num_imgs, batch, dim)
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model, key_feature_dim, self_posembed=None):
        super().__init__()
        self.self_attn = multihead_attn
        self.cross_attn = MultiheadAttention(
            feature_dim=d_model,
            n_head=1, key_feature_dim=key_feature_dim)

        self.FFN = FFN
        self.norm1 = nn.InstanceNorm1d(d_model)
        self.norm2 = nn.InstanceNorm1d(d_model)
        self.self_posembed = self_posembed

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def with_pos_embed(self, tensor, pos_embed: Optional[Tensor]):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, tgt, memory, query_pos=None):
        if self.self_posembed is not None and query_pos is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)
        else:
            query_pos_embed = None
        # NxBxC

        # self-attention
        query = key = value = self.with_pos_embed(tgt, query_pos_embed)

        tgt2 = self.self_attn(query=query, key=key, value=value)
        # tgt2 = self.dropout1(tgt2)
        tgt = tgt + tgt2
        # tgt = F.relu(tgt)
        # tgt = self.instance_norm(tgt, input_shape)
        # NxBxC
        # tgt = self.norm(tgt)
        tgt = self.norm1(tgt.permute(1, 2, 0)).permute(2, 0, 1)
        tgt = F.relu(tgt)

        mask = self.cross_attn(
            query=tgt, key=memory, value=memory)
        # mask = self.dropout2(mask)
        tgt2 = tgt + mask
        tgt2 = self.norm2(tgt2.permute(1, 2, 0)).permute(2, 0, 1)

        tgt2 = F.relu(tgt2)
        return tgt2


class TransformerDecoder(nn.Module):
    def __init__(self, multihead_attn, FFN,
                 d_model=512,
                 num_decoder_layers=6,
                 key_feature_dim=64,
                 self_posembed=None):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(
            multihead_attn, FFN, d_model, key_feature_dim, self_posembed=self_posembed)
        self.layers = _get_clones(decoder_layer, num_decoder_layers)

    def forward(self, tgt, memory, query_pos=None):
        assert tgt.dim() == 3, 'Expect 3 dimensional inputs'
        tgt_shape = tgt.shape
        num_imgs, batch, dim = tgt.shape

        output = tgt
        for layer in self.layers:
            output = layer(output, memory, query_pos=query_pos)
        return output


def _get_clones(module, N):
    return nn.ModuleList([module for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
