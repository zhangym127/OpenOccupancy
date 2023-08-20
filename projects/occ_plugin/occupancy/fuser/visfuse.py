import random
from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from mmdet3d.models.builder import FUSION_LAYERS
from mmcv.cnn import build_norm_layer


@FUSION_LAYERS.register_module()
class VisFuser(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if norm_cfg is None:
            norm_cfg = dict(type='BN3d', eps=1e-3, momentum=0.01)

        self.img_enc = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 7, padding=3, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            # nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
        )
        self.pts_enc = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 7, padding=3, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            # nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
        )
        self.vis_enc = nn.Sequential(
            nn.Conv3d(2*out_channels, 16, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, 16)[1],
            # nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.Conv3d(16, 1, 1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    # @实现图像和点云体素特征的自适应融合
    def forward(self, img_voxel_feats, pts_voxel_feats):
        # 分别对图像和点云体素特征进行3D卷积
        img_voxel_feats = self.img_enc(img_voxel_feats)
        pts_voxel_feats = self.pts_enc(pts_voxel_feats)
        # 沿特征通道对卷积后的特征进行拼接，并归一化，获得逐体素特征的融合比例系数
        vis_weight = self.vis_enc(torch.cat([img_voxel_feats, pts_voxel_feats], dim=1))
        # 依据比例系数进行逐体素特征融合
        voxel_feats = vis_weight * img_voxel_feats + (1 - vis_weight) * pts_voxel_feats

        return voxel_feats
