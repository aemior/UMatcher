import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.model.head import build_box_head
from lib.model.mobileone import mobileone
from lib.model.unet import LightweightUNet, LightweightUNetX
from lib.model.init_weights import init_weights

class SearchBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = None
        self.neck = None
        self.head = None 

    def forward(self, search_img, template_embedding):
        x = self.backbone(search_img)
        x = self.neck(x[1], template_embedding)
        x = self.head(x)
        return x

def build_search_branch(cfg):
    branch = SearchBranch()

    if cfg.MODEL.BACKBONE.TYPE == 'mobileone':
        branch.backbone = mobileone(backbone=True)
    else:
        raise ValueError(f"Invalid backbone type: {cfg.MODEL.BACKBONE.TYPE}")

    if cfg.MODEL.NECK.TYPE == 'unet':
        branch.neck = LightweightUNet(in_channels=cfg.MODEL.FEATUREMAP_CHANNELS,
                                      mid_channels=cfg.MODEL.EMBEDDING_DIM,
                                      out_channels=cfg.MODEL.HEAD.NUM_CHANNELS,
                                      normilize=cfg.MODEL.NECK.NORMILIZE)
    elif cfg.MODEL.NECK.TYPE == 'unetx':
        branch.neck = LightweightUNetX(in_channels=cfg.MODEL.FEATUREMAP_CHANNELS,
                                    mid_channels=cfg.MODEL.EMBEDDING_DIM,
                                    out_channels=cfg.MODEL.HEAD.NUM_CHANNELS,
                                    normilize=cfg.MODEL.NECK.NORMILIZE)
    init_weights(branch.neck)

    branch.head = build_box_head(cfg, cfg.MODEL.HIDDEN_DIM)
    init_weights(branch.head)

    return branch