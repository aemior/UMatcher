import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.model.mobileone import mobileone
from lib.model.autoencoder import EmbeddingHead
from lib.model.init_weights import init_weights

class TemplateBranch(nn.Module):
    def __init__(self, backbone:nn.Module, head:nn.Module, embed_dim:int=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.backbone = backbone
        self.head = head 
        self.head.output_type = 'embedding'

    def forward(self, template_img):
        x = self.backbone(template_img)
        x = self.head(x[3])
        return x


def build_template_branch(cfg):

    if cfg.MODEL.BACKBONE.TYPE == 'mobileone':
        backbone = mobileone(backbone=True, out_stages=[0,1,2,3])
    else:
        raise ValueError(f"Invalid backbone type: {cfg.MODEL.BACKBONE.TYPE}")

    head = EmbeddingHead(cfg.MODEL.EMBEDDING_DIM)
    branch = TemplateBranch(backbone, head, embed_dim=cfg.MODEL.EMBEDDING_DIM)
    init_weights(branch.head)
    return branch