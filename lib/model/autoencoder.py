import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.model.mobileone import mobileone
from lib.model.init_weights import init_weights

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = None
        self.head = None 
        self.decoder = None

    def forward(self, template_img, noisy_embedding=None):
        x_a = self.backbone(template_img)
        x_b = self.head(x_a[3])
        if noisy_embedding is not None:
            # If noisy embedding is provided, add it to the last feature map
            template_img_hat = self.decoder(x_a + x_b[:-1] + tuple(noisy_embedding))
        else:
            template_img_hat = self.decoder(x_a + x_b)
        return x_b[-1], template_img_hat


class UNetDecoder(nn.Module):
    """
    A UNet-based decoder module for upsampling feature maps to reconstruct an image.
    Args:
        embedding_dim (int): The dimension of the input feature map. Default is 128.
        feature_channels (list): List of feature channels from the encoder.
        out_channels_list (list): List of output channels for each upconv block.
    """
    def __init__(self, feature_channels=[128, 128, 128, 256, 128, 48, 48], out_channels_list=[128, 256, 128, 64, 64, 48]):
        super(UNetDecoder, self).__init__()
        
        # Define a block for upsampling and convolution
        def upconv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        # Define decoder layers using loops
        self.upconvs = nn.ModuleList()
        in_channels = 0
        last_out_channels = 0
        for i, out_channels in enumerate(out_channels_list):
            in_channels = last_out_channels + feature_channels[i]
            print(f"Decoder block {i}: {in_channels} -> {out_channels}")
            self.upconvs.append(upconv_block(in_channels, out_channels))
            last_out_channels = out_channels

        in_channels = last_out_channels + feature_channels[-1]
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

    def forward(self, features):
        x = features[-1]  # Start from the feature map of the neck layer

        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            x = torch.cat((x, features[-(i+2)]), dim=1)
        
        x = self.final_conv(x)  # No activation in the last step, directly output a 3-channel image
        return x

class EmbeddingHead(nn.Module):
    def __init__(self, embedding_dim=128, output_type='embedding'):
        super(EmbeddingHead, self).__init__()
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1)  # Output: n x 128 x 4 x 4
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)  # Output: n x 64 x 2 x 2
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(128, embedding_dim, kernel_size=3, stride=2, padding=1)  # Output: n x embedding_dim x 1 x 1
        self.bn3 = nn.BatchNorm2d(embedding_dim)
        self.relu3 = nn.ReLU(inplace=True)

        self.output_type = output_type

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu1(x)
        
        x = self.conv2(x0)
        x = self.bn2(x)
        x1 = self.relu2(x)
        
        x = self.conv3(x1)
        x = self.bn3(x)
        x2 = self.relu3(x)

        # Check if we're in ONNX export mode
        if torch.onnx.is_in_onnx_export():
            # For ONNX export, don't normalize
            if self.output_type == 'embedding':
                return x2
            return x0, x1, x2
        else:
            # Normalize the output embeddings
            x2 = F.normalize(x2, p=2, dim=1)
            
            if self.output_type == 'embedding':
                return x2
            return x0, x1, x2

def build_auto_encoder(cfg):
    branch = AutoEncoder()

    if cfg.MODEL.BACKBONE.TYPE == 'mobileone':
        branch.backbone = mobileone(backbone=True, out_stages=[0,1,2,3])
    else:
        raise ValueError(f"Invalid backbone type: {cfg.MODEL.BACKBONE.TYPE}")

    branch.head = EmbeddingHead(cfg.MODEL.EMBEDDING_DIM, output_type='autoencoder')
    branch.decoder = UNetDecoder()
    init_weights(branch.head)
    init_weights(branch.decoder)

    return branch