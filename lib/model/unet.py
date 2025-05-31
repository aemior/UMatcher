import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightUNet(nn.Module):
    def __init__(self, in_channels=1024, mid_channels=128, out_channels=64, normilize=True):
        super(LightweightUNet, self).__init__()
        self.normilize = normilize

        # Encoder
        self.enc1 = self.conv_block(in_channels, mid_channels, stride=2)
        self.enc2 = self.conv_block(mid_channels, mid_channels, stride=2)
        self.enc3 = self.conv_block(mid_channels, mid_channels, stride=2)
        self.enc4 = self.conv_block(mid_channels, mid_channels, stride=2)
        
        # Bottleneck
        self.bottleneck = self.conv_block(mid_channels*2, mid_channels)

        # Decoder
        self.upconv4 = self.upconv(mid_channels, mid_channels)
        self.dec4 = self.conv_block(mid_channels * 2, mid_channels)
        
        self.upconv3 = self.upconv(mid_channels, mid_channels)
        self.dec3 = self.conv_block(mid_channels * 2, mid_channels)
        
        self.upconv2 = self.upconv(mid_channels, mid_channels)
        self.dec2 = self.conv_block(mid_channels * 2, mid_channels)
        
        self.upconv1 = self.upconv(mid_channels, mid_channels)
        self.dec1 = self.conv_block(mid_channels + in_channels, out_channels)
        
        # Final layer
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels, stride=1, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, search_featuremap, template_embedding):
        # Encoder
        enc1 = self.enc1(search_featuremap)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        
        # Bottleneck
        # Normalize the output embeddings
        if self.normilize:
            enc4 = F.normalize(enc4, p=2, dim=1)
        fusion = torch.cat((enc4, template_embedding), dim=1)
        bottleneck = self.bottleneck(fusion)

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc3), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc2), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc1), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, search_featuremap), dim=1)
        dec1 = self.dec1(dec1)
        
        # Final layer
        out = self.final_conv(dec1)
        return out

class LightweightUNetX(nn.Module):
    def __init__(self, in_channels=1024, mid_channels=128, out_channels=64, normilize=True):
        super(LightweightUNetX, self).__init__()
        self.normilize = normilize

        # Encoder
        self.enc1 = self.conv_block(in_channels, mid_channels, stride=2)
        self.enc2 = self.conv_block(mid_channels, mid_channels, stride=2)
        self.enc3 = self.conv_block(mid_channels, mid_channels, stride=2)
        self.enc4 = self.conv_block(mid_channels, mid_channels, stride=2)
        
        # Bottleneck
        self.bottleneck = self.conv_block(mid_channels*2, mid_channels)

        # Decoder
        self.upconv4 = self.upconv(mid_channels, mid_channels)
        self.upconv4x = self.upconv(mid_channels, mid_channels)
        self.dec4 = self.conv_block(mid_channels * 3, mid_channels)
        
        self.upconv3 = self.upconv(mid_channels, mid_channels)
        self.upconv3x = self.upconv(mid_channels, mid_channels)
        self.dec3 = self.conv_block(mid_channels * 3, mid_channels)
        
        self.upconv2 = self.upconv(mid_channels, mid_channels)
        self.upconv2x = self.upconv(mid_channels, mid_channels)
        self.dec2 = self.conv_block(mid_channels * 3, mid_channels)
        
        self.upconv1 = self.upconv(mid_channels, mid_channels)
        self.upconv1x = self.upconv(mid_channels, mid_channels)
        self.dec1 = self.conv_block(mid_channels*2 + in_channels, out_channels)
        
        # Final layer
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels, stride=1, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, search_featuremap, template_embedding):
        # Encoder
        enc1 = self.enc1(search_featuremap)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        
        # Bottleneck
        # Normalize the output embeddings
        if self.normilize:
            enc4 = F.normalize(enc4, p=2, dim=1)
        fusion = torch.cat((enc4, template_embedding), dim=1)
        bottleneck = self.bottleneck(fusion)

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4x = self.upconv4x(template_embedding)
        dec4 = torch.cat((dec4, enc3, dec4x), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3x = self.upconv3x(dec4x)
        dec3 = torch.cat((dec3, enc2, dec3x), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2x = self.upconv2x(dec3x)
        dec2 = torch.cat((dec2, enc1, dec2x), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1x = self.upconv1x(dec2x)
        dec1 = torch.cat((dec1, search_featuremap, dec1x), dim=1)
        dec1 = self.dec1(dec1)
        
        # Final layer
        out = self.final_conv(dec1)
        return out