import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
from utils import Return_root_dir
from logger import setup_logger
root_dir = Return_root_dir()
train_log_path = os.path.join(root_dir,"Log.txt")
train_logger = setup_logger('Model_logger', train_log_path)

# Input ---->[batch_size, num_channels, num_frames, height, width]
class CombinedVideoModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CombinedVideoModel, self).__init__()

        # Video U-Net for video enhancement or segmentation
        self.unet = VideoUNet(in_channels, out_channels)

        # Video ESRGAN for super-resolution (upscaling)
        self.esrgan = VideoESRGAN(in_channels, out_channels)

        # Video Discriminator for adversarial loss
      #  self.discriminator = VideoDiscriminator(in_channels)

        # Video Patch Discriminator for patch-wise adversarial loss
        #self.patch_discriminator = VideoPatchDiscriminator(in_channels)

    def forward(self, x):
        # Forward pass through U-Net for enhancement or segmentation
        unet_output = self.unet(x)

        # Forward pass through ESRGAN for super-resolution
        esgran_output = self.esrgan(unet_output)

        # Forward pass through Discriminators (for adversarial training)
        #disc_output = self.discriminator(esgran_output)
        #patch_disc_output = self.patch_discriminator(esgran_output)

        return esgran_output, unet_output




#INPUT----->[batch_size, 3, 7, 448, 256]
class VideoUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VideoUNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose3d(1024, 512, kernel_size=(1,2,2), stride=(1,2,2))
        self.dec4 = self.conv_block(1024, 512)

        self.up3 = nn.ConvTranspose3d(512, 256, kernel_size=(1,2,2), stride=(1,2,2))
        self.dec3 = self.conv_block(512, 256)

        self.up2 = nn.ConvTranspose3d(256, 128, kernel_size=(1,2,2), stride=(1,2,2))
        self.dec2 = self.conv_block(256, 128)

        self.up1 = nn.ConvTranspose3d(128, 64, kernel_size=(1,2,2), stride=(1,2,2))
        self.dec1 = self.conv_block(128, 64)

        # Final convolution
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool3d(enc1, (1,2,2)))
        enc3 = self.enc3(F.max_pool3d(enc2, (1,2,2)))
        enc4 = self.enc4(F.max_pool3d(enc3, (1,2,2)))

        bottleneck = self.bottleneck(F.max_pool3d(enc4, (1,2,2)))

        dec4 = self.up4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.up3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        return self.final_conv(dec1)
    



#INPUT--->[batch_size, 3, 7, 448, 256]


class VideoESRGAN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VideoESRGAN, self).__init__()

        # Initial feature extraction
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)

        # Residual Blocks
        self.rrdb_blocks = nn.Sequential(
            *[ResidualDenseBlock(64) for _ in range(16)] 
        )

        # Upsampling layers
        self.upsample1 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.upsample1_scale = nn.Upsample(scale_factor=(1,2,2), mode="trilinear", align_corners=False)

        self.upsample2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.upsample2_scale = nn.Upsample(scale_factor=(1,2,2), mode="trilinear", align_corners=False)

        # Final convolution
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)  # Initial feature extraction
        x = self.rrdb_blocks(x)  # Residual processing

        # Upscaling
        x = self.upsample1(x)
        x = self.upsample1_scale(x)

        x = self.upsample2(x)
        x = self.upsample2_scale(x)

        return self.final_conv(x)





class VideoDiscriminator(nn.Module):
    def __init__(self, in_channels):
        super(VideoDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, stride=(1,2,2), padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(64, 128, kernel_size=3, stride=(1,2,2), padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(128, 256, kernel_size=3, stride=(1,2,2), padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(256, 512, kernel_size=3, stride=(1,2,2), padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(512, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x)





class VideoPatchDiscriminator(nn.Module):
    def __init__(self, in_channels):
        super(VideoPatchDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(512, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Outputs probability
        )

    def forward(self, x):
        return self.model(x)



class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.leaky_relu(self.conv1(x))
        out = self.leaky_relu(self.conv2(out))
        out = self.leaky_relu(self.conv3(out))
        return x + out  
