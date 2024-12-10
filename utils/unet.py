# U-Net Class
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil, floor
from openpyxl.styles.builtins import output

# We define a U-Net class with a classical architecture
class UNet(nn.Module):
    # We are dealing with RGB images and binary segmentation
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.input_channels = in_channels
        # In a regular U-Net there are 3 stages:
        # the downsampling, followed by the center and finally the upsampling
        # We follow the paper picture : https://apprendre-le-deep-learning.com/u-net-une-architecture-pour-la-segmentation-d-images/
        # Downsampling layers
        # 2 consecutive blue arrows
        self.double_conv1 = self.double_conv(in_channels, 64)
        # red arrow
        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 2 consecutive blue arrows
        self.double_conv2 = self.double_conv(64, 128)
        # red arrow
        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 2 consecutive blue arrows
        self.double_conv3 = self.double_conv(128, 256)
        # red arrow
        self.down3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 2 consecutive blue arrows
        self.double_conv4 = self.double_conv(256, 512)
        # red arrow
        self.down4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Center
        # 2 consecutive blue arrows
        self.center = self.double_conv(512, 1024)
        # Upsampling layers
        # up green arrow
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        # 2 consecutive blue arrows
        self.double_conv5 = self.double_conv(1024, 512)
        # up green arrow
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # 2 consecutive blue arrows
        self.double_conv6 = self.double_conv(512, 256)
        # up green arrow
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # 2 consecutive blue arrows
        self.double_conv7 = self.double_conv(256, 128)
        # up green arrow
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # 2 consecutive blue arrows
        self.double_conv8 = self.double_conv(128, 64)
        # Final layer (dark green arrow)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    @staticmethod
    def double_conv(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def copy_and_crop(left, right):
        # Padding to adjust tensor size to be able to concatenate tensors (grey arrow) along th channel
        y_padding = right.size(2) - left.size(2)
        x_padding = right.size(3) - left.size(3)
        left = F.pad(left, [floor(x_padding/2), ceil(x_padding/2), floor(y_padding/2), ceil(y_padding/2)])
        return torch.cat([left, right], dim=1)

    def forward(self, x):
        x1 = self.double_conv1(x)
        x2 = self.double_conv2(self.down1(x1))
        x3 = self.double_conv3(self.down2(x2))
        x4 = self.double_conv4(self.down3(x3))
        x_center = self.center(self.down4(x4))
        x_up1 = self.up1(x_center)
        # grey arrow 1
        x_up1 = self.double_conv5(self.copy_and_crop(x4, x_up1))
        x_up2 = self.up2(x_up1)
        # grey arrow 2
        x_up2 = self.double_conv6(self.copy_and_crop(x3, x_up2))
        x_up3 = self.up3(x_up2)
        # grey arrow 3
        x_up3 = self.double_conv7(self.copy_and_crop(x2, x_up3))
        x_up4 = self.up4(x_up3)
        # grey arrow 4
        x_up4 = self.double_conv8(self.copy_and_crop(x1, x_up4))
        return self.final_conv(x_up4)