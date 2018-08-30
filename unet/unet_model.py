#!/usr/bin/python
# full assembly of the sub-parts to form the complete net

import torch
import torch.nn as nn
import torch.nn.functional as F
# python 3 confusing imports :(
from .unet_parts import *
import time


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.scnn = scnn(256, 256)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(256, n_classes)
        self.upb = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        x1 = self.inc(x)  # 64, 480, 640
        x2 = self.down1(x1)  # 128, 240, 320
        x3 = self.down2(x2)  # 256, 120, 160
        x4 = self.down3(x3)  # 512, 60, 80
        x5 = self.down4(x4)  # 512, 30, 40
        #x5 = self.scnn(x5)  # 512, 30, 40

        x = self.up1(x5, x4)  # 256, 60, 80
        x = self.scnn(x)
        x = self.outc(x)  # 3, 480, 640
        x = self.upb(x)
        x = self.upb(x)
        x = self.upb(x)
        # x = self.up2(x, x3)  # 128, 120, 160
        # x = self.up3(x, x2)  # 64, 240, 320
        # x = self.up4(x, x1)  # 64, 480, 640
        return x
