""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, trilinear=False):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.trilinear = trilinear

        self.inc = (DoubleConv(n_channels, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        factor = 2 if trilinear else 1
        self.up1 = (Up(256, 128 // factor, trilinear))
        self.up2 = (Up(128, 64 // factor, trilinear))
        self.up3 = (Up(64, 32, trilinear))
        self.outc = (OutConv(32, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits