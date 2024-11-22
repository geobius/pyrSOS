
import torch
import torch.nn as nn
from torch.nn.modules import padding

class Double_Convolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_op(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = Double_Convolution(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip_connection = self.conv_op(x)
        downsampling = self.pool(skip_connection)

        return skip_connection, downsampling


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv_op = Double_Convolution(in_channels, out_channels)

    def forward(self, x, skip_connection):
        upsampling = self.upsample(x)
        merged = torch.cat([upsampling, skip_connection], 1)

        return self.conv_op(merged)

class Unet(nn.Module):
    def __init__(self, in_channels, out_classes):
        super().__init__()
        self.Down_Block1 = Encoder(in_channels, 64)
        self.Down_Block2 = Encoder(64, 128)
        self.Down_Block3 = Encoder(128, 256)
        self.Down_Block4 = Encoder(256, 512)

        self.Bottleneck = Double_Convolution(512, 1024)

        self.Up_Block4 = Decoder(1024, 512)
        self.Up_Block3 = Decoder(512, 256)
        self.Up_Block2 = Decoder(256, 128)
        self.Up_Block1 = Decoder(128, 64)

        self.Segmentation_Head = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, pre_image, post_image):
        #pre_image is not used but I keep it as a parameter for concistency of calling convention
        skip1, down1 = self.Down_Block1(post_image)
        skip2, down2 = self.Down_Block2(down1)
        skip3, down3 = self.Down_Block3(down2)
        skip4, down4 = self.Down_Block4(down3)

        b = self.Bottleneck(down4)

        up4 = self.Up_Block4(b, skip4)
        up3 = self.Up_Block3(up4, skip3)
        up2 = self.Up_Block2(up3, skip2)
        up1 = self.Up_Block1(up2, skip1)

        output = self.Segmentation_Head(up1)

        return output
