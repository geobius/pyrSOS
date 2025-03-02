
import torch
import torch.nn as nn
from torch.nn.modules import padding

class Double_Convolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        return self.conv_op(x)


class Unet(nn.Module):
    def __init__(self, n_channels, n_labels, depth):

        super().__init__()
        self.depth = depth

        self.encoders = nn.ModuleList(
            [Double_Convolution(n_channels if d == 0 else 64*2**(d-1), 64*2**d) for d in range(depth)])
        self.decoders = nn.ModuleList(
            [Double_Convolution(64*2**(d+1), 64*2**d) for d in reversed(range(depth))])
        self.down_convs = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=2) for _ in range(depth)])
        self.up_convs = nn.ModuleList(
            [nn.ConvTranspose2d(64*2**(d+1), 64*2**d, kernel_size=2, stride=2) for d in reversed(range(depth))])

        self.bottleneck = Double_Convolution(64*2**(depth-1), 64*2**depth)
        self.segmentation_head = nn.Conv2d(64, n_labels, kernel_size=1)


    def forward(self, pre_image, post_image):

        x = torch.cat((pre_image, post_image), dim=1)
        skip_connections = []

        for (encoder, down_conv) in zip(self.encoders, self.down_convs):
            x = encoder(x)
            skip_connections.append(x)
            x = down_conv(x)

        x = self.bottleneck(x)

        for (skip, up_conv, decoder) in zip(reversed(skip_connections), self.up_convs, self.decoders):
            x = up_conv(x)
            x = torch.cat((skip, x), dim=1)
            x = decoder(x)

        logit_segmentation = self.segmentation_head(x)

        return logit_segmentation

#pre = torch.rand(2,4,128,128)
#post= torch.rand(2,4,128,128)
#model = Unet(8,2,1)
#res = model(pre, post)
