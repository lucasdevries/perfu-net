import torch
import torch.nn as nn
from einops.einops import rearrange

class DownSampleConvBlock(nn.Module):
    """Depthwise_1DConv.
    """

    def __init__(self, in_channels, non_linear=True, stride=(2, 2)):
        super().__init__()
        convlayer = nn.Conv2d(in_channels=in_channels,
                              out_channels=in_channels,
                              kernel_size=3,
                              padding=(1, 1),
                              stride=stride,
                              groups=in_channels)
        if non_linear:
            bn = nn.BatchNorm2d(in_channels)
            relu = nn.ReLU()
            self.convlayers = nn.Sequential(*[convlayer, bn, relu])
        else:
            self.convlayers = nn.Sequential(*[convlayer])

    def forward(self, x):
        x = self.convlayers(x)
        return x

class UpSampleConvBlock(nn.Module):
    """Depthwise_1DConv.
    """

    def __init__(self, in_channels, non_linear=True, stride=(2, 2)):
        super().__init__()
        convlayer = torch.nn.ConvTranspose2d(in_channels=in_channels,
                                             out_channels=in_channels,
                                             kernel_size=3,
                                             stride=stride,
                                             padding=1,
                                             output_padding=1)
        if non_linear:
            bn = nn.BatchNorm2d(in_channels)
            relu = nn.ReLU()
            self.convlayers = nn.Sequential(*[convlayer, bn, relu])
        else:
            self.convlayers = nn.Sequential(*[convlayer])

    def forward(self, x):
        x = self.convlayers(x)
        return x

class RBNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_channels = 2
        self.out_channels = 2
        self.nonlinear_downsampling = self.config.nonlinear_downsampling

        self.conv1 = self.conv_block_2d(self.in_channels*self.config.clip_length, 32, 3, 1)
        self.pool1 = DownSampleConvBlock(in_channels=32, non_linear=self.nonlinear_downsampling)

        self.conv2 = self.conv_block_2d(32, 64, 3, 1)
        self.pool2 = DownSampleConvBlock(in_channels=64, non_linear=self.nonlinear_downsampling)

        self.conv3 = self.conv_block_2d(64, 128, 3, 1)
        self.pool3 = DownSampleConvBlock(in_channels=128, non_linear=self.nonlinear_downsampling)

        self.conv4 = self.conv_block_2d(128, 256, 3, 1)

        self.upconv3 = self.conv_block_2d(256, 256, 3, 1)
        self.upsample3= UpSampleConvBlock(in_channels=256, non_linear=self.nonlinear_downsampling)

        self.upconv2 = self.conv_block_2d(128, 128, 3, 1)
        self.upsample2= UpSampleConvBlock(in_channels=128, non_linear=self.nonlinear_downsampling)

        self.upconv1 = self.conv_block_2d(128 + 64, 64, 3, 1)
        self.upsample1= UpSampleConvBlock(in_channels=64, non_linear=self.nonlinear_downsampling)

        self.final = self.conv_block_2d_final(64 + 32, 2, 3, 1, mid=32)

    def __call__(self, x):
        # reshape to bertels shape
        x = rearrange(x, 'b c t h w -> b (c t) h w')
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)

        upconv2 = self.upconv2(conv3)  # 64 x 64
        upsample2 = self.upsample2(upconv2)  # 128 x 128
        concat2 = torch.cat([conv2, upsample2], 1)  # 128 x 128
        upconv1 = self.upconv1(concat2)  # 128 x 128
        upsample1 = self.upsample1(upconv1)  # 256 x 256
        concat1 = torch.cat([conv1, upsample1], 1)  # 256 x 256
        final = self.final(concat1)

        return final

    def conv_block_2d(self, in_channels, out_channels, kernel_size, padding):
        conv = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
        )
        return conv

    def conv_block_2d_final(self, in_channels, out_channels, kernel_size, padding, mid):
        conv = nn.Sequential(
            torch.nn.Conv2d(in_channels, mid, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(mid),
            torch.nn.ReLU(),
            torch.nn.Conv2d(mid, mid, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(mid),
            torch.nn.ReLU(),
            torch.nn.Conv2d(mid, out_channels, kernel_size=1, stride=1, padding=0),
        )
        return conv
