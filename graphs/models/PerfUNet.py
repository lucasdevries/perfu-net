import torch
import torch.nn as nn
from einops.einops import rearrange


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Sequential(nn.Conv3d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv3d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, timepoints, att_kernel):
        super(SpatialAttention, self).__init__()
        self.time_dim = timepoints // 2 + 1
        self.kernel_size = (self.time_dim, att_kernel, att_kernel)
        self.padding_size = (self.time_dim // 2, att_kernel // 2, att_kernel // 2)
        self.conv1 = nn.Conv3d(2, 1, self.kernel_size, padding=self.padding_size, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class AttentionModule(nn.Module):
    """AttentionModule.
    """

    def __init__(self, temp_att, chan_att, in_channels, timepoints, att_kernel):
        super().__init__()
        self.ca = ChannelAttention(in_channels) if chan_att else nn.Identity()
        self.sa = SpatialAttention(timepoints, att_kernel) if temp_att else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.mean = MeanModule(dim=2)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        x = self.mean(x)
        x = self.relu(x)
        return x


class ReduceTempModule(nn.Module):
    def __init__(self, channels, timepoints, att_kernel, groups):
        super(ReduceTempModule, self).__init__()
        self.time_dim = timepoints // 2 + 1
        self.num_blocks = timepoints // 2
        self.kernel_size = (self.time_dim, att_kernel, att_kernel)
        self.padding_size = (self.time_dim // 2, att_kernel // 2, att_kernel // 2)
        self.conv = []
        self.groups = channels if groups else 1
        for i in range(self.num_blocks):
            self.conv += [nn.Conv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0), bias=False, stride=(3, 1, 1), groups=self.groups),
                           nn.ReLU(inplace=True)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        x = self.conv(x)
        x = x.squeeze(dim=2)
        return x


class AttentionModule_test(nn.Module):
    """AttentionModule.
    """

    def __init__(self, temp_att, chan_att, in_channels, timepoints, att_kernel, groups):
        super().__init__()
        self.ca = ChannelAttention(in_channels) if chan_att else nn.Identity()
        self.sa = SpatialAttention(timepoints, att_kernel) if temp_att else nn.Identity()
        self.tr = ReduceTempModule(in_channels, timepoints, att_kernel, groups)
        self.relu = nn.ReLU(inplace=True)
        self.mean = MeanModule(dim=2)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        x = self.tr(x)
        return x

class AttentionModule_test2(nn.Module):
    """AttentionModule.
    """

    def __init__(self, temp_att, chan_att, in_channels, timepoints, att_kernel, groups):
        super().__init__()
        self.ca = ChannelAttention(in_channels) if chan_att else nn.Identity()
        self.sa = SpatialAttention(timepoints, att_kernel) if temp_att else nn.Identity()
        self.tr = ReduceTempModule(in_channels, timepoints, att_kernel, groups)
        self.relu = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.mean = MeanModule(dim=2)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        x = self.relu(x)
        x = self.tr(x)
        x = self.relu2(x)
        return x
class AttentionModule_test3(nn.Module):
    """AttentionModule.
    """

    def __init__(self, temp_att, chan_att, in_channels, timepoints, att_kernel, groups):
        super().__init__()
        self.ca = ChannelAttention(in_channels) if chan_att else nn.Identity()
        self.groupconv = nn.Conv3d(in_channels, in_channels, kernel_size=(timepoints, 1,1), padding=(0,0,0), groups=in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.relu(x)
        x = self.groupconv(x)
        x = self.relu(x)
        return x.squeeze(dim=2)
class MeanModule(nn.Module):
    """MeanModule.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.mean(x, dim=self.dim)


class MaxModule(nn.Module):
    """MaxModule.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        maxi, _ = torch.max(x, dim=self.dim)
        return maxi


class DownSampleConvBlock(nn.Module):
    """Depthwise_1DConv.
    """

    def __init__(self, in_channels, non_linear=True, stride=(2, 2, 2)):
        super().__init__()
        convlayer = nn.Conv3d(in_channels=in_channels,
                              out_channels=in_channels,
                              kernel_size=3,
                              padding=(1, 1, 1),
                              stride=stride,
                              groups=in_channels)
        if non_linear:
            bn = nn.BatchNorm3d(in_channels)
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


class UpSampleConvBlock3D(nn.Module):
    """Depthwise_1DConv.
    """

    def __init__(self, in_channels, non_linear=True, stride=(2, 2, 2)):
        super().__init__()
        convlayer = torch.nn.ConvTranspose3d(in_channels=in_channels,
                                             out_channels=in_channels,
                                             kernel_size=3,
                                             stride=stride,
                                             padding=1,
                                             output_padding=1)
        if non_linear:
            bn = nn.BatchNorm3d(in_channels)
            relu = nn.ReLU()
            self.convlayers = nn.Sequential(*[convlayer, bn, relu])
        else:
            self.convlayers = nn.Sequential(*[convlayer])

    def forward(self, x):
        x = self.convlayers(x)
        return x


class PerfUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.reduce = self.config.reduce
        self.dropout = self.config.dropout
        self.dropout_prob = self.config.dropout_prob
        self.att_kernel = self.config.att_kernel
        self.nonlinear_downsampling = self.config.nonlinear_downsampling
        self.in_channels = self.config.input_channels
        self.out_channels = 2
        self.att_kernel = self.config.att_kernel
        self.clip_length = self.config.clip_length

        self.channel_attention = self.config.channel_attention
        self.temporal_attention = self.config.temporal_attention

        self.conv1 = self.conv_block_3d(self.in_channels, 32, 3, 1)
        self.pool1 = DownSampleConvBlock(in_channels=32, non_linear=self.nonlinear_downsampling)

        self.conv2 = self.conv_block_3d(32, 64, 3, 1)
        self.pool2 = DownSampleConvBlock(in_channels=64, non_linear=self.nonlinear_downsampling)

        self.conv3 = self.conv_block_3d(64, 128, 3, 1, dropout=self.dropout, drop_prob=self.dropout_prob)
        self.pool3 = DownSampleConvBlock(in_channels=128, non_linear=self.nonlinear_downsampling)

        self.conv4 = self.conv_block_3d(128, 256, 3, 1, dropout=self.dropout, drop_prob=self.dropout_prob)

        if self.reduce == 'AttentionModule':
            self.reduce1 = AttentionModule(temp_att=self.temporal_attention, chan_att=self.channel_attention,
                                           in_channels=32, timepoints=self.clip_length, att_kernel=self.att_kernel)
            self.reduce2 = AttentionModule(temp_att=self.temporal_attention, chan_att=self.channel_attention,
                                           in_channels=64, timepoints=self.clip_length//2, att_kernel=self.att_kernel)
            self.reduce3 = AttentionModule(temp_att=self.temporal_attention, chan_att=self.channel_attention,
                                           in_channels=128, timepoints=self.clip_length//4, att_kernel=self.att_kernel)
        else:
            raise ValueError("Invalid Module type")

        self.upconv3 = self.conv_block_2d(256, 256, 3, 1)
        self.upsample3 = UpSampleConvBlock(in_channels=256, non_linear=self.nonlinear_downsampling)

        self.upconv2 = self.conv_block_2d(128, 128, 3, 1)
        self.upsample2 = UpSampleConvBlock(in_channels=128, non_linear=self.nonlinear_downsampling)

        self.upconv1 = self.conv_block_2d(128 + 64, 64, 3, 1)
        self.upsample1 = UpSampleConvBlock(in_channels=64, non_linear=self.nonlinear_downsampling)

        self.final = self.conv_block_2d_final(64 + 32, 2, 3, 1, mid=32)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def __call__(self, x):
        conv1 = self.conv1(x)
        red1 = self.reduce1(conv1)
        pool1 = self.pool1(self.relu1(conv1))

        conv2 = self.conv2(pool1)
        red2 = self.reduce2(conv2)
        pool2 = self.pool2(self.relu2(conv2))

        conv3 = self.conv3(pool2)
        red3 = self.reduce3(conv3)

        upconv2 = self.upconv2(red3)  # 64 x 64
        upsample2 = self.upsample2(upconv2)  # 128 x 128
        concat2 = torch.cat([red2, upsample2], 1)  # 128 x 128
        upconv1 = self.upconv1(concat2)  # 128 x 128
        upsample1 = self.upsample1(upconv1)  # 256 x 256
        concat1 = torch.cat([red1, upsample1], 1)  # 256 x 256
        final = self.final(concat1)

        return final

    def conv_block_3d(self, in_channels, out_channels, kernel_size, padding, dropout=False, drop_prob=0.2, groups=1):
        if dropout:
            conv = nn.Sequential(
                torch.nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding,
                                groups=groups),
                torch.nn.BatchNorm3d(out_channels),
                torch.nn.ReLU(),
                torch.nn.Dropout3d(p=drop_prob),
                torch.nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding,
                                groups=groups),
                torch.nn.BatchNorm3d(out_channels),
                torch.nn.ReLU(),
                torch.nn.Dropout3d(p=drop_prob),
            )
        else:
            conv = nn.Sequential(
                torch.nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding,
                                groups=groups),
                torch.nn.BatchNorm3d(out_channels),
                torch.nn.ReLU(),
                torch.nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding,
                                groups=groups),
                torch.nn.BatchNorm3d(out_channels),
                # torch.nn.ReLU(),
            )

        return conv

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
