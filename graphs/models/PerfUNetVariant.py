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
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
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

    def __init__(self, in_channels):
        super().__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        x = self.relu(x)
        return x


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


# class MeanMaxAttentionA(nn.Module):
#     def __init__(self, kernel_size=3, in_channels=32):
#         super(MeanMaxAttentionA, self).__init__()
#         self.in_channels = in_channels
#         self.conv1 = nn.Conv3d(2 * self.in_channels, self.in_channels, kernel_size, padding=kernel_size // 2,
#                                bias=False)
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=2, keepdim=True)
#         max_out, _ = torch.max(x, dim=2, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return x.squeeze(dim=2)
class MeanMaxAttentionA(nn.Module):
    def __init__(self, kernel_size=3, in_channels=32):
        super(MeanMaxAttentionA, self).__init__()
        self.in_channels = in_channels
        self.chan_att = ChannelAttention(self.in_channels)
        self.conv1 = nn.Conv3d(2 * self.in_channels, self.in_channels, kernel_size, padding=kernel_size // 2,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.chan_att(x) * x
        avg_out = torch.mean(x, dim=2, keepdim=True)
        max_out, _ = torch.max(x, dim=2, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.relu(x)

        return x.squeeze(dim=2)
class ConvModule(nn.Module):
    def __init__(self, in_timepoints, kernel_size=3, in_channels=32):
        super(ConvModule, self).__init__()
        self.in_channels = in_channels
        self.in_timepoints = in_timepoints
        self.conv1 = nn.Conv2d(self.in_timepoints * self.in_channels, self.in_channels, kernel_size, padding=kernel_size // 2,
                               bias=False)

    def forward(self, x):
        x = rearrange(x, 'b c t h w -> b (c t) h w')
        x = self.conv1(x)
        return x

class MeanMaxAttentionB(nn.Module):
    def __init__(self, in_timepoints):
        super(MeanMaxAttentionB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Sequential(nn.Conv3d(in_timepoints, in_timepoints // 2, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv3d(in_timepoints // 2, in_timepoints, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = rearrange(x, 'b c t h w -> b t c h w')
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out

        out = out.squeeze(dim=4).squeeze(dim=3).squeeze(dim=2)
        out = self.sigmoid(out)
        result = torch.einsum('btchw, bt->bchw', x, out)
        return result


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

class PerfUNetVariant(nn.Module):
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

        self.conv1 = self.conv_block_3d(self.in_channels, 32, 3, 1)
        self.pool1 = DownSampleConvBlock(in_channels=32, non_linear=self.nonlinear_downsampling)

        self.conv2 = self.conv_block_3d(32, 64, 3, 1)
        self.pool2 = DownSampleConvBlock(in_channels=64, non_linear=self.nonlinear_downsampling)

        self.conv3 = self.conv_block_3d(64, 128, 3, 1, dropout=self.dropout, drop_prob=self.dropout_prob)
        self.pool3 = DownSampleConvBlock(in_channels=128, non_linear=self.nonlinear_downsampling)

        self.conv4 = self.conv_block_3d(128, 256, 3, 1, dropout=self.dropout, drop_prob=self.dropout_prob)

        if self.reduce == 'MeanModule':
            self.reduce1 = nn.Sequential(MeanModule(dim=2))
            self.reduce2 = nn.Sequential(MeanModule(dim=2))
            self.reduce3 = nn.Sequential(MeanModule(dim=2))
            self.reduce4 = nn.Sequential(MeanModule(dim=2))
        elif self.reduce == 'MaxModule':
            self.reduce1 = nn.Sequential(MaxModule(dim=2))
            self.reduce2 = nn.Sequential(MaxModule(dim=2))
            self.reduce3 = nn.Sequential(MaxModule(dim=2))
            self.reduce4 = nn.Sequential(MaxModule(dim=2))
        elif self.reduce == 'MeanMaxAttentionA':
            self.reduce1 = nn.Sequential(MeanMaxAttentionA(kernel_size=self.att_kernel, in_channels=32))
            self.reduce2 = nn.Sequential(MeanMaxAttentionA(kernel_size=self.att_kernel, in_channels=64))
            self.reduce3 = nn.Sequential(MeanMaxAttentionA(kernel_size=self.att_kernel, in_channels=128))
            self.reduce4 = nn.Sequential(MeanMaxAttentionA(kernel_size=self.att_kernel, in_channels=256))
        elif self.reduce == 'MeanMaxAttentionB':
            self.reduce1 = nn.Sequential(MeanMaxAttentionB(in_timepoints=16))
            self.reduce2 = nn.Sequential(MeanMaxAttentionB(in_timepoints=8))
            self.reduce3 = nn.Sequential(MeanMaxAttentionB(in_timepoints=4))
            self.reduce4 = nn.Sequential(MeanMaxAttentionB(in_timepoints=2))
        elif self.reduce == 'ConvModuleB':
            self.reduce1 = nn.Sequential(ConvModule(in_timepoints=16, in_channels=32))
            self.reduce2 = nn.Sequential(ConvModule(in_timepoints=8, in_channels=64))
            self.reduce3 = nn.Sequential(ConvModule(in_timepoints=4, in_channels=128))
            self.reduce4 = nn.Sequential(ConvModule(in_timepoints=2, in_channels=256))
        else:
            raise ValueError("Invalid Module type")

        self.upconv3 = self.conv_block_2d(256, 256, 3, 1)
        self.upsample3= UpSampleConvBlock(in_channels=256, non_linear=self.nonlinear_downsampling)

        self.upconv2 = self.conv_block_2d(128, 128, 3, 1)
        self.upsample2= UpSampleConvBlock(in_channels=128, non_linear=self.nonlinear_downsampling)

        self.upconv1 = self.conv_block_2d(128 + 64, 64, 3, 1)
        self.upsample1= UpSampleConvBlock(in_channels=64, non_linear=self.nonlinear_downsampling)

        self.final = self.conv_block_2d_final(64 + 32, 2, 3, 1, mid=32)

    def __call__(self, x):
        conv1 = self.conv1(x)
        red1 = self.reduce1(conv1)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        red2 = self.reduce2(conv2)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        red3 = self.reduce3(conv3)
        # pool3 = self.pool3(conv3)
        #
        # conv4 = self.conv4(pool3)
        # red4 = self.reduce4(conv4)
        #
        # upconv3 = self.upconv3(red4)  # 32 x 32
        # upsample3 = self.upsample3(upconv3)  # 64 x 64
        # concat3 = torch.cat([red3, upsample3], 1)  # 64 x 64

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
                torch.nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=groups),
                torch.nn.BatchNorm3d(out_channels),
                torch.nn.ReLU(),
                torch.nn.Dropout3d(p=drop_prob),
                torch.nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=groups),
                torch.nn.BatchNorm3d(out_channels),
                torch.nn.ReLU(),
                torch.nn.Dropout3d(p=drop_prob),
            )
        else:
            conv = nn.Sequential(
                torch.nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=groups),
                torch.nn.BatchNorm3d(out_channels),
                torch.nn.ReLU(),
                torch.nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=groups),
                torch.nn.BatchNorm3d(out_channels),
                torch.nn.ReLU(),
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


class C3DU_reducer_dropout_smaller(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.reduce = self.config.reduce
        self.dropout = self.config.dropout
        self.dropout_prob = self.config.dropout_prob
        self.att_kernel = self.config.att_kernel
        self.nonlinear_downsampling = self.config.nonlinear_downsampling
        self.in_channels = 2
        self.out_channels = 2

        self.conv1 = self.conv_block_3d(self.in_channels, 32, 3, 1)
        self.pool1 = DownSampleConvBlock(in_channels=32, non_linear=self.nonlinear_downsampling)

        self.conv2 = self.conv_block_3d(32, 64, 3, 1)
        self.pool2 = DownSampleConvBlock(in_channels=64, non_linear=self.nonlinear_downsampling)

        self.conv3 = self.conv_block_3d(64, 128, 3, 1, dropout=self.dropout, drop_prob=self.dropout_prob)
        self.pool3 = DownSampleConvBlock(in_channels=128, non_linear=self.nonlinear_downsampling)

        self.conv4 = self.conv_block_3d(128, 256, 3, 1, dropout=self.dropout, drop_prob=self.dropout_prob)

        if self.reduce == 'MeanModule':
            self.reduce1 = nn.Sequential(MeanModule(dim=2))
            self.reduce2 = nn.Sequential(MeanModule(dim=2))
            self.reduce3 = nn.Sequential(MeanModule(dim=2))
            self.reduce4 = nn.Sequential(MeanModule(dim=2))
        elif self.reduce == 'MaxModule':
            self.reduce1 = nn.Sequential(MaxModule(dim=2))
            self.reduce2 = nn.Sequential(MaxModule(dim=2))
            self.reduce3 = nn.Sequential(MaxModule(dim=2))
            self.reduce4 = nn.Sequential(MaxModule(dim=2))
        elif self.reduce == 'MeanMaxAttentionA':
            self.reduce1 = nn.Sequential(MeanMaxAttentionA(kernel_size=self.att_kernel, in_channels=32))
            self.reduce2 = nn.Sequential(MeanMaxAttentionA(kernel_size=self.att_kernel, in_channels=64))
            self.reduce3 = nn.Sequential(MeanMaxAttentionA(kernel_size=self.att_kernel, in_channels=128))
            self.reduce4 = nn.Sequential(MeanMaxAttentionA(kernel_size=self.att_kernel, in_channels=256))
        elif self.reduce == 'MeanMaxAttentionB':
            self.reduce1 = nn.Sequential(MeanMaxAttentionB(in_timepoints=16))
            self.reduce2 = nn.Sequential(MeanMaxAttentionB(in_timepoints=8))
            self.reduce3 = nn.Sequential(MeanMaxAttentionB(in_timepoints=4))
            self.reduce4 = nn.Sequential(MeanMaxAttentionB(in_timepoints=2))
        elif self.reduce == 'ConvModuleB':
            self.reduce1 = nn.Sequential(ConvModule(in_timepoints=16, in_channels=32))
            self.reduce2 = nn.Sequential(ConvModule(in_timepoints=8, in_channels=64))
            self.reduce3 = nn.Sequential(ConvModule(in_timepoints=4, in_channels=128))
            self.reduce4 = nn.Sequential(ConvModule(in_timepoints=2, in_channels=256))
        else:
            raise ValueError("Invalid Module type")

        self.upconv3 = self.conv_block_2d(256, 256, 3, 1)
        self.upsample3 = UpSampleConvBlock(in_channels=256, non_linear=self.nonlinear_downsampling)

        self.upconv2 = self.conv_block_2d(128, 128, 3, 1)
        self.upsample2 = UpSampleConvBlock(in_channels=128, non_linear=self.nonlinear_downsampling)

        self.upconv1 = self.conv_block_2d(64, 64, 3, 1)
        self.upsample1 = UpSampleConvBlock(in_channels=64, non_linear=self.nonlinear_downsampling)

        self.final = self.conv_block_2d_final(64 + 32, 2, 3, 1, mid=32)

    def __call__(self, x):
        conv1 = self.conv1(x)
        red1 = self.reduce1(conv1)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        red2 = self.reduce2(conv2)
        # pool2 = self.pool2(conv2)
        #
        # conv3 = self.conv3(pool2)
        # red3 = self.reduce3(conv3)
        # pool3 = self.pool3(conv3)
        #
        # conv4 = self.conv4(pool3)
        # red4 = self.reduce4(conv4)
        #
        # upconv3 = self.upconv3(red4)  # 32 x 32
        # upsample3 = self.upsample3(upconv3)  # 64 x 64
        # concat3 = torch.cat([red3, upsample3], 1)  # 64 x 64

        # upconv2 = self.upconv2(red3)  # 64 x 64
        # upsample2 = self.upsample2(upconv2)  # 128 x 128
        # concat2 = torch.cat([red2, upsample2], 1)  # 128 x 128
        upconv1 = self.upconv1(red2)  # 128 x 128
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
                torch.nn.ReLU(),
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



class C3DU_reducer_dropout_3D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.reduce = self.config.reduce
        self.dropout = self.config.dropout
        self.dropout_prob = self.config.dropout_prob
        self.att_kernel = self.config.att_kernel
        self.nonlinear_downsampling = self.config.nonlinear_downsampling
        self.in_channels = 2
        self.out_channels = 2

        self.conv1 = self.conv_block_3d(self.in_channels, 32, 3, 1)
        self.pool1 = DownSampleConvBlock(in_channels=32, non_linear=self.nonlinear_downsampling)

        self.conv2 = self.conv_block_3d(32, 64, 3, 1)
        self.pool2 = DownSampleConvBlock(in_channels=64, non_linear=self.nonlinear_downsampling)

        self.conv3 = self.conv_block_3d(64, 128, 3, 1, dropout=self.dropout, drop_prob=self.dropout_prob)
        self.pool3 = DownSampleConvBlock(in_channels=128, non_linear=self.nonlinear_downsampling)

        self.conv4 = self.conv_block_3d(128, 256, 3, 1, dropout=self.dropout, drop_prob=self.dropout_prob)

        if self.reduce == 'MeanModule':
            self.reduce1 = MeanModule(dim=2)
        elif self.reduce == 'MaxModule':
            self.reduce1 = MaxModule(dim=2)
        else:
            raise ValueError("Invalid Module type")

        self.upconv3 = self.conv_block_3d(256, 256, 3, 1)
        self.upsample3= UpSampleConvBlock3D(in_channels=256, non_linear=self.nonlinear_downsampling)

        self.upconv2 = self.conv_block_3d(256 + 128, 128, 3, 1)
        self.upsample2= UpSampleConvBlock3D(in_channels=128, non_linear=self.nonlinear_downsampling)

        self.upconv1 = self.conv_block_3d(128 + 64, 64, 3, 1)
        self.upsample1= UpSampleConvBlock3D(in_channels=64, non_linear=self.nonlinear_downsampling)

        self.final = self.conv_block_2d_final(64 + 32, 2, 3, 1, mid=32, module=self.reduce1)

    def __call__(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)

        upconv3 = self.upconv3(conv4)  # 32 x 32
        upsample3 = self.upsample3(upconv3)  # 64 x 64
        concat3 = torch.cat([conv3, upsample3], 1)  # 64 x 64
        upconv2 = self.upconv2(concat3)  # 64 x 64
        upsample2 = self.upsample2(upconv2)  # 128 x 128
        concat2 = torch.cat([conv2, upsample2], 1)  # 128 x 128
        upconv1 = self.upconv1(concat2)  # 128 x 128
        upsample1 = self.upsample1(upconv1)  # 256 x 256

        concat1 = torch.cat([conv1, upsample1], 1)  # 256 x 256
        final = self.final(concat1)

        return final

    def conv_block_3d(self, in_channels, out_channels, kernel_size, padding, dropout=False, drop_prob=0.2,
                      groups=1):
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
                torch.nn.ReLU(),
            )

        return conv

    def conv_block_2d_final(self, in_channels, out_channels, kernel_size, padding, mid, module):
        conv = nn.Sequential(
                torch.nn.Conv3d(in_channels, mid, kernel_size=kernel_size, stride=1, padding=padding),
                torch.nn.BatchNorm3d(mid),
                torch.nn.ReLU(),
                module,
                torch.nn.Conv2d(mid, mid, kernel_size=kernel_size, stride=1, padding=padding),
                torch.nn.BatchNorm2d(mid),
                torch.nn.ReLU(),
                torch.nn.Conv2d(mid, out_channels, kernel_size=1, stride=1, padding=0),
        )
        return conv