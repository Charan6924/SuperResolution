import torch
import torch.nn as nn


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2),
                              kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention = self.avg_pool(x)
        attention = self.fc(attention)
        return x * attention


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU(num_parameters=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + 0.1 * residual


class Generator(nn.Module):
    def __init__(self, resblocks=16, channels=3, base_channels=64, use_attention=True):
        super(Generator, self).__init__()
        self.conv_input = nn.Sequential(
            nn.Conv2d(channels, base_channels, kernel_size=9, padding=4),
            nn.PReLU()
        )

        res_blocks = []
        for i in range(resblocks):
            res_blocks.append(ResBlock(base_channels))
            if use_attention and i in {7, 15}:  # only 2 attention blocks
                res_blocks.append(ChannelAttention(base_channels))

        self.resblocks = nn.Sequential(*res_blocks)

        self.conv_mid = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels)
        )

        self.upsample1 = UpsampleBlock(base_channels, base_channels)  # 64→128

        self.conv_output = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(base_channels, channels, kernel_size=9, padding=4)
        )

        self.final_resize = nn.Upsample(size=(125, 125), mode='bilinear', align_corners=False)

    def forward(self, x):
        x1 = self.conv_input(x)
        x2 = self.resblocks(x1)
        x3 = self.conv_mid(x2)
        x_mid = x1 + x3
        x_up = self.upsample1(x_mid)
        out = self.conv_output(x_up)
        out = self.final_resize(out)
        return out
