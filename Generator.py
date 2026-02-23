import torch.nn as nn


class UpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, scale=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * scale * scale, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.shuffle(self.conv(x)))


class ChannelAttention(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, ch // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // reduction, ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.PReLU(ch),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch)
        )

    def forward(self, x):
        return x + 0.1 * self.block(x)


class Generator(nn.Module):
    def __init__(self, num_blocks=16, in_ch=3, base_ch=64):
        super().__init__()

        self.input = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 9, padding=4),
            nn.PReLU()
        )

        blocks = []
        for i in range(num_blocks):
            blocks.append(ResBlock(base_ch))
            if i in {7, 15}:
                blocks.append(ChannelAttention(base_ch))
        self.blocks = nn.Sequential(*blocks)

        self.mid = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch)
        )

        self.upsample = UpsampleBlock(base_ch, base_ch)

        self.output = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(base_ch, in_ch, 9, padding=4)
        )

        self.resize = nn.Upsample(size=(125, 125), mode='bilinear', align_corners=False)

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.blocks(x1)
        x3 = self.mid(x2)
        out = self.upsample(x1 + x3)
        out = self.output(out)
        return self.resize(out)
