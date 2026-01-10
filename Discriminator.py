import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()

        def block(in_c, out_c, stride=1, bn=True):
            layers = [nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1)]
            if bn:
                layers.append(nn.BatchNorm2d(out_c)) # type: ignore
            layers.append(nn.LeakyReLU(0.2, inplace=True)) # type: ignore
            return layers

        self.model = nn.Sequential(
            *block(in_channels, base_channels, bn=False),
            *block(base_channels, base_channels, stride=2),

            *block(base_channels, base_channels*2),
            *block(base_channels*2, base_channels*2, stride=2),

            *block(base_channels*2, base_channels*4),
            *block(base_channels*4, base_channels*4, stride=2),

            *block(base_channels*4, base_channels*8),
            *block(base_channels*8, base_channels*8, stride=2),

            nn.Conv2d(base_channels*8, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.model(x)
    
d = Discriminator()