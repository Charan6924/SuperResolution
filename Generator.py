import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=64, growth_channels=32, num_blocks=8, num_class=2, residual_scale=0.2):
        super(Generator, self).__init__()
        self.residual_scale = residual_scale
        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[RRDB(num_features, growth_channels, residual_scale) for _ in range(num_blocks)])
        self.body_tail = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2), 
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.upsample_refine = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.tail = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features // 2, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        final_conv = self.tail[-2]  
        nn.init.xavier_uniform_(final_conv.weight, gain=0.1)
        nn.init.zeros_(final_conv.bias)

    def forward(self, x):
        x = self.head(x)
        feat = x
        x = self.body(x)
        x = self.body_tail(x) + feat
        x = self.upsample(x) 
        x = self.upsample_refine(x)
        x = self.tail(x)
        x = x[:, :, :125, :125]     
        return x


class RRDB(nn.Module):
    def __init__(self, num_features=64, growth_channels=32, residual_scale=0.2):
        super().__init__()
        self.residual_scale = residual_scale
        self.rdb1 = ResidualBlock(num_features, growth_channels, residual_scale)
        self.rdb2 = ResidualBlock(num_features, growth_channels, residual_scale)
        self.rdb3 = ResidualBlock(num_features, growth_channels, residual_scale)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + self.residual_scale * out


class ResidualBlock(nn.Module):
    def __init__(self, num_features=64, growth_channels=32, residual_scale=0.2):
        super().__init__()
        self.residual_scale = residual_scale
        self.layers = nn.ModuleList()
        for i in range(5):
            in_ch = num_features + i * growth_channels
            self.layers.append(DenseLayer(in_ch, growth_channels))
        fusion_in = num_features + 5 * growth_channels
        self.fusion = nn.Conv2d(fusion_in, num_features, kernel_size=1)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        fused = self.fusion(torch.cat(features, dim=1))
        return x + self.residual_scale * fused


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_channels=32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, growth_channels, kernel_size=3, padding=1)
        self.act  = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))