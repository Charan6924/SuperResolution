import torch
import torch.nn as nn


class DiscBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module):
    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()
        f = base_ch

        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, f, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            DiscBlock(f, f, stride=2),
            DiscBlock(f, f * 2),
            DiscBlock(f * 2, f * 2, stride=2),
            DiscBlock(f * 2, f * 4),
            DiscBlock(f * 4, f * 4, stride=2),
            DiscBlock(f * 4, f * 8),
            DiscBlock(f * 8, f * 8, stride=2),
        )

        self.head = nn.Sequential(
            nn.Conv2d(f * 8, f * 4, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(f * 4, 1, 3, padding=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.backbone(x))


class RelativisticAverageLoss(nn.Module):
    def __init__(self, real_label=0.9, fake_label=0.1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.real_label = real_label
        self.fake_label = fake_label

    def discriminator_loss(self, real, fake):
        r_vs_f = real - fake.mean()
        f_vs_r = fake - real.mean()
        loss_r = self.bce(r_vs_f, self.real_label * torch.ones_like(r_vs_f))
        loss_f = self.bce(f_vs_r, self.fake_label * torch.ones_like(f_vs_r))
        return (loss_r + loss_f) / 2

    def generator_loss(self, real, fake):
        r_vs_f = real - fake.mean()
        f_vs_r = fake - real.mean()
        loss_r = self.bce(r_vs_f, torch.zeros_like(r_vs_f))
        loss_f = self.bce(f_vs_r, torch.ones_like(f_vs_r))
        return (loss_r + loss_f) / 2
