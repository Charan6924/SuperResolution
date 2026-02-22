import torch
import torch.nn as nn


class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, base_features=64):
        super().__init__()
        f = base_features

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, f, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            DiscBlock(f, f, stride=2),        # 125 -> 63
            DiscBlock(f, f * 2, stride=1),
            DiscBlock(f * 2, f * 2, stride=2),  # 63 -> 32
            DiscBlock(f * 2, f * 4, stride=1),
            DiscBlock(f * 4, f * 4, stride=2),  # 32 -> 16
            DiscBlock(f * 4, f * 8, stride=1),
            DiscBlock(f * 8, f * 8, stride=2),  # 16 -> 8
        )

        self.head = nn.Sequential(
            nn.Conv2d(f * 8, f * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(f * 4, 1, kernel_size=3, padding=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


class RelativisticAverageLoss(nn.Module):
    def __init__(self, real_label=0.9, fake_label=0.1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.real_label = real_label
        self.fake_label = fake_label

    def discriminator_loss(self, real_logits, fake_logits):
        real_vs_fake = real_logits - fake_logits.mean()
        fake_vs_real = fake_logits - real_logits.mean()

        real_target = self.real_label * torch.ones_like(real_vs_fake)
        fake_target = self.fake_label * torch.ones_like(fake_vs_real)

        loss_real = self.bce(real_vs_fake, real_target)
        loss_fake = self.bce(fake_vs_real, fake_target)
        return (loss_real + loss_fake) / 2

    def generator_loss(self, real_logits, fake_logits):
        real_vs_fake = real_logits - fake_logits.mean()
        fake_vs_real = fake_logits - real_logits.mean()

        loss_real = self.bce(real_vs_fake, torch.zeros_like(real_vs_fake))
        loss_fake = self.bce(fake_vs_real, torch.ones_like(fake_vs_real))
        return (loss_real + loss_fake) / 2
