import torch
import torch.nn as nn
import torch.nn.functional as F

def spectral_conv(in_ch, out_ch, kernel_size, stride=1, padding=0):
    return nn.utils.spectral_norm(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
    )

class DiscBlock(nn.Module):
    def __init__(self, in_channels,out_channels, stride = 1):
        super(DiscBlock, self).__init__()
        self.block = nn.Sequential(
            spectral_conv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self,x):
        return self.block(x)

class Discriminator(nn.Module):
    def __init__(self,in_channels = 3, base_features = 64, num_classes = 2):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        f = base_features

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, f, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            DiscBlock(f,f,stride=2),   # 125 -> 63
            DiscBlock(f,f * 2,stride=1),
            DiscBlock(f * 2,f * 2,stride=2),   # 63  -> 32
            DiscBlock(f * 2,f * 4,stride=1),
            DiscBlock(f * 4,f * 4,stride=2),   # 32  -> 16
            DiscBlock(f * 4,f * 8,stride=1),
            DiscBlock(f * 8,f * 8,stride=2),   # 16  -> 8
        )

        self.head = nn.Sequential(
            spectral_conv(f*8,f*4,kernel_size = 3,padding = 1),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_conv(f*4,1,kernel_size = 3,padding = 1)
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')

    def forward(self,x):
        x = self.backbone(x)
        x = self.head(x)
        return x
    
class RelativisticAverageLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def discriminator_loss(self, real_logits, fake_logits):
        real_vs_fake = real_logits - fake_logits.mean()
        fake_vs_real = fake_logits - real_logits.mean()

        loss_real = self.bce(real_vs_fake, torch.ones_like(real_vs_fake))
        loss_fake = self.bce(fake_vs_real, torch.zeros_like(fake_vs_real))
        return (loss_real + loss_fake) / 2

    def generator_loss(self, real_logits, fake_logits):
        real_vs_fake = real_logits - fake_logits.mean()
        fake_vs_real = fake_logits - real_logits.mean()

        loss_real = self.bce(real_vs_fake, torch.zeros_like(real_vs_fake))
        loss_fake = self.bce(fake_vs_real, torch.ones_like(fake_vs_real))
        return (loss_real + loss_fake) / 2