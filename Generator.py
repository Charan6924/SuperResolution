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
            if use_attention and i in {7, 15}:  # only 2 attention
                res_blocks.append(ChannelAttention(base_channels))

        self.resblocks = nn.Sequential(*res_blocks)
        
        self.conv_mid = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels)
        )
        
        self.upsample1 = UpsampleBlock(base_channels, base_channels)  # 64→128
        
        # FIXED: No Tanh activation - let network learn the actual data range
        self.conv_output = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(base_channels, channels, kernel_size=9, padding=4)
            # Removed Tanh() - data is in [-0.24, 1.0], not [-1, 1]
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


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, use_spectral_norm=True):
        """
        Improved discriminator with:
        - Spectral normalization for training stability
        - Global average pooling for scalar output
        - Proper output size for PatchGAN-style discrimination
        
        Args:
            in_channels: Number of input channels (3 for RGB)
            base_channels: Base number of feature channels
            use_spectral_norm: Whether to use spectral normalization
        """
        super().__init__()

        def block(in_c, out_c, stride=1, bn=True):
            layers = []
            conv = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1)
            
            # Apply spectral normalization if enabled
            if use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            
            layers.append(conv)
            
            if bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.features = nn.Sequential(
            *block(in_channels, base_channels, bn=False),
            *block(base_channels, base_channels, stride=2),

            *block(base_channels, base_channels*2),
            *block(base_channels*2, base_channels*2, stride=2),

            *block(base_channels*2, base_channels*4),
            *block(base_channels*4, base_channels*4, stride=2),

            *block(base_channels*4, base_channels*8),
            *block(base_channels*8, base_channels*8, stride=2),
        )
        
        # FIXED: Add proper output head
        # After 4 stride-2 ops: 125 → 62 → 31 → 15 → 7
        # Output is [batch, 512, 7, 7]
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [batch, 512, 7, 7] → [batch, 512, 1, 1]
            nn.Flatten(),              # [batch, 512, 1, 1] → [batch, 512]
            nn.Linear(base_channels*8, 1)  # [batch, 512] → [batch, 1]
        )

    def forward(self, x):
        features = self.features(x)
        out = self.classifier(features)
        return out


# Test to verify output shapes
if __name__ == "__main__":
    print("Testing Generator...")
    g = Generator()
    lr_input = torch.randn(2, 3, 64, 64)  # Batch of 2, 64x64 LR images
    sr_output = g(lr_input)
    print(f"  Input shape:  {lr_input.shape}")
    print(f"  Output shape: {sr_output.shape}")
    print(f"  Expected:     torch.Size([2, 3, 125, 125])")
    assert sr_output.shape == torch.Size([2, 3, 125, 125]), "Generator output shape mismatch!"
    print("  ✓ Generator shape correct!\n")
    
    print("Testing Discriminator...")
    d = Discriminator()
    hr_input = torch.randn(2, 3, 125, 125)  # Batch of 2, 125x125 HR images
    d_output = d(hr_input)
    print(f"  Input shape:  {hr_input.shape}")
    print(f"  Output shape: {d_output.shape}")
    print(f"  Expected:     torch.Size([2, 1])")
    assert d_output.shape == torch.Size([2, 1]), "Discriminator output shape mismatch!"
    print("  ✓ Discriminator shape correct!\n")
    
    # Test data range
    print("Testing output range...")
    with torch.no_grad():
        test_output = g(torch.randn(1, 3, 64, 64))
        print(f"  Generator output range: [{test_output.min():.3f}, {test_output.max():.3f}]")
        print(f"  (Should be unbounded - will be clamped in training)")