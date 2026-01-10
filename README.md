# Super-Resolution GAN for Image Upscaling

A PyTorch implementation of SRGAN (Super-Resolution Generative Adversarial Network) for upscaling low-resolution images from 64×64 to 125×125 pixels with enhanced perceptual quality.

## Overview

This project implements a deep learning approach to single-image super-resolution using:
- **Generator**: 16-block residual network with channel attention mechanisms
- **Discriminator**: PatchGAN discriminator for realistic texture generation
- **Two-stage training**: MSE pre-training followed by adversarial fine-tuning
- **Dataset**: 111,000+ training images for robust learning

## Key Features

- Residual blocks with batch normalization for stable training
- Channel attention modules for adaptive feature refinement
- Comprehensive metrics: MSE, PSNR, SSIM
- PyTorch 2.0 compatible with torch.compile() optimization
- Early stopping to prevent overfitting
- Checkpoint saving for best model preservation

## Architecture

**Generator**
- Initial 9×9 convolution layer
- 16 residual blocks with PReLU activation
- Channel attention at blocks 8 and 16
- Pixel shuffle upsampling (2× scale)
- Bilinear interpolation for final resize to 125×125

**Discriminator**
- VGG-style architecture with strided convolutions
- LeakyReLU activation and batch normalization
- PatchGAN output for local realism assessment

## Training Strategy

1. **Phase 1**: Generator pre-training with MSE loss for pixel-perfect baseline
2. **Phase 2**: GAN training with combined adversarial and perceptual loss
   - Generator learning rate: 1e-5
   - Discriminator learning rate: 1e-4
   - Early stopping with patience of 8 epochs

## Requirements
```
torch>=2.0.0
torchvision
pillow
scikit-image
tqdm
numpy
```

## Usage
```python
# Load pretrained model
generator = Generator().to(device)
checkpoint = torch.load('best_generator_gan.pt')
generator.load_state_dict(checkpoint['generator'])

# Upscale image
with torch.no_grad():
    hr_image = generator(lr_image)
```

