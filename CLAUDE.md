# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyTorch SRGAN implementation for upscaling jet particle images from 64×64 to 125×125 pixels. Uses two-phase training: pre-training with pixel loss, then adversarial fine-tuning.

## Architecture

- **Generator**: 16-block residual network with channel attention at blocks 8 and 16, PixelShuffle upsampling (~1.4M params)
- **Discriminator**: PatchGAN with relativistic average loss (~5.9M params)
- **Training**: Mixed precision (bfloat16), gradient clipping (max_norm=10.0)

## Running Training

```bash
uv run TrainingLoop.py          # Local GPU
sbatch train.sh                 # SLURM cluster (H100)
```

Key hyperparameters in `TrainingLoop.py`:
- Pre-train: 50 epochs, batch 256, lr_g=1e-4
- GAN: 100 epochs, batch 256, lr_g=1e-4, lr_d=1e-5
- Loss weights: 0.0001×adversarial + 1×pixel + 0.1×SSIM

## Code Organization

- `TrainingLoop.py` - Main training orchestration (pre-train and GAN phases)
- `Generator.py` - Generator with ResBlocks, ChannelAttention, UpsampleBlock
- `Discriminator.py` - PatchGAN + RelativisticAverageLoss class
- `JetImageDataset.py` - IterableDataset for streaming .pt tensor files
- `utils.py` - PSNR/SSIM metric implementations
- `convert_to_pt.py` - Preprocessing script (parquet → .pt tensors)

## Data Pipeline

- Source: Parquet files with X_jets_LR (64×64) and X_jets (125×125)
- Preprocessing: `convert_to_pt.py` chunks into .pt files (500 samples each)
- Normalization: Clamp to [0,1] using 99.5th percentile (stats in `normalization_stats.pt`)
- Dataset splits: train 80%, val 10%, test 10%

## Training Phases

1. **Pre-training** (50 epochs): L1 + 0.1×SSIM loss, cosine annealing scheduler
2. **GAN Phase** (100 epochs):
   - Train discriminator first (generator frozen)
   - Train generator with gradient clipping
   - Both use mixed precision autocast

## Checkpoints

- `checkpoints/pretrain_final.pt` - Pre-training result
- `checkpoints/best_model.pt` - Best SSIM during GAN phase
- `checkpoints/epoch_XXX.pt` - Every 10 epochs (includes optimizer states)

## Monitoring

Weights & Biases integration logs loss components (d_loss, g_loss, g_adv, g_pix, g_ssim), PSNR, SSIM, and sample images.
