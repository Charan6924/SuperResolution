import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import get_worker_info
import torch.optim 
import torch.nn.functional as F
from utils import validate, train_generator, train_discriminator, set_epoch, get_current_adv_weight, should_train_discriminator
from Generator import Generator
from Discriminator import Discriminator
from JetImageDataset import JetImageDataset
from EarlyStopping import EarlyStopping
from tqdm import tqdm
from torch.amp import autocast, GradScaler #type: ignore
from torch.nn.utils import clip_grad_norm_
import os

if 'PFSDIR' in os.environ:
    tensor_dir = os.path.join(os.environ['PFSDIR'], 'tensors')
    print(f"Running in SLURM job - using: {tensor_dir}")
else:
    tensor_dir = '/tmp/tensor_data'
    print(f"Running locally - using: {tensor_dir}")


stats_path = os.path.join(tensor_dir, 'normalization_stats.pt')
if not os.path.exists(stats_path):
    raise FileNotFoundError(
        f"Normalization stats not found at {stats_path}\n"
        f"Run calculate_and_save_normalization_stats() first!"
    )

stats = torch.load(stats_path)
print("\n" + "="*60)
print("Loaded Normalization Statistics:")
print(f"  LR: min={stats['lr_min']:.4f}, max={stats['lr_max']:.4f}")
print(f"  HR: min={stats['hr_min']:.4f}, max={stats['hr_max']:.4f}")
print(f"  LR 99.5th percentile: {stats['lr_p995']:.4f} (using for normalization)")
print(f"  HR 99.5th percentile: {stats['hr_p995']:.4f} (using for normalization)")
print("="*60 + "\n")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True  
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

generator = Generator().to(device)
discriminator = Discriminator().to(device)

pixel_criterion = nn.MSELoss()
adv_criterion = nn.MSELoss()

g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.9, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-5, betas=(0.9, 0.999))  
scaler = GradScaler('cuda')

pretrain_epochs = 10
epochs = 100
os.makedirs('checkpoints', exist_ok=True)

resume_checkpoint = ""  
start_epoch = 0
best_val_mse = float("inf")
best_val_ssim = 0.0 
g_lossi = []
d_lossi = []
val_mse, val_psnr, val_ssim = [], [], []
adv_weights = []

torch.set_float32_matmul_precision('high')

if resume_checkpoint and os.path.exists(resume_checkpoint):
    print(f"Loading checkpoint from {resume_checkpoint}...")
    checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    g_optimizer.load_state_dict(checkpoint['g_optimizer'])
    d_optimizer.load_state_dict(checkpoint['d_optimizer'])
    start_epoch = checkpoint['epoch'] + 1  
    best_val_mse = checkpoint['val_mse']
    best_val_ssim = checkpoint.get('val_ssim', 0.0)  
    
    print(f"✓ Resumed from epoch {checkpoint['epoch']}")
    print(f"  Previous Val MSE: {checkpoint['val_mse']:.6f}")
    print(f"  Previous PSNR: {checkpoint['val_psnr']:.2f} dB")
    print(f"  Previous SSIM: {checkpoint['val_ssim']:.4f}")
else:
    print("No checkpoint found, starting from scratch\n")

early_stopper = EarlyStopping(patience=15, min_delta=1e-4)

print(f'Created models and loaded on {device}')

train_dataset = JetImageDataset(
    tensor_dir=tensor_dir,
    split='train',
    train_ratio=0.8,
    normalize=True,
    seed=42,
    max_batch_size=256,
    lr_max=stats['lr_p995'],  
    hr_max=stats['hr_p995']   
)

val_dataset = JetImageDataset(
    tensor_dir=tensor_dir,
    split='val',
    train_ratio=0.8,
    normalize=True,
    seed=42,
    max_batch_size=256,
    lr_max=stats['lr_p995'], 
    hr_max=stats['hr_p995']
)

train_loader = DataLoader(
    train_dataset,
    batch_size=None,
    num_workers=4, 
    pin_memory=True,
    prefetch_factor=2  
)

val_loader = DataLoader(
    val_dataset,
    batch_size=None,
    num_workers=2,  
    pin_memory=True,
    prefetch_factor=2
)
print('Created data loaders\n')

print("Checking actual data range after normalization...")
sample_loader = DataLoader(train_dataset, batch_size=None, num_workers=0)
for lr, hr in sample_loader:
    print(f"  LR: min={lr.min():.4f}, max={lr.max():.4f}, mean={lr.mean():.4f}")
    print(f"  HR: min={hr.min():.4f}, max={hr.max():.4f}, mean={hr.mean():.4f}")
    
    # Sanity check
    if lr.max() < 0.5:
        print("\n⚠️  WARNING: Data range looks too small!")
        print("   Expected range: approximately [-0.2, 1.0]")
        print("   Check normalization statistics calculation")
    elif lr.min() > 0 and hr.min() > 0:
        print("\n✓ Data range looks good (approximately [-0.2, 1.0])")
    break
print()

ssim_history = []
ssim_warning_threshold = 0.3  

for epoch in range(start_epoch, epochs):
    set_epoch(epoch)
    
    generator.train()
    discriminator.train()
    g_epoch_loss = 0.0
    d_epoch_loss = 0.0
    train_count = 0 

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for batch_idx, (lr, hr) in enumerate(pbar): 
        lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)

        if torch.isnan(lr).any() or torch.isnan(hr).any():
            print(f"Corrupt data in batch {batch_idx} - SKIPPING")
            continue

        if lr.min() < -1.0 or lr.max() > 2.0 or hr.min() < -1.0 or hr.max() > 2.0:
            print(f"Data out of expected range! lr: [{lr.min():.2f}, {lr.max():.2f}], hr: [{hr.min():.2f}, {hr.max():.2f}]")
            continue
        
        train_count += 1

        if epoch < pretrain_epochs:
            g_optimizer.zero_grad()
            
            with autocast('cuda', dtype=torch.bfloat16):
                fake_hr = generator(lr)
                loss = pixel_criterion(fake_hr, hr)
            
            scaler.scale(loss).backward()
            scaler.unscale_(g_optimizer)
            clip_grad_norm_(generator.parameters(), max_norm=1.0)
            scaler.step(g_optimizer)
            scaler.update()
            
            g_epoch_loss += loss.item()
            pbar.set_postfix({"Mode": "Pre-train", "MSE": f"{loss.item():.4f}"})
        else:
            with autocast('cuda', dtype=torch.bfloat16):
                fake_hr = generator(lr)
            
            train_d = should_train_discriminator(batch_idx, epoch, pretrain_epochs)
            d_loss = train_discriminator(discriminator, d_optimizer, hr, fake_hr, scaler, train_d)
            g_loss, pixel_loss, adv_loss = train_generator(generator, g_optimizer, discriminator, lr, hr, fake_hr, scaler)
            
            g_epoch_loss += g_loss
            d_epoch_loss += d_loss
            
            lambda_adv = get_current_adv_weight()
            pbar.set_postfix({
                "G": f"{g_loss:.4f}", 
                "D": f"{d_loss:.4f}", 
                "λ_adv": f"{lambda_adv:.6f}",
                "adv_loss": f"{adv_loss:.4f}"
            })

    if train_count > 0:
        avg_g_loss = g_epoch_loss / train_count
        avg_d_loss = d_epoch_loss / train_count if epoch >= pretrain_epochs else 0.0
    else:
        print(f"Epoch {epoch+1}: SKIPPING (No valid training data)")
        continue
    
    with torch.no_grad():
        metrics = validate(generator, val_loader)
    
    current_adv_weight = get_current_adv_weight()
    adv_weights.append(current_adv_weight)
    current_ssim = metrics["ssim"]
    ssim_history.append(current_ssim)
    
    # Monitor SSIM degradation
    if epoch >= pretrain_epochs:
        if len(ssim_history) > 1:
            ssim_drop = ssim_history[-2] - current_ssim
            if ssim_drop > 0.05:
                print(f"SSIM dropped {ssim_drop:.4f} this epoch!")
        
        if current_ssim < ssim_warning_threshold:
            print(f"WARNING: SSIM at {current_ssim:.4f} (threshold: {ssim_warning_threshold})")

    epoch_checkpoint = {
        "epoch": epoch,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "g_optimizer": g_optimizer.state_dict(),
        "d_optimizer": d_optimizer.state_dict(),
        "val_mse": metrics["mse"],
        "val_psnr": metrics["psnr"],
        "val_ssim": metrics["ssim"],
        "g_loss": avg_g_loss,
        "d_loss": avg_d_loss,
        "adv_weight": current_adv_weight,
        "normalization_stats": stats  
    }
    
    torch.save(epoch_checkpoint, f"checkpoints/checkpoint_epoch_{epoch+1:03d}.pt")
    torch.save(epoch_checkpoint, "checkpoints/latest.pt")

    if epoch >= pretrain_epochs:
        if current_ssim > best_val_ssim:
            best_val_ssim = current_ssim
            torch.save(epoch_checkpoint, "checkpoints/best_generator.pt")
            print(f"New best model (SSIM: {current_ssim:.4f})")
    else:
        if metrics["mse"] < best_val_mse:
            best_val_mse = metrics["mse"]
            torch.save(epoch_checkpoint, "checkpoints/best_generator.pt")
            print(f"New best model (MSE: {metrics['mse']:.6f})")

    val_mse.append(metrics["mse"])
    val_psnr.append(metrics["psnr"])
    val_ssim.append(metrics["ssim"])
    g_lossi.append(avg_g_loss)
    d_lossi.append(avg_d_loss)

    print(
        f"Epoch {epoch+1:03d} | "
        f"G Loss: {avg_g_loss:.4f} | "
        f"D Loss: {avg_d_loss:.4f} | "
        f"λ_adv: {current_adv_weight:.6f} | "
        f"Val MSE: {metrics['mse']:.6f} | "
        f"PSNR: {metrics['psnr']:.2f} dB | "
        f"SSIM: {metrics['ssim']:.4f}"
    )

    early_stopper.step(metrics["mse"])
    
    if early_stopper.should_stop:
        print(f"\nEarly stopping triggered at epoch {epoch+1}")
        break

print("\n" + "="*60)
print("Training completed!")
print(f"Best validation MSE: {best_val_mse:.6f}")
print(f"Best validation SSIM: {best_val_ssim:.4f}")
print(f"Checkpoints saved in: checkpoints/")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

axes[0, 0].plot(range(start_epoch, start_epoch + len(g_lossi)), g_lossi, label='Generator Loss', color='blue')
axes[0, 0].axvline(x=pretrain_epochs, color='gray', linestyle='--', label='GAN Start', alpha=0.5)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Generator Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(range(start_epoch, start_epoch + len(d_lossi)), d_lossi, label='Discriminator Loss', color='orange')
axes[0, 1].axvline(x=pretrain_epochs, color='gray', linestyle='--', label='GAN Start', alpha=0.5)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('Discriminator Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].plot(range(start_epoch, start_epoch + len(adv_weights)), adv_weights, label='λ_adv', color='purple')
axes[0, 2].axvline(x=pretrain_epochs, color='gray', linestyle='--', label='GAN Start', alpha=0.5)
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Weight')
axes[0, 2].set_title('Adversarial Loss Weight Over Time')
axes[0, 2].set_yscale('log') 
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

axes[1, 0].plot(range(start_epoch, start_epoch + len(val_mse)), val_mse, label='Validation MSE', color='red')
axes[1, 0].axvline(x=pretrain_epochs, color='gray', linestyle='--', alpha=0.5)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('MSE')
axes[1, 0].set_title('Validation MSE')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(range(start_epoch, start_epoch + len(val_psnr)), val_psnr, label='PSNR', color='green')
axes[1, 1].axvline(x=pretrain_epochs, color='gray', linestyle='--', alpha=0.5)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('PSNR (dB)')
axes[1, 1].set_title('Validation PSNR')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].plot(range(start_epoch, start_epoch + len(val_ssim)), val_ssim, label='SSIM', color='cyan')
axes[1, 2].axhline(y=ssim_warning_threshold, color='r', linestyle='--', 
                   label=f'Warning ({ssim_warning_threshold})', alpha=0.7)
axes[1, 2].axvline(x=pretrain_epochs, color='gray', linestyle='--', alpha=0.5)
axes[1, 2].set_xlabel('Epoch')
axes[1, 2].set_ylabel('SSIM')
axes[1, 2].set_title('Validation SSIM (Key Metric)')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
print("\n✓ Saved training curves to training_curves.png")