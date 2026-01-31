import torch

import torch.nn as nn

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from torch.utils.data import DataLoader

from torch.utils.data import get_worker_info

import torch.optim 

import torch.nn.functional as F

from utils import validate, train_generator, train_discriminator, set_epoch, should_train_discriminator, AdversarialScheduler, GAN_LRScheduler

from Generator import Generator

from Discriminator import Discriminator

from JetImageDataset import JetImageDataset

from EarlyStopping import EarlyStopping

from tqdm import tqdm

from torch.amp import autocast, GradScaler #type: ignore

from torch.nn.utils import clip_grad_norm_

import os

import logging

from datetime import datetime



# ============================================================================

# CONFIGURATION - These can stay at module level

# ============================================================================



device = 'cuda' if torch.cuda.is_available() else 'cpu'





# ============================================================================

# MAIN TRAINING FUNCTION

# ============================================================================

def main():

    """Main training function - prevents Windows multiprocessing issues"""

    

    log_dir = 'logs'

    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  

    log_file = os.path.join(log_dir, f'training_{timestamp}.log')



    logging.basicConfig(

        level=logging.INFO,

        format='%(asctime)s - %(levelname)s - %(message)s',

        handlers=[

            logging.FileHandler(log_file),

            logging.StreamHandler()

        ]

    )

    logger = logging.getLogger(__name__)



    metrics_log_file = os.path.join(log_dir, f'metrics_{timestamp}.csv')

    with open(metrics_log_file, 'w') as f:

        f.write("epoch,g_loss,d_loss,val_mse,val_psnr,val_ssim,adv_weight,g_lr,d_lr\n")



    logger.info(f"Logging to: {log_file}")

    logger.info(f"Metrics logging to: {metrics_log_file}")



    tensor_dir = "data/pt_tensors"

    pretrain_epochs = 10

    total_epochs = 200

    resume_checkpoint = r"D:\SuperResolution\checkpoints\checkpoint_epoch_010.pt"  # Set to "checkpoints/checkpoint_epoch_012.pt" to resume from epoch 12

    torch.backends.cudnn.benchmark = True  

    torch.backends.cuda.matmul.allow_tf32 = True

    torch.backends.cudnn.allow_tf32 = True

    torch.set_float32_matmul_precision('high')



    logger.info(f"Using device: {device}")



    # Load normalization stats

    stats = torch.load(os.path.join(tensor_dir, 'normalization_stats.pt'))

    lr_max = stats['lr_p995']

    hr_max = stats['hr_p995']

    logger.info(f"Normalization stats - LR max: {lr_max:.4f}, HR max: {hr_max:.4f}")



    # Create models

    generator = Generator().to(device)

    discriminator = Discriminator().to(device)

    logger.info(f"Generator params: {sum(p.numel() for p in generator.parameters()):,}")

    logger.info(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")



    # Loss functions

    pixel_criterion = nn.MSELoss()

    adv_criterion = nn.MSELoss()



    # Optimizers

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.9, 0.999))

    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-5, betas=(0.9, 0.999))  # Reduced from 5e-5



    # Gradient scaler

    scaler = GradScaler('cuda')



    # Checkpoint directory

    os.makedirs('checkpoints', exist_ok=True)

    start_epoch = 0

    best_val_mse = float("inf")

    best_val_ssim = 0.0 

    g_lossi = []

    d_lossi = []

    val_mse, val_psnr, val_ssim = [], [], []

    adv_weights = []

    g_lr_history = []  

    d_lr_history = []  



    # Schedulers - MORE CONSERVATIVE SETTINGS

    early_stopper = EarlyStopping(patience=1000, min_delta=1e-4)

    adv_scheduler = AdversarialScheduler(

        pretrain_epochs=10,

        warmup_epochs=150,     # MINIMAL GAN: Very slow ramp

        min_weight=0.000001,

        max_weight=0.00005,    # MINIMAL GAN: Capped at safe level (10x smaller)

        ssim_threshold=0.02,   # More sensitive

        patience=2,            # Faster response

        reduction_factor=0.5   # Standard reduction

    )



    lr_scheduler = GAN_LRScheduler(

        optimizer_G=g_optimizer,

        optimizer_D=d_optimizer,

        config={

            'lr_G': 1e-4,

            'lr_D': 2e-5,      # Reduced from 5e-5

            'pretrain_epochs': 10,

            'total_epochs': 200,

            'warmup_epochs': 15,  # Increased from 5

        }

    )



    logger.info("Created all schedulers")



    # Create datasets

    train_dataset = JetImageDataset(

        tensor_dir=tensor_dir,

        split='train',

        train_ratio=0.8,

        normalize=True,

        seed=42,

        max_batch_size=256,

        lr_max=lr_max,

        hr_max=hr_max

    )



    val_dataset = JetImageDataset(

        tensor_dir=tensor_dir,

        split='val',

        train_ratio=0.8,

        normalize=True,

        seed=42,

        max_batch_size=256,

        lr_max=lr_max,

        hr_max=hr_max

    )



    # Create data loaders

    train_loader = DataLoader(

        train_dataset,

        batch_size=None,

        num_workers=4,

        pin_memory=True,

        prefetch_factor=2,

        persistent_workers=True

    )



    val_loader = DataLoader(

        val_dataset,

        batch_size=None,

        num_workers=2,

        pin_memory=True,

        prefetch_factor=2,

        persistent_workers=True

    )



    logger.info(f"Created data loaders")



    # Load checkpoint if exists

    if resume_checkpoint and os.path.exists(resume_checkpoint):

        logger.info(f"Loading checkpoint from {resume_checkpoint}...")

        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)

        

        generator.load_state_dict(checkpoint['generator'])

        discriminator.load_state_dict(checkpoint['discriminator'])

        g_optimizer.load_state_dict(checkpoint['g_optimizer'])

        d_optimizer.load_state_dict(checkpoint['d_optimizer'])



        if 'adv_scheduler' in checkpoint:

            adv_scheduler.load_state_dict(checkpoint['adv_scheduler'])

            logger.info("Loaded adversarial scheduler state")

        

        start_epoch = checkpoint['epoch'] + 1

        best_val_mse = checkpoint.get('val_mse', float('inf'))

        best_val_ssim = checkpoint.get('val_ssim', 0.0)

        

        logger.info(f"Resumed from epoch {checkpoint['epoch']}")

        logger.info(f"Previous Val MSE: {checkpoint['val_mse']:.6f}")

        logger.info(f"Previous PSNR: {checkpoint['val_psnr']:.2f} dB")

        logger.info(f"Previous SSIM: {checkpoint['val_ssim']:.4f}")

        logger.info(f"Starting from epoch {start_epoch}")

    else:

        logger.info("No checkpoint found, starting from scratch")



    # Check data range

    logger.info("Checking data range...")

    sample_loader = DataLoader(train_dataset, batch_size=None, num_workers=0)

    for lr, hr in sample_loader:

        logger.info(f"LR: min={lr.min():.4f}, max={lr.max():.4f}, mean={lr.mean():.4f}")

        logger.info(f"HR: min={hr.min():.4f}, max={hr.max():.4f}, mean={hr.mean():.4f}")

        break



    ssim_warning_threshold = 0.3  



    logger.info(f"\nStarting training for {total_epochs} epochs")

    logger.info(f"Pretraining: {pretrain_epochs} epochs")

    logger.info(f"GAN training: {total_epochs - pretrain_epochs} epochs")

    logger.info(f"Starting from epoch {start_epoch}")



    # ========================================================================

    # TRAINING LOOP - WITH COLLAPSE PROTECTION

    # ========================================================================

    for epoch in range(start_epoch, total_epochs):

        set_epoch(epoch)

        

        generator.train()

        discriminator.train()

        lambda_adv = adv_scheduler.get_weight()

        

        current_g_lr = g_optimizer.param_groups[0]['lr']

        current_d_lr = d_optimizer.param_groups[0]['lr']

        g_lr_history.append(current_g_lr)

        d_lr_history.append(current_d_lr)

        

        g_epoch_loss = 0.0

        d_epoch_loss = 0.0

        train_count = 0



        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")

        for batch_idx, (lr_img, hr) in enumerate(pbar):

            lr_img, hr = lr_img.to(device, non_blocking=True), hr.to(device, non_blocking=True)

            

            if torch.isnan(lr_img).any() or torch.isnan(hr).any():

                logger.warning(f"NaN in batch {batch_idx} - SKIPPING")

                continue

            

            if lr_img.min() < -2 or lr_img.max() > 2 or hr.min() < -2 or hr.max() > 2:

                logger.warning(f"Unnormalized data in batch {batch_idx} - SKIPPING")

                continue



            train_count += 1

            

            # PRETRAINING: Only generator, MSE loss only

            if epoch < pretrain_epochs:

                g_optimizer.zero_grad()

                

                with autocast('cuda', dtype=torch.bfloat16):

                    fake_hr = generator(lr_img)

                    loss = pixel_criterion(fake_hr, hr)

                

                scaler.scale(loss).backward()

                scaler.unscale_(g_optimizer)

                clip_grad_norm_(generator.parameters(), max_norm=1.0)

                scaler.step(g_optimizer)

                scaler.update()

                

                g_epoch_loss += loss.item()

                pbar.set_postfix({

                    "Mode": "Pretrain",

                    "MSE": f"{loss.item():.4f}",

                    "G_LR": f"{current_g_lr:.2e}"

                })

            

            # GAN TRAINING: Both generator and discriminator

            else:

                with autocast('cuda', dtype=torch.bfloat16):

                    fake_hr = generator(lr_img)

                

                # Train discriminator (with frequency control)

                train_d = should_train_discriminator(batch_idx, epoch, pretrain_epochs)

                d_loss = train_discriminator(

                    discriminator, d_optimizer, hr, fake_hr, scaler, train_d, lambda_gp=0.0

                )

                

                # Train generator

                g_loss, pixel_loss, adv_loss = train_generator(

                    generator, g_optimizer, discriminator, lr_img, hr, fake_hr, scaler, lambda_adv

                )

                

                g_epoch_loss += g_loss

                d_epoch_loss += d_loss

                

                pbar.set_postfix({

                    "G": f"{g_loss:.4f}", 

                    "D": f"{d_loss:.4f}", 

                    "λ_adv": f"{lambda_adv:.6f}",

                    "G_LR": f"{current_g_lr:.2e}"

                })



        if train_count == 0:

            logger.error(f"Epoch {epoch+1}: No valid training data!")

            continue

        

        # Calculate average losses

        avg_g_loss = g_epoch_loss / train_count

        avg_d_loss = d_epoch_loss / train_count if epoch >= pretrain_epochs else 0.0

        

        # Validation

        with torch.no_grad():

            metrics = validate(generator, val_loader)

        

        current_ssim = metrics["ssim"]

        

        # ====================================================================

        # COLLAPSE DETECTION AND ROLLBACK

        # ====================================================================

        if epoch >= pretrain_epochs:

            ssim_drop = best_val_ssim - current_ssim

            

            # Major collapse detected - rollback and reduce adversarial weight

            if ssim_drop > 0.15:

                logger.error(f"MAJOR COLLAPSE DETECTED! SSIM dropped by {ssim_drop:.4f}")

                logger.error(f"Rolling back to best checkpoint...")

                

                # Load best checkpoint

                if os.path.exists("checkpoints/best_generator.pt"):

                    best_ckpt = torch.load("checkpoints/best_generator.pt", weights_only=False)

                    generator.load_state_dict(best_ckpt['generator'])

                    discriminator.load_state_dict(best_ckpt['discriminator'])

                    g_optimizer.load_state_dict(best_ckpt['g_optimizer'])

                    d_optimizer.load_state_dict(best_ckpt['d_optimizer'])

                    

                    # Drastically reduce adversarial weight

                    adv_scheduler.weight_multiplier *= 0.3

                    adv_scheduler.degradation_count = 0

                    logger.warning(f"Reduced adversarial weight multiplier to {adv_scheduler.weight_multiplier:.6f}")

                    logger.warning(f"Skipping checkpoint save for collapsed epoch {epoch+1}")

                    

                    # Skip to next epoch without saving

                    continue

                else:

                    logger.error("No best checkpoint found to rollback to!")

            

            # Moderate degradation - just warn

            elif ssim_drop > 0.05:

                logger.warning(f"Moderate degradation: {ssim_drop:.4f} - reducing adversarial weight")

                adv_scheduler.weight_multiplier *= 0.7  # Gentle reduction

                adv_scheduler.degradation_count = 0

        

        # Update schedulers

        lambda_adv = adv_scheduler.step(epoch, current_ssim=current_ssim)

        adv_weights.append(lambda_adv)

        current_g_lr, current_d_lr = lr_scheduler.step(

            epoch=epoch,

            d_loss=avg_d_loss,

            g_loss=avg_g_loss,

            lambda_adv=lambda_adv

        )

        

        # Log warning if SSIM is too low

        if epoch >= pretrain_epochs and current_ssim < ssim_warning_threshold:

            logger.warning(f"SSIM at {current_ssim:.4f} (threshold: {ssim_warning_threshold})")

        

        # Save epoch checkpoint

        epoch_checkpoint = {

            "epoch": epoch,

            "generator": generator.state_dict(),

            "discriminator": discriminator.state_dict(),

            "g_optimizer": g_optimizer.state_dict(),

            "d_optimizer": d_optimizer.state_dict(),

            "adv_scheduler": adv_scheduler.state_dict(), 

            "val_mse": metrics["mse"],

            "val_psnr": metrics["psnr"],

            "val_ssim": metrics["ssim"],

            "g_loss": avg_g_loss,

            "d_loss": avg_d_loss,

            "adv_weight": lambda_adv,

        }

        

        torch.save(epoch_checkpoint, f"checkpoints/checkpoint_epoch_{epoch+1:03d}.pt")

        logger.info(f"Saved checkpoint for epoch {epoch+1}")

        

        # Save latest checkpoint

        torch.save(epoch_checkpoint, "checkpoints/latest.pt")

        

        # Save best model based on SSIM (for GAN training) or MSE (for pretraining)

        if epoch >= pretrain_epochs:

            if current_ssim > best_val_ssim:

                best_val_ssim = current_ssim

                torch.save(epoch_checkpoint, "checkpoints/best_generator.pt")

                logger.info(f"New best model (SSIM: {current_ssim:.4f})")

        else:

            if metrics["mse"] < best_val_mse:

                best_val_mse = metrics["mse"]

                torch.save(epoch_checkpoint, "checkpoints/best_generator.pt")

                logger.info(f"New best model (MSE: {metrics['mse']:.6f})")



        # Update metric history

        val_mse.append(metrics["mse"])

        val_psnr.append(metrics["psnr"])

        val_ssim.append(metrics["ssim"])

        g_lossi.append(avg_g_loss)

        d_lossi.append(avg_d_loss)

        

        # Log to CSV

        with open(metrics_log_file, 'a') as f:

            f.write(f"{epoch+1},{avg_g_loss:.6f},{avg_d_loss:.6f},{metrics['mse']:.6f},"

                    f"{metrics['psnr']:.2f},{metrics['ssim']:.4f},{lambda_adv:.6f},"

                    f"{current_g_lr:.2e},{current_d_lr:.2e}\n")



        # Console output

        print(

            f"Epoch {epoch+1:03d} | "

            f"G: {avg_g_loss:.4f} | "

            f"D: {avg_d_loss:.4f} | "

            f"MSE: {metrics['mse']:.6f} | "

            f"SSIM: {metrics['ssim']:.4f} | "

            f"λ_adv: {lambda_adv:.6f}"

        )

        

        # Detailed logging

        logger.info(

            f"Epoch {epoch+1:03d} | "

            f"G Loss: {avg_g_loss:.4f} | "

            f"D Loss: {avg_d_loss:.4f} | "

            f"lambda_adv: {lambda_adv:.6f} | "

            f"Val MSE: {metrics['mse']:.6f} | "

            f"PSNR: {metrics['psnr']:.2f} dB | "

            f"SSIM: {metrics['ssim']:.4f} | "

            f"G_LR: {current_g_lr:.2e} | "

            f"D_LR: {current_d_lr:.2e}"

        )



        # Early stopping check

        early_stopper.step(metrics["mse"])

        

        if early_stopper.should_stop:

            logger.info(f"Early stopping triggered at epoch {epoch+1}")

            break



    # ========================================================================

    # TRAINING COMPLETE - PLOTTING

    # ========================================================================

    logger.info("\n" + "="*80)

    logger.info("Training completed!")

    logger.info(f"Best validation MSE: {best_val_mse:.6f}")

    logger.info(f"Best validation SSIM: {best_val_ssim:.4f}")

    logger.info(f"Checkpoints: checkpoints/")

    logger.info(f"Logs: {log_file}")

    logger.info(f"Metrics: {metrics_log_file}")

    logger.info("="*80)



    # Create plots

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))



    axes[0, 0].plot(range(start_epoch, start_epoch + len(g_lossi)), g_lossi, 

                    label='Generator Loss', color='blue', alpha=0.7)

    axes[0, 0].axvline(x=pretrain_epochs, color='gray', linestyle='--', 

                       label='GAN Start', alpha=0.5)

    axes[0, 0].set_xlabel('Epoch')

    axes[0, 0].set_ylabel('Loss')

    axes[0, 0].set_title('Generator Loss')

    axes[0, 0].legend()

    axes[0, 0].grid(True, alpha=0.3)



    axes[0, 1].plot(range(start_epoch, start_epoch + len(d_lossi)), d_lossi, 

                    label='Discriminator Loss', color='orange', alpha=0.7)

    axes[0, 1].axvline(x=pretrain_epochs, color='gray', linestyle='--', 

                       label='GAN Start', alpha=0.5)

    axes[0, 1].set_xlabel('Epoch')

    axes[0, 1].set_ylabel('Loss')

    axes[0, 1].set_title('Discriminator Loss')

    axes[0, 1].legend()

    axes[0, 1].grid(True, alpha=0.3)



    axes[0, 2].plot(range(start_epoch, start_epoch + len(adv_weights)), adv_weights, 

                    label='λ_adv', color='purple', alpha=0.7)

    axes[0, 2].axvline(x=pretrain_epochs, color='gray', linestyle='--', 

                       label='GAN Start', alpha=0.5)

    axes[0, 2].set_xlabel('Epoch')

    axes[0, 2].set_ylabel('Weight')

    axes[0, 2].set_title('Adversarial Loss Weight')

    axes[0, 2].set_yscale('log')

    axes[0, 2].legend()

    axes[0, 2].grid(True, alpha=0.3)



    axes[0, 3].plot(range(start_epoch, start_epoch + len(g_lr_history)), g_lr_history, 

                    label='Generator LR', color='blue', marker='o', markersize=3)

    axes[0, 3].plot(range(start_epoch, start_epoch + len(d_lr_history)), d_lr_history, 

                    label='Discriminator LR', color='orange', marker='s', markersize=3)

    axes[0, 3].axvline(x=pretrain_epochs, color='gray', linestyle='--', 

                       label='GAN Start', alpha=0.5)

    axes[0, 3].set_xlabel('Epoch')

    axes[0, 3].set_ylabel('Learning Rate')

    axes[0, 3].set_title('Learning Rate Schedule')

    axes[0, 3].set_yscale('log')

    axes[0, 3].legend()

    axes[0, 3].grid(True, alpha=0.3)



    axes[1, 0].plot(range(start_epoch, start_epoch + len(val_mse)), val_mse, 

                    label='Validation MSE', color='red', alpha=0.7)

    axes[1, 0].axvline(x=pretrain_epochs, color='gray', linestyle='--', alpha=0.5)

    axes[1, 0].set_xlabel('Epoch')

    axes[1, 0].set_ylabel('MSE')

    axes[1, 0].set_title('Validation MSE')

    axes[1, 0].legend()

    axes[1, 0].grid(True, alpha=0.3)



    axes[1, 1].plot(range(start_epoch, start_epoch + len(val_psnr)), val_psnr, 

                    label='PSNR', color='green', alpha=0.7)

    axes[1, 1].axvline(x=pretrain_epochs, color='gray', linestyle='--', alpha=0.5)

    axes[1, 1].set_xlabel('Epoch')

    axes[1, 1].set_ylabel('PSNR (dB)')

    axes[1, 1].set_title('Validation PSNR')

    axes[1, 1].legend()

    axes[1, 1].grid(True, alpha=0.3)



    axes[1, 2].plot(range(start_epoch, start_epoch + len(val_ssim)), val_ssim, 

                    label='SSIM', color='cyan', alpha=0.7)

    axes[1, 2].axhline(y=ssim_warning_threshold, color='r', linestyle='--', 

                       label=f'Warning ({ssim_warning_threshold})', alpha=0.7)

    axes[1, 2].axvline(x=pretrain_epochs, color='gray', linestyle='--', alpha=0.5)

    axes[1, 2].set_xlabel('Epoch')

    axes[1, 2].set_ylabel('SSIM')

    axes[1, 2].set_title('Validation SSIM')

    axes[1, 2].legend()

    axes[1, 2].grid(True, alpha=0.3)



    axes[1, 3].axis('off')

    summary_text = (

        f"Training Summary\n"

        f"{'='*30}\n"

        f"Total Epochs: {epoch+1}\n"

        f"Best MSE: {best_val_mse:.6f}\n"

        f"Best SSIM: {best_val_ssim:.4f}\n"

        f"Final G LR: {g_lr_history[-1]:.2e}\n"

        f"Final D LR: {d_lr_history[-1]:.2e}\n"

        f"Final lambda_adv: {adv_weights[-1]:.6f}"

    )

    axes[1, 3].text(0.1, 0.5, summary_text, fontsize=12, family='monospace',

                    verticalalignment='center')

    plt.tight_layout()

    plot_file = os.path.join(log_dir, f'training_curves_{timestamp}.png')

    plt.savefig(plot_file, dpi=150, bbox_inches='tight')

    logger.info(f"Saved training curves to {plot_file}")

    print(f"\nTraining complete! Check {log_dir}/ for logs and plots")





# ============================================================================

# ENTRY POINT - CRITICAL FOR WINDOWS

# ============================================================================

if __name__ == '__main__':

    main()