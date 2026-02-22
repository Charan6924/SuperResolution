import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from Generator import Generator
from Discriminator import Discriminator, RelativisticAverageLoss
from JetImageDataset import JetImageDataset
from utils import psnr, ssim
import os
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pixel_criterion = nn.L1Loss()
pretrain_pixel_loss = nn.L1Loss()
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

@torch.no_grad()
def log_images(generator, val_loader, run, epoch, num_samples=4):
    generator.eval()
    lr, hr = next(iter(val_loader))
    lr = lr[:num_samples].to(device)
    hr = hr[:num_samples].to(device)

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        sr = generator(lr)
    lr_upscaled = torch.nn.functional.interpolate(lr, size=(125, 125), mode='nearest')

    images = []
    for i in range(num_samples):
        lr_img = lr_upscaled[i].cpu().float()
        sr_img = sr[i].cpu().float()
        hr_img = hr[i].cpu().float()

        images.append(wandb.Image(lr_img, caption=f"LR_{i}"))
        images.append(wandb.Image(sr_img, caption=f"SR_{i}"))
        images.append(wandb.Image(hr_img, caption=f"HR_{i}"))

    run.log({f"samples/epoch_{epoch}": images})
    generator.train()

@torch.no_grad()
def validate(generator, val_loader):
    generator.eval()
    total_psnr = total_ssim = total_mse = 0.0
    count = 0

    for lr, hr in val_loader:
        lr = lr.to(device)
        hr = hr.to(device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            fake_hr = generator(lr)
            total_psnr += psnr(fake_hr, hr).item()
            total_ssim += ssim(fake_hr, hr).item()
            total_mse  += pixel_criterion(fake_hr, hr).item()
        count += 1

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    avg_mse  = total_mse  / count
    generator.train()
    print(f"Validation - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, MSE: {avg_mse:.6f}")
    return avg_psnr, avg_ssim, avg_mse

def pretrain_generator(num_epochs, generator, optimizer_G, train_loader, val_loader, run, scheduler):
    print('Generator Pretraining')
    generator.train()
    epoch_bar = tqdm(range(num_epochs), desc='Pretrain Epochs', position=0)

    for epoch in epoch_bar:
        total_g_loss = 0.0
        num_batches = 0
        batch_bar = tqdm(train_loader, desc=f'Pretrain Epoch {epoch+1}/{num_epochs}', position=1, leave=False)

        for lr, hr in batch_bar:
            lr = lr.to(device)
            hr = hr.to(device)

            optimizer_G.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                fake_hr  = generator(lr)
                g_loss   = pretrain_pixel_loss(fake_hr, hr)
            g_loss.backward()
            optimizer_G.step()

            total_g_loss += g_loss.item()
            num_batches += 1
            batch_bar.set_postfix({'G': f'{g_loss.item():.6f}'})

        avg_g_loss = total_g_loss / num_batches
        scheduler.step()
        run.log({
            'pretrain/g_loss': avg_g_loss,
            'pretrain/lr': scheduler.get_last_lr()[0],
            'pretrain/epoch': epoch + 1,
        })

    avg_psnr, avg_ssim, avg_mse = validate(generator, val_loader)
    run.log({'pretrain/val_psnr': avg_psnr, 'pretrain/val_ssim': avg_ssim, 'pretrain/val_mse': avg_mse})

    torch.save({'epoch':epoch,
        'generator': {k.replace('_orig_mod.', ''): v for k, v in generator.state_dict().items()},
        'g_optimizer': optimizer_G.state_dict(),
    }, f'{checkpoint_dir}/pretrain_final.pt')
    print(f"Pretrain checkpoint saved to {checkpoint_dir}/pretrain_final.pt")

def compute_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5

def train(num_epochs, generator, discriminator, optimizer_D, optimizer_G,
          train_loader, val_loader, criterion, run, scheduler_G, scheduler_D, log_images_every=10):
    print('Starting GAN training loop...')
    epoch_bar = tqdm(range(num_epochs), desc='Epochs', position=0)
    max_ssim  = 0.0
    generator.train()
    discriminator.train()

    for epoch in epoch_bar:
        total_d_loss = total_g_loss = total_g_adv_loss = total_g_pixel_loss = 0.0
        total_g_grad_norm = total_d_grad_norm = 0.0
        total_real_logits = total_fake_logits = 0.0
        num_batches = 0
        pixel_weight = max(0.1, 1.0 - epoch * (0.9 / num_epochs))

        batch_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', position=1, leave=False)

        for lr, hr in batch_bar:
            lr = lr.to(device)
            hr = hr.to(device)

            optimizer_D.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                fake_hr = generator(lr).detach() 
                real_logits = discriminator(hr)
                fake_logits = discriminator(fake_hr)
                d_loss = criterion.discriminator_loss(real_logits, fake_logits)
            d_loss.backward()
            d_grad_norm = compute_grad_norm(discriminator)
            optimizer_D.step()

            optimizer_G.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                fake_hr_g = generator(lr)
                fake_logits_g = discriminator(fake_hr_g)
                real_logits_g = real_logits.detach()
                g_adv_loss = criterion.generator_loss(real_logits_g, fake_logits_g)
                g_pixel_loss = pixel_criterion(fake_hr_g, hr)
                g_loss = g_adv_loss + g_pixel_loss * pixel_weight
            g_loss.backward()
            g_grad_norm = compute_grad_norm(generator)
            optimizer_G.step()

            # === TRACKING ===
            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
            total_g_adv_loss += g_adv_loss.item()
            total_g_pixel_loss += g_pixel_loss.item()
            total_g_grad_norm += g_grad_norm
            total_d_grad_norm += d_grad_norm
            total_real_logits += real_logits.mean().item()
            total_fake_logits += fake_logits.mean().item()
            num_batches += 1

            if num_batches == 1 and epoch == 0:
                print(f"\n{'='*60}")
                print(f"DEBUG FIRST BATCH:")
                print(f"  D grad norm: {d_grad_norm:.6f}")
                print(f"  G grad norm: {g_grad_norm:.6f}")
                print(f"  Real logits mean: {real_logits.mean().item():.4f}")
                print(f"  Fake logits mean: {fake_logits.mean().item():.4f}")
                print(f"  g_pixel_loss: {g_pixel_loss.item():.6f}")
                print(f"  g_adv_loss: {g_adv_loss.item():.6f}")
                print(f"  SR output range: [{fake_hr_g.min().item():.4f}, {fake_hr_g.max().item():.4f}]")
                print(f"{'='*60}\n")

            batch_bar.set_postfix({
                'D': f'{d_loss.item():.4f}',
                'G': f'{g_loss.item():.4f}',
                'Dg': f'{d_grad_norm:.2f}',
                'Gg': f'{g_grad_norm:.2f}'
            })

        scheduler_G.step()
        scheduler_D.step()

        avg_d_loss = total_d_loss / num_batches
        avg_g_loss = total_g_loss / num_batches
        avg_g_adv_loss = total_g_adv_loss / num_batches
        avg_g_pixel_loss = total_g_pixel_loss / num_batches
        avg_g_grad_norm = total_g_grad_norm / num_batches
        avg_d_grad_norm = total_d_grad_norm / num_batches
        avg_real_logits = total_real_logits / num_batches
        avg_fake_logits = total_fake_logits / num_batches

        print(f"Epoch {epoch+1}: D_loss={avg_d_loss:.4f}, G_loss={avg_g_loss:.4f}, "
              f"D_grad={avg_d_grad_norm:.4f}, G_grad={avg_g_grad_norm:.4f}, "
              f"Real={avg_real_logits:.4f}, Fake={avg_fake_logits:.4f}")

        avg_psnr, avg_ssim, avg_mse = validate(generator, val_loader)
        run.log({
            'train/d_loss': avg_d_loss,
            'train/g_loss': avg_g_loss,
            'train/g_adv_loss': avg_g_adv_loss,
            'train/g_pixel_loss': avg_g_pixel_loss,
            'train/pixel_weight': pixel_weight,
            'train/lr_G': scheduler_G.get_last_lr()[0],
            'debug/g_grad_norm': avg_g_grad_norm,
            'debug/d_grad_norm': avg_d_grad_norm,
            'debug/real_logits_mean': avg_real_logits,
            'debug/fake_logits_mean': avg_fake_logits,
            'train/lr_D': scheduler_D.get_last_lr()[0],
            'val/psnr': avg_psnr,
            'val/ssim': avg_ssim,
            'val/mse': avg_mse,
            'epoch': epoch + 1,
        })

        if (epoch + 1) % log_images_every == 0 or epoch == 0:
            log_images(generator, val_loader, run, epoch + 1)

        if avg_ssim > max_ssim:
            max_ssim = avg_ssim
            torch.save({'epoch': epoch,'generator': generator.state_dict(),'discriminator': discriminator.state_dict(),'g_optimizer':   optimizer_G.state_dict(),'d_optimizer':   optimizer_D.state_dict(),}, f'{checkpoint_dir}/best_model.pt')

        torch.save({'epoch':epoch,'generator':generator.state_dict(),'discriminator': discriminator.state_dict(),'g_optimizer':   optimizer_G.state_dict(),'d_optimizer':   optimizer_D.state_dict(),}, f'{checkpoint_dir}/epoch_{epoch+1:03d}.pt')


if __name__ == "__main__":
    num_epochs      = 200
    pretrain_epochs = 10

    tensor_dir = 'data/pt_tensors'
    hr_max = torch.load(f'{tensor_dir}/normalization_stats.pt')['hr_p995']
    train_dataset = JetImageDataset(tensor_dir=tensor_dir, split="train", train_ratio=0.8, hr_max=hr_max)
    val_dataset = JetImageDataset(tensor_dir=tensor_dir, split="val",   train_ratio=0.8, hr_max=hr_max)
    train_loader = DataLoader(train_dataset, batch_size=256, num_workers=3)
    val_loader = DataLoader(val_dataset,   batch_size=256, num_workers=3)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    optimizer_G = torch.optim.AdamW(generator.parameters(),     lr=1e-4, betas=(0.9, 0.999))
    optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=1e-6, betas=(0.9, 0.999))  # Reduced from 1e-5

    pretrain_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=pretrain_epochs)
    criterion = RelativisticAverageLoss()
    print('Loaded data and created models')

    run = wandb.init(
        entity="charanvardham",
        project="Super Resolution",
        config={
            "architecture": "ESRGAN",
            "epochs": num_epochs,
            "pretrain_epochs": pretrain_epochs,
            "batch_size": 256,
            "generator_lr": 1e-4,
            "discriminator_lr": 1e-5,
            "betas": (0.9, 0.999),
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "pixel_weight_start": 1.0,
            "pixel_weight_end": 0.1,
            "discriminator_update_freq": "every_2_epochs",
            "generator": {
                "num_features": 64,
                "growth_channels": 32,
                "num_blocks": 8,
                "residual_scale": 0.2,
            },
            "discriminator": {
                "base_features": 64,
            },
            "dataset": {
                "train_ratio": 0.8,
                "val_ratio": 0.1,
                "hr_max": hr_max,
                "num_workers": 3,
            },
        },
    )

    # Load pretrained checkpoint instead of retraining
    pretrain_ckpt = torch.load(f'{checkpoint_dir}/pretrain_final.pt', map_location=device)
    generator.load_state_dict(pretrain_ckpt['generator'])
    print(f"Loaded pretrained generator from {checkpoint_dir}/pretrain_final.pt")
    generator = torch.compile(generator)  
    for pg in optimizer_G.param_groups:
        pg['lr'] = 1e-4
    lr_scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=num_epochs)
    lr_scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=num_epochs)
    train(num_epochs=num_epochs,generator=generator,discriminator=discriminator,optimizer_D=optimizer_D,optimizer_G=optimizer_G,train_loader=train_loader,val_loader=val_loader,criterion=criterion,run=run,scheduler_G=lr_scheduler_G,scheduler_D=lr_scheduler_D,)

    run.finish()