import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import wandb
import matplotlib.pyplot as plt

from generator import Generator
from discriminator import Discriminator, RelativisticAverageLoss
from dataset import JetImageDataset
from utils import psnr, ssim
from config import TrainingConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weighted_l1(sr, hr):
    weight = (hr > 0.01).float() * 9 + 1
    return (weight * (sr - hr).abs()).mean()


def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


@torch.no_grad()
def log_images(generator, val_loader, run, epoch, cfg, num_samples=4):
    generator.eval()
    lr, hr = next(iter(val_loader))
    lr, hr = lr[:num_samples].to(device), hr[:num_samples].to(device)

    sr = generator(lr.float())
    sr = torch.clamp(sr, 0, 1)
    lr_up = nn.functional.interpolate(lr, size=(125, 125), mode='nearest')

    for i in range(num_samples):
        vmax = hr[i].max().item()
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(lr_up[i, 0].cpu() / vmax, cmap='hot', vmin=0, vmax=1)
        axes[0].set_title('LR')
        axes[1].imshow(sr[i, 0].cpu() / vmax, cmap='hot', vmin=0, vmax=1)
        axes[1].set_title('SR')
        axes[2].imshow(hr[i, 0].cpu() / vmax, cmap='hot', vmin=0, vmax=1)
        axes[2].set_title('HR')
        for ax in axes:
            ax.axis('off')
        plt.savefig(f'{cfg.samples_dir}/epoch_{epoch}_sample_{i}.png', dpi=150, bbox_inches='tight')
        plt.close()
    generator.train()


@torch.no_grad()
def validate(generator, val_loader):
    generator.eval()
    total_psnr, total_ssim, total_mse, count = 0.0, 0.0, 0.0, 0
    l1 = nn.L1Loss()

    for lr, hr in val_loader:
        lr, hr = lr.to(device), hr.to(device)
        sr = generator(lr.float())
        total_psnr += psnr(sr, hr).item()
        total_ssim += ssim(sr, hr).item()
        total_mse += l1(sr, hr).item()
        count += 1

    generator.train()
    return total_psnr / count, total_ssim / count, total_mse / count


def pretrain(epochs, generator, optimizer, train_loader, val_loader, run, cfg):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience, min_lr=cfg.scheduler_min_lr
    )
    generator.train()
    best_ssim = 0.0

    for epoch in range(epochs):
        total_loss, total_grad, n = 0.0, 0.0, 0

        pbar = tqdm(train_loader, desc=f'Pretrain {epoch+1}/{epochs}')
        for lr, hr in pbar:
            lr, hr = lr.to(device), hr.to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                sr = generator(lr)
                loss = weighted_l1(sr, hr) + cfg.pretrain_ssim_weight * (1 - ssim(sr, hr))
            loss.backward()

            g = grad_norm(generator)
            optimizer.step()

            total_loss += loss.item()
            total_grad += g
            n += 1
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        avg_psnr, avg_ssim, avg_mse = validate(generator, val_loader)
        scheduler.step(avg_ssim)
        current_lr = optimizer.param_groups[0]['lr']

        run.log({
            'pretrain/loss': total_loss / n,
            'pretrain/psnr': avg_psnr,
            'pretrain/ssim': avg_ssim,
            'pretrain/lr': current_lr,
            'epoch': epoch + 1,
        })
        print(f"Epoch {epoch+1}: loss={total_loss/n:.6f}, PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}, lr={current_lr:.2e}")

        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            torch.save({'generator': generator.state_dict()}, f'{cfg.checkpoint_dir}/pretrain_best.pt')

        if (epoch + 1) % 10 == 0 or epoch == 0:
            log_images(generator, val_loader, run, epoch + 1, cfg)

    torch.save({'generator': generator.state_dict()}, f'{cfg.checkpoint_dir}/pretrain_final.pt')


def train_gan(epochs, generator, discriminator, opt_g, opt_d, train_loader, val_loader, run, cfg):
    criterion = RelativisticAverageLoss()
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=epochs)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=epochs)

    generator.train()
    discriminator.train()
    best_ssim = 0.0

    for epoch in range(epochs):
        stats = {k: 0.0 for k in ['d_loss', 'g_loss', 'g_adv', 'g_pix', 'g_ssim']}
        n = 0
        d_updates = 0

        pbar = tqdm(train_loader, desc=f'GAN {epoch+1}/{epochs}')
        for lr, hr in pbar:
            lr, hr = lr.to(device), hr.to(device)

            if n % cfg.discriminator_update_freq == 0:
                opt_d.zero_grad()
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        fake = generator(lr)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    real_logits = discriminator(hr)
                    fake_logits = discriminator(fake)
                    d_loss = criterion.discriminator_loss(real_logits, fake_logits)

                d_loss.backward()
                opt_d.step()
                stats['d_loss'] += d_loss.item()
                d_updates += 1
            else:
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        real_logits = discriminator(hr)

            opt_g.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                fake = generator(lr)
                fake_logits = discriminator(fake)
                g_adv = criterion.generator_loss(real_logits.detach(), fake_logits)
                g_pix = weighted_l1(fake, hr)
                g_ssim = 1 - ssim(fake, hr)
                g_loss = cfg.weight_adversarial * g_adv + cfg.weight_pixel * g_pix + cfg.weight_ssim * g_ssim

            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=cfg.grad_clip_norm)
            opt_g.step()

            stats['g_loss'] += g_loss.item()
            stats['g_adv'] += g_adv.item()
            stats['g_pix'] += g_pix.item()
            stats['g_ssim'] += g_ssim.item()
            n += 1

            pbar.set_postfix({'D': f'{d_loss.item():.3f}', 'G': f'{g_loss.item():.3f}'})

        stats['d_loss'] /= max(d_updates, 1)
        for k in ['g_loss', 'g_adv', 'g_pix', 'g_ssim']:
            stats[k] /= n

        sched_g.step()
        sched_d.step()

        avg_psnr, avg_ssim, avg_mse = validate(generator, val_loader)

        run.log({
            'train/d_loss': stats['d_loss'],
            'train/g_loss': stats['g_loss'],
            'train/g_adv': stats['g_adv'],
            'train/g_pix': stats['g_pix'],
            'train/g_ssim': stats['g_ssim'],
            'val/psnr': avg_psnr,
            'val/ssim': avg_ssim,
            'epoch': epoch + 1,
        })

        print(f"Epoch {epoch+1}: D={stats['d_loss']:.4f}, G={stats['g_loss']:.4f}, PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            log_images(generator, val_loader, run, epoch + 1, cfg)

        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
            }, f'{cfg.checkpoint_dir}/best_model.pt')

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'opt_g': opt_g.state_dict(),
                'opt_d': opt_d.state_dict(),
            }, f'{cfg.checkpoint_dir}/epoch_{epoch+1:03d}.pt')


if __name__ == "__main__":
    cfg = TrainingConfig()

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.samples_dir, exist_ok=True)

    hr_max = torch.load(f'{cfg.tensor_dir}/normalization_stats.pt')['hr_p995']

    train_dataset = JetImageDataset(
        tensor_dir=cfg.tensor_dir, split="train",
        train_ratio=cfg.train_ratio, hr_max=hr_max
    )
    val_dataset = JetImageDataset(
        tensor_dir=cfg.tensor_dir, split="val",
        train_ratio=cfg.train_ratio, hr_max=hr_max
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    print(f"Generator: {sum(p.numel() for p in generator.parameters()):,} params")
    print(f"Discriminator: {sum(p.numel() for p in discriminator.parameters()):,} params")

    run = wandb.init(
        entity="charanvardham",
        project="Super Resolution",
        config={
            "pretrain_epochs": cfg.pretrain_epochs,
            "gan_epochs": cfg.gan_epochs,
            "batch_size": cfg.batch_size,
            "lr_g": cfg.lr_g,
            "lr_d": cfg.lr_d,
        },
    )

    pretrain_path = f'{cfg.checkpoint_dir}/pretrain_final.pt'
    if os.path.exists(pretrain_path):
        print(f"Loading pretrained generator from {pretrain_path}")
        generator.load_state_dict(torch.load(pretrain_path, map_location=device)['generator'])
    else:
        opt_g = torch.optim.AdamW(generator.parameters(), lr=cfg.lr_g)
        pretrain(cfg.pretrain_epochs, generator, opt_g, train_loader, val_loader, run, cfg)

    if cfg.gan_epochs > 0:
        opt_g = torch.optim.AdamW(generator.parameters(), lr=cfg.lr_g)
        opt_d = torch.optim.AdamW(discriminator.parameters(), lr=cfg.lr_d)
        train_gan(cfg.gan_epochs, generator, discriminator, opt_g, opt_d, train_loader, val_loader, run, cfg)

    run.finish()
    print("Done")
