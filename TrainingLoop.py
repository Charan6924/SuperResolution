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
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)


def compute_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


@torch.no_grad()
def log_images(generator, val_loader, run, epoch, num_samples=4):
    generator.eval()
    lr, hr = next(iter(val_loader))
    lr = lr[:num_samples].to(device)
    hr = hr[:num_samples].to(device)

    sr = generator(lr)
    lr_upscaled = torch.nn.functional.interpolate(lr, size=(125, 125), mode='nearest')

    images = []
    for i in range(num_samples):
        images.append(wandb.Image(lr_upscaled[i].cpu().float(), caption=f"LR_{i}"))
        images.append(wandb.Image(sr[i].cpu().float(), caption=f"SR_{i}"))
        images.append(wandb.Image(hr[i].cpu().float(), caption=f"HR_{i}"))

    run.log({f"samples/epoch_{epoch}": images})
    generator.train()


@torch.no_grad()
def validate(generator, val_loader):
    generator.eval()
    total_psnr = total_ssim = total_mse = 0.0
    count = 0
    pixel_criterion = nn.L1Loss()

    for lr, hr in val_loader:
        lr, hr = lr.to(device), hr.to(device)
        fake_hr = generator(lr)
        total_psnr += psnr(fake_hr, hr).item()
        total_ssim += ssim(fake_hr, hr).item()
        total_mse += pixel_criterion(fake_hr, hr).item()
        count += 1

    generator.train()
    avg_psnr, avg_ssim, avg_mse = total_psnr/count, total_ssim/count, total_mse/count
    print(f"Validation - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, MSE: {avg_mse:.6f}")
    return avg_psnr, avg_ssim, avg_mse


def pretrain_generator(num_epochs, generator, optimizer, train_loader, val_loader, run):
    pixel_loss_fn = nn.L1Loss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    generator.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_grad_norm = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f'Pretrain {epoch+1}/{num_epochs}')
        for lr, hr in pbar:
            lr, hr = lr.to(device), hr.to(device)

            optimizer.zero_grad()
            sr = generator(lr)
            loss = pixel_loss_fn(sr, hr)
            loss.backward()

            grad_norm = compute_grad_norm(generator)
            optimizer.step()

            total_loss += loss.item()
            total_grad_norm += grad_norm
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'grad': f'{grad_norm:.4f}'
            })

        scheduler.step()
        avg_loss = total_loss / num_batches
        avg_grad = total_grad_norm / num_batches

        # Validate
        avg_psnr, avg_ssim, avg_mse = validate(generator, val_loader)

        # Log to wandb
        run.log({
            'pretrain/loss': avg_loss,
            'pretrain/grad_norm': avg_grad,
            'pretrain/lr': scheduler.get_last_lr()[0],
            'pretrain/psnr': avg_psnr,
            'pretrain/ssim': avg_ssim,
            'pretrain/mse': avg_mse,
            'pretrain/epoch': epoch + 1,
        })

        print(f"Epoch {epoch+1}: loss={avg_loss:.6f}, grad={avg_grad:.4f}, PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}")

    # Save pretrained checkpoint
    torch.save({
        'generator': generator.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, f'{checkpoint_dir}/pretrain_final.pt')
    print(f"Saved pretrained generator to {checkpoint_dir}/pretrain_final.pt")


def train_gan(num_epochs, generator, discriminator, opt_G, opt_D,
              train_loader, val_loader, run, log_images_every=10):
    criterion = RelativisticAverageLoss()
    pixel_loss_fn = nn.L1Loss()

    sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=num_epochs)
    sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=num_epochs)

    generator.train()
    discriminator.train()
    best_ssim = 0.0

    for epoch in range(num_epochs):
        pixel_weight = max(0.1, 1.0 - epoch * (0.9 / num_epochs))

        stats = {k: 0.0 for k in ['d_loss', 'g_loss', 'g_adv', 'g_pix',
                                   'd_grad', 'g_grad', 'real_logits', 'fake_logits']}
        num_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for lr, hr in pbar:
            lr, hr = lr.to(device), hr.to(device)

            opt_D.zero_grad()
            with torch.no_grad():
                fake_hr = generator(lr)

            real_logits = discriminator(hr)
            fake_logits = discriminator(fake_hr)
            d_loss = criterion.discriminator_loss(real_logits, fake_logits)

            d_loss.backward()
            d_grad = compute_grad_norm(discriminator)
            opt_D.step()
            opt_G.zero_grad()

            fake_hr = generator(lr)
            fake_logits_g = discriminator(fake_hr)

            g_adv = criterion.generator_loss(real_logits.detach(), fake_logits_g)
            g_pix = pixel_loss_fn(fake_hr, hr)
            g_loss = g_adv + pixel_weight * g_pix

            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=10.0)  # Prevent explosion
            g_grad = compute_grad_norm(generator)
            opt_G.step()
            stats['d_loss'] += d_loss.item()
            stats['g_loss'] += g_loss.item()
            stats['g_adv'] += g_adv.item()
            stats['g_pix'] += g_pix.item()
            stats['d_grad'] += d_grad
            stats['g_grad'] += g_grad
            stats['real_logits'] += real_logits.mean().item()
            stats['fake_logits'] += fake_logits.mean().item()
            num_batches += 1

            pbar.set_postfix({
                'D': f'{d_loss.item():.3f}',
                'G': f'{g_loss.item():.3f}',
                'Dg': f'{d_grad:.1f}',
                'Gg': f'{g_grad:.1f}'
            })

        # Average stats
        for k in stats:
            stats[k] /= num_batches

        sched_G.step()
        sched_D.step()

        # Validate
        avg_psnr, avg_ssim, avg_mse = validate(generator, val_loader)

        # Print epoch summary
        print(f"Epoch {epoch+1}: D={stats['d_loss']:.4f}, G={stats['g_loss']:.4f}, "
              f"Dg={stats['d_grad']:.2f}, Gg={stats['g_grad']:.2f}, "
              f"Real={stats['real_logits']:.2f}, Fake={stats['fake_logits']:.2f}")

        # Log to wandb
        run.log({
            'train/d_loss': stats['d_loss'],
            'train/g_loss': stats['g_loss'],
            'train/g_adv_loss': stats['g_adv'],
            'train/g_pixel_loss': stats['g_pix'],
            'train/pixel_weight': pixel_weight,
            'train/lr_G': sched_G.get_last_lr()[0],
            'train/lr_D': sched_D.get_last_lr()[0],
            'debug/d_grad_norm': stats['d_grad'],
            'debug/g_grad_norm': stats['g_grad'],
            'debug/real_logits': stats['real_logits'],
            'debug/fake_logits': stats['fake_logits'],
            'val/psnr': avg_psnr,
            'val/ssim': avg_ssim,
            'val/mse': avg_mse,
            'epoch': epoch + 1,
        })

        # Log images periodically
        if (epoch + 1) % log_images_every == 0 or epoch == 0:
            log_images(generator, val_loader, run, epoch + 1)

        # Save best model
        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            torch.save({
                'epoch': epoch,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
            }, f'{checkpoint_dir}/best_model.pt')

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'opt_G': opt_G.state_dict(),
                'opt_D': opt_D.state_dict(),
            }, f'{checkpoint_dir}/epoch_{epoch+1:03d}.pt')


if __name__ == "__main__":
    PRETRAIN_EPOCHS = 10
    GAN_EPOCHS = 200
    BATCH_SIZE = 256
    LR_G = 1e-4
    LR_D = 1e-5  # Lower than G to prevent D from dominating

    tensor_dir = 'data/pt_tensors'
    hr_max = torch.load(f'{tensor_dir}/normalization_stats.pt')['hr_p995']

    train_dataset = JetImageDataset(tensor_dir=tensor_dir, split="train", train_ratio=0.8, hr_max=hr_max)
    val_dataset = JetImageDataset(tensor_dir=tensor_dir, split="val", train_ratio=0.8, hr_max=hr_max)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=3)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=3)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    print(f"Generator params: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")

    run = wandb.init(
        entity="charanvardham",
        project="Super Resolution",
        config={
            "architecture": "ESRGAN",
            "pretrain_epochs": PRETRAIN_EPOCHS,
            "gan_epochs": GAN_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr_G": LR_G,
            "lr_D": LR_D,
        },
    )

    # Load pretrained generator (skip pretraining if checkpoint exists)
    pretrain_path = f'{checkpoint_dir}/pretrain_final.pt'
    if os.path.exists(pretrain_path):
        print(f"Loading pretrained generator from {pretrain_path}")
        ckpt = torch.load(pretrain_path, map_location=device)
        generator.load_state_dict(ckpt['generator'])
    else:
        opt_G = torch.optim.AdamW(generator.parameters(), lr=LR_G, betas=(0.9, 0.999))
        pretrain_generator(PRETRAIN_EPOCHS, generator, opt_G, train_loader, val_loader, run)

    opt_G = torch.optim.AdamW(generator.parameters(), lr=LR_G, betas=(0.9, 0.999))
    opt_D = torch.optim.AdamW(discriminator.parameters(), lr=LR_D, betas=(0.9, 0.999))

    train_gan(GAN_EPOCHS, generator, discriminator, opt_G, opt_D,
              train_loader, val_loader, run)

    run.finish()
    print("Training complete!")
