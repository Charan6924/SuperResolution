import torch
import torch.nn as nn
import torch.nn.functional as F
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

def pretrain_generator(num_epochs, generator, optimizer_G, train_loader, run, scheduler):
    print('Generator Pretraining')
    generator.train()
    epoch_bar = tqdm(range(num_epochs), desc='Pretrain Epochs', position=0)

    for epoch in epoch_bar:
        g_loss_val = 0.0
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

            g_loss_val = g_loss.item()
            batch_bar.set_postfix({'G': f'{g_loss_val:.6f}'})

        scheduler.step()
        run.log({'pretrain/g_loss': g_loss_val, 'pretrain/epoch': epoch + 1})

    avg_psnr, avg_ssim, avg_mse = validate(generator, val_loader)
    run.log({'pretrain/val_psnr': avg_psnr, 'pretrain/val_ssim': avg_ssim, 'pretrain/val_mse': avg_mse})

    torch.save({'epoch':epoch,
        'generator': {k.replace('_orig_mod.', ''): v for k, v in generator.state_dict().items()},
        'g_optimizer': optimizer_G.state_dict(),
    }, f'{checkpoint_dir}/pretrain_final.pt')
    print(f"Pretrain checkpoint saved to {checkpoint_dir}/pretrain_final.pt")

def train(num_epochs, generator, discriminator, optimizer_D, optimizer_G,
          train_loader, criterion, run, scheduler_G, scheduler_D):
    print('Starting GAN training loop...')
    epoch_bar = tqdm(range(num_epochs), desc='Epochs', position=0)
    max_ssim  = 0.0
    generator.train()
    discriminator.train()

    for epoch in epoch_bar:
        d_loss_val = g_loss_val = 0.0
        pixel_weight = max(0.1, 1.0 - epoch * (0.9 / num_epochs))

        batch_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', position=1, leave=False)

        for lr, hr in batch_bar:
            lr = lr.to(device)
            hr = hr.to(device)

            if epoch % 2 == 0:
                optimizer_D.zero_grad()
                with torch.no_grad():
                    fake_hr = generator(lr)         
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    real_logits = discriminator(hr)
                    fake_logits = discriminator(fake_hr)
                    d_loss      = criterion.discriminator_loss(real_logits, fake_logits)
                d_loss.backward()
                optimizer_D.step()

            optimizer_G.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                fake_hr_g    = generator(lr)             
                fake_logits_g = discriminator(fake_hr_g)
                real_logits_g = real_logits.detach()     
                g_loss  = criterion.generator_loss(real_logits_g, fake_logits_g)
                g_loss += pixel_criterion(fake_hr_g, hr) * pixel_weight
            g_loss.backward()
            optimizer_G.step()

            d_loss_val = d_loss.item()
            g_loss_val = g_loss.item()
            batch_bar.set_postfix({'D': f'{d_loss_val:.4f}', 'G': f'{g_loss_val:.4f}'})

        scheduler_G.step()
        scheduler_D.step()

        avg_psnr, avg_ssim, avg_mse = validate(generator, val_loader)
        run.log({
            'train/d_loss':  d_loss_val,
            'train/g_loss':  g_loss_val,
            'train/pixel_weight': pixel_weight,
            'val/psnr':      avg_psnr,
            'val/ssim':      avg_ssim,
            'val/mse':       avg_mse,
            'epoch':         epoch + 1,
        })

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
    optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=1e-5, betas=(0.9, 0.999))

    pretrain_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=pretrain_epochs)
    criterion = RelativisticAverageLoss()
    print('Loaded data and created models')

    run = wandb.init(
        entity="charanvardham",
        project="Super Resolution",
        config={
            "learning_rate": 1e-4,
            "architecture":  "ESRGAN",
            "epochs":        num_epochs,
            "batch_size":    256,
            "betas":         (0.9, 0.999),
        },
    )

    pretrain_generator(num_epochs=pretrain_epochs,generator=generator,optimizer_G=optimizer_G,train_loader=train_loader,run=run,scheduler=pretrain_scheduler,)
    generator = torch.compile(generator)
    for pg in optimizer_G.param_groups:
        pg['lr'] = 1e-4
    lr_scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=num_epochs)
    lr_scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=num_epochs)
    train(num_epochs=num_epochs,generator=generator,discriminator=discriminator,optimizer_D=optimizer_D,optimizer_G=optimizer_G,train_loader=train_loader,criterion=criterion,run=run,scheduler_G=lr_scheduler_G,scheduler_D=lr_scheduler_D,)

    run.finish()