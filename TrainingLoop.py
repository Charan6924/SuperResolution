import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler # type: ignore
import torch.optim.optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from Generator import Generator
from Discriminator import Discriminator, RelativisticAverageLoss
from JetImageDataset import JetImageDataset
from utils import  psnr,ssim
import logging
import time
import os
import wandb

num_epochs = 200
pretrain_epochs = 10
device = 'cuda'
generator = Generator().to(device)
generator = torch.compile(generator)
discriminator = Discriminator().to(device)
tensor_dir = 'data/pt_tensors'
hr_max = torch.load(f'{tensor_dir}/normalization_stats.pt')['hr_p995']
train_dataset = JetImageDataset(tensor_dir=tensor_dir, split="train", train_ratio=0.8, hr_max=hr_max)
val_dataset   = JetImageDataset(tensor_dir=tensor_dir, split="val",   train_ratio=0.8, hr_max=hr_max)
train_loader  = DataLoader(train_dataset, batch_size=256,num_workers = 3)
val_loader    = DataLoader(val_dataset,   batch_size=256, num_workers = 3)
optimizer_G = torch.optim.AdamW(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
lr_scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=num_epochs)
lr_scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=num_epochs)
pretrain_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=pretrain_epochs)
criterion = RelativisticAverageLoss()
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pixel_criterion = nn.MSELoss()
pretrain_pixel_loss = nn.L1Loss()
adv_criterion = nn.MSELoss()
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint = torch.load('/mnt/vstor/courses/csds312/cxv166/assignment/datascience/SuperResolution/checkpoints/pretrain_final.pt')
generator.load_state_dict(checkpoint['generator'])
#discriminator.load_state_dict(checkpoint['discriminator'])
optimizer_G.load_state_dict(checkpoint['g_optimizer'])
#optimizer_G.load_state_dict(checkpoint['d_optimizer'])
#start_epoch = checkpoint['epoch'] + 1
print('loaded data and created models')

@torch.no_grad()
def validate(generator, val_loader):
    generator.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total_mse = 0.0
    count = 0
    for lr, hr in val_loader:
        lr = lr.to(device)
        hr = hr.to(device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            fake_hr = generator(lr)
            total_psnr += psnr(fake_hr, hr).item()
            total_ssim += ssim(fake_hr, hr).item()
            total_mse += pixel_criterion(fake_hr, hr).item()
            count += 1
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    avg_mse = total_mse / count
    generator.train()
    print(f"Validation - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, MSE: {avg_mse:.6f}")
    return avg_psnr, avg_ssim, avg_mse

def pretrain_generator(num_epochs, generator, optimizer_G, train_loader, criterion, run):
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
                fake_hr = generator(lr)
                g_loss = pretrain_pixel_loss(fake_hr, hr)
            g_loss.backward()
            optimizer_G.step()

            g_loss_val = g_loss.item()

            batch_bar.set_postfix({
                'G': f'{g_loss_val:.4f}'
            })
        pretrain_scheduler.step()
        run.log({'pretrain/g_loss': g_loss_val, 'pretrain/epoch': epoch + 1})

    avg_psnr, avg_ssim, avg_mse = validate(generator, val_loader)
    run.log({'pretrain/val_psnr': avg_psnr, 'pretrain/val_ssim': avg_ssim, 'pretrain/val_mse': avg_mse})
    torch.save({'epoch': epoch,'generator': generator.state_dict(),'g_optimizer': optimizer_G.state_dict(),}, f'{checkpoint_dir}/pretrain_final.pt')


def train(num_epochs, generator, discriminator, optimizer_D, optimizer_G, train_loader, criterion, run):
    print('Starting training loop...')
    epoch_bar = tqdm(range(num_epochs), desc='Epochs', position=0)
    max_ssim = 0.0
    generator.train()

    for epoch in epoch_bar: 
        d_loss_val = g_loss_val = 0.0

        batch_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', position=1, leave=False)

        for lr, hr in batch_bar:
            lr = lr.to(device)
            hr = hr.to(device)

            optimizer_D.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                fake_hr = generator(lr)
                real_logits = discriminator(hr)
                fake_logits = discriminator(fake_hr.detach())
                d_loss = criterion.discriminator_loss(real_logits, fake_logits)
            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                fake_logits_g = discriminator(fake_hr)
                real_logits_g = discriminator(hr)       
                g_loss = criterion.generator_loss(real_logits_g, fake_logits_g)
            g_loss.backward()
            optimizer_G.step()

            d_loss_val = d_loss.item()
            g_loss_val = g_loss.item()

            batch_bar.set_postfix({
                'D': f'{d_loss_val:.4f}',
                'G': f'{g_loss_val:.4f}',
            })
        
        lr_scheduler_G.step()
        lr_scheduler_D.step()   
        avg_psnr, avg_ssim, avg_mse = validate(generator, val_loader)
        run.log({
            'train/d_loss': d_loss_val,
            'train/g_loss': g_loss_val,
            'val/psnr': avg_psnr,
            'val/ssim': avg_ssim,
            'val/mse': avg_mse,
            'epoch': epoch + 1,
        })

        if avg_ssim > max_ssim:
            max_ssim = avg_ssim
            torch.save({'epoch': epoch,'generator': generator.state_dict(),'discriminator': discriminator.state_dict(),'g_optimizer': optimizer_G.state_dict(),'d_optimizer': optimizer_D.state_dict(),}, f'{checkpoint_dir}/best_model.pt')
        torch.save({'epoch': epoch,'generator': generator.state_dict(),'discriminator': discriminator.state_dict(),'g_optimizer': optimizer_G.state_dict(),'d_optimizer': optimizer_D.state_dict(),}, f'{checkpoint_dir}/epoch_{epoch+1:03d}.pt')


if __name__ == "__main__":
    run = wandb.init(
        entity="charanvardham",
        project="Super Resolution",
        id="6o6cdyhb",
        resume="must",
        config={
            "learning_rate": 1e-4,
            "architecture": "ESRGAN",
            "epochs": num_epochs,
            "batch_size": 256,
            "betas": (0.5, 0.999),
        },
    )
    #pretrain_generator(num_epochs=pretrain_epochs, generator=generator, optimizer_G=optimizer_G, train_loader=train_loader, criterion=criterion, run=run)
    for pg in optimizer_G.param_groups:
        pg['lr'] = 1e-4
    train(num_epochs=num_epochs, generator=generator, discriminator=discriminator, optimizer_D=optimizer_D, optimizer_G=optimizer_G, train_loader=train_loader, criterion=criterion, run=run)
    run.finish()