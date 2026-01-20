import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.amp import autocast, GradScaler # type: ignore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pixel_criterion = nn.MSELoss()
adv_criterion = nn.MSELoss()

# Track training state
training_state = {
    "current_epoch": 0,
    "pretrain_epochs": 10,
    "nan_count": {"generator": 0, "discriminator": 0},
    "best_ssim": 0.0,
    "ssim_degradation_count": 0
}

def psnr(sr, hr, max_val=1.24):  # 1.24 covers [-0.24, 1.0] range
    """
    Calculate PSNR for data in range [-0.24, 1.0]
    """
    mse = F.mse_loss(sr, hr)
    if mse == 0:
        return torch.tensor(100.0, device=sr.device)
    return 10 * torch.log10((max_val ** 2) / mse)

def ssim(sr, hr, C1=0.01**2, C2=0.03**2):
    """Compute SSIM with"""
    eps = 1e-8
    
    mu_x = sr.mean(dim=(-1, -2), keepdim=True)
    mu_y = hr.mean(dim=(-1, -2), keepdim=True)

    sigma_x = ((sr - mu_x) ** 2).mean(dim=(-1, -2), keepdim=True)
    sigma_y = ((hr - mu_y) ** 2).mean(dim=(-1, -2), keepdim=True)
    sigma_xy = ((sr - mu_x) * (hr - mu_y)).mean(dim=(-1, -2), keepdim=True)

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    denominator = torch.clamp(denominator, min=eps)
    
    ssim_map = torch.clamp(numerator / denominator, -1.0, 1.0)
    return ssim_map.mean()

@torch.no_grad()
def validate(generator, val_loader):
    generator.eval()

    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n = 0

    for lr, hr in val_loader:
        lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)

        with autocast('cuda'):
            sr = generator(lr)
            sr = torch.clamp(sr, -0.3, 1.0)
            mse_loss = F.mse_loss(sr, hr, reduction="mean")

        total_mse += mse_loss.item() * lr.size(0)
        total_psnr += psnr(sr, hr).item() * lr.size(0)  # Uses max_val=1.24
        total_ssim += ssim(sr, hr).item() * lr.size(0)
        n += lr.size(0)

    avg_ssim = total_ssim / n

    if avg_ssim > training_state["best_ssim"]:
        training_state["best_ssim"] = avg_ssim
        training_state["ssim_degradation_count"] = 0
    elif avg_ssim < training_state["best_ssim"] - 0.02:
        training_state["ssim_degradation_count"] += 1
    
    return {
        "mse": total_mse / n,
        "psnr": total_psnr / n,
        "ssim": avg_ssim,
    }

def get_adversarial_weight(epoch, pretrain_epochs, warmup_epochs=50):
    if epoch < pretrain_epochs:
        return 0.0
    
    gan_epoch = epoch - pretrain_epochs
    
    # Check for SSIM degradation - pause adversarial training if needed
    if training_state["ssim_degradation_count"] >= 3:
        print("SSIM degrading - reducing adversarial weight")
        return 0.00001 
    
    if gan_epoch < warmup_epochs:
        min_weight = 0.00001
        max_weight = 0.001
        progress = (gan_epoch / warmup_epochs) ** 0.5
        return min_weight + (max_weight - min_weight) * progress
    else:
        return 0.001  

def set_epoch(epoch):
    """Call this at the start of each epoch"""
    training_state["current_epoch"] = epoch

lambda_pixel = 1.0


def train_generator(generator, g_optimizer, discriminator, lr, hr, fake_hr, scaler):
    g_optimizer.zero_grad()
    
    current_epoch = training_state["current_epoch"]
    pretrain_epochs = training_state["pretrain_epochs"]
    lambda_adv = get_adversarial_weight(current_epoch, pretrain_epochs)

    with autocast('cuda'):
        fake_hr_clamped = torch.clamp(fake_hr, -0.3, 1.0)
        if torch.isnan(fake_hr_clamped).any() or torch.isinf(fake_hr_clamped).any():
            print(f"SKIPPING BATCH: NaN/Inf in fake_hr")
            return 0.01, 0.01, 0.0  
        
        if torch.isnan(hr).any() or torch.isinf(hr).any():
            print(f"SKIPPING BATCH: NaN/Inf in hr (data corruption!)")
            return 0.01, 0.01, 0.0

        pixel_loss = pixel_criterion(fake_hr_clamped, hr)
        if current_epoch >= pretrain_epochs:
            pred_fake = discriminator(fake_hr_clamped)

            if torch.isnan(pred_fake).any() or torch.isinf(pred_fake).any():
                print(f"SKIPPING BATCH: NaN/Inf from discriminator")
                return 0.01, 0.01, 0.0
            
            adv_targets = torch.ones_like(pred_fake)
            adv_loss = adv_criterion(pred_fake, adv_targets)
        else:
            adv_loss = torch.tensor(0.0, device=lr.device)
    
        g_loss = lambda_pixel * pixel_loss + lambda_adv * adv_loss
    
    if torch.isnan(g_loss) or torch.isinf(g_loss):
        training_state["nan_count"]["generator"] += 1
        print(f"NaN/Inf in G loss! pixel={pixel_loss.item():.4f}, adv={adv_loss.item():.4f}")
        return 0.01, 0.01, 0.0
    
    if g_loss.item() > 2.0: 
        print(f"EXTREME G loss: {g_loss.item():.4f} - SKIPPING BATCH")
        print(f"   pixel={pixel_loss.item():.4f}, adv={adv_loss.item():.4f}")
        return 0.01, 0.01, 0.0
    
    scaler.scale(g_loss).backward()
    scaler.unscale_(g_optimizer)
    
    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=0.5)
    total_norm = 0.0
    for p in generator.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    if total_norm > 10.0:  
        print(f"Gradient explosion: {total_norm:.2f} - SKIPPING STEP")
        g_optimizer.zero_grad()
        return 0.01, 0.01, 0.0
    
    scaler.step(g_optimizer)
    scaler.update()

    return g_loss.item(), pixel_loss.item(), adv_loss.item() if isinstance(adv_loss, torch.Tensor) else 0.0


def train_discriminator(discriminator, d_optimizer, real_hr, fake_hr, scaler, train_d=True):
    if not train_d:
        return 0.0
    
    d_optimizer.zero_grad()
    
    with autocast('cuda'):
        real_hr = torch.clamp(real_hr, -0.3, 1.0)
        fake_hr = torch.clamp(fake_hr, -0.3, 1.0)
        
        if torch.isnan(real_hr).any() or torch.isinf(real_hr).any():
            print(f"SKIPPING D: NaN/Inf in real_hr")
            return 0.0
        
        if torch.isnan(fake_hr).any() or torch.isinf(fake_hr).any():
            print(f"SKIPPING D: NaN/Inf in fake_hr")
            return 0.0
        
        pred_real = discriminator(real_hr)
        real_targets = torch.ones_like(pred_real) * 0.85  
        loss_real = adv_criterion(pred_real, real_targets)
        
        pred_fake = discriminator(fake_hr.detach())
        fake_targets = torch.zeros_like(pred_fake) + 0.15  
        loss_fake = adv_criterion(pred_fake, fake_targets)
        
        d_loss = 0.5 * (loss_real + loss_fake)
    
    if torch.isnan(d_loss) or torch.isinf(d_loss) or d_loss.item() > 1.0:
        training_state["nan_count"]["discriminator"] += 1
        print(f"Bad D loss: {d_loss.item():.4f}")
        return 0.0
    
    scaler.scale(d_loss).backward()
    scaler.unscale_(d_optimizer)
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=0.5)
    scaler.step(d_optimizer)
    scaler.update()

    return d_loss.item()

def should_train_discriminator(batch_idx, epoch, pretrain_epochs):
    if epoch < pretrain_epochs:
        return False
    
    gan_epoch = epoch - pretrain_epochs

    if gan_epoch < 10:
        return batch_idx % 3 == 0
    elif gan_epoch < 20:
        return batch_idx % 2 == 0
    else:
        return True

def get_current_adv_weight():
    return get_adversarial_weight(
        training_state["current_epoch"], 
        training_state["pretrain_epochs"]
    )

def collate_fn(batch):
    lr_batch = torch.stack([item[0] for item in batch])
    hr_batch = torch.stack([item[1] for item in batch])
    return lr_batch, hr_batch