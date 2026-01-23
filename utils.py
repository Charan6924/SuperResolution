import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.amp import autocast, GradScaler # type: ignore
import logging
import numpy as np

logger = logging.getLogger(__name__)

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


def train_generator(generator, g_optimizer, discriminator, lr, hr, fake_hr, scaler, lambda_adv):
    g_optimizer.zero_grad()

    with autocast('cuda'):
        fake_hr_clamped = torch.clamp(fake_hr, -0.3, 1.0)
        if torch.isnan(fake_hr_clamped).any() or torch.isinf(fake_hr_clamped).any():
            print(f"SKIPPING BATCH: NaN/Inf in fake_hr")
            return 0.01, 0.01, 0.0  
        
        if torch.isnan(hr).any() or torch.isinf(hr).any():
            print(f"SKIPPING BATCH: NaN/Inf in hr (data corruption!)")
            return 0.01, 0.01, 0.0

        pixel_loss = pixel_criterion(fake_hr_clamped, hr)
        
        # USE lambda_adv TO DETERMINE IF WE'RE IN GAN TRAINING
        # If lambda_adv > 0, we're past pretraining
        if lambda_adv > 0:  # CHANGED THIS LINE
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

class AdversarialScheduler:
    def __init__(
        self, 
        pretrain_epochs=10,
        warmup_epochs=50,
        min_weight=0.00001,
        max_weight=0.001,
        ssim_threshold=0.05,
        patience=3,
        reduction_factor=0.1
    ):
        self.pretrain_epochs = pretrain_epochs
        self.warmup_epochs = warmup_epochs
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.ssim_threshold = ssim_threshold
        self.patience = patience
        self.reduction_factor = reduction_factor
        
        self.current_epoch = 0
        self.ssim_history = []
        self.degradation_count = 0
        self.current_weight = 0.0
        self.weight_multiplier = 1.0
        
    def step(self, epoch, current_ssim=None):
        """Update adversarial weight for the current epoch."""
        self.current_epoch = epoch
        
        if current_ssim is not None:
            self._update_ssim_monitoring(current_ssim)
        
        base_weight = self._calculate_base_weight(epoch)
        self.current_weight = base_weight * self.weight_multiplier
        
        return self.current_weight
    
    def _calculate_base_weight(self, epoch):
        """Calculate base adversarial weight without SSIM adjustment."""
        if epoch < self.pretrain_epochs:
            return 0.0
        
        gan_epoch = epoch - self.pretrain_epochs
        
        if gan_epoch < self.warmup_epochs:
            progress = gan_epoch / self.warmup_epochs
            progress = progress ** 0.5  # Square root for smoother warmup
            return self.min_weight + (self.max_weight - self.min_weight) * progress
        
        return self.max_weight
    
    def _update_ssim_monitoring(self, current_ssim):
        """Monitor SSIM and adjust weight multiplier if degrading."""
        self.ssim_history.append(current_ssim)
        
        if len(self.ssim_history) < 2:
            return
        
        ssim_drop = self.ssim_history[-2] - current_ssim
        
        if ssim_drop > self.ssim_threshold:
            self.degradation_count += 1
            logger.warning(f"SSIM dropped by {ssim_drop:.4f} (count: {self.degradation_count}/{self.patience})")
            
            if self.degradation_count >= self.patience:
                old_multiplier = self.weight_multiplier
                self.weight_multiplier *= self.reduction_factor
                logger.warning(f"Reducing adversarial weight: {old_multiplier:.6f} → {self.weight_multiplier:.6f}")
                self.degradation_count = 0
        else:
            if self.degradation_count > 0:
                self.degradation_count = max(0, self.degradation_count - 1)
            
            if current_ssim > 0.7 and self.weight_multiplier < 1.0:
                self.weight_multiplier = min(1.0, self.weight_multiplier * 1.1)
    
    def get_weight(self):
        """Get current adversarial weight."""
        return self.current_weight
    
    def state_dict(self):
        """Get scheduler state for checkpointing."""
        return {
            'current_epoch': self.current_epoch,
            'ssim_history': self.ssim_history,
            'degradation_count': self.degradation_count,
            'current_weight': self.current_weight,
            'weight_multiplier': self.weight_multiplier,
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state from checkpoint."""
        self.current_epoch = state_dict.get('current_epoch', 0)
        self.ssim_history = state_dict.get('ssim_history', [])
        self.degradation_count = state_dict.get('degradation_count', 0)
        self.current_weight = state_dict.get('current_weight', 0.0)
        self.weight_multiplier = state_dict.get('weight_multiplier', 1.0)

class GAN_LRScheduler:
    def __init__(self, optimizer_G, optimizer_D, config):
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.base_lr_G = config.get('lr_G', 1e-4)
        self.base_lr_D = config.get('lr_D', 1e-5)
        self.pretrain_epochs = config.get('pretrain_epochs', 10)
        self.total_epochs = config.get('total_epochs', 200)
        self.warmup_epochs = config.get('warmup_epochs', 5)
        self.d_loss_history = []
        self.g_loss_history = []
        self.patience_counter = 0
        
    def step(self, epoch, d_loss, g_loss, lambda_adv):        
        self.d_loss_history.append(d_loss)
        self.g_loss_history.append(g_loss)
        
        if epoch < self.pretrain_epochs:
            lr_G = self.base_lr_G
            lr_D = 0 
            
        else:
            gan_epoch = epoch - self.pretrain_epochs
            gan_total = self.total_epochs - self.pretrain_epochs
            progress = gan_epoch / gan_total
            adv_factor = 1.0 / (1.0 + lambda_adv * 100)
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            if gan_epoch < self.warmup_epochs:
                warmup_factor = (gan_epoch + 1) / self.warmup_epochs
            else:
                warmup_factor = 1.0
            
            lr_G = self.base_lr_G * adv_factor * cosine_factor * warmup_factor
            if len(self.d_loss_history) >= 3:
                recent_d_loss = np.mean(self.d_loss_history[-3:])
                
                if recent_d_loss < 0.01:  # D too strong
                    d_strength_factor = 0.5
                elif recent_d_loss > 0.05:  # D too weak
                    d_strength_factor = 2.0
                else:
                    d_strength_factor = 1.0
            else:
                d_strength_factor = 1.0
            
            lr_D = self.base_lr_D * d_strength_factor * cosine_factor * warmup_factor
            lr_G = np.clip(lr_G, 1e-7, self.base_lr_G)
            lr_D = np.clip(lr_D, 1e-7, self.base_lr_D * 2)
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr_G
        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr_D
        
        return lr_G, lr_D