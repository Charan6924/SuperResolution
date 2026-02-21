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

def psnr(sr, hr, max_val=1.24):  # 1.24 covers [-0.24, 1.0] range
    """
    Calculate PSNR for data in range [-0.24, 1.0]
    """
    mse = F.mse_loss(sr, hr)
    if mse == 0:
        return torch.tensor(100.0, device=sr.device)
    return 10 * torch.log10((max_val ** 2) / mse)

def ssim(sr, hr, C1=0.01**2, C2=0.03**2):
    """Compute SSIM"""
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
    return


def collate_fn(batch):
    lr_batch = torch.stack([item[0] for item in batch])
    hr_batch = torch.stack([item[1] for item in batch])
    return lr_batch, hr_batch
