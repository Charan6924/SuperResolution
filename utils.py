import torch
import torch.nn.functional as F
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pixel_criterion = nn.MSELoss()
adv_criterion = nn.MSELoss()

def psnr(sr, hr, max_val=1.0):
    mse = F.mse_loss(sr, hr)
    if mse == 0:
        return torch.tensor(100.0, device=sr.device)
    return 10 * torch.log10((max_val ** 2) / mse)

def ssim(sr, hr, C1=0.01**2, C2=0.03**2):
    mu_x = sr.mean(dim=(-1, -2), keepdim=True)
    mu_y = hr.mean(dim=(-1, -2), keepdim=True)

    sigma_x = ((sr - mu_x) ** 2).mean(dim=(-1, -2), keepdim=True)
    sigma_y = ((hr - mu_y) ** 2).mean(dim=(-1, -2), keepdim=True)
    sigma_xy = ((sr - mu_x) * (hr - mu_y)).mean(dim=(-1, -2), keepdim=True)

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x ** 2 + mu_y ** 2 + C1) *
        (sigma_x + sigma_y + C2)
    )

    return ssim_map.mean()

@torch.no_grad()
def validate(generator, val_loader):
    generator.eval()

    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n = 0

    for lr, hr in val_loader:
        lr, hr = lr.to(device), hr.to(device)

        sr = generator(lr)

        mse_loss = F.mse_loss(sr, hr, reduction="mean")
        total_mse += mse_loss.item() * lr.size(0)

        total_psnr += psnr(sr, hr).item() * lr.size(0)
        total_ssim += ssim(sr, hr).item() * lr.size(0)

        n += lr.size(0)

    return {
        "mse": total_mse / n,
        "psnr": total_psnr / n,
        "ssim": total_ssim / n,
    }

lambda_pixel = 1.0
lambda_adv = 0.001
def train_generator(generator, g_optimizer,discriminator, lr, hr,fake_hr):
    g_optimizer.zero_grad()

    pixel_loss = pixel_criterion(fake_hr, hr)
    pred_fake = discriminator(fake_hr)
    adv_targets = torch.ones_like(pred_fake)
    adv_loss = adv_criterion(pred_fake, adv_targets)

    g_loss = lambda_pixel * pixel_loss + lambda_adv * adv_loss
    g_loss.backward()
    g_optimizer.step()

    return g_loss.item(), pixel_loss.item(), adv_loss.item()

def train_discriminator(discriminator, d_optimizer,real_hr, fake_hr):
    d_optimizer.zero_grad()
    pred_real = discriminator(real_hr)
    real_targets = torch.ones_like(pred_real)
    loss_real = adv_criterion(pred_real, real_targets)

    pred_fake = discriminator(fake_hr.detach())
    fake_targets = torch.zeros_like(pred_fake)
    loss_fake = adv_criterion(pred_fake, fake_targets)

    d_loss = 0.5 * (loss_real + loss_fake)
    d_loss.backward()
    d_optimizer.step()

    return d_loss.item()

def collate_fn(batch):
    lr_batch = torch.stack([item[0] for item in batch])
    hr_batch = torch.stack([item[1] for item in batch])
    return lr_batch, hr_batch