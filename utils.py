import torch
import torch.nn.functional as F


def psnr(sr, hr, max_val=1.0):
    mse = F.mse_loss(sr, hr)
    if mse == 0:
        return torch.tensor(100.0, device=sr.device)
    return 10 * torch.log10(max_val ** 2 / mse)


def _gaussian_kernel(size=11, sigma=1.5, channels=3, device='cpu'):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g.outer(g)
    kernel = kernel.view(1, 1, size, size).repeat(channels, 1, 1, 1)
    return kernel


def ssim(sr, hr, window_size=11, sigma=1.5):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    channels = sr.size(1)
    kernel = _gaussian_kernel(window_size, sigma, channels, sr.device)

    mu_x = F.conv2d(sr, kernel, padding=window_size // 2, groups=channels)
    mu_y = F.conv2d(hr, kernel, padding=window_size // 2, groups=channels)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(sr ** 2, kernel, padding=window_size // 2, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(hr ** 2, kernel, padding=window_size // 2, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(sr * hr, kernel, padding=window_size // 2, groups=channels) - mu_xy

    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)

    ssim_map = num / den.clamp(min=1e-8)
    return ssim_map.mean()
