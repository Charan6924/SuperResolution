import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.data import get_worker_info
import torch.optim 
import torch.nn.functional as F
from utils import validate, train_generator, train_discriminator
from Generator import Generator
from Discriminator import Discriminator
from JetImageDataset import JetImageDataset
from EarlyStopping import EarlyStopping

all_files = []
device = 'cuda' if torch.cuda.is_available() else 'cpu'
generator = Generator().to(device)
checkpoint = torch.load("/content/sr_model_best.pth")
discriminator = Discriminator().to(device)
pixel_criterion = nn.MSELoss()
adv_criterion = nn.MSELoss()
g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.9, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.9, 0.999))
g_lossi = []
d_lossi = []
epochs = 100
val_mse, val_psnr, val_ssim = [], [], []
early_stopper = EarlyStopping(
    patience=8,
    min_delta=1e-4
)
best_val_mse = float("inf")
generator = torch.compile(generator)
generator.load_state_dict(checkpoint)
discriminator = torch.compile(discriminator)

train_dataset = JetImageDataset(
    parquet_files=all_files,
    split='train',
    train_ratio=0.8,
    chunk_size=512,
    normalize=False
)

val_dataset = JetImageDataset(
    parquet_files=all_files,
    split='val',
    train_ratio=0.8,
    chunk_size=512,
    normalize=False
)

train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, pin_memory=True,prefetch_factor = 2,persistent_workers=True,  )
val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4, pin_memory=True, prefetch_factor = 2,persistent_workers=True,)

for epoch in range(epochs):
    generator.train()
    discriminator.train()
    g_epoch_loss = 0.0
    d_epoch_loss = 0.0
    train_count = 0 

    for lr, hr in train_loader:
        lr, hr = lr.to(device), hr.to(device)

        fake_hr = generator(lr)
        d_loss = train_discriminator(discriminator, d_optimizer,hr, fake_hr.detach())
        g_loss, p_loss, a_loss = train_generator(generator, g_optimizer,discriminator, lr, hr,fake_hr)
        g_epoch_loss += g_loss
        train_count += 1
        d_epoch_loss += d_loss

    g_loss = g_epoch_loss / train_count
    d_loss = d_epoch_loss / train_count
    with torch.no_grad():
      metrics = validate(generator, val_loader)
    if metrics["mse"] < best_val_mse:
        best_val_mse = metrics["mse"]
        torch.save(
            {
                "epoch": epoch,
                "generator": generator.state_dict(),
                "val_mse": metrics["mse"],
            },
            "best_generator.pt"
        )

    val_mse.append(metrics["mse"])
    val_psnr.append(metrics["psnr"])
    val_ssim.append(metrics["ssim"])
    g_lossi.append(g_loss)
    d_lossi.append(d_loss)

    print(
        f"Epoch {epoch+1:03d} | "
        f"G Loss: {g_loss:.4f} | "
        f"D Loss: {d_loss:.4f} | "
        f"Val MSE: {metrics['mse']:.6f} | "
        f"PSNR: {metrics['psnr']:.2f} dB | "
        f"SSIM: {metrics['ssim']:.4f}"
    )

    early_stopper.step(metrics["mse"])

    if early_stopper.should_stop:
        print("Early stopping triggered")
        break