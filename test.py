import torch
import torch.nn as nn
from Generator import Generator
from JetImageDataset import JetImageDataset
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
tensor_dir = 'data/pt_tensors'
hr_max = torch.load(f'{tensor_dir}/normalization_stats.pt')['hr_p995']
val_dataset = JetImageDataset(tensor_dir=tensor_dir, split="val", train_ratio=0.8, hr_max=hr_max)
val_loader = DataLoader(val_dataset, batch_size=4, num_workers=0)

# Load model
generator = Generator().to(device)
checkpoint = torch.load('checkpoints/pretrain_final.pt', map_location=device)
generator.load_state_dict(checkpoint['generator'])
print("Loaded pretrained generator\n")

# Get a batch
lr, hr = next(iter(val_loader))
lr, hr = lr.to(device), hr.to(device)

# === DEBUG 1: Check generator output statistics ===
print("=" * 50)
print("DEBUG 1: Generator output statistics")
print("=" * 50)
generator.eval()
with torch.no_grad():
    sr = generator(lr)

print(f"SR output - min: {sr.min().item():.6f}, max: {sr.max().item():.6f}, mean: {sr.mean().item():.6f}, std: {sr.std().item():.6f}")
print(f"HR target - min: {hr.min().item():.6f}, max: {hr.max().item():.6f}, mean: {hr.mean().item():.6f}, std: {hr.std().item():.6f}")
print(f"LR input  - min: {lr.min().item():.6f}, max: {lr.max().item():.6f}, mean: {lr.mean().item():.6f}, std: {lr.std().item():.6f}")

if sr.std().item() < 1e-5:
    print("\n⚠️  WARNING: Generator outputting near-constant values!")

# === DEBUG 2: Check gradient flow ===
print("\n" + "=" * 50)
print("DEBUG 2: Gradient flow check")
print("=" * 50)
generator.train()
optimizer = torch.optim.AdamW(generator.parameters(), lr=1e-4)
optimizer.zero_grad()

sr = generator(lr)
loss = nn.L1Loss()(sr, hr)
loss.backward()

print(f"Loss value: {loss.item():.10f}")

grad_stats = []
for name, p in generator.named_parameters():
    if p.grad is not None:
        grad_mean = p.grad.abs().mean().item()
        grad_stats.append((name, grad_mean))

print(f"\nGradient stats (first 5 layers):")
for name, grad_mean in grad_stats[:5]:
    status = "✓" if grad_mean > 1e-8 else "⚠️ ZERO"
    print(f"  {name}: {grad_mean:.2e} {status}")

if all(g < 1e-8 for _, g in grad_stats):
    print("\n⚠️  WARNING: All gradients are near zero!")

# === DEBUG 3: Check if weights update ===
print("\n" + "=" * 50)
print("DEBUG 3: Weight update check")
print("=" * 50)
w_before = generator.head.weight.data.clone()
optimizer.step()
w_after = generator.head.weight.data
weight_diff = (w_after - w_before).abs().max().item()

print(f"Max weight change in head layer: {weight_diff:.2e}")
if weight_diff < 1e-10:
    print("⚠️  WARNING: Weights not updating!")
else:
    print("✓ Weights are updating")

# === DEBUG 4: Check for NaN/Inf ===
print("\n" + "=" * 50)
print("DEBUG 4: NaN/Inf check")
print("=" * 50)
has_nan = torch.isnan(sr).any().item()
has_inf = torch.isinf(sr).any().item()
print(f"Output has NaN: {has_nan}")
print(f"Output has Inf: {has_inf}")

# === DEBUG 5: Per-channel statistics ===
print("\n" + "=" * 50)
print("DEBUG 5: Per-channel output statistics")
print("=" * 50)
for c in range(sr.shape[1]):
    ch = sr[:, c, :, :]
    print(f"Channel {c}: min={ch.min().item():.6f}, max={ch.max().item():.6f}, std={ch.std().item():.6f}")
