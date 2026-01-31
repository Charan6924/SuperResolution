import torch
from torch.utils.data import IterableDataset, get_worker_info
import glob
import os

class JetImageDataset(IterableDataset):
    def __init__(
        self,
        tensor_dir,
        split="train",
        train_ratio=0.8,
        normalize=True,
        seed=42,
        max_batch_size=256,
        lr_max=None,
        hr_max=None
    ):
        super().__init__()
        self.tensor_files = sorted(glob.glob(os.path.join(tensor_dir, "*.pt")))
        self.tensor_files = [f for f in self.tensor_files if not f.endswith('normalization_stats.pt')]
        if not self.tensor_files:
            raise ValueError(f"No .pt files found in {tensor_dir}")
        
        self.split = split
        self.train_ratio = train_ratio
        self.normalize = normalize
        self.seed = seed
        self.max_batch_size = max_batch_size

        if normalize:
            if lr_max is None or hr_max is None:
                raise RuntimeError(
            "Normalization enabled but lr_max/hr_max not provided. "
            "Run calculate_and_save_normalization_stats() once and pass the values.")
            self.lr_max = lr_max
            self.hr_max = hr_max
        else:
            self.lr_max = lr_max
            self.hr_max = hr_max
        
        print(f"Found {len(self.tensor_files)} tensor files")

    def _calculate_global_stats(self):
        all_lr_vals = []
        all_hr_vals = []
        
        for file_path in self.tensor_files:
            try:
                data = torch.load(file_path, map_location='cpu')
                lr_nonzero = data['lr'][data['lr'] != 0]
                hr_nonzero = data['hr'][data['hr'] != 0]
                
                if len(lr_nonzero) > 0:
                    all_lr_vals.append(lr_nonzero.abs())
                if len(hr_nonzero) > 0:
                    all_hr_vals.append(hr_nonzero.abs())
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

        all_lr = torch.cat(all_lr_vals)
        all_hr = torch.cat(all_hr_vals)
        
        lr_max = torch.quantile(all_lr, 0.995).item()
        hr_max = torch.quantile(all_hr, 0.995).item()
        
        print(f"  Using 99.5th percentile for normalization")
        print(f"  (This clips ~0.5% of extreme outliers)")
        
        return lr_max, hr_max

    def _get_worker_files(self):
        worker = get_worker_info()
        if worker is None:
            return self.tensor_files
        return [f for i, f in enumerate(self.tensor_files) if i % worker.num_workers == worker.id]

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = torch.Generator().manual_seed(self.seed + worker_id)
        files = self._get_worker_files()
        
        for file_path in files:
            try:
                data = torch.load(file_path, map_location='cpu')
                lr = data['lr']  # [N, 3, 25, 25]
                hr = data['hr']  # [N, 3, 125, 125]
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

            batch_size = lr.size(0)

            if self.normalize:
                lr = torch.clamp(lr / self.lr_max, -1.0, 1.0)
                hr = torch.clamp(hr / self.hr_max, -1.0, 1.0)

            indices = torch.randperm(batch_size, generator=rng)
            split_idx = int(batch_size * self.train_ratio)
            
            if self.split == "train":
                indices = indices[:split_idx]
            else:
                indices = indices[split_idx:]
            
            if len(indices) == 0:
                continue
                
            lr = lr[indices]
            hr = hr[indices]
            
            for i in range(0, len(lr), self.max_batch_size):
                yield lr[i:i+self.max_batch_size], hr[i:i+self.max_batch_size]


def calculate_and_save_normalization_stats(tensor_dir):
    tensor_files = sorted(glob.glob(os.path.join(tensor_dir, "*.pt")))
    tensor_files = [f for f in tensor_files if not f.endswith('normalization_stats.pt')]
    lr_samples = []
    hr_samples = []
    max_samples_per_file = 100000 
    
    print(f"Scanning {len(tensor_files)} files...")
    
    lr_abs_max = -float('inf')
    hr_abs_max = -float('inf')
    lr_min = float('inf')
    hr_min = float('inf')
    
    for i, file_path in enumerate(tensor_files):
        if i % 5 == 0:
            print(f"Progress: {i}/{len(tensor_files)}")
        
        try:
            data = torch.load(file_path, map_location='cpu')
            lr_abs_max = max(lr_abs_max, data['lr'].max().item())
            hr_abs_max = max(hr_abs_max, data['hr'].max().item())
            lr_min = min(lr_min, data['lr'].min().item())
            hr_min = min(hr_min, data['hr'].min().item())
            lr_nonzero = data['lr'][data['lr'] != 0].abs()
            hr_nonzero = data['hr'][data['hr'] != 0].abs()
            
            if len(lr_nonzero) > max_samples_per_file:
                indices = torch.randperm(len(lr_nonzero))[:max_samples_per_file]
                lr_nonzero = lr_nonzero[indices]
            
            if len(hr_nonzero) > max_samples_per_file:
                indices = torch.randperm(len(hr_nonzero))[:max_samples_per_file]
                hr_nonzero = hr_nonzero[indices]
            
            if len(lr_nonzero) > 0:
                lr_samples.append(lr_nonzero)
            if len(hr_nonzero) > 0:
                hr_samples.append(hr_nonzero)
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    print("Calculating percentiles from samples...")
    all_lr = torch.cat(lr_samples)
    all_hr = torch.cat(hr_samples)
    
    print(f"  LR: {len(all_lr):,} samples")
    print(f"  HR: {len(all_hr):,} samples")
    
    lr_p995 = torch.quantile(all_lr, 0.995).item()
    hr_p995 = torch.quantile(all_hr, 0.995).item()
    lr_p99 = torch.quantile(all_lr, 0.99).item()
    hr_p99 = torch.quantile(all_hr, 0.99).item()
    lr_median = torch.median(all_lr).item()
    hr_median = torch.median(all_hr).item()
    
    stats = {
        'lr_min': lr_min,
        'lr_max': lr_abs_max,
        'hr_min': hr_min,
        'hr_max': hr_abs_max,
        'lr_p995': lr_p995,  
        'hr_p995': hr_p995,
        'lr_p99': lr_p99,
        'hr_p99': hr_p99,
        'lr_median': lr_median,
        'hr_median': hr_median,
    }
    
    stats_path = os.path.join(tensor_dir, 'normalization_stats.pt')
    torch.save(stats, stats_path)
    
    print("\n" + "="*60)
    print("Global Statistics:")
    print(f"  LR range: [{lr_min:.6f}, {lr_abs_max:.6f}]")
    print(f"  HR range: [{hr_min:.6f}, {hr_abs_max:.6f}]")
    print(f"\n  LR median: {lr_median:.6f}")
    print(f"  HR median: {hr_median:.6f}")
    print(f"\n  LR 99th percentile: {lr_p99:.6f}")
    print(f"  HR 99th percentile: {hr_p99:.6f}")
    print(f"\n  LR 99.5th percentile: {lr_p995:.6f}")
    print(f"  HR 99.5th percentile: {hr_p995:.6f}")
    print(f"\n   WARNING: Your max values are HUGE!")
    print(f"     LR max: {lr_abs_max:.1f} (vs p99.5: {lr_p995:.1f})")
    print(f"     HR max: {hr_abs_max:.1f} (vs p99.5: {hr_p995:.1f})")
    print(f"\n  RECOMMENDATION: Use p99 or p99.5 for normalization")
    print(f"    lr_max={lr_p995:.6f}")
    print(f"    hr_max={hr_p995:.6f}")
    print(f"\nSaved to: {stats_path}")
    print("="*60)
    
    return stats


if __name__ == "__main__":
    tensor_dir = "data/pt_tensors"
    stats = calculate_and_save_normalization_stats(tensor_dir)
    train_dataset = JetImageDataset(
        tensor_dir=tensor_dir,
        split="train",
        train_ratio=0.8,
        max_batch_size=256,
        lr_max=stats['lr_p995'],  
        hr_max=stats['hr_p995']
    )
    
    val_dataset = JetImageDataset(
        tensor_dir=tensor_dir,
        split="val",
        train_ratio=0.8,
        max_batch_size=256,
        lr_max=stats['lr_p995'],  
        hr_max=stats['hr_p995']
    )
    
    print("\nTesting data loading:")
    for lr_batch, hr_batch in train_dataset:
        print(f"LR batch shape: {lr_batch.shape}")
        print(f"HR batch shape: {hr_batch.shape}")
        print(f"LR range: [{lr_batch.min():.6f}, {lr_batch.max():.6f}]")
        print(f"HR range: [{hr_batch.min():.6f}, {hr_batch.max():.6f}]")
        break