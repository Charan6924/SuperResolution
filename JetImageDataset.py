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
        val_ratio=0.1,
        normalize=True,
        seed=42,
        hr_max=None,
    ):
        super().__init__()
        all_files = sorted(glob.glob(os.path.join(tensor_dir, "*.pt")))
        all_files = [f for f in all_files if not f.endswith('normalization_stats.pt')]

        if not all_files:
            raise ValueError(f"No .pt files found in {tensor_dir}")

        n = len(all_files)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        
        if split == "train":
            self.tensor_files = all_files[:n_train]
        elif split == "val":
            self.tensor_files = all_files[n_train:n_train + n_val]
        elif split == "test":
            self.tensor_files = all_files[n_train + n_val:]
        else:
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

        self.split     = split
        self.normalize = normalize
        self.seed      = seed

        if normalize:
            if hr_max is None:
                raise RuntimeError(
                    "normalize=True requires hr_max. "
                    "Run calculate_and_save_normalization_stats() and pass stats['hr_p995']."
                )
            self.hr_max = hr_max
        else:
            self.hr_max = None

        end_idx = (n_train if split == "train"
                   else n_train + n_val if split == "val"
                   else n)
        start_idx = (0 if split == "train"
                     else n_train if split == "val"
                     else n_train + n_val)

        print(f"[{split}] {len(self.tensor_files)} files "
              f"(files {start_idx}–{end_idx})")

    def _get_worker_files(self):
        worker = get_worker_info()
        if worker is None:
            return self.tensor_files
        return [f for i, f in enumerate(self.tensor_files)
                if i % worker.num_workers == worker.id]

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = torch.Generator().manual_seed(self.seed + worker_id)

        for file_path in self._get_worker_files():
            try:
                data = torch.load(file_path, map_location='cpu')
                lr   = data['lr'].float()
                hr   = data['hr'].float()
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

            if self.normalize:
                lr = torch.clamp(lr, min=0.0) / self.hr_max #type: ignore
                hr = torch.clamp(hr, min=0.0) / self.hr_max #type: ignore
                lr = torch.clamp(lr, 0.0, 1.0) 
                hr = torch.clamp(hr, 0.0, 1.0)

            n = lr.size(0)
            perm = torch.randperm(n, generator=rng)
            lr = lr[perm]
            hr = hr[perm]

            for i in range(n):
                yield lr[i], hr[i]


def calculate_and_save_normalization_stats(tensor_dir):
    tensor_files = sorted(glob.glob(os.path.join(tensor_dir, "*.pt")))
    tensor_files = [f for f in tensor_files if not f.endswith('normalization_stats.pt')]

    lr_samples, hr_samples = [], []
    lr_abs_max = -float('inf')
    hr_abs_max = -float('inf')
    lr_min = float('inf')
    hr_min = float('inf')
    max_samples_per_file = 100_000

    print(f"Scanning {len(tensor_files)} files...")

    for i, file_path in enumerate(tensor_files):
        if i % 5 == 0:
            print(f"  Progress: {i}/{len(tensor_files)}")
        try:
            data = torch.load(file_path, map_location='cpu')
            lr_abs_max = max(lr_abs_max, data['lr'].max().item())
            hr_abs_max = max(hr_abs_max, data['hr'].max().item())
            lr_min = min(lr_min, data['lr'].min().item())
            hr_min = min(hr_min, data['hr'].min().item())

            lr_nonzero = data['lr'][data['lr'] != 0].abs()
            hr_nonzero = data['hr'][data['hr'] != 0].abs()

            if len(lr_nonzero) > max_samples_per_file:
                lr_nonzero = lr_nonzero[torch.randperm(len(lr_nonzero))[:max_samples_per_file]]
            if len(hr_nonzero) > max_samples_per_file:
                hr_nonzero = hr_nonzero[torch.randperm(len(hr_nonzero))[:max_samples_per_file]]

            if len(lr_nonzero) > 0: lr_samples.append(lr_nonzero)
            if len(hr_nonzero) > 0: hr_samples.append(hr_nonzero)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    all_lr = torch.cat(lr_samples)
    all_hr = torch.cat(hr_samples)

    MAX_Q = 5_000_000
    if len(all_lr) > MAX_Q: all_lr = all_lr[torch.randperm(len(all_lr))[:MAX_Q]]
    if len(all_hr) > MAX_Q: all_hr = all_hr[torch.randperm(len(all_hr))[:MAX_Q]]

    stats = {
        'lr_min':  lr_min,   'lr_max':  lr_abs_max,
        'hr_min':  hr_min,   'hr_max':  hr_abs_max,
        'lr_p995': torch.quantile(all_lr, 0.995).item(),
        'hr_p995': torch.quantile(all_hr, 0.995).item(),
        'lr_p99':  torch.quantile(all_lr, 0.99).item(),
        'hr_p99':  torch.quantile(all_hr, 0.99).item(),
        'lr_median': torch.median(all_lr).item(),
        'hr_median': torch.median(all_hr).item(),
    }

    stats_path = os.path.join(tensor_dir, 'normalization_stats.pt')
    torch.save(stats, stats_path)
    print(f"\nSaved stats to {stats_path}")
    print(f"hr_p995 = {stats['hr_p995']:.6f}")
    return stats

