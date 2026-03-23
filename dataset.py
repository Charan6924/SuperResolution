import torch
from torch.utils.data import IterableDataset, get_worker_info
import glob
import os


class JetImageDataset(IterableDataset):
    def __init__(self, tensor_dir, split="train", train_ratio=0.8, val_ratio=0.1, hr_max=None, seed=42):
        super().__init__()
        all_files = sorted(glob.glob(os.path.join(tensor_dir, "*.pt")))
        all_files = [f for f in all_files if not f.endswith('normalization_stats.pt')]

        if not all_files:
            raise ValueError(f"No .pt files found in {tensor_dir}")

        n = len(all_files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        if split == "train":
            self.files = all_files[:n_train]
        elif split == "val":
            self.files = all_files[n_train:n_train + n_val]
        elif split == "test":
            self.files = all_files[n_train + n_val:]
        else:
            raise ValueError(f"Invalid split: {split}")

        self.hr_max = hr_max
        self.seed = seed
        print(f"[{split}] {len(self.files)} files")

    def _get_worker_files(self):
        worker = get_worker_info()
        if worker is None:
            return self.files
        return [f for i, f in enumerate(self.files) if i % worker.num_workers == worker.id]

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        rng = torch.Generator().manual_seed(self.seed + worker_id)

        for path in self._get_worker_files():
            try:
                data = torch.load(path, map_location='cpu')
                lr, hr = data['lr'].float(), data['hr'].float()
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue

            if self.hr_max:
                lr = torch.clamp(lr / self.hr_max, 0, 1)
                hr = torch.clamp(hr / self.hr_max, 0, 1)

            n = lr.size(0)
            perm = torch.randperm(n, generator=rng)
            lr, hr = lr[perm], hr[perm]

            for i in range(n):
                yield lr[i], hr[i]


def calculate_and_save_normalization_stats(tensor_dir):
    files = sorted(glob.glob(os.path.join(tensor_dir, "*.pt")))
    files = [f for f in files if not f.endswith('normalization_stats.pt')]

    lr_samples, hr_samples = [], []
    lr_max, hr_max = -float('inf'), -float('inf')
    max_per_file = 100_000

    print(f"Scanning {len(files)} files...")

    for i, path in enumerate(files):
        if i % 5 == 0:
            print(f"  {i}/{len(files)}")
        try:
            data = torch.load(path, map_location='cpu')
            lr_max = max(lr_max, data['lr'].max().item())
            hr_max = max(hr_max, data['hr'].max().item())

            lr_nz = data['lr'][data['lr'] != 0].abs()
            hr_nz = data['hr'][data['hr'] != 0].abs()

            if len(lr_nz) > max_per_file:
                lr_nz = lr_nz[torch.randperm(len(lr_nz))[:max_per_file]]
            if len(hr_nz) > max_per_file:
                hr_nz = hr_nz[torch.randperm(len(hr_nz))[:max_per_file]]

            if len(lr_nz) > 0:
                lr_samples.append(lr_nz)
            if len(hr_nz) > 0:
                hr_samples.append(hr_nz)
        except Exception as e:
            print(f"Error: {path}: {e}")

    all_lr = torch.cat(lr_samples)
    all_hr = torch.cat(hr_samples)

    max_q = 5_000_000
    if len(all_lr) > max_q:
        all_lr = all_lr[torch.randperm(len(all_lr))[:max_q]]
    if len(all_hr) > max_q:
        all_hr = all_hr[torch.randperm(len(all_hr))[:max_q]]

    stats = {
        'lr_max': lr_max,
        'hr_max': hr_max,
        'lr_p995': torch.quantile(all_lr, 0.995).item(),
        'hr_p995': torch.quantile(all_hr, 0.995).item(),
    }

    stats_path = os.path.join(tensor_dir, 'normalization_stats.pt')
    torch.save(stats, stats_path)
    print(f"Saved to {stats_path}, hr_p995={stats['hr_p995']:.6f}")
    return stats
