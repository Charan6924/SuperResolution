import pyarrow.parquet as pq
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

class JetImageDataset(IterableDataset):
    def __init__(
        self,
        parquet_files,
        split="train",
        train_ratio=0.8,
        chunk_size=512,
        normalize=True,
        seed=42,
    ):
        super().__init__()
        self.parquet_files = parquet_files
        self.split = split
        self.train_ratio = train_ratio
        self.chunk_size = chunk_size
        self.normalize = normalize
        self.seed = seed

        self.lr_col = "X_jets_LR"
        self.hr_col = "X_jets"

        self.lr_shape = (3, 64, 64)
        self.hr_shape = (3, 125, 125)

        assert split in {"train", "val"}
    def _get_worker_files(self):
        worker = get_worker_info()
        if worker is None:
            return self.parquet_files
        return self.parquet_files[worker.id :: worker.num_workers]
    def _convert_chunk(self, batch):
      lr_col = batch.column(self.lr_col).to_numpy(zero_copy_only=False)
      hr_col = batch.column(self.hr_col).to_numpy(zero_copy_only=False)

      lr_out = np.empty((len(lr_col), 3, 64, 64), dtype=np.float32)
      hr_out = np.empty((len(hr_col), 3, 125, 125), dtype=np.float32)

      for i in range(len(lr_col)):
          # LR
          for c in range(3):
              lr_out[i, c] = np.stack(lr_col[i][c], axis=0)

          # HR
          for c in range(3):
              hr_out[i, c] = np.stack(hr_col[i][c], axis=0)

      lr = torch.from_numpy(lr_out)
      hr = torch.from_numpy(hr_out)

      if self.normalize:
          lr /= 255.0
          hr /= 255.0
      return lr, hr
    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        files = self._get_worker_files()

        for file_path in files:
            parquet = pq.ParquetFile(file_path)

            for batch in parquet.iter_batches(
                batch_size=self.chunk_size,
                columns=[self.lr_col, self.hr_col],
            ):
                lr, hr = self._convert_chunk(batch)
                mask = rng.random(len(lr)) < self.train_ratio
                if self.split == "train":
                    idxs = torch.where(torch.from_numpy(mask))[0]
                else:
                    idxs = torch.where(torch.from_numpy(~mask))[0]

                for i in idxs:
                    yield lr[i], hr[i]