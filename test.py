import pyarrow.parquet as pq
import numpy as np
import torch
import os


pq_file = pq.ParquetFile('data/QCDToGGQQ_IMGjet_RH1all_jet0_run0_n36272_LR.parquet')
batch = next(pq_file.iter_batches(batch_size=5000))

lr_raw = batch.column("X_jets_LR").to_pylist()
lr_from_parquet = np.array(lr_raw, dtype=np.float32)

data = torch.load(f"{os.environ['PFSDIR']}/tensors/QCDToGGQQ_IMGjet_RH1all_jet0_run0_n36272_LR_chunk000.pt")
lr_from_tensor = data['lr'].numpy()

print("First 5000 samples - Parquet to numpy:")
print(f"  Shape: {lr_from_parquet.shape}")
print(f"  Range: [{lr_from_parquet.min():.6f}, {lr_from_parquet.max():.6f}]")
print(f"  Mean: {lr_from_parquet.mean():.6f}")

print("\nFirst 5000 samples - Loaded tensor:")
print(f"  Shape: {lr_from_tensor.shape}")
print(f"  Range: [{lr_from_tensor.min():.6f}, {lr_from_tensor.max():.6f}]")
print(f"  Mean: {lr_from_tensor.mean():.6f}")

print(f"\nAre they identical? {np.allclose(lr_from_parquet, lr_from_tensor)}")
print(f"Max absolute difference: {np.abs(lr_from_parquet - lr_from_tensor).max():.10f}")