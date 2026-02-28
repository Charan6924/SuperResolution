import pyarrow.parquet as pq
import numpy as np
import torch
from tqdm import tqdm
import os

parquet_files = [
    '/mnt/vstor/courses/csds312/cxv166/assignment/datascience/SuperResolution/data/QCDToGGQQ_IMGjet_RH1all_jet0_run0_n36272_LR.parquet',
    '/mnt/vstor/courses/csds312/cxv166/assignment/datascience/SuperResolution/data/QCDToGGQQ_IMGjet_RH1all_jet0_run1_n47540_LR.parquet',
    '/mnt/vstor/courses/csds312/cxv166/assignment/datascience/SuperResolution/data/QCDToGGQQ_IMGjet_RH1all_jet0_run2_n55494_LR.parquet'
]

output_dir = "data/pt_tensors"
os.makedirs(output_dir, exist_ok=True)
rows_per_chunk = 500

print(f"Processing {rows_per_chunk} rows at a time")
print(f"Saving to: {output_dir}")

for file_idx, file_path in enumerate(parquet_files):
    print(f"\n[File {file_idx+1}/{len(parquet_files)}] {file_path}")
    
    parquet_file = pq.ParquetFile(file_path)
    total_rows = parquet_file.metadata.num_rows
    num_chunks = (total_rows + rows_per_chunk - 1) // rows_per_chunk
    
    print(f"  Total rows: {total_rows:,}")
    print(f"  Will create {num_chunks} chunk files")
    
    chunk_idx = 0
    for batch in tqdm(parquet_file.iter_batches(batch_size=rows_per_chunk, columns=["X_jets_LR", "X_jets"]),total=num_chunks, desc=f"  File {file_idx+1}"):
        lr_raw = batch.column("X_jets_LR").to_pylist()
        hr_raw = batch.column("X_jets").to_pylist()
        lr_tensor = torch.from_numpy(np.array(lr_raw, dtype=np.float32))
        hr_tensor = torch.tensor(hr_raw, dtype=torch.float32)
        del lr_raw, hr_raw
        base_name = os.path.basename(file_path).replace('.parquet', '')
        output_file = os.path.join(output_dir, f'{base_name}_chunk{chunk_idx:03d}.pt')
        torch.save({'lr': lr_tensor,'hr': hr_tensor,'num_samples': len(lr_tensor)}, output_file)
        del lr_tensor, hr_tensor
        chunk_idx += 1
    
    print(f"created {chunk_idx} chunks for this file")

print(f"Output directory: {output_dir}")
pt_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.pt')])
total_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in pt_files)
print(f"Total files created: {len(pt_files)}")
print(f"Total size: {total_size / 1e9:.2f} GB")