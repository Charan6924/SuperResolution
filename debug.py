# debug_parquet.py
import pyarrow.parquet as pq
import numpy as np
import sys

# Use your actual file path
FILE_PATH = 'data/QCDToGGQQ_IMGjet_RH1all_jet0_run0_n36272_LR.parquet'

print(f"--- Inspecting {FILE_PATH} ---")

try:
    parquet_file = pq.ParquetFile(FILE_PATH)
    print(f"Metadata: {parquet_file.metadata}")
    print(f"Schema names: {parquet_file.schema.names}")
    
    # Try to load just ONE row
    print("\nAttempting to load 1 row...")
    batch = next(parquet_file.iter_batches(batch_size=1, columns=['X_jets_LR']))
    
    # Convert to python list first (safest way to see structure)
    val = batch.column('X_jets_LR').to_pylist()[0]
    
    print(f"Top level type: {type(val)}")
    if isinstance(val, list):
        print(f"Top level length: {len(val)}")
        print(f"Index 0 type: {type(val[0])}")
        if isinstance(val[0], list):
             print(f"Index 0 length: {len(val[0])}")
             print(f"Index 0,0 type: {type(val[0][0])}")
    
    print("\nSUCCESS: File is readable.")

except Exception as e:
    print(f"\nCRITICAL FAIL: {e}")