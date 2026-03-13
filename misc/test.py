# Load and inspect cache
import pandas as pd
import numpy as np

# Check metadata
meta_df = pd.read_parquet("/home/cs21d002_eashaan/PhD/Objective1/data/processed/cache/huggingface_transformers_target_train_meta.parquet")
print(f"Total samples in source cache: {len(meta_df)}")
print(f"Unique bugs: {meta_df['bug_id'].nunique()}")
print(f"Unique blobs: {meta_df['blob_sha'].nunique()}")
print(f"Avg candidates per bug: {len(meta_df) / meta_df['bug_id'].nunique():.1f}")

# Check code_ids size
code_ids = np.load("/home/cs21d002_eashaan/PhD/Objective1/data/processed/cache/huggingface_transformers_target_train_code_ids.npy", mmap_mode='r')
print(f"\nCode IDs shape: {code_ids.shape}")
print(f"Expected: (num_samples, max_lines * max_line_len)")
print(f"File size: {code_ids.nbytes / 1e9:.2f} GB")