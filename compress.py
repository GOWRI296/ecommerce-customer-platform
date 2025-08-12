import pandas as pd
import numpy as np
import os

def get_file_size(filepath):
    return os.path.getsize(filepath) / (1024*1024)  # Size in MB

print("Starting compression...")

# Define specific file paths
csv_files = [
    r"D:\projects\streamlit project\customer_products.csv",
    r"D:\projects\streamlit project\online_retail.csv"
]

npy_files = [
    r"D:\projects\streamlit project\product_similarities.npy"
]

# Compress CSV files
for csv_file in csv_files:
    if os.path.exists(csv_file):
        print(f"Compressing {os.path.basename(csv_file)}...")
        try:
            df = pd.read_csv(csv_file)
            parquet_file = csv_file.replace('.csv', '.parquet')
            df.to_parquet(parquet_file)
            
            old_size = get_file_size(csv_file)
            new_size = get_file_size(parquet_file)
            compression_ratio = ((old_size - new_size) / old_size) * 100
            print(f"  {os.path.basename(csv_file)}: {old_size:.1f}MB → {os.path.basename(parquet_file)}: {new_size:.1f}MB ({compression_ratio:.1f}% reduction)")
        except Exception as e:
            print(f"  Error compressing {os.path.basename(csv_file)}: {e}")
    else:
        print(f"  File not found: {csv_file}")

# Compress NPY files
for npy_file in npy_files:
    if os.path.exists(npy_file):
        print(f"Compressing {os.path.basename(npy_file)}...")
        try:
            data = np.load(npy_file)
            npz_file = npy_file.replace('.npy', '.npz')
            np.savez_compressed(npz_file, data=data)
            
            old_size = get_file_size(npy_file)
            new_size = get_file_size(npz_file)
            compression_ratio = ((old_size - new_size) / old_size) * 100
            print(f"  {os.path.basename(npy_file)}: {old_size:.1f}MB → {os.path.basename(npz_file)}: {new_size:.1f}MB ({compression_ratio:.1f}% reduction)")
        except Exception as e:
            print(f"  Error compressing {os.path.basename(npy_file)}: {e}")
    else:
        print(f"  File not found: {npy_file}")

print("Compression complete!")

# Optional: Print total space saved
print("\nSummary:")
print("- CSV files converted to Parquet format")
print("- NPY files converted to compressed NPZ format")
print("- Original files are preserved")
print("- Use the compressed versions in your Streamlit app for faster loading")