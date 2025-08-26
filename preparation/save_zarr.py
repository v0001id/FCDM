import zarr
import numpy as np
import os
from glob import glob

latent_files = sorted(glob("/path/to/features/imagenet256_features/*.npy"))
label_files  = sorted(glob("/path/to/features/imagenet256_labels/*.npy"))

store_path = "/path/to/features/imagenet_latents.zarr"
z = zarr.open(store_path, mode="w")

n = len(latent_files)
z.create_dataset(
    "latents",
    shape=(n, 4, 32, 32),
    chunks=(1000, 4, 32, 32),
    dtype="float32"
)
z.create_dataset(
    "labels",
    shape=(n,),
    chunks=(1000,),
    dtype="int64"
)

for idx, (lf, yf) in enumerate(zip(latent_files, label_files)):
    z["latents"][idx, ...] = np.load(lf)
    z["labels"][idx]       = np.load(yf)
    if idx % 100_000 == 0:
        print(f"Written {idx}/{n}")

print("Zarr store ready at:", store_path)