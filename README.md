# All Convolution, No Attention (ICLR 2026 Prep)

This repository provides training and inference scripts for the **"All Convolution, No Attention"** model. It supports both local and cluster environments.

### 1. Environment setup

```bash
conda env create -f environment.yml
conda activate ACD
```

### 1. (Optional) Manual Snapshot Download of EQVAE

Some cluster environments may restrict automatic downloads from Hugging Face due to security policies. If so, follow these steps to manually download the EQVAE (or SDVAE) snapshot:

1. **Download locally**  
Run the following script on your local machine:
  ```bash
  python preparation/download_snapshot.py
  ```
This will create a folder (e.g., `zelaki-eq-vae/`) containing model files like `config.json`, `pytorch_model.bin`, etc.

2. **Transfer to cluster**

Upload the folder to your cluster:
  ```bash
  scp -r zelaki-eq-vae /path/to/your_cluster
  ```
3. **Specify the snapshot path**

When launching any script, use the `--hf-model-dir` argument to point to the snapshot directory (default: `None`):
  ```bash
  --hf-model-dir /path/on/cluster/zelaki-eq-vae
  ```

### 2. Extract Latent Representations
  ```bash
  torchrun --nnodes=1 --nproc_per_node=4  preparation/extract_features.py \
      --data-path /path/to/imagenet \
      --features-path /path/to/latents
  ```
This will generate folders containing encoded features and corresponding labels. These paths should be provided to `--feature-path` and `--label-path` during training.

### 2-1 (Optional) Cache Latents to Zarr Format
If your cluster imposes a file number limit, it's recommended to store the latents in Zarr format. Manually set the feature, label, and output paths:
  ```bash
  python preparation/save_zarr.py
  ```
Once caching is complete, pass the Zarr path in any script using:
  ```bash
  --zarr-path /path/to/cache.zarr
  ```

### 3. Train ACD on Precomputed Latents
  ```bash
  accelerate launch --mixed_precision bf16 --num_processes 4 train_gen/train.py \
    --model ACD-XL \
    --label-path /path/to/labels \
    --feature-path /path/to/features \
    --results-dir results
  ```
*Optional arguments*: `--hf-model-dir`, `--zarr-path`

We found that `--mixed_precision` bf16 is more stable than fp16, especially when resuming training. We recommend using bf16 for mixed precision.

### 4. Sample 50k Images for Evaluation
Run the following to generate 50,000 samples for evaluation:
  ```bash
torchrun --nnodes=1 --nproc_per_node=4 train_gen/sample_ddp.py \
    --model ADC-XL \
    --num-fid-samples 50000 \
    --ckpt /path/to/checkpoint \
    --sample-dir samples \
    --ddpm True \
    --cfg-scale 1.0
  ```
*Optional argument*: `--hf-model-dir`

This script automatically saves the generated images and a corresponding `.npz` file.

### 5. Evaluation:
Before running evaluation, download the reference `.npz` file from [ADM's TensorFlow evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) and place it in the appropriate directory. 

> **Note**: Evaluation requires a TensorFlow environment. We recommend setting up a separate environment for this to avoid potential conflicts with PyTorch.

Run evaluation using:
  ```bash
  python train_gen/evaluate.py \
    --num-fid-samples 50000 \
    --sample-dir samples \
    --sample-batch /path/to/sample/sample.npz \
    --ref-batch /path/to/ref.npz \
    --output_fid /path/to/fid_output.txt
  ```