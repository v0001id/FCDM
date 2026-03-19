"""
Code adapted from https://github.com/chuanyangjin/fast-DiT
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
from accelerate import Accelerator
from models.fcdm_models import FCDM_models

from diffusers.models import AutoencoderKL
from torchvision.utils import save_image
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import wandb

from flow_matching import create_transport, Sampler
from train_utils import parse_transport_args, parse_sde_args
import zarr

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class CustomDataset(Dataset):
    def __init__(self, features_dir, labels_dir):
        self.features_dir = features_dir
        self.labels_dir = labels_dir

        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]

        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))
        return torch.from_numpy(features), torch.from_numpy(labels)

class ZarrCustomDataset(Dataset):
    def __init__(self, store_path):
        self.store = zarr.open(store_path, mode="r")
        self.features = self.store["latents"]
        self.labels  = self.store["labels"]
        assert self.features.shape[0] == self.labels.shape[0], \
            "features and labels must have the same length"

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        features = self.features[idx]
        labels  = self.labels[idx]
        return torch.from_numpy(features), torch.from_numpy(labels)

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
    x = x.permute(1,2,0).numpy()
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


class HuggingFaceImageTextDataset(Dataset):
    def __init__(self, dataset_name, split="train", image_column="image", text_column="text", dataset_config_name=None):
        self.dataset = load_dataset(dataset_name, dataset_config_name, split=split)
        self.image_column = image_column
        self.text_column = text_column

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item[self.image_column]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))
        text = item[self.text_column]
        if text is None:
            text = ""
        return image, str(text)

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new FCDM model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup accelerator:
    accelerator = Accelerator()
    device = accelerator.device

    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.model.replace("/", "-")  # e.g., FCDM-XL/2 --> FCDM-XL-2 (for naming folders)

    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder

    if args.vae_name:
        experiment_dir += f"-{args.vae_name}"

    # Setup an experiment folder:
    print(accelerator.is_main_process)
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        sample_dir = f"{experiment_dir}/samples"  # Stores generated samples
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        print(f"Experiment directory created at {experiment_dir}")

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = FCDM_models[args.model](
        in_channels=args.in_channels,
        text_embed_dim=args.text_embed_dim,
        class_dropout_prob=args.class_dropout_prob,
        learn_sigma=False
    )
    # Note that parameter initialization is done within the FCDM constructor
    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )  # default: velocity; 
    transport_sampler = Sampler(transport)

    if args.hf_model_dir is not None:
        vae = AutoencoderKL.from_pretrained(
            args.hf_model_dir,
            local_files_only=True
        ).to(device)
    else:
        vae = AutoencoderKL.from_pretrained(args.hf_model_name).to(device)
    vae.eval()
    requires_grad(vae, False)

    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_name)
    text_encoder = AutoModel.from_pretrained(args.text_encoder_name).to(device)
    text_encoder.eval()
    requires_grad(text_encoder, False)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    if args.ckpt is not None:
        ckpt_path = args.ckpt
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict["model"])
        ema.load_state_dict(state_dict["ema"])
        opt.load_state_dict(state_dict["opt"])
        args = state_dict["args"]

    requires_grad(ema, False)

    if accelerator.is_main_process:
        logger.info(f"ConvDiff Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):

    # Setup data:
    if args.dataset_name is None:
        raise ValueError("--dataset-name is required for text-conditioned training.")

    dataset = HuggingFaceImageTextDataset(
        dataset_name=args.dataset_name,
        split=args.dataset_split,
        image_column=args.dataset_image_column,
        text_column=args.dataset_text_column,
        dataset_config_name=args.dataset_config_name,
    )
    sample_prompts = [dataset[i][1] for i in range(min(len(dataset), args.prompt_pool_size)) if dataset[i][1].strip()]
    train_transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
    ])
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} samples")

    if accelerator.is_main_process and args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            dir=experiment_dir,
        )

    # Prepare models for training:
    if args.ckpt is None:
        update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model, opt, loader = accelerator.prepare(model, opt, loader)
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
   
    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    
    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        for batch in loader:
            images, texts = batch
            pixel_values = torch.stack([train_transform(img.convert("RGB")) for img in images]).to(device)
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * args.vae_scaling_factor
                tokens = tokenizer(
                    list(texts),
                    padding=True,
                    truncation=True,
                    max_length=args.max_text_length,
                    return_tensors="pt",
                )
                tokens = {k: v.to(device) for k, v in tokens.items()}
                text_outputs = text_encoder(**tokens)
                text_embeddings = text_outputs.last_hidden_state.mean(dim=1)
            x = latents
            model_kwargs = dict(text_embeddings=text_embeddings)

            loss_dict = transport.training_losses(model, x, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item() / accelerator.num_processes
                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    if args.use_wandb:
                        wandb.log({"train/loss": avg_loss, "train/steps_per_sec": steps_per_sec}, step=train_steps)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save FCDM checkpoint:
            if train_steps % args.ckpt_every == 0:
                if accelerator.is_main_process:
                    # Save checkpoint
                    checkpoint = {
                        "model": accelerator.unwrap_model(model).state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

                    # Generate and save samples
                    gen = torch.Generator(device=device).manual_seed(42)
                    with torch.no_grad():
                        if sample_prompts:
                            n = min(4, len(sample_prompts))
                            eval_prompts = np.random.choice(sample_prompts, size=n, replace=False).tolist()
                        else:
                            n = 4
                            eval_prompts = ["a high-quality photo"] * n

                        latent_size = args.image_size // 8
                        z = torch.randn((n, 4, latent_size, latent_size), generator=gen, device=device)
                        tokenized = tokenizer(
                            eval_prompts,
                            padding=True,
                            truncation=True,
                            max_length=args.max_text_length,
                            return_tensors="pt",
                        )
                        tokenized = {k: v.to(device) for k, v in tokenized.items()}
                        text_outputs = text_encoder(**tokenized)
                        text_embeddings = text_outputs.last_hidden_state.mean(dim=1)

                         # Setup classifier-free guidance:
                        if args.cfg_scale > 1.0:
                            z = torch.cat([z, z], 0)
                            null_embeddings = torch.zeros_like(text_embeddings)
                            text_embeddings = torch.cat([text_embeddings, null_embeddings], 0)
                            sample_model_kwargs = dict(text_embeddings=text_embeddings, cfg_scale=args.cfg_scale)
                            model_fn = ema.forward_with_cfg
                        else:
                            sample_model_kwargs = dict(text_embeddings=text_embeddings)
                            model_fn = ema.forward

                        start_time = time()
                        sample_fn = transport_sampler.sample_sde()
                        samples = sample_fn(z, model_fn, **sample_model_kwargs)[-1]
                        end_time = time()
                        sampling_time = end_time - start_time

                        if args.cfg_scale > 1.0:
                            samples, _ = samples.chunk(2, dim=0)
                        samples = vae.decode(samples / args.vae_scaling_factor).sample

                    logger.info(f"Sampling {len(eval_prompts)} images took {sampling_time:.2f} seconds")
                    sample_path = f"{sample_dir}/{train_steps:07d}.png"
                    save_image(samples, sample_path, nrow=max(1, len(eval_prompts)//2), normalize=True, value_range=(-1, 1))
                    logger.info(f"Generated samples at step {train_steps}")

                    if args.use_wandb:
                        wandb.log(
                            {
                                "eval/samples": [wandb.Image(custom_to_pil(img), caption=prompt) for img, prompt in zip(samples, eval_prompts)],
                                "eval/prompts": eval_prompts,
                            },
                            step=train_steps,
                        )

            if train_steps > args.max_train_steps:
                break
    
    if accelerator.is_main_process:
        logger.info("Done!")
        if args.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    # Default args here will train FCDM-XL with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=str, default="features")
    parser.add_argument("--label-path", type=str, default="labels")
    parser.add_argument("--zarr-path", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--dataset-config-name", type=str, default=None)
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--dataset-image-column", type=str, default="image")
    parser.add_argument("--dataset-text-column", type=str, default="text")
    parser.add_argument("--prompt-pool-size", type=int, default=2048)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--in-channels", type=int, default=4)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--max-train-steps", type=int, default=1_000_000)
    parser.add_argument("--model", type=str, choices=list(FCDM_models.keys()), default="FCDM-XL")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--class-dropout-prob", type=float, default=0.1)
    parser.add_argument("--text-encoder-name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--text-embed-dim", type=int, default=384)
    parser.add_argument("--max-text-length", type=int, default=77)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae-name", type=str, default="ema")  # Choice doesn't affect training
    parser.add_argument("--hf-model-name", type=str, default="stabilityai/sd-vae-ft-ema")
    parser.add_argument("--hf-model-dir", type=str, default=None)
    parser.add_argument("--vae-scaling-factor", type=float, default=0.18215)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--use-wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", type=str, default="fcdm-text-flow")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    parse_transport_args(parser)
    parse_sde_args(parser)
    args = parser.parse_args()
    main(args)
