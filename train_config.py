from pathlib import Path
import random
import datetime

import numpy as np
import torch
from torch.backends import cudnn

from loss import VisualLoss
from image_variants import ImageVariants

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# When evaluating the performance of the SR model, whether to verify only the Y channel image data



experiment_name = "espcn_x4_rgb_3_area"

upscale_factor = 4
image_variant = ImageVariants.RGB

in_channels = ImageVariants.get_variant_channels(image_variant)
out_channels = ImageVariants.get_variant_channels(image_variant)
channels = 128
only_test_y_channel = False

hr_image_size = int(32 * upscale_factor)
batch_size = 64
total_epochs = 400

train_hr_images_dir = f"./data/DIV2K/original"
valid_hr_images_dir = f"./data/DIV2K-val"

experiment_path = Path(f"experiments/{experiment_name}")
experiment_path.mkdir(exist_ok=True, parents=True)

backup_base_dir = experiment_path / "backup"
backup_base_dir.mkdir(exist_ok=True, parents=True)

# Trainers definition
model_trainer_kwargs = [
    {
        "image_variant": image_variant,
        "upscale_factor": upscale_factor,
        "loss_kwargs": {
            "lpips_weight": 1,
            "mse_weight": 0,
            "l1_weight": 0,
            "net_type": "alex",
            "max_pre_loss_shift": 8,
        },
        "internal_channels": channels,
        # "load_path": "experiments/espcn_x4_rgb_baselines/backup-long-1/espcnx4-lpips_alex-rgb-2024-02-15-00-58.pt",
    },
    {
        "image_variant": image_variant,
        "upscale_factor": upscale_factor,
        "loss_kwargs": {
            "lpips_weight": 1,
            "mse_weight": 0,
            "l1_weight": 0,
            "net_type": "vgg",
            "max_pre_loss_shift": 8,
        },
        "internal_channels": channels,
        # "load_path": "experiments/espcn_x4_rgb_baselines/backup-long-1/espcnx4-lpips_alex-rgb-2024-02-15-00-58.pt",
    },
    {
        "image_variant": image_variant,
        "upscale_factor": upscale_factor,
        "loss_kwargs": {
            "lpips_weight": 1,
            "mse_weight": 0,
            "l1_weight": 0,
            "net_type": "alex",
            "max_pre_loss_shift": 0,
        },
        "internal_channels": channels,
        # "load_path": "experiments/espcn_x4_rgb_baselines/backup-long-1/espcnx4-lpips_alex-rgb-2024-02-15-00-58.pt",
    },
    {
        "image_variant": image_variant,
        "upscale_factor": upscale_factor,
        "loss_kwargs": {
            "lpips_weight": 1,
            "mse_weight": 0,
            "l1_weight": 0,
            "net_type": "vgg",
            "max_pre_loss_shift": 0,
        },
        "internal_channels": channels,
        # "load_path": "experiments/espcn_x4_rgb_baselines/backup-long-1/espcnx4-lpips_vgg-rgb-2024-02-15-00-58.pt",
    },
    {
        "image_variant": image_variant,
        "upscale_factor": upscale_factor,
        "loss_kwargs": {
            "lpips_weight": 0,
            "mse_weight": 1,
            "l1_weight": 0,
            "net_type": "vgg",
            "max_pre_loss_shift": 0,
        },
        "internal_channels": channels,
        # "load_path": "experiments/espcn_x4_rgb_baselines/backup-long-1/espcnx4-mse-rgb-2024-02-15-00-58.pt",
    },
    {
        "image_variant": image_variant,
        "upscale_factor": upscale_factor,
        "loss_kwargs": {
            "lpips_weight": 0.2,
            "mse_weight": 0.8,
            "l1_weight": 0,
            "net_type": "vgg",
            "max_pre_loss_shift": 0,
        },
        "internal_channels": channels,
        # "load_path": "experiments/espcn_x4_rgb_baselines/backup-long-1/espcnx4-lpips_vgg+mse-rgb-2024-02-15-00-58.pt",
    },
    {
        "image_variant": image_variant,
        "upscale_factor": upscale_factor,
        "loss_kwargs": {
            "lpips_weight": 0.2,
            "mse_weight": 0.8,
            "l1_weight": 0,
            "net_type": "vgg",
            "max_pre_loss_shift": 8,
        },
        "internal_channels": channels,
        # "load_path": "experiments/espcn_x4_rgb_baselines/backup-long-1/espcnx4-slpips_8_vgg+mse-rgb-2024-02-15-00-58.pt",
    },
    {
        "image_variant": image_variant,
        "upscale_factor": upscale_factor,
        "loss_kwargs": {
            "lpips_weight": 0.2,
            "mse_weight": 0.8,
            "l1_weight": 0,
            "net_type": "alex",
            "max_pre_loss_shift": 0,
        },
        "internal_channels": channels,
        # "load_path": "experiments/espcn_x4_rgb_baselines/backup-long-1/espcnx4-lpips_alex+mse-rgb-2024-02-15-00-58.pt",
    },
    {
        "image_variant": image_variant,
        "upscale_factor": upscale_factor,
        "loss_kwargs": {
            "lpips_weight": 0.2,
            "mse_weight": 0.8,
            "l1_weight": 0,
            "net_type": "alex",
            "max_pre_loss_shift": 8,
        },
        "internal_channels": channels,
        # "load_path": "experiments/espcn_x4_rgb_baselines/backup-long-1/espcnx4-slpips_8_alex+mse-rgb-2024-02-15-00-58.pt",
    },
]