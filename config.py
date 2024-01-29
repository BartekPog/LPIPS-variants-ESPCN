# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import random
import datetime

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# When evaluating the performance of the SR model, whether to verify only the Y channel image data
only_test_y_channel = False
# Model architecture name
model_arch_name = "espcn_x4"
# Model arch config
in_channels = 1
out_channels = 1
channels = 64
upscale_factor = 4
# Current configuration parameter method
mode = "train"
# Experiment name, easy to save weights and log files

random_string = "".join([str(random.randint(0, 9)) for _ in range(6)])
date_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
loss_type = "mse"

exp_name = f"{model_arch_name}-DIV2K-{loss_type}-{date_string}-{random_string}"

if mode == "train":
    # Dataset address
    train_gt_images_dir = f"./data/DIV2K/original"

    # test_gt_images_dir = f"./data/Set5/GTmod12"
    # test_lr_images_dir = f"./data/Set5/LRbicx{upscale_factor}"

    test_gt_images_dir = f"./data/DIV2K-test"
    # test_lr_images_dir = f"./data/DIV2K-valid/LRx4m"

    # gt_image_size = int(17 * upscale_factor)
    gt_image_size = int(32 * upscale_factor)
    batch_size = 256
    num_workers = 4

    # The address to load the pretrained model
    pretrained_model_weights_path = f""

    # Incremental training and migration training
    resume_model_weights_path = f""

    # Total num epochs
    epochs = 100

    # loss function weights
    loss_weights = 1.0

    # Optimizer parameter
    model_lr = 1e-3
    # model_momentum = 0.9
    model_weight_decay = 1e-7
    # model_nesterov = False

    # EMA parameter
    # model_ema_decay = 0.999

    # Dynamically adjust the learning rate policy
    lr_scheduler_milestones = [int(epochs * 0.1), int(epochs * 0.8)]
    lr_scheduler_gamma = 0.1

    # How many iterations to print the training result
    train_print_frequency = 100
    test_print_frequency = 10

if mode == "test":
    # Test data address
    lr_dir = f"./data/Set5/LRbicx{upscale_factor}"
    sr_dir = f"./results/test/{exp_name}"
    gt_dir = "./data/Set5/GTmod12"

    model_weights_path = "./results/pretrained_models/ESPCN_x4-DIV2K-64bf5ee4.pth.tar"
