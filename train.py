import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import imgproc
import train_config as config

from image_variants import ImageVariants
from logger import MetricLogger
from model_trainer import ModelTrainer


from dataset import TrainValidRGBImageDataset
from image_quality_assessment import PSNR, SSIM


class Training:
    def __init__(self, model_trainers, train_dataset_path, valid_dataset_path, backup_base_dir, hr_image_size, batch_size, total_epochs, metrics, device='cuda'):
        self.model_trainers = model_trainers
        self.train_dataset_path = train_dataset_path
        self.valid_dataset_path = valid_dataset_path
        self.backup_dir = backup_base_dir
        self.device = device
        self.batch_size = batch_size
        self.hr_image_size = hr_image_size
        self.metrics = metrics

        self.total_epochs = total_epochs
        self.trained_epochs = 0

        self.logger = MetricLogger()
        
        upscale_factors = {model_trainer.upscale_factor for model_trainer in model_trainers}
        assert len(upscale_factors) == 1, f"Multiple upscale factors detected: {upscale_factors}"
        self.upscale_factor = upscale_factors.pop()

        image_variants = {model_trainer.image_variant for model_trainer in model_trainers}
        assert len(image_variants) == 1, f"Multiple image variants detected: {image_variants}"
        self.image_variant = image_variants.pop()

        assert self.image_variant == ImageVariants.RGB, f"Unsupported image variant: {self.image_variant}, only YCbCr is now supported."

        self.train_loader = self._load_train_dataloader()
        self.valid_loader = self._load_valid_dataloader()

    def _increase_epoch(self):
        for model_trainer in self.model_trainers:
            model_trainer.increment_epoch()
        self.trained_epochs += 1

    def _backup_models(self, incl_epoch=True):
        for model_trainer in self.model_trainers:
            model_trainer.save(self.backup_dir, incl_epoch=incl_epoch)

    def _load_train_dataloader(self):
        train_dataset = TrainValidRGBImageDataset(
            self.train_dataset_path,
            hr_image_size=self.hr_image_size,
            upscale_factor=self.upscale_factor,
            only_use_y_channel=True,
            mode="Train"
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        return train_loader

    def _load_valid_dataloader(self):
        valid_dataset = TrainValidRGBImageDataset(
            self.valid_dataset_path,
            hr_image_size=self.hr_image_size,
            upscale_factor=self.upscale_factor,
            only_use_y_channel=False,
            mode="Valid"
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        return valid_loader
    
    def train_step(self, lr, target):
        for model_trainer in self.model_trainers:
            model_trainer.training_step(lr, target) 
        
    def train_epoch(self):
        for model_trainer in self.model_trainers:
            model_trainer.model.train()

        for lr, hr in tqdm(self.train_loader, total=len(self.train_loader), desc=f"Training epoch {self.trained_epochs + 1} of {self.total_epochs}, {(self.trained_epochs) / self.total_epochs:.2%} complete"):
            lr = lr.to(self.device)
            hr = hr.to(self.device)
            
            self.train_step(lr, hr)

    def prediction_merge_cbcr(self, prediction_y, hr_ycbcrcv):
        prediction_ycbcr = torch.cat([prediction_y, hr_ycbcrcv[:, 1:, :, :].detach(), hr_ycbcrcv[:, 2:, :, :].detach()], dim=1)

        return prediction_ycbcr

    def valid_epoch(self):
        for model_trainer in self.model_trainers:
            model_trainer.model.eval()
        
        with torch.no_grad():
            for batch_index, (lr, hr) in tqdm(enumerate(self.valid_loader), total=len(self.valid_loader)):
                lr = lr.to(self.device)
                hr = hr.to(self.device)
                
                for model_trainer in self.model_trainers:
                    if model_trainer.image_variant == ImageVariants.YCbCr:
                        prediction = model_trainer.model(lr[:, :1, :, :]) # Only Y channel
                        prediction = self.prediction_merge_cbcr(prediction, hr)

                        hr = imgproc.ycbcr_to_rgb_torch(hr)
                        prediction = imgproc.ycbcr_to_bgr_torch(prediction)

                    else:
                        prediction = model_trainer.model(lr)

                    for metric in self.metrics:
                        value = metric(prediction, hr)

                        self.logger.log_row(model_trainer.descritpion_string, metric, value.mean().item(), batch_index, self.trained_epochs)                       
            
        self.logger.print_mean_last_epoch()
        self.logger.to_csv('last_epoch.csv')
        

    def train(self):
        for _ in range(self.total_epochs):
            self.train_epoch()
            self.valid_epoch()

            self._increase_epoch()
            self._backup_models(incl_epoch=False) # Backup without epoch number to overwrite the last backup
        
        self._backup_models(incl_epoch=True) # Backup with epoch number to keep the last backup
            
if __name__ == "__main__":
    model_trainers = [
        ModelTrainer(**kwargs) for kwargs in config.model_trainer_kwargs
    ]

    metrics = [
        PSNR(config.upscale_factor, config.only_test_y_channel),
        SSIM(config.upscale_factor, config.only_test_y_channel)
    ]

    ycbcr_training = Training(
        model_trainers,
        config.train_hr_images_dir,
        config.valid_hr_images_dir,
        config.backup_base_dir,
        config.hr_image_size,
        config.batch_size,
        config.total_epochs,
        metrics
    )

    ycbcr_training.train()