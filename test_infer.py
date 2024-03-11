from pathlib import Path

import cv2
import numpy as np 
import torch
from tqdm import tqdm 

import imgproc

from logger import MetricLogger
from loss import VisualLoss, ShiftLPIPS
from image_variants import ImageVariants
from model_trainer import ModelTrainer
from image_quality_assessment import PSNR, SSIM

import test_config as config


class SingleImageDataLoader:
    def __init__(self, images_base_path, upscale_factor, device):
        self.images_base_path = Path(images_base_path)
        self.upscale_factor = upscale_factor
        self.device = device

        self.image_paths = self._get_all_image_paths()
        self.iteration_index = 0

    def _get_all_image_paths(self):
        paths = list(self.images_base_path.glob('*'))
        sorted_paths = sorted(paths) # Sort the paths to ensure the same order every time
        return sorted_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def _load_image_pair(self, image_path: Path):
        hr_image = cv2.imread(str(image_path)).astype(np.float32) / 255.

        # Convert BGR to RGB
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
        lr_image = imgproc.image_resize(hr_image, 1 / self.upscale_factor)

        # Note: The range of input and output is between [0, 1]
        hr_tensor = imgproc.image_to_tensor(hr_image, False, False)
        lr_tensor = imgproc.image_to_tensor(lr_image, False, False)

        return lr_tensor, hr_tensor
    
    def __getitem__(self, index):
        lr_tensor, hr_tensor = self._load_image_pair(self.image_paths[index])

        return lr_tensor, hr_tensor
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.iteration_index < len(self):
            lr_tensor, hr_tensor = self[self.iteration_index]
            self.iteration_index += 1
            return lr_tensor, hr_tensor
        else:
            raise StopIteration

    def reset(self):
        self.iteration_index = 0



class InferenceTester:
    def __init__(self, models, metrics, test_images_dir, upscale_factor, output_images_dir, output_metrics_path, device):
        self.models = models # Dict[str, nn.Module]
        self.metrics = metrics # Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
        self.upscale_factor = upscale_factor
        self.output_images_dir = output_images_dir
        self.output_metrics_path = output_metrics_path
        self.device = device

        self.metric_logger = MetricLogger()

        self.test_images_loader = SingleImageDataLoader(test_images_dir, upscale_factor, device)

    def _save_image(self, image, image_path):
        img_rgb = imgproc.tensor_to_image(image, range_norm=False, half=True)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(image_path), img_bgr)


    def test(self):
        for lr_image, hr_image in tqdm(self.test_images_loader):
            lr_image = lr_image.to(self.device).unsqueeze(0) # Add batch dimension to the images
            hr_image = hr_image.to(self.device).unsqueeze(0)

            for model_name, model in self.models.items():
                with torch.no_grad():
                    sr_image = model(lr_image)

                output_image_path = self.output_images_dir / f'{self.test_images_loader.iteration_index:04d}' / f"{model_name}_.png"
                output_image_path.parent.mkdir(parents=True, exist_ok=True)
                
                self._save_image(sr_image, output_image_path)

                for metric_name, metric in self.metrics.items():
                    metric_value = metric(sr_image, hr_image).item()
                    self.metric_logger.log_row(model_name, metric_name, metric_value, self.test_images_loader.iteration_index, 0)

            # Add Bicubic for comparison
                    
            bicubic_sr_image = torch.nn.functional.interpolate(lr_image, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
            bicubic_output_image_path = self.output_images_dir / f'{self.test_images_loader.iteration_index:04d}' / f"Bicubic_.png"

            self._save_image(bicubic_sr_image, bicubic_output_image_path)

            for metric_name, metric in self.metrics.items():
                metric_value = metric(bicubic_sr_image, hr_image).item()
                self.metric_logger.log_row("Bicubic", metric_name, metric_value, self.test_images_loader.iteration_index, 0)

            self.metric_logger.to_csv(self.output_metrics_path) # Save the metrics to a file -- just in case the process is interrupted


if __name__ == "__main__":
    model_trainers = [
        ModelTrainer(
            **kwargs
        ) for kwargs in config.model_trainer_kwargs
    ]

    models = {
        model_trainer.descritpion_string: model_trainer.model
        for model_trainer in model_trainers
    }

    metrics = {
        "PSNR": PSNR(config.upscale_factor, config.only_test_y_channel),
        "SSIM": SSIM(config.upscale_factor, config.only_test_y_channel),
        "LPIPS_vgg": ShiftLPIPS(net_type="vgg", max_pre_loss_shift=0).to(config.device),
        "LPIPS_alex": ShiftLPIPS(net_type="alex", max_pre_loss_shift=0).to(config.device)
    }


    inference_tester = InferenceTester(
        models=models,
        metrics=metrics,
        test_images_dir=config.test_images_dir,
        upscale_factor=config.upscale_factor,
        output_images_dir=config.out_images_dir,
        output_metrics_path=config.out_metrics_path,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    inference_tester.test()