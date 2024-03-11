import datetime
from pathlib import Path

import torch

from loss import VisualLoss 
from image_variants import ImageVariants
import model


class ModelTrainer(torch.nn.Module):
    def __init__(self, load_path=None, image_variant=ImageVariants.RGB, upscale_factor=4, loss_kwargs=None, internal_channels=64, device='cuda', lr=1e-3, **kwargs):
        super().__init__(**kwargs)

        self.upscale_factor = upscale_factor
        self.device = device
        self.image_variant = image_variant

        self.init_date_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        self.trained_epochs = 0

        self.internal_channels = internal_channels

        self.model = self._build_model()
        self.loss = VisualLoss(**loss_kwargs)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        if load_path is not None:
            print(f"Loading model from {load_path}")
            self.load(load_path)
        else:
            assert loss_kwargs is not None, "loss_kwargs must be provided when creating a new model"
            print(f"New model initialized with {self.loss.description} loss")
            

        self.model.to(self.device)
        self.loss.to(self.device)
        

    def _build_model(self):
        input_output_channels = ImageVariants.get_variant_channels(self.image_variant)

        espcn_model = model.ESPCN(in_channels=input_output_channels,
                                  out_channels=input_output_channels,
                                  channels=self.internal_channels,
                                  upscale_factor=self.upscale_factor)
        
        espcn_model = espcn_model.to(device=self.device)

        return espcn_model
    
    def increment_epoch(self): 
        self.trained_epochs += 1
    
    def training_step(self, lr, target):
        assert lr.shape[:2] == target.shape[:2], f"Misaligned batch size or channels {lr.shape}[:2] != target shape {target.shape}:[:2]"

        self.optimizer.zero_grad()

        loss = self.loss(self.model(lr), target)
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def save(self, save_dir, save_name=None, incl_epoch=True):
        if save_name is None:
            save_name = self.descritpion_string

        if incl_epoch:
            save_name += f"_epoch-{self.trained_epochs}"
            
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / f"{save_name}.pt"
        torch.save(self.state_dict(), save_path)

        return save_path

    def load(self, load_path):
        self.load_state_dict(torch.load(load_path))
        self.trained_epochs = 0 #int(Path(load_path).stem.split("_")[-1].split(".")[0])

        return self
    
    def __str__(self):
        return f"ModelTrainer({self.init_date_string}, {self.trained_epochs} epochs, {self.upscale_factor}x, {self.image_variant})"
    
    @property 
    def descritpion_string(self):
        return f"espcnx{self.upscale_factor}-{self.loss.description}-{self.image_variant}-{self.init_date_string}"
    