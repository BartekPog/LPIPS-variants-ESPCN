from collections import deque

import numpy as np
import torch 
import torch.nn.functional as F

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity 


class ShiftLPIPS(LearnedPerceptualImagePatchSimilarity): # lpips.LPIPS
    '''LPIPS, but randomly rolls image before computing loss. Accepts unscaled bnw or color images.'''
    def __init__(self, max_pre_loss_shift=8, net_type='alex', **kwargs):
        super().__init__(normalize=False, net_type=net_type, **kwargs)

        assert max_pre_loss_shift >= 0
        assert isinstance (max_pre_loss_shift, int)

        self.max_pre_loss_shift = max_pre_loss_shift
        self.net_type = net_type
        
    def get_random_shifts(self, batch_size):
        return torch.randint(-self.max_pre_loss_shift, self.max_pre_loss_shift, (batch_size, 2))
    
    def roll_image(self, x, shift):
        """Roll single image (c,w,h) by shift amount."""
        return torch.roll(x, shift.tolist(), dims=(1,2))
    
    def rescale_images(self, img, ref_mean, ref_std):
        norm_img = (img - ref_mean) / ref_std # normalize across batch

        return F.tanh(norm_img) # scale to [-1,1]
    
    def expand_channels_to_rgb(self, img):
        return img.expand(-1, 3, -1, -1)
    
    def random_roll_batch(self, x, y):
        """Randomly roll image batches before computing loss."""
        batch_size = x.shape[0]
        shifts = self.get_random_shifts(batch_size) 
        
        x_rolled = torch.stack([self.roll_image(img, shift) for img, shift in zip(x, shifts)])
        y_rolled = torch.stack([self.roll_image(img, shift) for img, shift in zip(y, shifts)])

        return x_rolled, y_rolled

    def forward(self, x, y):
        """Compute LPIPS loss between two image batches"""
        ref_mean = y.mean()
        ref_std = y.std()

        x = self.rescale_images(x, ref_mean, ref_std)
        y = self.rescale_images(y, ref_mean, ref_std)

        if x.shape[1] == 1:
            x = self.expand_channels_to_rgb(x)
            y = self.expand_channels_to_rgb(y)

        if self.max_pre_loss_shift != 0:
            x, y = self.random_roll_batch(x, y)

        return super().forward(x, y) # add batch dimension

    @property
    def metric_name(self):
        if self.max_pre_loss_shift == 0:
            return f'lpips_{self.net_type}'

        return f'slpips_{self.max_pre_loss_shift}_{self.net_type}'
    

class VisualLoss(torch.nn.Module):
    '''Wrapper for LPIPS, MSE and L1 losses. Accepts unscaled images.'''
    def __init__(self, lpips_weight=0, mse_weight=0, l1_weight=0, device='cuda', past_inferences_avg_num=50, **kwargs):
        super().__init__()

        raw_weights = {
            'lpips_weight': lpips_weight,
            'mse_weight': mse_weight,
            'l1_weight': l1_weight,
        }
        

        # filter out zero weights
        filtered_weights = {
            k: v
            for k, v  
            in raw_weights.items() if v > 0
        }

        self.loss_weights = {}
        self.losses = torch.nn.ModuleDict({})

        for k, v in filtered_weights.items(): # add metrics and weights, we want to avoid initializing a metric if it's zero-weighted
            if k == 'lpips_weight':
                lpips = ShiftLPIPS(**kwargs)
                self.losses[lpips.metric_name] = lpips
                self.loss_weights[lpips.metric_name] = torch.tensor(v, device=device)
            
            elif k == 'mse_weight':
                self.losses['mse'] = torch.nn.MSELoss()
                self.loss_weights['mse'] = torch.tensor(v, device=device)
            
            elif k == 'l1_weight':
                self.losses['l1'] = torch.nn.L1Loss()
                self.loss_weights['l1'] = torch.tensor(v, device=device)

        assert any(torch.stack(list(self.loss_weights.values())) > 0), 'At least one loss must be weighted > 0'
        assert all(torch.stack(list(self.loss_weights.values())) >= 0), 'All loss weights must be >= 0'

        self.device = device
        for loss in self.losses.values():
            loss.to(self.device)

    def _get_loss_normalizer_factor(self, loss_name):
        if loss_name not in self.loss_raw_values:
            raise ValueError(f"Metric {loss_name} not found in loss_raw_values")

        return torch.tensor(1, device=self.device) # with empty deque, we assume it has values around 1 


    def forward(self, x, y):
        total_loss = 0
        for loss_name, loss_func in self.losses.items():
            loss_value = loss_func(x, y)

            weighted_loss = self.loss_weights[loss_name] * loss_value
            total_loss += weighted_loss

        return total_loss
    
    @property
    def description(self):
        desc_parts = [
            metric_name 
            for metric_name, weight in self.loss_weights.items()
            if weight > 0
        ]

        return '+'.join(desc_parts)