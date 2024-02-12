
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
    def __init__(self, lpips_weight=0, mse_weight=0, l1_weight=0, device='cuda', **kwargs):
        super().__init__()

        raw_weights = {
            'lpips_weight': lpips_weight,
            'mse_weight': mse_weight,
            'l1_weight': l1_weight,
        }
        
        assert any(torch.stack(list(self.metric_weights.values())) > 0), 'At least one loss must be weighted > 0'
        assert all(torch.stack(list(self.metric_weights.values())) >= 0), 'All loss weights must be >= 0'

        # filter out zero weights
        filtered_weights = {
            k: v
            for k, v  
            in raw_weights.items() if v > 0
        }

        self.metric_weights = {}
        self.metrics = torch.nn.ModuleDict({})

        for k, v in filtered_weights.items(): # add metrics and weights, we want to avoid initializing a metric if it's zero-weighted
            if k == 'lpips_weight':
                lpips = ShiftLPIPS(**kwargs)
                self.metrics[lpips.metric_name] = lpips
                self.metric_weights[lpips.metric_name] = torch.tensor(v, device=device)
            
            elif k == 'mse_weight':
                self.metrics['mse'] = torch.nn.MSELoss()
                self.metric_weights['mse'] = torch.tensor(v, device=device)
            
            elif k == 'l1_weight':
                self.metrics['l1'] = torch.nn.L1Loss()
                self.metric_weights['l1'] = torch.tensor(v, device=device)

        self.device = device
        for metric in self.metrics.values():
            metric.to(self.device)

    def forward(self, x, y):
        total_loss = 0
        for metric_name, metric in self.metrics.items():
            loss = metric(x, y)
            total_loss += loss * self.metric_weights[metric_name]

        return total_loss
    
    @property
    def description(self):
        desc_parts = [
            metric_name 
            for metric_name, weight in self.metric_weights.items()
            if weight > 0
        ]

        return '+'.join(desc_parts)