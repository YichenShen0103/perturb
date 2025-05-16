import torch.nn as nn
from pytorch_msssim import ssim

class SSIMLoss(nn.Module):
    def __init__(self, data_range=1.0):
        super().__init__()
        self.data_range = data_range

    def forward(self, x, y):
        return 1 - ssim(x, y, data_range=self.data_range, size_average=True)