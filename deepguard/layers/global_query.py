import math
import torch
import torch.nn as nn
from .se import SE

class FeatExtract(nn.Module):
    """
        Resize: Feature map -> window size for global attention
        Repeat X K (MBconv -> MaxPool 2X2)
        K: log2(feature map Height // window_size)
    """
    def __init__(self, dim, keep_dim=False):
        super().__init__()
        self.keep_dim = keep_dim
        self.mb_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                      groups=dim, bias=False),
            nn.GELU(),
            SE(dim),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False)
        )
        if not self.keep_dim:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        x = x.contiguous()
        x = x + self.mb_conv(x)
        if not self.keep_dim:
            x = self.pool(x)
            
        return x
    
class GlobalQueryGen(nn.Module):
    """
        inp: (B, D, H, W)
        out: (B, 1, num_heads, window_size**2, head_dim)
    """
    def __init__(self,
                 dim: int,
                 input_resolution: int | float,
                 window_size: int,
                 num_heads: int):
        super().__init__()

        self.resolution = input_resolution
        self.num_heads = num_heads
        self.N = window_size ** 2
        self.dim_head = dim // num_heads
        num_stages = int(math.log2(input_resolution // window_size))
        
        layers = []
        if num_stages > 0:
            for i in range(num_stages):
                layers.append(FeatExtract(dim, keep_dim=False))
        else:
            layers.append(FeatExtract(dim, keep_dim=True))
        
        self.to_q_global = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.to_q_global(x) # (B, D, H, W) -> (B, D, window_size, window_size)
        x = x.permute(0,2,3,1) # (B, D, window_size, window_size) -> (B, window_size, window_size, D)
        B = x.shape[0]
        x = x.reshape(B, 1, self.N, self.num_heads, self.dim_head).permute(0,1,3,2,4)
        return x