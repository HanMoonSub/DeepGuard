import torch.nn as nn
from typing import Type, Optional
from .reducesize import ReduceSize
from .se import SE

class FeatExtract(nn.Module):
    
    """
        MBConv -> MaxPool 2X2
        (B,C,H,W) -> (B,C,H//2,W//2)
    """
    def __init__(self, dim):
        super().__init__()
        self.mb_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                      groups=dim, bias=False),
            nn.GELU(),
            SE(dim),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        x = x.contiguous()
        x =  x + self.mb_conv(x) # skip connection
        x = self.pool(x)
            
        return x

class MidLevelPatchEmbed(nn.Module):
    
    """
    Feature map: block.0 output, (B,C,H,W) -> (B,H//8,W//8,D)
    Feature map: block.1 output, (B,C,H,W) -> (B,H//4,W//4,D)
    Feature map: block.2 output, (B,C,H,W) -> (B,H//2,W//2,D)
    
    """
    
    def __init__(
                self,
                in_chs: int,
                dim: int,
                block_idx: int,
                ):
        super().__init__()
        
        if block_idx == 0:
            self.proj = nn.Conv2d(in_chs, dim, kernel_size=7, stride=4, padding=3)
        elif block_idx == 1:
            self.proj = nn.Conv2d(in_chs, dim, kernel_size=3, stride=2, padding=1)
        elif block_idx == 2:
            self.proj = nn.Conv2d(in_chs, dim, kernel_size=1, stride=1, padding=0)
        
        self.conv_down = ReduceSize(dim=dim, keep_dim=True)
        
    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0,2,3,1) # (B,C,H,W) -> (B,H,W,C)
        x = self.conv_down(x)
        
        return x
    
    
class LowLevelPatchEmbed(nn.Module):
    
    """
    Feature map: block.5 output, (B,C,H,W) -> (B,H//2,W//2,D)
    Feature map: block.6 output, (B,C,H,W) -> (B,H,W,D
    
    """
    
    
    def __init__(
                self,
                in_chs: int,
                dim: int, 
                block_idx: int, 
                ):
        super().__init__()
        
        if block_idx == 5:
            self.block = nn.Sequential(
                FeatExtract(in_chs),
                nn.Conv2d(in_chs, dim, kernel_size=1, stride=1, padding=0)
            )
        
        elif block_idx == 6:
            self.block = nn.Conv2d(in_chs, dim, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        
        x = self.block(x)
        x = x.permute(0,2,3,1).contiguous() # (B,C,H,W) -> (B,H,W,C)
        
        return x