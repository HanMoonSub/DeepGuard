import torch
import torch.nn as nn
import math

## Spatial Attention Module
class SAM(nn.Module):
    def __init__(self, kernel_size=3):
        super(SAM, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding='same', bias=False)
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, x):

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)

        cat = torch.cat([max_out, avg_out], dim=1)

        out = self.conv(cat)
        
        return x * self.sigmoid(out)