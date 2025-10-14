import torch
import torch.nn as nn
import math

## Efficient Channel Attention Module
class ECA(nn.Module):
    def __init__(self, in_planes, gamma=2, b=1):
        super(ECA, self).__init__()
        t = int(abs((math.log2(in_planes) / gamma) + b))
        k = t if t % 2 else t + 1  

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.conv = nn.Conv1d(2, 1, kernel_size=k, padding=(k-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, _, _ = x.size()
        
        avg_out = self.avg_pool(x).squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        max_out = self.max_pool(x).squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        
        out = torch.cat([avg_out, max_out], dim=1)

        out = self.conv(out)   # (B, 1, C)
        out = self.sigmoid(out).transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)

        return x * out