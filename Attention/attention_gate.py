import torch
import torch.nn as nn
import math

## Attention Gate(HxW, CxH, CxW)
class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
   
        self.conv = nn.Sequential(
            nn.Conv2d(2,1, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU(inplace=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        max_out, _ = torch.max(x, dim=1, keepdim=True) # (B,1,H,W)
        avg_out = torch.mean(x, dim=1, keepdim=True) # (B,1,H,W)

        cat = torch.cat([max_out, avg_out], dim=1)

        x_out = self.conv(cat)

        return x * self.sigmoid(x_out)