import torch
import torch.nn as nn
import math
from .attention_gate import AttentionGate

## Triplet Attention Module
class TripletAttention(nn.Module):
    def __init__(self):
        super(TripletAttention, self).__init__()
        
        self.cw = AttentionGate() # Channel, Width
        self.hc = AttentionGate() # Height, Channel
        self.hw = AttentionGate() # Height, Width

    def forward(self, x):
        # x: (Batch, Channel, Height, Width)
        
        # Channel, Width
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out1 = x_out1.permute(0,2,1,3).contiguous()

        # Height, Channel
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out2 = x_out2.permute(0,3,2,1).contiguous()

        # Height, Width
        x_out = self.hw(x)
        x_out = (x_out + x_out1 + x_out2) / 3

        return x_out