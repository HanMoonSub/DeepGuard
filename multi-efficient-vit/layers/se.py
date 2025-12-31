import torch.nn as nn

class SE(nn.Module):
    """
        Squeeze and excitation block
    """
    
    def __init__(
        self,
        in_chs: int,
        rd_ratio: float = 0.25
    ):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_chs, int(in_chs * rd_ratio), bias=False),
            nn.GELU(),
            nn.Linear(int(in_chs * rd_ratio), in_chs, bias=False),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        B, C, _, _ = x.shape
        
        out = self.pool(x).view(B,C)
        out = self.fc(out).view(B,C,1,1)
        
        return x * out