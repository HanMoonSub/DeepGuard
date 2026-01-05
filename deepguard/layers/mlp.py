import torch.nn as nn
from typing import Optional, Type

class Mlp(nn.Module):
    
    """
    Multi-Layer Perceptron for vision-transformer
    
    """
    
    
    def __init__(
        self,
        in_dims: int,
        hidden_dims: Optional[int] = None,
        out_dims: Optional[int] = None,   
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.,
        ):
        
        super().__init__()
        
        hidden_dims = hidden_dims or in_dims
        out_dims = out_dims or in_dims
    
        self.fc1 = nn.Linear(in_dims, hidden_dims)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dims, out_dims)
        self.drop = nn.Dropout(drop)    
        
        
    def forward(self, x):
        
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
    
        return x

