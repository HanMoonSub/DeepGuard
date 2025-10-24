from typing import Optional, Type
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention module
    """
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8,
        qkv_bias: bool = False, 
        attn_drop: float = 0., 
        proj_drop: float = 0.
        ):
        
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5  # Scaled dot-product

        # Q, K, V
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape  # Batch, Sequence Length, Feature Dim
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # Attention 
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Output projection
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    
    """
    Feed Forward Network (MLP) used in Transformer blocks
    """
    
    def __init__(
            self, 
            dim: int, 
            mlp_ratio: float = 4.0, 
            act_layer: Type[nn.Module] = nn.GELU,
            drop: float = 0.,
            ):
        super.__init__()
        
        hidden_dim = int(dim * mlp_ratio)
        
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop2(x)
        
        return x
     
class TransformerBlock(nn.Module):
    """
    Transformer Encoder Block
    """
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8, 
        mlp_ratio: float = 4., 
        qkv_bias: bool = False,
        act_type: str = 'gelu', 
        attn_drop: float = 0., 
        drop: float = 0.
        ):
    
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)  # Pre LayerNorm
        self.attn = MultiHeadAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)  # Pre LayerNorm
        
        act_dict = {
            'gelu': nn.GELU,
            'relu': nn.RELU,
            'silu': nn.SiLU,
        }
        if act_type.lower() not in act_dict:
             raise ValueError(f"Invalid act_type '{act_type}'. Choose from {list(act_dict.keys())}.")
        
        act_layer = act_dict[act_type.lower()]
        
        # Feed-Forward Network (MLP)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, act_layer = act_layer, drop=drop)

    def forward(self, x):
        # Residual connection + Attention
        x = x + self.attn(self.norm1(x))
        # Residual connection + MLP
        x = x + self.mlp(self.norm2(x))
        return x
