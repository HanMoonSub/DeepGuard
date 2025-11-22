import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.
                ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context=None):

        B, Nq, C = x.shape
        context = x if context is None else context
        Nk = context.shape[1]

        q = self.to_q(x).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        k = k.reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        v = v.reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0,2,1,3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x