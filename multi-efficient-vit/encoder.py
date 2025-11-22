import torch.nn as nn
from .transformer import TransformerBlock

class MultiScaleEncoder(nn.Module):
    def __init__(self,
                 h_branch,
                 m_branch,
                ):
        super().__init__()

        # High-branch Transformer Encoder
        self.h_branch_encoder = nn.ModuleList([
            TransformerBlock(
                dim = h_branch.embed_dim,
                num_heads = h_branch.num_heads,
                mlp_ratio = h_branch.mlp_ratio,
                qkv_bias = h_branch.qkv_bias,
                attn_drop = h_branch.attn_drop,
                drop = h_branch.drop,
            ) for _ in range(h_branch.num_transformer_layers)
        ])

        # Middle-branch Transformer Encoder
        self.m_branch_encoder = nn.ModuleList([
            TransformerBlock(
                dim = m_branch.embed_dim,
                num_heads = m_branch.num_heads,
                mlp_ratio = m_branch.mlp_ratio,
                qkv_bias = m_branch.qkv_bias,
                attn_drop = m_branch.attn_drop,
                drop = m_branch.drop,
            ) for _ in range(m_branch.num_transformer_layers)
        ])


    def forward(self, h_token, m_token):

        # High-branch encoding
        res_h_token = h_token.clone()

        for h_block in self.h_branch_encoder:
            h_token = h_block(h_token)

        h_token_out = h_token * 0.9 + res_h_token * 0.1

        # Middle-branch encoding
        res_m_token = m_token.clone()

        for m_block in self.m_branch_encoder:
            m_token = m_block(m_token)

        m_token_out = m_token * 0.9 + res_m_token * 0.1

        return h_token_out, m_token_out
