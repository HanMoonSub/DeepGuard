import torch.nn as nn
from .branch import EfficientBranch
from .encoder import MultiScaleEncoder

class MultiScaleEffViT(nn.Module):
    def __init__(self, config):

        super().__init__()

        self.config = config

        self.cnn_branch = EfficientBranch(self.config.cnn_backbone, self.config.img_size, self.config.h_branch, self.config.m_branch)

        self.multi_encoder = MultiScaleEncoder(self.config.h_branch, self.config.m_branch)

        # Head
        self.h_branch_head = self.create_branch_head(self.config.h_branch.embed_dim)
        self.m_branch_head = self.create_branch_head(self.config.m_branch.embed_dim)

        self._init_weights()

    def _init_weights(self):
        modules_to_init = [
            self.multi_encoder,
            self.h_branch_head,
            self.m_branch_head,
        ]

        for module in modules_to_init:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def create_branch_head(embed_dim, dropout=0.5):
        return nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.LayerNorm(embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Linear(embed_dim // 4, 1)
            )

    def forward(self, x):

        h_branch_token, m_branch_token, = self.cnn_branch(x)

        h_branch_out, m_branch_out = self.multi_encoder(h_branch_token, m_branch_token)

        h_branch_out = self.h_branch_head(h_branch_out[:,0,:]) # cls_token
        m_branch_out = self.m_branch_head(m_branch_out[:,0,:]) # cls_token

        return h_branch_out + m_branch_out