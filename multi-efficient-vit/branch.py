import timm
import torch
import torch.nn as nn
from einops import rearrange


class EfficientBranch(nn.Module):
    def __init__(self, cnn_backbone, img_size, h_branch, m_branch):
        super().__init__()
        self.h_branch = h_branch
        self.m_branch = m_branch

        # CNN Backbone
        self.cnn_backbone = timm.create_model(
            cnn_backbone,
            pretrained=True,
            features_only=True,
        )
        num_features = len(self.cnn_backbone.feature_info)

        # Check Branch Block Idx Error
        for branch_name, branch in zip(["h_branch", "m_branch"], [self.h_branch, self.m_branch]):
            assert 0 <= branch.block_idx < num_features, \
                f"{branch_name}.block_idx {branch.block_idx} out of range (0~{num_features-1})"

        # Initialize Branch Info
        h_info = self._init_branch(self.h_branch, img_size)
        m_info = self._init_branch(self.m_branch, img_size)

        for k, v in h_info.items():
            setattr(self, f"h_branch_{k}", v)
        for k, v in m_info.items():
            setattr(self, f"m_branch_{k}", v)

        # Initialize cls_token, pos_embed weight
        self._init_weights()

    def _init_weights(self):
        for branch in ['h_branch', 'm_branch']:
            cls_token = getattr(self, f"{branch}_cls_token")
            pos_embed = getattr(self, f"{branch}_pos_embed")
            patch_to_embedding = getattr(self, f"{branch}_patch_to_embedding")

            nn.init.trunc_normal_(cls_token, std=0.02)
            nn.init.trunc_normal_(pos_embed, std=0.02)

            nn.init.trunc_normal_(patch_to_embedding.weight, std=0.02)
            if patch_to_embedding.bias is not None:
                nn.init.constant_(patch_to_embedding.bias, 0)

    def _init_branch(self, branch, img_size):
        C, H, W = self.get_branch_feature_info(branch, img_size)
        patch_dim = C * (branch.patch_size ** 2)
        patch_to_embedding = nn.Linear(patch_dim, branch.embed_dim)
        cls_token = nn.Parameter(torch.zeros(1, 1, branch.embed_dim))
        pos_embed = nn.Parameter(torch.zeros(1, (H // branch.patch_size) * (W // branch.patch_size) + 1, branch.embed_dim))
        pos_dropout = nn.Dropout(branch.pos_dropout)

        return dict(
            channels=C, height=H, width=W,
            patch_dim=patch_dim,
            patch_to_embedding=patch_to_embedding,
            cls_token=cls_token,
            pos_embed=pos_embed,
            pos_dropout=pos_dropout
        )

    def get_branch_feature_info(self, branch, img_size):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, *img_size)
            feat = self.cnn_backbone(dummy_input)[branch.block_idx]
        C, H, W = feat.shape[1], feat.shape[2], feat.shape[3]
        assert H % branch.patch_size == 0, f"Feature map height {H} must be divisible by patch size {branch.patch_size}"
        assert W % branch.patch_size == 0, f"Feature map width {W} must be divisible by patch size {branch.patch_size}"
        return C, H, W

    def forward(self, x):
        # Extract feature maps for each branch
        h_branch_feature = self.cnn_backbone(x)[self.h_branch.block_idx]
        m_branch_feature = self.cnn_backbone(x)[self.m_branch.block_idx]

        branch_list = []

        for branch, feature in zip(['h_branch', 'm_branch'], [h_branch_feature, m_branch_feature]):
            B, C, H, W = feature.shape

            p = getattr(self, branch).patch_size
            patch_to_embedding = getattr(self, f"{branch}_patch_to_embedding")
            cls_token = getattr(self, f"{branch}_cls_token")
            pos_embed = getattr(self, f"{branch}_pos_embed")
            pos_dropout = getattr(self, f"{branch}_pos_dropout")

            patches = rearrange(feature, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
            x_branch = patch_to_embedding(patches)

            cls_token = cls_token.expand(B, -1, -1)
            x_branch = torch.cat((cls_token, x_branch), dim=1)

            x_branch += pos_embed
            x_branch = pos_dropout(x_branch)

            branch_list.append(x_branch)

        return branch_list
