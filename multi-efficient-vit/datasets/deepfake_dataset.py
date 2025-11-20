import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, Tuple
import albumentations as A
import cv2
from collections import defaultdict

class DeepFakeDataset(Dataset):
    """
    Generic DeepFake Image Dataset

    Args:
        meta_df (pd.DataFrame): DataFrame with image paths and labels.
        transforms (albumentations.Compose): Preprocessing & augmentation pipeline.
        image_key (str): Column name for image paths.
        label_key (str): Column name for labels.
        return_label (bool): If False → return only image (inference mode).
    """

    def __init__(
        self,
        meta_df,
        transforms: Optional[A.Compose] = None,
        image_col: str = "image_path",
        label_col: str = "label",
        origin_col: str = "origin_vid",
        frame_idx_col: str = "frame_idx",
        return_label: bool = True,
        mixup_prob: float = 0.0,
        mixup_alpha: float = 0.5,
        img_size: tuple =(224,224),
    ):
        self.meta_df = meta_df
        self.transforms = transforms
        self.image_col = image_col
        self.label_col = label_col
        self.origin_col = origin_col
        self.frame_idx_col = frame_idx_col
        self.return_label = return_label
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha
        self.img_size = img_size
        
        if self.transforms is None:
            raise ValueError("Transforms must be provided for preprocessing")

        if self.mixup_prob > 0:
            self.original_groups = defaultdict(lambda: {'real': [], 'fake': []})
            for _, row in self.meta_df.iterrows():
                original_id = row[self.origin_col]
                label = row[self.label_col]
                image_path = row[self.image_col]
                frame_idx = row[self.frame_idx_col]

                if label == 0:
                    self.original_groups[original_id]['real'].append((frame_idx, image_path))
                else:
                    self.original_groups[original_id]['fake'].append((frame_idx, image_path))
                
            for vid in self.original_groups:
                self.original_groups[vid]["real"] = sorted(self.original_groups[vid]["real"], key=lambda x: x[0])
                self.original_groups[vid]["fake"] = sorted(self.original_groups[vid]["fake"], key=lambda x: x[0])
    
    def __len__(self) -> int:
        return len(self.meta_df)

    def load_image(self, path: str) -> np.ndarray:
        """Load image safely and convert to numpy array."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")

        img = Image.open(path).convert("RGB")
        return np.array(img, dtype=np.float32)

    def resize_image(self, img):
        H, W = self.img_size

        transform = A.Compose([
            A.LongestMaxSize(max_size=max(H,W)),
            A.PadIfNeeded(min_height=H, min_width=W, border_mode=cv2.BORDER_CONSTANT)
        ])
        return transform(image=img)['image']

    def _apply_mixup(self, img, label, original_id, frame_idx):
        group = self.original_groups[original_id]

        # pick opposite label group
        if label == 0:
            pool = group['fake']
            opposite_label = 1.0
        else:
            pool = group['real']
            opposite_label = 0.0

        pool_paths = [p for idx, p in pool if idx == frame_idx]
        
        if len(pool_paths) == 0:
            return img, label

        opp_path = np.random.choice(pool_paths)
        opp_img = self.load_image(opp_path)
        opp_img = self.resize_image(opp_img)

        # mixup
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        mixed_img = img * lam + opp_img * (1 - lam)
        mixed_label = label * lam + opposite_label * (1 - lam)

        return mixed_img.astype(np.float32), float(mixed_label)

    def __getitem__(self, idx: int):
        row = self.meta_df.iloc[idx]
        image_path = row[self.image_col]
        label = row[self.label_col]
        original_id = row[self.origin_col]
        frame_idx = row[self.frame_idx_col]

        image = self.load_image(image_path)
        image = self.resize_image(image)

        # Apply MixUp with probability
        if np.random.rand() < self.mixup_prob:
            image, label = self._apply_mixup(image, label, original_id, frame_idx)

        # Apply transforms
        image = self.transforms(image=image)["image"]

        if not self.return_label:
            return image

        label = torch.tensor(label, dtype=torch.float32)
        return image, label
