import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, Tuple
import albumentations as A


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
        image_key: str = "image_path",
        label_key: str = "label",
        return_label: bool = True,
    ):
        self.meta_df = meta_df
        self.transforms = transforms
        self.image_key = image_key
        self.label_key = label_key
        self.return_label = return_label

        if self.transforms is None:
            raise ValueError("Transforms must be provided for preprocessing")

    def __len__(self) -> int:
        return len(self.meta_df)

    def load_image(self, path: str) -> np.ndarray:
        """Load image safely and convert to numpy array."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")

        img = Image.open(path).convert("RGB")
        img = np.array(img, dtype=np.float32)

        return img

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:

        row = self.meta_df.iloc[index]
        img_path = row[self.image_key]

        # Load image
        img = self.load_image(img_path)

        # Apply transforms
        img = self.transforms(image=img)["image"]

        # For Testing
        if not self.return_label:
            return img

        # Load label
        label = torch.tensor(row[self.label_key], dtype=torch.float32)

        return img, label