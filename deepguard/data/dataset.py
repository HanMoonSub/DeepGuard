import numpy as np
import pandas as pd
from typing import List
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A

class DeepFakeDataset(Dataset):
    """
    Dataset class for Deepfake detection.
    Reads image paths and labels from a pandas DataFrame, loads images, 
    and applies specified transformations.

    Attributes:
        meta_df (pd.DataFrame): Dataframe containing image paths and labels.
        transforms (A.Compose): Albumentations transformation pipeline.
        img_size (List[int]): Default [Height, Width] to use if image loading fails.
        img_col (str): Name of the column containing image file paths.
        label_col (str): Name of the column containing labels.
        return_label (bool): Whether to return (image, label) or just image.
    """
    def __init__(
                self,
                meta_df: pd.DataFrame,
                img_size: List[int],
                transforms: A.Compose,
                img_col: str = "img_path",
                label_col: str = "label",
                return_label: bool = True,
                ):
        self.meta_df = meta_df
        self.img_size = img_size
        self.transforms = transforms(img_size)
        self.img_col = img_col
        self.label_col = label_col
        self.return_label = return_label

    def __len__(self):
        return len(self.meta_df)
    
    def _load_image(self, path: str):
        img_bgr = cv2.imread(path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        return img_rgb
    
    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        img_path = str(row[self.img_col])
        
        img = self._load_image(img_path)
        img = self.transforms(image=img)['image']
        
        if not self.return_label:
            return img
        else:
            label = row[self.label_col]
            return img, torch.tensor(label, dtype=torch.float32)
        
    def __str__(self):
        return f"DeepFakeDataset with {len(self.meta_df)} samples"