import numpy as np
import pandas as pd
from typing import List
from collections import defaultdict
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A

class MixUpDeepFakeDataset(Dataset):
    """
    Dataset class for Deepfake detection.
    Reads image paths and labels from a pandas DataFrame, loads images, 
    and applies specified transformations and Mixup augmentation.

    The Mixup implementation here is specialized for Deepfakes: it pairs a frame 
    with its counterpart (Real vs. Fake) from the same video and frame index 
    to help the model learn fine-grained discriminative features.

    Attributes:
        meta_df (pd.DataFrame): Dataframe containing image paths, labels, original video IDs, and frame indices.
        img_size (List[int]): Target [Height, Width] for the image transformation.
        transforms (A.Compose): Albumentations transformation pipeline.
        img_col (str): Column name for image file paths.
        label_col (str): Column name for labels (0 for Real, 1 for Fake).
        ori_vid_col (str): Column name identifying the source video.
        frame_idx_col (str): Column name for the frame index within the video.
        return_label (bool): Whether to return (image, label) or just the image.
        mixup_prob (float): Probability of applying Mixup augmentation (0.0 to 1.0).
        mixup_alpha (float): Alpha parameter for the Beta distribution used in Mixup.
        groups (dict): Pre-computed mapping of (video, frame, label) to image paths for efficient Mixup pairing.
    """
    def __init__(
                self,
                meta_df: pd.DataFrame,
                img_size: List[int],
                transforms: A.Compose,
                img_col: str = "img_path",
                label_col: str = "label",
                ori_vid_col: str = "ori_vid",
                frame_idx_col: str = "frame_idx",
                return_label: bool = True,
                mixup_prob: float = 0.0,
                mixup_alpha: float = 0.5,
                ):
        self.meta_df = meta_df
        self.img_size = img_size
        self.transforms = transforms(img_size)
        self.img_col = img_col
        self.label_col = label_col
        self.ori_vid_col = ori_vid_col
        self.frame_idx_col = frame_idx_col
        self.return_label = return_label
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha
        
        if self.mixup_prob > 0.:
            self.groups = self.meta_df.groupby([self.ori_vid_col, self.frame_idx_col, self.label_col])[self.img_col].apply(list).to_dict()
    
    def __len__(self):
        return len(self.meta_df)
    
    def _load_image(self, path: str):
        img_bgr = cv2.imread(path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
        return img_rgb
      
    def _apply_mixup(self, 
                     img: np.ndarray, 
                     label: float,
                     ori_vid: str,
                     frame_idx: int,
                     ):
        opposite_label = 1 - int(label)
        mu_paths = self.groups.get((ori_vid, frame_idx, opposite_label))
        
        if not mu_paths:
            return img, label
        
        mu_path = np.random.choice(mu_paths)
        mu_img = self._load_image(mu_path)
        mu_label = float(opposite_label)
        
        if img.shape != mu_img.shape:
            return img, label
        
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        
        mixed_img = img.astype(np.float32) * lam + mu_img.astype(np.float32) * (1 - lam)
        mixed_label = label * lam + mu_label * (1 - lam)
        
        return mixed_img.astype(np.uint8), mixed_label
        
    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        img_path = str(row[self.img_col])
        label = float(row[self.label_col])
        
        img = self._load_image(img_path)
        
        if np.random.rand() < self.mixup_prob:
            ori_vid = str(row[self.ori_vid_col])
            frame_idx = int(row[self.frame_idx_col])
            img, label = self._apply_mixup(img, label, ori_vid, frame_idx)
            
        img = self.transforms(image=img)['image']
        
        if not self.return_label:
            return img
        else:
            return img, torch.tensor(label, dtype=torch.float32)
        
    def __str__(self):
        return f"MixUp DeepFakeDataset with {len(self.meta_df)} samples"