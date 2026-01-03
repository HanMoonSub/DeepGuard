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
    def __init__(
        self,
        meta_df,
        transforms: A.Compose,
        image_col: str = "image_path",
        label_col: str = "label",
        origin_col: str = "origin_vid",
        frame_idx_col: str = "frame_idx",
        landmark_col: str = "landmark_path",
        return_label: bool = True,
     
        mixup_prob: float = 0.0,
        mixup_alpha: float = 0.5,
        cutout_prob: float = 0.0,
        mixcut_prob: float = 0.0, 
        
        img_size: tuple = (384, 384),
    ):
        self.meta_df = meta_df.reset_index(drop=True)
        self.transforms = transforms
        self.image_col = image_col
        self.label_col = label_col
        self.origin_col = origin_col
        self.frame_idx_col = frame_idx_col
        self.landmark_col = landmark_col
        self.return_label = return_label
        
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha
        self.cutout_prob = cutout_prob
        self.mixcut_prob = mixcut_prob
        
        self.img_size = img_size
        
        if self.transforms is None:
            raise ValueError("Albumentations transforms must be provided!")
        
        self.original_groups = defaultdict(lambda: {"real": [], "fake": []})
        
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

    def __len__(self):
        return len(self.meta_df)

    def load_image(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")

        try:
            img = Image.open(path).convert("RGB")
            return np.array(img, dtype=np.uint8)

        except UnidentifiedImageError:
            return np.zeros((*self.img_size, 3), dtype=np.uint8)
            
    def resize_image(self, img: np.ndarray):
        H, W = self.img_size
        transform = A.Compose([
            A.LongestMaxSize(max_size=max(H, W)),
            A.PadIfNeeded(min_height=H, min_width=W, border_mode=cv2.BORDER_CONSTANT, value=0)
        ])
        return transform(image=img)["image"]

    def load_landmark(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Landmark not found: {path}")
        
        lm = np.load(path, allow_pickle=True).astype(np.float32)
        
        if lm.shape[0] != 5:
            return None
        else:
            return lm

    def _apply_mixup(self, img: np.ndarray, label, original_id, frame_idx):

        if np.sum(img) == 0:
            return img, label
        
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

        if np.sum(opp_img) == 0:
            return img, label

        # mixup
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        mixed_img = img * lam + opp_img * (1 - lam)
        mixed_label = label * lam + opposite_label * (1 - lam)

        return mixed_img.astype(np.uint8), float(mixed_label)

    def _choose_cutout_part(self):
        parts = ["left_eye", "right_eye", "nose", "mouth"]
        return np.random.choice(parts)


    def _apply_face_cutout(self, img: np.ndarray, landmark):
   
        if np.sum(img) == 0 or landmark is None:
            return img

        out = img.copy()
        part = self._choose_cutout_part()

        le, re, nose, lm, rm = landmark
        h, w = img.shape[:2]

        if part == "left_eye":
            center = le
            radius = int(np.linalg.norm(le - re) * 0.3)
        elif part == "right_eye":
            center = re
            radius = int(np.linalg.norm(le - re) * 0.3)
        elif part == "nose":
            center = nose
            radius = int(np.linalg.norm(le - nose) * 0.6)
        elif part == "mouth":
            center = ((lm + rm) / 2)
            radius = int(np.linalg.norm(lm - rm) * 0.6)

    
        num_points = np.random.randint(4, 7)
        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        angles += np.random.rand(num_points) * (2*np.pi / num_points)  

        polygon = []
        for angle in angles:
            x = int(center[0] + radius * np.cos(angle) * np.random.uniform(0.7,1.0))
            y = int(center[1] + radius * np.sin(angle) * np.random.uniform(0.7,1.0))
            x = np.clip(x, 0, w-1)
            y = np.clip(y, 0, h-1)
            polygon.append([x, y])

        polygon = np.array([polygon], dtype=np.int32)
        cv2.fillPoly(out, polygon, (0,0,0))

        return out.astype(np.uint8)



    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        img_path = row[self.image_col]
        label = float(row[self.label_col])
    
        img = self.load_image(img_path)
        
        if np.random.rand() < self.mixcut_prob:

            aug = np.random.choice(['mixup', 'cutout'])

            if aug == 'mixup' and np.random.rand() < self.mixup_prob:
                original_id = row[self.origin_col]
                frame_idx = row[self.frame_idx_col]
                img = self.resize_image(img)
                img, label = self._apply_mixup(img, label, original_id, frame_idx)
            
            elif aug == 'cutout' and np.random.rand() < self.cutout_prob:
                lm = self.load_landmark(row[self.landmark_col])
                img = self._apply_face_cutout(img, lm)
        
        img = self.resize_image(img)
        img = self.transforms(image=img)["image"]

        if not self.return_label:
            return img
            
        return img, torch.tensor(label, dtype=torch.float32)