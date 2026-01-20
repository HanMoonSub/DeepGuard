import numpy as np
import pandas as pd
import random
from typing import List
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A

class DeepFakeDataset(Dataset):
    """
    Dataset class for Deepfake detection with Landmark-based Cutout.
    
    This class loads images and their corresponding labels, applying a specialized 
    Cutout augmentation targeting specific facial features (eyes, nose, mouth) 
    based on pre-extracted landmarks. This encourages the model to learn robust 
    representations rather than over-relying on a single facial region.

    Attributes:
        meta_df (pd.DataFrame): Dataframe containing 'img_path', 'label', and 'landmark_path'.
        img_size (List[int]): Target [Height, Width] for the transformation pipeline.
        transforms (A.Compose): Albumentations augmentation pipeline.
        img_col (str): Column name for the image file paths.
        label_col (str): Column name for the labels (0 for Real, 1 for Fake).
        landmark_col (str): Column name for the .npy files containing 5-point facial landmarks.
        cutout_prob (float): Probability of applying the landmark-based cutout augmentation.
        return_label (bool): If True, returns (image, label); otherwise, returns only image.
    """
    
    def __init__(
                self,
                meta_df: pd.DataFrame,
                img_size: List[int],
                transforms: A.Compose,
                img_col: str = "img_path",
                label_col: str = "label",
                landmark_col: str = "landmark_path",
                cutout_prob: float = 0.,
                return_label: bool = True,
                ):
        self.meta_df = meta_df
        self.img_size = img_size
        self.transforms = transforms(img_size)
        self.img_col = img_col
        self.label_col = label_col
        self.landmark_col = landmark_col
        self.cutout_prob = cutout_prob
        self.return_label = return_label

    def __len__(self):
        return len(self.meta_df)
    
    def _load_image(self, path: str):
        img_bgr = cv2.imread(path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        return img_rgb
    
    def _load_landmark(self, path: str):
        return np.load(path, allow_pickle=True).astype(np.float32)
        
    def _generate_hull(self, center, base_scale, shape_type='circle'):
        cx, cy = center
        points = []
        
        num_points = random.randint(6,12)
        
        for _ in range(num_points):
            theta = random.uniform(0, 2 * np.pi)
            distortion = random.uniform(0.8, 1.3)
            
            if shape_type == 'horizontal': # eye, mouth
                r_x = base_scale * random.uniform(1.2, 1.8) * distortion
                r_y = base_scale * random.uniform(0.5, 0.9) * distortion
            elif shape_type == 'vertical': # nose
                r_x = base_scale * random.uniform(0.6, 1.0) * distortion
                r_y = base_scale * random.uniform(1.2, 1.6) * distortion
            else: # random
                r_x = base_scale * random.uniform(0.8, 1.4) * distortion
                r_y = base_scale * random.uniform(0.8, 1.4) * distortion
                
            # angle jitter    
            rot_angle = random.uniform(-0.3, 0.3)
            
            local_x = r_x * np.cos(theta)
            local_y = r_y * np.sin(theta)
            
            rot_x = local_x * np.cos(rot_angle) - local_y * np.sin(rot_angle)
            rot_y = local_x * np.sin(rot_angle) + local_y * np.cos(rot_angle)
            
            pt_x = int(cx + rot_x)
            pt_y = int(cy + rot_y)
            points.append([pt_x, pt_y])
            
        points = np.array(points, dtype=np.int32)
        hull = cv2.convexHull(points)
        return hull
            
    def _apply_cutout(self, img: np.ndarray, landmark: np.ndarray):
        if landmark.shape[0] != 5 or np.sum(landmark) == 0:
            return img 
        
        out = img.copy()
        h, w, _ = out.shape
        
        eye_dist = np.linalg.norm(landmark[0] - landmark[1])
        base_scale = max(eye_dist * 0.20, 10.0)
        
        choice = random.choices(['l_eye', 'r_eye', 'nose', 'mouth', 'random'], 
                              weights=[0.225, 0.225, 0.225, 0.225, 0.1], k=1)[0]
        
        hulls = []
        if choice == "l_eye":
            hulls.append(self._generate_hull(landmark[0], base_scale, "horizontal"))
        elif choice == "r_eye":
            hulls.append(self._generate_hull(landmark[1], base_scale, "horizontal"))
        elif choice == "nose":
            hulls.append(self._generate_hull(landmark[2], base_scale, 'vertical'))
        elif choice == "mouth":
            mouth_center = (landmark[3] + landmark[4]) / 2
            mouth_w = np.linalg.norm(landmark[3] - landmark[4])
            hulls.append(self._generate_hull(mouth_center, mouth_w * 0.5, "horizontal"))
        elif choice == "random":
            idxs = np.random.choice(5,2, replace=False)
            for idx in idxs:
                hulls.append(self._generate_hull(landmark[idx], base_scale, 'circle'))
            
        cv2.fillPoly(out, hulls, (0,0,0))
        
        return out.astype(np.uint8)
        
        
    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        img_path = str(row[self.img_col])

        img = self._load_image(img_path)

        if np.random.rand() < self.cutout_prob:
            landmark_path = str(row[self.landmark_col])
            landmark = self._load_landmark(landmark_path)
            img = self._apply_cutout(img, landmark)
                
        img = self.transforms(image=img)['image']

        if not self.return_label:
            return img
        else:
            label = row[self.label_col]
            return img, torch.tensor(label, dtype=torch.float32)