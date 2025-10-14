import numpy as np
from PIL import Image
from typing import Optional, Tuple
import albumentations as A
import torch
from torch.utils.data import DataLoader, Dataset

class DFDCDataset(Dataset):

    """
    DFDC Dataset for Deepfake Image Classification.

    Args:
        df (pd.DataFrame): DataFrame containing 'image_path' and 'label' columns.
        transforms (albumentations.Compose, optional): Albumentations transforms including
            Resize, Normalize, ToTensorV2, etc. Must be provided.
    """
    
    def __init__(self, df, transforms: Optional[A.Compose]=None):
        self.df = df
        self.transforms = transforms
        if self.transforms is None:
            raise ValueError("Transforms must be provided for preprocessing")
    
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:

        row = self.df.iloc[index]
        img_path = row['image_path']
        label = row['label']

        # Load Image and convert to numpy array
        img = Image.open(img_path).convert("RGB")
        img = np.array(img, dtype=np.float32)

        # Apply transforms
        img = self.transforms(image=img)['image']

        # Convert Label to Tensor
        label = torch.tensor(label, dtype=torch.float32)

        return img, label