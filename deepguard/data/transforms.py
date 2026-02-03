from typing import List
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

def _get_isotropical_resize(img_size: List[int]):
    h, w = img_size
    return [
        A.LongestMaxSize(max_size=max(h, w)),
        A.PadIfNeeded(min_height=h, min_width=w,
                      border_mode=cv2.BORDER_CONSTANT, fill=0)
    ]

def _get_normalization():
    return [
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]

def get_train_transforms(img_size: List[int]):
    """
        Create a data augmentation pipeline for model training
    """
    transforms = [
        *_get_isotropical_resize(img_size),
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, scale=(0.8, 1.2), rotate=(-10, 10), border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.25),
    ]
    
    return A.Compose([
        *transforms,
        *_get_normalization(),
    ])

def get_valid_transforms(img_size: List[int]):
    """
        Create a minimal preprocessing pipeline for validation
    """
    
    return A.Compose([
        *_get_isotropical_resize(img_size),
        *_get_normalization()
    ])

def get_test_transforms(
                        img_size: List[int],
                        tta_hflip: float = 0.0,
                        tta_scale: float = 1.0,
                    ):  
    """ Creates a pipeline for inference and Test Time Augmentation (TTA).

    Args:
        img_size (List[int]): Base image size [height, width].
        tta_hflip (float): Probability of applying a horizontal flip. Defaults to 0.0.
        tta_scale (float): Scaling factor for the image. Defaults to 1.0.

    Returns:
        A.Compose: Pipeline with TTA options and normalization.
    """

    
    transforms = _get_isotropical_resize(img_size)
        
    if tta_scale != 1.0:
        h, w = img_size        
        transforms.append(A.Resize(height=int(h * tta_scale), width=int(w * tta_scale)))
    
    if tta_hflip != 0:
        transforms.append(A.HorizontalFlip(p=tta_hflip))
        
    return A.Compose([
        *transforms,
        *_get_normalization(),
        ])
