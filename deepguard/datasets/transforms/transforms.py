import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transform(cfg, 
                  mode: str = "train", 
                  tta_hflip: bool = False,
                  tta_scale: float = 1.0):
    """
    Returns an Albumentations Compose transform.

    Args:
        cfg: config object with img_size, mean, std
        mode (str): 'train', 'valid', 'test'
        tta_hflip (bool): apply horizontal flip (TTA option)
        tta_scale (float): scale factor for resizing (TTA option)
    """
    mode = str(mode).lower()
    if mode not in ["train", "valid", "test"]:
        raise ValueError("mode must be one of ['train', 'valid', 'test']")
    
    
    transforms_list = []
    
    if not hasattr(cfg, "img_size"):
        raise ValueError("You need to Img Size for Image Augmentation")
    
    if not hasattr(cfg, "mean") or not hasattr(cfg, "std"):
        raise ValueError("You need to mean, std for Image Preprocessing")
    

    # Resize
    if tta_scale != 1.0:
        h, w = cfg.img_size
        new_h, new_w = int(h * tta_scale), int(w * tta_scale)
        transforms_list.append(A.Resize(height=new_h, width=new_w))

    # Horizontal flip
    if mode == "train":
        transforms_list.append(A.HorizontalFlip(p=0.5))
        transforms_list.append(A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5))
        transforms_list.append(A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3))
    elif mode == "valid":
        pass
    elif mode == "test" and tta_hflip:
        transforms_list.append(A.HorizontalFlip(p=0.5))

    # Normalize + ToTensor
    transforms_list.extend([
        A.Normalize(mean=cfg.mean, std=cfg.std),
        ToTensorV2()
    ])

    return A.Compose(transforms_list)
