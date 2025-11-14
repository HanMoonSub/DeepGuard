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
    transforms_list = []

    # Resize
    h, w = cfg.img_size
    new_h, new_w = int(h * tta_scale), int(w * tta_scale)
    transforms_list.append(A.Resize(new_h, new_w))

    # Horizontal flip
    if mode == "train":
        transforms_list.append(A.HorizontalFlip(p=0.5))
    elif mode in ["valid", "test"] and tta_hflip:
        transforms_list.append(A.HorizontalFlip(p=1.0))

    # Normalize + ToTensor
    transforms_list.extend([
        A.Normalize(mean=cfg.mean, std=cfg.std),
        ToTensorV2()
    ])

    return A.Compose(transforms_list)
