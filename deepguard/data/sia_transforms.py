from typing import List

import cv2
import albumentations as A

from .transforms import _get_isotropical_resize, _get_normalization


def get_sia_train_transforms(
                            img_size: List[int],
                            mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5),
                        ):
    """
        Port of DeepfakeBench's training augmentation (init_data_aug_method) with sia.yaml values.

        DeepfakeBench resizes every face to the target resolution at load time, which makes its
        IsotropicResize step a no-op; DeepGuard crops are not square, so the isotropic resize
        (LongestMaxSize + PadIfNeeded) is applied first instead, same as the valid/test pipeline.
    """
    transforms = [
        *_get_isotropical_resize(img_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=(-10, 10), border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1)),
            A.FancyPCA(),
            A.HueSaturationValue(),
        ], p=0.5),
        A.ImageCompression(quality_range=(40, 100), p=0.5),
    ]

    return A.Compose([
        *transforms,
        *_get_normalization(mean, std),
    ])
