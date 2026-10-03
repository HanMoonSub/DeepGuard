import random
from typing import List, Optional

import cv2
import numpy as np
import albumentations as A
import torchvision.transforms as T

from .transforms import _get_isotropical_resize


class _CoreView:
    """
        One augmented view, following the original CORE pipeline order:
        geometric/photometric aug on uint8 RGB -> ToTensor([0,1]) -> (RandomErasing) -> Normalize.
    """

    def __init__(self, aug: A.Compose, mean, std, erasing: Optional[T.RandomErasing] = None):
        self.aug = aug
        self.to_tensor = T.ToTensor()
        self.erasing = erasing
        self.normalize = T.Normalize(mean=mean, std=std)

    def __call__(self, image: np.ndarray):
        x = self.aug(image=image)['image']
        x = self.to_tensor(x)
        if self.erasing is not None:
            x = self.erasing(x)
        return self.normalize(x)


class _OneOfView:
    """Randomly select one of the given views (original CORE's OneOfTrans)"""

    def __init__(self, views: List[_CoreView]):
        self.views = views

    def __call__(self, image: np.ndarray):
        return self.views[random.randint(0, len(self.views) - 1)](image)


class TwoTransform:
    """
        Apply the same random augmentation twice to get two views of one image.
        Keeps the albumentations-style interface used by DeepFakeDataset:
        transforms(image=img)['image'] -> [view1, view2]
    """

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, image: np.ndarray):
        return {'image': [self.base_transform(image), self.base_transform(image)]}


def _random_erasing(p: float):
    return T.RandomErasing(p=p, scale=(0.02, 0.20), ratio=(0.5, 2.0))


def _rand_crop(img_size: List[int]):
    h, w = img_size
    return A.RandomResizedCrop(size=(h, w), scale=(1 / 1.3, 1.0), ratio=(0.9, 1.1))


def get_core_train_transforms(
                            img_size: List[int],
                            aug_name: str = "RE",
                            mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5),
                        ):
    """
        Port of the original CORE (niyunsheng/CORE, src/transform.py) training augmentations,
        wrapped in TwoTransform for the consistency loss.

        The original uses Resize(IMG_SIZE) on square face crops; DeepGuard crops are not square,
        so the isotropic resize (LongestMaxSize + PadIfNeeded) is used instead to avoid distortion.

    Args:
        img_size (List[int]): [Height, Width].
        aug_name (str): One of "None", "RE", "RandCrop", "RaAug", "DFDC_selim".
        mean, std: Normalization values (CORE uses 0.5).
    """
    h, w = img_size

    if aug_name == "None":
        view = _CoreView(A.Compose(_get_isotropical_resize(img_size)), mean, std)

    elif aug_name == "RE":
        view = _CoreView(
            A.Compose([*_get_isotropical_resize(img_size), A.HorizontalFlip(p=0.5)]),
            mean, std,
            erasing=_random_erasing(p=0.8),
        )

    elif aug_name == "RandCrop":
        view = _CoreView(A.Compose([_rand_crop(img_size), A.HorizontalFlip(p=0.5)]), mean, std)

    elif aug_name == "RaAug":
        view = _OneOfView([
            _CoreView(A.Compose([*_get_isotropical_resize(img_size), A.HorizontalFlip(p=0.5)]), mean, std),
            _CoreView(
                A.Compose([*_get_isotropical_resize(img_size), A.HorizontalFlip(p=0.5)]),
                mean, std,
                erasing=_random_erasing(p=1.0),
            ),
            _CoreView(A.Compose([_rand_crop(img_size), A.HorizontalFlip(p=0.5)]), mean, std),
        ])

    elif aug_name == "DFDC_selim":
        # DFDC 1st place (selimsef) augmentation, as used by the original CORE repo
        view = _CoreView(
            A.Compose([
                A.ImageCompression(quality_range=(60, 100), p=0.5),
                A.GaussNoise(p=0.1),
                A.GaussianBlur(blur_limit=(3, 3), p=0.05),
                A.HorizontalFlip(p=0.5),
                A.OneOf([
                    A.LongestMaxSize(max_size=max(h, w), interpolation=cv2.INTER_CUBIC),
                    A.LongestMaxSize(max_size=max(h, w), interpolation=cv2.INTER_AREA),
                    A.LongestMaxSize(max_size=max(h, w), interpolation=cv2.INTER_LINEAR),
                ], p=1.0),
                A.PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_CONSTANT, fill=0),
                A.OneOf([A.RandomBrightnessContrast(), A.FancyPCA(), A.HueSaturationValue()], p=0.7),
                A.ToGray(p=0.2),
                A.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, scale=(0.8, 1.2),
                         rotate=(-10, 10), border_mode=cv2.BORDER_CONSTANT, p=0.5),
            ]),
            mean, std,
        )

    else:
        raise NotImplementedError(f"Unsupported CORE aug_name: {aug_name}")

    return TwoTransform(view)
