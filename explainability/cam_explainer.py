import torch
import cv2
import numpy as np
from abc import ABC, abstractmethod
from explainability.utils.model_targets import BinaryClassifierOutputTarget
from explainability.utils.image import show_cam_on_image, deprocess_image, remove_padding_and_resize
from .base_explainer import BaseExplainer
class CAMExplainer(BaseExplainer, ABC):
    def __init__(self, 
                 aug_smooth: bool = False,
                 eigen_smooth: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.aug_smooth = aug_smooth # aug_smooth has the effect of beter centering the CAM around the objects
        self.eigen_smooth = eigen_smooth # # eigen_smooth has the effect of removing a lot of noise
        self.cam = self._build_cam()
    
    @abstractmethod
    def _build_cam(self):
        ...

    def _build_grayscale_cam(self, img_path: str):
        """
        Generate a grayscale CAM and its associated image tensors.

        Returns:
            tuple: (grayscale_cam, tensor, face)
                - grayscale_cam (np.ndarray): CAM result in (H, W), range [0, 1].
                - tensor (torch.Tensor): Preprocessed input in (1, 3, H, W).
                - face (np.ndarray): Original image in (H, W, 3), uint8.
        """
        face, tensor = self._get_transform(img_path)
        tensor = tensor.to(self.device) # (1,3,H,W)
        targets = [BinaryClassifierOutputTarget(self.category)]
        
        grayscale_cam = self.cam(
            input_tensor = tensor, # (1,3,H,W)
            targets = targets,
            aug_smooth = self.aug_smooth,
            eigen_smooth = self.eigen_smooth
        ) # (1,H,W)
        
        return grayscale_cam[0], tensor, face 
        
    def display_heatmap(self, img_path: str, colormap: int = cv2.COLORMAP_JET, image_weight: float = 0.5,):
        grayscale_cam, tensor, face = self._build_grayscale_cam(img_path)     
        
        cam_recovered = remove_padding_and_resize(grayscale_cam, face.shape, tensor.shape)   
        
        heatmap = show_cam_on_image(
            np.float32(face) / 255.0,  # (H,W,3), range(0~1), np.ndarray
            cam_recovered, # (H,W), range(0~1), np.ndarray
            use_rgb =True, 
            colormap = colormap,
            image_weight= image_weight)
        
        return heatmap # (H,W,3), range(0~255), np.uint8
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"model={self.model_name}, dataset={self.dataset}, "
                f"margin_ratio={self.margin_ratio}, conf_thres={self.conf_thres}, "
                f"branch_level={self.branch_level}, l_stage_idx={self.l_stage_idx}, block_idx={self.block_idx},"
                f"aug_smooth={self.aug_smooth}, eigen_smooth={self.eigen_smooth}, "
                f"colormap={self.colormap}, image_weight={self.image_weight}, "
                f"category={self.category},device={self.device})")
        