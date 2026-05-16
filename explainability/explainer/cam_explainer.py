import torch
import cv2
import numpy as np
from abc import ABC, abstractmethod
from explainability.utils.model_targets import BinaryClassifierOutputTarget
from explainability.utils.image import show_cam_on_image, deprocess_image, remove_padding_and_resize
from explainability.explainer.base_explainer import BaseExplainer


def _draw_label(img: np.ndarray, text: str, x: int, y: int, prob_ratio: float = 1.0):
    """Confidence label with a clean pill-style background.
    
    prob_ratio: 0~1, controls bg color (green→red)
    """
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness  = 1
    pad        = 4

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    rx1, ry1 = x - pad,      y - th - pad
    rx2, ry2 = x + tw + pad, y + baseline + pad

    r = int(min(255, 2 * prob_ratio * 255))
    g = int(min(255, 2 * (1 - prob_ratio) * 255))
    bg_color = (r, g, 30)  # RGB: green → red

    overlay = img.copy()
    cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), bg_color, -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


class CAMExplainer(BaseExplainer, ABC):
    def __init__(self, 
                 aug_smooth: bool = False,
                 eigen_smooth: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.aug_smooth = aug_smooth
        self.eigen_smooth = eigen_smooth
        self.cam = self._build_cam()
    
    @abstractmethod
    def _build_cam(self):
        ...

    def _build_grayscale_cam(self, img_path: str):
        face, tensor = self._get_transform(img_path)
        tensor = tensor.to(self.device)
        targets = [BinaryClassifierOutputTarget(self.category)]
        
        grayscale_cam = self.cam(
            input_tensor=tensor,
            targets=targets,
            aug_smooth=self.aug_smooth,
            eigen_smooth=self.eigen_smooth,
        )
        return grayscale_cam[0], tensor, face

    def _prepare_cam(self, img_path: str):
        """Build grayscale CAM and recover it to original face resolution."""
        grayscale_cam, tensor, face = self._build_grayscale_cam(img_path)
        cam_recovered = remove_padding_and_resize(grayscale_cam, face.shape, tensor.shape)
        return cam_recovered, face
    
    def _get_binary_mask(self, cam: np.ndarray, threshold: float | str):
        if threshold != 'auto':
            t = np.percentile(cam, threshold * 100)
            binary_mask = (cam > t).astype(np.uint8) * 255
        else:
            _, binary_mask = cv2.threshold(
                np.uint8(cam * 255), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        return binary_mask

    def display_heatmap_on_image(self, 
                                 img_path: str, 
                                 colormap: int = cv2.COLORMAP_JET, 
                                 image_weight: float = 0.5,
                                 threshold: float | str = "auto"
                                 ):
        cam_recovered, face = self._prepare_cam(img_path)
        
        binary_mask = self._get_binary_mask(cam_recovered, threshold)
        cam_recovered = np.where(binary_mask > 0, cam_recovered, 0.0)
                
        heatmap = show_cam_on_image(
            np.float32(face) / 255.0,
            cam_recovered,
            use_rgb=True,
            colormap=colormap,
            image_weight=image_weight,
        )
        return heatmap  # (H,W,3), range(0~255), np.uint8
    
    def display_contour_on_image(self, 
                                 img_path: str,
                                 threshold: float | str = 'auto',
                                 thickness: int = 2):
        cam_recovered, face = self._prepare_cam(img_path)
        
        binary_mask = self._get_binary_mask(cam_recovered, threshold)
            
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        annotated_img = cv2.drawContours(face.copy(), contours, -1, (255, 0, 0), thickness)
        return annotated_img
    
    def display_bbox_on_image(self, 
                              img_path: str,
                              threshold: float | str = 'auto',
                              thickness: int = 2):
        cam_recovered, face = self._prepare_cam(img_path)
        
        binary_mask = self._get_binary_mask(cam_recovered, threshold)
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        annotated_img = face.copy()
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            roi_prob = float(np.max(cam_recovered[y:y + h, x:x + w]))

            r = int(min(255, 2 * roi_prob * 255))
            g = int(min(255, 2 * (1 - roi_prob) * 255))
            box_color = (r, g, 30)  # RGB

            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), box_color, thickness)
            _draw_label(
                annotated_img,
                text=f"{roi_prob * 100:.1f}%",
                x=x,
                y=max(y - 6, 14),
                prob_ratio=roi_prob,
            )

        return annotated_img

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"model={self.model_name}, dataset={self.dataset}, "
                f"margin_ratio={self.margin_ratio}, conf_thres={self.conf_thres}, "
                f"branch_level={self.branch_level}, l_stage_idx={self.l_stage_idx}, block_idx={self.block_idx},"
                f"aug_smooth={self.aug_smooth}, eigen_smooth={self.eigen_smooth}, "
                f"category={self.category},device={self.device})")