from typing import Union, Literal
import torch
import cv2
import numpy as np
from abc import ABC, abstractmethod
from pytorch_grad_cam import (
    HiResCAM, GradCAMElementWise, LayerCAM,
    EigenGradCAM, GradCAMPlusPlus, XGradCAM,
)
from explainability.utils.model_targets import BinaryClassifierOutputTarget
from explainability.utils.image import (
    show_cam_on_image, show_bbox_on_image,
    remove_padding_and_resize, draw_label
)
from explainability.explainer.base_explainer import BaseExplainer

# Fixed per-branch forgery artifact thresholds (top CAM percentile)
#   - low (local structure): 0.95 → only the top 5% is a suspected forgery region
#   - high (global structure): 0.8 → the top 20% is a suspected forgery region
DEFAULT_THRESHOLD = {"low": 0.95, "high": 0.8}


def _cam_kwargs(kwargs: dict) -> dict:
    """Pick out only the arguments used for CAM generation from display_* kwargs."""
    return {
        "category": kwargs.get("category", 1),
        "aug_smooth": kwargs.get("aug_smooth", False),
        "eigen_smooth": kwargs.get("eigen_smooth", False),
    }

class CAMDisplayMixin:
    """Shared functionality for visualizing CAM maps as heatmaps/bounding boxes.

    Classes using it only need to implement `_build_grayscale_cam(img_path, ...)` and
    `_build_grayscale_cam_from_array(face, ...)`.
    (CAMExplainer: single branch, BranchFusionExplainer: low/high branches combined)
    """

    def _prepare_cam(self, img_path: str,
                     category: int = 1,
                     aug_smooth: bool = False,
                     eigen_smooth: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Generate a grayscale CAM, remove the padding added for the model input, and restore it to the original face resolution.

        Args:
            img_path (str): input image file path.
            category (int, optional): target class index. Defaults to 1.
            aug_smooth (bool, optional): whether to run test-time augmentation. Defaults to False.
            eigen_smooth (bool, optional): whether to run PCA-based denoising. Defaults to False.

        Returns:
            tuple[np.ndarray, np.ndarray]: 
                - cam_recovered: CAM map restored to the original face size (H_face, W_face).
                - face: cropped original face image array (RGB).
        """
        grayscale_cam, tensor, face = self._build_grayscale_cam(img_path, category, aug_smooth, eigen_smooth)
        cam_recovered = remove_padding_and_resize(grayscale_cam, face.shape, tensor.shape)
        return cam_recovered, face
    
    def _prepare_cam_from_array(self, face: np.ndarray,
                                category: int = 1,
                                aug_smooth: bool = False,
                                eigen_smooth: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Generate a grayscale CAM from a face array and restore it to the original face resolution.

        Args:
            face (np.ndarray): preprocessed (cropped) face image array (RGB).
            category (int, optional): target class index. Defaults to 1.
            aug_smooth (bool, optional): whether to run test-time augmentation. Defaults to False.
            eigen_smooth (bool, optional): whether to run PCA-based denoising. Defaults to False.

        Returns:
            tuple[np.ndarray, np.ndarray]: 
                - cam_recovered: CAM map restored to the original face size (H_face, W_face).
                - face: the original face image array that was passed in.
        """
        grayscale_cam, tensor, face = self._build_grayscale_cam_from_array(face, category, aug_smooth, eigen_smooth)
        cam_recovered = remove_padding_and_resize(grayscale_cam, face.shape, tensor.shape)
        return cam_recovered, face
    
    def _get_binary_mask(self, cam: np.ndarray, threshold: Union[float, Literal["auto"]]) -> np.ndarray:
        """Binarize the likely forgery regions (binary mask) based on the given CAM map.

        Args:
            cam (np.ndarray): restored CAM map with values between 0 and 1.
            threshold (Union[float, 'auto']): 
                - float (0.0~1.0): sets the threshold by top percentile (e.g. 0.7 selects the top 30% region).
                - 'auto': computes the optimal threshold automatically with Otsu's algorithm.

        Returns:
            np.ndarray: binary mask map with values 0 or 255 (np.uint8).
        """
        if threshold != 'auto':
            t = np.percentile(cam, threshold * 100)
            binary_mask = (cam > t).astype(np.uint8) * 255
        else:
            _, binary_mask = cv2.threshold(
                np.uint8(cam * 255), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        return binary_mask

    def _threshold_of(self, kwargs: dict) -> Union[float, Literal["auto"]]:
        """If kwargs has no threshold, use this explainer branch's fixed value (DEFAULT_THRESHOLD)."""
        threshold = kwargs.get("threshold")
        return DEFAULT_THRESHOLD[self.branch_level] if threshold is None else threshold

    def _prepare_masked_cam(self, img_path: str, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Restore the CAM to the original face resolution and build the suspected forgery region binary mask with it.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: (cam_recovered, binary_mask, face)
        """
        cam_recovered, face = self._prepare_cam(img_path, **_cam_kwargs(kwargs))
        return cam_recovered, self._get_binary_mask(cam_recovered, self._threshold_of(kwargs)), face

    def _prepare_masked_cam_from_array(self, face: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Face array input version of _prepare_masked_cam."""
        cam_recovered, face = self._prepare_cam_from_array(face, **_cam_kwargs(kwargs))
        return cam_recovered, self._get_binary_mask(cam_recovered, self._threshold_of(kwargs)), face

    def display_heatmap_on_image(self, img_path: str, **kwargs) -> np.ndarray:
        """Take an image path and produce an image with the forgery artifact heatmap overlaid on the original face.

        Args:
            img_path (str): input image file path.
            **kwargs:
                category (int): target class index (default: 1).
                aug_smooth (bool): whether to run test-time augmentation (default: False).
                eigen_smooth (bool): whether to run PCA-based denoising (default: False).
                threshold (float | str): binarization threshold (default: fixed per-branch value, low 0.95 / high 0.8).
                colormap (int): OpenCV colormap constant (default: cv2.COLORMAP_JET).
                image_weight (float): opacity weight of the original image (default: 0.8).

        Returns:
            np.ndarray: face image with the heatmap blended in (RGB, np.uint8).
        """
        cam_recovered, binary_mask, face = self._prepare_masked_cam(img_path, **kwargs)
        cam_recovered = np.where(binary_mask > 0, cam_recovered, 0.0)
                
        return show_cam_on_image(
            np.float32(face) / 255.0,
            cam_recovered,
            use_rgb = True,
            colormap = kwargs.get("colormap", cv2.COLORMAP_JET),
            image_weight = kwargs.get("image_weight", 0.8),
        )
        
    def display_heatmap_from_array(self, face: np.ndarray, **kwargs) -> np.ndarray:
        """Take a face array directly and produce an image with the forgery artifact heatmap overlaid on the original face.

        Args:
            face (np.ndarray): preprocessed (cropped) face image array (RGB).
            **kwargs:
                category (int): target class index (default: 1).
                aug_smooth (bool): whether to run test-time augmentation (default: False).
                eigen_smooth (bool): whether to run PCA-based denoising (default: False).
                threshold (float | str): binarization threshold (default: fixed per-branch value, low 0.95 / high 0.8).
                colormap (int): OpenCV colormap constant (default: cv2.COLORMAP_JET).
                image_weight (float): opacity weight of the original image (default: 0.8).

        Returns:
            np.ndarray: face image with the heatmap blended in (RGB, np.uint8).
        """
        
        cam_recovered, binary_mask, face = self._prepare_masked_cam_from_array(face, **kwargs)
        cam_recovered = np.where(binary_mask > 0, cam_recovered, 0.0)
                
        return show_cam_on_image(
            np.float32(face) / 255.0,
            cam_recovered,
            use_rgb = True,
            colormap = kwargs.get("colormap", cv2.COLORMAP_JET),
            image_weight = kwargs.get("image_weight", 0.8),
        )  
    
    def display_bbox_on_image(self, img_path: str, **kwargs) -> np.ndarray:
        """Take an image path, find suspected forgery regions, and draw rectangular bounding boxes with probabilities.

        Args:
            img_path (str): input image file path.
            **kwargs:
                category (int): target class index (default: 1).
                aug_smooth (bool): whether to run test-time augmentation (default: False).
                eigen_smooth (bool): whether to run PCA-based denoising (default: False).
                threshold (float | str): binarization threshold (default: fixed per-branch value, low 0.95 / high 0.8).
                thickness (int): rectangle line thickness (default: 2).

        Returns:
            np.ndarray: original face image with bounding boxes and annotations added (RGB).
        """
        cam_recovered, binary_mask, face = self._prepare_masked_cam(img_path, **kwargs)
        annotated_img = face.copy()
        show_bbox_on_image(annotated_img, cam_recovered, binary_mask, kwargs.get("thickness", 1))

        return annotated_img

    def display_bbox_from_array(self, face: np.ndarray, **kwargs) -> np.ndarray:
        """Take a face array directly, find suspected forgery regions, and draw rectangular bounding boxes with probabilities.

        Args:
            face (np.ndarray): preprocessed (cropped) face image array (RGB).
            **kwargs:
                category (int): target class index (default: 1).
                aug_smooth (bool): whether to run test-time augmentation (default: False).
                eigen_smooth (bool): whether to run PCA-based denoising (default: False).
                threshold (float | str): binarization threshold (default: fixed per-branch value, low 0.95 / high 0.8).
                thickness (int): rectangle line thickness (default: 2).

        Returns:
            np.ndarray: original face image with bounding boxes and annotations added (RGB).
        """
        cam_recovered, binary_mask, face = self._prepare_masked_cam_from_array(face, **kwargs)
        annotated_img = face.copy()
        show_bbox_on_image(annotated_img, cam_recovered, binary_mask, kwargs.get("thickness", 1))

        return annotated_img
    
    def display_heatmap_bbox_on_image(self, img_path: str, **kwargs) -> np.ndarray:
        cam_recovered, binary_mask, face = self._prepare_masked_cam(img_path, **kwargs)
        cam_recovered = np.where(binary_mask > 0, cam_recovered, 0.0)
         
        heatmap = show_cam_on_image(
            np.float32(face) / 255.0,
            cam_recovered,
            use_rgb = True,
            colormap = kwargs.get("colormap", cv2.COLORMAP_JET),
            image_weight = kwargs.get("image_weight", 0.8),
        )
        
        show_bbox_on_image(heatmap, cam_recovered, binary_mask, kwargs.get("thickness", 1))
        return heatmap
    
    def display_heatmap_bbox_from_array(self, face: np.ndarray, **kwargs) -> np.ndarray:
        cam_recovered, binary_mask, face = self._prepare_masked_cam_from_array(face, **kwargs)
        cam_recovered = np.where(binary_mask > 0, cam_recovered, 0.0)
         
        heatmap = show_cam_on_image(
            np.float32(face) / 255.0,
            cam_recovered,
            use_rgb = True,
            colormap = kwargs.get("colormap", cv2.COLORMAP_JET),
            image_weight = kwargs.get("image_weight", 0.8),
        )
        
        show_bbox_on_image(heatmap, cam_recovered, binary_mask, kwargs.get("thickness", 1))
        return heatmap


class CAMExplainer(BaseExplainer, CAMDisplayMixin, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cam = self._build_cam()
    
    @abstractmethod
    def _build_cam(self) -> any:
        """Create the CAM object for each algorithm (GradCAM, HiResCAM, etc.) in subclasses.

        Returns:
            Any: CAM-based instance from the pytorch-grad-cam package.
        """
        pass

    def _build_grayscale_cam(self, img_path: str,
                             category: int = 1,
                             aug_smooth: bool = False,
                             eigen_smooth: bool = False) -> tuple[np.ndarray, torch.Tensor, np.ndarray]:
        """Take an image path, preprocess it, and generate a raw grayscale CAM.

        Args:
            img_path (str): input image file path.
            category (int, optional): target class index (fake: 1, real: 0). Defaults to 1.
            aug_smooth (bool, optional): whether to run test-time augmentation. Defaults to False.
            eigen_smooth (bool, optional): whether to run PCA-based denoising. Defaults to False.

        Returns:
            tuple[np.ndarray, torch.Tensor, np.ndarray]: 
                - grayscale_cam: CAM map of shape (H, W) normalized to 0~1.
                - tensor: GPU model input tensor of shape (1, 3, H_model, W_model).
                - face: cropped original face image array (RGB).
        """
        
        face, tensor = self._get_transform(img_path)
        tensor = tensor.to(self.device)
        targets = [BinaryClassifierOutputTarget(category)]
        
        grayscale_cam = self.cam(
            input_tensor=tensor,
            targets=targets,
            aug_smooth=aug_smooth,
            eigen_smooth=eigen_smooth,
        )
        return grayscale_cam[0], tensor, face
    
    def _build_grayscale_cam_from_array(self,
                                        face: np.ndarray,
                                        category: int = 1,
                                        aug_smooth: bool = False,
                                        eigen_smooth: bool = False) -> tuple[np.ndarray, torch.Tensor, np.ndarray]:
        """Generate a raw grayscale CAM from an already-cropped face image array.

        Args:
            face (np.ndarray): preprocessed (cropped) face image array (RGB).
            category (int, optional): target class index. Defaults to 1.
            aug_smooth (bool, optional): whether to run test-time augmentation. Defaults to False.
            eigen_smooth (bool, optional): whether to run PCA-based denoising. Defaults to False.

        Returns:
            tuple[np.ndarray, torch.Tensor, np.ndarray]: 
                - grayscale_cam: CAM map of shape (H, W) normalized to 0~1.
                - tensor: GPU model input tensor of shape (1, 3, H_model, W_model).
                - face: the original face image array that was passed in.
        """
        face, tensor = self._get_transform_from_array(face)
        tensor = tensor.to(self.device)
        targets = [BinaryClassifierOutputTarget(category)]
        
        grayscale_cam = self.cam(
            input_tensor=tensor,
            targets=targets,
            aug_smooth=aug_smooth,
            eigen_smooth=eigen_smooth,
        )
        return grayscale_cam[0], tensor, face


# ---------------------------------------------------------------------------
# Low-level branch: precise local activation (texture, blending, compression traces)
# ---------------------------------------------------------------------------

class HiResCAMExplainer(CAMExplainer):
    """
    Like GradCAM but element-wise multiply the activations with the gradients; provably guaranteed faithfulness for certain models
    """
    def _build_cam(self):
        return HiResCAM(model=self.model, target_layers=self.target_layers, reshape_transform=self.reshape_fn)


class GradCAMElementWiseExplainer(CAMExplainer):
    """
    Like GradCAM but element-wise multiply the activations with the gradients then apply a ReLU operation before summing
    """
    def _build_cam(self):
        return GradCAMElementWise(model=self.model, target_layers=self.target_layers, reshape_transform=self.reshape_fn)


class LayerCAMExplainer(CAMExplainer):
    """
    Spatially weight the activations by positive gradients.
    Works better especially in lower layers
    """
    def _build_cam(self):
        return LayerCAM(model=self.model, target_layers=self.target_layers, reshape_transform=self.reshape_fn)


# ---------------------------------------------------------------------------
# High-level branch: clean class-discriminative global map (lighting, geometry, shadows)
# ---------------------------------------------------------------------------

class EigenGradCAMExplainer(CAMExplainer):
    """
    Like EigenCAM but with class discrimination
    First principle component of Activations*Grad
    Looks like GradCAM, but cleaner
    """
    def _build_cam(self):
        return EigenGradCAM(model=self.model, target_layers=self.target_layers, reshape_transform=self.reshape_fn)


class GradCAMPlusPlusExplainer(CAMExplainer):
    """
    Like GradCAM but uses second order gradients
    """
    def _build_cam(self):
        return GradCAMPlusPlus(model=self.model, target_layers=self.target_layers, reshape_transform=self.reshape_fn)


class XGradCAMExplainer(CAMExplainer):
    """
    Like GradCAM but scale the gradients by the normalized activations
    """
    def _build_cam(self):
        return XGradCAM(model=self.model, target_layers=self.target_layers, reshape_transform=self.reshape_fn)
