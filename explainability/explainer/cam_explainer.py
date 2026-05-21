from typing import Union, Literal
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cam = self._build_cam()
    
    @abstractmethod
    def _build_cam(self) -> any:
        """하위 클래스에서 각 알고리즘(GradCAM, HiResCAM 등)에 맞는 CAM 객체를 생성한다.

        Returns:
            Any: pytorch-grad-cam 패키지의 CAM 기반 인스턴스.
        """
        pass

    def _build_grayscale_cam(self, img_path: str,
                             category: int = 1,
                             aug_smooth: bool = False,
                             eigen_smooth: bool = False) -> tuple[np.ndarray, torch.Tensor, np.ndarray]:
        """이미지 경로를 받아 전처리를 수행하고 raw 흑백(Grayscale) CAM을 생성한다.

        Args:
            img_path (str): 입력 이미지 파일 경로.
            category (int, optional): 타겟 클래스 인덱스 (위조: 1, 정상: 0). Defaults to 1.
            aug_smooth (bool, optional): Test-time augmentation 가동 여부. Defaults to False.
            eigen_smooth (bool, optional): PCA 기반 노이즈 제거 가동 여부. Defaults to False.

        Returns:
            tuple[np.ndarray, torch.Tensor, np.ndarray]: 
                - grayscale_cam: (H, W) 형태의 0~1 사이로 정규화된 CAM 맵.
                - tensor: (1, 3, H_model, W_model) 형태의 모델 입력용 GPU 텐서.
                - face: 크롭된 원본 얼굴 이미지 배열 (RGB).
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
        """이미 크롭된 얼굴 이미지 배열로부터 raw 흑백(Grayscale) CAM을 생성한다.

        Args:
            face (np.ndarray): 전처리(크롭)가 완료된 얼굴 이미지 배열 (RGB).
            category (int, optional): 타겟 클래스 인덱스. Defaults to 1.
            aug_smooth (bool, optional): Test-time augmentation 가동 여부. Defaults to False.
            eigen_smooth (bool, optional): PCA 기반 노이즈 제거 가동 여부. Defaults to False.

        Returns:
            tuple[np.ndarray, torch.Tensor, np.ndarray]: 
                - grayscale_cam: (H, W) 형태의 0~1 사이로 정규화된 CAM 맵.
                - tensor: (1, 3, H_model, W_model) 형태의 모델 입력용 GPU 텐서.
                - face: 입력받은 원본 얼굴 이미지 배열.
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

    def _prepare_cam(self, img_path: str,
                     category: int = 1,
                     aug_smooth: bool = False,
                     eigen_smooth: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """흑백 CAM을 생성하고, 모델 입력 시 발생한 패딩 제거 및 원래 얼굴 해상도로 복원한다.

        Args:
            img_path (str): 입력 이미지 파일 경로.
            category (int, optional): 타겟 클래스 인덱스. Defaults to 1.
            aug_smooth (bool, optional): Test-time augmentation 가동 여부. Defaults to False.
            eigen_smooth (bool, optional): PCA 기반 노이즈 제거 가동 여부. Defaults to False.

        Returns:
            tuple[np.ndarray, np.ndarray]: 
                - cam_recovered: 원본 얼굴 크기로 복원된 CAM 맵 (H_face, W_face).
                - face: 크롭된 원본 얼굴 이미지 배열 (RGB).
        """
        grayscale_cam, tensor, face = self._build_grayscale_cam(img_path, category, aug_smooth, eigen_smooth)
        cam_recovered = remove_padding_and_resize(grayscale_cam, face.shape, tensor.shape)
        return cam_recovered, face
    
    def _prepare_cam_from_array(self, face: np.ndarray,
                                category: int = 1,
                                aug_smooth: bool = False,
                                eigen_smooth: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """얼굴 배열로부터 흑백 CAM을 생성하고 원래 얼굴 해상도로 복원한다.

        Args:
            face (np.ndarray): 전처리(크롭)가 완료된 얼굴 이미지 배열 (RGB).
            category (int, optional): 타겟 클래스 인덱스. Defaults to 1.
            aug_smooth (bool, optional): Test-time augmentation 가동 여부. Defaults to False.
            eigen_smooth (bool, optional): PCA 기반 노이즈 제거 가동 여부. Defaults to False.

        Returns:
            tuple[np.ndarray, np.ndarray]: 
                - cam_recovered: 원본 얼굴 크기로 복원된 CAM 맵 (H_face, W_face).
                - face: 입력받은 원본 얼굴 이미지 배열.
        """
        grayscale_cam, tensor, face = self._build_grayscale_cam_from_array(face, category, aug_smooth, eigen_smooth)
        cam_recovered = remove_padding_and_resize(grayscale_cam, face.shape, tensor.shape)
        return cam_recovered, face
    
    def _get_binary_mask(self, cam: np.ndarray, threshold: Union[float, Literal["auto"]]) -> np.ndarray:
        """입력된 CAM 맵을 기반으로 위조 유력 부위를 이진화(Binary Mask)한다.

        Args:
            cam (np.ndarray): 0~1 사이 값을 가진 복원된 CAM 맵.
            threshold (Union[float, 'auto']): 
                - float (0.0~1.0): 상위 백분위수 기준으로 임계값 설정 (예: 0.7 이면 상위 30% 영역 선택).
                - 'auto': Otsu 알고리즘을 사용해 최적의 임계값을 자동으로 계산.

        Returns:
            np.ndarray: 0 또는 255 값을 가지는 이진화 마스크 맵 (np.uint8).
        """
        if threshold != 'auto':
            t = np.percentile(cam, threshold * 100)
            binary_mask = (cam > t).astype(np.uint8) * 255
        else:
            _, binary_mask = cv2.threshold(
                np.uint8(cam * 255), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        return binary_mask

    def display_heatmap_on_image(self, img_path: str, **kwargs) -> np.ndarray:
        """이미지 경로를 입력받아 원본 얼굴 위에 위조 흔적 히트맵을 중첩한 이미지를 생성한다.

        Args:
            img_path (str): 입력 이미지 파일 경로.
            **kwargs:
                category (int): 타겟 클래스 인덱스 (기본값: 1).
                aug_smooth (bool): Test-time augmentation 가동 여부 (기본값: False).
                eigen_smooth (bool): PCA 기반 노이즈 제거 가동 여부 (기본값: False).
                threshold (float | str): 이진화 임계값 (기본값: "auto").
                colormap (int): OpenCV 컬러맵 상수 (기본값: cv2.COLORMAP_JET).
                image_weight (float): 원본 이미지 불투명도 가중치 (기본값: 0.5).

        Returns:
            np.ndarray: 히트맵이 합성된 얼굴 이미지 (RGB, np.uint8).
        """
        cam_recovered, face = self._prepare_cam(
            img_path, 
            category = kwargs.get("category", 1), 
            aug_smooth = kwargs.get("aug_smooth", False), 
            eigen_smooth = kwargs.get("eigen_smooth", False))
        
        binary_mask = self._get_binary_mask(cam_recovered, kwargs.get("threshold", "auto"))
        cam_recovered = np.where(binary_mask > 0, cam_recovered, 0.0)
                
        return show_cam_on_image(
            np.float32(face) / 255.0,
            cam_recovered,
            use_rgb = True,
            colormap = kwargs.get("colormap", cv2.COLORMAP_JET),
            image_weight = kwargs.get("image_weight", 0.5),
        )
        
    def display_heatmap_from_array(self, face: np.ndarray, **kwargs) -> np.ndarray:
        """얼굴 배열을 직접 입력받아 원본 얼굴 위에 위조 흔적 히트맵을 중첩한 이미지를 생성한다.

        Args:
            face (np.ndarray): 전처리(크롭)가 완료된 얼굴 이미지 배열 (RGB).
            **kwargs:
                category (int): 타겟 클래스 인덱스 (기본값: 1).
                aug_smooth (bool): Test-time augmentation 가동 여부 (기본값: False).
                eigen_smooth (bool): PCA 기반 노이즈 제거 가동 여부 (기본값: False).
                threshold (float | str): 이진화 임계값 (기본값: "auto").
                colormap (int): OpenCV 컬러맵 상수 (기본값: cv2.COLORMAP_JET).
                image_weight (float): 원본 이미지 불투명도 가중치 (기본값: 0.5).

        Returns:
            np.ndarray: 히트맵이 합성된 얼굴 이미지 (RGB, np.uint8).
        """
        
        cam_recovered, face = self._prepare_cam_from_array(
            face, 
            category = kwargs.get("category", 1), 
            aug_smooth = kwargs.get("aug_smooth", False), 
            eigen_smooth = kwargs.get("eigen_smooth", False))
        
        binary_mask = self._get_binary_mask(cam_recovered, kwargs.get("threshold", "auto"))
        cam_recovered = np.where(binary_mask > 0, cam_recovered, 0.0)
                
        return show_cam_on_image(
            np.float32(face) / 255.0,
            cam_recovered,
            use_rgb = True,
            colormap = kwargs.get("colormap", cv2.COLORMAP_JET),
            image_weight = kwargs.get("image_weight", 0.5),
        )  
    
    def display_bbox_on_image(self, img_path: str, **kwargs) -> np.ndarray:
        """이미지 경로를 입력받아 위조 의심 부위를 찾아 사각형 바운딩 박스와 확률을 그린다.

        Args:
            img_path (str): 입력 이미지 파일 경로.
            **kwargs:
                category (int): 타겟 클래스 인덱스 (기본값: 1).
                aug_smooth (bool): Test-time augmentation 가동 여부 (기본값: False).
                eigen_smooth (bool): PCA 기반 노이즈 제거 가동 여부 (기본값: False).
                threshold (float | str): 이진화 임계값 (기본값: "auto").
                thickness (int): 사각형 선 굵기 (기본값: 2).

        Returns:
            np.ndarray: 바운딩 박스와 어노테이션이 추가된 원본 얼굴 이미지 (RGB).
        """
        cam_recovered, face = self._prepare_cam(
            img_path, 
            category = kwargs.get("category", 1), 
            aug_smooth = kwargs.get("aug_smooth", False), 
            eigen_smooth = kwargs.get("eigen_smooth", False)
            )
        
        binary_mask = self._get_binary_mask(cam_recovered, kwargs.get("threshold", "auto"))
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        annotated_img = face.copy()
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            roi_prob = float(np.max(cam_recovered[y:y + h, x:x + w]))

            r = int(min(255, 2 * roi_prob * 255))
            g = int(min(255, 2 * (1 - roi_prob) * 255))
            box_color = (r, g, 30)  # RGB

            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), box_color, kwargs.get("thickness", 1))
            
            _draw_label(
                annotated_img,
                text=f"{roi_prob * 100:.1f}%",
                x=x,
                y=max(y - 6, 14),
                prob_ratio=roi_prob,
            )

        return annotated_img

    def display_bbox_from_array(self, face: np.ndarray, **kwargs) -> np.ndarray:
        """얼굴 배열을 직접 입력받아 위조 의심 부위를 찾아 사각형 바운딩 박스와 확률을 그린다.

        Args:
            face (np.ndarray): 전처리(크롭)가 완료된 얼굴 이미지 배열 (RGB).
            **kwargs:
                category (int): 타겟 클래스 인덱스 (기본값: 1).
                aug_smooth (bool): Test-time augmentation 가동 여부 (기본값: False).
                eigen_smooth (bool): PCA 기반 노이즈 제거 가동 여부 (기본값: False).
                threshold (float | str): 이진화 임계값 (기본값: "auto").
                thickness (int): 사각형 선 굵기 (기본값: 2).

        Returns:
            np.ndarray: 바운딩 박스와 어노테이션이 추가된 원본 얼굴 이미지 (RGB).
        """
        cam_recovered, face = self._prepare_cam_from_array(
            face, 
            category = kwargs.get("category", 1), 
            aug_smooth = kwargs.get("aug_smooth", False), 
            eigen_smooth = kwargs.get("eigen_smooth", False)
            )
        
        binary_mask = self._get_binary_mask(cam_recovered, kwargs.get("threshold", "auto"))
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        annotated_img = face.copy()
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            roi_prob = float(np.max(cam_recovered[y:y + h, x:x + w]))

            r = int(min(255, 2 * roi_prob * 255))
            g = int(min(255, 2 * (1 - roi_prob) * 255))
            box_color = (r, g, 30)  # RGB

            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), box_color, kwargs.get("thickness", 1))
            
            _draw_label(
                annotated_img,
                text=f"{roi_prob * 100:.1f}%",
                x=x,
                y=max(y - 6, 14),
                prob_ratio=roi_prob,
            )

        return annotated_img