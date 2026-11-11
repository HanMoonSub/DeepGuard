import os
import cv2
import timm
import torch
import numpy as np
from pathlib import Path
from typing import List
from preprocess.face_detector import FaceDetector2
from preprocess.landmark_detector import LandmarkDetector
from deepguard.data import get_test_transforms
from .utils import ClipPredictorMixin, PredictorError

class ImagePredictor:
    def __init__(
        self,
        margin_ratio: float, 
        conf_thres: float, 
        model_name: str, 
        dataset: str
    ):  
        self.device = "cuda:0" if torch.cuda.is_available() else 'cpu'
        self.face_detector = FaceDetector2(conf_thres)
        self.landmark_detector = LandmarkDetector()
        self.margin_ratio = margin_ratio
        self.model_name = model_name
        self.model = self._load_model(model_name, dataset)
        self.img_size = [224,224] if model_name.split("_")[-1] == "b0" else [384,384]
        
        # Model Inference Mode
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self, model_name: str, dataset: str):
        return timm.create_model(model_name, pretrained=True, dataset=dataset)

    def _forward(self, batch: torch.Tensor) -> np.ndarray:
        """(B,3,H,W) -> (B,) fake probabilities"""
        with torch.no_grad():
            return torch.sigmoid(self.model(batch).view(-1)).cpu().numpy()

    def _get_face_bbox(self, img: np.ndarray) -> List[float]:
        """
            return [xmin, ymin, xmax, ymax, confidence] or None
        """
        
        result = self.face_detector.detect_batch([img], [1.0])
        if len(result[0]) == 0:
            return None
        
        return result[0]
    
    def _crop_face(self, img: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Crops the face. Returns None instead of raising an error if the crop is invalid.
        """
        xmin, ymin, xmax, ymax = bbox
            
        img_h, img_w = img.shape[:2]
        
        # --- Crop logic (apply the margin and clip safely) ---
        w = xmax - xmin
        h = ymax - ymin
    
        pad_w = int(w * self.margin_ratio)
        pad_h = int(h * self.margin_ratio)
    
        # Include the margin, but simply cut off any part outside the image (similar to np.clip)
        y1 = max(int(ymin - pad_h), 0)
        y2 = min(int(ymax + pad_h), img_h)
        x1 = max(int(xmin - pad_w), 0)
        x2 = min(int(xmax + pad_w), img_w)
    
        cropped_img = img[y1:y2, x1:x2]
    
        return cropped_img
        
    def _preprocess_img(self, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path)
        if img is None:
            raise PredictorError(
            "이미지를 불러올 수 없습니다. 파일이 손상되었거나 지원하지 않는 형식인지 확인해 주세요."
        )
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        detect_result = self._get_face_bbox(img)
        if detect_result is None:
            raise PredictorError(
                "이미지에서 얼굴을 인식하지 못해 분석을 진행할 수 없습니다. "
                "얼굴이 정면을 향하고 이목구비가 뚜렷하게 보이는 이미지가 필요합니다."
            )
                    
        bbox = detect_result[:4]
        cropped = self._crop_face(img, bbox)
        
        keypoint = self.landmark_detector.detect_batch([cropped])[0]
        if len(keypoint) == 0:
            raise PredictorError(
                "얼굴은 감지되었으나, 일부분이 가려져 있거나 옆모습을 향하고 있어 정밀 분석이 어렵습니다. "
                "이목구비가 전체적으로 잘 보이는 이미지가 필요합니다."
            )

        return cropped
                  
        
    def predict_img(self, img_path: str, tta_hflip: float = 0) -> float:
        
        try:
            img = self._preprocess_img(img_path)

            transforms = get_test_transforms(img_size=self.img_size, tta_hflip=tta_hflip)
            img = transforms(image=img)['image']
        
            img = img.unsqueeze(0).to(self.device)
            pred = float(self._forward(img)[0])

            return {"prob": pred}
        
        except PredictorError as e:
            raise e
        except Exception as e:
            raise e
        
    def __repr__(self):
        return (f"ImagePredictor(model_name='{self.model_name}', "
                f"device='{self.device}', "
                f"img_size={self.img_size})")

class ClipImagePredictor(ClipPredictorMixin, ImagePredictor):
    def __init__(
        self,
        margin_ratio: float,
        conf_thres: float,
        model_name: str,
    ):
        self._check_model_name(model_name)
        super().__init__(margin_ratio, conf_thres, model_name, dataset=None)

    def predict_img_artifacts(self, img_path: str, tta_hflip: float = 0) -> dict:
        """
        Returns the probabilities of the 20 artifact texts:
            artifact_probs : list of 20 floats in PAIR_ORDER (idx = 5 * region + type)
        """
        img = self._preprocess_img(img_path)

        transforms = get_test_transforms(img_size=self.img_size, tta_hflip=tta_hflip)
        img = transforms(image=img)['image'].unsqueeze(0).to(self.device)
        _, artifact_probs = self._forward_artifacts(img)

        return {"artifact_probs": artifact_probs[0].tolist()}

    def __repr__(self):
        return super().__repr__().replace("ImagePredictor", "ClipImagePredictor", 1)
