import os
import cv2
import timm
import torch
import numpy as np
from pathlib import Path
from typing import List
from preprocess.face_detector import FaceDetector
from deepguard.data import get_test_transforms
from .utils import PredictorError

class ImagePredictor:
    def __init__(
        self,
        margin_ratio: float, 
        conf_thres: float, 
        min_face_ratio: float,
        model_name: str, 
        dataset: str
    ):  
        self.device = "cuda:0" if torch.cuda.is_available() else 'cpu'
        self.face_detector = FaceDetector(conf_thres, min_face_ratio)
        self.margin_ratio = margin_ratio
        self.model = timm.create_model(model_name, pretrained=True, dataset=dataset)
        self.img_size = [224,224] if model_name.split("_")[-1] == "b0" else [384,384]
        
        # Model Inference Mode
        self.model.to(self.device)
        self.model.eval()
                
    def _get_face_bbox(self, img: np.ndarray) -> List[float]:
        """
            return [xmin, ymin, xmax, ymax] or None
        """
        
        face_bbox = self.face_detector.detect_batch([img], [1.0])
        if len(face_bbox[0]) == 0:
            return None
        
        return face_bbox[0]
    
    def _crop_face(self, img: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Crops the face. Returns None instead of raising an error if the crop is invalid.
        """
        xmin, ymin, xmax, ymax = bbox
            
        w = xmax - xmin
        h = ymax - ymin
        pad_w = int(w * self.margin_ratio)
        pad_h = int(h * self.margin_ratio)
            
        y1 = max(int(ymin - pad_h), 0)
        y2 = min(int(ymax + pad_h), img.shape[0]) 
        x1 = max(int(xmin - pad_w), 0)
        x2 = min(int(xmax + pad_w), img.shape[1])
            
        cropped_img = img[y1:y2, x1:x2]
            
        if cropped_img.size == 0:
            return None
        return cropped_img
        
    def _preprocess_img(self, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path)
        if img is None:
            raise PredictorError("이미지 파일을 읽을 수 없습니다")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        bbox = self._get_face_bbox(img)
        if bbox is None:
            raise PredictorError("이미지에서 얼굴을 감지하지 못했습니다.")
            
        cropped = self._crop_face(img, bbox)
        if cropped is None:
            raise PredictorError("얼굴이 화면 끝에 너무 붙어 있거나 가려져 있어 분석이 어렵습니다.")
                  
        
    def predict_img(self, img_path: str, tta_hflip: float = 0) -> float:
        
        try:
            img = self._preprocess_img(img_path)
            
            transforms = get_test_transforms(img_size=self.img_size, tta_hflip=tta_hflip)
            img = transforms(image=img)['image']
        
            with torch.no_grad():
                img = img.unsqueeze(0).to(self.device)
            
                out = self.model(img)
                pred = torch.sigmoid(out).item()
            
            return pred
        
        except PredictorError as e:
            raise e
        except Exception as e:
            raise PredictorError("분석 중 예상치 못한 오류 발생")