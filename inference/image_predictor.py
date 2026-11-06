import os
import cv2
import timm
import torch
import numpy as np
from pathlib import Path
from typing import List
from preprocess.face_detector import FaceDetector
from deepguard.data import get_test_transforms
from .utils import ClipPredictorMixin, to_artifact_scores

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
        try:
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
                print(f"[CropError] Resulting crop size is zero.")
                return None
            
            return cropped_img
        except Exception as e:
            print(f"[CropError] Cropping failed: {e}")
            return None
    
    def _preprocess_img(self, img_path: str) -> np.ndarray:
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"[ImageError] Image not found or invalid: {img_path}")
                return None
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            bbox = self._get_face_bbox(img)
            if bbox is None:
                return None
            
            return self._crop_face(img, bbox)
                  
        except Exception as e:
            print(f"[Error] Preprocess Failed for {img_path}: {e}")
            return None
        
    def predict_img(self, img_path: str, tta_hflip: float = 0) -> float:
        
        try:
            img = self._preprocess_img(img_path)
            if img is None:
                return 0.5
            
            transforms = get_test_transforms(img_size=self.img_size, tta_hflip=tta_hflip)
            img = transforms(image=img)['image']
        
            img = img.unsqueeze(0).to(self.device)
            pred = float(self._forward(img)[0])

            return pred
        
        except Exception as e:
           print(f"[Critical Skip] Inference failed for {img_path}: {e}")
           return 0.5
       
    def __repr__(self):
        return (f"ImagePredictor(model_name='{self.model_name}', "
                f"device='{self.device}', "
                f"img_size={self.img_size})")

class ClipImagePredictor(ClipPredictorMixin, ImagePredictor):
    def __init__(
        self,
        margin_ratio: float,
        conf_thres: float,
        min_face_ratio: float,
        model_name: str,
    ):
        self._check_model_name(model_name)
        super().__init__(margin_ratio, conf_thres, min_face_ratio, model_name, dataset=None)

    def predict_img_artifacts(self, img_path: str, tta_hflip: float = 0) -> dict:
        """
        Returns {"prob": fake probability, "artifacts": {region: {artifact type: probability}}}.
        artifacts is None when no face is found (prob is 0.5, same as predict_img).
        """
        try:
            img = self._preprocess_img(img_path)
            if img is None:
                return {"prob": 0.5, "artifacts": None}

            transforms = get_test_transforms(img_size=self.img_size, tta_hflip=tta_hflip)
            img = transforms(image=img)['image'].unsqueeze(0).to(self.device)
            probs, artifact_probs = self._forward_artifacts(img)

            return {"prob": float(probs[0]), "artifacts": to_artifact_scores(artifact_probs[0])}

        except Exception as e:
            print(f"[Critical Skip] Inference failed for {img_path}: {e}")
            return {"prob": 0.5, "artifacts": None}

    def __repr__(self):
        return super().__repr__().replace("ImagePredictor", "ClipImagePredictor", 1)
