import os
import cv2
import timm
import torch
import numpy as np
from pathlib import Path
from typing import List
from preprocess.face_detector import FaceDetector2
from deepguard.data import get_test_transforms

class ImageExplainer:
    def __init__(
        self,
        model_name: str, 
        dataset: str, # celeb_df_v2, ff++, kodf
        cam_type: str, # [GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad]
        aug_smooth: bool, # This has the effect of better centering the CAM around the objects
        eigen_smooth: bool, # This has the effect of removing a lot of noise
        margin_ratio: float = 0.2, 
        conf_thres: float = 0.5, 
    ):  
        self.device = "cuda:0" if torch.cuda.is_available() else 'cpu'
        self.face_detector = FaceDetector2(conf_thres)
        self.margin_ratio = margin_ratio
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=True, dataset=dataset)
        self.img_size = [224,224] if model_name.split("_")[-1] == "b0" else [384,384]
        
        # Model Inference Mode
        self.model.to(self.device)
        self.model.eval()
                
    def _get_face_bbox(self, img: np.ndarray) -> List[float]:
        """
            return [xmin, ymin, xmax, ymax, confidence] or None
        """
        
        result = self.face_detector.detect_batch([img], [1.0])
        return result[0]
    
    def _crop_face(self, img: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Crops the face. Returns None instead of raising an error if the crop is invalid.
        """
        xmin, ymin, xmax, ymax = bbox
            
        img_h, img_w = img.shape[:2]
        
        # --- 이후 크롭 로직 (마진 적용 및 안전하게 클리핑) ---
        w = xmax - xmin
        h = ymax - ymin
    
        pad_w = int(w * self.margin_ratio)
        pad_h = int(h * self.margin_ratio)
    
        # 마진을 포함하되, 이미지 밖으로 나가는 부분은 그냥 잘라냄 (np.clip과 유사)
        y1 = max(int(ymin - pad_h), 0)
        y2 = min(int(ymax + pad_h), img_h)
        x1 = max(int(xmin - pad_w), 0)
        x2 = min(int(xmax + pad_w), img_w)
    
        cropped_img = img[y1:y2, x1:x2]
    
        return cropped_img
        
    def _preprocess_img(self, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path)        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detect_result = self._get_face_bbox(img)                
        bbox = detect_result[:4]
        cropped = self._crop_face(img, bbox)
        
        return cropped
                  
        
    def _get_transform(self, img_path: str) -> float:
        
        img = self._preprocess_img(img_path)
            
        transforms = get_test_transforms(img_size=self.img_size, tta_hflip=0)
        img = transforms(image=img)['image'] # (3,H,W)
        return img.unsqueeze(0) # (1,3,H,W)
            
        
    def __repr__(self):
        return "Display Fogery Region where AI model focus on watching"