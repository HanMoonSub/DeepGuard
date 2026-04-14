import os
import cv2
import timm
import torch
import numpy as np
from pathlib import Path
from typing import List
from preprocess.face_detector import FaceDetector2
from deepguard.data import get_test_transforms
from .utils import PredictorError

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
        self.margin_ratio = margin_ratio
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
        if len(result[0]) == 0:
            return None
        
        return result[0]
    
    def _crop_face(self, img: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Crops the face. Returns None instead of raising an error if the crop is invalid.
        """
        xmin, ymin, xmax, ymax = bbox
            
        img_h, img_w = img.shape[:2]
        
        threshold = 5 
    
        out_of_bounds = []
    
        # 마진 계산 전, bbox 자체가 경계에 너무 붙어있는지 검사
        if ymin <= threshold: out_of_bounds.append("상단")
        if ymax >= img_h - threshold: out_of_bounds.append("하단")
        if xmin <= threshold: out_of_bounds.append("왼쪽")
        if xmax >= img_w - threshold: out_of_bounds.append("오른쪽")
    
        # bbox가 구석에 너무 치우쳐 있다면 차단
        if out_of_bounds:
            return None, out_of_bounds
        
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
    
        return cropped_img, []
        
    def _preprocess_img(self, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path)
        if img is None:
            raise PredictorError(
            "이미지를 불러올 수 없습니다. 파일이 손상되었거나 지원하지 않는 형식인지 확인해 주세요."
        )
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]
        total_area = img_h * img_w
            
        detect_result = self._get_face_bbox(img)
        if detect_result is None:
            raise PredictorError(
                "이미지에서 얼굴을 인식하지 못해 분석을 진행할 수 없습니다. "
                "얼굴이 정면을 향하고 이목구비가 뚜렷하게 보이는 이미지가 필요합니다."
            )
        
        bbox = detect_result[:4]; conf = detect_result[4] * 100
        
        x1, y1, x2, y2 = bbox
        face_w = (x2 - x1); face_h = (y2 - y1); face_area = face_w * face_h
        face_ratio = (face_area / total_area) * 100
        
        if face_ratio < 3:
            raise PredictorError(
                    f"얼굴 영역이 분석 기준치(3%)에 미달합니다. (현재: {face_ratio:.1f}%) "
                    "신뢰도 높은 판별을 위해 얼굴이 더 크게 부각된 이미지가 필요합니다."
            )
                
        cropped, directions = self._crop_face(img, bbox)
        if cropped is None:
            dir_msg = ", ".join(directions)
            raise PredictorError(
                f"얼굴이 화면 {dir_msg}에 너무 가까이 붙어 있어 정밀 분석이 불가능합니다."
                "얼굴이 화면 중앙에 위치한 이미지가 필요합니다"
            )
            
        face_gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
        face_brightness = (np.mean(face_gray) / 255) * 100
        
        return cropped, conf, face_ratio, face_brightness
                  
        
    def predict_img(self, img_path: str, tta_hflip: float = 0) -> float:
        
        try:
            img, face_conf, face_ratio, face_brightness = self._preprocess_img(img_path)
            
            transforms = get_test_transforms(img_size=self.img_size, tta_hflip=tta_hflip)
            img = transforms(image=img)['image']
        
            with torch.no_grad():
                img = img.unsqueeze(0).to(self.device)
            
                out = self.model(img)
                pred = torch.sigmoid(out).item()
            
            return {"prob": pred, "face_conf": face_conf, "face_ratio": face_ratio, "face_brightness": face_brightness}
        
        except PredictorError as e:
            raise e
        except Exception as e:
            raise e