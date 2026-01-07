import os
from ultralytics import YOLO
from typing import List
import numpy as np
from colorama import Fore, Style

c_ = Fore.BLUE
s_ = Style.BRIGHT
r_ = Style.RESET_ALL

class FaceDetector:
    """
    YOLO-based Face Detector.

    This class performs batch inference to detect FACE bounding boxes.
    It uses a custom YOLO model trained on face datasets (e.g., yolov8n-face.pt).
    """
    
    def __init__(
                self,
                root_dir: str,
                conf_thres: float,
                min_face_ratio: float
                ):
        """
            Args:
                root_dir: Root directory containing video files.
                verbose: print detailed logs and error messages
                conf_thres: Confidence threshold for YOLO face detection
                min_face_ratio: Minimum face bounding box area ratio relative to the full frame area.
        """
        
        
        model_path = os.path.join(root_dir, "yolov8n-face.pt")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        self.model = YOLO(model_path, task='detect')
        self.conf_thres = conf_thres
        self.min_face_ratio = min_face_ratio
        
    def detect_batch(
                    self, 
                    frames: List[np.ndarray],
                    scales: List[float]
                    ):
        
        results = self.model(frames, conf=self.conf_thres, verbose=False)
   
        batch_results = []
    
        for i, result in enumerate(results):
            scale_factor = scales[i]
            
            if len(result.boxes) == 0:
                batch_results.append([])
                continue
            
            # (N, 6) -> [xmin, ymin, xmax, ymax, conf, cls]
            data = result.boxes.data.cpu().numpy()
            
            face_boxes = data[data[:, 5] == 0] # Category: Face
                
            if len(face_boxes) == 0:
                batch_results.append([])
                continue
            
            ws = face_boxes[:, 2] - face_boxes[:, 0]
            hs = face_boxes[:, 3] - face_boxes[:, 1]
            areas = ws * hs
            
            max_area_idx = np.argmax(areas)
            max_area = areas[max_area_idx]
        
            frame_h, frame_w = frames[i].shape[:2]
            frame_area = frame_h * frame_w
    
            if (max_area / frame_area) < self.min_face_ratio:
                batch_results.append([])
            else:
                best_face_box = face_boxes[max_area_idx, :4]
                
                final_box = best_face_box / scale_factor
                
                batch_results.append(final_box.tolist())
                
        return batch_results