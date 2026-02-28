import os
from pathlib import Path
from ultralytics import YOLO
from typing import List
import numpy as np

class LandmarkDetector:
    """
    YOLO-based Face Landmark Detector.
    
    This class performs batch inference to detect 5 facial landmarks.
    """
    
    def __init__(self):
        
        weights_path = Path(__file__).resolve().parent / "yolov8n-face.pt"
        
        self.model = YOLO(str(weights_path), task='pose')
        
    def detect_batch(
                    self, 
                    frames: List[np.ndarray],
                    ):
        """
        Detects 5 facial landmarks for a batch of face images.

        Args:
            frames: List of Cropped Face Images (Numpy arrays, RGB format)
                
        Returns:
            List of landmarks.
            - Shape: (B, 5, 2) if detected.
            - Each landmark: [x, y] coordinates.
            - If no face detected, returns an empty list []
            
            Keypoint Order: [Left Eye, Right Eye, Nose, Left Mouth, Right Mouth]
        """
        
        results = self.model(frames, verbose=False)
   
        batch_landmarks = []
    
        for i, result in enumerate(results):
            
            if result.keypoints is None or len(result.keypoints) == 0:
                batch_landmarks.append([])
                continue
            
            keypoints = result.keypoints.xy[0].cpu().numpy()
            
            batch_landmarks.append(keypoints)
                
        return batch_landmarks
    
class LandmarkDetector_KODF:
    """
    YOLO-based Face Landmark Detector.
    
    This class performs inference to detect 5 facial landmarks.
    """
    
    def __init__(self):
        
        weights_path = Path(__file__).resolve().parent / "yolov8n-face.pt"
        
        self.model = YOLO(str(weights_path), task='pose')
        
    def detect_single_landmark(
                    self, 
                    frame: np.ndarray,
                    ):
        """
        Detects 5 facial landmarks for a single croppe face image

        Args:
            frame: Cropped Face Images (Numpy arrays, RGB format)
                
        Returns:
            - Shape: (5, 2) if detected.
            - Each landmark: [x, y] coordinates.
            Keypoint Order: [Left Eye, Right Eye, Nose, Left Mouth, Right Mouth]
        """
        
        results = self.model(frame, verbose=False)
        
        if not results or len(results) == 0:
            return []
        
        result = results[0]
        
        if result.keypoints is None or len(result.keypoints) == 0:
            return []
            
        keypoints = result.keypoints.xy[0].cpu().numpy()
            
        return keypoints