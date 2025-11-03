import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List

import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

from PIL import Image
from facenet_pytorch import MTCNN
from torch.utils.data import Dataset

class VideoFaceDetector(ABC):
    
    def __init__(self, **kwargs) -> None:
        super().__init__()
    
    @property
    @abstractmethod
    def _batch_size(self) -> int:
        pass     
    
    @abstractmethod
    def _detect_faces(self, frames) -> List:
        pass
    
class FacenetDetector(VideoFaceDetector):
    
    def __init__(self, device="cuda:0", batch_size=32) -> None:
        super().__init__()
        self.detector = MTCNN(margin=0, thresholds=[0.85, 0.95, 0.95], device=device)
        self.batch_size = batch_size
        
    def _detect_faces(self, frames) -> List:
        """
        Arguments:
            frames {list of PIL.Image} -- List of input video frames.
        
        Returns:
            list -- A list of face bounding box coordinates for each frame.
                    Each element is a list of boxes in the format [[xmin, ymin, xmax, ymax], ...].
                    Returns None for frames where no face is detected.
        """
        
        batch_boxes, probs = self.detector.detect(frames, landmarks=False)
        return [b.tolist() if b is not None else None for b in batch_boxes]
    
    @property  
    def _batch_size(self):
        return self.batch_size
    
class VideoDataset(Dataset):
    
    def __init__(self, videos) -> None:
        super().__init__()
        self.videos = videos
    
    def __getitem__(self, index):
        video = self.videos[index]
        cap = cv2.VideoCapture(video)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = OrderedDict()
        for i in range(frame_count):
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            frame = Image.fromarray(frame)
            new_size = tuple(int(s * 0.5) for s in frame.size)  # (width, height)
            frame = frame.resize(new_size)
            frames[i] = frame
        
        cap.release()
            
        return video, list(frames.keys()), list(frames.values())
    
    def __len__(self) -> int:
        return len(self.videos) 