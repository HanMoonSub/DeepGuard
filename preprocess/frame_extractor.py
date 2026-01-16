import numpy as np
from typing import List
import cv2

class FrameExtractor:
    def __init__(self, 
                 jitter: int = 0, 
                 ):
        """
            Args:
                jitter: Random offest to apply when selecting frames
                verbose: print detailed logs and error messages 
        """

        self.jitter = jitter
        
    def _get_frame_indices(self, 
                           frame_cnt: int,
                           num_frames: int,
                           frame_endpoint: int
                           ):
        if frame_endpoint == 0:
            frame_indices = np.linspace(0, frame_cnt - 1, num=num_frames,
                    endpoint=True, dtype=np.int32)
        else:
            frame_indices = np.linspace(0, frame_endpoint - 1, num=num_frames,
                    endpoint=True, dtype=np.int32)
        
        if self.jitter > 0:
            jitter_offset = np.random.randint(-self.jitter, self.jitter+1, num_frames)
            frame_indices = np.clip(frame_indices + jitter_offset, 0, frame_cnt - 1)
        
        frame_indices = sorted(list(set(frame_indices)))
            
        return frame_indices
    
    def extract_frames_for_detect(self, 
                                  video_path: str,
                                  num_frames: int,
                                  frame_endpoint: int = 0,
                                  ):
        """
            For Faster Detecting Faces, Apply Rescaling
        """
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {}
        
        frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_cnt <= 0:
            cap.release()
            return {}

        frame_indices = self._get_frame_indices(frame_cnt, num_frames, frame_endpoint)
        
        return self._extract_frames(cap, frame_indices, rescale=True)
            
    def extract_frames_for_crop(self,
                                video_path: str,
                                frame_indices: List[int]):
        """
            No apply rescaling, for Cropping Frames
        """
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {}
        
        frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_cnt <= 0:
            cap.release()
            return {}

        valid_indices = [idx for idx in frame_indices if idx < frame_cnt]
        valid_indices = sorted(set(valid_indices))
        
        if not valid_indices:
            cap.release()
            return {}
            
        return self._extract_frames(cap, valid_indices, rescale=False)
        
    def _extract_frames(self,
                        cap: cv2.VideoCapture,
                        frame_indices: List[int],
                        rescale: bool):
        """
            For Speed up in extractinf fraemes
            We selected not cap.read() but cap.grab() & cap.retrieve()
        """
        
        frames = {}
        target_idx = 0
        max_target = frame_indices[-1]
        
        for frame_idx in range(max_target + 1):
            
            ret = cap.grab()
            if not ret: break 
            
            if frame_idx == frame_indices[target_idx]:
                ret, frame = cap.retrieve()   
                
                if ret and frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    scale_factor = 1.0
                    
                    if not rescale:
                        frames[frame_idx] = frame_rgb
                    else:
                        h, w = frame_rgb.shape[:2]
                        size = max(h, w)
                        
                        if size < 300:
                            scale_factor = 2.0
                        elif 300 <= size < 700:
                            scale_factor = 1.0
                        elif 700 <= size < 1500:
                            scale_factor = 0.5
                        else:
                            scale_factor = 0.33 
                        
                        if scale_factor != 1.0:
                            resized_frame = cv2.resize(frame_rgb, (0,0), fx=scale_factor, fy=scale_factor)
                            frames[frame_idx] = (resized_frame, scale_factor)
                        else:
                            frames[frame_idx] = (frame_rgb, scale_factor)

                target_idx += 1
                
                if target_idx >= len(frame_indices):
                    break
        
        cap.release()
        return frames
        