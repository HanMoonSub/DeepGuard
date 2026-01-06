import numpy as np
import cv2
from typing import List, Optional, Dict

class FrameExtractor:
    def __init__(self,
                 jitter: int = 0,
                 verbose: bool = False):
        """
        A class to efficiently extract specific frames from a video
            
        Args:
            jitter: The range of random offset applied to each frame indx.
            verbose: If True, prints error messages during the extraction
        
        return:
            Dict[int, np.ndarray]:
                A Dicitonary mapping { frame_index: RGB_image_array }.
                Example: { 0: array([...]), 15: array([...])}
        """
        
        self.jitter = jitter
        self.verbose = verbose
        
    def _get_frame_indices(self, 
                           frame_count: int, 
                           num_frames: int) -> List[int]:
        frame_idxs = np.linspace(0, frame_count-1, num=num_frames,
                                 endpoint=True, dtype=np.int32)
        
        if self.jitter > 0:
            jitter_offset = np.random.randint(-self.jitter, self.jitter + 1, len(frame_idxs))
            frame_idxs = np.clip(frame_idxs + jitter_offset, 0, frame_count-1)
            
        return sorted(set(frame_idxs))
        
    def extract_frames(self, 
                       video_path: str,
                       num_frames: int) -> Optional[Dict[int, np.ndarray]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            if self.verbose:
                print(f"[Error] Cannot open video: {video_path}")
            return None
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            if self.verbose:
                print(f"[Error] Invalid frame count for: {video_path}")
            cap.release()
            return None
        
        frame_idxs = self._get_frame_indices(frame_count, num_frames)
        frames = {}
        current = 0
        
        for frame_idx in range(frame_idxs[-1] + 1):
            
            ret = cap.grab()
            if not ret:
                if self.verbose:
                    print(f"[Error] Failed to grab frame {frame_idx}")
                break
            
            if frame_idx == frame_idxs[current]: 
                ret, frame = cap.retrieve()
                if not ret or frame is None:
                    if self.verbose:
                        print(f"[Error] Failed to retrieve frame {frame_idx}")
                    break
                
                frames[frame_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                current += 1
                if current >= len(frame_idxs):
                    break

        cap.release()
        return frames                