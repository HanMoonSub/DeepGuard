import os
import cv2
import numpy as np
import timm
import torch
from typing import Dict, List
from preprocess.frame_extractor import FrameExtractor
from preprocess.face_detector import FaceDetector
from deepguard.data import get_test_transforms

class VideoPredictor:
    def __init__(self,
                 margin_ratio: float,
                 conf_thres: float,
                 min_face_ratio: float,
                 model_name: str,
                 dataset: str,
                 ):
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"        
        self.margin_ratio = margin_ratio
        self.frame_extractor = FrameExtractor(jitter=0)
        self.face_detector = FaceDetector(conf_thres, min_face_ratio)
        self.model = timm.create_model(model_name, pretrained=True, dataset=dataset)
        self.img_size = [224,224] if model_name.split("_")[-1] == "b0" else [384,384]
        
        # Model Inference Mode
        self.model.to(self.device)
        self.model.eval()
        
    def _get_frame_indices(self, frame_cnt: int, num_frames: int):
        
        frame_indices = np.linspace(0, frame_cnt - 1, num=num_frames,
                    endpoint=True, dtype=np.int32)
        return sorted(list(set(frame_indices)))
        
    def _extract_frames(self, video_path: str, num_frames: int):
        cap = None
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"The video file does not exist at path: {video_path}")
            
            cap = cv2.VideoCapture(filename=video_path)
    
            if not cap.isOpened():
               raise ConnectionError(f"Failed to open video stream for: {video_path}")
            
            frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if frame_cnt <= 0:
                raise ValueError(f"Invalid frame count ({frame_cnt}). The video file might be corrupted: {video_path}")
            
            frame_indices = self._get_frame_indices(frame_cnt, num_frames)
            
            frames = {}
            target_idx = 0
            max_target = frame_indices[-1]
            
            for frame_idx in range(max_target + 1):
                ret = cap.grab()
                if not ret: 
                    break
                
                if frame_idx == frame_indices[target_idx]:
                    ret, frame = cap.retrieve()
                    
                    if ret and frame is not None:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                        
                        frames[frame_idx] = frame_rgb
                    
                    target_idx += 1
                
                    if target_idx >= len(frame_indices):
                        break
            
            if not frames:
                raise RuntimeError(f"No frames were successfully extracted from the video: {video_path}")
            
            return frames
            
        except (FileNotFoundError, ConnectionError, ValueError, RuntimeError) as e:
            print(f"[VideoError] {str(e)}")
            raise e
        except Exception as e:
            print(f"[UnexpectedError] An unidentified error occurred: {str(e)}")
            raise e
        finally:
            if cap is not None and cap.isOpened():
                cap.release()
                
    def _get_face_bboxes(self, frames_dict: Dict):
        frames = list(frames_dict.values())
        face_bboxes = self.face_detector.detect_batch(frames, [1.0] * len(frames))
        return face_bboxes, frames
    
    def _crop_face(self, frame: np.ndarray, bbox: List[float]):
        xmin, ymin, xmax, ymax = bbox
        
        w = xmax - xmin
        h = ymax - ymin
        pad_w = int(w * self.margin_ratio)
        pad_h = int(h * self.margin_ratio)
        
        y1 = max(int(ymin - pad_h), 0)
        y2 = min(int(ymax + pad_h), frame.shape[0]) 
        x1 = max(int(xmin - pad_w), 0)
        x2 = min(int(xmax + pad_w), frame.shape[1])
        
        cropped_frame = frame[y1:y2,x1:x2]
        
        return cropped_frame
    
    def _preprocess_frames(self, video_path: str, num_frames: int):
        try:
            frames_dict = self._extract_frames(video_path, num_frames)
            bboxes, frames = self._get_face_bboxes(frames_dict)
            
            cropped_frames = []
            
            for i, bbox in enumerate(bboxes):
                if len(bbox) == 0:
                    continue
                cropped_frames.append(self._crop_face(frames[i], bbox))
                
            if len(cropped_frames) == 0:
                raise RuntimeError(f"No faces were detected: {video_path}")
            return cropped_frames
            
        except Exception as e:
            print(f"[Error] Preprocess Failed for {video_path}: {e}")
            raise e
        
    def _frame_aggregate(self, preds: np.ndarray, agg_mode: str) -> np.ndarray:
        
        sz = len(preds)
        
        if agg_mode == "mean":
            return np.mean(preds)
        
        elif agg_mode == "vote":
            return np.sum(preds > 0.5) / sz
        
        elif agg_mode == 'conf':
            t_fake = 0.8
            t_real = 0.2
            
            fakes = preds[preds > t_fake]
            num_fakes = len(fakes)
            
            if num_fakes > (sz // 2.5):
                return np.mean(fakes)
            elif np.count_nonzero(preds < t_real) > 0.9 * sz:
                return np.mean(preds[preds < t_real])
            else:
                return np.mean(preds)
        
    def _get_predict(self, video_path: str, num_frames: int, tta_hflip: float) -> np.ndarray:
        frames = self._preprocess_frames(video_path, num_frames)
        transforms = get_test_transforms(img_size=self.img_size, tta_hflip=tta_hflip)
            
        frames_list = [transforms(image=f)['image'] for f in frames]
        batch_frames = torch.stack(frames_list).to(self.device) # (B,3,H,W)
                            
        with torch.no_grad():
            outs = self.model(batch_frames) # (B,1)
            preds = torch.sigmoid(outs.view(-1)).cpu().numpy() # (B,)
            
        return preds
    
    def predict_video(self, video_path: str, num_frames: int = 20, agg_mode: str = 'conf', tta_hflip: float = 0) -> np.ndarray:
        try:
            preds = self._get_predict(video_path, num_frames, tta_hflip)
            
            return self._frame_aggregate(preds, agg_mode)
        
        except Exception as e:
            print(f"Error in predict_video: {e}")
            raise e
        
    def predict_detail(self, video_path: str, num_frames_per_sec: int = 3, agg_mode: str = 'conf', tta_hflip: float = 0) -> dict[str, float]:
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            cap.release()
            
            total_sample_frames = int(duration * num_frames_per_sec)
            
            preds = self._get_predict(video_path, total_sample_frames, tta_hflip)

            timestamps = np.linspace(0, duration, len(preds))
            secondly_data = {}
            
            for i, t in enumerate(timestamps):
                second_key = f"{int(t)}s"
                if second_key not in secondly_data:
                    secondly_data[second_key] = []
                secondly_data[second_key].append(preds[i])
                
            report = {sec: float(self._frame_aggregate(np.array(scores), agg_mode))
                      for sec, scores in secondly_data.items()}
            
            return report 
        except Exception as e:
            print(f"Error in predict_second: {e}")
            raise e