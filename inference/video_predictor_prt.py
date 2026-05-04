import os
import cv2
import numpy as np
import timm
import torch
from typing import Dict, List
from preprocess.frame_extractor import FrameExtractor
from preprocess.face_detector import FaceDetector2
from preprocess.landmark_detector import LandmarkDetector
from deepguard.data import get_test_transforms
from .utils import PredictorError

class VideoPredictor:
    def __init__(self,
                 margin_ratio: float,
                 conf_thres: float,
                 model_name: str,
                 dataset: str,
                 ):
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"        
        self.margin_ratio = margin_ratio
        self.frame_extractor = FrameExtractor(jitter=0)
        self.face_detector = FaceDetector2(conf_thres)
        self.landmark_detector = LandmarkDetector()
        self.model = timm.create_model(model_name, pretrained=True, dataset=dataset)
        self.img_size = [224,224] if model_name.split("_")[-1] == "b0" else [384,384]
        
        # Model Inference Mode
        self.model.to(self.device)
        self.model.eval()
        
    def _get_frame_indices(self, frame_cnt: int, num_frames: int):
        
        frame_indices = np.linspace(0, frame_cnt - 1, num=num_frames,
                    endpoint=True, dtype=np.int32)
        return sorted(list(set(frame_indices)))
        
    def _extract_frames(self, video_path: str) -> Dict[int, np.ndarray]:
        """
        Extracts frames from the video and returns a dictionary in the format {frame_index: frame_data}.
        Returns an empty dictionary {} if the extraction fails.
        """
        cap = None
        try:
            if not os.path.exists(video_path):
                raise PredictorError("해당 비디오 파일 경로가 존재하지 않습니다")
            
            cap = cv2.VideoCapture(filename=video_path)
            if not cap.isOpened():
                raise PredictorError("비디오를 불러올 수 없습니다. 파일이 손상되었거나 지원하지 않는 형식인지 확인해 주세요.")
            
            frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            duration_sec = int(frame_cnt / fps)
            num_frames = max(10, min(duration_sec, 60))
            
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
                raise PredictorError(
                    f"해당 비디오 내 {num_frames}개의 Frame에서 모두 추출에 실패하였습니다"
                )
            
            return frames, fps
        
        except PredictorError as e:
            raise e
        except Exception as e:
            raise e
        finally:
            if cap is not None and cap.isOpened():
                cap.release()
                
    def _get_face_bboxes(self, frames_dict: Dict):
        frames = list(frames_dict.values())
        face_results = self.face_detector.detect_batch(frames, [1.0] * len(frames))
        return face_results, frames
    
    def _crop_face(self, frame: np.ndarray, bbox: List[float]):
        
        img_h, img_w = frame.shape[:2]
        xmin, ymin, xmax, ymax = bbox
        
        w = xmax - xmin
        h = ymax - ymin
        
        # Calculate Face Ratio in Frame
        face_area = w * h; total_area = img_w * img_h
        face_ratio = (face_area / total_area) * 100
        
        pad_w = int(w * self.margin_ratio)
        pad_h = int(h * self.margin_ratio)
        
        y1 = max(int(ymin - pad_h), 0)
        y2 = min(int(ymax + pad_h), frame.shape[0]) 
        x1 = max(int(xmin - pad_w), 0)
        x2 = min(int(xmax + pad_w), frame.shape[1])
        
        cropped_frame = frame[y1:y2,x1:x2]
        
        return cropped_frame, face_ratio
    
    def _preprocess_frames(self, video_path: str) -> List[np.ndarray]:
        """
        Detects and crops faces from extracted frames, returning them as a list.
        Returns an empty list [] if no faces are detected or an error occurs.
        """
        try:
            frames_dict, fps = self._extract_frames(video_path)
            frame_indices = list(frames_dict.keys())
            
            face_results, frames = self._get_face_bboxes(frames_dict)
            
            if not any(face_results):
                raise PredictorError(f"추출한 {len(frames)}개의 Frame에서 모두 얼굴 탐지에 실패하였습니다 ")
            
            cropped_frames, face_confs, face_ratios, face_brightnesses, frame_times = [], [], [], [], []
            
            for i, face_result in enumerate(face_results):
                if len(face_result) == 0: continue
                
                bbox = face_result[:4]
                conf = face_result[4] * 100
                cropped_frame, face_ratio = self._crop_face(frames[i], bbox)
                
                # 랜드마크 추출
                keypoint = self.landmark_detector.detect_batch([cropped_frame])[0]
                if len(keypoint) == 0: 
                    continue
                
                face_gray = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2GRAY)
                
                cropped_frames.append(cropped_frame)
                face_confs.append(conf)
                face_ratios.append(face_ratio)
                face_brightnesses.append((np.mean(face_gray) / 255) * 100)
                frame_times.append(round(frame_indices[i] / fps, 2))
                
            if not cropped_frames:
                raise PredictorError(f"추출한 {len(frames)}개의 Frame에서 얼굴이 옆모습이거나 가려져 있어 탐지가 어렵습니다")
        
            return cropped_frames, face_confs, face_ratios, face_brightnesses, frame_times
        
        except PredictorError as e:
            raise e
        except Exception as e:
            raise e
        
    def _frame_aggregate(self, preds: np.ndarray, face_confs: List[float], face_ratios: List[float],
                        face_brightnesses: List[float], agg_mode: str) -> np.ndarray:
        
        sz = len(preds) # num_frames
        
        if agg_mode == "mean":
            return np.mean(preds), np.mean(face_confs), np.mean(face_ratios), np.mean(face_brightnesses)
        
        elif agg_mode == "vote":
            return np.sum(preds > 0.5) / sz, np.mean(face_confs), np.mean(face_ratios), np.mean(face_brightnesses)
        
        elif agg_mode == 'conf':
            t_fake = 0.8
            t_real = 0.2
            
            fakes = preds[preds > t_fake]
            num_fakes = len(fakes)
            
            if num_fakes > (sz // 2.5):
                return np.mean(fakes), np.mean(face_confs), np.mean(face_ratios), np.mean(face_brightnesses)
            elif np.count_nonzero(preds < t_real) > 0.9 * sz:
                return np.mean(preds[preds < t_real]), np.mean(face_confs), np.mean(face_ratios), np.mean(face_brightnesses)
            else:
                return np.mean(preds), np.mean(face_confs), np.mean(face_ratios), np.mean(face_brightnesses)
        
    def _get_predict(self, video_path: str, tta_hflip: float) -> np.ndarray:
        try:
            cropped_frames, face_confs, face_ratios, face_brightnesses, frame_times = self._preprocess_frames(video_path)
    
            transforms = get_test_transforms(img_size=self.img_size, tta_hflip=tta_hflip)
            frames_list = [transforms(image=f)['image'] for f in cropped_frames]
            batch_frames = torch.stack(frames_list).to(self.device) # (num_frames,3,H,W)
                            
            with torch.no_grad():
                outs = self.model(batch_frames) # (num_frames,1)
                preds = torch.sigmoid(outs.view(-1)).cpu().numpy() # (num_frames,)
            
            return preds, face_confs, face_ratios, face_brightnesses, frame_times
        
        except PredictorError as e:
            raise e
        except Exception as e:
            raise e 
    
    def predict_video(self, video_path: str, agg_mode: str = 'conf', tta_hflip: float = 0) -> np.ndarray:
        try:
            preds, face_confs, face_ratios, face_brightnesses, frame_times = self._get_predict(video_path, tta_hflip)
            
            agg_prob, agg_conf, agg_ratio, agg_brightness = self._frame_aggregate(
                preds, face_confs, face_ratios, face_brightnesses, agg_mode
            )
            
            frame_results = [
                {
                    "frame_index":     i,
                    "frame_time":      frame_times[i],
                    "score":           float(preds[i]),
                    "face_conf":       float(face_confs[i]),
                    "face_ratio":      float(face_ratios[i]),
                    "face_brightness": float(face_brightnesses[i]),
                }
                for i in range(len(preds))
            ]
            
            return {
                "prob":            float(agg_prob),
                "face_conf":       float(agg_conf),
                "face_ratio":      float(agg_ratio),
                "face_brightness": float(agg_brightness),
                "frame_results":   frame_results, 
            }
            
        except PredictorError as e:
            raise e
        except Exception as e :
            raise e