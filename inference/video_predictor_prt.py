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
from .utils import ClipPredictorMixin, PredictorError

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
            num_frames = int(frame_cnt / fps)

            frame_indices = [min(int(sec * fps), frame_cnt - 1) for sec in range(num_frames)]
            frame_indices = sorted(set(frame_indices))     
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
            
            return frames, fps, frame_cnt, len(frame_indices), len(frames)
        
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
    
    def _preprocess_frames(self, video_path: str) -> List[np.ndarray]:
        """
        Detects and crops faces from extracted frames, returning them as a list.
        Returns an empty list [] if no faces are detected or an error occurs.
        """
        try:
            frames_dict, fps, total_frames, num_sampled, num_extracted = self._extract_frames(video_path)
            frame_indices = list(frames_dict.keys())
            
            face_results, frames = self._get_face_bboxes(frames_dict)
            
            if not any(face_results):
                raise PredictorError(f"추출한 {len(frames)}개의 Frame에서 모두 얼굴 탐지에 실패하였습니다 ")
            
            cropped_frames, frame_times = [], []

            for i, face_result in enumerate(face_results):
                if len(face_result) == 0: continue

                bbox = face_result[:4]
                cropped_frame = self._crop_face(frames[i], bbox)

                # 랜드마크 추출
                keypoint = self.landmark_detector.detect_batch([cropped_frame])[0]
                if len(keypoint) == 0:
                    continue

                cropped_frames.append(cropped_frame)
                frame_times.append(i)
                
            if not cropped_frames:
                raise PredictorError(f"추출한 {len(frames)}개의 Frame에서 얼굴이 옆모습이거나 가려져 있어 탐지가 어렵습니다")
        
            num_detected = len(cropped_frames)

            return (cropped_frames, frame_times,
                fps, total_frames, num_sampled, num_extracted, num_detected)

        except PredictorError as e:
            raise e
        except Exception as e:
            raise e

    def _frame_aggregate(self, preds: np.ndarray, agg_mode: str) -> float:

        sz = len(preds) # num_frames

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

    def _get_predict(self, video_path: str, tta_hflip: float) -> np.ndarray:
        try:
            (cropped_frames, frame_times,
            fps, total_frames, num_sampled, num_extracted, num_detected) = self._preprocess_frames(video_path)

            transforms = get_test_transforms(img_size=self.img_size, tta_hflip=tta_hflip)
            frames_list = [transforms(image=f)['image'] for f in cropped_frames]
            batch_frames = torch.stack(frames_list).to(self.device) # (num_frames,3,H,W)
                            
            preds = self._forward(batch_frames) # (num_frames,)
            
            return (preds, frame_times,
                fps, total_frames, num_sampled, num_extracted, num_detected)
        
        except PredictorError as e:
            raise e
        except Exception as e:
            raise e 
    
    def _build_result(self, preds: np.ndarray, frame_times: List[int], fps: float, total_frames: int,
                      num_sampled: int, num_extracted: int, num_detected: int, agg_mode: str) -> dict:
        """프레임별 예측값을 집계해 predict_video 결과 dict를 만든다."""
        frame_results = [
            {
                "frame_index": i,
                "frame_time":  frame_times[i],
                "score":       float(preds[i]),
            }
            for i in range(len(preds))
        ]

        return {
            "prob":            float(self._frame_aggregate(preds, agg_mode)),
            "frame_results":   frame_results,
            "fps":             round(fps, 2),
            "total_frames":    total_frames,
            "num_sampled":     num_sampled,
            "num_extracted":   num_extracted,
            "num_detected":    num_detected,
        }

    def predict_video(self, video_path: str, agg_mode: str = 'conf', tta_hflip: float = 0) -> dict:
        return self._build_result(*self._get_predict(video_path, tta_hflip), agg_mode)
        
    def __repr__(self):
        return (f"VideoPredictor(model_name='{self.model_name}', "
                f"margin_ratio='{self.margin_ratio}', "
                f"device='{self.device}', "
                f"img_size={self.img_size})")

class ClipVideoPredictor(ClipPredictorMixin, VideoPredictor):
    def __init__(
        self,
        margin_ratio: float,
        conf_thres: float,
        model_name: str,
    ):
        self._check_model_name(model_name)
        super().__init__(margin_ratio, conf_thres, model_name, dataset=None)

    def predict_video_artifacts(self, video_path: str, tta_hflip: float = 0) -> dict:
        """
        Returns the probabilities of the 20 artifact texts for the whole video (mean over the frames):
            artifact_probs : list of 20 floats in PAIR_ORDER (idx = 5 * region + type)
        """
        cropped_frames = self._preprocess_frames(video_path)[0]

        transforms = get_test_transforms(img_size=self.img_size, tta_hflip=tta_hflip)
        batch_frames = torch.stack([transforms(image=f)['image'] for f in cropped_frames]).to(self.device)
        _, artifact_probs = self._forward_artifacts(batch_frames) # (num_frames,), (num_frames,20)

        return {"artifact_probs": artifact_probs.mean(axis=0).tolist()}

    def __repr__(self):
        return super().__repr__().replace("VideoPredictor", "ClipVideoPredictor", 1)
