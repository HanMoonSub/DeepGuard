import cv2
import timm
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Literal, Callable
from preprocess.face_detector import FaceDetector2
from deepguard.data import get_test_transforms
from explainability.utils.reshape_transforms import reshape_transform_vit, reshape_transform_gcvit

class BaseExplainer:
    def __init__(
        self,
        model_name: Literal["ms_eff_vit_b0", "ms_eff_vit_b5", "ms_eff_gcvit_b0", "ms_eff_gcvit_b5"],
        dataset: Literal["celeb_df_v2", "ff++", "kodf"], 
        margin_ratio: float = 0.2,
        conf_thres: float = 0.5,
        branch_level: Literal["low", "high"] = "high",
        l_stage_idx: int = -1, # gcvit low branch stage index (0~3)
        block_idx: int = -1, # vit, gcvit block index
    ):  
        
        self.device = "cuda:0" if torch.cuda.is_available() else 'cpu'
        self.margin_ratio = margin_ratio
        self.conf_thres = conf_thres
        self.model_name = model_name
        self.dataset = dataset
        self.branch_level = branch_level
        self.l_stage_idx = l_stage_idx
        self.block_idx = block_idx
                
        # Parse Model Metadata
        self.model_variant = model_name.split("_")[-1] # b0, b5
        self.transformer_type = model_name.split("_")[-2] # vit, gcvit 
                
        # Model & Tools setup
        self.model = timm.create_model(model_name, pretrained=True, dataset=dataset)
        self.face_detector = FaceDetector2(conf_thres)
        self.img_size = [224,224] if self.model_variant == "b0" else [384,384]
        self.transforms = get_test_transforms(img_size=self.img_size)

        self.model.to(self.device).eval()
        
        # Explainability setup
        self.reshape_fn = self._get_reshape_fn()
        self.target_layers = self._resolve_target_layers()

    def _get_reshape_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """모델 종류에 맞는 Feature map reshape 함수를 반환한다.

        Returns:
            Callable: 3D/4D 텐서를 GradCAM 연산에 적합한 형태로 변환하는 함수.
        """
        return reshape_transform_vit if self.transformer_type == "vit" else reshape_transform_gcvit    
    
    def _resolve_target_layers(self) -> list[torch.nn.Module]:
        """시각화(XAI)의 대상이 될 Target Layer들을 선택한다.

        Returns:
            list[torch.nn.Module]: 분석 대상 정규화(Normalization) 레이어 리스트.

        Raises:
            ValueError: l_stage_idx나 block_idx가 모델 범위를 벗어날 경우 발생.
        """
        layers = []

        if self.transformer_type == "gcvit":
            n_stages  = len(self.model.l_gcvit.stages)
            stage_idx = self.l_stage_idx if self.l_stage_idx >= 0 else n_stages + self.l_stage_idx

            if not (0 <= stage_idx < n_stages):
                raise ValueError(f"l_stage_idx={self.l_stage_idx} 범위 초과 (stage 수: {n_stages})")

            if self.branch_level in ("low", "both"):
                n_blocks = len(self.model.l_gcvit.stages[stage_idx].blocks)
                blk      = self._resolve_block_idx(self.block_idx, n_blocks, f"l_gcvit stage[{stage_idx}]")
                layers.append(self.model.l_gcvit.stages[stage_idx].blocks[blk].norm2)

            if self.branch_level in ("high", "both"):
                n_blocks = len(self.model.h_gcvit.stages[0].blocks)
                blk      = self._resolve_block_idx(self.block_idx, n_blocks, "h_gcvit stage[0]")
                layers.append(self.model.h_gcvit.stages[0].blocks[blk].norm2)

        else:  # vit
            if self.branch_level in ("low", "both"):
                n_blocks = len(self.model.l_vit.blocks)
                blk      = self._resolve_block_idx(self.block_idx, n_blocks, "l_vit")
                layers.append(self.model.l_vit.blocks[blk].norm1)

            if self.branch_level in ("high", "both"):
                n_blocks = len(self.model.h_vit.blocks)
                blk      = self._resolve_block_idx(self.block_idx, n_blocks, "h_vit")
                layers.append(self.model.h_vit.blocks[blk].norm1)

        return layers

    @staticmethod
    def _resolve_block_idx(block_idx: int, n_blocks: int, name: str) -> int:
        """음수 인덱싱을 지원하며, 해당 블록의 유효한 인덱스를 계산 및 검증한다.

        Args:
            block_idx (int): 접근하려는 블록 인덱스 (음수 가능).
            n_blocks (int): 해당 스테이지/모델의 총 블록 개수.
            name (str): 에러 메시지 출력용 모듈 이름.

        Returns:
            int: 정제된 양수 형태의 블록 인덱스.

        Raises:
            ValueError: 인덱스가 유효 범위를 벗어난 경우 발생.
        """
        idx = block_idx if block_idx >= 0 else n_blocks + block_idx
        if not (0 <= idx < n_blocks):
            raise ValueError(f"block_idx={block_idx} 범위 초과 ({name} block 수: {n_blocks})")
        return idx
    
    def _get_face_bbox(self, img: np.ndarray) -> List[float]:
        """얼굴 탐지기(YOLO)를 통해 이미지에서 얼굴 바운딩 박스를 좌표를 추출한다.

        Args:
            img (np.ndarray): HWC 형태의 RGB 이미지 데이터.

        Returns:
            list[float]: [xmin, ymin, xmax, ymax, confidence] 형태의 좌표 리스트.
        """
        result = self.face_detector.detect_batch([img], [1.0])
        return result[0]
    
    def _crop_face(self, img: np.ndarray, bbox: List[float]) -> np.ndarray:
        """마진 비율을 고려하여 바운딩 박스 기준으로 얼굴 영역을 크롭한다.

        Args:
            img (np.ndarray): 전체 원본 이미지.
            bbox (list[float]): 탐지된 [xmin, ymin, xmax, ymax] 얼굴 좌표.

        Returns:
            np.ndarray: 크롭된 얼굴 이미지 영역.
        """
        xmin, ymin, xmax, ymax = bbox
        h, w = img.shape[:2]
        pw = int((xmax - xmin) * self.margin_ratio)
        ph = int((ymax - ymin) * self.margin_ratio)
        return img[
            max(int(ymin - ph), 0) : min(int(ymax + ph), h),
            max(int(xmin - pw), 0) : min(int(xmax + pw), w),
        ]
        
    def _preprocess_img(self, img_path: str) -> np.ndarray:
        """이미지 경로를 읽어 얼굴을 탐지하고 크롭하는 전과정을 수행한다.

        Args:
            img_path (str): 입력 이미지 파일 경로.

        Returns:
            np.ndarray: 전처리가 완료된 크롭된 얼굴 이미지 (RGB).
        """
        img = cv2.imread(img_path)        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detect_result = self._get_face_bbox(img)                
        bbox = detect_result[:4]
        cropped = self._crop_face(img, bbox)
        return cropped
                  
    def _get_transform(self, img_path: str) -> tuple[np.ndarray, torch.Tensor]:
        """이미지 경로를 받아 크롭 이미지와 모델 입력용 텐서를 동시에 생성한다.

        Args:
            img_path (str): 입력 이미지 파일 경로.

        Returns:
            tuple[np.ndarray, torch.Tensor]: (크롭된 얼굴 이미지, (1, 3, H, W) 형태의 배치 텐서).
        """
        face = self._preprocess_img(img_path)
        tensor = self.transforms(image=face)['image'].unsqueeze(0) # (1,3,H,W)
        return face, tensor
    
    def _get_transform_from_array(self, face: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
        """이미 크롭된 얼굴 배열을 받아 모델 입력용 텐서로 변환한다.

        Args:
            face (np.ndarray): 이미 잘려진 얼굴 이미지 배열 (RGB).

        Returns:
            tuple[np.ndarray, torch.Tensor]: (원본 얼굴 이미지, (1, 3, H, W) 형태의 배치 텐서).
        """
        tensor = self.transforms(image=face)['image'].unsqueeze(0) # (1,3,H,W)
        return face, tensor
            
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"model={self.model_name}, dataset={self.dataset}, "
                f"margin_ratio={self.margin_ratio}, conf_thres={self.conf_thres}, "
                f"branch_level={self.branch_level}, l_stage_idx={self.l_stage_idx}, block_idx={self.block_idx},"
                f"category={self.category},device={self.device})")