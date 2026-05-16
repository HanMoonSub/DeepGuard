import cv2
import timm
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from preprocess.face_detector import FaceDetector2
from deepguard.data import get_test_transforms
from explainability.utils.reshape_transforms import reshape_transform_vit, reshape_transform_gcvit

DATASETS = {"celeb_df_v2", "ff++", "kodf"}
MODELS   = {"ms_eff_vit_b0", "ms_eff_vit_b5", "ms_eff_gcvit_b0", "ms_eff_gcvit_b5"}

class BaseExplainer:
    def __init__(
        self,
        model_name: str, # ms_eff_vit_b0 | ms_eff_vit_b5 | ms_eff_gcvit_b0 | ms_eff_gcvit_b5 
        dataset: str,  # celeb_df_v2 | ff++ | both
        margin_ratio: float = 0.2,
        conf_thres: float = 0.5,
        branch_level: str = "both", # low | high | both
        l_stage_idx: int = -1, # gcvit low branch stage index (0~3)
        block_idx: int = -1 # vit, gcvit block index 
    ):  
        self._validate_inputs(model_name, dataset, branch_level)
        
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
    
    def _validate_inputs(self, model_name: str, dataset: str, branch_level: str):
        if model_name not in MODELS:
            raise ValueError(f"Unsupported model: '{model_name}'. Must be one of {MODELS}")
        if dataset not in DATASETS:
            raise ValueError(f"Invalid dataset: '{dataset}'. Choose from {DATASETS}")
        if branch_level not in ("low", "high", "both"):
            raise ValueError(f"Invalid branch_level: '{branch_level}'. Expected 'low', 'high', or 'both'.")
        
    def _get_reshape_fn(self):
        return reshape_transform_vit if self.transformer_type == "vit" else reshape_transform_gcvit    
    
    def _resolve_target_layers(self) -> list:
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
        idx = block_idx if block_idx >= 0 else n_blocks + block_idx
        if not (0 <= idx < n_blocks):
            raise ValueError(f"block_idx={block_idx} 범위 초과 ({name} block 수: {n_blocks})")
        return idx
    
    def _get_face_bbox(self, img: np.ndarray) -> List[float]:
        result = self.face_detector.detect_batch([img], [1.0])
        return result[0]
    
    def _crop_face(self, img: np.ndarray, bbox: List[float]) -> np.ndarray:
        xmin, ymin, xmax, ymax = bbox
        h, w = img.shape[:2]
        pw = int((xmax - xmin) * self.margin_ratio)
        ph = int((ymax - ymin) * self.margin_ratio)
        return img[
            max(int(ymin - ph), 0) : min(int(ymax + ph), h),
            max(int(xmin - pw), 0) : min(int(xmax + pw), w),
        ]
        
    def _preprocess_img(self, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path)        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detect_result = self._get_face_bbox(img)                
        bbox = detect_result[:4]
        cropped = self._crop_face(img, bbox)
        return cropped
                  
    def _get_transform(self, img_path: str):
        face = self._preprocess_img(img_path)
        tensor = self.transforms(image=face)['image'].unsqueeze(0) # (1,3,H,W)
        return face, tensor
            
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"model={self.model_name}, dataset={self.dataset}, "
                f"margin_ratio={self.margin_ratio}, conf_thres={self.conf_thres}, "
                f"branch_level={self.branch_level}, l_stage_idx={self.l_stage_idx}, block_idx={self.block_idx},"
                f"category={self.category},device={self.device})")