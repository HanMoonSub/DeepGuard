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
        model_name: Literal[
            "ms_eff_vit_b0", "ms_eff_vit_b5", "ms_eff_gcvit_b0", "ms_eff_gcvit_b5",
            # ablation models
            "eff_b0", "eff_b5",
            "low_eff_gcvit_b0", "low_eff_gcvit_b5",
            "high_eff_gcvit_b0", "high_eff_gcvit_b5",
            # DeepClip models (ClipExplainerMixin)
            "deepclip_b0", "deepclip_b5",
        ],
        dataset: Optional[Literal["celeb_df_v2", "ff++", "kodf"]] = None, # DeepClip 모델은 dataset 불필요
        margin_ratio: float = 0.2,
        conf_thres: float = 0.5,
        branch_level: Literal["low", "high"] = "high", # 두 branch 결합은 BranchFusionExplainer 사용
        model: Optional[torch.nn.Module] = None,
    ):

        self.device = "cuda:0" if torch.cuda.is_available() else 'cpu'
        self.margin_ratio = margin_ratio
        self.conf_thres = conf_thres
        self.model_name = model_name
        self.dataset = dataset
        self.branch_level = branch_level

        # Parse Model Metadata
        self.model_variant = model_name.split("_")[-1] # b0, b5
        self.transformer_type = self._get_transformer_type()
        self._validate_branch_level()

        # Model & Tools setup
        self.model = model if model is not None else self._load_model(model_name, dataset)
        self.face_detector = FaceDetector2(conf_thres)
        self.img_size = [224,224] if self.model_variant == "b0" else [384,384]
        self.transforms = get_test_transforms(img_size=self.img_size)

        self.model.to(self.device).eval()
        
        # Explainability setup
        self.reshape_fn = self._get_reshape_fn()
        self.target_layers = self._resolve_target_layers()

    def _load_model(self, model_name: str, dataset: Optional[str]) -> torch.nn.Module:
        """timm에 등록된 사전학습 모델을 불러온다."""
        return timm.create_model(model_name, pretrained=True, dataset=dataset)

    def _get_transformer_type(self) -> Literal["gcvit", "vit", "cnn"]:
        """모델 이름으로 branch의 transformer 종류를 판별한다.

        Returns:
            str: "gcvit", "vit", "cnn" (eff_b0/eff_b5: transformer 없는 EfficientNet baseline).
        """
        if "gcvit" in self.model_name:
            return "gcvit"
        if "vit" in self.model_name:
            return "vit"
        return "cnn"

    def _branch_root(self) -> torch.nn.Module:
        """feat_extractor와 l_/h_ branch를 가진 모듈을 반환한다."""
        return self.model

    def _get_reshape_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """모델 종류에 맞는 Feature map reshape 함수를 반환한다.

        Returns:
            Callable: 3D/4D 텐서를 GradCAM 연산에 적합한 형태로 변환하는 함수.
        """
        if self.transformer_type == "cnn":
            return None  # backbone feature map은 이미 (B,C,H,W)
        return reshape_transform_vit if self.transformer_type == "vit" else reshape_transform_gcvit

    def _validate_branch_level(self) -> None:
        """Ablation 모델이 실제로 가진 branch와 branch_level이 일치하는지 검증한다.

        Raises:
            ValueError: low_/high_ 단일 branch 모델에 존재하지 않는 branch를 요청한 경우 발생.
        """
        if self.branch_level not in ("low", "high"):
            raise ValueError(f"branch_level은 'low' 또는 'high'만 가능합니다 (입력: {self.branch_level!r}). 두 branch 결합은 BranchFusionExplainer를 사용하세요.")
        if self.transformer_type == "cnn" and self.branch_level != "high":
            raise ValueError(f"{self.model_name}은 마지막 feature map만 사용하므로 branch_level='high'만 가능합니다.")
        if self.model_name.startswith("low_") and self.branch_level != "low":
            raise ValueError(f"{self.model_name}은 low-level branch만 가지므로 branch_level='low'만 가능합니다.")
        if self.model_name.startswith("high_") and self.branch_level != "high":
            raise ValueError(f"{self.model_name}은 high-level branch만 가지므로 branch_level='high'만 가능합니다.")

    def _get_backbone_layer(self, feature_idx: int) -> torch.nn.Module:
        """EfficientNet backbone에서 feature_idx 단계의 출력 모듈을 찾는다."""
        backbone = self._branch_root().feat_extractor.backbone
        name = backbone.feature_info[feature_idx]["module"]  # e.g. "blocks.6"
        modules = dict(backbone.named_modules())
        # timm FeatureListNet은 이름의 '.'를 '_'로 바꾸는 경우가 있어 둘 다 시도
        for key in (name, name.replace(".", "_")):
            if key in modules:
                return modules[key]
        raise ValueError(f"backbone에서 '{name}' 모듈을 찾을 수 없습니다.")

    def _resolve_target_layers(self) -> list[torch.nn.Module]:
        """시각화(XAI)의 대상이 될 Target Layer들을 선택한다.

        각 branch의 마지막 block 정규화 레이어로 고정한다.
            - gcvit: 마지막 stage 마지막 GCViTBlock의 norm2
            - vit: 마지막 ViTBlock의 norm1
            - cnn: backbone의 마지막 feature map

        Returns:
            list[torch.nn.Module]: 분석 대상 레이어 리스트.
        """
        root = self._branch_root()
        layers = []

        if self.transformer_type == "cnn":
            # eff_b0/eff_b5: transformer가 없으므로 backbone의 마지막 feature map을 직접 사용
            layers.append(self._get_backbone_layer(root.h_meta["feature_idx"]))

        elif self.transformer_type == "gcvit":
            if self.branch_level == "low":
                layers.append(root.l_gcvit.stages[-1].blocks[-1].norm2)
            else:
                layers.append(root.h_gcvit.stages[-1].blocks[-1].norm2)

        else:  # vit
            if self.branch_level == "low":
                layers.append(root.l_vit.blocks[-1].norm1)
            else:
                layers.append(root.h_vit.blocks[-1].norm1)

        return layers

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
                f"branch_level={self.branch_level}, device={self.device})")