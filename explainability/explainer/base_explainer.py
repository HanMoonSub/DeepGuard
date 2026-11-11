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
        dataset: Optional[Literal["celeb_df_v2", "ff++", "kodf"]] = None, # DeepClip models don't need a dataset
        margin_ratio: float = 0.2,
        conf_thres: float = 0.5,
        branch_level: Literal["low", "high"] = "high", # use BranchFusionExplainer to combine both branches
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
        """Load a pretrained model registered with timm."""
        return timm.create_model(model_name, pretrained=True, dataset=dataset)

    def _get_transformer_type(self) -> Literal["gcvit", "vit", "cnn"]:
        """Determine the branch transformer type from the model name.

        Returns:
            str: "gcvit", "vit", "cnn" (eff_b0/eff_b5: EfficientNet baseline without a transformer).
        """
        if "gcvit" in self.model_name:
            return "gcvit"
        if "vit" in self.model_name:
            return "vit"
        return "cnn"

    def _branch_root(self) -> torch.nn.Module:
        """Return the module that holds feat_extractor and the l_/h_ branches."""
        return self.model

    def _get_reshape_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return the feature map reshape function for the model type.

        Returns:
            Callable: function that converts 3D/4D tensors into a shape suitable for GradCAM.
        """
        if self.transformer_type == "cnn":
            return None  # backbone feature maps are already (B,C,H,W)
        return reshape_transform_vit if self.transformer_type == "vit" else reshape_transform_gcvit

    def _validate_branch_level(self) -> None:
        """Check that branch_level matches the branches the ablation model actually has.

        Raises:
            ValueError: raised when a branch that doesn't exist in a low_/high_ single-branch model is requested.
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
        """Find the output module of stage feature_idx in the EfficientNet backbone."""
        backbone = self._branch_root().feat_extractor.backbone
        name = backbone.feature_info[feature_idx]["module"]  # e.g. "blocks.6"
        modules = dict(backbone.named_modules())
        # timm FeatureListNet sometimes replaces '.' in names with '_', so try both
        for key in (name, name.replace(".", "_")):
            if key in modules:
                return modules[key]
        raise ValueError(f"backbone에서 '{name}' 모듈을 찾을 수 없습니다.")

    def _resolve_target_layers(self) -> list[torch.nn.Module]:
        """Select the target layers for visualization (XAI).

        Fixed to the normalization layer of the last block in each branch.
            - gcvit: norm2 of the last GCViTBlock in the last stage
            - vit: norm1 of the last ViTBlock
            - cnn: the last backbone feature map

        Returns:
            list[torch.nn.Module]: list of target layers.
        """
        root = self._branch_root()
        layers = []

        if self.transformer_type == "cnn":
            # eff_b0/eff_b5: no transformer, so use the last backbone feature map directly
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
        """Extract face bounding box coordinates from the image with the face detector (YOLO).

        Args:
            img (np.ndarray): RGB image data in HWC layout.

        Returns:
            list[float]: coordinate list in [xmin, ymin, xmax, ymax, confidence] form.
        """
        result = self.face_detector.detect_batch([img], [1.0])
        return result[0]
    
    def _crop_face(self, img: np.ndarray, bbox: List[float]) -> np.ndarray:
        """Crop the face region around the bounding box, taking the margin ratio into account.

        Args:
            img (np.ndarray): full original image.
            bbox (list[float]): detected [xmin, ymin, xmax, ymax] face coordinates.

        Returns:
            np.ndarray: cropped face image region.
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
        """Read the image path and run the full face detection and cropping process.

        Args:
            img_path (str): input image file path.

        Returns:
            np.ndarray: preprocessed, cropped face image (RGB).
        """
        img = cv2.imread(img_path)        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detect_result = self._get_face_bbox(img)                
        bbox = detect_result[:4]
        cropped = self._crop_face(img, bbox)
        return cropped
                  
    def _get_transform(self, img_path: str) -> tuple[np.ndarray, torch.Tensor]:
        """Take an image path and produce both the cropped image and the model input tensor.

        Args:
            img_path (str): input image file path.

        Returns:
            tuple[np.ndarray, torch.Tensor]: (cropped face image, batch tensor of shape (1, 3, H, W)).
        """
        face = self._preprocess_img(img_path)
        tensor = self.transforms(image=face)['image'].unsqueeze(0) # (1,3,H,W)
        return face, tensor
    
    def _get_transform_from_array(self, face: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
        """Convert an already-cropped face array into a model input tensor.

        Args:
            face (np.ndarray): already-cropped face image array (RGB).

        Returns:
            tuple[np.ndarray, torch.Tensor]: (original face image, batch tensor of shape (1, 3, H, W)).
        """
        tensor = self.transforms(image=face)['image'].unsqueeze(0) # (1,3,H,W)
        return face, tensor
            
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"model={self.model_name}, dataset={self.dataset}, "
                f"margin_ratio={self.margin_ratio}, conf_thres={self.conf_thres}, "
                f"branch_level={self.branch_level}, device={self.device})")