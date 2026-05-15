import numpy as np
import torch
from typing import List, Dict, Literal
from explainability.utils.model_targets import BinaryClassifierOutputSigmoidTarget
from explainability.explainer.cam_explainer import CAMExplainer
from explainability.metrics.road import (
    ROADLeastRelevantFirstAverage, ROADMostRelevantFirstAverage,
    ROADCombined
)

METRIC_REGISTRY = {
    "MORF": ROADMostRelevantFirstAverage,
    "LERF": ROADLeastRelevantFirstAverage,
    "Combined": ROADCombined,
}

class CAMEvaluator:
    def __init__(self,
                 cam_explainer: CAMExplainer,
                 cam_metric: Literal["Combined", "MORF", "LERF"] = "Combined", # Combined, MORF, LERF
                 percentiles: List[int] = [10,20,30,40,50,60,70,80,90],
                 seed: int = 2026
                 ):
        self.seed = seed
        self.cam_explainer = cam_explainer
        self.cam_metric_name = cam_metric
        self.percentiles = percentiles
        self.cam_metric = METRIC_REGISTRY[cam_metric](percentiles)
        
    def evaluate(self, img_path: str) -> float:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        grayscale_cam, tensor, _ = self.cam_explainer._build_grayscale_cam(img_path)
        targets = [BinaryClassifierOutputSigmoidTarget()]
        
        return self.cam_metric(
            input_tensor = tensor, # (1,C,H,W)
            cams = grayscale_cam[np.newaxis, ...], #(1,H,W)
            targets = targets,
            model = self.cam_explainer.model, 
        )[0]
    
    def __repr__(self):
        return (
            f"CAMEvaluator(explainer={self.cam_explainer.__class__.__name__}, "
            f"metric={self.cam_metric_name}, percentiles={self.percentiles})"
        )