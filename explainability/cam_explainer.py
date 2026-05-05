import numpy as np
import torch
from .utils.model_targets import BinaryClassifierOutputTarget
from .base_explainer import BaseExplainer
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus

class CAMExpaliner(BaseExplainer):
    def __init__(self, 
                 cam_type: str,
                 aug_smooth: bool = False,
                 eigen_smooth: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.aug_smooth = aug_smooth
        self.eigen_smooth = eigen_smooth
        
    
        
        