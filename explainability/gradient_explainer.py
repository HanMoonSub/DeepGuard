from pytorch_grad_cam import (
    GradCAM, GradCAMPlusPlus, GradCAMElementWise,
    XGradCAM, HiResCAM
)
from explainability.cam_explainer import CAMExplainer


class GradCAMExplainer(CAMExplainer):
    """
    Weight the 2D Activations by the average gradient
    """
    def _build_cam(self):
        return GradCAM(model=self.model, target_layers=self.target_layers, reshape_transform=self.reshape_fn)


class GradCAMPlusPlusExplainer(CAMExplainer):
    """
    Like GradCAM but uses second order gradients
    """    
    def _build_cam(self):
        return GradCAMPlusPlus(model=self.model, target_layers=self.target_layers, reshape_transform=self.reshape_fn)

class GradCAMElementWiseExplainer(CAMExplainer):
    """
    Like GradCAM but element-wise multiply the activations with the gradients then apply a ReLU operation before summing
    """
    def _build_cam(self):
        return GradCAMElementWise(model=self.model, target_layers=self.target_layers, reshape_transform=self.reshape_fn)


class XGradCAMExplainer(CAMExplainer):
    """
    Like GradCAM but scale the gradients by the normalized activations
    """
    def _build_cam(self):
        return XGradCAM(model=self.model, target_layers=self.target_layers, reshape_transform=self.reshape_fn)


class HiResCAMExplainer(CAMExplainer):
    """
    Like GradCAM but element-wise multiply the activations with the gradients; provably guaranteed faithfulness for certain models
    """
    def _build_cam(self):
        return HiResCAM(model=self.model, target_layers=self.target_layers, reshape_transform=self.reshape_fn)


    
