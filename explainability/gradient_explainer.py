from pytorch_grad_cam import (
    GradCAM, GradCAMPlusPlus, GradCAMElementWise,
    XGradCAM, HiResCAM, FullGrad, ShapleyCAM, FinerCAM,
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


class FullGradExplainer(CAMExplainer):
    """
    Computes the gradients of the biases from all over the network, and then sums them
    """
    def _build_cam(self):
        return FullGrad(model=self.model, target_layers=self.target_layers, reshape_transform=self.reshape_fn)


class ShapleyCAMExplainer(CAMExplainer):
    """
    Weight the activations using the gradient and Hessian-vector product.
    """
    def _build_cam(self):
        return ShapleyCAM(model=self.model, target_layers=self.target_layers, reshape_transform=self.reshape_fn)


class FinerCAMExplainer(CAMExplainer):
    """
    Improves fine-grained classification by comparing similar classes, suppressing shared features and highlighting discriminative details.
    """
    def __init__(self, base_method=GradCAM, **kwargs):
        self.base_method = base_method
        super().__init__(**kwargs)

    def _build_cam(self):
        return FinerCAM(model=self.model, target_layers=self.target_layers, reshape_transform=self.reshape_fn, base_method=self.base_method)