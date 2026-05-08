from pytorch_grad_cam import LayerCAM, RandomCAM
from explainability.cam_explainer import CAMExplainer


class LayerCAMExplainer(CAMExplainer):
    """
    Spatially weight the activations by positive gradients.
    Works better especially in lower layers
    """
    def _build_cam(self):
        return LayerCAM(model=self.model, target_layers=self.target_layers, reshape_transform=self.reshape_fn)


