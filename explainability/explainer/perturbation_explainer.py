from pytorch_grad_cam import ScoreCAM, FEM
from explainability.explainer.cam_explainer import CAMExplainer
class ScoreCAMExplainer(CAMExplainer):
    """
    Perbutate the image by the scaled activation
    mesaure how the output drops 
    """
    def _build_cam(self):
        return ScoreCAM(model=self.model, target_layers=self.target_layers, reshape_transform=self.reshape_fn)

class FEMExplainer(CAMExplainer):
    """
    A gradient free method that binarizes activations
    by an activation > mean + k * std rule
    """
    def __init__(self, k: int = 2, **kwargs):
        self.k = k
        super().__init__(**kwargs)

    def _build_cam(self):
        return FEM(model=self.model, target_layers=self.target_layers, reshape_transform=self.reshape_fn, k=self.k)
    
    def __repr__(self):
        return super().__repr__().rstrip(")") + f", k={self.k})"