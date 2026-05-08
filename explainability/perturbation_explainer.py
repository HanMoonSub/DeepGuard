from pytorch_grad_cam import AblationCAM, ScoreCAM, FEM
from explainability.cam_explainer import CAMExplainer


class AblationCAMExplainer(CAMExplainer):
    """
    Zero out activations and measure how the output drops
    (this repository includes a fast batched implementation)
    """
    def __init__(self, batch_size: int = 32, ratio_channels_to_ablate: float = 1.0, **kwargs):
        self.batch_size = batch_size
        self.ratio_channels_to_ablate = ratio_channels_to_ablate
        super().__init__(**kwargs)

    def _build_cam(self):
        return AblationCAM(model=self.model, target_layers=self.target_layers, reshape_transform=self.reshape_fn,
                           batch_size=self.batch_size, ratio_channels_to_ablate=self.ratio_channels_to_ablate)


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