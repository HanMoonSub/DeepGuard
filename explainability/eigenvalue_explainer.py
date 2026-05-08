from pytorch_grad_cam import EigenCAM, EigenGradCAM, KPCA_CAM, SegEigenCAM
from explainability.cam_explainer import CAMExplainer

class EigenCAMExplainer(CAMExplainer):
    """
    Takes the first principle component of the 2D Activations
    (no class discrimination, but seems to give great results)
    """
    def _build_cam(self):
        return EigenCAM(model=self.model, target_layers=self.target_layers, reshape_transform=self.reshape_fn)
    
class EigenGradCAMExplainer(CAMExplainer):
    """
    Like EigenCAM but with class discrimination
    First principle component of Activations*Grad
    Looks like GradCAM, but cleaner
    """
    def _build_cam(self):
        return EigenGradCAM(model=self.model, target_layers=self.target_layers, reshape_transform=self.reshape_fn)
    
class KPCACAMExplainer(CAMExplainer):
    """
    Like EigenCAM but with kernel PCA instead of PCA
    """
    def __init__(self, kernel: str = "sigmoid", gamma: float = None, **kwargs):
        self.kernel = kernel # sigmoid | rbf | poly 
        self.gamma  = gamma
        super().__init__(**kwargs)

    def _build_cam(self):
        return KPCA_CAM(model=self.model, target_layers=self.target_layers, reshape_transform=self.reshape_fn,
                        kernel=self.kernel, gamma=self.gamma)
    
class SegEigenCAMExplainer(CAMExplainer):
    """
    Like EigenCAM but with gradient weighting (absolute gradients ⊙ activations )
    before SVD and sign correction to fix SVD sign ambiguity
    designed for semantic segmentation
    """
    def _build_cam(self):
        return SegEigenCAM(model=self.model, target_layers=self.target_layers, reshape_transform=self.reshape_fn)
    
    

