import numpy as np
import timm
import torch

# deepclip is an optional extra: pip install deepguard[deepclip]
try:
    import deepclip  # noqa: F401  (registers deepclip_b0 / deepclip_b5 in timm)
    from deepclip.data.text import PAIR_ORDER
    from deepclip.models.deepclip import available_models
except ImportError:
    deepclip = None

class PredictorError(Exception):
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        super().__init__(self.message)

def to_artifact_scores(artifact_probs: np.ndarray) -> dict[str, dict[str, float]]:
    """
    (20,) artifact probabilities of a DeepClip model -> {region: {artifact type: probability}}

        region        : face, eyes, nose, mouth
        artifact type : blur, color, texture, structure, blend
    """
    scores = {}
    for (region, type_), prob in zip(PAIR_ORDER, artifact_probs):
        scores.setdefault(region, {})[type_] = float(prob)
    return scores

class ClipPredictorMixin:
    """
    DeepClip parts shared by the Clip predictors (pip and web versions).

    DeepClip has no dataset variants, and besides the fake probability it gives artifact scores:
    for each face region (face, eyes, nose, mouth), how likely each forgery artifact
    (blur, color, texture, structure, blend) is visible.
    """

    @staticmethod
    def _check_model_name(model_name: str):
        if deepclip is None:
            raise ImportError("DeepClip models need the deepclip package: pip install deepguard[deepclip]")
        if model_name not in available_models():
            raise ValueError(f"Unknown DeepClip model {model_name!r}. Available: {available_models()}")

    def _load_model(self, model_name: str, dataset: str | None = None):
        return timm.create_model(model_name, pretrained=True)

    def _forward_artifacts(self, batch: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """(B,3,H,W) -> (B,) fake probabilities, (B,20) artifact probabilities"""
        with torch.no_grad():
            img_emb = self.model.encode_image(batch)
            probs = torch.sigmoid(self.model.real_fake_logits(img_emb))
            artifact_probs = torch.sigmoid(self.model.pair_logits(img_emb))
        return probs.cpu().numpy(), artifact_probs.cpu().numpy()
