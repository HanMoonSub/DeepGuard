from typing import List, Tuple

import timm
import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PredictorError(Exception):
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        super().__init__(self.message)


def load_pretrained_weights(model: nn.Module, weight_path: str) -> nn.Module:
    """
    Loads a manually supplied checkpoint into the model in place.

    Used for model families that aren't registered with timm (e.g. Effort),
    so there is no per-dataset weight_registry / automatic pretrained lookup -
    the caller points this at whichever checkpoint they trained/downloaded.

    Args:
        weight_path: Local file path, or an http(s) URL (downloaded and
            cached via torch.hub) to a state_dict saved with torch.save().
    """
    if weight_path.startswith("http://") or weight_path.startswith("https://"):
        state_dict = torch.hub.load_state_dict_from_url(weight_path, map_location="cpu")
    else:
        state_dict = torch.load(weight_path, map_location="cpu")

    model.load_state_dict(state_dict)
    return model


def build_model(model_name: str, dataset: str, weight_path: str = None) -> Tuple[nn.Module, List[int], Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Builds a DeepGuard model and loads its pretrained weights for inference.

    Most model families (ms_eff_gcvit_*, ms_eff_vit_*, ...) are registered
    with timm and go through timm's own pretrained-loading path. Effort is a
    plain nn.Module wrapping a Hugging Face CLIP backbone instead, so it is
    built directly and its checkpoint is loaded from a manually supplied path.

    Args:
        weight_path: Required when model_name == "effort" - a local file path
            or URL pointing at a trained Effort checkpoint. Ignored otherwise.

    Returns:
        model: eval-mode-ready model (caller still moves it to device / calls .eval()).
        img_size: expected square input resolution [H, W].
        mean, std: normalization statistics this model's backbone was pretrained with.
    """
    if model_name == "effort":
        from deepguard.models.effort import Effort, CLIP_MEAN, CLIP_STD

        if not weight_path:
            raise ValueError("effort requires an explicit weight_path (local file path or URL) - there is no automatic pretrained weight lookup for this model.")

        model = Effort()
        load_pretrained_weights(model, weight_path)
        return model, [224, 224], CLIP_MEAN, CLIP_STD

    model = timm.create_model(model_name, pretrained=True, dataset=dataset)
    img_size = [224, 224] if model_name.split("_")[-1] == "b0" else [384, 384]
    return model, img_size, IMAGENET_MEAN, IMAGENET_STD
