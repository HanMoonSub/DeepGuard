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

    Used for model families that aren't registered with timm (e.g. CORE), so
    there is no per-dataset weight_registry / automatic pretrained lookup -
    the caller points this at whichever checkpoint they trained/downloaded.

    Args:
        weight_path: Local file path, or an http(s) URL (downloaded and
            cached via torch.hub) to a state_dict saved with torch.save().

    Note:
        Loaded with strict=False, but any missing/unexpected keys are surfaced
        as a warning since they indicate a checkpoint/model mismatch.
    """
    if weight_path.startswith("http://") or weight_path.startswith("https://"):
        state_dict = torch.hub.load_state_dict_from_url(weight_path, map_location="cpu")
    else:
        state_dict = torch.load(weight_path, map_location="cpu")

    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(f"⚠️  load_pretrained_weights: missing keys in checkpoint (not loaded): {result.missing_keys}")
    if result.unexpected_keys:
        print(f"⚠️  load_pretrained_weights: unexpected keys in checkpoint (not in model): {result.unexpected_keys}")
    return model


def build_model(
        model_name: str,
        dataset: str,
        weight_path: str = None,
        ) -> Tuple[nn.Module, List[int], Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Builds a DeepGuard model and loads its pretrained weights for inference.

    Most model families (ms_eff_gcvit_*, ms_eff_vit_*, ...) are registered
    with timm and go through timm's own pretrained-loading path. CORE is a
    plain nn.Module instead, so it is built directly and its checkpoint is
    loaded from a manually supplied path.

    Args:
        weight_path: Required when model_name == "core" - a local file path or
            URL pointing at CORE's trained checkpoint (full state_dict).
            Ignored for timm-registered models.

    Returns:
        model: eval-mode-ready model (caller still moves it to device / calls .eval()).
        img_size: expected square input resolution [H, W].
        mean, std: normalization statistics this model was trained with.
    """
    if model_name == "core":
        from deepguard.models.core import CORE, CORE_MEAN, CORE_STD

        if not weight_path:
            raise ValueError("core requires an explicit weight_path (trained checkpoint) - there is no automatic pretrained weight lookup for this model.")

        model = CORE()
        load_pretrained_weights(model, weight_path)
        return model, [299, 299], CORE_MEAN, CORE_STD

    model = timm.create_model(model_name, pretrained=True, dataset=dataset)
    img_size = [224, 224] if model_name.split("_")[-1] == "b0" else [384, 384]
    return model, img_size, IMAGENET_MEAN, IMAGENET_STD
