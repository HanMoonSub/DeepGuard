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

    Used for model families that aren't registered with timm (e.g. IID), so
    there is no per-dataset weight_registry / automatic pretrained lookup -
    the caller points this at whichever checkpoint they trained/downloaded.

    Args:
        weight_path: Local file path, or an http(s) URL (downloaded and
            cached via torch.hub) to a state_dict saved with torch.save().

    Note:
        Loaded with strict=False since some checkpoints only contain the
        trainable subset of parameters - the rest of the model (a frozen
        sub-network) is already correctly initialized by the model's own
        constructor. Any unexpected_keys are still surfaced as a warning,
        since those would indicate a genuine mismatch rather than an
        intentionally-omitted frozen parameter.
    """
    if weight_path.startswith("http://") or weight_path.startswith("https://"):
        state_dict = torch.hub.load_state_dict_from_url(weight_path, map_location="cpu")
    else:
        state_dict = torch.load(weight_path, map_location="cpu")

    result = model.load_state_dict(state_dict, strict=False)
    if result.unexpected_keys:
        print(f"⚠️  load_pretrained_weights: unexpected keys in checkpoint (not in model): {result.unexpected_keys}")
    return model


def build_model(
        model_name: str,
        dataset: str,
        weight_path: str = None,
        explicit_extractor_path: str = None,
        ) -> Tuple[nn.Module, List[int], Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Builds a DeepGuard model and loads its pretrained weights for inference.

    Most model families (ms_eff_gcvit_*, ms_eff_vit_*, ...) are registered
    with timm and go through timm's own pretrained-loading path. IID is a
    plain nn.Module instead, so it is built directly and its checkpoint is
    loaded from manually supplied paths.

    Args:
        weight_path: Required when model_name == "iid" - a local file path or
            URL pointing at IID's trained checkpoint (Xception backbone + head
            only; the frozen explicit extractor isn't saved in it). Ignored
            for timm-registered models.
        explicit_extractor_path: Required when model_name == "iid" - a local
            path to the frozen ArcFace-style IResNet-50 checkpoint. Without
            it, IID's explicit identity branch is randomly initialized and
            the explicit-vs-implicit comparison is meaningless. Ignored for
            every other model.

    Returns:
        model: eval-mode-ready model (caller still moves it to device / calls .eval()).
        img_size: expected square input resolution [H, W].
        mean, std: normalization statistics this model's backbone was pretrained with.
    """
    if model_name == "iid":
        from deepguard.models.iid import IID, IID_MEAN, IID_STD

        if not explicit_extractor_path:
            raise ValueError("iid requires an explicit explicit_extractor_path (frozen ArcFace checkpoint) - there is no automatic lookup for this model.")
        if not weight_path:
            raise ValueError("iid requires an explicit weight_path (trained backbone+head checkpoint) - there is no automatic pretrained weight lookup for this model.")

        model = IID(explicit_extractor_path=explicit_extractor_path)
        load_pretrained_weights(model, weight_path)
        return model, [256, 256], IID_MEAN, IID_STD

    model = timm.create_model(model_name, pretrained=True, dataset=dataset)
    img_size = [224, 224] if model_name.split("_")[-1] == "b0" else [384, 384]
    return model, img_size, IMAGENET_MEAN, IMAGENET_STD
