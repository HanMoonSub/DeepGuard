# deepclip is an optional extra: pip install deepguard[deepclip]
try:
    import deepclip  # registers deepclip_b0 / deepclip_b5 in timm
    from deepclip.models.deepclip import available_models
except ImportError:
    deepclip = None


def check_deepclip_model_name(model_name: str) -> None:
    """Check that the deepclip package is installed and the DeepClip model name is valid.

    Raises:
        ImportError: raised when the deepclip package is not installed.
        ValueError: raised when the DeepClip model name doesn't exist.
    """
    if deepclip is None:
        raise ImportError(
            "DeepClip models need the deepclip package: pip install deepguard[deepclip]"
        )
    if model_name not in available_models():
        raise ValueError(
            f"Unknown DeepClip model {model_name!r}. Available: {available_models()}"
        )
