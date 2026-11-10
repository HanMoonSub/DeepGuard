# deepclip is an optional extra: pip install deepguard[deepclip]
try:
    import deepclip  # registers deepclip_b0 / deepclip_b5 in timm
    from deepclip.models.deepclip import available_models
except ImportError:
    deepclip = None


def check_deepclip_model_name(model_name: str) -> None:
    """deepclip 패키지 설치 여부와 DeepClip 모델 이름을 검증한다.

    Raises:
        ImportError: deepclip 패키지가 설치되어 있지 않은 경우 발생.
        ValueError: 존재하지 않는 DeepClip 모델 이름인 경우 발생.
    """
    if deepclip is None:
        raise ImportError(
            "DeepClip models need the deepclip package: pip install deepguard[deepclip]"
        )
    if model_name not in available_models():
        raise ValueError(
            f"Unknown DeepClip model {model_name!r}. Available: {available_models()}"
        )
