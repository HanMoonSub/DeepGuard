import yaml
from types import SimpleNamespace

def read_yaml(yaml_path):
    """
    YAML 파일을 읽어서 Python dict로 반환하는 함수
    Args:
        yaml_path (str): 불러올 yaml 파일 경로
    Returns:
        dict: yaml 내용을 담은 딕셔너리
    """
    with open(yaml_path, "r") as f:
        yaml_dict = yaml.safe_load(f)
    return yaml_dict


def dict_to_namespace(d: dict):
    """
    dict를 재귀적으로 SimpleNamespace로 변환하는 함수
    - dict의 key를 속성처럼 접근할 수 있게 함 (ex. cfg.model.name)
    Args:
        d (dict): 변환할 딕셔너리
    Returns:
        SimpleNamespace: dot notation으로 접근 가능한 객체
    """
    if isinstance(d, dict):
        # dict의 각 key, value를 재귀적으로 SimpleNamespace로 변환
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    else:
        # dict가 아니면 그대로 반환 (ex. 숫자, 문자열 등)
        return d


def merge_dict(base: dict, specific: dict):
    """
    두 개의 dict를 병합하는 함수
    - base의 내용을 유지하면서, specific의 key가 있으면 덮어씀
    Args:
        base (dict): 기본 설정 (예: base.yaml)
        specific (dict): 덮어쓸 설정 (예: dfdc.yaml)
    Returns:
        dict: 병합된 딕셔너리
    """
    merged = base.copy()  # base 내용을 복사하여 시작
    for k, v in specific.items():
        merged[k] = v  # specific의 key로 base 값을 덮어씀
    return merged