import timm
import asyncio
from inference.image_predictor import ImagePredictor
from fastapi import status
from fastapi.exceptions import HTTPException

model_cache = {}

# 사용자 모델 설정 변수명 
MODEL_CONFIG = {
    'v1': {
        'fast': {'서양인': ("ms_eff_vit_b0", "ff++")},
        'pro':  {'서양인': ("ms_eff_vit_b5", "ff++")}
    },
    'v2': {
        'fast': {
            '서양인': ("ms_eff_gcvit_b0", "ff++"),
            '동양인': ("ms_eff_gcvit_b0", "kodf")
        },
        'pro': {
            '서양인': ("ms_eff_gcvit_b5", "ff++"),
            '동양인': ("ms_eff_gcvit_b5", "kodf")
        }
    }
}

async def predict_image(image_loc: str, version_type: str, model_type: str, domain_type: str):
    
    # version_type: v1, v2
    # model_type: fast, pro
    # domain_type: 서양인, 동양인
    
    try:
        model_name, dataset = MODEL_CONFIG[version_type][model_type][domain_type]
    except KeyError:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail = f"지원하지 않는 모델 설정입니다. (입력값: {version_type}, {model_type}, {domain_type}). "
                     f"v1은 '서양인'만 지원하며, v2는 '서양인', '동양인' 모두 지원합니다.")

    # 캐시 확인 및 모델 초기화
    cache_key = (model_name, dataset)
    if cache_key not in model_cache:
        model_cache[cache_key] = ImagePredictor(
            margin_ratio=0.2,
            conf_thres=0.5,
            min_face_ratio=0.01,
            model_name=model_name,
            dataset=dataset
        )
    # 캐시 내 모델 존재 시, 바로 사용
    predictor = model_cache[cache_key]
    
    # 비동기 이미지 추론 실행
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, 
        predictor.predict_img, 
        f"./{image_loc}", 
        0.0
    )
    
    # 최종 출력값 출력
    print(f"Deepfake Probability: {result:.4f}")
    return result