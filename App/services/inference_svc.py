import timm
import asyncio
from inference.image_predictor_prt import ImagePredictor
from inference.utils import PredictorError
from fastapi import status
from fastapi.exceptions import HTTPException
from sqlalchemy import text, Connection
from sqlalchemy.exc import SQLAlchemyError
from services.image_svc import delete_image


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

# 사용자 이미지 딥페이크 여부 판단 로직
async def predict_image(image_loc: str, version_type: str, model_type: str, domain_type: str):
    
    # version_type: v1, v2
    # model_type: fast, pro
    # domain_type: 서양인, 동양인
    
    model_name, dataset = MODEL_CONFIG[version_type][model_type][domain_type]

    # 캐시 확인 및 모델 초기화
    cache_key = (model_name, dataset)
    if cache_key not in model_cache:
        model_cache[cache_key] = ImagePredictor(
            margin_ratio=0.2,
            conf_thres=0.5,
            model_name=model_name,
            dataset=dataset
        )
    predictor = model_cache[cache_key]
    
    
    # 비동기 이미지 추론 실행
    loop = asyncio.get_running_loop()
    try:
        analysis = await loop.run_in_executor(
            None, 
            predictor.predict_img, 
            "." + image_loc, 
            0.0
        )
        
        print(f"딥페이크 확률 값: {analysis["prob"]}, 얼굴 신뢰도: {analysis["face_conf"]}, 얼굴 비율: {analysis["face_ratio"]}, 얼굴 밝기: {analysis["face_brightness"]}")
    
        # 분석 성공 시
        return {
            "analysis": analysis,
            "message": "이미지 분석에 성공하였습니다",
            "status": "success"
        }
    except PredictorError as e:
        print(e.message)
        return {
            "analysis": {"prob": -1, "face_conf": -1, "face_ratio": -1, "face_brightness": -1},
            "message": e.message,
            "status": "warning",
        }
    
    # CUDA Out of Memory, CUDA Device Error, Timm 모드 로델 실패 등..
    except Exception as e:
        print(str(e))
        await delete_image(image_loc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="서버 분석 중 치명적인 오류가 발생했습니다."
        )

# 이미지 메타데이터 + 추론 결과값 DB에 저장
async def register_image_result(conn: Connection, user_id: int, image_loc: str, score: float, face_conf: float, face_ratio: float,
                                face_brightness: float, version_type: str, model_type: str, domain_type: str, result_msg: str):
    if score > 0.5: # 0.5 ~ 1.0 구간 
        label = "FAKE"
    elif score >= 0: # 0.0 ~ 0.5 구간
        label = "REAL"
    else: # score가 -1인 경우 
        label = "UNKNOWN" 
        
    try:
        query = """
        INSERT INTO image_result(user_id, image_loc, label, score, face_conf, face_ratio, face_brightness, version_type, model_type, domain_type, result_msg)
        values (:user_id, :image_loc, :label, :score, :face_conf, :face_ratio, :face_brightness, :version_type, :model_type, :domain_type, :result_msg)
        """
        
        stmt = text(query)
        bind_stmt = stmt.bindparams(user_id=user_id, image_loc=image_loc, label=label, score=score, face_conf=face_conf, face_ratio=face_ratio,
                                    face_brightness=face_brightness, version_type=version_type, model_type=model_type, domain_type=domain_type, result_msg=result_msg)
        await conn.execute(bind_stmt)
        await conn.commit()
        
    except SQLAlchemyError as e:
        print(e)
        await conn.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="요청데이터가 제대로 전달되지 않았습니다")

    
    
    