import asyncio
from typing import Literal
from fastapi import status, Depends
from fastapi.exceptions import HTTPException
from sqlalchemy import text, Connection
from sqlalchemy.exc import SQLAlchemyError
from services import image_svc, video_svc
from db.database import celery_db_conn
from celery_app import celery_app
from inference.image_predictor_prt import ImagePredictor
from inference.video_predictor_prt import VideoPredictor
from inference.utils import PredictorError

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

_image_cache: dict = {}
_video_cache: dict = {}

 # 캐시 확인 및 모델 초기화
def _get_or_create_predictor(
    model_name: str,
    dataset: str,
    predictor_mode: Literal["image", "video"],
) -> ImagePredictor | VideoPredictor:
    cache_key = (model_name, dataset)
 
    if predictor_mode == "image":
        if cache_key not in _image_cache:
            _image_cache[cache_key] = ImagePredictor(
                margin_ratio=0.2,
                conf_thres=0.5,
                model_name=model_name,
                dataset=dataset,
            )
        return _image_cache[cache_key]
 
    if cache_key not in _video_cache:
        _video_cache[cache_key] = VideoPredictor(
            margin_ratio=0.2,
            conf_thres=0.5,
            model_name=model_name,
            dataset=dataset,
        )
    return _video_cache[cache_key]

# 사용자 이미지 딥페이크 여부 판단 로직
def predict_image(image_loc: str, version_type: str, model_type: str, domain_type: str):
    
    model_name, dataset = MODEL_CONFIG[version_type][model_type][domain_type]

    predictor = _get_or_create_predictor(model_name, dataset, "image")
    image_path = "." + image_loc
    
    try:
        analysis = predictor.predict_img(image_path)
        
        print(f"딥페이크 확률 값: {analysis['prob']}, 얼굴 신뢰도: {analysis['face_conf']}, 얼굴 비율: {analysis['face_ratio']}, 얼굴 밝기: {analysis['face_brightness']}")
    
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
        return {
            "analysis": {"prob": -1, "face_conf": -1, "face_ratio": -1, "face_brightness": -1},
            "message": f"이미지 분석 중 치명적인 오류가 발생했습니다.",
            "status": "failed"
        }

# Celery 이미지 추론 Task
@celery_app.task(name="process_image_task")
def process_image_task(image_id: int, image_loc: str, version_type: str, model_type: str, 
                             domain_type: str, user_id: int | None):
    async def run_inference():    
        try:
            # 이미지 추론, DeepFake 결과값 반환
            result = predict_image(image_loc, version_type, model_type, domain_type)
        
            async with celery_db_conn() as conn:    
                # 로그인 상관없이 이미지 추론 결과값 DB에 저장하기
                await image_svc.update_image_result(
                    conn, 
                    image_id, 
                    result["analysis"], 
                    result["message"], 
                    result["status"]
                    )
        except SQLAlchemyError as e:
            print(f"[DB Error] Image Result Update Failed: {str(e)}")
            try:
                async with celery_db_conn() as conn:  
                    await image_svc.update_image_result(
                        conn, 
                        image_id, 
                        {"prob": -1, "face_conf": -1, "face_ratio": -1, "face_brightness": -1}, 
                        "이미지 추론 결과 업데이트 중 오류가 발생하였습니다.", 
                        "failed"
                    )
            except Exception as db_err:
                print(f"Final Emergency DB Update Failed: {db_err}")
        
        except Exception as e:
            print(f"[Unknown Error] Image Task Failed: {str(e)}")
        
        finally:
            if not user_id:
                image_svc.cleanup_anonymous_image.apply_async(args=[image_id, image_loc], countdown=60)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_inference())
    finally:
        loop.close()
    
        
# 사용자 비디오 딥페이크 여부 판단 로직
def predict_video(video_loc: str, version_type: str, model_type: str, domain_type: str):
    
    model_name, dataset = MODEL_CONFIG[version_type][model_type][domain_type]

    predictor = _get_or_create_predictor(model_name, dataset, "video")
    video_path = "." + video_loc
    
    try:
        analysis = predictor.predict_video(video_path)
        
        print(f"딥페이크 확률 값: {analysis['prob']}, 얼굴 신뢰도: {analysis['face_conf']}, 얼굴 비율: {analysis['face_ratio']}, 얼굴 밝기: {analysis['face_brightness']}")
    
        # 분석 성공 시
        return {
            "analysis": analysis,
            "message": "비디오 분석에 성공하였습니다",
            "status": "success"
        }
    except PredictorError as e:
        print(e.message)
        return {
            "analysis": {"prob": -1, "face_conf": -1, "face_ratio": -1, "face_brightness": -1, "frame_results": [],
                         "fps": None, "total_frames": None, "num_sampled": None, "num_extracted": None, "num_detected": None,},
            "message": e.message,
            "status": "warning",
        }
    
    # CUDA Out of Memory, CUDA Device Error, Timm 모드 로델 실패 등..
    except Exception as e:
        print(str(e))
        return {
            "analysis": {"prob": -1, "face_conf": -1, "face_ratio": -1, "face_brightness": -1, "frame_results": [],
                         "fps": None, "total_frames": None, "num_sampled": None, "num_extracted": None, "num_detected": None,},
            "message": f"비디오 분석 중 치명적인 오류가 발생했습니다.",
            "status": "failed"
        }
        
# Celery 비디오 추론 Task
@celery_app.task(name="process_video_task")
def process_video_task(video_id: int, video_loc: str, version_type: str, model_type: str, 
                             domain_type: str, user_id: int | None):
    async def run_inference():
        try:
            # 비디오 추론, DeepFake 결과값 반환
            result = predict_video(video_loc, version_type, model_type, domain_type)
        
            async with celery_db_conn() as conn:    
                # 로그인 상관없이 추론 결과값 DB에 저장하기
                await video_svc.update_video_result(
                    conn, 
                    video_id, 
                    result["analysis"], 
                    result["message"], 
                    result["status"]
                    )
            if result["status"] == "success":
                if result["analysis"].get("fps") is not None:
                    try:
                        async with celery_db_conn() as conn:
                            await video_svc.save_video_meta_result(conn, video_id, result["analysis"])
                    except Exception as e:
                        print(f"[Video Meta Save Error] video_id={video_id}: {e}")

                if result["analysis"].get("frame_results"):
                    try:
                        async with celery_db_conn() as conn:
                            await video_svc.save_video_frame_result(
                                conn, video_id, result["analysis"]["frame_results"]
                            )
                    except Exception as e:
                        print(f"[Video Frame Save Error] video_id={video_id}: {e}")
                        
        except SQLAlchemyError as e:
            print(f"Video DB Update Error: {e}")
            try:
                async with celery_db_conn() as conn:  
                    await video_svc.update_video_result(
                        conn, 
                        video_id, 
                        {"prob": -1, "face_conf": -1, "face_ratio": -1, "face_brightness": -1}, 
                        "비디오 추론 결과 업데이트 중 오류가 발생하였습니다.", 
                        "failed"
                    )          
            except Exception as db_err:
                print(f"Final Emergency DB Update Failed: {db_err}")
        
        except Exception as e:
            print(f"[Unknown Error] Video Task Failed: {str(e)}")
          
        finally:
            if not user_id:
                is_success = result.get("status") == "success"
                video_svc.cleanup_anonymous_video.apply_async(args=[video_id, video_loc, is_success], countdown=60)

                

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_inference())
    finally:
        loop.close()
    