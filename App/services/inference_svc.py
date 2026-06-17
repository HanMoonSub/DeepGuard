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

def _get_or_create_predictor(
    model_name: str,
    dataset: str,
    predictor_mode: Literal["image", "video"],
) -> ImagePredictor | VideoPredictor:
    """캐시에서 Predictor를 조회하거나, 없으면 새로 생성해 캐싱 후 반환한다.

    이미지/비디오 캐시를 분리 관리해 모드별 독립적인 인스턴스를 유지한다.

    Args:
        model_name (str): 추론에 사용할 모델 아키텍처 이름.
        dataset (str): 모델 학습에 사용된 데이터셋 이름.
        predictor_mode (Literal["image", "video"]): Predictor 종류 선택.

    Returns:
        ImagePredictor | VideoPredictor: 캐시된 또는 새로 생성된 Predictor 인스턴스.
    """
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

def predict_image(image_loc: str, version_type: str, model_type: str, domain_type: str) -> dict:
    """이미지 경로를 받아 딥페이크 여부를 추론하고 분석 결과를 반환한다.

    Args:
        image_loc (str): 추론 대상 이미지의 서버 저장 경로 (DB 기준, '/'로 시작).
        version_type (str): 모델 버전 타입 (MODEL_CONFIG 키값).
        model_type (str): 추론 모드 ("fast" | "pro").
        domain_type (str): 탐지 도메인 ("서양인" | "동양인").

    Returns:
        dict:
            - analysis (dict): prob, face_conf, face_ratio, face_brightness 포함.
              얼굴 미감지 또는 오류 발생 시 모든 값 -1.
            - message (str): 처리 결과 메세지.
            - status (str): "success" | "warning" | "failed"
    """
    
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

@celery_app.task(name="process_image_task")
def process_image_task(image_id: int, image_loc: str, version_type: str, model_type: str, 
                             domain_type: str, user_id: int | None) -> None:
    """딥페이크 이미지 추론을 처리하는 Celery 비동기 태스크.

    추론 결과를 DB에 저장하며, SQLAlchemy 오류 발생 시 실패 상태로 업데이트한다.
    Celery 워커(동기) 내에서 asyncio 이벤트 루프를 직접 구동해 비동기 로직을 실행한다.

    Args:
        image_id (int): 이미지 레코드 PK.
        image_loc (str): 추론 대상 이미지의 서버 저장 경로 (DB 기준, '/'로 시작).
        version_type (str): 모델 버전 타입 (MODEL_CONFIG 키값).
        model_type (str): 추론 모드 ("fast" | "pro").
        domain_type (str): 탐지 도메인 ("서양인" | "동양인").
        user_id (int | None): 로그인 사용자 ID. 비회원이면 None.

    Note:
        비회원(user_id=None) 요청은 추론 완료 60초 후 이미지 파일 및 DB 레코드가 자동 삭제된다.
    """
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
    
        
def predict_video(video_loc: str, version_type: str, model_type: str, domain_type: str) -> dict:
    """비디오 경로를 받아 딥페이크 여부를 추론하고 분석 결과를 반환한다.

    Args:
        video_loc (str): 추론 대상 비디오의 서버 저장 경로 (DB 기준, '/'로 시작).
        version_type (str): 모델 버전 타입 (MODEL_CONFIG 키값).
        model_type (str): 추론 모드 ("fast" | "pro").
        domain_type (str): 탐지 도메인 ("서양인" | "동양인").

    Returns:
        dict:
            - analysis (dict): prob, face_conf, face_ratio, face_brightness,
              frame_results, fps, total_frames, num_sampled, num_extracted, num_detected 포함.
              오류 발생 시 주요 수치 값 -1.
            - message (str): 처리 결과 메세지.
            - status (str): "success" | "warning" | "failed"
    """
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
        
@celery_app.task(name="process_video_task")
def process_video_task(video_id: int, video_loc: str, version_type: str, model_type: str, 
                             domain_type: str, user_id: int | None) -> None:
    """딥페이크 비디오 추론을 처리하는 Celery 비동기 태스크.

    추론 결과를 DB에 저장하며, 성공 시 메타 데이터와 프레임별 결과를 추가로 저장한다.
    메타/프레임 저장 실패는 독립적으로 처리되어 메인 추론 결과에 영향을 주지 않는다.
    Celery 워커(동기) 내에서 asyncio 이벤트 루프를 직접 구동해 비동기 로직을 실행한다.

    Args:
        video_id (int): 비디오 레코드 PK.
        video_loc (str): 추론 대상 비디오의 서버 저장 경로 (DB 기준, '/'로 시작).
        version_type (str): 모델 버전 타입 (MODEL_CONFIG 키값).
        model_type (str): 추론 모드 ("fast" | "pro").
        domain_type (str): 탐지 도메인 ("서양인" | "동양인").
        user_id (int | None): 로그인 사용자 ID. 비회원이면 None.

    Note:
        비회원(user_id=None) 요청은 추론 완료 60초 후 비디오 파일 및 DB 레코드가 자동 삭제된다.
    """
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
    