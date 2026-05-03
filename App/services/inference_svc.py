import timm
import asyncio
from inference.image_predictor_prt import ImagePredictor
from inference.video_predictor_prt import VideoPredictor
from inference.utils import PredictorError
from fastapi import status, Depends
from fastapi.exceptions import HTTPException
from sqlalchemy import text, Connection
from sqlalchemy.exc import SQLAlchemyError
from services import image_svc, video_svc
from db.database import context_get_conn, background_db_conn, engine
from schemas.image_schema import ImageData_indi
from schemas.video_schema import VideoData, VideoData_indi
from celery_app import celery_app

image_cache = {}
video_cache = {}

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
    
    # image_loc: static 폴더 내 저장된 사용자 이미지 경로
    # version_type: v1, v2
    # model_type: fast, pro
    # domain_type: 서양인, 동양인
    
    model_name, dataset = MODEL_CONFIG[version_type][model_type][domain_type]

    # 캐시 확인 및 모델 초기화
    cache_key = (model_name, dataset)
    if cache_key not in image_cache:
        image_cache[cache_key] = ImagePredictor(
            margin_ratio=0.2,
            conf_thres=0.5,
            model_name=model_name,
            dataset=dataset
        )
    predictor = image_cache[cache_key]
    
    # 비동기 이미지 추론 실행
    loop = asyncio.get_running_loop()
    try:
        analysis = await loop.run_in_executor(
            None, predictor.predict_img, "." + image_loc, 0.0
        )
        
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
            # 이미지 비동기 추론, DeepFake 결과값 반환
            result = await predict_image(image_loc, version_type, model_type, domain_type)
        
            async with background_db_conn() as conn:    
                # 로그인 상관없이 이미지 추론 결과값 DB에 저장하기
                await image_svc.update_image_result(
                    conn, 
                    image_id, 
                    result["analysis"], 
                    result["message"], 
                    result["status"]
                    )
        except Exception as e:
            print(f"Image Result Update Error: {str(e)}")
            try:
                async with background_db_conn() as conn:  
                    await image_svc.update_image_result(
                        conn, 
                        image_id, 
                        {"prob": -1, "face_conf": -1, "face_ratio": -1, "face_brightness": -1}, 
                        "이미지 추론 결과 업데이트 중 오류가 발생하였습니다.", 
                        "failed"
                    )
            except Exception as db_err:
                print(f"Final Emergency DB Update Failed: {db_err}")
            
        finally:
            if not user_id:
                await image_svc.delete_image(image_loc)

            # 비로그인 추론 결과값 반환만 하고 서버 내 이미지 파일 삭제
            # 다만, 바로 삭제하지 말고 특정 시간 이후에 삭제해야 한다.
            # if not user_id:
            #   await image_svc.delete_image_db(image_loc)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_inference())
    finally:
        loop.close()
    
# 이미지 결과 값 가져오기
async def get_image_result(conn: Connection, 
                           image_id: int):
    try:
        query = text("SELECT * FROM image_result WHERE id = :image_id")
        result = await conn.execute(query, {"image_id": image_id})
        
        if result.rowcount == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                                detail=f"요청하신 이미지 분석 결과(ID: {image_id})를 찾을 수 없습니다. ID를 다시 확인해주세요.")
        row = result.fetchone()
        
        image_data = ImageData_indi(
            image_id = row.id,
            user_id = row.user_id,
            image_loc = row.image_loc,
            status = row.status,
            label = row.label,
            score = row.score,
            face_conf = row.face_conf,
            face_ratio = row.face_ratio,
            face_brightness = row.face_brightness,
            version_type = row.version_type,
            model_type = row.model_type,
            domain_type = row.domain_type,
            result_msg = row.result_msg,
            created_at = row.created_at,
        )
        
        result.close()
        return image_data
        
    except SQLAlchemyError as e:
        print(f"이미지 분석 결과값 가져오기 실패: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                            detail="요청하신 서비스가 잠시 내부적으로 문제가 발생하였습니다.")
    except HTTPException:
        raise
    except Exception as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                            detail="알수없는 이유로 문제가 발생하였습니다.")
        
# 사용자 비디오 딥페이크 여부 판단 로직
async def predict_video(video_loc: str, version_type: str, model_type: str, domain_type: str):
    
    # video_loc: static 폴더 내 저장된 사용자 비디오 경로
    # version_type: v1, v2
    # model_type: fast, pro
    # domain_type: 서양인, 동양인
    
    model_name, dataset = MODEL_CONFIG[version_type][model_type][domain_type]

    # 캐시 확인 및 모델 초기화
    cache_key = (model_name, dataset)
    if cache_key not in video_cache:
        video_cache[cache_key] = VideoPredictor(
            margin_ratio=0.2,
            conf_thres=0.5,
            model_name=model_name,
            dataset=dataset
        )
    predictor = video_cache[cache_key]
    
    
    # 비동기 비디오 추론 실행
    loop = asyncio.get_running_loop()
    try:
        analysis = await loop.run_in_executor(
            None, predictor.predict_video, "." + video_loc, 10, 'conf', 0.0
        )
        
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
            "analysis": {"prob": -1, "face_conf": -1, "face_ratio": -1, "face_brightness": -1},
            "message": e.message,
            "status": "warning",
        }
    
    # CUDA Out of Memory, CUDA Device Error, Timm 모드 로델 실패 등..
    except Exception as e:
        print(str(e))
        return {
            "analysis": {"prob": -1, "face_conf": -1, "face_ratio": -1, "face_brightness": -1},
            "message": f"비디오 분석 중 치명적인 오류가 발생했습니다.",
            "status": "failed"
        }
        
# Celery 비디오 추론 Task
@celery_app.task(name="process_video_task")
def process_video_task(video_id: int, video_loc: str, version_type: str, model_type: str, 
                             domain_type: str, user_id: int | None):
    async def run_inference():
        try:
            # 비디오 비동기 추론, DeepFake 결과값 반환
            result = await predict_video(video_loc, version_type, model_type, domain_type)
        
            async with background_db_conn() as conn:    
                # 로그인 상관없이 추론 결과값 DB에 저장하기
                await video_svc.update_video_result(
                    conn, 
                    video_id, 
                    result["analysis"], 
                    result["message"], 
                    result["status"]
                    )
        except Exception as e:
            print(f"Video Result Update Error: {str(e)}")
            try:
                async with background_db_conn() as conn:  
                    await video_svc.update_video_result(
                        conn, 
                        video_id, 
                        {"prob": -1, "face_conf": -1, "face_ratio": -1, "face_brightness": -1}, 
                        "비디오 추론 결과 업데이트 중 오류가 발생하였습니다.", 
                        "failed"
                    )
            except Exception as db_err:
                print(f"Final Emergency DB Update Failed: {db_err}")
            
        finally:
            if not user_id:
                await video_svc.delete_video(video_loc)

            # 비로그인 추론 결과값 반환만 하고 서버 내 비디오 파일 삭제
            # 다만, 바로 삭제하지 말고 특정 시간 이후에 삭제해야 한다.
            # if not user_id:
            #   await video_svc.delete_video_db(video_loc)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_inference())
    finally:
        loop.close()
    
# 비디오 결과 값 가져오기
async def get_video_result(conn: Connection, 
                           video_id: int):
    try:
        query = text("SELECT * FROM video_result WHERE id = :video_id")
        result = await conn.execute(query, {"video_id": video_id})
        
        if result.rowcount == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                                detail=f"요청하신 비디오 분석 결과(ID: {video_id})를 찾을 수 없습니다. ID를 다시 확인해주세요.")
        row = result.fetchone()
        
        video_data = VideoData_indi(
            id = row.id,
            user_id = row.user_id,
            video_loc = row.video_loc,
            status = row.status,
            label = row.label,
            score = row.score,
            face_conf = row.face_conf,
            face_ratio = row.face_ratio,
            face_brightness = row.face_brightness,
            version_type = row.version_type,
            model_type = row.model_type,
            domain_type = row.domain_type,
            result_msg = row.result_msg,
            created_at = row.created_at,
        )
        
        result.close()
        return video_data
        
    except SQLAlchemyError as e:
        print(f"비디오 분석 결과값 가져오기 실패: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                            detail="요청하신 서비스가 잠시 내부적으로 문제가 발생하였습니다.")
    except HTTPException:
        raise
    except Exception as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                            detail="알수없는 이유로 문제가 발생하였습니다.")
    