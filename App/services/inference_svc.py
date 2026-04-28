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
from db.database import context_get_conn, background_db_conn
from schemas.video_schema import VideoData, VideoData_indi

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

# 빈 이미지 DB 생성 이후, image_id 반환(접수 완료)
async def register_image_result(conn: Connection, user_id: int | None, image_loc: str, 
                                version_type: str, model_type: str, domain_type: str):
    try:
        query = """
            INSERT INTO image_result (
                user_id, image_loc, status, label, score, face_conf, face_ratio,
                face_brightness, version_type, model_type, domain_type, result_msg
            )
            VALUES (
                :user_id, :image_loc, 'PENDING', 'UNKNOWN', -1.0, -1.0, -1.0,
                -1.0, :version_type, :model_type, :domain_type, '분석 대기 중...')
        """
        
        stmt = text(query)
        result = await conn.execute(stmt, {
            "user_id": user_id, 
            "image_loc": image_loc,
            "version_type": version_type,
            "model_type": model_type,
            "domain_type": domain_type
        })
        await conn.commit()
        
        # MySQL에서 방금 생성된 auto_increment ID 가져오기
        return result.lastrowid
        
    except SQLAlchemyError as e:
        print(f"DB Insert Error: {e}")
        await conn.rollback()
        await image_svc.delete_image(image_loc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="요청데이터가 제대로 전달되지 않았습니다")

# 사용자 이미지 딥페이크 여부 판단 로직
async def predict_image(image_loc: str, version_type: str, model_type: str, domain_type: str):
    
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
        await image_svc.delete_image(image_loc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="이미지 분석 중 치명적인 오류가 발생했습니다."
        )

# 이미지 메타데이터 + 추론 결과값 DB에 저장
async def update_image_result(conn: Connection, image_id: int, analysis: dict,
                              result_msg: str, status: str): 
    
    # status 종류: success, warning, failed
    # success: 이미지 추론 성공
    # warning: 이미지 추론 과정 PredictError 발생
    # failed: 이미지 추로 과정 알수 없는 오류 발생 
    
    if status == 'success':
        label = "FAKE" if analysis["prob"] > 0.5 else "REAL"
    else:
        label = "UNKNOWN"
    try:
        query = """
            UPDATE image_result 
            SET status = :status,
                label = :label, 
                score = :score, 
                face_conf = :face_conf, 
                face_ratio = :face_ratio, 
                face_brightness = :face_brightness, 
                result_msg = :result_msg
            WHERE id = :image_id
        """
        
        db_status = status.upper()
        
        stmt = text(query)
        await conn.execute(stmt, {
            "status": db_status,
            "label": label,
            "score": analysis["prob"],
            "face_conf": analysis["face_conf"],
            "face_ratio": analysis["face_ratio"],
            "face_brightness": analysis["face_brightness"],
            "result_msg": result_msg,
            "image_id": image_id
        })
        await conn.commit()
        
    except SQLAlchemyError as e:
        await conn.rollback()
        raise e

# 빈 비디오 DB 생성 이후, video_id 반환(접수 완료)
async def register_video_result(conn: Connection, user_id: int | None, video_loc: str,
                                version_type: str, model_type: str, domain_type: str):
    try:
        query = """
            INSERT INTO video_result (
                user_id, video_loc, status, label, score, face_conf, face_ratio, 
                face_brightness, version_type, model_type, domain_type, result_msg
            )
            VALUES (
                :user_id, :video_loc, 'PENDING', 'UNKNOWN', -1.0, -1.0, -1.0, 
                -1.0, :version_type, :model_type, :domain_type, '분석 대기 중...'
            )
        """
        stmt = text(query)
        result = await conn.execute(stmt, {
            "user_id": user_id, 
            "video_loc": video_loc,
            "version_type": version_type,
            "model_type": model_type,
            "domain_type": domain_type
        })
        await conn.commit()
        
        # MySQL에서 방금 생성된 auto_increment ID 가져오기
        return result.lastrowid
        
    except SQLAlchemyError as e:
        print(f"DB Insert Error: {e}")
        await conn.rollback()
        await video_svc.delete_video(video_loc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="요청데이터가 제대로 전달되지 않았습니다")
    
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
        
# 비디오 메타데이터 + 추론 결과값 DB에 저장
async def update_video_result(conn: Connection, video_id: int, analysis: dict,
                              result_msg: str, status: str):
    
    if status == 'success':
        label = "FAKE" if analysis["prob"] > 0.5 else "REAL"
    else:
        label = "UNKNOWN"
        
    try:
        query = """
            UPDATE video_result 
            SET status = :status,
                label = :label, 
                score = :score, 
                face_conf = :face_conf, 
                face_ratio = :face_ratio, 
                face_brightness = :face_brightness, 
                result_msg = :result_msg
            WHERE id = :video_id
        """
        
        db_status = status.upper()
        
        stmt = text(query)
        await conn.execute(stmt, {
            "status": db_status,
            "label": label,
            "score": analysis["prob"],
            "face_conf": analysis["face_conf"],
            "face_ratio": analysis["face_ratio"],
            "face_brightness": analysis["face_brightness"],
            "result_msg": result_msg,
            "video_id": video_id
        })
        await conn.commit()
        
    except SQLAlchemyError as e:
        await conn.rollback()
        raise e
        
# 비디오 딥페이크 분석 프로세스: 비디오 추론 + 비디오 DB 내 저장(백그라운드 실행 함수)
async def process_video_task(video_id: int, video_loc: str, version_type: str, model_type: str, 
                             domain_type: str, user_id: int | None):
    
    try:
        # 비디오 비동기 추론, DeepFake 결과값 반환
        result = await predict_video(video_loc, version_type, model_type, domain_type)
    
        async with background_db_conn() as conn:    
            # 로그인 상관없이 추론 결과값 DB에 저장하기
            await update_video_result(
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
                await update_video_result(
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
        
# 비디오 결과 값 가져오기
async def get_video_result(conn: Connection, 
                           video_id: int):
    try:
        query = text("SELECT * FROM video_result WHERE id = :video_id")
        result = await conn.execute(query, {"video_id": video_id})
        
        if result.rowcount == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                                detail="결과를 찾을 수 없습니다.")
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

    except Exception as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                            detail="알수없는 이유로 문제가 발생하였습니다.")
    