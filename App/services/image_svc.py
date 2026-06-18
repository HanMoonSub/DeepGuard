import os
import time
import cv2
import numpy as np
import aiofiles as aio
import asyncio
from dotenv import load_dotenv
from fastapi import UploadFile, status
from fastapi.exceptions import HTTPException
from sqlalchemy import text, Connection
from sqlalchemy.exc import SQLAlchemyError
from schemas.image_schema import ImageData, ImageDetailData, ImageResultData
from db.database import celery_db_conn
from celery_app import celery_app


load_dotenv()
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./static/uploads")
EXPLAIN_UPLOAD_DIR = os.getenv("EXPLAIN_UPLOAD_DIR", "./static/explain")

os.makedirs(EXPLAIN_UPLOAD_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

async def upload_image(user_email: str | None, imagefile: UploadFile) -> str:
    """업로드된 이미지를 서버에 저장하고 DB 저장용 경로를 반환한다. 회원/비회원 공통.

    Args:
        user_email (str | None): 로그인 사용자 이메일. 비회원이면 None.
        imagefile (UploadFile): FastAPI 업로드 파일 객체.

    Returns:
        str: 저장된 이미지의 DB 기준 경로 ('/'로 시작, '\\' 정규화됨).

    Raises:
        HTTPException 500: 파일 쓰기 또는 예기치 못한 오류 발생 시.
    """
    try:
        # 사용자별 하위 디렉토리 결정
        sub_dir = user_email if user_email else "anonymous"
        
        user_dir = os.path.join(UPLOAD_DIR, sub_dir)

        # 디렉토리 존재 확인 및 생성
        os.makedirs(user_dir, exist_ok=True)
    
        # 파일명 중복 방지 (파일명_타임스탬프.확장자)
        filename_only, ext = os.path.splitext(imagefile.filename)
        
        upload_filename = f"{filename_only}_{int(time.time())}{ext}"
        upload_image_loc = os.path.join(user_dir, upload_filename)

        # 비동기 이미지 저장 (Chunk 단위 읽기)
        try:
            async with aio.open(upload_image_loc, "wb") as outfile:
                while content := await imagefile.read(1024 * 1024):
                    await outfile.write(content)
        except Exception as e:
            print(f"[File Error] 파일 쓰기 실패: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="이미지 파일을 저장하는 중 오류가 발생했습니다."
            )

        # 6. DB 저장용 경로 반환
        return upload_image_loc[1:].replace("\\", "/")
    
    except HTTPException:
        raise
    
    except Exception as e:
        print(f"[Unknown Error] {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="이미지 업로드 과정에서 예상치 못한 오류가 발생했습니다.")

async def upload_image_cam(user_email: str, image_id: int, image: np.ndarray) -> str:
    """CAM 시각화 이미지를 서버에 저장하고 DB 저장용 경로를 반환한다. 회원 전용.

    Args:
        user_email (str): 요청 사용자 이메일 (저장 경로 구분용).
        image_id (int): 이미지 레코드 PK (파일명 구성에 사용).
        image (np.ndarray): 저장할 시각화 이미지 배열 (RGB).

    Returns:
        str: 저장된 CAM 이미지의 DB 기준 경로 ('/'로 시작).

    Raises:
        Exception: 파일 쓰기 실패 시 그대로 re-raise.
    """
    user_dir = os.path.join(EXPLAIN_UPLOAD_DIR, user_email)
    os.makedirs(user_dir, exist_ok=True)
    
    cam_filename = f"{image_id}_{int(time.time())}.png"
    cam_loc = os.path.join(user_dir, cam_filename)
            
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", image_bgr)

    try:
        async with aio.open(cam_loc, "wb") as outfile:
            await outfile.write(buf.tobytes())
    except Exception as e:
        raise e

    return cam_loc[1:].replace("\\", "/")

async def upload_frame_cam(user_email: str, video_id: int, frame_time: float, image: np.ndarray) -> str:
    """비디오 프레임 CAM 시각화 이미지를 서버에 저장하고 DB 저장용 경로를 반환한다. 회원 전용.

    Args:
        user_email (str): 요청 사용자 이메일 (저장 경로 구분용).
        video_id (int): 비디오 레코드 PK (파일명 구성에 사용).
        frame_time (float): 시각화한 프레임의 타임스탬프 (초 단위, 파일명 구성에 사용).
        image (np.ndarray): 저장할 시각화 이미지 배열 (RGB).

    Returns:
        str: 저장된 CAM 이미지의 DB 기준 경로 ('/'로 시작).

    Raises:
        Exception: 파일 쓰기 실패 시 그대로 re-raise.
    """
    user_dir = os.path.join(EXPLAIN_UPLOAD_DIR, user_email)
    os.makedirs(user_dir, exist_ok=True)
    
    cam_filename = f"v{video_id}_t{frame_time}_{int(time.time())}.png"
    cam_loc = os.path.join(user_dir, cam_filename)
            
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", image_bgr)

    try:
        async with aio.open(cam_loc, "wb") as outfile:
            await outfile.write(buf.tobytes())
    except Exception as e:
        raise e

    return cam_loc[1:].replace("\\", "/")

async def delete_image(image_loc: str):
    """서버에서 이미지 물리 파일을 삭제한다.

    파일이 존재하지 않으면 경고 로그만 출력하고 정상 종료한다.

    Args:
        image_loc (str): 삭제할 이미지의 DB 기준 경로 ('/'로 시작).

    Raises:
        HTTPException 500: 파일 삭제 중 예기치 못한 오류 발생 시.
    """
    try:
        
        file_path = "." + image_loc 

        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            print(f"File not found: {file_path}")

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="알수없는 이유로 문제가 발생하였습니다."
        )

async def delete_image_db(conn: Connection, image_id: int):
    """DB에서 이미지 레코드를 삭제한다.

    Args:
        conn (Connection): SQLAlchemy 비동기 DB 커넥션.
        image_id (int): 삭제할 이미지 레코드 PK.

    Raises:
        HTTPException 404: 해당 image_id 레코드가 없을 때.
        HTTPException 503: DB 쿼리 중 SQLAlchemy 오류 발생 시.
        HTTPException 500: 예기치 못한 오류 발생 시.
    """
    try:
        delete_query = text("DELETE FROM image_result WHERE id = :image_id")
        result = await conn.execute(delete_query, {"image_id": image_id})

        if result.rowcount == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"해당 이미지 id {image_id}는(은) 존재하지 않아 삭제할 수 없습니다.")
            
        await conn.commit()

    except SQLAlchemyError as e:
        await conn.rollback()
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="요청하신 서비스가 잠시 내부적으로 문제가 발생하였습니다.")

    except HTTPException:
        raise
    
    except Exception as e:
        await conn.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="알수없는 이유로 문제가 발생하였습니다.")

@celery_app.task(name="cleanup_anonymous_image")
def cleanup_anonymous_image(image_id: int, image_loc: str):
    """비회원 이미지 DB 레코드 및 물리 파일을 삭제하는 Celery 태스크.

    DB 삭제와 파일 삭제는 독립적으로 처리되어 한쪽 실패가 다른쪽에 영향을 주지 않는다.

    Args:
        image_id (int): 삭제할 이미지 레코드 PK.
        image_loc (str): 삭제할 이미지의 DB 기준 경로.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        async def _delete():
            async with celery_db_conn() as conn:
                success = True
                try:
                    await delete_image_db(conn, image_id)
                except Exception as e:
                    print(f"[Cleanup] 이미지 DB 삭제 실패 - image_id: {image_id}, error: {e}")
                    success=False
            try:
                await delete_image(image_loc)
            except Exception as e:
                print(f"[Cleanup] 이미지 파일 삭제 실패 - image_loc: {image_loc}, error: {e}")
                success=False

            if success: 
                print(f"[Cleanup] 비회원 이미지 삭제 프로세스 완료 - image_id: {image_id}, image_loc: {image_loc}")

        loop.run_until_complete(_delete())
    
    finally:
        loop.close()

@celery_app.task(name="cleanup_image_cam")
def cleanup_image_cam(cam_loc: str):
    """CAM 시각화 파일을 삭제하는 Celery 태스크.

    이미지/비디오 프레임 시각화 파일 모두 이 태스크로 정리한다.

    Args:
        cam_loc (str): 삭제할 CAM 시각화 파일의 DB 기준 경로.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        async def _delete():
            try:
                await delete_image(cam_loc)
                print(f"[Cleanup] 이미지 시각화 파일 삭제 완료 - cam_loc: {cam_loc}")
            except Exception as e:
                print(f"[Cleanup] 이미지 시각화 파일 삭제 실패 - cam_loc: {cam_loc}, error: {e}")
        
        loop.run_until_complete(_delete())
    
    finally:
        loop.close()    
    
async def register_image_result(conn: Connection, user_id: int | None, image_loc: str, 
                                version_type: str, model_type: str, domain_type: str):
    """PENDING 상태의 빈 이미지 레코드를 DB에 생성하고 image_id를 반환한다.

    Args:
        conn (Connection): SQLAlchemy 비동기 DB 커넥션.
        user_id (int | None): 로그인 사용자 ID. 비회원이면 None.
        image_loc (str): 업로드된 이미지의 DB 기준 경로.
        version_type (str): 모델 버전 타입.
        model_type (str): 추론 모드 ("fast" | "pro").
        domain_type (str): 탐지 도메인.

    Returns:
        int: 생성된 이미지 레코드의 PK (auto_increment).

    Raises:
        HTTPException 400: DB INSERT 실패 시. 업로드된 물리 파일도 함께 롤백 삭제된다.
    """
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
        await delete_image(image_loc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="요청데이터가 제대로 전달되지 않았습니다")
        
async def update_image_result(conn: Connection, image_id: int, analysis: dict,
                              result_msg: str, status: str): 
    """추론 완료 후 이미지 레코드에 분석 결과를 업데이트한다.

    Args:
        conn (Connection): SQLAlchemy 비동기 DB 커넥션.
        image_id (int): 업데이트할 이미지 레코드 PK.
        analysis (dict): 추론 결과값. prob, face_conf, face_ratio, face_brightness 포함.
        result_msg (str): 처리 결과 메세지.
        status (str): 추론 상태 ("success" | "warning" | "failed").

    Raises:
        SQLAlchemyError: DB 업데이트 실패 시 그대로 re-raise.
    """
    # status 종류: success, warning, failed
    # success: 이미지 추론 성공
    # warning: 이미지 추론 과정 PredictError 발생
    # failed: 이미지 추론 과정 알수 없는 오류 발생 
    
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

async def get_image_result(conn: Connection, image_id: int):
    """image_id로 이미지 분석 결과를 조회해 반환한다.

    Args:
        conn (Connection): SQLAlchemy 비동기 DB 커넥션.
        image_id (int): 조회할 이미지 레코드 PK.

    Returns:
        ImageData_indi: 이미지 분석 결과 스키마 객체.

    Raises:
        HTTPException 404: 해당 image_id 레코드가 없을 때.
        HTTPException 503: DB 쿼리 중 SQLAlchemy 오류 발생 시.
        HTTPException 500: 예기치 못한 오류 발생 시.
    """
    try:
        query = text("""
            SELECT id, user_id, image_loc, status, label, score, face_conf, face_ratio,
                   face_brightness, version_type, model_type, domain_type, result_msg, created_at
            FROM image_result
            WHERE id = :image_id
        """)
        result = await conn.execute(query, {"image_id": image_id})
        
        if result.rowcount == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                                detail=f"요청하신 이미지 분석 결과(ID: {image_id})를 찾을 수 없습니다. ID를 다시 확인해주세요.")
        row = result.fetchone()
        
        image_data = ImageResultData(
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

async def get_user_histories(conn: Connection, user_id: int):
    """사용자의 전체 이미지 분석 히스토리를 최신순으로 조회한다.

    Args:
        conn (Connection): SQLAlchemy 비동기 DB 커넥션.
        user_id (int): 조회할 사용자 PK.

    Returns:
        list[ImageData]: 이미지 히스토리 스키마 리스트. 결과 없으면 빈 리스트.

    Raises:
        HTTPException 503: DB 쿼리 중 SQLAlchemy 오류 발생 시.
        HTTPException 500: 예기치 못한 오류 발생 시.
    """
    try:
        # SQL에 맞춰 테이블명과 컬럼 변경
        query = ("""
            SELECT id, image_loc, label, version_type, model_type, domain_type, created_at
            FROM image_result
            WHERE user_id = :user_id
            ORDER BY created_at DESC;
        """)
        stmt = text(query)
        result = await conn.execute(stmt, {"user_id": user_id})
        
        user_histories = [ImageData(
            image_id = row.id,
            image_loc = row.image_loc,
            label = row.label,
            version_type = row.version_type,
            model_type = row.model_type,
            domain_type = row.domain_type,
            created_at = row.created_at
        ) for row in result]
        
        result.close()            
        return user_histories
    
    except SQLAlchemyError as e:
        print(f"히스토리 조회 실패: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="요청하신 서비스가 잠시 내부적으로 문제가 발생하였습니다.")

    except Exception as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="알수없는 이유로 문제가 발생하였습니다.")

async def get_user_history(conn: Connection, image_id: int):
    """image_id로 이미지 개별 상세 히스토리를 조회한다.

    Args:
        conn (Connection): SQLAlchemy 비동기 DB 커넥션.
        image_id (int): 조회할 이미지 레코드 PK.

    Returns:
        ImageDetailData | None: 상세 히스토리 스키마 객체. 레코드 없으면 None.

    Raises:
        HTTPException 503: DB 쿼리 중 SQLAlchemy 오류 발생 시.
        HTTPException 500: 예기치 못한 오류 발생 시.
    """
    try:
        query = """
            SELECT id, image_loc, status, label, score, face_conf, face_ratio, face_brightness, version_type, model_type, domain_type, result_msg, created_at
            FROM image_result
            WHERE id = :image_id;
        """
        stmt = text(query)
        result = await conn.execute(stmt, {"image_id": image_id})
        
        row = result.fetchone()
        if row is None:
            return None
        
        user_history = ImageDetailData(
            image_id = row.id,
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
            created_at = row.created_at
        )
        
        result.close()
        
        return user_history
    
    except SQLAlchemyError as e:
        print(f"히스토리 조회 실패: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="요청하신 서비스가 잠시 내부적으로 문제가 발생하였습니다.")

    except HTTPException:
        raise

    except Exception as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="알수없는 이유로 문제가 발생하였습니다.")

        