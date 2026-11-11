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
    """Save an uploaded image on the server and return its path for the DB. Shared by members and guests.

    Args:
        user_email (str | None): Logged-in user's email, or None for guests.
        imagefile (UploadFile): FastAPI upload file object.

    Returns:
        str: DB path of the saved image (starts with '/', '\\' normalized).

    Raises:
        HTTPException 500: When writing the file fails or an unexpected error occurs.
    """
    try:
        # Pick the per-user subdirectory
        sub_dir = user_email if user_email else "anonymous"
        
        user_dir = os.path.join(UPLOAD_DIR, sub_dir)

        # Make sure the directory exists, creating it if needed
        os.makedirs(user_dir, exist_ok=True)
    
        # Avoid file name collisions (name_timestamp.ext)
        filename_only, ext = os.path.splitext(imagefile.filename)
        
        upload_filename = f"{filename_only}_{int(time.time())}{ext}"
        upload_image_loc = os.path.join(user_dir, upload_filename)

        # Save the image asynchronously (reading in chunks)
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

        # 6. Return the path for the DB
        return upload_image_loc[1:].replace("\\", "/")
    
    except HTTPException:
        raise
    
    except Exception as e:
        print(f"[Unknown Error] {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="이미지 업로드 과정에서 예상치 못한 오류가 발생했습니다.")

async def upload_image_cam(user_email: str, image_id: int, image: np.ndarray) -> str:
    """Save a CAM visualization image on the server and return its path for the DB. Members only.

    Args:
        user_email (str): Requesting user's email (used to separate storage paths).
        image_id (int): Image record PK (used in the file name).
        image (np.ndarray): Visualization image array to save (RGB).

    Returns:
        str: DB path of the saved CAM image (starts with '/').

    Raises:
        Exception: Re-raised as is when writing the file fails.
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
    """Save a video frame CAM visualization image on the server and return its path for the DB. Members only.

    Args:
        user_email (str): Requesting user's email (used to separate storage paths).
        video_id (int): Video record PK (used in the file name).
        frame_time (float): Timestamp of the visualized frame (in seconds, used in the file name).
        image (np.ndarray): Visualization image array to save (RGB).

    Returns:
        str: DB path of the saved CAM image (starts with '/').

    Raises:
        Exception: Re-raised as is when writing the file fails.
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
    """Delete an image file from the server.

    If the file does not exist, only a warning is logged and the function returns normally.

    Args:
        image_loc (str): DB path of the image to delete (starts with '/').

    Raises:
        HTTPException 500: When an unexpected error occurs while deleting the file.
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
    """Delete an image record from the DB.

    Args:
        conn (Connection): SQLAlchemy async DB connection.
        image_id (int): PK of the image record to delete.

    Raises:
        HTTPException 404: When no record exists for image_id.
        HTTPException 503: When a SQLAlchemy error occurs during the DB query.
        HTTPException 500: When an unexpected error occurs.
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
    """Celery task that deletes a guest's image DB record and file.

    The DB deletion and file deletion are handled independently, so one failing does not affect the other.

    Args:
        image_id (int): PK of the image record to delete.
        image_loc (str): DB path of the image to delete.
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
    """Celery task that deletes a CAM visualization file.

    Cleans up both image and video frame visualization files.

    Args:
        cam_loc (str): DB path of the CAM visualization file to delete.
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
    """Create an empty image record in the PENDING state and return its image_id.

    Args:
        conn (Connection): SQLAlchemy async DB connection.
        user_id (int | None): Logged-in user ID, or None for guests.
        image_loc (str): DB path of the uploaded image.
        version_type (str): Model version type.
        model_type (str): Inference mode ("fast" | "pro").
        domain_type (str): Detection domain.

    Returns:
        int: PK of the created image record (auto_increment).

    Raises:
        HTTPException 400: When the DB INSERT fails. The uploaded file is also deleted as a rollback.
    """
    try:
        query = """
            INSERT INTO image_result (
                user_id, image_loc, status, label, score, version_type, model_type, domain_type, result_msg
            )
            VALUES (
                :user_id, :image_loc, 'PENDING', 'UNKNOWN', -1.0, :version_type, :model_type, :domain_type, '분석 대기 중...')
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
        
        # Get the auto_increment ID just created by MySQL
        return result.lastrowid
        
    except SQLAlchemyError as e:
        print(f"DB Insert Error: {e}")
        await conn.rollback()
        await delete_image(image_loc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="요청데이터가 제대로 전달되지 않았습니다")
        
async def update_image_result(conn: Connection, image_id: int, analysis: dict,
                              result_msg: str, status: str): 
    """Update the image record with the analysis result after inference finishes.

    Args:
        conn (Connection): SQLAlchemy async DB connection.
        image_id (int): PK of the image record to update.
        analysis (dict): Inference result, including prob.
        result_msg (str): Result message.
        status (str): Inference status ("success" | "warning" | "failed").

    Raises:
        SQLAlchemyError: Re-raised as is when the DB update fails.
    """
    # status values: success, warning, failed
    # success: image inference succeeded
    # warning: a PredictError was raised during image inference
    # failed: an unknown error occurred during image inference
    
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
                result_msg = :result_msg
            WHERE id = :image_id
        """
        
        db_status = status.upper()
        
        stmt = text(query)
        await conn.execute(stmt, {
            "status": db_status,
            "label": label,
            "score": analysis["prob"],
            "result_msg": result_msg,
            "image_id": image_id
        })
        await conn.commit()
        
    except SQLAlchemyError as e:
        await conn.rollback()
        raise e

async def get_image_result(conn: Connection, image_id: int):
    """Fetch and return the image analysis result for image_id.

    Args:
        conn (Connection): SQLAlchemy async DB connection.
        image_id (int): PK of the image record to fetch.

    Returns:
        ImageData_indi: Image analysis result schema object.

    Raises:
        HTTPException 404: When no record exists for image_id.
        HTTPException 503: When a SQLAlchemy error occurs during the DB query.
        HTTPException 500: When an unexpected error occurs.
    """
    try:
        query = text("""
            SELECT id, user_id, image_loc, status, label, score, version_type, model_type, domain_type, result_msg, created_at
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
    """Fetch the user's full image analysis history, newest first.

    Args:
        conn (Connection): SQLAlchemy async DB connection.
        user_id (int): PK of the user to look up.

    Returns:
        list[ImageData]: List of image history schemas. Empty list if there are no results.

    Raises:
        HTTPException 503: When a SQLAlchemy error occurs during the DB query.
        HTTPException 500: When an unexpected error occurs.
    """
    try:
        # Table and column names follow the SQL schema
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
    """Fetch the detailed history of a single image by image_id.

    Args:
        conn (Connection): SQLAlchemy async DB connection.
        image_id (int): PK of the image record to fetch.

    Returns:
        ImageDetailData | None: Detailed history schema object, or None if there is no record.

    Raises:
        HTTPException 503: When a SQLAlchemy error occurs during the DB query.
        HTTPException 500: When an unexpected error occurs.
    """
    try:
        query = """
            SELECT id, image_loc, status, label, score, version_type, model_type, domain_type, result_msg, created_at
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

        