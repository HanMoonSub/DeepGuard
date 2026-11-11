import os
import time
from dotenv import load_dotenv
import asyncio
import aiofiles as aio
from fastapi import UploadFile, status
from fastapi.exceptions import HTTPException
from sqlalchemy import text, Connection
from sqlalchemy.exc import SQLAlchemyError
from schemas.video_schema import (
    VideoData, VideoDetailData, VideoFrameData, VideoMetaData)
from db.database import celery_db_conn
from celery_app import celery_app

load_dotenv()
UPLOAD_DIR = os.getenv("UPLOAD_DIR")


async def upload_video(user_email: str | None, videofile: UploadFile) -> str:
    """Save an uploaded video on the server and return its path for the DB. Shared by members and guests.

    Args:
        user_email (str | None): Logged-in user's email, or None for guests.
        videofile (UploadFile): FastAPI upload file object.

    Returns:
        str: DB path of the saved video (starts with '/', '\\' normalized).

    Raises:
        HTTPException 500: When creating the directory, writing the file, or an unexpected error fails.
    """
    try:
        # 1. Pick the per-user subdirectory
        sub_dir = user_email if user_email else "anonymous"
        
        user_dir = os.path.join(UPLOAD_DIR, sub_dir)

        # 3. Make sure the directory exists, creating it if needed
        if not os.path.exists(user_dir):
            try:
                os.makedirs(user_dir, exist_ok=True)
            except OSError as e:
                print(f"[Storage Error] 디렉토리 생성 실패: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="서버 내 저장 공간을 준비하지 못했습니다."
                )

        # 4. Avoid file name collisions (name_timestamp.ext)
        filename_only, ext = os.path.splitext(videofile.filename)
        
        upload_filename = f"{filename_only}_{int(time.time())}{ext}"
        upload_video_loc = os.path.join(user_dir, upload_filename)

        # 5. Save the video asynchronously (reading in chunks)
        try:
            async with aio.open(upload_video_loc, "wb") as outfile:
                while content := await videofile.read(1024 * 1024):
                    await outfile.write(content)
        except Exception as e:
            print(f"[File Error] 파일 쓰기 실패: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="비디오 파일을 저장하는 중 오류가 발생했습니다."
            )

        # 6. Return the path for the DB
        return upload_video_loc[1:].replace("\\", "/")
    
    except HTTPException:
        raise
    
    except Exception as e:
        print(f"[Unknown Error] {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="비디오 업로드 과정에서 예상치 못한 오류가 발생했습니다.")


async def delete_video(video_loc: str) -> None:
    """Delete a video file from the server.

    If the file does not exist, only a warning is logged and the function returns normally.

    Args:
        video_loc (str): DB path of the video to delete (starts with '/').

    Raises:
        HTTPException 500: When an error occurs while deleting the file.
    """
    try:
        file_path = "." + video_loc 

        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            print(f"Video file not found: {file_path}")

    except Exception as e:
        print(f"[Delete Error] {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="비디오 파일 삭제 중 오류가 발생했습니다."
        )


async def get_user_histories(conn: Connection, user_id: int) -> list[VideoData]:
    """Fetch the user's full video analysis history, newest first.

    Args:
        conn (Connection): SQLAlchemy async DB connection.
        user_id (int): PK of the user to look up.

    Returns:
        list[VideoData]: List of video history schemas. Empty list if there are no results.

    Raises:
        HTTPException 503: When a SQLAlchemy error occurs during the DB query.
        HTTPException 500: When an unexpected error occurs.
    """
    try:
        query = """
            SELECT id, user_id, video_loc, status, label, 
                   version_type, model_type, domain_type, result_msg, created_at
            FROM video_result
            WHERE user_id = :user_id
            ORDER BY created_at DESC;
        """
        stmt = text(query)
        result = await conn.execute(stmt, {"user_id": user_id})

        video_histories = [VideoData(
            id = row.id,
            user_id = row.user_id,
            video_loc = row.video_loc,
            status = row.status,
            label = row.label,
            version_type = row.version_type,
            model_type = row.model_type,
            domain_type = row.domain_type,
            created_at = row.created_at
        ) for row in result]
        
        result.close()
        return video_histories
    
    except SQLAlchemyError as e:
        print(f"비디오 목록 조회 실패: {e}")
        raise HTTPException(status_code=503, detail="데이터베이스 조회 중 문제가 발생했습니다.")
    
    except Exception as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="알수없는 이유로 문제가 발생하였습니다.")


async def get_user_history(conn: Connection, user_id: int, video_id: int) -> VideoDetailData | None:
    """Fetch the detailed history of a single video by video_id and user_id.

    Also filters by user_id so only the user's own data is returned.

    Args:
        conn (Connection): SQLAlchemy async DB connection.
        user_id (int): PK of the requesting user.
        video_id (int): PK of the video record to fetch.

    Returns:
        VideoDetailData | None: Detailed history schema object, or None if there is no record.

    Raises:
        HTTPException 503: When a SQLAlchemy error occurs during the DB query.
        HTTPException 500: When an unexpected error occurs.
    """
    try:
        # With a user_id, return only the user's own record; with None, look up by video_id only (guest detail allowed)
        if user_id is not None:
            stmt = text("""
                SELECT id, user_id, video_loc, status, label, score, version_type, model_type, domain_type, result_msg, created_at
                FROM video_result
                WHERE id = :video_id AND user_id = :user_id
            """)
            params = {"video_id": video_id, "user_id": user_id}
        else:
            stmt = text("""
                SELECT id, user_id, video_loc, status, label, score, version_type, model_type, domain_type, result_msg, created_at
                FROM video_result
                WHERE id = :video_id
            """)
            params = {"video_id": video_id}
        result = await conn.execute(stmt, params)
        row = result.fetchone()
        if row is None:
            return None
            
        video_history = VideoDetailData(
            id = row.id,
            user_id = row.user_id,
            video_loc = row.video_loc,
            status = row.status,
            label = row.label,
            score = row.score,
            version_type = row.version_type,
            model_type = row.model_type,
            domain_type = row.domain_type,
            result_msg = row.result_msg,
            created_at = row.created_at
        )
        
        return video_history
    
    except SQLAlchemyError as e:
        print(f"비디오 상세 조회 실패: {e}")
        raise HTTPException(status_code=503, detail="데이터베이스 조회 중 문제가 발생했습니다.")
    
    except Exception as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="알수없는 이유로 문제가 발생하였습니다.")


async def register_video_result(conn: Connection, user_id: int | None, video_loc: str,
                                version_type: str, model_type: str, domain_type: str) -> int:
    """Create an empty video record in the PENDING state and return its video_id.

    Args:
        conn (Connection): SQLAlchemy async DB connection.
        user_id (int | None): Logged-in user ID, or None for guests.
        video_loc (str): DB path of the uploaded video.
        version_type (str): Model version type.
        model_type (str): Inference mode ("fast" | "pro").
        domain_type (str): Detection domain.

    Returns:
        int: PK of the created video record (auto_increment).

    Raises:
        HTTPException 400: When the DB INSERT fails. The uploaded file is also deleted as a rollback.
    """
    try:
        query = """
            INSERT INTO video_result (
                user_id, video_loc, status, label, score, version_type, model_type, domain_type, result_msg
            )
            VALUES (
                :user_id, :video_loc, 'PENDING', 'UNKNOWN', -1.0, :version_type, :model_type, :domain_type, '분석 대기 중...'
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
        
        # Get the auto_increment ID just created by MySQL
        return result.lastrowid
        
    except SQLAlchemyError as e:
        print(f"Video DB Register Error: {e}")
        await conn.rollback()
        await delete_video(video_loc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="요청데이터가 제대로 전달되지 않았습니다")


async def update_video_result(conn: Connection, video_id: int, analysis: dict,
                              result_msg: str, status: str) -> None:
    """Update the video record with the analysis result after inference finishes.

    Args:
        conn (Connection): SQLAlchemy async DB connection.
        video_id (int): PK of the video record to update.
        analysis (dict): Inference result, including prob.
        result_msg (str): Result message.
        status (str): Inference status ("success" | "warning" | "failed").

    Raises:
        SQLAlchemyError: Re-raised as is when the DB update fails.
    """
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
                result_msg = :result_msg
            WHERE id = :video_id
        """
        
        db_status = status.upper()
  
        stmt = text(query)
        await conn.execute(stmt, {
            "status": db_status,
            "label": label,
            "score": analysis["prob"],
            "result_msg": result_msg,
            "video_id": video_id
        })
        await conn.commit()
        
    except SQLAlchemyError as e:
        await conn.rollback()
        raise e


async def get_video_result(conn: Connection, video_id: int) -> VideoDetailData:
    """Fetch and return the video analysis result for video_id.

    Args:
        conn (Connection): SQLAlchemy async DB connection.
        video_id (int): PK of the video record to fetch.

    Returns:
        VideoDetailData: Video analysis result schema object.

    Raises:
        HTTPException 404: When no record exists for video_id.
        HTTPException 503: When a SQLAlchemy error occurs during the DB query.
        HTTPException 500: When an unexpected error occurs.
    """
    try:
        query = text("SELECT * FROM video_result WHERE id = :video_id")
        result = await conn.execute(query, {"video_id": video_id})
        
        if result.rowcount == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                                detail=f"요청하신 비디오 분석 결과(ID: {video_id})를 찾을 수 없습니다. ID를 다시 확인해주세요.")
        row = result.fetchone()
        
        video_data = VideoDetailData(
            id = row.id,
            user_id = row.user_id,
            video_loc = row.video_loc,
            status = row.status,
            label = row.label,
            score = row.score,
            version_type = row.version_type,
            model_type = row.model_type,
            domain_type = row.domain_type,
            result_msg = row.result_msg,
            created_at = row.created_at
        )
        
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


async def save_video_meta_result(conn: Connection, video_id: int, analysis: dict) -> None:
    """Save the frame summary statistics of one video to the video_meta_result table.

    Args:
        conn (Connection): SQLAlchemy async DB connection.
        video_id (int): Video record PK.
        analysis (dict): Inference result dict. Must contain the keys fps, total_frames, num_sampled,
            num_extracted and num_detected.

    Raises:
        SQLAlchemyError: Re-raised as is when the DB INSERT fails.
    """
    try:
        query = """
            INSERT INTO video_meta_result (video_id, fps, total_frames, num_sampled, num_extracted, num_detected)
            VALUES (:video_id, :fps, :total_frames, :num_sampled, :num_extracted, :num_detected)
        """
        await conn.execute(text(query), {
            "video_id":     video_id,
            "fps":          analysis["fps"],
            "total_frames": analysis["total_frames"],
            "num_sampled":  analysis["num_sampled"],
            "num_extracted":analysis["num_extracted"],
            "num_detected": analysis["num_detected"],
        })
        await conn.commit()

    except SQLAlchemyError as e:
        await conn.rollback()
        raise e


async def save_video_frame_result(conn: Connection, video_id: int, frame_results: list) -> None:
    """Bulk-save the analysis results of every frame in a video to the video_frame_result table.

    Args:
        conn (Connection): SQLAlchemy async DB connection.
        video_id (int): Video record PK.
        frame_results (list): List of per-frame analysis result dicts.
            Each item must contain the keys frame_index, frame_time and score.

    Raises:
        SQLAlchemyError: Re-raised as is when the DB INSERT fails.
    """
    try:
        query = """
            INSERT INTO video_frame_result
                (video_id, frame_index, frame_time, score)
            VALUES
                (:video_id, :frame_index, :frame_time, :score)
        """
        rows = [
            {
                "video_id":        video_id,
                "frame_index":     frame["frame_index"],
                "frame_time":      frame["frame_time"],
                "score":           frame["score"],
            }
            for frame in frame_results
        ]
        await conn.execute(text(query), rows)
        await conn.commit()

    except SQLAlchemyError as e:
        await conn.rollback()
        raise e


async def delete_video_meta_result(conn: Connection, video_id: int) -> None:
    """Delete the video's meta data from the video_meta_result table.

    Args:
        conn (Connection): SQLAlchemy async DB connection.
        video_id (int): PK of the video record to delete.

    Raises:
        HTTPException 404: When no record exists for video_id.
        HTTPException 503: When a SQLAlchemy error occurs during the DB query.
        HTTPException 500: When an unexpected error occurs.
    """
    try:
        delete_query = text("""
            DELETE FROM video_meta_result
            WHERE video_id = :video_id
        """)
        result = await conn.execute(delete_query, {"video_id": video_id})
        if result.rowcount == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"해당 비디오 id {video_id}는(은) 존재하지 않아 삭제할 수 없습니다.")

        await conn.commit()

    except SQLAlchemyError as e:
        print(e)
        await conn.rollback()
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="요청하신 서비스가 잠시 내부적으로 문제가 발생하였습니다.")

    except HTTPException:
        raise
    
    except Exception as e:
        print(e)
        await conn.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="알수없는 이유로 문제가 발생하였습니다.")


async def delete_video_frame_result(conn: Connection, video_id: int) -> None:
    """Delete the video's frame results from the video_frame_result table.

    Args:
        conn (Connection): SQLAlchemy async DB connection.
        video_id (int): PK of the video record to delete.

    Raises:
        HTTPException 404: When no record exists for video_id.
        HTTPException 503: When a SQLAlchemy error occurs during the DB query.
        HTTPException 500: When an unexpected error occurs.
    """
    try:
        delete_query = text("""
            DELETE FROM video_frame_result
            WHERE video_id = :video_id
        """)
        result = await conn.execute(delete_query, {"video_id": video_id})
        if result.rowcount == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"해당 비디오 id {video_id}는(은) 존재하지 않아 삭제할 수 없습니다.")
        
        await conn.commit()

    except SQLAlchemyError as e:
        print(e)
        await conn.rollback()
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="요청하신 서비스가 잠시 내부적으로 문제가 발생하였습니다.")

    except HTTPException:
        raise
    
    except Exception as e:
        print(e)
        await conn.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="알수없는 이유로 문제가 발생하였습니다.")


async def get_video_meta_result(conn: Connection, video_id: int) -> VideoMetaData:
    """Fetch the video meta data by video_id.

    Returns fps, total_frames, num_sampled, num_extracted and num_detected.

    Args:
        conn (Connection): SQLAlchemy async DB connection.
        video_id (int): PK of the video record to fetch.

    Returns:
        VideoMetaData: Video meta data schema object.

    Raises:
        HTTPException 404: When no meta record exists for video_id.
        HTTPException 503: When a SQLAlchemy error occurs during the DB query.
        HTTPException 500: When an unexpected error occurs.
    """
    try:
        query = text("""
            SELECT fps, total_frames, num_sampled, num_extracted, num_detected
            FROM video_meta_result
            WHERE video_id = :video_id
        """)
        result = await conn.execute(query, {"video_id": video_id})
        row = result.fetchone()

        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"요청하신 비디오 메타 정보(ID: {video_id})를 찾을 수 없습니다. ID를 다시 확인해주세요.")

        return VideoMetaData(
            fps=row.fps,
            total_frames=row.total_frames,
            num_sampled=row.num_sampled,
            num_extracted=row.num_extracted,
            num_detected=row.num_detected,
        )

    except SQLAlchemyError as e:
        print(f"[Meta Query Error] {e}")
        raise HTTPException(status_code=503, detail="데이터베이스 조회 중 문제가 발생했습니다.")
    except HTTPException:
        raise
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="알수없는 이유로 문제가 발생하였습니다.")


async def get_video_frame_result(conn: Connection, video_id: int) -> list[VideoFrameData]:
    """Fetch every frame analysis result for video_id, ordered by frame_index ascending.

    Used for visualizations such as the per-frame score chart and suspicious segment markers.

    Args:
        conn (Connection): SQLAlchemy async DB connection.
        video_id (int): PK of the video record to fetch.

    Returns:
        list[VideoFrameData]: List of per-frame analysis result schemas.

    Raises:
        HTTPException 503: When a SQLAlchemy error occurs during the DB query.
        HTTPException 500: When an unexpected error occurs.
    """
    try:
        query = text("""
            SELECT frame_index, frame_time, score
            FROM video_frame_result
            WHERE video_id = :video_id
            ORDER BY frame_index ASC
        """)
        result = await conn.execute(query, {"video_id": video_id})

        return [
            VideoFrameData(
                frame_index=r.frame_index,
                frame_time=r.frame_time,
                score=r.score,
            )
            for r in result
        ]

    except SQLAlchemyError as e:
        print(f"[Frame Query Error] {e}")
        raise HTTPException(status_code=503, detail="데이터베이스 조회 중 문제가 발생했습니다.")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="알수없는 이유로 문제가 발생하였습니다.")


async def get_video_frame_by_index(conn: Connection, video_id: int, frame_index: int) -> float:
    """Fetch the frame_time of a specific frame by video_id and frame_index.

    Args:
        conn (Connection): SQLAlchemy async DB connection.
        video_id (int): Video record PK.
        frame_index (int): Frame index to look up.

    Returns:
        float: Timestamp of the frame (in seconds).

    Raises:
        HTTPException 404: When the frame_index does not exist.
        HTTPException 503: When a SQLAlchemy error occurs during the DB query.
        HTTPException 500: When an unexpected error occurs.
    """
    try:
        query = text("""
            SELECT frame_time
            FROM video_frame_result
            WHERE video_id = :video_id AND frame_index = :frame_index
        """)
        
        result = await conn.execute(query, {"video_id": video_id, "frame_index": frame_index})        
        if result.rowcount == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                                detail=f"해당 ID: {video_id} Video 내 Frame Index {frame_index}에 해당하는 프레임을 찾을 수 없습니다")
        
        row = result.fetchone()
        
        return row.frame_time
        
    except SQLAlchemyError as e:
        print(f"[Frame Query Error] {e}")
        raise HTTPException(status_code=503, detail="데이터베이스 조회 중 문제가 발생했습니다.")
    except HTTPException:
        raise
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="알수없는 이유로 문제가 발생하였습니다.")


@celery_app.task(name="cleanup_anonymous_video")
def cleanup_anonymous_video(video_id: int, video_loc: str, is_success: bool) -> None:
    """Celery task that deletes a guest's video DB record and file.

    On successful inference, the meta/frame results are deleted as well.
    Each deletion step is handled independently, so one failing does not affect the others.

    Args:
        video_id (int): PK of the video record to delete.
        video_loc (str): DB path of the video to delete.
        is_success (bool): Whether inference succeeded. If True, the meta/frame results are deleted too.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        async def _delete():
            success = True
            async with celery_db_conn() as conn:
                if is_success:
                    try:
                        await delete_video_meta_result(conn, video_id)
                    except Exception as e:
                        print(f"[Cleanup] video meta 삭제 실패 - video_id: {video_id}, error: {e}")
                        success=False

                    try:  
                        await delete_video_frame_result(conn, video_id)   
                    except Exception as e:
                        print(f"[Cleanup] video frame 삭제 실패 - video_id: {video_id}, error: {e}")
                        success=False
                try:
                    await delete_video_db(conn, video_id)
                except Exception as e:
                    print(f"[Cleanup] video_result 삭제 실패 - video_id: {video_id}, error:{e}")
                    success=False
            try:
                await delete_video(video_loc)
            except Exception as e:
                print(f"[Cleanup] 비디오 파일 삭제 실패 - video_loc: {video_loc}, error: {e}")
                success=False

            if success:        
                print(f"[Cleanup] 비회원 비디오 삭제 완료 - video_id: {video_id}, video_loc: {video_loc}")

        loop.run_until_complete(_delete())
    finally:
        loop.close()


async def delete_video_db(conn: Connection, video_id: int) -> None:
    """Delete a video record from the DB.

    Args:
        conn (Connection): SQLAlchemy async DB connection.
        video_id (int): PK of the video record to delete.

    Raises:
        HTTPException 404: When no record exists for video_id.
        HTTPException 503: When a SQLAlchemy error occurs during the DB query.
        HTTPException 500: When an unexpected error occurs.
    """
    try:
        delete_query = text("DELETE FROM video_result WHERE id = :video_id")
        result = await conn.execute(delete_query, {"video_id": video_id})

        if result.rowcount == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"해당 비디오 id {video_id}는(은) 존재하지 않아 삭제할 수 없습니다.")
            
        await conn.commit()

    except SQLAlchemyError as e:
        print(e)
        await conn.rollback()
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="요청하신 서비스가 잠시 내부적으로 문제가 발생하였습니다.")

    except HTTPException:
        raise
    
    except Exception as e:
        print(e)
        await conn.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="알수없는 이유로 문제가 발생하였습니다.")