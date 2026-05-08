import os
import asyncio
import time
import aiofiles as aio
from dotenv import load_dotenv

from fastapi import UploadFile, status
from fastapi.exceptions import HTTPException
from sqlalchemy import text, Connection
from sqlalchemy.exc import SQLAlchemyError, DBAPIError
from schemas.video_schema import (
    VideoData, VideoDetailData, VideoFrameData, VideoMetaData )
from db.database import celery_db_conn
from celery_app import celery_app
load_dotenv()
UPLOAD_DIR = os.getenv("UPLOAD_DIR")

# 사용자 업로드 동영상 서버 내 저장 (회원/비회원 공통)
async def upload_video(user_email: str | None, videofile: UploadFile) -> str:
    try:
        # 1. 사용자별 하위 디렉토리 결정
        sub_dir = user_email if user_email else "anonymous"
        
        user_dir = os.path.join(UPLOAD_DIR, sub_dir)

        # 3. 디렉토리 존재 확인 및 생성
        if not os.path.exists(user_dir):
            try:
                os.makedirs(user_dir, exist_ok=True)
            except OSError as e:
                print(f"[Storage Error] 디렉토리 생성 실패: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="서버 내 저장 공간을 준비하지 못했습니다."
                )

        # 4. 파일명 중복 방지 (파일명_타임스탬프.확장자)
        filename_only, ext = os.path.splitext(videofile.filename)
        
        upload_filename = f"{filename_only}_{int(time.time())}{ext}"
        upload_video_loc = os.path.join(user_dir, upload_filename)

        # 5. 비동기 이미지 저장 (Chunk 단위 읽기)
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

        print(f"Upload Succeeded: {upload_video_loc}")

        # 6. DB 저장용 경로 반환
        return upload_video_loc[1:].replace("\\", "/")
    
    except HTTPException:
        raise
    
    except Exception as e:
        print(f"[Unknown Error] {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="비디오 업로드 과정에서 예상치 못한 오류가 발생했습니다.")

# 사용자 업로드 비디오 서버 내 삭제
async def delete_video(video_loc: str):
    try:
        file_path = "." + video_loc 

        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Video file removed: {file_path}")
        else:
            print(f"Video file not found: {file_path}")

    except Exception as e:
        print(f"[Delete Error] {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="비디오 파일 삭제 중 오류가 발생했습니다."
        )

# 사용자 전체 비디오 히스토리 조회
async def get_user_histories(conn: Connection, user_id: int):
    try:
        query = """
            SELECT id, user_id, video_loc, status, label, 
                   version_type, model_type, domain_type, result_msg, created_at
            FROM video_result
            WHERE user_id = :user_id
            ORDER BY created_at DESC;
        """
        stmt = text(query)
        bind_stmt = stmt.bindparams(user_id=user_id)
        result = await conn.execute(bind_stmt)

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
 

# 사용자 개별 비디오 히스토리 조회
async def get_user_history(conn: Connection, user_id: int, video_id: int):
    try:
        stmt = text("""
            SELECT id, user_id, video_loc, status, label, score, face_conf, face_ratio,
                    face_brightness, version_type, model_type, domain_type, result_msg, created_at
            FROM video_result
            WHERE id = :video_id AND user_id = :user_id
        """)
        result = await conn.execute(stmt, {"video_id": video_id, "user_id": user_id})
        
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
            face_conf = row.face_conf,
            face_ratio = row.face_ratio,
            face_brightness = row.face_brightness,
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
    
# 비회원 데이터 5분 후 자동 삭제 태스크
@celery_app.task(name="cleanup_anonymous_video")
def cleanup_anonymous_video(video_id: int, video_loc: str):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        async def _delete():
            async with celery_db_conn() as conn:
                await delete_video_db(conn, video_id)
            await delete_video(video_loc)
            print(f"[Cleanup] 비회원 비디오 삭제 완료 - video_id: {video_id}, video_loc: {video_loc}")
        loop.run_until_complete(_delete())
    except Exception as e:
        print(f"[Cleanup Error] 비회원 비디오 삭제 실패 - video_id: {video_id}, error: {e}")
    finally:
        loop.close()

# 비디오 DB 레코드 및 물리 파일 완전 삭제
async def delete_video_db(conn: Connection, video_id: int):
    try:
        delete_query = text("DELETE FROM video_result WHERE id = :video_id")
        result = await conn.execute(delete_query.bindparams(video_id=video_id))

        if result.rowcount == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"해당 비디오 id {id}는(은) 존재하지 않아 삭제할 수 없습니다.")
            
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
        print(f"Video DB Register Error: {e}")
        await conn.rollback()
        await delete_video(video_loc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="요청데이터가 제대로 전달되지 않았습니다")
        
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
        
        video_data = VideoDetailData(
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

# 비디오 메타 데이터 정보 저장하기
async def save_video_meta_result(conn: Connection, video_id: int, analysis: dict):
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
    
# 비디오 상세 결과 값 저장하기 
# 비디오 프레임 별 딥페이크 점수, 얼굴 신뢰도, 얼굴 비율, 얼굴 조도 반환
async def save_video_frame_results(conn: Connection, video_id: int, frame_results: list):
    try:
        query = """
            INSERT INTO video_frame_result
                (video_id, frame_index, frame_time, score, face_conf, face_ratio, face_brightness)
            VALUES
                (:video_id, :frame_index, :frame_time, :score, :face_conf, :face_ratio, :face_brightness)
        """
        rows = [
            {
                "video_id":        video_id,
                "frame_index":     frame["frame_index"],
                "frame_time":      frame["frame_time"],
                "score":           frame["score"],
                "face_conf":       frame["face_conf"],
                "face_ratio":      frame["face_ratio"],
                "face_brightness": frame["face_brightness"],
            }
            for frame in frame_results
        ]
        await conn.execute(text(query), rows)
        await conn.commit()

    except SQLAlchemyError as e:
        await conn.rollback()
        raise e
        
# 비디오 메타 데이터 정보 가져오기
async def get_video_meta_result(conn: Connection, video_id: int):         
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
 
# 비디오 상세 결과 값 가져오기
async def get_video_frame_results(conn: Connection, video_id: int):
    try:
        query = text("""
            SELECT frame_index, frame_time, score, face_conf, face_ratio, face_brightness
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
                face_conf=r.face_conf,
                face_ratio=r.face_ratio,
                face_brightness=r.face_brightness,
            )
            for r in result
        ]

        
        
    except SQLAlchemyError as e:
        print(f"[Frame Query Error] {e}")
        raise HTTPException(status_code=503, detail="데이터베이스 조회 중 문제가 발생했습니다.")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="알수없는 이유로 문제가 발생하였습니다.")