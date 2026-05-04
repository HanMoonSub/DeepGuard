import os
import asyncio
import time
import aiofiles as aio
from dotenv import load_dotenv

from fastapi import UploadFile, status
from fastapi.exceptions import HTTPException
from sqlalchemy import text, Connection
from sqlalchemy.exc import SQLAlchemyError, DBAPIError
from schemas.video_schema import VideoData
from schemas.video_schema import VideoData_indi
from db.database import celery_db_conn

load_dotenv()
UPLOAD_DIR = os.getenv("UPLOAD_DIR")

# [1] 사용자 업로드 동영상 서버 내 저장 (회원/비회원 공통)
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

# [2] 사용자 업로드 비디오 서버 내 삭제
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

# [3] 사용자 전체 비디오 히스토리 조회
async def get_user_histories(conn: Connection, user_id: int):
    try:
        query = """
            SELECT id, user_id, video_loc, status, label, score, face_conf, face_ratio, face_brightness, version_type, model_type, domain_type, result_msg, created_at
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
 

# [4] 사용자 개별 비디오 히스토리 조회
async def get_user_history(conn: Connection, user_id: int, video_id: int):
    try:
        query = """
            SELECT id, user_id, video_loc, status, label, score, face_conf, face_ratio, face_brightness, version_type, model_type, domain_type, result_msg, created_at
            FROM video_result
            WHERE id = :video_id AND user_id = :user_id;
        """
        stmt = text(query)
        # 보안을 위해 video_id뿐만 아니라 user_id도 바인딩하여 본인 확인
        bind_stmt = stmt.bindparams(video_id=video_id, user_id=user_id)
        result = await conn.execute(bind_stmt)
        
        row = result.fetchone()
        if row is None:
            return None
            
        video_history = VideoData_indi(
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
        
        result.close()
        return video_history
    
    except SQLAlchemyError as e:
        print(f"비디오 상세 조회 실패: {e}")
        raise HTTPException(status_code=503, detail="데이터베이스 조회 중 문제가 발생했습니다.")
    
    except Exception as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="알수없는 이유로 문제가 발생하였습니다.")
    
# [5] 비회원 데이터 5분 후 자동 삭제 태스크
async def delete_video_delayed_db(video_id: int):
    await asyncio.sleep(300)  # 300s (5m) 대기
    try:
        async with celery_db_conn() as conn:
            # video_svc의 통합 삭제 함수 호출
            await delete_video_db(conn, video_id)
            print(f"Non-user auto-cleanup finished: {video_id}")
    except Exception as e:
        print(f"[Cleanup Error] Video ID {video_id}: {e}") 

# [6] 비디오 DB 레코드 및 물리 파일 완전 삭제
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


# [6] 빈 비디오 DB 생성 이후, video_id 반환(접수 완료)
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
        await delete_video(video_loc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="요청데이터가 제대로 전달되지 않았습니다")
        
# [7] 비디오 메타데이터 + 추론 결과값 DB에 저장
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