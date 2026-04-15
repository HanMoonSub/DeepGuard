import os
import time
import aiofiles as aio
from dotenv import load_dotenv

from fastapi import UploadFile, status
from fastapi.exceptions import HTTPException
from sqlalchemy import text, Connection
from sqlalchemy.exc import SQLAlchemyError, DBAPIError
from schemas.video_schema import VideoData

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
            score = row.score,
            face_conf = row.face_conf,
            face_ratio = row.face_ratio,
            face_brightness = row.face_brightness,
            version_type = row.version_type,
            model_type = row.model_type,
            domain_type = row.domain_type,
            result_msg = row.result_msg,
            created_at = row.created_at
        ) for row in result]
        
        result.close()
        return video_histories
    
    except SQLAlchemyError as e:
        print(f"비디오 목록 조회 실패: {e}")
        raise HTTPException(status_code=503, detail="데이터베이스 조회 중 문제가 발생했습니다.")

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
            
        video_history = VideoData(
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