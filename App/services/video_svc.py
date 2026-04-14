import os
import time
import aiofiles as aio
from dotenv import load_dotenv

from fastapi import UploadFile, status
from fastapi.exceptions import HTTPException
from sqlalchemy import text, Connection
from sqlalchemy.exc import SQLAlchemyError, DBAPIError

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

