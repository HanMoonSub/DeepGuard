import os
import time
import aiofiles as aio
from dotenv import load_dotenv

from fastapi import UploadFile, status
from fastapi.exceptions import HTTPException
from sqlalchemy import text, Connection
from sqlalchemy.exc import SQLAlchemyError, DBAPIError
from schemas.image_schema import UserHistory

load_dotenv()
# .env에서 UPLOAD_DIR을 가져오되, 경로가 설정되지 않았을 경우를 대비해 'static' 기본값 설정
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "static")

# [1] 사용자 업로드 이미지 서버 내 저장 (회원/비회원 공통)
async def upload_image(user_email: str | None, imagefile: UploadFile) -> str:
    try:
        # 1. 사용자별 하위 디렉토리 결정
        sub_dir = user_email if user_email else "anonymous"
        
        # 2.경로 중복 방지 로직
        # UPLOAD_DIR에 이미 'uploads'가 포함되어 있는지 확인하여 중복 생성을 막습니다.
        if "uploads" in UPLOAD_DIR:
            user_dir = os.path.join(UPLOAD_DIR, sub_dir)
        else:
            user_dir = os.path.join(UPLOAD_DIR, "uploads", sub_dir)

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
        filename_only, ext = os.path.splitext(imagefile.filename)
        if not ext: ext = ".png"
        
        upload_filename = f"{filename_only}_{int(time.time())}{ext}"
        upload_image_loc = os.path.join(user_dir, upload_filename)

        # 5. 비동기 이미지 저장 (Chunk 단위 읽기)
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

        print(f"Upload Succeeded: {upload_image_loc}")

        # 6. DB 저장용 경로 반환 (역슬래시를 슬래시로 통일)
        return upload_image_loc.replace("\\", "/")

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Unknown Error] {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="이미지 업로드 과정에서 예상치 못한 오류가 발생했습니다.")


# [2] 이미지 메타데이터 DB 저장
async def register_user_image(conn: Connection, user_id: int | None, image_loc: str):
    try:
        query = text("""
            INSERT INTO user_image (user_id, image_loc)
            VALUES (:user_id, :image_loc)
        """)
        
        await conn.execute(query, {"user_id": user_id, "image_loc": image_loc})
        await conn.commit()
        
    except DBAPIError as e:
        print(f"[Database Connection Error] {e}")
        await conn.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="데이터베이스 연결이 원활하지 않습니다."
        )
    except SQLAlchemyError as e:
        print(f"[SQL Execution Error] {e}")
        await conn.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="이미지 정보를 저장하는 형식이 올바르지 않습니다.")


# [3] 사용자 전체 히스토리 조회
async def get_user_histories(conn: Connection, user_id: int):
    try:
        query = text("""
            SELECT image_loc, created_at
            FROM user_image
            WHERE user_id = :user_id
            ORDER BY created_at DESC;
        """)
        result = await conn.execute(query, {"user_id": user_id})
        
        return [
            UserHistory(image_loc=row.image_loc, created_at=row.created_at) 
            for row in result
        ]
    except SQLAlchemyError as e:
        print(f"[SQL Error] {e}")
        raise HTTPException(status_code=503, detail="데이터베이스 조회 중 문제가 발생했습니다.")

# [4] 사용자 개별 히스토리 조회
async def get_user_individual_history(conn: Connection, user_id: int, image_id: int):
    try:
        query = text("""
            SELECT image_loc, created_at
            FROM user_image
            WHERE id = :image_id AND user_id = :user_id
        """)
        result = await conn.execute(query, {"image_id": image_id, "user_id": user_id})
        row = result.fetchone()
        
        if row:
            return UserHistory(image_loc=row.image_loc, created_at=row.created_at)
        return None
    except SQLAlchemyError as e:
        print(f"[SQL Error] {e}")
        raise HTTPException(status_code=503, detail="데이터베이스 상세 조회 중 문제가 발생했습니다.")