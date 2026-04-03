import os
from dotenv import load_dotenv
import time
import aiofiles as aio

from fastapi import UploadFile, status
from fastapi.exceptions import HTTPException
from sqlalchemy import text, Connection
from sqlalchemy.exc import SQLAlchemyError
from schemas.image_schema import UserHistory

load_dotenv()
UPLOAD_DIR = os.getenv("UPLOAD_DIR")

# 사용자 업로드 이미지 서버 내 저장(회원/비회원)
async def upload_image(user_email: str | None, imagefile: UploadFile):
    try:
        # 회원 저장 경로 (user_email 이용)
        # user_name의 경우 중복될 가능성이 존재한다.
        if user_email:
            user_dir = f"{UPLOAD_DIR}/{user_email}/"
        # 비회원 저장 경로 (anonymous 고정)
        else:
            user_dir = f"{UPLOAD_DIR}/anonymous/"
        # 해당 회원 저장 폴더 없을 시, 새로 생성한다.
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)

        # sample_img.png => sample_img (filename_only), png (ext)
        filename_only, ext = os.path.splitext(imagefile.filename)
        # 회원이 동일한 이름의 사진을 업로드 해도 time을 통해 새로운 이름으로 저장 
        upload_filename = f"{filename_only}_{(int)(time.time())}{ext}"
        # 실제 서버 내 저장 경로
        upload_image_loc = user_dir + upload_filename

        # 비동기 이미지 저장 
        async with aio.open(upload_image_loc, "wb") as outfile:
            while content := await imagefile.read(1024):
                await outfile.write(content)
        print("upload succeeded:", upload_image_loc)

        return upload_image_loc[1:]
    
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="이미지 파일이 제대로 업로드되지 않았습니다. ")
      
# 이미지 메타데이터 DB 저장 
async def register_user_image(conn: Connection, user_id: int | None, image_loc: str):
    try:
        query = f"""
        INSERT INTO user_image(user_id, image_loc)
        values (:user_id, :image_loc)
        """
        
        # :user_id와 :image_loc는 Bind Parameter
        stmt = text(query)
        bind_stmt = stmt.bindparams(user_id=user_id, image_loc=image_loc)
        
        # 쿼리 실행 및 변경사항 확정(Commit)
        await conn.execute(bind_stmt)
        await conn.commit()
        
    except SQLAlchemyError as e:
        print(e)
        await conn.rollback()
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="요청하신 서비스가 잠시 내부적으로 문제가 발생하였습니다.")

async def get_user_histories(conn: Connection, user_id: int):
    try:
        query = """
        SELECT image_loc, created_at
        FROM user_image
        WHERE user_id = :user_id
        ORDER by created_at DESC;
        """
        stmt = text(query)
        bind_stmt = stmt.bindparams(user_id=user_id)
        result = await conn.execute(bind_stmt)
        
        user_histories = [UserHistory(
            image_loc = row.image_loc,
            created_at = row.created_at
        )
                          for row in result]
        
        result.close()
        return user_histories
        
    except SQLAlchemyError as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="요청하신 서비스가 잠시 내부적으로 문제가 발생하였습니다.")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="알수없는 이유로 서비스 오류가 발생하였습니다")