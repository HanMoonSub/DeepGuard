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
    """업로드된 비디오를 서버에 저장하고 DB 저장용 경로를 반환한다. 회원/비회원 공통.

    Args:
        user_email (str | None): 로그인 사용자 이메일. 비회원이면 None.
        videofile (UploadFile): FastAPI 업로드 파일 객체.

    Returns:
        str: 저장된 비디오의 DB 기준 경로 ('/'로 시작, '\\' 정규화됨).

    Raises:
        HTTPException 500: 디렉토리 생성, 파일 쓰기, 또는 예기치 못한 오류 발생 시.
    """
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

        # 6. DB 저장용 경로 반환
        return upload_video_loc[1:].replace("\\", "/")
    
    except HTTPException:
        raise
    
    except Exception as e:
        print(f"[Unknown Error] {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="비디오 업로드 과정에서 예상치 못한 오류가 발생했습니다.")


async def delete_video(video_loc: str) -> None:
    """서버에서 비디오 물리 파일을 삭제한다.

    파일이 존재하지 않으면 경고 로그만 출력하고 정상 종료한다.

    Args:
        video_loc (str): 삭제할 비디오의 DB 기준 경로 ('/'로 시작).

    Raises:
        HTTPException 500: 파일 삭제 중 오류 발생 시.
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
    """사용자의 전체 비디오 분석 히스토리를 최신순으로 조회한다.

    Args:
        conn (Connection): SQLAlchemy 비동기 DB 커넥션.
        user_id (int): 조회할 사용자 PK.

    Returns:
        list[VideoData]: 비디오 히스토리 스키마 리스트. 결과 없으면 빈 리스트.

    Raises:
        HTTPException 503: DB 쿼리 중 SQLAlchemy 오류 발생 시.
        HTTPException 500: 예기치 못한 오류 발생 시.
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
    """video_id와 user_id로 비디오 개별 상세 히스토리를 조회한다.

    본인 소유 데이터만 조회되도록 user_id를 함께 필터링한다.

    Args:
        conn (Connection): SQLAlchemy 비동기 DB 커넥션.
        user_id (int): 요청 사용자 PK.
        video_id (int): 조회할 비디오 레코드 PK.

    Returns:
        VideoDetailData | None: 상세 히스토리 스키마 객체. 레코드 없으면 None.

    Raises:
        HTTPException 503: DB 쿼리 중 SQLAlchemy 오류 발생 시.
        HTTPException 500: 예기치 못한 오류 발생 시.
    """
    try:
        # user_id 있으면 본인 것만, None이면 video_id로만 조회 (비회원 상세 허용)
        if user_id is not None:
            stmt = text("""
                SELECT id, user_id, video_loc, status, label, score, face_conf, face_ratio,
                        face_brightness, version_type, model_type, domain_type, result_msg, created_at
                FROM video_result
                WHERE id = :video_id AND user_id = :user_id
            """)
            params = {"video_id": video_id, "user_id": user_id}
        else:
            stmt = text("""
                SELECT id, user_id, video_loc, status, label, score, face_conf, face_ratio,
                        face_brightness, version_type, model_type, domain_type, result_msg, created_at
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


async def register_video_result(conn: Connection, user_id: int | None, video_loc: str,
                                version_type: str, model_type: str, domain_type: str) -> int:
    """PENDING 상태의 빈 비디오 레코드를 DB에 생성하고 video_id를 반환한다.

    Args:
        conn (Connection): SQLAlchemy 비동기 DB 커넥션.
        user_id (int | None): 로그인 사용자 ID. 비회원이면 None.
        video_loc (str): 업로드된 비디오의 DB 기준 경로.
        version_type (str): 모델 버전 타입.
        model_type (str): 추론 모드 ("fast" | "pro").
        domain_type (str): 탐지 도메인.

    Returns:
        int: 생성된 비디오 레코드의 PK (auto_increment).

    Raises:
        HTTPException 400: DB INSERT 실패 시. 업로드된 물리 파일도 함께 롤백 삭제된다.
    """
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


async def update_video_result(conn: Connection, video_id: int, analysis: dict,
                              result_msg: str, status: str) -> None:
    """추론 완료 후 비디오 레코드에 분석 결과를 업데이트한다.

    Args:
        conn (Connection): SQLAlchemy 비동기 DB 커넥션.
        video_id (int): 업데이트할 비디오 레코드 PK.
        analysis (dict): 추론 결과값. prob, face_conf, face_ratio, face_brightness 포함.
        result_msg (str): 처리 결과 메세지.
        status (str): 추론 상태 ("success" | "warning" | "failed").

    Raises:
        SQLAlchemyError: DB 업데이트 실패 시 그대로 re-raise.
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


async def get_video_result(conn: Connection, video_id: int) -> VideoDetailData:
    """video_id로 비디오 분석 결과를 조회해 반환한다.

    Args:
        conn (Connection): SQLAlchemy 비동기 DB 커넥션.
        video_id (int): 조회할 비디오 레코드 PK.

    Returns:
        VideoDetailData: 비디오 분석 결과 스키마 객체.

    Raises:
        HTTPException 404: 해당 video_id 레코드가 없을 때.
        HTTPException 503: DB 쿼리 중 SQLAlchemy 오류 발생 시.
        HTTPException 500: 예기치 못한 오류 발생 시.
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


async def save_video_meta_result(conn: Connection, video_id: int, analysis: dict) -> None:
    """비디오 1개의 프레임 요약 통계를 video_meta_result 테이블에 저장한다.

    Args:
        conn (Connection): SQLAlchemy 비동기 DB 커넥션.
        video_id (int): 비디오 레코드 PK.
        analysis (dict): 추론 결과 딕셔너리. fps, total_frames, num_sampled,
            num_extracted, num_detected 키를 포함해야 한다.

    Raises:
        SQLAlchemyError: DB INSERT 실패 시 그대로 re-raise.
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
    """비디오에 속한 모든 프레임의 분석 결과를 video_frame_result 테이블에 일괄 저장한다.

    Args:
        conn (Connection): SQLAlchemy 비동기 DB 커넥션.
        video_id (int): 비디오 레코드 PK.
        frame_results (list): 프레임별 분석 결과 딕셔너리 리스트.
            각 항목은 frame_index, frame_time, score, face_conf,
            face_ratio, face_brightness 키를 포함해야 한다.

    Raises:
        SQLAlchemyError: DB INSERT 실패 시 그대로 re-raise.
    """
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


async def delete_video_meta_result(conn: Connection, video_id: int) -> None:
    """video_meta_result 테이블에서 해당 비디오의 메타 데이터를 삭제한다.

    Args:
        conn (Connection): SQLAlchemy 비동기 DB 커넥션.
        video_id (int): 삭제할 비디오 레코드 PK.

    Raises:
        HTTPException 404: 해당 video_id 레코드가 없을 때.
        HTTPException 503: DB 쿼리 중 SQLAlchemy 오류 발생 시.
        HTTPException 500: 예기치 못한 오류 발생 시.
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
    """video_frame_result 테이블에서 해당 비디오의 프레임 결과를 삭제한다.

    Args:
        conn (Connection): SQLAlchemy 비동기 DB 커넥션.
        video_id (int): 삭제할 비디오 레코드 PK.

    Raises:
        HTTPException 404: 해당 video_id 레코드가 없을 때.
        HTTPException 503: DB 쿼리 중 SQLAlchemy 오류 발생 시.
        HTTPException 500: 예기치 못한 오류 발생 시.
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
    """video_id로 비디오 메타 데이터를 조회한다.

    fps, total_frames, num_sampled, num_extracted, num_detected를 반환한다.

    Args:
        conn (Connection): SQLAlchemy 비동기 DB 커넥션.
        video_id (int): 조회할 비디오 레코드 PK.

    Returns:
        VideoMetaData: 비디오 메타 데이터 스키마 객체.

    Raises:
        HTTPException 404: 해당 video_id 메타 레코드가 없을 때.
        HTTPException 503: DB 쿼리 중 SQLAlchemy 오류 발생 시.
        HTTPException 500: 예기치 못한 오류 발생 시.
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
    """video_id에 속한 모든 프레임 분석 결과를 frame_index 오름차순으로 조회한다.

    프레임별 점수 그래프, 의심 구간 표시 등 시각화에 활용된다.

    Args:
        conn (Connection): SQLAlchemy 비동기 DB 커넥션.
        video_id (int): 조회할 비디오 레코드 PK.

    Returns:
        list[VideoFrameData]: 프레임별 분석 결과 스키마 리스트.

    Raises:
        HTTPException 503: DB 쿼리 중 SQLAlchemy 오류 발생 시.
        HTTPException 500: 예기치 못한 오류 발생 시.
    """
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


async def get_video_frame_by_index(conn: Connection, video_id: int, frame_index: int) -> float:
    """video_id와 frame_index로 특정 프레임의 frame_time을 조회한다.

    Args:
        conn (Connection): SQLAlchemy 비동기 DB 커넥션.
        video_id (int): 비디오 레코드 PK.
        frame_index (int): 조회할 프레임 인덱스.

    Returns:
        float: 해당 프레임의 타임스탬프 (초 단위).

    Raises:
        HTTPException 404: 해당 frame_index가 없을 때.
        HTTPException 503: DB 쿼리 중 SQLAlchemy 오류 발생 시.
        HTTPException 500: 예기치 못한 오류 발생 시.
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
    """비회원 비디오 DB 레코드 및 물리 파일을 삭제하는 Celery 태스크.

    추론 성공 시 meta/frame 결과도 함께 삭제한다.
    각 삭제 단계는 독립적으로 처리되어 한쪽 실패가 다른쪽에 영향을 주지 않는다.

    Args:
        video_id (int): 삭제할 비디오 레코드 PK.
        video_loc (str): 삭제할 비디오의 DB 기준 경로.
        is_success (bool): 추론 성공 여부. True면 meta/frame 결과도 함께 삭제.
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
    """DB에서 비디오 레코드를 삭제한다.

    Args:
        conn (Connection): SQLAlchemy 비동기 DB 커넥션.
        video_id (int): 삭제할 비디오 레코드 PK.

    Raises:
        HTTPException 404: 해당 video_id 레코드가 없을 때.
        HTTPException 503: DB 쿼리 중 SQLAlchemy 오류 발생 시.
        HTTPException 500: 예기치 못한 오류 발생 시.
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