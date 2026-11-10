import asyncio
import json
from typing import Literal

from celery_app import celery_app
from db.database import celery_db_conn
from fastapi import status
from fastapi.exceptions import HTTPException
from schemas.artifact_schema import ArtifactResultData
from services import inference_svc
from sqlalchemy import Connection, text
from sqlalchemy.exc import SQLAlchemyError

from inference.utils import PredictorError

# 미디어 종류별 보조 테이블 / FK 컬럼 (쿼리 문자열에만 사용, 사용자 입력 아님)
_TABLES = {
    "image": ("image_artifact_result", "image_id"),
    "video": ("video_artifact_result", "video_id"),
}


async def get_artifact(conn: Connection, media: Literal["image", "video"], media_id: int) -> ArtifactResultData | None:
    """저장된 DeepClip 20개 텍스트 확률을 조회한다.

    Args:
        conn (Connection): SQLAlchemy 비동기 DB 커넥션.
        media (Literal["image", "video"]): 미디어 종류.
        media_id (int): image_result.id 또는 video_result.id.

    Returns:
        ArtifactResultData | None: 저장된 결과. 없으면 None.

    Raises:
        HTTPException 503: DB 쿼리 중 SQLAlchemy 오류 발생 시.
    """
    table, fk = _TABLES[media]
    try:
        query = text(f"SELECT {fk}, model_name, artifact_probs, created_at FROM {table} WHERE {fk} = :media_id")
        row = (await conn.execute(query, {"media_id": media_id})).fetchone()
        if row is None:
            return None

        probs = row.artifact_probs
        return ArtifactResultData(
            media_id=getattr(row, fk),
            model_name=row.model_name,
            artifact_probs=json.loads(probs) if isinstance(probs, str) else probs,
            created_at=row.created_at,
        )
    except SQLAlchemyError as e:
        print(f"[Artifact Query Error] {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="데이터베이스 조회 중 문제가 발생했습니다.")


async def save_artifact(conn: Connection, media: Literal["image", "video"], media_id: int,
                        model_name: str, artifact_probs: list[float]) -> None:
    """DeepClip 20개 텍스트 확률을 저장한다. 이미 있으면 덮어쓴다.

    Args:
        conn (Connection): SQLAlchemy 비동기 DB 커넥션.
        media (Literal["image", "video"]): 미디어 종류.
        media_id (int): image_result.id 또는 video_result.id.
        model_name (str): 사용한 DeepClip 모델 이름 (deepclip_b0 / deepclip_b5).
        artifact_probs (list[float]): PAIR_ORDER 순서의 20개 확률 (idx = 5*region + type).
    """
    table, fk = _TABLES[media]
    try:
        query = text(f"""
            INSERT INTO {table} ({fk}, model_name, artifact_probs)
            VALUES (:media_id, :model_name, :artifact_probs)
            ON DUPLICATE KEY UPDATE model_name = VALUES(model_name), artifact_probs = VALUES(artifact_probs)
        """)
        await conn.execute(query, {"media_id": media_id, "model_name": model_name,
                                   "artifact_probs": json.dumps(artifact_probs)})
        await conn.commit()
    except SQLAlchemyError:
        await conn.rollback()
        raise


async def delete_artifact(conn: Connection, media: Literal["image", "video"], media_id: int) -> None:
    """저장된 DeepClip 20개 텍스트 확률을 삭제한다. 없으면 아무것도 하지 않는다.

    Raises:
        HTTPException 503: DB 쿼리 중 SQLAlchemy 오류 발생 시.
    """
    table, fk = _TABLES[media]
    try:
        await conn.execute(text(f"DELETE FROM {table} WHERE {fk} = :media_id"), {"media_id": media_id})
        await conn.commit()
    except SQLAlchemyError as e:
        print(f"[Artifact Delete Error] {e}")
        await conn.rollback()
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="요청하신 서비스가 잠시 내부적으로 문제가 발생하였습니다.")


def _run_artifact_task(media: Literal["image", "video"], media_id: int, media_loc: str, model_type: str) -> dict:
    """DeepClip으로 20개 텍스트 확률을 추론하고 DB에 저장한다 (Celery 태스크 공통 로직).

    Returns:
        dict:
            - status (str): "SUCCESS" | "WARNING" | "FAILED"
            - message (str): 처리 결과 메세지.
            - artifact_probs (list[float] | None): 20개 확률. 실패 시 None.
    """
    model_name = inference_svc.DEEPCLIP_MODEL[model_type]
    path = "." + media_loc

    try:
        predictor = inference_svc._get_or_create_predictor(model_name, None, media)
        if media == "image":
            artifact_probs = predictor.predict_img_artifacts(path)["artifact_probs"]
        else:
            artifact_probs = predictor.predict_video_artifacts(path)["artifact_probs"]
    except PredictorError as e:
        print(e.message)
        return {"status": "WARNING", "message": e.message, "artifact_probs": None}
    except Exception as e:
        print(str(e))
        return {"status": "FAILED", "message": "부위별 위조 흔적 분석 중 오류가 발생했습니다.", "artifact_probs": None}

    async def _save():
        async with celery_db_conn() as conn:
            await save_artifact(conn, media, media_id, model_name, artifact_probs)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_save())
    except Exception as e:
        print(f"[Artifact Save Error] {e}")
        return {"status": "FAILED", "message": "부위별 위조 흔적 분석 결과를 저장하는 중 오류가 발생했습니다.", "artifact_probs": None}
    finally:
        loop.close()

    return {"status": "SUCCESS", "message": "부위별 위조 흔적 분석에 성공하였습니다", "artifact_probs": artifact_probs}


@celery_app.task(name="process_artifact_image_task")
def process_artifact_image_task(image_id: int, image_loc: str, model_type: str) -> dict:
    """FAKE 이미지의 DeepClip 20개 텍스트 확률을 추론·저장하는 Celery 태스크."""
    return _run_artifact_task("image", image_id, image_loc, model_type)


@celery_app.task(name="process_artifact_video_task")
def process_artifact_video_task(video_id: int, video_loc: str, model_type: str) -> dict:
    """FAKE 비디오의 DeepClip 20개 텍스트 확률(영상 전체 프레임 평균)을 추론·저장하는 Celery 태스크."""
    return _run_artifact_task("video", video_id, video_loc, model_type)
