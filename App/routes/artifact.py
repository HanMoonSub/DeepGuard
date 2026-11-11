import os

from celery.result import AsyncResult
from celery_app import celery_app
from db.database import context_get_conn
from fastapi import APIRouter, Depends, status
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from schemas.artifact_schema import ArtifactResultData
from services import artifact_svc, image_svc, session_svc, video_svc
from sqlalchemy import Connection

# DeepClip 20 text probabilities (region x type) - served on demand for FAKE verdicts only
router = APIRouter(prefix="/artifact", tags=["artifact"])


def _check_target(result, session_user: dict, media_name: str, media_loc: str) -> None:
    """Check that the record belongs to the user, the verdict succeeded as FAKE, and the original file still exists."""
    if result.user_id != session_user["id"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"본인의 {media_name} 기록만 조회할 수 있습니다.")

    if result.status != "SUCCESS" or result.label != "FAKE":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"부위별 위조 흔적 분석은 FAKE로 판정된 {media_name}에서만 가능합니다",
        )

    if not os.path.exists("." + media_loc):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"요청하신 {media_name} 파일을 찾을 수 없습니다. 삭제하였는지 다시 확인해주세요.",
        )


async def _get_saved(conn: Connection, media: str, media_id: int) -> ArtifactResultData:
    saved = await artifact_svc.get_artifact(conn, media, media_id)
    if saved is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="아직 부위별 위조 흔적 분석 결과가 없습니다.")
    return saved


@router.get("/image/{image_id}", response_model=ArtifactResultData, summary="이미지 부위별 위조 흔적(20개 텍스트 확률) 조회")
async def get_image_artifact(
    image_id: int,
    conn: Connection = Depends(context_get_conn),
    session_user = Depends(session_svc.get_session_user_prt), # login required
):
    result = await image_svc.get_image_result(conn, image_id)
    if result.user_id != session_user["id"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="본인의 이미지 기록만 조회할 수 있습니다.")
    return await _get_saved(conn, "image", image_id)


@router.get("/video/{video_id}", response_model=ArtifactResultData, summary="비디오 부위별 위조 흔적(영상 전체 평균 20개 텍스트 확률) 조회")
async def get_video_artifact(
    video_id: int,
    conn: Connection = Depends(context_get_conn),
    session_user = Depends(session_svc.get_session_user_prt), # login required
):
    result = await video_svc.get_video_result(conn, video_id)
    if result.user_id != session_user["id"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="본인의 비디오 기록만 조회할 수 있습니다.")
    return await _get_saved(conn, "video", video_id)


@router.post("/image/{image_id}", status_code=status.HTTP_202_ACCEPTED,
             response_class=JSONResponse, summary="이미지 부위별 위조 흔적 분석 비동기 접수")
async def analyze_image_artifact(
    image_id: int,
    conn: Connection = Depends(context_get_conn),
    session_user = Depends(session_svc.get_session_user_prt), # login required
):
    result = await image_svc.get_image_result(conn, image_id)
    _check_target(result, session_user, "이미지", result.image_loc)

    # Return the stored result without re-running inference if one exists
    saved = await artifact_svc.get_artifact(conn, "image", image_id)
    if saved is not None:
        return JSONResponse(status_code=status.HTTP_200_OK, content=saved.model_dump(mode="json"))

    task = artifact_svc.process_artifact_image_task.delay(
        image_id=image_id, image_loc=result.image_loc, model_type=result.model_type)
    return {"message": "이미지 부위별 위조 흔적 분석 접수 완료. 분석 시작 ...", "task_id": task.id}


@router.post("/video/{video_id}", status_code=status.HTTP_202_ACCEPTED,
             response_class=JSONResponse, summary="비디오 부위별 위조 흔적 분석 비동기 접수")
async def analyze_video_artifact(
    video_id: int,
    conn: Connection = Depends(context_get_conn),
    session_user = Depends(session_svc.get_session_user_prt), # login required
):
    result = await video_svc.get_video_result(conn, video_id)
    _check_target(result, session_user, "비디오", result.video_loc)

    # Return the stored result without re-running inference if one exists
    saved = await artifact_svc.get_artifact(conn, "video", video_id)
    if saved is not None:
        return JSONResponse(status_code=status.HTTP_200_OK, content=saved.model_dump(mode="json"))

    task = artifact_svc.process_artifact_video_task.delay(
        video_id=video_id, video_loc=result.video_loc, model_type=result.model_type)
    return {"message": "비디오 부위별 위조 흔적 분석 접수 완료. 분석 시작 ...", "task_id": task.id}


@router.get("/result/{task_id}", status_code=status.HTTP_200_OK,
            response_class=JSONResponse, summary="부위별 위조 흔적 분석 결과 조회")
async def get_artifact_result(
    task_id: str,
    session_user = Depends(session_svc.get_session_user_prt), # login required
):
    # Fetch the async task status for this task ID from the Redis broker
    task = AsyncResult(task_id, app=celery_app)

    if task.state in ("PENDING", "STARTED", "RETRY"):
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={"message": "부위별 위조 흔적 분석 중 ..."},
        )

    if task.state == "FAILURE":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="부위별 위조 흔적 분석 중 알 수 없는 오류가 발생하였습니다")

    result = task.result

    # Analysis not possible (WARNING, e.g. no face detected) -> 422, any other error -> 500
    if result["status"] == "WARNING":
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=result["message"])
    if result["status"] == "FAILED":
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result["message"])

    return {
        "status": result["status"],
        "message": result["message"],
        "artifact_probs": result["artifact_probs"],
    }
