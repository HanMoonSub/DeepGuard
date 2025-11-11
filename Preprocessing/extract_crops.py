import os, json
from os import cpu_count
from pathlib import Path
from glob import glob

import argparse

# --- 환경 변수 설정: 멀티스레딩 관련 ---
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from functools import partial
from multiprocessing.pool import Pool

import cv2
cv2.ocl.setUseOpenCL(False)  # OpenCL 사용 비활성화
cv2.setNumThreads(0)         # OpenCV 자체 스레드 사용 제한
from tqdm import tqdm
from colorama import Fore, Style

from .utils import get_video_paths  # utils.py에서 비디오 경로 가져오는 함수


def extract_video(param, frame_interval, margine_ratio, root_dir, crops_dir):
    """
    주어진 비디오에서 얼굴 영역만 crop하여 저장하는 함수.

    Args:
        param (tuple): (video_path, bboxes_json_path)
        frame_interval (int): 몇 프레임마다 추출할지 간격
        margine_ratio (float): crop 영역 확대 비율
        root_dir (str): crop 저장할 root 디렉토리
        crops_dir (str): crop 하위 디렉토리 이름

    Example:
        param = ("root_dir/video_folder/video.mp4", "root_dir/box_dir/video.json")
        extract_video(param, frame_interval=10, margine_ratio=0.3, root_dir="root_dir", crops_dir="crops")
    """
    video_path, bboxes_path = param

    # --- bounding box JSON 로드 ---
    try:
        with open(bboxes_path, "r") as f:
            bboxes_dict = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load bboxes file {bboxes_path}: {e}")
        return

    # --- 비디오 캡쳐 초기화 ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Failed to open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_id = os.path.splitext(os.path.basename(video_path))[0]

    # --- 출력 디렉토리 생성 ---
    output_dir = os.path.join(root_dir, crops_dir, video_id)
    os.makedirs(output_dir, exist_ok=True)

    # --- 프레임 처리 루프 ---
    for frame_idx in range(total_frames):
        cap.grab()  # 프레임을 효율적으로 건너뜀

        # 지정된 간격(frame_interval)마다 처리
        if frame_idx % frame_interval != 0:
            continue

        ret, frame = cap.retrieve()
        if not ret:
            continue

        # 현재 프레임에 bbox 정보가 없으면 건너뜀
        if str(frame_idx) not in bboxes_dict:
            continue

        bboxes = bboxes_dict[str(frame_idx)]
        if not bboxes:
            continue

        # --- 각 얼굴 bbox별 crop ---
        for face_idx, bbox in enumerate(bboxes):
            # 원본 프레임 크기로 좌표 복원 (2배 스케일)
            xmin, ymin, xmax, ymax = [int(coord * 2) for coord in bbox]

            # 마진 적용
            w = xmax - xmin
            h = ymax - ymin
            pad_w = int(w * margine_ratio)
            pad_h = int(h * margine_ratio)

            # 이미지 범위 내로 좌표 제한
            y1 = max(ymin - pad_h, 0)
            y2 = min(ymax + pad_h, frame.shape[0])
            x1 = max(xmin - pad_w, 0)
            x2 = min(xmax + pad_w, frame.shape[1])

            crop = frame[y1:y2, x1:x2]

            # 빈 crop이면 건너뜀
            if crop.size == 0:
                continue

            # crop 저장
            save_path = os.path.join(output_dir, f"{frame_idx}_{face_idx}.png")
            cv2.imwrite(save_path, crop)

    cap.release()

    # --- crop 결과 확인 ---
    cropped_files = glob(os.path.join(output_dir, '*.png'))
    if len(cropped_files) == 0:
        try:
            os.rmdir(output_dir)  # 비어있으면 디렉토리 삭제
            print(f"[INFO] Removed empty directory: {output_dir}")
        except OSError as e:
            print(f"[WARN] Failed to remove empty directory: {output_dir} | {e}")
    else:
        print(f"{Fore.BLUE}{Style.BRIGHT}[Info] Total Cropped Images are {len(cropped_files)} [Saved] {output_dir}")


if __name__ == "__main__":
    # --- 명령행 인자 파서 설정 ---
    parser = argparse.ArgumentParser(description="Extracts crops from video")
    parser.add_argument("--root_dir", help="root directory")  # 데이터셋 루트
    parser.add_argument("--crops_dir", help="crops directory")  # crop 저장 디렉토리
    parser.add_argument("--frame_interval", default=10, type=int, help="Interval between processed frames")
    parser.add_argument("--margine_ratio", default=0.3, type=float, help="crop margine")

    args = parser.parse_args()

    # crop 최상위 폴더 생성
    os.makedirs(os.path.join(args.root_dir, args.crops_dir), exist_ok=True)

    # --- 비디오 경로 목록 가져오기 ---
    params = get_video_paths(args.root_dir)  # (video_path, bbox_path) 리스트

    # --- 멀티 프로세싱으로 영상 처리 ---
    with Pool(processes=cpu_count()) as p:
        with tqdm(total=len(params)) as pbar:  # 진행률 표시
            # partial로 extract_video에 인자 전달
            for _ in p.imap_unordered(
                partial(
                    extract_video,
                    frame_interval=args.frame_interval,
                    margine_ratio=args.margine_ratio,
                    root_dir=args.root_dir,
                    crops_dir=args.crops_dir
                ),
                params
            ):
                pbar.update()  # 한 영상 처리될 때마다 진행률 업데이트
