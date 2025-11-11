import os
import argparse
from functools import partial
from multiprocessing.pool import Pool
from glob import glob

import cv2
from tqdm import tqdm
from PIL import Image
from facenet_pytorch import MTCNN
import numpy as np
import torch
from colorama import Style, Fore

from .utils import get_original_video_paths


# ==========================================
# 환경 설정
# ==========================================
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

# ==========================================
# Face Landmark Detector 초기화
# ==========================================
device = "cuda:0" if torch.cuda.is_available() else "cpu"
detector = MTCNN(
    margin=0,
    thresholds=[0.65, 0.75, 0.75],
    device=device
)


# ==========================================
# 각 영상의 얼굴 랜드마크 저장 함수
# ==========================================
def save_landmarks(ori_id, root_dir):
    """
    각 영상의 crop된 이미지(.png)에서 얼굴 랜드마크를 추출하여
    root_dir/landmarks/<video_id>/ 경로에 저장합니다.
    """
    
    # video.mp4 → video
    ori_id = ori_id[:-4]
    ori_dir = os.path.join(root_dir, "crops", ori_id)

    # crops/<video> 폴더 존재 확인
    if not os.path.exists(ori_dir):
        raise FileNotFoundError(f"Crop directory not found: {ori_dir}")
    

    # landmarks/<video> 경로 생성
    output_dir = os.path.join(root_dir, "landmarks", ori_id)
    os.makedirs(output_dir, exist_ok=True)

    # crops/<video>/*.png 파일 불러오기
    frame_files = sorted(glob(os.path.join(ori_dir, "*.png")))
    assert len(frame_files) > 0, print(f"{Fore.RED}[ERROR] No frame images found in: {ori_dir}{Style.RESET_ALL}")
        

    # 각 프레임 이미지 처리
    for frame_path in frame_files:
        filename = os.path.basename(frame_path)

        # 파일명 형식: {frame_idx}_{face_idx}.png
        frame_idx, face_idx = filename.replace(".png", "").split("_")


        landmark_id = f"{frame_idx}_{face_idx}"
        landmark_path = os.path.join(output_dir, landmark_id)

        # 이미지 읽기 및 RGB 변환
        image_ori = cv2.imread(frame_path, cv2.IMREAD_COLOR)[..., ::-1]
        frame_img = Image.fromarray(image_ori)

        # MTCNN을 이용한 얼굴 랜드마크 탐지
        _, _, landmarks = detector.detect(frame_img, landmarks=True)

        # 랜드마크가 탐지된 경우 저장
        if landmarks is not None:
            landmarks = np.around(landmarks[0]).astype(np.int16)
            np.save(landmark_path, landmarks)

    # 저장된 랜드마크 파일 확인
    landmark_files = glob(os.path.join(output_dir, '*'))
    if len(landmark_files) == 0:
        # landmarks 폴더 비어 있으면 삭제
        try:
            os.rmdir(output_dir)
            print(f"{Fore.YELLOW}[INFO] Removed empty directory: {output_dir}{Style.RESET_ALL}")
        except OSError as e:
            print(f"{Fore.RED}[WARN] Failed to remove empty directory: {output_dir} | {e}{Style.RESET_ALL}")
    else:
        print(f"{Fore.CYAN}{Style.BRIGHT}[INFO] Saved {len(landmark_files)} landmark files from {len(frame_files)} cropped files→ {output_dir}{Style.RESET_ALL}")

# ==========================================
# 전체 프로세스 실행
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Extract image landmarks from cropped video frames")
    parser.add_argument("--root-dir", required=True, help="Root directory containing crops/ and landmarks/ folders")
    args = parser.parse_args()

    # root_dir/videos/ 내부의 원본 영상 목록 가져오기
    ids = get_original_video_paths(args.root_dir, cropped=True, basename=True)

    # landmarks/ 폴더 생성
    os.makedirs(os.path.join(args.root_dir, "landmarks"), exist_ok=True)

    # 병렬 처리로 landmarks 추출
    with Pool(processes=os.cpu_count()) as pool:
        with tqdm(total=len(ids), desc="Extracting Landmarks") as pbar:
            func = partial(save_landmarks, root_dir=args.root_dir)
            for _ in pool.imap_unordered(func, ids):
                pbar.update()


# ==========================================
# 엔트리 포인트
# ==========================================
if __name__ == "__main__":
    main()
