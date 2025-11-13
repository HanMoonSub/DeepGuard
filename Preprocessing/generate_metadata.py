import os, json
import random
import argparse
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path

# --- 멀티스레딩 관련 환경 변수 설정 ---
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
from tqdm import tqdm
from glob import glob

from .utils import get_original_with_fakes  # metadata에서 (real, fake) 쌍 가져오는 함수

import cv2
cv2.ocl.setUseOpenCL(False)  # OpenCL 비활성화
cv2.setNumThreads(0)         # OpenCV 스레드 제한


def get_paths(vid, label, root_dir):
    """
    특정 비디오의 frame 이미지 경로와 라벨 정보를 수집하는 함수.

    Args:
        vid (tuple): (original_video, fake_video)
        label (int): 0=REAL, 1=FAKE
        root_dir (str): crop 이미지가 저장된 root 경로

    Returns:
        list: [[frame_path, label, ori_vid], ...] 형태
    """
    ori_vid, fake_vid = vid
    base_dir = os.path.join(root_dir, "crops")  # crop 이미지 저장 경로

    data = []

    # target_dir 결정: REAL이면 ori_vid, FAKE이면 fake_vid
    target_vid = ori_vid if label == 0 else fake_vid
    target_dir = os.path.join(base_dir, target_vid)

    # crop 폴더가 없으면 에러
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"Target video directory not found: {target_dir}")

    # 프레임 이미지 리스트
    frame_files = sorted(glob(os.path.join(target_dir, "*.png")))
    assert len(frame_files) > 0, f"[ERROR] No frame images found in directory: {target_dir}"

    # 각 frame 처리
    for frame_path in frame_files:
        filename = os.path.basename(frame_path)
        frame_idx, face_idx = filename.replace(".png", "").split("_")
        image_id = f"{frame_idx}_{face_idx}.png"
        img_path = os.path.join(target_dir, image_id)

        try:
            if os.path.exists(img_path):
                data.append([img_path, label, ori_vid])
        except:
            pass

    return data


def collect_metadata(pairs, label, root_dir, desc):
    """
    멀티프로세싱으로 비디오들의 frame metadata를 수집하는 함수.

    Args:
        pairs (list): [(ori_vid, fake_vid), ...] 리스트
        label (int): 0=REAL, 1=FAKE
        root_dir (str): crop root 경로
        desc (str): tqdm description

    Returns:
        list: [[frame_path, label, ori_vid], ...] 형태
    """
    func = partial(get_paths, label=label, root_dir=root_dir)  # get_paths에 label, root_dir 고정
    metadata = []

    # 멀티프로세싱 Pool 사용
    with Pool(processes=os.cpu_count()) as p:
        for result in tqdm(p.imap_unordered(func, pairs), total=len(pairs), desc=desc):
            if result:
                metadata.extend(result)
    return metadata


def main():
    """
    전체 dataset metadata 생성 및 CSV 파일로 저장
    """
    parser = argparse.ArgumentParser(description="Generate CSV File and Move Frame into output directory")
    parser.add_argument("--root_dir", help="root directory")  # 데이터셋 root
    parser.add_argument("--output_dir", help="output(metadata) directory")  # CSV 저장 경로

    args = parser.parse_args()

    # 입력/출력 경로 출력
    print(f"📂 Source dataset: {args.root_dir}")
    print(f"💾 Output dataset: {args.output_dir}") 

    # metadata에서 (ori, fake) 쌍 가져오기
    ori_fakes = get_original_with_fakes(args.root_dir, cropped=True)
    ori_ori = set([(ori, ori) for ori, fake in ori_fakes])  # REAL용 쌍

    # --- REAL / FAKE frame metadata 수집 ---
    real_meta = collect_metadata(ori_ori, label=0, root_dir=args.root_dir, desc="Collecting REAL frames")
    fake_meta = collect_metadata(ori_fakes, label=1, root_dir=args.root_dir, desc="Collecting FAKE frames")

    print(f"[INFO] Total Real Frames: {len(real_meta)}, Total Fake Frames: {len(fake_meta)}")

    # --- 전체 metadata 합치기 ---
    metadata = real_meta + fake_meta

    data = []
    for img_path, label, ori_vid in metadata:
        path = Path(img_path)
        video = path.parent.name
        file = path.name
        frame_idx, face_idx = file.replace(".png", "").split("_")

        # 각 frame의 정보 리스트
        data.append([video, file, frame_idx, face_idx, label, ori_vid])

        """
        Columns 설명:
        video: 선택된 비디오 이름
        file: {frame_idx}_{face_idx}.png
        frame_idx: 0,1,2...
        face_idx: 0,1...
        label: 0(Real), 1(Fake)
        ori_vid: 원본 비디오 이름
        """

    # --- DataFrame 생성 및 CSV 저장 ---
    df = pd.DataFrame(data, columns=["video", "file", "frame_idx", "face_idx", "label", "ori_vid"])
    df.sort_values(by=['video', 'frame_idx', 'face_idx'], inplace=True)

    # output 디렉토리 생성
    os.makedirs(os.path.join(args.root_dir, args.output_dir), exist_ok=True)
    csv_path = os.path.join(args.output_dir, "train_metadata.csv")
    df.to_csv(csv_path, index=False)

    print(f"✅ [Summary] Total Videos: {df['video'].nunique()}, Total Frames: {len(df['file'])}")


if __name__ == "__main__":
    main()
