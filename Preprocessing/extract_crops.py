import os,json
from os import cpu_count
from pathlib import Path
from glob import glob

import argparse

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from functools import partial
from multiprocessing.pool import Pool

import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
from tqdm import tqdm
from colorama import Fore, Style

def extract_video(param, frame_interval, margine_ratio, root_dir, crops_dir):
    """
    Extract face crops from a video using precomputed bounding boxes.

    Args:
        param (tuple): (video_path, bboxes_json_path)
        root_dir (str): Root directory for saving crops
        crops_dir (str): Subfolder name for cropped images

    Example:
        param = ("root_dir/video_folder/video.mp4", "root_dir/box_dir/video.json")
        extract_video(param, "output_root", "crops")
    """
    video_path, bboxes_path = param

    # --- Load bounding boxes ---
    try:
        with open(bboxes_path, "r") as f:
            bboxes_dict = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load bboxes file {bboxes_path}: {e}")
        return

    # --- Initialize video capture ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Failed to open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_id = os.path.splitext(os.path.basename(video_path))[0]

    # --- Prepare output directory ---
    output_dir = os.path.join(root_dir, crops_dir, video_id)
    os.makedirs(output_dir, exist_ok=True)

    # --- Frame processing loop ---
    for frame_idx in range(total_frames):
        cap.grab()  # Efficiently skip frame

        # Process every 10th frame only
        if frame_idx % frame_interval != 0:
            continue

        ret, frame = cap.retrieve()
        if not ret:
            continue

        # If there is no bbox info for this frame, skip
        if str(frame_idx) not in bboxes_dict:
            continue

        bboxes = bboxes_dict[str(frame_idx)]
        if not bboxes:
            continue

        for face_idx, bbox in enumerate(bboxes):
            # Scale coordinates (detection was done on half-size frames)
            xmin, ymin, xmax, ymax = [int(coord * 2) for coord in bbox] # 2X Scaling for Restore

            # Add margin
            w = xmax - xmin
            h = ymax - ymin
            pad_w = int(w * margine_ratio)
            pad_h = int(h * margine_ratio)

            # Ensure coordinates stay inside image bounds
            y1 = max(ymin - pad_h, 0)
            y2 = min(ymax + pad_h, frame.shape[0])
            x1 = max(xmin - pad_w, 0)
            x2 = min(xmax + pad_w, frame.shape[1])

            crop = frame[y1:y2, x1:x2]

            # Skip empty crops
            if crop.size == 0:
                continue

            # Save crop
            # path: Data_root/crops_dir/video/frame_idx_face_idx
            save_path = os.path.join(output_dir, f"{frame_idx}_{face_idx}.png")
            cv2.imwrite(save_path, crop)
    
    cap.release()
    cnt = glob(os.path.join(output_dir, '*.png'))
    print(f"{Fore.BLUE}{Style.BRIGHT}[Info] Total Cropped Images are {len(cnt)} [Saved] {output_dir}")
            
def get_video_paths(root_dir):
    
    paths = []
    
    for json_path in glob(os.path.join(root_dir, "*/metadata.json")):
        dir = Path(json_path).parent # root_dir/video_folder
        
        with open(json_path, "r") as f:
            metadata = json.load(f)
        
        for k, v in metadata.items():
            # Real Video: None, Fake Video: Original
            original = v.get("original", None)
            if not original:
                original = k # video.mp4
            bboxes_path = os.path.join(root_dir, "boxes", original[:-4] + ".json")
            
            if not os.path.exists(bboxes_path):
                continue
            
            # root_dir/video_folder/video.mp4, root_dir/boxes/video.json
            paths.append((os.path.join(dir, k), bboxes_path))
            
    return paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracts crops from video")
    parser.add_argument("--root_dir", help="root directory")
    parser.add_argument("--crops_dir", help="crops directory")
    parser.add_argument("--frame_interval", default=10, type=int, help="Interval between processed frames")
    parser.add_argument("--margine_ratio", default=0.3, type=float, help="crop margine")
    
    args = parser.parse_args()
    os.makedirs(os.path.join(args.root_dir, args.crops_dir), exist_ok=True)
    params = get_video_paths(args.root_dir)
        
    with Pool(processes=cpu_count()) as p:
        with tqdm(total=len(params)) as pbar:
            for _ in p.imap_unordered(partial(extract_video, frame_interval=args.frame_interval, margine_ratio=args.margine_ratio, root_dir=args.root_dir, crops_dir=args.crops_dir), params):
                pbar.update()      