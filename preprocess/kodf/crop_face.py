import os
import time
import argparse
import json
import cv2
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed

from colorama import Fore, Style
from .utils import get_video_paths
from ..frame_extractor import FrameExtractor_KODF
from ..landmark_detector import LandmarkDetector_KODF

c_ = Fore.BLUE
g_ = Fore.GREEN
r_ = Fore.RED
y_ = Fore.YELLOW
s_ = Style.BRIGHT
rs_ = Style.RESET_ALL

frame_extractor = None
landmark_detector = None

def init_worker():
    global frame_extractor, landmark_detector
    frame_extractor = FrameExtractor_KODF()
    landmark_detector = LandmarkDetector_KODF()

def crop_frame_with_margin(frame: np.ndarray, 
               margin_ratio: float, 
               margin_jitter: float, 
               ori_box: List[float]) -> np.ndarray:
    
    xmin, ymin, xmax, ymax = ori_box
    
    current_margin = margin_ratio
    if margin_jitter > 0:
        jitter_scale = margin_ratio * margin_jitter
        noise = np.random.uniform(low=-jitter_scale, high=jitter_scale)
        current_margin = max(0.0, margin_ratio + noise)
    
    h, w = frame.shape[:2]
    box_w = xmax - xmin
    box_h = ymax - ymin
    pad_w = int(box_w * current_margin)
    pad_h = int(box_h * current_margin)
    y1 = max(int(ymin - pad_h), 0)
    y2 = min(int(ymax + pad_h), h)
    x1 = max(int(xmin - pad_w), 0)
    x2 = min(int(xmax + pad_w), w)
        
    return frame[y1:y2, x1:x2]

def process_crop(args_tuple: Tuple) -> Dict[str, int]:
    video_path, ori_video_id, boxes_dir, crop_base, landmark_base, margin_ratio, margin_jitter = args_tuple
    
    res = {
        "processed": 0, "skipped": 0, "frames": 0, "landmarks": 0,
        "message": None,  # Log Message
        "status": "OK"    # (INFO, SKIP, WARN, ERROR)
    }
    
    try:
        json_path = boxes_dir / f"{ori_video_id}.json"
        with open(json_path, "r") as f:
            ori_box_dict = json.load(f)

        ori_idx = next(iter(ori_box_dict))
        ori_box = ori_box_dict[ori_idx]
        
        frame_dict = frame_extractor.extract_frame(str(video_path), int(ori_idx))
        video_id = video_path.stem
        
        if not frame_dict:
            res["status"] = "SKIP"
            res["message"] = f"{y_}[SKIP]{rs_} Extraction failed: {c_}{video_id}{rs_}"
            return res

        frame_idx = next(iter(frame_dict))
        frame = frame_dict[frame_idx]
            
        crop = crop_frame_with_margin(frame, margin_ratio, margin_jitter, ori_box)
        
        lm_result = landmark_detector.detect_single_landmark(crop)
        
        if lm_result is not None:
            save_crop_dir = crop_base / video_id
            save_landmark_dir = landmark_base / video_id
            save_crop_dir.mkdir(parents=True, exist_ok=True)
            save_landmark_dir.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(save_crop_dir / f"{frame_idx}.png"), 
                        cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_PNG_COMPRESSION, 1])
            
            np.save(save_landmark_dir / f"{frame_idx}.npy", lm_result)
            res.update({"processed": 1, "frames": 1, "landmarks": 1})
            
        else:
            res.update({"status": "WARN", "skipped": 1, "message": f"{r_}[WARN]{rs_} Landmark not found: {c_}{video_id}{rs_}"})
    
    except FileNotFoundError: # No Bbox JSON
        res["status"] = "SKIP"
        res["message"] = f"{y_}[SKIP]{rs_} No Bbox JSON: {c_}{ori_video_id}{rs_}"
        
    except Exception as e:
        res["status"] = "ERROR"
        res["message"] = f"{r_}[ERROR]{rs_} {video_path.name}: {e}"
        res["skipped"] = 1
        
    return res
        
def main():
    parser = argparse.ArgumentParser(description="Crop Face and Extract Landmark")
    parser.add_argument("--root-dir", required=True, help="Root Directory with KODF")
    parser.add_argument("--work-dir", required=True, help="Directory containing the video files to process")
    parser.add_argument("--margin-ratio", default=0.2, type=float, help="Margin Ratio for Detected Face")
    parser.add_argument("--margin-jitter", default=0.0, type=float, help="Noise for Margin Ratio")
    parser.add_argument("--num-workers", default=1, type=int)
    parser.add_argument("--verbose", action="store_true", help="Print Log Information")
    
    args = parser.parse_args()
    print(f"\n{c_}{s_}>>> Step 1: Scanning video paths...{rs_}")
    video_paths, ori_video_ids = get_video_paths(args.root_dir, args.work_dir)
    
    num_videos = len(video_paths)
    print(f"{g_}🔍 Total {num_videos} videos found for processing.{rs_}")
    
    root_path = Path(args.root_dir)
    boxes_dir = root_path / "boxes"
    crop_base = root_path / "crops"
    landmark_base = root_path / "landmarks"
    
    
    tasks = [
        (vp, vid, boxes_dir, crop_base, landmark_base, args.margin_ratio, args.margin_jitter)
        for vp, vid in zip(video_paths, ori_video_ids)
    ]
    
    summary = {
        "total_videos": len(video_paths),
        "processed_videos": 0,
        "skipped_videos": 0,
        "total_frames_saved": 0,
        "total_landmarks_found": 0
    }
    
    print(f"{c_}{s_}>>> Step 2: Parallel Processing with {args.num_workers} workers...{rs_}\n")
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=args.num_workers, initializer=init_worker) as executor:
        futures = [executor.submit(process_crop, task) for task in tasks]

        pbar = tqdm(as_completed(futures), total=len(futures), desc="🔄 Processing", mininterval=2.0)
        for i, future in enumerate(pbar, 1):
            try:
                res = future.result()
                
                summary["processed_videos"] += res["processed"]
                if res["processed"] == 0:
                    summary["skipped_videos"] += 1
                
                summary["total_frames_saved"] += res["frames"]
                summary["total_landmarks_found"] += res["landmarks"]
                
                if i % 10 == 0 or i == len(futures):
                    pbar.set_postfix({
                        "✅ Success": f"{summary['processed_videos']}",
                        "⚠️ Skip": f"{summary['skipped_videos']}",
                        "📍 LMs": summary["total_landmarks_found"]
                    })
                
                if args.verbose and res["message"]:
                    pbar.write(res["message"])
                    
            except Exception as e:
                pbar.write(f"{r_}[CRITICAL ERROR]{rs_} Task failed: {e}")

    total_time = time.time() - start_time

    print(f"\n{g_}{s_}{'='*50}{rs_}")
    print(f"{g_}{s_}📊 Final Processing Summary Report{rs_}")
    print(f"{g_}{s_}{'='*50}{rs_}")
    print(f" 📂 Total Videos Found     : {summary['total_videos']}")
    print(f" ✅ Successfully Processed : {summary['processed_videos']}")
    print(f" ⚠️ Skipped (No Box/Fail)  : {r_}{summary['skipped_videos']}{rs_}")
    print(f" 🖼️  Total Face Crops Saved : {summary['total_frames_saved']}")
    print(f" 📍 Total Landmarks Saved  : {summary['total_landmarks_found']}")
    print(f" ⏱️  Total Time Taken       : {y_}{total_time:.2f}s{rs_}")
    
    if summary['processed_videos'] > 0:
        avg_time = total_time / summary['processed_videos']
        print(f" ⚡ Average Time per Video : {avg_time:.2f}s")
    
    print(f"{g_}{s_}{'='*50}{rs_}\n")
    
if __name__ == "__main__":
    main()