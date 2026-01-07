import os 
from os import cpu_count
import argparse
import time
import numpy as np
from tqdm import tqdm
from pathlib import Path
from multiprocessing.pool import Pool
from functools import partial
from typing import List, Tuple
import json
import cv2
from colorama import Fore, Style
from .utils import get_video_paths
from .frame_extractor import FrameExtractor

c_ = Fore.BLUE
g_ = Fore.GREEN
r_ = Fore.RED
s_ = Style.BRIGHT
rs_ = Style.RESET_ALL

def process_crop(video_pair, root_dir, margin_ratio, margin_jitter):
    """
    Returns:
        bool: True if processing was successful and frames were saved, False otherwise.
    """
    video_path, ori_video_path = video_pair
    
    try:
        root_path = Path(root_dir)
        boxes_dir = root_path / "boxes"
        crops_dir = root_path / "crops"
        
        video_id = video_path.stem
        ori_video_id = ori_video_path.stem
        
        frame_extractor = FrameExtractor(jitter=0)
         
        json_path = boxes_dir / f"{ori_video_id}.json"
            
        if json_path.exists():   
            with open(json_path, "r") as f:
                ori_box_dict = json.load(f)
                    
            ori_frame_indices = sorted([int(x) for x in list(ori_box_dict.keys())])
        
            frames_dict = frame_extractor.extract_frames_for_crop(str(video_path), ori_frame_indices)                    
            save_video_dir = crops_dir / video_id
                
            if len(frames_dict) > 0: 
                save_video_dir.mkdir(parents=True, exist_ok=True)
        
                for idx, frame in frames_dict.items():
                    if str(idx) not in ori_box_dict: continue

                    ori_box = ori_box_dict[str(idx)]
                    
                    xmin, ymin, xmax, ymax = ori_box 
                        
                    current_margin = margin_ratio
                    if margin_jitter > 0:
                        jitter_scale = margin_ratio * margin_ratio
                        noise = np.random.uniform(-jitter_scale, jitter_scale)
                        current_margin = max(0.0, margin_ratio + noise)
                        
                    w = xmax - xmin
                    h = ymax - ymin
                    pad_w = int(w * current_margin)
                    pad_h = int(h * current_margin)
                        
                    y1 = max(int(ymin - pad_h), 0)
                    y2 = min(int(ymax + pad_h), frame.shape[0])
                    x1 = max(int(xmin - pad_w), 0)
                    x2 = min(int(xmax + pad_w), frame.shape[1])
                        
                    crop = frame[y1:y2, x1:x2]
                    
                    save_path = save_video_dir / f"{idx}.png"
                    cv2.imwrite(str(save_path), crop)
                
                return True 
            else:
                return False # No Extract Frame
        else:
            return False # No Json File
            
    except Exception as e:
        return False


def main():
    parser = argparse.ArgumentParser(description="Crop Face and save")
    parser.add_argument("--root-dir", required=True, help="Root Directory with Celeb DF(V2)")
    parser.add_argument("--margin-ratio", default=0.2, type=float, help="Margin Ratio for Detected Face")
    parser.add_argument("--margin-jitter", default=0.2, type=float, help="noise for margin ratio")
    
    args = parser.parse_args()
    
    Path(args.root_dir).joinpath("crops").mkdir(parents=True, exist_ok=True)

    print(f"{c_}{s_}[Info] Scanning video paths...{rs_}")
    video_paths, ori_video_paths = get_video_paths(args.root_dir)
    params = list(zip(video_paths, ori_video_paths))
    total_videos = len(params)
    
    print(f"{c_}{s_}[Info] Found {total_videos} videos. Starting multiprocessing...{rs_}")
    
    start_time = time.time()
    
    success_count = 0
    fail_count = 0
    
    with Pool(processes=cpu_count()) as p:
        with tqdm(total=total_videos, desc="Cropping Faces", unit="vid") as pbar:
        
            for result in p.imap_unordered(
                partial(
                    process_crop,
                    root_dir = args.root_dir,
                    margin_ratio = args.margin_ratio,
                    margin_jitter = args.margin_jitter
                ),
                params
            ):
       
                if result:
                    success_count += 1
                else:
                    fail_count += 1
                
                pbar.set_postfix({"Success": success_count, "Fail": fail_count}, refresh=False)
                pbar.update()
                
    end_time = time.time()
    total_duration = end_time - start_time
    avg_speed = total_videos / total_duration if total_duration > 0 else 0

    print("\n" + "="*50)
    print(f"{s_}PROCESSING SUMMARY{rs_}")
    print("="*50)
    print(f"Total Videos   : {total_videos}")
    print(f"Success        : {g_}{success_count}{rs_}")
    print(f"Failed/Skipped : {r_}{fail_count}{rs_}")
    print("-" * 50)
    print(f"Total Time     : {total_duration:.2f} sec")
    print(f"Average Speed  : {c_}{avg_speed:.2f} videos/sec{rs_}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()