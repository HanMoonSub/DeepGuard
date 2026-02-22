import os
import time
import argparse
from pathlib import Path
from typing import List
import json
from tqdm import tqdm
from colorama import Fore, Style
from functools import partial
import multiprocessing as mp
from .utils import get_original_video_paths

c_ = Fore.BLUE
s_ = Style.BRIGHT
r_ = Style.RESET_ALL

def process_single_video(
                            video_path: Path,
                            root_dir: str,
                            conf_thres: float,
                            min_face_ratio: float,
                        ):
    from ..frame_extractor import FrameExtractor_KODF
    from ..face_detector import FaceDetector_KODF
    
    try:
        extractor = FrameExtractor_KODF()
        detector = FaceDetector_KODF(conf_thres, min_face_ratio)
        
        video_id = video_path.stem
        boxes_dir = Path(root_dir) / "boxes"
        json_path = boxes_dir / f"{video_id}.json"
        
        frame_dict = extractor.extract_frame(video_path)
        if not frame_dict:
            return {"status": "failed_extraction", "id": video_id}
        
        index, frame = next(iter(frame_dict.items()))
        result = detector.detect_single_face(frame)
        
        if result:
            with open(json_path, mode="w") as f:
                json.dump({int(index): result}, f)
            return {"status": "success", "id": video_id}
        else:
            return {"status": "no_face", "id": video_id}
        
    except Exception as e:
        return {"status": "error", "id": video_id, "msg": str(e)}
    
def process_videos(video_paths: List[Path],
                   root_dir: str,
                   conf_thres: float,
                   min_face_ratio: float):
    """
    Extracts frame from videos and performs face detection to save metadata
    
    This function iterates through a list of videos, extracts a specific number of using
    the provided Frame Extractor, and runs face detection via FaceDetector.
    The resulting bounding box coordinates are serialized into individual Json file
    for each video
    
    Args:
        video_paths(List[Path]): List of paths to the real video files
        root_dir(str): Root directory of the dataset
    """
    
    (Path(root_dir) / "boxes").mkdir(parents=True, exist_ok=True)
    
    num_workers = os.cpu_count() # If Colab Notebook, there is 2 cpu core
    worker_func = partial(
        process_single_video, 
        root_dir = root_dir,
        conf_thres = conf_thres,
        min_face_ratio = min_face_ratio)
    
    stats = {
        "total": len(video_paths),
        "success": 0, 
        "no_face": 0, 
        "failed_extraction": 0, 
        "error": 0
    }
    start_time = time.time()
    
    print(f"\n{s_}[*] CPU Multiprocessing: Using {num_workers} CPU Cores{r_}")
    
    with mp.Pool(num_workers) as pool:
        pbar = tqdm(pool.imap_unordered(worker_func, video_paths),
                    total=len(video_paths),
                    desc='Processing KODF Real Videos')
        for result in pbar:
            status = result['status']
            if status in stats:
                stats[status] += 1
            else:
                stats["error"] += 1
        
            pbar.set_postfix({"Success": stats["success"], "NoFace": stats["no_face"]})
    
    elapsed_time = time.time() - start_time
    print_summary(stats, elapsed_time)  
        
def print_summary(stats, elapsed_time):
    avg_time = elapsed_time / stats["total"] if stats["total"] > 0 else 0
    
    print("\n" + "="*50)
    print(f"       PROCESSING SUMMARY REPORT (CPU Parallel)")
    print("="*50)
    print(f" Total Videos Processed : {stats['total']}")
    print(f" [-] Successfully Saved : {stats['success']}")
    print(f" [-] Skipped (No Face)  : {stats['no_face']}")
    print(f" [-] Failed (Extract)   : {stats['failed_extraction']}")
    print(f" [-] Runtime Errors     : {stats['error']}")
    print("-" * 50)
    print(f" Total Elapsed Time     : {elapsed_time:.2f} sec")
    print(f" Average Time per Video : {avg_time:.2f} sec")
    print("="*50 + "\n")
    
def main():
    parser = argparse.ArgumentParser(description="Extract frame from videos and save Face Bounding Box metadata to JSON")
    parser.add_argument("--root-dir", required=True, help="Root Directory with KODF")
    parser.add_argument("--work-dir", required=True, help="Directory containing the video files to process")
    parser.add_argument("--conf-thres", default=0.5, type=float, help="Confidence threshold for YOLO face detection")
    parser.add_argument("--min-face-ratio", default=0.01, type=float, help="Minimum face bounding box area ratio")

    args = parser.parse_args()
    
    video_paths = get_original_video_paths(args.root_dir, args.work_dir)
    
    if not video_paths:
        print(f"{c_}[!] No videos found in the specified directory.{r_}")
        return
    
    process_videos(video_paths, args.root_dir, args.conf_thres, args.min_face_ratio)
    
if __name__ == "__main__":
    main()