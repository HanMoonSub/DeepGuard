import os
import time
import argparse
from pathlib import Path
from typing import List
import json
from tqdm import tqdm
from colorama import Fore, Style
from ..frame_extractor import FrameExtractor
from ..face_detector import FaceDetector
from .utils import get_original_video_paths

c_ = Fore.BLUE
s_ = Style.BRIGHT
r_ = Style.RESET_ALL

def process_videos(
                    video_paths: List[Path],
                    root_dir: str, 
                    frame_endpoints: List[str],
                    num_frames: int,
                    frame_extractor: FrameExtractor,
                    face_detector: FaceDetector, 
):  
    """
    Extracts frames from videos and performs batch face detection to save metadata
    
    This function iterates through a list of videos, extracts a specific number of using
    the provided Frame Extractor, and runs face detection via FaceDetector.
    The resulting bounding box coordinates are serialized into indivudal Json file
    for each video
    
    Args:
        video_paths(List[Path]): List of paths to the real video files
        root_dir(str): Root dictory of the dataset
        frame_endpoints(List[Path]): valid frame ranges
        num_frames(str): Number of frames to samples
    """
    
    boxes_dir = Path(root_dir) / "boxes"
    boxes_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "total_videos": len(video_paths),
        "success_saved": 0,
        "skipped_no_faces": 0,
        "failed_extraction": 0,
        "total_frames_ext": 0,
        "total_faces_det": 0,
    }
    
    start_time = time.time()
    
    pbar = tqdm(zip(video_paths, frame_endpoints), total=len(video_paths), desc="Detecting Faces")
    
    for i, (video_path, frame_endpoint) in enumerate(pbar):
        
        video_id = video_path.stem
        json_path = os.path.join(boxes_dir, f"{str(video_id)}.json")
        
        # (idx, (frame, scale_factor))
        frames_dict = frame_extractor.extract_frames_for_detect(video_path, num_frames, frame_endpoint)

        extracted_cnt = len(frames_dict)
        detected_cnt = 0
        
        stats["total_frames_ext"] += extracted_cnt
        
        if len(frames_dict) > 0:
            indices = list(frames_dict.keys())
            
            raw_values = list(frames_dict.values())
            frames = [val[0] for val in raw_values]
            scales = [val[1] for val in raw_values]
            
            results = face_detector.detect_batch(frames, scales)
            
            bbox_json = {}
            
            for idx, bbox in zip(indices, results):
                if bbox:
                    bbox_json[int(idx)] = bbox
                    detected_cnt +=1
                    
            stats["total_faces_det"] += detected_cnt
                    
            if bbox_json:
                with open(json_path, "w") as f:
                    json.dump(bbox_json, f)
                    stats["success_saved"] += 1
            else:
                stats["skipped_no_faces"] += 1
                pbar.write(f"{c_}{s_}[Skip] No faces detected in any frame: {video_id}{r_}")
        else:
            stats['failed_extraction'] += 1
            pbar.write(f"{c_}{s_}[Warning] No frames extracted: {video_id}{r_}")
        
        pbar.set_postfix({
            "Video Id": video_id,
            "Saved": f"{detected_cnt}/{extracted_cnt}"
        })
    
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / stats["total_videos"] if stats["total_videos"] > 0 else 0

    print("\n" + "="*50)
    print(f"       PROCESSING SUMMARY REPORT")
    print("="*50)
    print(f" Total Videos Processed : {stats['total_videos']}")
    print(f" [-] Successfully Saved : {stats['success_saved']}/{stats['total_videos']}")
    print(f" [-] Skipped (No Face)  : {stats['skipped_no_faces']}/{stats['total_videos']}")
    print(f" [-] Failed (No Frame)  : {stats['failed_extraction']}/{stats['total_videos']}")
    print("-" * 50)
    print(f" Total Frames Extracted : {stats['total_frames_ext']}/{stats['total_videos']*num_frames}")
    print(f" Total Faces Detected   : {stats['total_faces_det']}/{stats['total_videos']*num_frames}")
    print("-" * 50)
    print(f" Total Elapsed Time     : {elapsed_time:.2f} sec")
    print(f" Average Time per Video : {avg_time:.2f} sec")
    print("="*50 + "\n")
    
def main():
    parser = argparse.ArgumentParser(description="Extract frames from videos and save Face Bounding Box metadata to JSON")
    parser.add_argument("--root-dir", required=True, help="Root Directory with FaceForensics++")
    parser.add_argument("--num-frames", default=10, type=int, help="Number of frames to extract")
    parser.add_argument("--conf-thres", default=0.5, type=float, help="Confidence threshold for YOLO face detection")
    parser.add_argument("--min-face-ratio", default=0.01, type=float, help="Minimum face bounding box area ratio")
    parser.add_argument("--jitter", default=0, type=int, help="Random offset range for frame index")

    args = parser.parse_args()
    
    face_detector = FaceDetector(args.conf_thres, args.min_face_ratio)
    frame_extractor = FrameExtractor(args.jitter)
    
    video_paths, frame_endpoints = get_original_video_paths(args.root_dir)
    
    process_videos(video_paths, args.root_dir, frame_endpoints, args.num_frames,
                   frame_extractor, face_detector)
    
if __name__ == "__main__":
    main()