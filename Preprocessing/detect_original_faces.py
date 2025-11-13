import os, json
from os import cpu_count
import argparse
from typing import Type

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from colorama import Fore, Style

from . import face_detector
from .face_detector import VideoFaceDetector, VideoDataset
from .utils import get_original_video_paths

def process_videos(videos, root_dir, device, batch_size, apply_clahe, detector_cls: Type[VideoFaceDetector]):

    detector = face_detector.__dict__[detector_cls](device=device, batch_size=batch_size)
    
    dataset = VideoDataset(videos, apply_clahe) # Get List of PIL.Image(x0.5 Rescaling with Original Frame)
    
    loader = DataLoader(dataset, shuffle=False, num_workers=cpu_count(), batch_size=1, collate_fn=lambda x: x)
    
    for item in tqdm(loader, total=len(loader)):
        result = {}
        video, indices, frames = item[0]
        
        batches = [
            (indices[i:i + detector._batch_size], frames[i:i + detector._batch_size])
            for i in range(0, len(frames), detector._batch_size)
        ]
        
        total_boxes = 0
        total_frames = len(frames)
        no_face_frames = 0
        
        for batch_indices, frames_batch in batches:
            batch_boxes = detector._detect_faces(frames_batch)

            for idx, b in zip(batch_indices, batch_boxes):
                result[int(idx)] = b
                if b is None:
                    no_face_frames += 1
                else:
                    total_boxes += len(b)
                 
        id = os.path.splitext(os.path.basename(video))[0]
        out_dir = os.path.join(root_dir, "boxes")
        os.makedirs(out_dir, exist_ok = True)
        json_path = os.path.join(out_dir, f"{id}.json")
        
        # Skip saving if no faces detected in any frame
        if no_face_frames == total_frames:
            print(
                f"{Fore.YELLOW}{Style.BRIGHT}[Warning]{Style.RESET_ALL} "
                f"Video: {id} — No faces detected in any frame. Skipping save."
            )
            continue

        # Save JSON result
        with open(json_path, "w") as f:
            json.dump(result, f)

        print(
            f"{Fore.BLUE}{Style.BRIGHT}[Info]{Style.RESET_ALL} "
            f"Video: {id} | Total frames: {total_frames}, "
            f"Detected boxes: {total_boxes}, No-face frames: {no_face_frames} "
            f"[Saved] {json_path}"
        )
        
def main():
    
    parser = argparse.ArgumentParser(
        description="Process a original videos with face detector"
    )
    parser.add_argument("--root_dir", help='root directory')
    parser.add_argument("--detector_type", help='type of the detector', default="FacenetDetector",
                        choices=["FacenetDetector"])
    parser.add_argument("--batch_size", default=32, type=int, help="Face Detector Batch Size")
    parser.add_argument("--apply_clahe", default=False, type=bool, help="Apply clahe for find calibrated bbox coordinates")
    args = parser.parse_args()
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # originals: real video path(ex: root_dir/video_folder/video.mp4)
    originals = get_original_video_paths(args.root_dir)
    # return: bbox json(ex: root_dir/boxes/video.json)
    process_videos(originals, args.root_dir, device, args.batch_size, args.apply_clahe, args.detector_type)
    
if __name__ == "__main__":
    main()
    