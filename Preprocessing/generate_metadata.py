import os, json
import random
import argparse
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
from tqdm import tqdm
from glob import glob

from .utils import get_original_with_fakes

import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

def get_paths(vid, label, root_dir):
    
    # ex) (ori_video, fake_video)
    ori_vid, fake_vid = vid
    # root_dir/crops/
    base_dir = os.path.join(root_dir, "crops")
    
    data = []
    
    # target_dir: root_dir/crops/video
    target_vid = ori_vid if label == 0 else fake_vid
    target_dir = os.path.join(base_dir, target_vid)
    
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"Target video directory not found: {target_dir}")
    
    frame_files = sorted(glob(os.path.join(target_dir, "*.png")))
    
    assert len(frame_files) > 0, f"[ERROR] No frame images found in directory: {target_dir}"
    
    for frame_path in frame_files:
        # root_dir/crops/video/{frame_idx}_{face_idx}.png
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

def filter_existing_pairs(ori_fakes, crops_dir):
    valid_videos = set(os.listdir(crops_dir))
    filtered = [(ori, fake) for ori, fake in ori_fakes if ori in valid_videos and fake in valid_videos]
    return filtered

def collect_metadata(pairs, label, root_dir, desc):
    func = partial(get_paths, label=label, root_dir=root_dir)
    metadata = []
    with Pool(processes=os.cpu_count()) as p:
        for result in tqdm(p.imap_unordered(func, pairs), total=len(pairs), desc=desc):
            if result:
                metadata.extend(result)
    return metadata
        
    
def main():
    parser = argparse.ArgumentParser(description="Generate CSV File and Move Frame into output directory")
    
    parser.add_argument("--root_dir", help="root directory")
    parser.add_argument("--output_dir", help="output(metadata) directory")
    parser.add_argument("--dataset", default="DFDC", choices=["DFDC"], help="type of deepfake dataset")
    
    args = parser.parse_args()    
    
    print(f"📂 Source dataset: {args.root_dir}")
    print(f"💾 Output dataset: {args.out_dir}")
    
    # ex) [(ori_video, fake_video), ...] 
    ori_fakes = get_original_with_fakes(args.root_dir)
    crop_dir = os.path.join(args.root_dir, "crops")
    
    ori_fakes = filter_existing_pairs(ori_fakes, crop_dir)
    ori_ori = set([(ori,ori) for ori, fake in ori_fakes]) 

    real_meta = collect_metadata(ori_ori, label=0, root_dir=args.root_dir, desc="Collecting REAL frames")
    fake_meta = collect_metadata(ori_fakes, label=1, root_dir=args.root_dir, desc="Collecting FAKE frames")
        
    print(f"[INFO] Total Real Frames: {len(real_meta)}, Total Fake Frames: {len(fake_meta)}")
    
    metadata = real_meta + fake_meta    
        
    data = []
    for img_path, label, ori_vid in metadata:
            
        path = Path(img_path)
        video = path.parent.name
        file = path.name
        frame_idx, face_idx = file.replace(".png", "").split("_")
        
        data.append([video, file, frame_idx, face_idx, label, ori_vid])

        """
        video: selected video name
        filename: {frame_idx}_{face_idx}.png
        frame_idx: 0, 1, ,2 ...
        face_idx: 0, 1 ...
        label: 0(Real), 1(Fake)
        ori_vid: asdfa
        """
        
        df = pd.DataFrame(data, columns=["video", "file","frame_idx","face_idx","label","ori_vid"]).to_csv(csv_path, index=False)
        df.sort_values(by=['video','frame_idx','face_idx'], inplace=True)
        
        os.makedirs(os.path.join(args.root_dir, args.out_dir), exist_ok=True)
        csv_path = os.path.join(args.out_dir, "train_metadata.csv")
        df.to_csv(csv_path, index=False)
        
        print(f"✅ [Summary] Total Videos: {df['video'].nunique()}, Total Frames: {len(df['file'])}")
        
        
if __name__ == "__main__":
    main()
    