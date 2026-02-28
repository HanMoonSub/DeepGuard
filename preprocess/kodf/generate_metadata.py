import os
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def generate_metadata(root_dir: str,
                      vid_col: str = "vid",
                      ori_vid_col: str = "ori_vid",
                      label_col: str = "label",
                      source_col: str = "source",
                      frame_col: str = "frame_idx",
                      video_meta: str = "train_metadata.csv",
                      frame_meta: str = "train_frame_metadata.csv",
                      ) -> None:
    """Generate Frame MetaData For Model Training"""
    
    root_path = Path(root_dir)
    crop_dir = root_path / "crops"
    
    print(f"[*] Scanning images in {crop_dir}...")
    frame_paths = list(crop_dir.rglob("*.png"))
    
    video_meta_path = root_path / video_meta
    if not video_meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {video_meta_path}")
    
    meta_df = pd.read_csv(video_meta_path)
    meta_dict = meta_df.set_index(vid_col).to_dict("index")
    meta_info = []
    
    for p in tqdm(frame_paths, total=len(frame_paths), desc="Generate Frame Metadata"):
        vid = p.parent.name
        frame_idx = p.stem
        
        if vid in meta_dict:
            info = meta_dict[vid]
            meta_info.append(
                {
                    vid_col: vid,
                    ori_vid_col: info[ori_vid_col],
                    label_col: info[label_col],
                    source_col: info[source_col],
                    frame_col: int(frame_idx),
                }
            )
        
    if meta_info:
        result_df = pd.DataFrame(meta_info)
        result_df = result_df.sort_values(by=[vid_col, frame_col])
        
        output_path = root_path / frame_meta
        result_df.to_csv(output_path, index=False)
        print(f"[+] Successfully saved frame metadata to: {output_path}")
    else:
        print("[!] No matching metadata found for the images.")

def main():
    parser = argparse.ArgumentParser(description="Generate Frame-level Metadata from cropped image and video-level csv")
    parser.add_argument("--root-dir", required=True, help = "Root Directory containing crops/ and metadata.csv")
    parser.add_argument("--vid-col", default='vid', type=str)
    parser.add_argument("--ori-vid-col", default='ori_vid', type=str)
    parser.add_argument("--label-col", default='label', type=str)
    parser.add_argument("--source-col", default='source', type=str)
    parser.add_argument("--frame-col", default='frame_idx', type=str)
    parser.add_argument("--video-meta", default='train_metadata.csv', type=str)
    parser.add_argument("--frame-meta", default='train_frame_metadata.csv', type=str)
    
    
    args = parser.parse_args()
    
    generate_metadata(**vars(args))
    
if __name__ == "__main__":
    main()