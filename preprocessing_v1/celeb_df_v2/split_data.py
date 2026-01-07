import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from pathlib import Path
from typing import List
import argparse        
from colorama import Fore, Style

c_ = Fore.BLUE
s_ = Style.BRIGHT
r_ = Style.RESET_ALL
  
def generate_metadata(root_dir: str) -> pd.DataFrame:
    """
    Generates metadata DataFrame for the Celeb-DF dataset.
    
    Args:
        root_dir (str): Root directory path of the dataset.
        print_info (bool): Whether to print video counts.
        
    Returns:
        pd.DataFrame: Metadata containing ['label', 'source', 'vid', 'origin_vid']
    """
    
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Directory not found: {root_path}")
    
    target_dirs = {
            "Celeb-real": "REAL",
            "Celeb-synthesis": "FAKE",
            "YouTube-real": "REAL"
        }
    
    video_dict = []
    for folder_name, label_type in target_dirs.items():
        folder_path = root_path / folder_name
        
        if not folder_path.exists():
            print(f"Warning: '{folder_name}' folder not found in {root_dir}")
            continue
        
        video_files = list(folder_path.rglob("*.mp4"))
        
        print(f"Total {folder_name} Videos: {len(video_files)}")
        
        for file_path in tqdm(video_files, total=len(video_files), desc=f"Processing {folder_name}"):
            vid_name = file_path.stem
            
            if folder_name == 'Celeb-synthesis':
                parts = vid_name.split("_")
                origin_vid = f"{parts[0]}_{parts[-1]}"
            else:
                origin_vid = vid_name
                
            video_dict.append({
                "label": label_type,       # REAL / FAKE
                "source": folder_name,     # Celeb-real, Celeb-synthesis, etc.
                "vid": vid_name,           # Video Name
                "origin_vid": origin_vid,  # Target Video name
            })
        
    return pd.DataFrame(video_dict)
    
def get_test_vids(root_dir: str) -> List:
    test_path = Path(root_dir) / "List_of_testing_videos.txt"

    if not test_path.exists():
        raise FileNotFoundError(f"testing_videos.txt not found in {root_dir}")
        
    with open(test_path, "r") as f:
        test_list = f.read()

    test_vids =[]
        
    for i, line in enumerate(test_list.strip().splitlines()):
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            print(f"Warning: Line {i} has wrong format!!\n")
            print(f"=> {line}")
            continue
        else:
            _, video = parts
            test_vid = Path(video).stem
            test_vids.append(test_vid)
        
    return test_vids 
    
def split_metadata(meta_df: pd.DataFrame, root_dir: str, test_vids: List):
        
    train_meta = meta_df[~meta_df['vid'].isin(test_vids)].reset_index(drop=True)
    
    # Remove Orphan Fake Video
    valid_real_ids = set(train_meta[train_meta['label'] == 'REAL']['origin_vid'])
    condition = (train_meta['label'] == 'REAL') | (train_meta['origin_vid'].isin(valid_real_ids))
    train_meta = train_meta[condition].copy().reset_index(drop=True)
    
    test_meta = meta_df[meta_df['vid'].isin(test_vids)].reset_index(drop=True)
        
    train_meta.to_csv(os.path.join(root_dir, "train_metadata.csv"), index=False)
    test_meta.to_csv(os.path.join(root_dir, "test_metadata.csv"), index=False)

    return train_meta, test_meta

def main():
    parser = argparse.ArgumentParser(description="Generating Train, Test MetaData")
    parser.add_argument("--root-dir", required=True, help="Root Directory with Celeb DF(V2)")
    parser.add_argument("--print-info", default=True, type=bool, help="Display MetaData")
    
    args = parser.parse_args()
    
    print(f"{c_}{s_}1. Generating Total MetaData... {r_}")
    total_metadata = generate_metadata(args.root_dir)
    
    print(f"\n{c_}{s_}2. Reading Test Video List...{r_}")
    test_vids = get_test_vids(args.root_dir)
    
    print(f"\n{c_}{s_}3. Splitting and Saving...{r_}")
    train_meta, test_meta = split_metadata(total_metadata, args.root_dir, test_vids)
    
    if args.print_info:
        print(f"\n{c_}{s_}[INFO] Dataset Statistics{r_}")
        
        for name, meta in [("Train", train_meta), ("Test", test_meta)]:
            n_total = len(meta)
            n_celeb_real = len(meta[meta['source'] == 'Celeb-real'])
            n_yt_real = len(meta[meta['source'] == 'YouTube-real'])
            n_fake = len(meta[meta['source'] == 'Celeb-synthesis'])
            
            print(f"\n🔹 {name} Set (Total: {n_total})")
            print(f"   - Real (Celeb): {n_celeb_real}")
            print(f"   - Real (YouTube): {n_yt_real}")
            print(f"   - Fake (Synthesis): {n_fake}")
            
            if n_total > 0:
                real_ratio = (n_celeb_real + n_yt_real) / n_total * 100
                print(f"   - Real/Fake Ratio: {real_ratio:.1f}% / {100-real_ratio:.1f}%")
                
if __name__ == "__main__":
    main()

    