import pandas as pd
from pathlib import Path

def get_original_video_paths(
                            root_dir: str,
                            work_dir: str = None, 
                            csv_filename: str = "train_metadata.csv",
                            label_col: str = 'label'
                            ):
    
    """
    Retrieves a list of file paths for original (REAL) videos based on metadata.
    
    Args:
        root_dir (str): The base directory containing the metadata and video folders.
        work_dir (str): The directory to search for video files. If None, defaults to root_dir.
        csv_filename (str): The name of the metadata CSV file.
        label_col (str): The Column name of Label of the metadat CSV file
        
    Returns:
        list: A list of Path objects pointing to the .mp4 files.
    """
    
    root_path = Path(root_dir)
    work_dir = work_dir or root_dir
    work_path = Path(work_dir)
    
    csv_path = root_path / csv_filename
    
    if not csv_path.exists():
        raise FileExistsError(f"Metadata file not found at: {csv_path}")

    meta_df = pd.read_csv(csv_path)
    
    real_videos = meta_df[meta_df[label_col] == 'REAL'].copy()
    
    if real_videos.empty:
        print("Warning: No entries with label 'REAL' were found.")
        return []
    
    real_paths = work_path.rglob("*.mp4")
    real_paths_dict = {p.stem: p for p in real_paths}
    
    video_list = real_videos['vid'].map(real_paths_dict).dropna().tolist()
        
    return sorted(set(video_list))

def get_video_paths(
                    root_dir: str,
                    work_dir: str, 
                    csv_filename: str = "train_metadata.csv"):
    
    root_path = Path(root_dir)
    work_dir = work_dir or root_dir
    work_path = Path(work_dir)
    csv_path = root_path / csv_filename
    
    if not csv_path.exists():
        raise FileExistsError(f"Metadata file not found at: {csv_path}")

    meta_df = pd.read_csv(csv_path).copy()
     
    video_paths = work_path.rglob("*.mp4")
    video_paths_dict = {p.stem: p for p in video_paths}
    
    mask = meta_df['vid'].isin(video_paths_dict.keys())
    filtered_df = meta_df[mask].copy()
    
    video_list = filtered_df['vid'].map(video_paths_dict).tolist()
    original_video_list = filtered_df['ori_vid'].tolist()
    
    return video_list, original_video_list