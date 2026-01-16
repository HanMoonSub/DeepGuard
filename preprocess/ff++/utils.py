from pathlib import Path
import pandas as pd


def get_original_video_paths(
                            root_dir: str,
                            csv_filename: str = "train_metadata.csv"):
    
    """
    Retrieves a list of file paths for original (REAL) videos based on metadata.
    
    Args:
        root_dir (str): The base directory containing the metadata and video folders.
        csv_filename (str): The name of the metadata CSV file.
        
    Returns:
        list: A list of Path objects pointing to the .mp4 files.
    """
    
    root_path = Path(root_dir)
    csv_path = root_path / csv_filename
    
    if not csv_path.exists():
        raise FileExistsError(f"Metadata file not found at: {csv_path}")

    meta_df = pd.read_csv(csv_path, dtype={'ori_vid': str})
    min_frames = meta_df.groupby(['ori_vid'])['frame_cnt'].min() 
    
    real_videos = meta_df[meta_df['label'] == 'REAL'].copy()
    real_videos['min_frame_cnt'] = real_videos['vid'].map(min_frames)
    
    if real_videos.empty:
        print("Warning: No entries with label 'REAL' were found.")
        return [], []
    
    
    video_list = real_videos.apply(
        lambda x: root_path / x['source'] / f"{x['vid']}.mp4", 
        axis=1).tolist()
        
    frame_cnt_list = real_videos['min_frame_cnt'].tolist()
    
    return video_list, frame_cnt_list

def get_video_paths(
                    root_dir: str,
                    csv_filename: str = "train_metadata.csv"
):
    root_path = Path(root_dir)
    csv_path = root_path / csv_filename
    
    if not csv_path.exists():
        raise FileExistsError(f"Metadata file not found at: {csv_path}")

    meta_df = pd.read_csv(csv_path, dtype={'ori_vid': str})
    
    video_list = meta_df.apply(
        lambda x: root_path / x['source'] / f"{x['d_vid']}.mp4", 
        axis=1).tolist()
    
    original_video_list = meta_df.apply(
        lambda x: root_path / 'original' / f"{x['ori_vid']}.mp4", 
        axis=1).tolist()
    
    return video_list, original_video_list