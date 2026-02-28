import os
import argparse
import pandas as pd
from pathlib import Path
import json

base_dir = Path(__file__).resolve().parent
label2id = {
    'Deepfakes': '0_',
    'Face2Face': '1_',
    'FaceShifter': '2_',
    'FaceSwap': '3_',
    'NeuralTextures': '4_'
}

def generate_metadata(root_dir: str) -> pd.DataFrame:
    
    """
    FaceForensics++ metadata 
    Columns:
        label(str): Ground truth label ('FAKE' or 'TRUE')
        frame_cnt(int): Total number of frames in the video
        d_vid(str): Duplicated Video name for each deepfake model
        source(str): Deepfake model name(e.g., FaceSwap, Face2Face)
        ori_vid: Video name used for manipulation
        ori_frame_cnt: Total number of frames in the original video
        vid(str): Unique Video name
    """

    df = pd.read_csv(os.path.join(root_dir, "csv/FF++_Metadata.csv"))
    df = df[['File Path', 'Label', 'Frame Count']].rename(columns={'Frame Count': 'frame_cnt', 'Label': 'label'})

    df['d_vid'] = df['File Path'].apply(lambda x: str(Path(x).stem))
    df['source'] = df['File Path'].apply(lambda x: str(Path(x).parent.name))
    df['ori_vid'] = df['d_vid'].apply(lambda x: x.split("_")[0].zfill(3))

    df = df[df['source'] != "DeepFakeDetection"].drop(columns=['File Path']).reset_index(drop=True)
    
    ref_df = df[['d_vid', 'frame_cnt']].drop_duplicates('d_vid')
    df = df.merge(
        ref_df.rename(columns={'d_vid': 'ori_vid', 'frame_cnt': 'ori_frame_cnt'}),
        on='ori_vid',
        how='left'
    )
    
    df['ori_vid'] = df['ori_vid'].astype(str)
    
    df['vid'] = df.apply(
        lambda x: label2id.get(x['source'], '') + x['d_vid'], 
        axis=1)
    
    return df

def filter_metadata(meta_df, json_data):
    vid_list = []
    
    for item in json_data:
        source, vid = item.split("/") 
        
        prefix_vid = label2id.get(source, '') + vid
        vid_list.append(prefix_vid)
    
    return meta_df[meta_df['vid'].isin(vid_list)].reset_index(drop=True)
    

def split_metadata(meta_df: pd.DataFrame, root_dir: str):
    """
    Splits the metadata into train and test sets based on JSON split files.
    
    Args:
        meta_df: The DataFrame returned by generate_metadata()
    Returns:
        train_meta, test_meta: Filtered DataFrames
    """
    
    train_json_path = base_dir / "train.json"
    test_json_path = base_dir / "test.json"
    
    with open(train_json_path, "r") as f:
        train_json = json.load(f)
        
    with open(test_json_path, "r") as f:
        test_json = json.load(f)
    
    train_meta = filter_metadata(meta_df, train_json)
    test_meta = filter_metadata(meta_df, test_json)
    
    train_meta.to_csv(os.path.join(root_dir, "train_metadata.csv"), index=False)
    test_meta.to_csv(os.path.join(root_dir, "test_metadata.csv"), index=False)
    
    return train_meta, test_meta

def main():
    parser = argparse.ArgumentParser(description="Generating Train, Test MetaData")
    parser.add_argument("--root-dir", required=True, help="Root Directory with FF++")
    parser.add_argument("--print-info", default=True, type=bool, help="Display MetaData")
    
    args = parser.parse_args()
    
    meta_df = generate_metadata(args.root_dir)
    train_meta, test_meta = split_metadata(meta_df, args.root_dir)
    
    if args.print_info:
        print("\n" + "="*50)
        print("🔍 FaceForensics++ Metadata Summary")
        print("="*50)
        
        print(f"✅ Total processed videos: {len(meta_df)}")
        print(f"📂 Train set: {len(train_meta)} videos")
        print(f"📂 Test set:  {len(test_meta)} videos")
        
        print("\n📊 Distribution by Source (Train):")
        print(train_meta['source'].value_counts())
        
        print("\n📊 Distribution by Label (Train):")
        print(train_meta['label'].value_counts())
        
        print("\n📝 Sample Data (Train Set - Top 5):")
        cols_to_show = ['vid', 'label', 'source', 'frame_cnt', 'ori_vid']
        print(train_meta[cols_to_show].head())
        
        print("="*50)
        print(f"🚀 Metadata CSV files saved to: {args.root_dir}")
        print("="*50 + "\n")
    
if __name__ == "__main__":
    main()