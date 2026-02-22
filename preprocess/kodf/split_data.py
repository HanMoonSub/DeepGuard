import os
import argparse
import pandas as pd

def generate_train_metadata(root_dir: str) -> pd.DataFrame:
    """
    Build kodf(Korean Deepfake Detection Dataset) train metadata.csv
    """
    ori_path = os.path.join(root_dir, "원본영상_training_메타데이터.csv")
    syn_path = os.path.join(root_dir, "변조영상_training_메타데이터.csv")
    
    ori_train_df = pd.read_csv(ori_path)
    syn_train_df = pd.read_csv(syn_path)
    
    ori_train_df = ori_train_df[['영상ID']].rename(columns={'영상ID': 'vid'})
    ori_train_df['ori_vid'] = ori_train_df['vid']
    ori_train_df['label'] = 'REAL'
    ori_train_df['source'] = 'original'
    
    syn_train_df = syn_train_df[['영상ID', '타겟영상', '변조모델']]
    syn_train_df = syn_train_df.rename(columns={
        '영상ID': 'vid', 
        '타겟영상': 'ori_vid', 
        '변조모델': 'source'
    })
    syn_train_df['label'] = 'FAKE'
    
    train_df = pd.concat([ori_train_df, syn_train_df], ignore_index=True)
    
    train_df['vid'] = train_df['vid'].apply(lambda x: x.split(".")[0])
    train_df['ori_vid'] = train_df['ori_vid'].apply(lambda x: x.split(".")[0])
    
    return train_df

def generate_test_metadata(root_dir: str) -> pd.DataFrame:
    """
    Build kodf(Korean Deepfake Detection Dataset) test metadata.csv
    """
    ori_path = os.path.join(root_dir, "원본영상_validation_메타데이터.csv")
    syn_path = os.path.join(root_dir, "변조영상_validation_메타데이터.csv")
    
    ori_test_df = pd.read_csv(ori_path)
    syn_test_df = pd.read_csv(syn_path)
    
    ori_test_df = ori_test_df[['영상ID']].rename(columns={'영상ID': 'vid'})
    ori_test_df['ori_vid'] = ori_test_df['vid']
    ori_test_df['label'] = 'REAL'
    ori_test_df['source'] = 'original'
    
    syn_test_df = syn_test_df[['영상ID', '타겟영상', '변조모델']]
    syn_test_df = syn_test_df.rename(columns={
        '영상ID': 'vid', 
        '타겟영상': 'ori_vid', 
        '변조모델': 'source'
    })
    syn_test_df['label'] = 'FAKE'
    
    test_df = pd.concat([ori_test_df, syn_test_df], ignore_index=True)
    
    test_df['vid'] = test_df['vid'].apply(lambda x: x.split(".")[0])
    test_df['ori_vid'] = test_df['ori_vid'].apply(lambda x: x.split(".")[0])
    
    return test_df

def main():
    parser = argparse.ArgumentParser(description="Generating Train, Test MetaData")
    parser.add_argument("--root-dir", required=True, help="Root Directory with kodf")
    parser.add_argument("--print-info", default=True, type=bool, help="Display MetaData")
    
    args = parser.parse_args()
    
    train_meta = generate_train_metadata(args.root_dir)
    test_meta = generate_test_metadata(args.root_dir)
    
    train_meta.to_csv(os.path.join(args.root_dir, "train_metadata.csv"), index=False)
    test_meta.to_csv(os.path.join(args.root_dir, "test_metadata.csv"), index=False)
    
    if args.print_info:
        print("\n" + "="*50)
        print("🔍 FaceForensics++ Metadata Summary")
        print("="*50)
        
        print(f"📂 Train set: {len(train_meta)} videos")
        print(f"📂 Test set:  {len(test_meta)} videos")
        
        print("\n📊 Distribution by Source (Train):")
        print(train_meta['source'].value_counts())
        
        print("\n📊 Distribution by Label (Train):")
        print(train_meta['label'].value_counts())
        
        print("\n📊 Distribution by Source (Test):")
        print(test_meta['source'].value_counts())
        
        print("\n📊 Distribution by Label (Test):")
        print(test_meta['label'].value_counts())
        
        print("\n📝 Sample Data (Train Set - Top 5):")
        cols_to_show = ['vid', 'label', 'source', 'ori_vid']
        print(train_meta[cols_to_show].head())
        
        print("="*50)
        print(f"🚀 Metadata CSV files saved to: {args.root_dir}")
        print("="*50 + "\n")
    
if __name__ == "__main__": 
    main()