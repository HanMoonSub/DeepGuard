import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from colorama import Fore, Style
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, 
    PrecisionRecallDisplay, RocCurveDisplay
    )
from sklearn.metrics import (
    roc_auc_score, accuracy_score, 
    recall_score, precision_score, f1_score, 
    log_loss
)
from .video_predictor import VideoPredictor

c_ = Fore.BLUE
g_ = Fore.GREEN
r_ = Fore.RED
y_ = Fore.YELLOW
s_ = Style.BRIGHT
rs_ = Style.RESET_ALL

def main():
    parser = argparse.ArgumentParser(description="Predict DeepFake Video")
    # --- Data & Path Settings ---
    parser.add_argument("--root-dir", required=True, help="Root directory containing the dataset files")
    parser.add_argument("--label-col", default='label', type=str, help="Column name for target labels")
    parser.add_argument("--source-col", default='source', type=str, help="Column name for video source/folder")
    parser.add_argument("--video-col", default='vid', type=str, help="Column name for video filenames")
    parser.add_argument("--dataset", default='celeb_df_v2', type=str, help="Target dataset for evaluation (e.g., celeb_df_v2, ff++, kodf)")

    # --- Model & Detection Hyperparameters ---
    parser.add_argument("--margin-ratio", default=0.2, type=float, help="Margin ratio around the detected face crop")
    parser.add_argument("--conf-thres", default=0.5, type=float, help="Confidence threshold for face detection")
    parser.add_argument("--min-face-ratio", default=0.01, type=float, help="Minimum face-to-frame size ratio to process")
    parser.add_argument("--model-name", default='ms_eff_vit_b0', type=str, help="Name of the model architecture")
    parser.add_argument("--model-dataset", default='celeb_df_v2', type=str, help="Dataset used for model pre-training")

    # --- Inference Strategy ---
    parser.add_argument("--num-frames", default=20, type=int, help="Number of frames to sample per video")
    parser.add_argument("--agg-mode", default='conf', type=str, help="Aggregation method for frame-level predictions")
    parser.add_argument("--tta-hflip", default=0.0, type=float, help="Probability of horizontal flip for Test-Time Augmentation (TTA)")
    
    # --- Display Result ---
    parser.add_argument("--save-result", default=True, type=bool, help="Save figure file")
    
    args = parser.parse_args()
    
    
    
    print(f"\n🚀 {s_}{c_}[1/4] 데이터셋 준비 중...{rs_} (Target: {y_}{args.dataset}{rs_})")
    
    if args.dataset == "celeb_df_v2":
        meta_df = pd.read_csv(os.path.join(args.root_dir, "Celeb_DF(v2)", "test_metadata.csv"))
        meta_df[args.label_col] = meta_df[args.label_col].map({"REAL": 0, "FAKE": 1})
        meta_df['video_path'] = meta_df.apply(lambda x: os.path.join(args.root_dir, "Celeb_DF(v2)", f"{x[args.source_col]}", f"{x[args.video_col]}.mp4"), axis=1)
    
    if args.dataset == "ff++":
        meta_df = pd.read_csv(os.path.join(args.root_dir, "FF++", "test_metadata.csv"))
        meta_df[args.label_col] = meta_df[args.label_col].map({"REAL": 0, "FAKE": 1})
        mask = meta_df[args.source_col] != "original"
        meta_df.loc[mask, args.video_col] = meta_df.loc[mask, args.video_col].str[2:]
        meta_df['video_path'] = meta_df.apply(lambda x: os.path.join(args.root_dir, "FF++",  f"{x[args.source_col]}", f"{x[args.video_col]}.mp4"), axis=1)   
    
    if args.dataset == "kodf":
        meta_df = pd.read_csv(os.path.join(args.root_dir, "kodf", "test_metadata.csv"))
        meta_df[args.label_col] = meta_df[args.label_col].map({"REAL": 0, "FAKE": 1})
        meta_df['pid'] = meta_df[args.video_col].apply(lambda x: x.split("_")[0])
        meta_df['video_path'] = meta_df.apply(lambda x: os.path.join(args.root_dir, "kodf",  x[args.source_col], x["pid"], f"{x[args.video_col]}.mp4"), axis=1)   
    
    
    print(f"\n🤖 {s_}{c_}[2/4] 모델 로딩 중...{rs_} ({g_}{args.model_name}{rs_})")
    video_predictor = VideoPredictor(args.margin_ratio, args.conf_thres, args.min_face_ratio,
                   args.model_name, args.model_dataset)
    
    print(f"\n🔍 {s_}{c_}[3/4] 비디오 분석 시작{rs_} (프레임: {y_}{args.num_frames}{rs_}, 집계: {y_}{args.agg_mode}{rs_}, TTA: {y_}{args.tta_hflip}{rs_})")
    all_pred = []
    all_true = meta_df[args.label_col]
    
    for video_path in tqdm(meta_df['video_path'], total=len(meta_df), desc=f"Test Evaluation (Data: {args.dataset} | Model: {args.model_dataset})"):
        pred = video_predictor.predict_video(video_path, args.num_frames, args.agg_mode, args.tta_hflip)
        all_pred.append(pred)
        
    all_pred = np.array(all_pred)
    all_true = np.array(all_true)
    
    print(f"\n📊 {s_}{c_}[4/4] 최종 메트릭 계산 중...{rs_}")
    y_pred_bin = (all_pred >= 0.5).astype(int)
    
    auc = roc_auc_score(all_true, all_pred)
    acc = accuracy_score(all_true, y_pred_bin)
    loss = log_loss(all_true, all_pred)
    
    # REAL(0) recall, precision, f1
    real_recall = recall_score(all_true, y_pred_bin, pos_label=0)
    real_precision = precision_score(all_true, y_pred_bin, pos_label=0)
    real_f1 = f1_score(all_true, y_pred_bin, pos_label=0)
    
    # FAKE(1) recall, precision, f1
    fake_recall = recall_score(all_true, y_pred_bin, pos_label=1)
    fake_precision = precision_score(all_true, y_pred_bin, pos_label=1)
    fake_f1 = f1_score(all_true, y_pred_bin, pos_label=1)
    
    print(f"{s_}{y_}" + "=" * 65 + f"{rs_}")
    print(f"🏆 {s_}최종 평가 결과 (Final Metrics){rs_}")
    print(f"{s_}{y_}" + "-" * 65 + f"{rs_}")

    print(f"   Overall  ▶  {s_}AUC:{rs_} {g_}{auc:.4f}{rs_}  |  {s_}ACC:{rs_} {g_}{acc * 100:.2f}%{rs_}  |  {s_}Log LOSS:{rs_} {g_}{loss:.4f}{rs_}")
    print(f"{s_}{y_}" + "-" * 65 + f"{rs_}")

    print(f"   CATEGORY  │  PRECISION  │   RECALL     │     F1    ")
    print(f"   ----------│-------------│-------------│-----------")
    print(f"   {s_}REAL (0){rs_}  │   {real_precision:.4f}     │   {real_recall:.4f}     │   {real_f1:.4f}")
    print(f"   {s_}FAKE (1){rs_}  │   {fake_precision:.4f}     │   {fake_recall:.4f}     │   {fake_f1:.4f}")

    print(f"{s_}{y_}" + "=" * 65 + f"{rs_}")
    print(f"✨ {s_}{g_}모든 분석이 완료되었습니다!{rs_}\n")
    
    if args.save_result:
        plt.style.use('seaborn-v0_8-whitegrid')
        
        fig, ax = plt.subplots(1, 3, figsize=(20, 6))
        
        # [1] Confusion Matrix
        cm = confusion_matrix(all_true.astype(int), (all_pred >= 0.5).astype(int), labels=[0,1], normalize='true')
        cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Real', 'Fake'])
        cm_disp.plot(ax=ax[0], cmap='Blues', colorbar=False)
        ax[0].set_title("Confusion Matrix")
        
        # [2] ROC Curve
        RocCurveDisplay.from_predictions(all_true, all_pred, pos_label=0, ax=ax[1], name="Class 0 (Real)", color='royalblue')
        RocCurveDisplay.from_predictions(all_true, all_pred, pos_label=1, ax=ax[1], name="Class 1 (Fake)", color='darkorange')
        ax[1].set_title("ROC Curve (Real vs Fake)")
        ax[1].grid(True, linestyle='--', alpha=0.5)
        
        # [3] Precision-Recall Curve
        PrecisionRecallDisplay.from_predictions(all_true, all_pred, pos_label=0, ax=ax[2], name="Class 0 (Real)", color='royalblue')
        PrecisionRecallDisplay.from_predictions(all_true, all_pred, pos_label=1, ax=ax[2], name="Class 1 (Fake)", color='darkorange')
        ax[2].set_title("Precision-Recall Curve (Real vs Fake)")
        ax[2].grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        save_path = f"result_{args.dataset}_{args.model_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"📈 시각화 결과가 저장되었습니다: {s_}{g_}{save_path}{rs_}")
        
if __name__ == "__main__":
    main()