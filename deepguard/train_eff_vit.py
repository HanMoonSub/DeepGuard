import os
import sys
from pathlib import Path
import re
import wandb
import argparse
import numpy as np
import pandas as pd
from typing import List
import webbrowser
from omegaconf import OmegaConf
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .data import split_data, class_imbalance_handle
from .data import CutMixDeepFakeDataset, get_train_transforms, get_valid_transforms
from .utils import seed_everything, AverageMeter, build_metrics, EarlyStopping
from .optimizer import build_ms_eff_vit_optimizer, add_to_optim_groups
from .scheduler import fetch_scheduler
from .models import MultiScaleEffViT, ms_eff_vit_b0, ms_eff_vit_b5

from colorama import Fore, Style

c_ = Fore.BLUE
s_ = Style.BRIGHT
r_ = Style.RESET_ALL

current_dir = Path(__file__).resolve().parent
class Trainer:
    """DeepFake Classification Model Training"""
    
    def __init__(self, model, cfg):
        
        self.cfg = cfg
        self.model = model.to(cfg.device)
        
        # ======= Loss =======
        self.best_loss = float("inf")
        self.criterion = getattr(nn, cfg.train.loss)()
        self.train_epoch_loss = AverageMeter()
        self.valid_epoch_loss = AverageMeter()
        
        # ======= Early Stopping ======
        self.early_stopping = EarlyStopping(
            patience = cfg.train.patience,
            min_delta = cfg.train.min_delta,
        )
        
        # ======= Metrics =======
        base_metrics = build_metrics(cfg.device, task="binary")
        self.train_metrics = base_metrics.clone(prefix="train_")
        self.valid_metrics = base_metrics.clone(prefix="valid_")
        
        # ======= Optimizer =======
        self.optimizer = build_ms_eff_vit_optimizer(self.model, cfg)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.cfg.train.use_amp)
        
        # ======= Scheduler ========
        self.scheduler = fetch_scheduler(self.optimizer, cfg)
        
        # ======= Gradient Watch =======
        wandb.watch(self.model, log='all', log_freq = 100)
        
        print(f"{c_}{s_}\n✅ Trainer initialized | Device: {cfg.device}\n{r_}")
    
    def _save_checkpoint(self, ckpt_path, loss, epoch, metrics):
        
        torch.save(self.model.state_dict(), ckpt_path)
                
        artifact = wandb.Artifact(
            name = self.cfg.wandb_artifact_name, 
            type = 'model',
            metadata = {"epoch": epoch, "loss": loss, **metrics}
            )
        artifact.add_file(ckpt_path)
        wandb.log_artifact(artifact, aliases=['best'])
    
    def fit(self, train_df, valid_df):
        # ====== Calculate Ratio Real[0]/Fake[1] ======
        real_train_df = train_df[train_df[self.cfg.label_col] == 0].reset_index(drop=True)
        fake_train_df = train_df[train_df[self.cfg.label_col] == 1].reset_index(drop=True)
        n_splits = int(np.ceil(len(fake_train_df) / len(real_train_df)))
        
        # ====== Valid Dataset, DataLoader ======
        valid_ds = CutMixDeepFakeDataset(
                        meta_df = valid_df,
                        img_size = self.cfg.model.img_size,
                        transforms = get_valid_transforms,
                        mixup_prob = 0.,
                        cutout_prob = 0.,
                        )
        valid_loader = DataLoader(
                        valid_ds,
                        shuffle = False,
                        batch_size = self.cfg.train.valid_bs,
                        num_workers = os.cpu_count() - 1,
                        pin_memory = True,
                        drop_last = False,
        )
        
        print(f"{c_}{s_}\n🚀 Starting Training for {self.cfg.train.epochs}...{r_}")
        for epoch in range(1, self.cfg.train.epochs + 1):
            print(f"{c_}{s_}\n{'#' * 25}\n🌙 Epoch [{epoch}/{self.cfg.train.epochs}]\n{'#' * 25}{r_}")

            # ====== Handling Class Imbalance =======
            k = (epoch - 1) % n_splits
            
            balanced_df = class_imbalance_handle(real_train_df, fake_train_df, k)
            
            # ====== Train Dataset, DataLoader ======
            train_ds = CutMixDeepFakeDataset(
                        meta_df = balanced_df,
                        img_size = self.cfg.model.img_size,
                        transforms = get_train_transforms,
                        cutout_prob = self.cfg.train.cutout_prob,
                        mixup_prob = self.cfg.train.mixup_prob,
                        mixup_alpha = self.cfg.train.mixup_alpha,
            )
            train_loader = DataLoader(
                        train_ds,
                        shuffle = True, 
                        batch_size = self.cfg.train.train_bs,
                        num_workers = os.cpu_count() - 1,
                        pin_memory = True,
                        drop_last = False,
            )
            
            # ====== Training / Validation ======
            train_loss, train_metrics = self.train_one_epoch(train_loader)
            valid_loss, valid_metrics = self.valid_one_epoch(valid_loader)
            
            # ====== Scheduler ======
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(valid_loss)
            else:
                self.scheduler.step()
                
            # ====== Wandb Logging ======
            wandb.log({"train_loss": train_loss,
                       "valid_loss": valid_loss,
                       **train_metrics,
                       **valid_metrics,})
            
            # ====== Checkpoint ======
            ckpt_path = "best-checkpoint.bin"
            if valid_loss < self.best_loss:
                print(f"🎊{s_} Best Score Updated! ({self.best_loss:.5f} -> {valid_loss:.5f}){r_}")
                self.best_loss = valid_loss
                self._save_checkpoint(ckpt_path, valid_loss, epoch, valid_metrics)
            
            print(f"{c_}{s_}📉 Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}{r_}")

            # ====== Early Stopping ======
            if self.early_stopping(valid_loss):
                print(f"{c_}{s_}🛑 Early stopping triggered! Training stopped.{r_}")
                break

        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.cfg.device))
        return self.model
    
    def train_one_epoch(self, loader):
        self.model.train()
        
        # ====== reset loss, metric ======
        self.train_epoch_loss.reset()
        self.train_metrics.reset()
        
        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Training for {self.cfg.dataset}")
        for step, (imgs, labels) in pbar:
            imgs = imgs.to(self.cfg.device) # (B,C,H,W)
            labels = labels.to(self.cfg.device).float() # (B,)
        
            # ====== Auto Mixed Precision ======
            with torch.amp.autocast(device_type="cuda", enabled=self.cfg.train.use_amp):
                preds = self.model(imgs) # (B,1)
                preds = preds.view(-1) # (B,)
                loss = self.criterion(preds, labels)
                loss = loss / self.cfg.train.n_accumulate
            
            if self.cfg.train.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # ====== Gradient Accumulation ======
            if (step + 1) % self.cfg.train.n_accumulate == 0 or (step + 1) == len(loader):
                if self.cfg.train.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
            
                self.optimizer.zero_grad()
            
            preds_sig = torch.sigmoid(preds).detach()
            self.train_epoch_loss.update(loss.item() * self.cfg.train.n_accumulate, imgs.size(0))
            self.train_metrics.update(preds_sig, labels.int())

            avg_loss = self.train_epoch_loss.avg
            avg_metrics = self.train_metrics.compute()
        
            # ====== Learning Rate ======
            backbone_lr = self.optimizer.param_groups[0]['lr']
            l_vit_lr = self.optimizer.param_groups[2]['lr']
            h_vit_lr = self.optimizer.param_groups[4]['lr']
            head_lr = self.optimizer.param_groups[6]['lr']
            
            # ====== Calculate GPU Memory Usage =======
            mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
            
            # ====== Display Info ======
            pbar.set_postfix(
                train_loss = f'{avg_loss:.4f}',
                train_acc = f'{avg_metrics["train_acc"]:.4f}',
                train_auc = f'{avg_metrics["train_auc"]:.4f}',
                train_precision = f'{avg_metrics["train_precision"]:.4f}',
                train_recall = f'{avg_metrics["train_recall"]:.4f}',
                train_f1 = f'{avg_metrics["train_f1"]:.4f}',
                backbone_lr = f'{backbone_lr:.1e}',
                l_vit_lr = f'{l_vit_lr:.1e}',
                h_vit_lr = f'{h_vit_lr:.1e}',
                head_lr = f'{head_lr:.1e}',
                gpu_mem = f'{mem:.1f} GB'
            )
            
        torch.cuda.empty_cache()
        return avg_loss, avg_metrics
    
    @torch.no_grad()
    def valid_one_epoch(self, loader): 
        self.model.eval()
        
        # ====== reset loss, metric ======
        self.valid_epoch_loss.reset()
        self.valid_metrics.reset()
        
        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Validation for {self.cfg.dataset}")
        
        for steps, (imgs, labels) in pbar:
            imgs = imgs.to(self.cfg.device)
            labels = labels.to(self.cfg.device).float()
            
            preds = self.model(imgs) #(B,1)
            preds = preds.view(-1) # (B,)
            loss = self.criterion(preds, labels)
            
            preds_sig = torch.sigmoid(preds)
            self.valid_epoch_loss.update(loss.item(), imgs.size(0))
            self.valid_metrics.update(preds_sig, labels.int())
            
            avg_loss = self.valid_epoch_loss.avg
            avg_metrics = self.valid_metrics.compute()  
                        
            # ====== Calculate GPU Memory Usage =======
            mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
            
            # ====== Display Info ======
            pbar.set_postfix(
                valid_loss = f'{avg_loss:.4f}',
                valid_acc = f'{avg_metrics["valid_acc"]:.4f}',
                valid_auc = f'{avg_metrics["valid_auc"]:.4f}',
                valid_precision = f'{avg_metrics["valid_precision"]:.4f}',
                valid_recall = f'{avg_metrics["valid_recall"]:.4f}',
                valid_f1 = f'{avg_metrics["valid_f1"]:.4f}',
                gpu_mem = f'{mem:.1f} GB'
            )
            
        torch.cuda.empty_cache()
        return avg_loss, avg_metrics 
            
        
    def __str__(self):
        return "DeepFake Classification Trainer"
        

def main():
    parser = argparse.ArgumentParser(description="Multi-Scale EffViT Training")
   
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--model-ver", default="ms_eff_vit_b0", choices=['ms_eff_vit_b0','ms_eff_vit_b5'])
    parser.add_argument("--dataset", default="celeb_df_v2", choices=["celeb_df_v2","ff++","kodf"])
    
    parser.add_argument("--seed", default=2025, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--wandb-api-key", required=True)
    
    # Metadata
    parser.add_argument("--label-col", default='label', type=str)
    parser.add_argument("--img-col", default='img_path', type=str)
    parser.add_argument("--landmark-col", default='landmark_path', type=str)
    parser.add_argument("--ori-vid-col", default='ori_vid', type=str)
    parser.add_argument("--group-col", default='pid', type=str)
    parser.add_argument("--crop-dir", default='crops', type=str)
    parser.add_argument("--landmark-dir", default="landmarks", type=str)
        
    parser.add_argument("--result-info", default=True, type=bool)
    
    args = parser.parse_args()
    seed_everything(args.seed)
    config_path = current_dir / "config" / args.model_ver / f"{args.dataset}.yaml"

    cfg = OmegaConf.load(config_path)
    cfg.device = "cuda:0" if torch.cuda.is_available() else 'cpu'
    cfg.label_col = args.label_col
    
    # ==================== Build MetaData ==================================
    meta_df = pd.read_csv(os.path.join(args.root_dir, "train_frame_metadata.csv"), dtype={args.ori_vid_col: str})    
    meta_df[args.label_col] = meta_df[args.label_col].map({"REAL": 0, "FAKE": 1})
    meta_df[args.img_col] = meta_df.apply(lambda x: os.path.join(args.root_dir, f"{args.crop_dir}/{x['vid']}/{x['frame_idx']}.png"), axis=1)
    meta_df[args.landmark_col] = meta_df.apply(lambda x: os.path.join(args.root_dir, f"{args.landmark_dir}/{x['vid']}/{x['frame_idx']}.npy"), axis=1)
    
    # ==================== Build Dataset ==================================
    train_df, valid_df = split_data(meta_df=meta_df, seed=args.seed, label_col=args.label_col, 
                                    ori_vid=args.ori_vid_col, dataset=args.dataset, 
                                    group_col=args.group_col, debug=args.debug)
    
    # ==================== Build Model ===============================
    model = MultiScaleEffViT(**cfg.model)
    
    # =================== Setting Weighted & Bias ======================
    wandb.login(key=args.wandb_api_key)
    cfg.wandb_artifact_name = f"{args.model_ver}_L{cfg.model.l_block_idx}_H{cfg.model.h_block_idx}"
        
    run = wandb.init(
        project = f"{args.model_ver}_{args.dataset}",
        name= cfg.wandb_artifact_name,
        config = OmegaConf.to_container(cfg, resolve=True),
    )
    
    # =================== Training ============================
    trainer = Trainer(model, cfg)
    trainer.fit(train_df, valid_df)
    
    run.finish()
  
if __name__ == "__main__":
    main()
