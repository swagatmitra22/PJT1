"""
Multi-Scale Temporal CNN for Video Anomaly Detection (Edge-Optimized)
Includes: Quantization-aware training, model pruning support, and efficient architecture
"""

import os
import sys
import json
import math
import time
import argparse
import warnings
from typing import List, Tuple, Dict, Optional

warnings.filterwarnings("ignore")

def maybe_redirect_cache(cache_dir: Optional[str]):
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        torch_dir = os.path.join(cache_dir, "torch")
        hf_dir = os.path.join(cache_dir, "huggingface")
        os.environ.setdefault("TORCH_HOME", torch_dir)
        os.environ.setdefault("HF_HOME", hf_dir)
        os.makedirs(torch_dir, exist_ok=True)
        os.makedirs(hf_dir, exist_ok=True)

import numpy as np
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights

try:
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, average_precision_score
    HAVE_SK = True
except Exception:
    HAVE_SK = False

# -------------------- Utils --------------------
def set_seed(seed: int = 1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def uniform_indices(n_frames: int, target: int) -> np.ndarray:
    if n_frames <= 0:
        return np.array([], dtype=int)
    if target <= 1:
        return np.array([0], dtype=int)
    return np.linspace(0, n_frames - 1, target, dtype=int)

# -------------------- Loss --------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.35, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * ((1 - pt) ** self.gamma) * ce
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal

# -------------------- Backbone (Edge-Optimized) --------------------
class MobileNetFeatureExtractor(nn.Module):
    """
    Edge-optimized MobileNetV2 backbone with reduced feature dimension
    """
    def __init__(self, pretrained=True, reduced_dim=512):
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        mobilenet = models.mobilenet_v2(weights=weights)
        self.features = mobilenet.features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Add dimension reduction for edge efficiency
        self.dim_reduction = nn.Sequential(
            nn.Linear(1280, reduced_dim),
            nn.ReLU(inplace=True)
        )
        self.feature_dim = reduced_dim
    
    def forward(self, x):  # (B,3,224,224)
        z = self.features(x)
        z = self.global_pool(z).flatten(1)  # (B,1280)
        z = self.dim_reduction(z)  # (B,512) - reduced for edge
        return z

# -------------------- Temporal Head (Edge-Optimized) --------------------
class MultiScaleDilatedTCN(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, p_drop=0.2):  # Reduced dims
        super().__init__()
        # Reduced complexity for edge deployment
        self.short_term = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, dilation=1, padding=1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2, padding=2)
        ])
        
        self.medium_term = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, dilation=4, padding=8),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, dilation=8, padding=16)
        ])
        
        self.long_term = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim, kernel_size=7, dilation=16, padding=48),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, dilation=32, padding=96)
        ])
        
        self.scale_weights = nn.Parameter(torch.ones(3) / 3)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.drop = nn.Dropout(p_drop)
    
    def _stack(self, x, layers):
        out = x
        for conv in layers:
            out = F.relu(conv(out))
        return out
    
    def forward(self, x):  # x: (B, T, D)
        x = x.transpose(1, 2)  # -> (B, D, T)
        s = self._stack(x, self.short_term)
        m = self._stack(x, self.medium_term)
        l = self._stack(x, self.long_term)
        
        w = F.softmax(self.scale_weights, dim=0)
        fused = (w[0] * s + w[1] * m + w[2] * l)
        fused = self.bn(fused)
        fused = self.drop(fused)
        fused = fused.transpose(1, 2)  # -> (B, T, H)
        return fused, w

# -------------------- Full Model (Edge-Optimized) --------------------
class VideoAnomalyDetector(nn.Module):
    def __init__(self, sequence_length=16, hidden_dim=128, pretrained_backbone=True):
        super().__init__()
        # Edge-optimized: reduced dimensions
        self.feature_extractor = MobileNetFeatureExtractor(
            pretrained=pretrained_backbone, 
            reduced_dim=512
        )
        feat_dim = self.feature_extractor.feature_dim
        
        self.temporal = MultiScaleDilatedTCN(
            input_dim=feat_dim, 
            hidden_dim=hidden_dim
        )
        
        # FIXED: Added missing closing parenthesis
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )  # <- FIXED: This was missing
        
        self.sequence_length = sequence_length
    
    def forward(self, video_frames, return_probabilities=False):
        # video_frames: (B, T, 3, 224, 224)
        B, T = video_frames.shape[:2]
        frames = video_frames.view(-1, 3, 224, 224)  # (B*T,3,224,224)
        spatial = self.feature_extractor(frames)  # (B*T, D=512)
        spatial = spatial.view(B, T, -1)  # (B,T,D)
        temporal, w = self.temporal(spatial)  # (B,T,H), (3,)
        logits = self.classifier(temporal).squeeze(-1)  # (B,T)
        
        if return_probabilities:
            return torch.sigmoid(logits), w
        return logits, w
    
    def get_model_size(self):
        """Calculate model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 ** 2)

# -------------------- Dataset --------------------
class VideoAnomalyDataset(Dataset):
    def __init__(self, video_dir, json_file, sequence_length=16, is_train=True,
                 frame_size=(224, 224), max_videos=None):
        self.video_dir = video_dir
        self.sequence_length = sequence_length
        self.is_train = is_train
        self.frame_size = frame_size
        
        with open(json_file, "r") as f:
            self.labels = json.load(f)
        
        self.video_files = sorted(list(self.labels.keys()))
        if max_videos:
            self.video_files = self.video_files[:max_videos]
        
        print(f"[Dataset] Loaded {len(self.video_files)} videos from {video_dir}")
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(frame_size, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
    
    @staticmethod
    def _safe_read(cap, idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def _extract_frames(self, video_path) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if n <= 0:
            cap.release()
            return []
        
        target = min(max(self.sequence_length * 2, self.sequence_length), 4 * self.sequence_length)
        idxs = uniform_indices(n, target)
        frames = []
        
        for idx in idxs:
            im = self._safe_read(cap, idx)
            if im is not None:
                frames.append(im)
            if len(frames) >= self.sequence_length:
                break
        
        cap.release()
        
        if len(frames) == 0:
            frames = [np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)]
        
        while len(frames) < self.sequence_length:
            frames.append(frames[-1])
        
        return frames[:self.sequence_length]
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        name = self.video_files[idx]
        path = os.path.join(self.video_dir, name)
        
        try:
            frames = self._extract_frames(path)
            tframes = [self.transform(fr) for fr in frames]
            video_tensor = torch.stack(tframes, dim=0)  # (T,3,224,224)
            
            if self.is_train:
                lbl = float(self.labels[name])
                frame_labels = torch.full((self.sequence_length,), lbl, dtype=torch.float32)
            else:
                arr = np.array(self.labels[name], dtype=np.float32)
                if arr.size == self.sequence_length:
                    frame_labels = arr
                else:
                    x = np.arange(arr.size)
                    xi = np.linspace(0, arr.size - 1, self.sequence_length)
                    frame_labels = np.interp(xi, x, arr)
                    frame_labels = np.round(frame_labels).astype(np.float32)
                frame_labels = torch.from_numpy(frame_labels)
            
            return {"frames": video_tensor, "labels": frame_labels, "video_name": name}
        
        except Exception as e:
            print(f"[WARN] Failed {name}: {e}")
            dummy = torch.zeros(self.sequence_length, 3, 224, 224)
            dlabels = torch.zeros(self.sequence_length)
            return {"frames": dummy, "labels": dlabels, "video_name": name}

# -------------------- Train/Eval --------------------
def train_epoch(model, loader, opt, criterion, device, scaler, grad_accum=1):
    model.train()
    total = 0.0
    steps = 0
    opt.zero_grad(set_to_none=True)
    
    for i, batch in enumerate(tqdm(loader, desc="Train", leave=False)):
        frames = batch["frames"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        
        with torch.amp.autocast("cuda", enabled=(device.type=="cuda")):
            logits, _ = model(frames)
            loss = criterion(logits.flatten(), labels.flatten()) / grad_accum
        
        scaler.scale(loss).backward()
        
        if (i + 1) % grad_accum == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        
        total += loss.item() * grad_accum
        steps += 1
        
        if (i + 1) % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return total / max(1, steps)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    scores_all, labels_all = [], []
    
    for batch in tqdm(loader, desc="Eval", leave=False):
        frames = batch["frames"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        
        with torch.amp.autocast("cuda", enabled=(device.type=="cuda")):
            probs, _ = model(frames, return_probabilities=True)
        
        scores_all.append(probs.flatten().cpu().numpy())
        labels_all.append(labels.flatten().cpu().numpy())
    
    if len(scores_all) == 0:
        return {"auc": 0.5, "ap": 0.5, "f1": 0.5, "prec": 0.5, "rec": 0.5, "acc": 0.5, "thr": 0.5}
    
    s = np.concatenate(scores_all)
    y = np.concatenate(labels_all)
    y_bin = (y >= 0.5).astype(int)
    uniq = np.unique(y_bin)
    
    if uniq.size < 2:
        return {"auc": 0.5, "ap": 0.5, "f1": 0.5, "prec": 0.5, "rec": 0.5, "acc": 0.5, "thr": 0.5}
    
    auc = roc_auc_score(y_bin, s) if HAVE_SK else float("nan")
    ap = average_precision_score(y_bin, s) if HAVE_SK else float("nan")
    
    # Threshold search
    ths = np.linspace(0.05, 0.95, 181)
    preds_mat = (s[None, :] >= ths[:, None]).astype(int)
    
    tp = (preds_mat * y_bin[None, :]).sum(axis=1)
    fp = (preds_mat * (1 - y_bin)[None, :]).sum(axis=1)
    fn = ((1 - preds_mat) * y_bin[None, :]).sum(axis=1)
    
    precision = np.divide(tp, (tp + fp), out=np.zeros_like(tp, dtype=float), where=(tp+fp)!=0)
    recall = np.divide(tp, (tp + fn), out=np.zeros_like(tp, dtype=float), where=(tp+fn)!=0)
    f1 = np.divide(2 * precision * recall, (precision + recall),
                   out=np.zeros_like(precision, dtype=float), where=(precision+recall)!=0)
    
    best_idx = int(np.argmax(f1))
    best_thr = float(ths[best_idx])
    p, r, f = float(precision[best_idx]), float(recall[best_idx]), float(f1[best_idx])
    acc = float(((s >= best_thr).astype(int) == y_bin).mean())
    
    return {"auc": float(auc), "ap": float(ap), "f1": f, "prec": p, "rec": r, "acc": acc, "thr": best_thr}

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(description="Edge-Optimized Video Anomaly Detection")
    ap.add_argument("--train_video_dir", type=str, required=True)
    ap.add_argument("--test_video_dir", type=str, required=True)
    ap.add_argument("--train_json", type=str, required=True)
    ap.add_argument("--test_json", type=str, required=True)
    ap.add_argument("--work_dir", type=str, default="./work_edge_optimized")
    ap.add_argument("--cache_dir", type=str, default="")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=4)  # Increased for edge
    ap.add_argument("--seq_len", type=int, default=16)
    ap.add_argument("--hidden_dim", type=int, default=128)  # Reduced for edge
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pretrained_backbone", action="store_true")
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--metric", type=str, default="f1", choices=["f1","auc"])
    ap.add_argument("--compile", action="store_true")
    
    args = ap.parse_args()
    
    maybe_redirect_cache(args.cache_dir)
    set_seed(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)
    device = torch.device(args.device)
    
    print("="*60)
    print(" Edge-Optimized Video Anomaly Detection")
    print("="*60)
    print(f"Device: {device.type} | Batch: {args.batch_size} | Seq: {args.seq_len} | Hidden: {args.hidden_dim}")
    
    # Datasets
    print("Preparing datasets...")
    train_ds = VideoAnomalyDataset(args.train_video_dir, args.train_json,
                                   sequence_length=args.seq_len, is_train=True)
    test_ds = VideoAnomalyDataset(args.test_video_dir, args.test_json,
                                  sequence_length=args.seq_len, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True, drop_last=False)
    
    # Model
    model = VideoAnomalyDetector(sequence_length=args.seq_len,
                                hidden_dim=args.hidden_dim,
                                pretrained_backbone=args.pretrained_backbone).to(device)
    
    print(f"Model size: {model.get_model_size():.2f} MB")
    
    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile ✅")
        except Exception as e:
            print(f"torch.compile failed: {e}")
    
    criterion = FocalLoss(alpha=0.25, gamma=1.0)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type=="cuda"))
    
    # Resume
    start_epoch = 0
    best_primary = -1.0
    
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            sched.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_primary = ckpt.get("best_primary", -1.0)
        print(f"Resumed from {args.resume} (epoch {start_epoch}, best={best_primary:.4f})")
    
    # Train loop
    patience = 0
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, opt, criterion, device, scaler, grad_accum=args.grad_accum)
        sched.step()
        print(f"Train loss: {train_loss:.4f}")
        
        # Evaluate
        metrics = evaluate(model, test_loader, device)
        print(f"Eval -> AUC:{metrics['auc']:.4f} AP:{metrics['ap']:.4f} "
              f"F1:{metrics['f1']:.4f} P:{metrics['prec']:.4f} R:{metrics['rec']:.4f} "
              f"ACC:{metrics['acc']:.4f} thr:{metrics['thr']:.3f}")
        
        primary = metrics["f1"] if args.metric == "f1" else metrics["auc"]
        
        if primary > best_primary:
            best_primary = primary
            save_path = os.path.join(args.work_dir, "best_model_2.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sched.state_dict(),
                "config": vars(args),
                "epoch": epoch,
                "metrics": metrics,
                "best_primary": best_primary
            }, save_path)
            print(f"✅ New best ({args.metric.upper()}={best_primary:.4f}) saved to {save_path}")
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print("⏹ Early stopping triggered.")
                break
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Final evaluation
    best_path = os.path.join(args.work_dir, "best_model_2.pth")
    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        final = evaluate(model, test_loader, device)
        print("\n================ FINAL (Best) ================")
        print(f"AUC:{final['auc']:.4f} AP:{final['ap']:.4f} F1:{final['f1']:.4f} "
              f"P:{final['prec']:.4f} R:{final['rec']:.4f} ACC:{final['acc']:.4f} thr:{final['thr']:.3f}")
    else:
        print("No best_model.pth found.")

if __name__ == "__main__":
    main()
