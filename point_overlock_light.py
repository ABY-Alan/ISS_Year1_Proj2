# point_overlock_light.py
# Lightweight "OverLoCK for Point Clouds": DGCNN backbone + Overview (global context) + Gate (channel-wise)
# Author: you + ChatGPT
# Python >= 3.9, PyTorch >= 2.0

import os, argparse, json, time, random
import joblib
import numpy as np
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# Utils
# ---------------------------


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_class_weights(y_indices, num_classes):
    counts = np.bincount(y_indices, minlength=num_classes).astype(np.float64)
    weights = counts.sum() / (counts + 1e-8)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def split_indices(n, train_ratio=0.7, val_ratio=0.15, seed=42, y=None):
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    if y is None:
        rng.shuffle(idx)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]
    # stratified split
    labels = np.array(y)
    uniq = np.unique(labels)
    train_idx, val_idx, test_idx = [], [], []
    for c in uniq:
        c_idx = np.where(labels == c)[0]
        rng.shuffle(c_idx)
        n_c = len(c_idx)
        n_train = int(n_c * train_ratio)
        n_val = int(n_c * val_ratio)
        train_idx += list(c_idx[:n_train])
        val_idx += list(c_idx[n_train : n_train + n_val])
        test_idx += list(c_idx[n_train + n_val :])
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


# ---------------------------
# Dataset
# ---------------------------


class PointFrameDataset(Dataset):
    """
    data: tuple (X_list, y_list)
    - X_list: list of ndarray, each (54,5) -> [x,y,z,v,a]
    - y_list: list of str labels
    Transforms:
      - normalize xyz to [-1,1] given observed bounds (optional robust scaling for v,a)
      - simple augment: small rotation around z, jitter, small translate
    """

    def __init__(
        self,
        X,
        y,
        label_encoder=None,
        split="train",
        norm=True,
        augment=False,
        xyz_bounds=([-1.0, -1.0, 0.0], [1.0, 1.0, 2.0]),
        robust_va=True,
    ):
        self.X = X
        self.y_str = y
        self.split = split
        self.norm = norm
        self.augment = augment and (split == "train")
        self.xyz_min = np.array(xyz_bounds[0], dtype=np.float32)
        self.xyz_max = np.array(xyz_bounds[1], dtype=np.float32)
        self.robust_va = robust_va

        if label_encoder is None:
            self.le = LabelEncoder()
            self.y = self.le.fit_transform(self.y_str)
        else:
            self.le = label_encoder
            self.y = self.le.transform(self.y_str)

        # precompute robust stats for v,a if needed
        if self.robust_va:
            va = []
            for arr in self.X:
                va.append(arr[:, 3:5])  # v,a
            va = np.concatenate(va, axis=0)
            self.va_med = np.median(va, axis=0)
            self.va_iqr = np.percentile(va, 75, axis=0) - np.percentile(va, 25, axis=0)
            self.va_iqr[self.va_iqr == 0] = 1.0
        else:
            self.va_med = np.zeros(2, dtype=np.float32)
            self.va_iqr = np.ones(2, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def _normalize(self, pts):
        # xyz to [-1,1] by min/max; v,a robust standardization
        xyz = pts[:, :3]
        v_a = pts[:, 3:5]
        xyz_n = 2.0 * (xyz - self.xyz_min) / (self.xyz_max - self.xyz_min + 1e-8) - 1.0
        va_n = (v_a - self.va_med) / (self.va_iqr + 1e-6)
        return np.concatenate([xyz_n, va_n], axis=1)

    @staticmethod
    def _rotate_z(pts, max_deg=10.0):
        theta = np.deg2rad(np.random.uniform(-max_deg, max_deg))
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        xyz = pts[:, :3] @ R.T
        pts[:, :3] = xyz
        return pts

    @staticmethod
    def _jitter(pts, sigma=0.005, clip=0.02):
        noise = np.clip(sigma * np.random.randn(*pts.shape), -clip, clip).astype(
            np.float32
        )
        pts = pts + noise
        return pts

    @staticmethod
    def _translate(pts, shift_range=0.02):
        shift = np.random.uniform(-shift_range, shift_range, size=(1, 5)).astype(
            np.float32
        )
        shift[0, 3:] = 0.0  # do not shift v,a
        return pts + shift

    def __getitem__(self, idx):
        arr = self.X[idx].astype(np.float32)  # (54,5)
        N_target = 54
        N = arr.shape[0]
        if N < N_target:
            # 不足时随机重复补齐
            pad_idx = np.random.choice(N, N_target - N, replace=True)
            arr = np.concatenate([arr, arr[pad_idx]], axis=0)
        elif N > N_target:
            # 超过则随机采样
            arr = arr[np.random.choice(N, N_target, replace=False)]
        if self.norm:
            arr = self._normalize(arr)
        if self.augment:
            if np.random.rand() < 0.5:
                arr = self._rotate_z(arr)
            if np.random.rand() < 0.5:
                arr = self._translate(arr)
            if np.random.rand() < 0.5:
                arr = self._jitter(arr)
        y = self.y[idx]
        # PyTorch expects (C,N); we use (N,C) then permute later in model
        return arr, y


# ---------------------------
# DGCNN Backbone (simplified)
# ---------------------------


def knn(x, k):
    # x: (B,C,N)
    # returns idx: (B,N,k)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=10):
    # x: (B,C,N)
    B, C, N = x.size()
    idx = knn(x, k=k)  # (B,N,k)
    idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
    idx = idx + idx_base
    idx = idx.view(-1)
    x = x.transpose(2, 1).contiguous()  # (B,N,C)
    feature = x.view(B * N, -1)[idx, :].view(B, N, k, C)
    x = x.view(B, N, 1, C).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)  # (B,2C,N,k)
    return feature


class DGCNNBackbone(nn.Module):
    """
    Simplified DGCNN backbone with dynamic-safe channel handling.
    Fixed all conv definitions, added dynamic MLP init.
    """

    def __init__(self, in_channels=5, k=10, feat_dim=256):
        super().__init__()
        self.k = k

        # === EdgeConv layers ===
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.LeakyReLU(0.2),
        )

        # === dynamic MLP will be created after first forward ===
        self.mlp = None
        self.mlp_in = None

    def forward(self, x):
        # x: (B, N, 5)
        x = x.permute(0, 2, 1)  # (B, 5, N)

        # --- Four EdgeConv stages ---
        x1 = self.conv1(get_graph_feature(x, k=self.k)).max(dim=-1)[0]  # (B, 64, N)
        x2 = self.conv2(get_graph_feature(x1, k=self.k)).max(dim=-1)[0]  # (B, 64, N)
        x3 = self.conv3(get_graph_feature(x2, k=self.k)).max(dim=-1)[0]  # (B, 128, N)
        x4 = self.conv4(get_graph_feature(x3, k=self.k)).max(dim=-1)[
            0
        ]  # (B, feat_dim, N)

        # --- Concatenate all scales ---
        x_cat = torch.cat((x1, x2, x3, x4), dim=1)  # (B, total_channels, N)
        x_global = F.adaptive_max_pool1d(x_cat, 1).squeeze(-1)  # (B, total_channels)

        # === Lazy init of MLP ===
        if self.mlp is None:
            in_dim = x_global.shape[1]
            print(f"[Init] DGCNNBackbone detected global feature dim = {in_dim}")
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2)
            ).to(x.device)
            self.mlp_in = in_dim

        # --- Global embedding ---
        x_global = self.mlp(x_global)  # (B, 1024)

        return x_global, x4


# ---------------------------
# Overview + Gate
# ---------------------------


class OverviewNet(nn.Module):
    """Global context from mean-pool of inputs + small MLP."""

    def __init__(self, in_channels=5, ctx_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 128), nn.ReLU(), nn.Linear(128, ctx_dim), nn.ReLU()
        )

    def forward(self, x):
        # x: (B,N,5) normalized points
        mean_feat = x.mean(dim=1)  # (B,5)
        ctx = self.mlp(mean_feat)  # (B,ctx_dim)
        return ctx


class ContextGate(nn.Module):
    """Channel-wise gate: use context to scale local token features."""

    def __init__(self, ctx_dim=256, feat_dim=256):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(ctx_dim, feat_dim), nn.Sigmoid())

    def forward(self, ctx, local_feat):
        # ctx: (B,ctx_dim)
        # local_feat: (B,feat_dim,N)
        scale = self.fc(ctx)[:, :, None]  # (B,feat_dim,1)
        return local_feat * scale


# ---------------------------
# Full Model
# ---------------------------


class PointOverLoCKLight(nn.Module):
    def __init__(
        self,
        num_classes,
        use_overview=True,
        use_gate=True,
        in_channels=5,
        k=10,
        feat_dim=256,
        ctx_dim=256,
        dropout=0.4,
    ):
        super().__init__()
        self.backbone = DGCNNBackbone(in_channels=in_channels, k=k, feat_dim=feat_dim)
        self.use_overview = use_overview
        self.use_gate = use_gate
        if use_overview:
            self.overview = OverviewNet(in_channels=in_channels, ctx_dim=ctx_dim)
            fusion_in = 1024 + ctx_dim
        else:
            self.overview = None
            fusion_in = 1024
        self.gate = (
            ContextGate(ctx_dim=ctx_dim, feat_dim=feat_dim)
            if (use_overview and use_gate)
            else None
        )
        self.head = nn.Sequential(
            nn.Linear(fusion_in, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (B,N,5)
        global_feat, local_feat = self.backbone(
            x
        )  # global: (B,1024), local: (B,feat_dim,N)
        if self.use_overview:
            ctx = self.overview(x)  # type: ignore # (B,ctx_dim)
            if self.gate is not None:
                local_feat = self.gate(ctx, local_feat)  # (B,feat_dim,N)
            fused = torch.cat([global_feat, ctx], dim=1)
        else:
            fused = global_feat
        logits = self.head(fused)
        return logits


# ---------------------------
# Training / Eval
# ---------------------------


def train_one_epoch(model, loader, optimizer, device, criterion):
    model.train()
    losses, y_true, y_pred = [], [], []
    for pts, y in loader:
        pts = pts.to(device).float()  # 强制转为 float32
        y = y.to(device)
        optimizer.zero_grad()
        out = model(pts)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(out.argmax(1).detach().cpu().numpy().tolist())
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    return np.mean(losses), acc, mf1


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    losses, y_true, y_pred = [], [], []
    for pts, y in loader:
        pts = pts.to(device).float()
        y = y.to(device)
        out = model(pts)
        loss = criterion(out, y)
        losses.append(loss.item())
        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(out.argmax(1).detach().cpu().numpy().tolist())
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    return np.mean(losses), acc, mf1, cm, (y_true, y_pred)


def save_confusion_matrix(cm, labels, out_png):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(7, 6))
        sns.heatmap(
            cm,
            annot=False,
            cmap="Blues",
            fmt="d",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.xlabel("Pred")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
    except Exception as e:
        print(f"[warn] failed to save confusion matrix: {e}")


# ---------------------------
# Main
# ---------------------------


# def main():
#     parser = argparse.ArgumentParser()
#     # parser.add_argument("--joblib_path", type=str, default="all_dataset.joblib")
#     parser.add_argument("--batch_size", type=int, default=256)
#     parser.add_argument("--epochs", type=int, default=20)
#     parser.add_argument("--lr", type=float, default=1e-3)
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--k", type=int, default=10)
#     parser.add_argument("--feat_dim", type=int, default=256)
#     parser.add_argument("--ctx_dim", type=int, default=256)
#     parser.add_argument("--dropout", type=float, default=0.4)
#     parser.add_argument("--no_overview", action="store_true")
#     parser.add_argument("--no_gate", action="store_true")
#     parser.add_argument("--save_dir", type=str, default="runs_overlock_light")
#     parser.add_argument("--val_ratio", type=float, default=0.15)
#     parser.add_argument("--test_ratio", type=float, default=0.15)
#     parser.add_argument(
#         "--weighting", type=str, default="none", choices=["none", "class", "focal"]
#     )
#     parser.add_argument("--workers", type=int, default=4)
#     args = parser.parse_args()

#     set_seed(args.seed)
#     os.makedirs(args.save_dir, exist_ok=True)

#     # Load data
#     X_list, y_list = joblib.load(args.joblib_path)
#     assert isinstance(X_list, list) and isinstance(y_list, list)
#     n = len(X_list)

#     # Label encoder
#     le = LabelEncoder()
#     y_idx = le.fit_transform(y_list)
#     labels = list(le.classes_)
#     num_classes = len(labels)

#     # Splits
#     train_idx, val_idx, test_idx = split_indices(
#         n,
#         train_ratio=1.0 - args.val_ratio - args.test_ratio,
#         val_ratio=args.val_ratio,
#         seed=args.seed,
#         y=y_idx,
#     )
#     X_train = [X_list[i] for i in train_idx]
#     y_train = [y_list[i] for i in train_idx]
#     X_val = [X_list[i] for i in val_idx]
#     y_val = [y_list[i] for i in val_idx]
#     X_test = [X_list[i] for i in test_idx]
#     y_test = [y_list[i] for i in test_idx]

#     # Datasets
#     ds_train = PointFrameDataset(
#         X_train, y_train, label_encoder=le, split="train", norm=True, augment=True
#     )
#     ds_val = PointFrameDataset(
#         X_val,
#         y_val,
#         label_encoder=le,
#         split="val",
#         norm=True,
#         augment=False,
#         xyz_bounds=(ds_train.xyz_min, ds_train.xyz_max),
#     )
#     ds_test = PointFrameDataset(
#         X_test,
#         y_test,
#         label_encoder=le,
#         split="test",
#         norm=True,
#         augment=False,
#         xyz_bounds=(ds_train.xyz_min, ds_train.xyz_max),
#     )

#     # Loaders
#     dl_train = DataLoader(
#         ds_train,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=args.workers,
#         drop_last=False,
#     )
#     dl_val = DataLoader(
#         ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
#     )
#     dl_test = DataLoader(
#         ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
#     )

#     # Model
#     model = PointOverLoCKLight(
#         num_classes=num_classes,
#         use_overview=not args.no_overview,
#         use_gate=(not args.no_overview) and (not args.no_gate),
#         in_channels=5,
#         k=args.k,
#         feat_dim=args.feat_dim,
#         ctx_dim=args.ctx_dim,
#         dropout=args.dropout,
#     )
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)

#     # Loss
#     if args.weighting == "class":
#         w = compute_class_weights(le.transform(y_train), num_classes).to(device)
#         criterion = nn.CrossEntropyLoss(weight=w)
#     elif args.weighting == "focal":
#         # simple focal wrapper
#         class FocalLoss(nn.Module):
#             def __init__(self, gamma=2.0, weight=None):
#                 super().__init__()
#                 self.gamma = gamma
#                 self.ce = nn.CrossEntropyLoss(weight=weight)

#             def forward(self, logits, target):
#                 logpt = -self.ce(logits, target)
#                 pt = torch.exp(logpt)
#                 loss = -((1 - pt) ** self.gamma) * logpt
#                 return loss

#         w = compute_class_weights(le.transform(y_train), num_classes).to(device)
#         criterion = FocalLoss(gamma=2.0, weight=w)
#     else:
#         criterion = nn.CrossEntropyLoss()

#     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

#     best_val_mf1, best_state = -1.0, None
#     log = []

#     print(f"Labels: {labels}")
#     print(f"Train/Val/Test sizes: {len(ds_train)}/{len(ds_val)}/{len(ds_test)}")

#     for epoch in range(1, args.epochs + 1):
#         t0 = time.time()
#         tr_loss, tr_acc, tr_mf1 = train_one_epoch(
#             model, dl_train, optimizer, device, criterion
#         )
#         val_loss, val_acc, val_mf1, val_cm, _ = evaluate(
#             model, dl_val, device, criterion
#         )
#         scheduler.step()

#         log.append(
#             {
#                 "epoch": epoch,
#                 "train_loss": tr_loss,
#                 "train_acc": tr_acc,
#                 "train_mf1": tr_mf1,
#                 "val_loss": val_loss,
#                 "val_acc": val_acc,
#                 "val_mf1": val_mf1,
#                 "lr": scheduler.get_last_lr()[0],
#             }
#         )
#         print(
#             f"[{epoch:03d}] tr_loss={tr_loss:.4f} acc={tr_acc:.4f} mf1={tr_mf1:.4f} | "
#             f"val_loss={val_loss:.4f} acc={val_acc:.4f} mf1={val_mf1:.4f} | lr={scheduler.get_last_lr()[0]:.2e} "
#             f"({time.time()-t0:.1f}s)"
#         )

#         if val_mf1 > best_val_mf1:
#             best_val_mf1 = val_mf1
#             best_state = {
#                 "model": model.state_dict(),
#                 "args": vars(args),
#                 "labels": labels,
#                 "epoch": epoch,
#             }
#             torch.save(best_state, os.path.join(args.save_dir, "best.pt"))
#             save_confusion_matrix(
#                 val_cm, labels, os.path.join(args.save_dir, "val_cm.png")
#             )

#     # Test
#     if best_state is not None:
#         model.load_state_dict(best_state["model"])
#     test_loss, test_acc, test_mf1, test_cm, (yt, yp) = evaluate(
#         model, dl_test, device, criterion
#     )
#     print("\n===== TEST =====")
#     print(f"loss={test_loss:.4f} acc={test_acc:.4f} macroF1={test_mf1:.4f}")
#     print(classification_report(yt, yp, target_names=labels, digits=4))
#     save_confusion_matrix(test_cm, labels, os.path.join(args.save_dir, "test_cm.png"))

#     # Save log
#     with open(os.path.join(args.save_dir, "log.json"), "w", encoding="utf-8") as f:
#         json.dump(log, f, ensure_ascii=False, indent=2)


# if __name__ == "__main__":
#     main()
