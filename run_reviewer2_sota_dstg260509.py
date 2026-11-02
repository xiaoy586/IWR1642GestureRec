#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reviewer-2 SoTA comparison for the submitted 260509 lightweight DSTG-Net.

Compared methods:
  - Ref31-P4Transformer
  - Ref41-TD-CNN-Transformer
  - Ref42-TAGEGestureNet
  - DSTG-Net, the submitted lightweight model

Protocol:
  - public mmWave radar gesture dataset by default: data/
  - shared stratified split for all methods
  - same optimizer, scheduler, epochs, batch size, metrics, and result format
  - DSTG-Net uses the submitted 260509 features:
      LSTM branch: [80, 240]
      Attention-BiGRU statistical-difference branch: [80, 58]
  - point/graph baselines use adapted reproductions under the same local data
    protocol; do not report these values as numbers copied from the original
    papers.

Example:
  python run_reviewer2_sota_dstg260509.py
  python run_reviewer2_sota_dstg260509.py --epochs 200
  python run_reviewer2_sota_dstg260509.py --methods Ref41-TD-CNN-Transformer DSTG-Net --epochs 5
"""

import argparse
import copy
import importlib.util
import os
import random
import sys
import time
import types
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, SubsetRandomSampler
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

DATA_PATH = "data"
RESULT_DIR = "Result-Reviewer2-SoTA-DSTG260509"
SEED = 42

MAX_FRAMES = 80
MAX_POINTS = 80
NUM_CLASSES = 12

BATCH_SIZE = 32
GRAPH_BATCH_SIZE = 64
P4_BATCH_SIZE = 14
P4_MEMORY_BATCH_SIZE = 4
EPOCHS = 200
LR = 5e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

LR_PLATEAU_FACTOR = 0.5
LR_PLATEAU_PATIENCE = 8
LR_PLATEAU_THRESHOLD = 1e-4
LR_PLATEAU_COOLDOWN = 2
MIN_LR = 1e-6
EARLY_STOPPING_PATIENCE = None
CHECKPOINT_CRITERION = "Best validation accuracy"

BENCHMARK_WARMUP_BATCHES = 5
BENCHMARK_REPEATS = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = torch.cuda.is_available()

GESTURES = [
    "arm_to_left",
    "arm_to_right",
    "close_fist_horizontally",
    "close_fist_perpendicularly",
    "hand_away",
    "hand_closer",
    "hand_down",
    "hand_rotation_palm_down",
    "hand_rotation_palm_up",
    "hand_to_left",
    "hand_to_right",
    "hand_up",
]

GESTURE_ABBREVIATIONS = {
    "arm_to_left": "AL",
    "arm_to_right": "AR",
    "close_fist_horizontally": "CFH",
    "close_fist_perpendicularly": "CFP",
    "hand_away": "HA",
    "hand_closer": "HC",
    "hand_down": "HD",
    "hand_rotation_palm_down": "HRPD",
    "hand_rotation_palm_up": "HRPU",
    "hand_to_left": "HL",
    "hand_to_right": "HR",
    "hand_up": "HU",
}

DISPLAY_GESTURES = [GESTURE_ABBREVIATIONS[g] for g in GESTURES]
LABELS_DICT = {gesture: idx for idx, gesture in enumerate(GESTURES)}

METHOD_SPECS = {
    "Ref31-P4Transformer": {
        "Reference": "[31]",
        "Family": "Point 4D Transformer",
        "Input": "Sparse radar point sequence",
        "Note": "Adapted reproduction using x, y, and Doppler velocity as 3-D point coordinates.",
    },
    "Ref41-TD-CNN-Transformer": {
        "Reference": "[41]",
        "Family": "Time-distributed CNN plus temporal Transformer",
        "Input": "Projected radar frame maps",
        "Note": "Adapted reproduction using projected occupancy, Doppler, and peak-intensity maps.",
    },
    "Ref42-TAGEGestureNet": {
        "Reference": "[42]",
        "Family": "Trajectory-aware graph encoder",
        "Input": "Graph-based radar point cloud",
        "Note": "Adapted reproduction using local TAGEGestureNet graph construction and model.",
    },
    "DSTG-Net": {
        "Reference": "Proposed",
        "Family": "260509 lightweight dual-stream LSTM + Attention-BiGRU",
        "Input": "DSTG [80,240] + [80,58]",
        "Note": "Submitted lightweight DSTG-Net setting.",
    },
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def sanitize_name(name):
    return name.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def is_cuda_oom(exc):
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading and feature extraction
# ---------------------------------------------------------------------------

def collect_samples(data_path, max_samples_per_class=None):
    samples = []
    labels = []
    for gesture, label in LABELS_DICT.items():
        folder = os.path.join(data_path, gesture)
        if not os.path.exists(folder):
            continue
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".csv")])
        if max_samples_per_class is not None:
            files = files[:max_samples_per_class]
        for file_name in files:
            samples.append((os.path.join(folder, file_name), label))
            labels.append(label)
    if not samples:
        raise RuntimeError(f"No CSV files found in {data_path}")
    return samples, np.asarray(labels, dtype=np.int64)


def read_radar_csv(file_path):
    df = pd.read_csv(file_path)
    required_cols = {"FrameNumber", "Velocity", "PeakValue", "x", "y"}
    if required_cols.issubset(set(df.columns)):
        return df
    return pd.read_csv(file_path, skiprows=1)


def parse_frames(file_path):
    df = read_radar_csv(file_path)
    cols = df.columns.tolist()
    frame_col = "FrameNumber" if "FrameNumber" in cols else cols[0]
    vel_col = "Velocity" if "Velocity" in cols else (cols[3] if len(cols) > 3 else None)
    peak_col = "PeakValue" if "PeakValue" in cols else (cols[4] if len(cols) > 4 else None)
    x_col = "x" if "x" in cols else (cols[5] if len(cols) > 5 else None)
    y_col = "y" if "y" in cols else (cols[6] if len(cols) > 6 else None)
    range_col = "Range" if "Range" in cols else None
    if any(col is None for col in [frame_col, vel_col, peak_col, x_col, y_col]):
        raise ValueError(f"Missing required radar columns in {file_path}")

    frames_data = []
    for _, group in df.groupby(frame_col, sort=True):
        x_vals = group[x_col].values.astype(np.float32)
        y_vals = group[y_col].values.astype(np.float32)
        v_vals = group[vel_col].values.astype(np.float32)
        p_vals = group[peak_col].values.astype(np.float32)
        if range_col is not None and range_col in group:
            r_vals = group[range_col].values.astype(np.float32)
        else:
            r_vals = np.sqrt(x_vals ** 2 + y_vals ** 2).astype(np.float32)
        frames_data.append(list(zip(x_vals, y_vals, v_vals, r_vals, p_vals)))
    return frames_data


def process_sample(file_path):
    """Return point features, masks, and submitted 260509 DSTG features."""
    frames_data = parse_frames(file_path)

    points = np.zeros((MAX_FRAMES, MAX_POINTS, 4), dtype=np.float32)
    point_mask = np.zeros((MAX_FRAMES, MAX_POINTS), dtype=bool)
    lstm_data = np.zeros((MAX_FRAMES, 3, MAX_POINTS), dtype=np.float32)

    frame_features = []
    for frame_idx, frame in enumerate(frames_data[:MAX_FRAMES]):
        limited = frame[:MAX_POINTS]
        if limited:
            x_p = np.asarray([p[0] for p in limited], dtype=np.float32)
            y_p = np.asarray([p[1] for p in limited], dtype=np.float32)
            v_p = np.asarray([p[2] for p in limited], dtype=np.float32)
            r_p = np.asarray([p[3] for p in limited], dtype=np.float32)
            peak_p = np.asarray([p[4] for p in limited], dtype=np.float32)

            n = len(limited)
            points[frame_idx, :n, :] = np.stack([x_p, y_p, v_p, peak_p], axis=-1)
            point_mask[frame_idx, :n] = True

            pad_width = MAX_POINTS - n
            lstm_data[frame_idx] = np.asarray([
                np.pad(x_p, (0, pad_width)),
                np.pad(y_p, (0, pad_width)),
                np.pad(v_p, (0, pad_width)),
            ], dtype=np.float32)

            feature_list = [v_p, r_p, peak_p, x_p, y_p]
            non_zero_mask_total = np.zeros(n, dtype=bool)
            fv = []
            for feat in feature_list:
                non_zero_mask = feat != 0
                non_zero_mask_total = non_zero_mask_total | non_zero_mask
                non_zero_values = feat[non_zero_mask]
                if len(non_zero_values) > 0:
                    fv.extend([
                        float(np.mean(non_zero_values)),
                        float(np.std(non_zero_values)),
                        float(np.median(non_zero_values)),
                        float(np.max(non_zero_values)),
                        float(np.min(non_zero_values)),
                        float(np.percentile(non_zero_values, 25)),
                        float(np.percentile(non_zero_values, 75)),
                    ])
                else:
                    fv.extend([0.0] * 7)
            fv.append(float(np.sum(non_zero_mask_total)))
        else:
            fv = [0.0] * 36
        frame_features.append(fv)

    while len(frame_features) < MAX_FRAMES:
        frame_features.append([0.0] * 36)

    lstm_feature = lstm_data.reshape(MAX_FRAMES, 240).astype(np.float32)
    ff_arr = np.asarray(frame_features, dtype=np.float32)
    diff_arr = np.diff(ff_arr, axis=0)
    diff_arr = np.vstack([np.zeros((1, 36), dtype=np.float32), diff_arr])
    gru_feature = np.concatenate([ff_arr, diff_arr[:, :22]], axis=1).astype(np.float32)
    frame_mask = point_mask.any(axis=1)

    return points, frame_mask, point_mask, lstm_feature, gru_feature


class SotaRadarDataset(Dataset):
    def __init__(self, samples):
        self.points = []
        self.frame_masks = []
        self.point_masks = []
        self.x_lstm = []
        self.x_gru = []
        self.y = []

        print("Loading shared SoTA/DSTG dataset...")
        for file_path, label in tqdm(samples, leave=False):
            try:
                points, frame_mask, point_mask, lstm_feature, gru_feature = process_sample(file_path)
            except Exception as exc:
                print(f"[Warning] failed to process {file_path}: {exc}")
                continue
            self.points.append(points)
            self.frame_masks.append(frame_mask)
            self.point_masks.append(point_mask)
            self.x_lstm.append(lstm_feature)
            self.x_gru.append(gru_feature)
            self.y.append(label)

        self.points = np.asarray(self.points, dtype=np.float32)
        self.frame_masks = np.asarray(self.frame_masks, dtype=bool)
        self.point_masks = np.asarray(self.point_masks, dtype=bool)
        self.x_lstm = np.asarray(self.x_lstm, dtype=np.float32)
        self.x_gru = np.asarray(self.x_gru, dtype=np.float32)
        self.y = np.asarray(self.y, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.points[idx], dtype=torch.float32),
            torch.tensor(self.frame_masks[idx], dtype=torch.bool),
            torch.tensor(self.point_masks[idx], dtype=torch.bool),
            torch.tensor(self.x_lstm[idx], dtype=torch.float32),
            torch.tensor(self.x_gru[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


class TrainSetNormalizedSotaDataset(Dataset):
    """Normalize point and DSTG features with train split statistics only."""

    def __init__(self, source_dataset, train_idx):
        self.y = source_dataset.y
        self.frame_masks = source_dataset.frame_masks
        self.point_masks = source_dataset.point_masks
        self.points = source_dataset.points.copy()
        self.x_lstm = source_dataset.x_lstm.copy()
        self.x_gru = source_dataset.x_gru.copy()

        train_idx = np.asarray(train_idx, dtype=np.int64)

        train_points = self.points[train_idx]
        train_point_mask = self.point_masks[train_idx]
        valid_points = train_points[train_point_mask]
        if len(valid_points) > 0:
            self.point_mean = valid_points.mean(axis=0).astype(np.float32)
            self.point_std = valid_points.std(axis=0).astype(np.float32)
        else:
            self.point_mean = np.zeros(4, dtype=np.float32)
            self.point_std = np.ones(4, dtype=np.float32)
        self.point_std[self.point_std < 1e-6] = 1.0

        train_lstm = self.x_lstm[train_idx]
        train_gru = self.x_gru[train_idx]
        self.lstm_mean = train_lstm.mean(axis=(0, 1), keepdims=True).astype(np.float32)
        self.lstm_std = train_lstm.std(axis=(0, 1), keepdims=True).astype(np.float32)
        self.gru_mean = train_gru.mean(axis=(0, 1), keepdims=True).astype(np.float32)
        self.gru_std = train_gru.std(axis=(0, 1), keepdims=True).astype(np.float32)
        self.lstm_std[self.lstm_std < 1e-6] = 1.0
        self.gru_std[self.gru_std < 1e-6] = 1.0

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        point_mask = self.point_masks[idx]
        points = np.zeros_like(self.points[idx], dtype=np.float32)
        points[point_mask] = (self.points[idx][point_mask] - self.point_mean) / self.point_std
        x_lstm = (self.x_lstm[idx] - self.lstm_mean.squeeze(0)) / self.lstm_std.squeeze(0)
        x_gru = (self.x_gru[idx] - self.gru_mean.squeeze(0)) / self.gru_std.squeeze(0)
        return (
            torch.tensor(points, dtype=torch.float32),
            torch.tensor(self.frame_masks[idx], dtype=torch.bool),
            torch.tensor(point_mask, dtype=torch.bool),
            torch.tensor(x_lstm, dtype=torch.float32),
            torch.tensor(x_gru, dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Projection utilities and TD-CNN-Transformer baseline
# ---------------------------------------------------------------------------

def project_points_to_grid(x, point_mask, grid_size=32):
    """Build [B,T,3,H,W] maps: occupancy, mean Doppler, mean peak intensity."""
    B, T, P, _ = x.shape
    H = W = grid_size
    device = x.device

    valid = point_mask.bool()
    xs = x[..., 0]
    ys = x[..., 1]
    vel = x[..., 2]
    peak = x[..., 3]

    inf = torch.tensor(float("inf"), device=device, dtype=x.dtype)
    neg_inf = torch.tensor(float("-inf"), device=device, dtype=x.dtype)

    xs_min = torch.where(valid, xs, inf).reshape(B, -1).amin(dim=1)
    xs_max = torch.where(valid, xs, neg_inf).reshape(B, -1).amax(dim=1)
    ys_min = torch.where(valid, ys, inf).reshape(B, -1).amin(dim=1)
    ys_max = torch.where(valid, ys, neg_inf).reshape(B, -1).amax(dim=1)
    empty = ~valid.reshape(B, -1).any(dim=1)
    xs_min = torch.where(empty, torch.zeros_like(xs_min), xs_min)
    xs_max = torch.where(empty, torch.ones_like(xs_max), xs_max)
    ys_min = torch.where(empty, torch.zeros_like(ys_min), ys_min)
    ys_max = torch.where(empty, torch.ones_like(ys_max), ys_max)

    x_range = (xs_max - xs_min).clamp(min=1e-6).view(B, 1, 1)
    y_range = (ys_max - ys_min).clamp(min=1e-6).view(B, 1, 1)
    ix = ((xs - xs_min.view(B, 1, 1)) / x_range * (W - 1)).long().clamp(0, W - 1)
    iy = ((ys - ys_min.view(B, 1, 1)) / y_range * (H - 1)).long().clamp(0, H - 1)

    maps = torch.zeros(B, T, 3, H, W, device=device, dtype=x.dtype)
    counts = torch.zeros(B, T, 1, H, W, device=device, dtype=x.dtype)
    b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, T, P)
    t_idx = torch.arange(T, device=device).view(1, T, 1).expand(B, T, P)
    valid_f = valid.float()

    maps_flat = maps.reshape(-1)
    counts_flat = counts.reshape(-1)

    def flat_map_index(channel):
        return (((((b_idx * T + t_idx) * 3 + channel) * H + iy) * W + ix)).reshape(-1)

    count_index = (((((b_idx * T + t_idx) * 1) * H + iy) * W + ix)).reshape(-1)
    maps_flat.scatter_add_(0, flat_map_index(0), valid_f.reshape(-1))
    maps_flat.scatter_add_(0, flat_map_index(1), torch.where(valid, vel, torch.zeros_like(vel)).reshape(-1))
    maps_flat.scatter_add_(0, flat_map_index(2), torch.where(valid, peak, torch.zeros_like(peak)).reshape(-1))
    counts_flat.scatter_add_(0, count_index, valid_f.reshape(-1))

    maps = maps_flat.reshape(B, T, 3, H, W)
    counts = counts_flat.reshape(B, T, 1, H, W)
    denom = counts.clamp(min=1.0)
    maps[:, :, 1:2] = maps[:, :, 1:2] / denom
    maps[:, :, 2:3] = maps[:, :, 2:3] / denom
    maps[:, :, 0:1] = maps[:, :, 0:1].clamp(max=1.0)
    return maps


class FrameMapEncoder(nn.Module):
    def __init__(self, out_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, maps):
        B, T, C, H, W = maps.shape
        h = self.net(maps.reshape(B * T, C, H, W)).flatten(1)
        return h.reshape(B, T, -1)


class Ref41TDCNNTransformer(nn.Module):
    def __init__(self, num_classes=12, grid_size=32, d_model=128, dropout=0.1):
        super().__init__()
        self.grid_size = grid_size
        self.encoder = FrameMapEncoder(out_dim=d_model, dropout=dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, MAX_FRAMES + 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=256,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=3)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x, frame_mask, point_mask):
        maps = project_points_to_grid(x, point_mask, self.grid_size)
        feat = self.encoder(maps)
        B, T, _ = feat.shape
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, feat], dim=1)
        tokens = tokens + self.pos_embed[:, : T + 1]
        cls_valid = torch.ones(B, 1, dtype=torch.bool, device=x.device)
        pad_mask = ~torch.cat([cls_valid, frame_mask], dim=1)
        out = self.transformer(tokens, src_key_padding_mask=pad_mask)
        return self.head(out[:, 0])


# ---------------------------------------------------------------------------
# P4Transformer adapter
# ---------------------------------------------------------------------------

def _unique_existing_parent_candidates(paths):
    seen = set()
    out = []
    for path in paths:
        if not path:
            continue
        path = os.path.abspath(os.path.expanduser(path))
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def _nearby_p4_search_roots(script_dir, cwd):
    roots = [script_dir, cwd, os.path.dirname(script_dir), os.path.dirname(cwd)]
    roots = [root for root in roots if root and os.path.isdir(root)]
    return _unique_existing_parent_candidates(roots)


def _looks_like_official_p4_dir(module_dir):
    parts = os.path.abspath(module_dir).split(os.sep)
    base = os.path.basename(module_dir)
    return (
        base.startswith("modules-pytorch")
        or any(part.lower().startswith("p4transformer") for part in parts)
    )


def _discover_nearby_p4_module_dirs(script_dir, cwd, max_depth=5):
    """Find official P4Transformer module dirs near the script for zero-argument runs."""
    found = []
    skip_dirs = {
        ".git",
        "__pycache__",
        ".ipynb_checkpoints",
        "data",
        "data_12_self",
        "Result_Contrast_Study-200-pub",
        "Result_Ablation_Study-200-pub",
        "Result-Reviewer2-SoTA-DSTG260509",
    }
    for root in _nearby_p4_search_roots(script_dir, cwd):
        root_depth = root.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, filenames in os.walk(root):
            depth = dirpath.rstrip(os.sep).count(os.sep) - root_depth
            if depth >= max_depth:
                dirnames[:] = []
            dirnames[:] = [
                dirname for dirname in dirnames
                if dirname not in skip_dirs and not dirname.startswith("Result")
            ]
            if "point_4d_convolution.py" not in filenames:
                continue
            if not _looks_like_official_p4_dir(dirpath):
                continue
            found.append(dirpath)
    return _unique_existing_parent_candidates(found)


def _p4_module_dir_candidates(p4_root=None):
    """Return candidate directories that may contain official point_4d_convolution.py."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    roots = _unique_existing_parent_candidates([
        p4_root,
        os.environ.get("P4TRANSFORMER_ROOT"),
        os.path.join(script_dir, "P4Transformer-main"),
        os.path.join(cwd, "P4Transformer-main"),
        os.path.join(script_dir, "P4Transformer"),
        os.path.join(cwd, "P4Transformer"),
        os.path.join(script_dir, "P4Transformer-master"),
        os.path.join(cwd, "P4Transformer-master"),
    ])

    candidates = []
    for root in roots:
        candidates.extend([
            os.path.join(root, "modules-pytorch-1.8.1"),
            os.path.join(root, "modules-pytorch-1.4.0"),
            os.path.join(root, "P4Transformer-main", "modules-pytorch-1.8.1"),
            os.path.join(root, "P4Transformer-main", "modules-pytorch-1.4.0"),
            os.path.join(root, "P4Transformer", "modules-pytorch-1.8.1"),
            os.path.join(root, "P4Transformer", "modules-pytorch-1.4.0"),
            os.path.join(root, "P4Transformer-master", "modules-pytorch-1.8.1"),
            os.path.join(root, "P4Transformer-master", "modules-pytorch-1.4.0"),
        ])
        if _looks_like_official_p4_dir(root):
            candidates.append(root)

    candidates.extend(_discover_nearby_p4_module_dirs(script_dir, cwd))

    return _unique_existing_parent_candidates(candidates)


def _try_load_official_p4dconv(module_dir):
    p4d_path = os.path.join(module_dir, "point_4d_convolution.py")
    if not os.path.exists(p4d_path):
        return None, "point_4d_convolution.py not found"

    original_sys_path = list(sys.path)
    try:
        sys.path.insert(0, module_dir)
        try:
            import pointnet2_ops._ext  # noqa: F401
        except ImportError:
            try:
                import pointnet2._ext as pointnet2_ext
                pointnet2_ops_pkg = types.ModuleType("pointnet2_ops")
                pointnet2_ops_pkg.__path__ = []
                pointnet2_ops_pkg._ext = pointnet2_ext
                sys.modules.setdefault("pointnet2_ops", pointnet2_ops_pkg)
                sys.modules.setdefault("pointnet2_ops._ext", pointnet2_ext)
            except ImportError:
                pass

        module_name = "official_p4dconv_" + str(abs(hash(p4d_path)))
        spec = importlib.util.spec_from_file_location(module_name, p4d_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.P4DConv, None
    except Exception as exc:
        return None, str(exc).splitlines()[0]
    finally:
        sys.path = original_sys_path


def _load_p4dconv_backend(p4_root=None):
    checked = []
    last_error = "official P4Transformer point_4d_convolution.py not found"
    for module_dir in _p4_module_dir_candidates(p4_root):
        p4d_path = os.path.join(module_dir, "point_4d_convolution.py")
        checked.append(p4d_path)
        backend, error = _try_load_official_p4dconv(module_dir)
        if backend is not None:
            version = os.path.basename(module_dir)
            if not version.startswith("modules-pytorch"):
                version = "official"
            return backend, f"official {version}", f"loaded from {p4d_path}"
        if os.path.exists(p4d_path):
            last_error = f"{p4d_path}: {error}"

    from point_4d_convolution import P4DConv
    checked_msg = "; checked: " + " | ".join(checked[:6])
    if len(checked) > 6:
        checked_msg += f" | ... ({len(checked)} candidates)"
    return P4DConv, "pure PyTorch fallback", last_error + checked_msg


P4DConvBackend = None
P4_BACKEND_NAME = "not initialized"
P4_BACKEND_DETAIL = "call _load_p4dconv_backend() before building Ref31-P4Transformer"


class P4Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class P4PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class P4FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class P4Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        b, n, _ = x.shape
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.reshape(b, n, h, -1).permute(0, 2, 1, 3) for t in qkv]
        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, n, -1)
        return self.to_out(out)


class P4TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    P4Residual(P4PreNorm(dim, P4Attention(dim, heads, dim_head, dropout=0.0))),
                    P4Residual(P4PreNorm(dim, P4FeedForward(dim, mlp_dim, dropout=dropout))),
                ])
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


class P4Transformer(nn.Module):
    def __init__(
        self,
        radius=0.7,
        nsamples=32,
        spatial_stride=32,
        temporal_kernel_size=3,
        temporal_stride=2,
        dim=1024,
        depth=5,
        heads=8,
        dim_head=128,
        mlp_dim=2048,
        num_classes=12,
        dropout=0.0,
    ):
        super().__init__()
        global P4DConvBackend, P4_BACKEND_NAME, P4_BACKEND_DETAIL
        if P4DConvBackend is None:
            P4DConvBackend, P4_BACKEND_NAME, P4_BACKEND_DETAIL = _load_p4dconv_backend()
        self.tube_embedding = P4DConvBackend(
            in_planes=0,
            mlp_planes=[dim],
            mlp_batch_norm=[False],
            mlp_activation=[False],
            spatial_kernel_size=[radius, nsamples],
            spatial_stride=spatial_stride,
            temporal_kernel_size=temporal_kernel_size,
            temporal_stride=temporal_stride,
            temporal_padding=[1, 0],
            operator="+",
            spatial_pooling="max",
            temporal_pooling="max",
        )
        self.pos_embedding = nn.Conv1d(4, dim, kernel_size=1)
        self.transformer = P4TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout=dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, x, frame_mask, point_mask):
        xyz = x[..., :3] * point_mask.unsqueeze(-1).float()
        xyzs, features = self.tube_embedding(xyz)
        xyzts = []
        for t, xyz_t in enumerate(torch.unbind(xyzs, dim=1)):
            t_feature = torch.ones((xyz_t.size(0), xyz_t.size(1), 1), dtype=x.dtype, device=x.device) * (t + 1)
            xyzts.append(torch.cat((xyz_t.contiguous(), t_feature), dim=2))
        xyzts = torch.stack(xyzts, dim=1)
        xyzts = xyzts.reshape(xyzts.shape[0], xyzts.shape[1] * xyzts.shape[2], xyzts.shape[3])
        features = features.permute(0, 1, 3, 2)
        features = features.reshape(features.shape[0], features.shape[1] * features.shape[2], features.shape[3])
        xyzts = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)
        output = self.transformer(xyzts + features)
        output = torch.max(output, dim=1, keepdim=False)[0]
        return self.mlp_head(output)


# ---------------------------------------------------------------------------
# Submitted DSTG-Net
# ---------------------------------------------------------------------------

class FusionGestureModel(nn.Module):
    def __init__(
        self,
        lstm_in=240,
        gru_in=58,
        lstm_hidden=32,
        gru_hidden=64,
        num_layers=2,
        num_classes=12,
        dropout_rate=0.35,
    ):
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = num_layers
        self.lstm = nn.LSTM(lstm_in, lstm_hidden, num_layers, batch_first=True, dropout=0.1)
        self.lstm_fc = nn.Sequential(
            nn.Linear(lstm_hidden * MAX_FRAMES, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.gru_proj = nn.Sequential(
            nn.Linear(gru_in, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            64,
            gru_hidden,
            num_layers,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(gru_hidden * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Dropout(dropout_rate),
        )
        self.gru_fc = nn.Sequential(
            nn.Linear(gru_hidden * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.fusion_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )

    def forward(self, x_lstm, x_gru):
        h0 = torch.zeros(self.lstm_layers, x_lstm.size(0), self.lstm_hidden, device=x_lstm.device)
        c0 = torch.zeros(self.lstm_layers, x_lstm.size(0), self.lstm_hidden, device=x_lstm.device)
        out_lstm, _ = self.lstm(x_lstm, (h0, c0))
        feat_lstm = self.lstm_fc(out_lstm.reshape(out_lstm.shape[0], -1))
        batch_size, seq_len, _ = x_gru.shape
        x_g_proj = self.gru_proj(x_gru.reshape(-1, x_gru.size(-1))).reshape(batch_size, seq_len, -1)
        gru_out, _ = self.gru(x_g_proj)
        attn_w = torch.softmax(self.attention(gru_out).squeeze(-1), dim=1).unsqueeze(-1)
        feat_gru = self.gru_fc(torch.sum(attn_w * gru_out, dim=1))
        return self.fusion_classifier(torch.cat([feat_lstm, feat_gru], dim=1))


# ---------------------------------------------------------------------------
# Optional TAGEGestureNet support
# ---------------------------------------------------------------------------

try:
    from torch_geometric.loader import DataLoader as GraphDataLoader
    from TAGEGestureNet import (
        GestureGraphDataset,
        GraphConstructor,
        TAGEGestureNet,
        compute_graph_flops,
    )
    TAGE_AVAILABLE = True
except Exception as exc:
    print(f"Warning: TAGEGestureNet dependencies unavailable: {exc}")
    GraphDataLoader = None
    GestureGraphDataset = None
    GraphConstructor = None
    TAGEGestureNet = None
    compute_graph_flops = None
    TAGE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Forward, training, evaluation, latency
# ---------------------------------------------------------------------------

def forward_model(model, batch, input_type):
    if input_type == "point":
        points, frame_mask, point_mask, _, _, y = batch
        points = points.to(DEVICE, non_blocking=PIN_MEMORY)
        frame_mask = frame_mask.to(DEVICE, non_blocking=PIN_MEMORY)
        point_mask = point_mask.to(DEVICE, non_blocking=PIN_MEMORY)
        y = y.to(DEVICE, non_blocking=PIN_MEMORY)
        return model(points, frame_mask, point_mask), y
    if input_type == "dstg":
        _, _, _, x_lstm, x_gru, y = batch
        x_lstm = x_lstm.to(DEVICE, non_blocking=PIN_MEMORY)
        x_gru = x_gru.to(DEVICE, non_blocking=PIN_MEMORY)
        y = y.to(DEVICE, non_blocking=PIN_MEMORY)
        return model(x_lstm, x_gru), y
    raise ValueError(f"Unsupported input_type: {input_type}")


def train_one_epoch(model, loader, criterion, optimizer, input_type, model_name):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    for batch in tqdm(loader, desc=f"{model_name} Train", leave=False):
        optimizer.zero_grad()
        outputs, y = forward_model(model, batch, input_type)
        loss = criterion(outputs, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(y.detach().cpu().numpy())
    return (
        total_loss / max(1, len(loader)),
        accuracy_score(all_targets, all_preds),
        f1_score(all_targets, all_preds, average="macro", zero_division=0),
    )


@torch.no_grad()
def evaluate(model, loader, criterion, input_type, model_name):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    for batch in tqdm(loader, desc=f"{model_name} Eval", leave=False):
        outputs, y = forward_model(model, batch, input_type)
        loss = criterion(outputs, y)
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(y.detach().cpu().numpy())
    return {
        "loss": total_loss / max(1, len(loader)),
        "acc": accuracy_score(all_targets, all_preds),
        "macro_precision": precision_score(all_targets, all_preds, average="macro", zero_division=0),
        "macro_recall": recall_score(all_targets, all_preds, average="macro", zero_division=0),
        "macro_f1": f1_score(all_targets, all_preds, average="macro", zero_division=0),
        "preds": np.asarray(all_preds, dtype=np.int64),
        "targets": np.asarray(all_targets, dtype=np.int64),
    }


@torch.no_grad()
def benchmark_inference(model, loader, input_type, warmup_batches=BENCHMARK_WARMUP_BATCHES, repeats=BENCHMARK_REPEATS):
    model.eval()
    elapsed_times = []
    per_sample_ms = []
    details = []
    total_samples = 0
    for repeat_idx in range(repeats):
        for batch_idx, batch in enumerate(loader):
            if input_type == "point":
                points, frame_mask, point_mask, _, _, _ = batch
                points = points.to(DEVICE, non_blocking=PIN_MEMORY)
                frame_mask = frame_mask.to(DEVICE, non_blocking=PIN_MEMORY)
                point_mask = point_mask.to(DEVICE, non_blocking=PIN_MEMORY)
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(points, frame_mask, point_mask)
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                n = points.size(0)
            elif input_type == "dstg":
                _, _, _, x_lstm, x_gru, _ = batch
                x_lstm = x_lstm.to(DEVICE, non_blocking=PIN_MEMORY)
                x_gru = x_gru.to(DEVICE, non_blocking=PIN_MEMORY)
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(x_lstm, x_gru)
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                n = x_lstm.size(0)
            else:
                raise ValueError(input_type)

            elapsed = time.perf_counter() - start
            is_warmup = repeat_idx == 0 and batch_idx < warmup_batches
            if not is_warmup:
                elapsed_ms = elapsed * 1000.0
                sample_latency = elapsed_ms / max(1, n)
                elapsed_times.append(elapsed)
                per_sample_ms.append(sample_latency)
                total_samples += n
                details.append({
                    "Repeat": repeat_idx + 1,
                    "Batch": batch_idx + 1,
                    "Samples": n,
                    "Batch Latency (ms)": elapsed_ms,
                    "Per-Sample Latency (ms)": sample_latency,
                })

    if not elapsed_times or total_samples == 0:
        return 0.0, 0.0, 0.0, 0.0, []
    total_time = sum(elapsed_times)
    mean_latency_ms = (total_time / total_samples) * 1000.0
    fps = total_samples / total_time
    return (
        mean_latency_ms,
        fps,
        float(np.median(per_sample_ms)),
        float(np.percentile(per_sample_ms, 95)),
        details,
    )


def estimate_flops_with_thop(model, input_type):
    try:
        from thop import profile
        model.eval()
        if input_type == "point":
            dummy_points = torch.randn(1, MAX_FRAMES, MAX_POINTS, 4, device=DEVICE)
            dummy_frame_mask = torch.ones(1, MAX_FRAMES, dtype=torch.bool, device=DEVICE)
            dummy_point_mask = torch.ones(1, MAX_FRAMES, MAX_POINTS, dtype=torch.bool, device=DEVICE)
            flops, _ = profile(model, inputs=(dummy_points, dummy_frame_mask, dummy_point_mask), verbose=False)
        elif input_type == "dstg":
            dummy_lstm = torch.randn(1, MAX_FRAMES, 240, device=DEVICE)
            dummy_gru = torch.randn(1, MAX_FRAMES, 58, device=DEVICE)
            flops, _ = profile(model, inputs=(dummy_lstm, dummy_gru), verbose=False)
        else:
            return -1, -1.0
        return int(flops), flops / 1e6
    except Exception as exc:
        print(f"[Warning] FLOPs calculation failed: {exc}")
        return estimate_flops_analytic(model)


def linear_flops(in_features, out_features, tokens=1):
    return int(tokens * in_features * out_features * 2)


def lstm_flops(lstm, seq_len):
    num_directions = 2 if lstm.bidirectional else 1
    total = 0
    for layer_idx in range(lstm.num_layers):
        layer_input_size = lstm.input_size if layer_idx == 0 else lstm.hidden_size * num_directions
        gates = 4
        macs_per_step = gates * lstm.hidden_size * (layer_input_size + lstm.hidden_size)
        total += seq_len * num_directions * macs_per_step * 2
    return int(total)


def gru_flops(gru, seq_len):
    num_directions = 2 if gru.bidirectional else 1
    total = 0
    for layer_idx in range(gru.num_layers):
        layer_input_size = gru.input_size if layer_idx == 0 else gru.hidden_size * num_directions
        gates = 3
        macs_per_step = gates * gru.hidden_size * (layer_input_size + gru.hidden_size)
        total += seq_len * num_directions * macs_per_step * 2
    return int(total)


def transformer_encoder_flops(seq_len, d_model, num_heads, dim_feedforward, num_layers):
    head_dim = d_model // num_heads
    mha_per_layer = (
        3 * linear_flops(d_model, d_model, seq_len) +
        linear_flops(d_model, d_model, seq_len) +
        2 * num_heads * seq_len * seq_len * head_dim * 2
    )
    ffn_per_layer = (
        linear_flops(d_model, dim_feedforward, seq_len) +
        linear_flops(dim_feedforward, d_model, seq_len)
    )
    return int(num_layers * (mha_per_layer + ffn_per_layer))


def conv2d_flops(h, w, in_ch, out_ch, kernel_size):
    return int(h * w * out_ch * in_ch * kernel_size * kernel_size * 2)


def estimate_flops_analytic(model):
    """Fallback single-sample FLOPs estimate when thop is unavailable."""
    if isinstance(model, FusionGestureModel):
        total = 0
        total += lstm_flops(model.lstm, MAX_FRAMES)
        total += linear_flops(model.lstm_hidden * MAX_FRAMES, 128)

        total += linear_flops(58, 128, MAX_FRAMES)
        total += linear_flops(128, 64, MAX_FRAMES)
        total += gru_flops(model.gru, MAX_FRAMES)
        total += linear_flops(model.gru.hidden_size * 2, 64, MAX_FRAMES)
        total += linear_flops(64, 1, MAX_FRAMES)
        total += linear_flops(model.gru.hidden_size * 2, 128)

        total += linear_flops(256, 128)
        total += linear_flops(128, 64)
        total += linear_flops(64, NUM_CLASSES)
        return int(total), total / 1e6

    if isinstance(model, Ref41TDCNNTransformer):
        d_model = model.cls_token.shape[-1]
        total = 0
        total += MAX_FRAMES * conv2d_flops(32, 32, 3, 32, 3)
        total += MAX_FRAMES * conv2d_flops(16, 16, 32, 64, 3)
        total += MAX_FRAMES * conv2d_flops(8, 8, 64, d_model, 3)
        seq_len = MAX_FRAMES + 1
        first_layer = model.transformer.layers[0]
        total += transformer_encoder_flops(
            seq_len=seq_len,
            d_model=d_model,
            num_heads=first_layer.self_attn.num_heads,
            dim_feedforward=first_layer.linear1.out_features,
            num_layers=len(model.transformer.layers),
        )
        total += linear_flops(d_model, NUM_CLASSES)
        return int(total), total / 1e6

    if isinstance(model, P4Transformer):
        # Coarse estimate for the official-style P4Transformer adapter. The
        # P4DConv grouping cost is implementation-dependent, so the estimate
        # covers the dominant embedding, temporal Transformer, and classifier.
        dim = model.pos_embedding.out_channels
        spatial_tokens = max(1, MAX_POINTS // model.tube_embedding.spatial_stride)
        temporal_tokens = ((MAX_FRAMES + sum(model.tube_embedding.temporal_padding) -
                            model.tube_embedding.temporal_kernel_size) //
                           model.tube_embedding.temporal_stride) + 1
        seq_len = max(1, spatial_tokens * temporal_tokens)
        total = 0
        total += seq_len * linear_flops(4, dim)
        total += transformer_encoder_flops(
            seq_len=seq_len,
            d_model=dim,
            num_heads=8,
            dim_feedforward=2048,
            num_layers=5,
        )
        total += linear_flops(dim, 2048)
        total += linear_flops(2048, NUM_CLASSES)
        return int(total), total / 1e6

    return -1, -1.0


def save_confusion_matrix(y_true, y_pred, model_name, result_dir):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    pd.DataFrame(cm, index=DISPLAY_GESTURES, columns=DISPLAY_GESTURES).to_csv(
        os.path.join(result_dir, f"Confusion_Matrix_Count_{sanitize_name(model_name)}.csv"),
        encoding="utf-8-sig",
    )
    cm_percent = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    pd.DataFrame(cm_percent * 100.0, index=DISPLAY_GESTURES, columns=DISPLAY_GESTURES).to_csv(
        os.path.join(result_dir, f"Confusion_Matrix_Percent_{sanitize_name(model_name)}.csv"),
        encoding="utf-8-sig",
    )


def per_class_accuracy(y_true, y_pred):
    rows = []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    for cls_idx, cls_name in enumerate(DISPLAY_GESTURES):
        mask = y_true == cls_idx
        acc = float(np.mean(y_pred[mask] == y_true[mask]) * 100.0) if np.any(mask) else np.nan
        rows.append({"Class": cls_name, "Accuracy (%)": acc, "Samples": int(mask.sum())})
    return rows


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_model(method_name):
    if method_name == "Ref31-P4Transformer":
        model = P4Transformer(
            radius=0.7,
            nsamples=32,
            spatial_stride=32,
            temporal_kernel_size=3,
            temporal_stride=2,
            dim=1024,
            depth=5,
            heads=8,
            dim_head=128,
            mlp_dim=2048,
            num_classes=NUM_CLASSES,
        )
        return model.to(DEVICE), "point"
    if method_name == "Ref41-TD-CNN-Transformer":
        return Ref41TDCNNTransformer(NUM_CLASSES).to(DEVICE), "point"
    if method_name == "DSTG-Net":
        return FusionGestureModel(num_classes=NUM_CLASSES).to(DEVICE), "dstg"
    raise ValueError(f"Unknown non-graph method: {method_name}")


def method_batch_size(method_name, memory_adjusted=False):
    if method_name == "Ref31-P4Transformer":
        return P4_MEMORY_BATCH_SIZE if memory_adjusted else P4_BATCH_SIZE
    return BATCH_SIZE


# ---------------------------------------------------------------------------
# TAGE graph method
# ---------------------------------------------------------------------------

def run_tage_method(shared_split, result_dir):
    method_name = "Ref42-TAGEGestureNet"
    spec = METHOD_SPECS[method_name]
    if not TAGE_AVAILABLE:
        return {
            "Model": method_name,
            **spec,
            "Note": "Skipped: TAGEGestureNet dependencies are unavailable.",
        }

    print("\n" + "=" * 100)
    print(f"Running adapted reproduction: {method_name}")
    print("=" * 100)

    graph_constructor = GraphConstructor(
        d_th=0.3,
        min_samples=7,
        alpha1=4.0,
        alpha2=4.0,
        beta=0.8,
        frame_interval=0.033,
    )
    graph_dataset = GestureGraphDataset(
        data_dir=DATA_PATH,
        gesture_classes=GESTURES,
        graph_constructor=graph_constructor,
        transform=None,
    )
    train_idx, val_idx = shared_split
    train_dataset = Subset(graph_dataset, train_idx)
    val_dataset = Subset(graph_dataset, val_idx)
    train_loader = GraphDataLoader(train_dataset, batch_size=GRAPH_BATCH_SIZE, shuffle=True)
    val_loader = GraphDataLoader(val_dataset, batch_size=GRAPH_BATCH_SIZE, shuffle=False)

    model = TAGEGestureNet(
        in_dim=4,
        edge_dim=10,
        hidden_dim=64,
        graph_dim=1024,
        num_layers=2,
        num_classes=NUM_CLASSES,
        descriptor_dim=5,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_PLATEAU_FACTOR,
        patience=LR_PLATEAU_PATIENCE,
        threshold=LR_PLATEAU_THRESHOLD,
        cooldown=LR_PLATEAU_COOLDOWN,
        min_lr=MIN_LR,
    )

    best_acc = -1.0
    best_state = None
    best_epoch = 0
    best_result = None
    history = []
    logs = []
    no_improve = 0
    safe_name = sanitize_name(method_name)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        for batch in tqdm(train_loader, desc=f"{method_name} Train", leave=False):
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item()
            all_preds.extend(logits.argmax(dim=1).detach().cpu().numpy())
            all_targets.extend(batch.y.detach().cpu().numpy())
        train_loss = total_loss / max(1, len(train_loader))
        train_acc = accuracy_score(all_targets, all_preds)
        train_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"{method_name} Eval", leave=False):
                batch = batch.to(DEVICE)
                logits = model(batch)
                loss = criterion(logits, batch.y)
                val_loss += loss.item()
                val_preds.extend(logits.argmax(dim=1).detach().cpu().numpy())
                val_targets.extend(batch.y.detach().cpu().numpy())
        val_loss = val_loss / max(1, len(val_loader))
        val_acc = accuracy_score(val_targets, val_preds)
        val_precision = precision_score(val_targets, val_preds, average="macro", zero_division=0)
        val_recall = recall_score(val_targets, val_preds, average="macro", zero_division=0)
        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        scheduler.step(val_loss)

        line = (
            f"[{method_name}] Epoch [{epoch}/{EPOCHS}] - Loss: {train_loss:.4f} - "
            f"Train Acc: {train_acc*100:.2f}% - Val Loss: {val_loss:.4f} - "
            f"Val Acc: {val_acc*100:.2f}% - Val F1: {val_f1*100:.2f}% - "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )
        print(line, flush=True)
        logs.append(line)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc * 100,
            "train_f1": train_f1 * 100,
            "val_loss": val_loss,
            "val_acc": val_acc * 100,
            "val_f1": val_f1 * 100,
        })

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_result = {
                "loss": val_loss,
                "acc": val_acc,
                "macro_precision": val_precision,
                "macro_recall": val_recall,
                "macro_f1": val_f1,
                "preds": np.asarray(val_preds, dtype=np.int64),
                "targets": np.asarray(val_targets, dtype=np.int64),
            }
            no_improve = 0
            torch.save(
                {
                    "model_name": method_name,
                    "model_state_dict": best_state,
                    "best_epoch": best_epoch,
                    "checkpoint_criterion": CHECKPOINT_CRITERION,
                    "accuracy": best_acc * 100,
                    "parameters": count_trainable_params(model),
                },
                os.path.join(result_dir, f"best_model_{safe_name}.pth"),
            )
        else:
            no_improve += 1
            if EARLY_STOPPING_PATIENCE is not None and no_improve >= EARLY_STOPPING_PATIENCE:
                break

    with open(os.path.join(result_dir, f"{safe_name}_training_logs.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(logs))
    pd.DataFrame(history).to_csv(os.path.join(result_dir, f"history_{safe_name}.csv"), index=False)

    if best_state is not None:
        model.load_state_dict(best_state)
    if best_result is not None:
        val_loss = best_result["loss"]
        val_acc = best_result["acc"]
        val_precision = best_result["macro_precision"]
        val_recall = best_result["macro_recall"]
        val_f1 = best_result["macro_f1"]
        val_preds = best_result["preds"]
        val_targets = best_result["targets"]

    params = count_trainable_params(model)
    try:
        flops = compute_graph_flops(model, DEVICE, num_nodes=128, num_edges=512)
        mflops = flops / 1e6
    except Exception as exc:
        print(f"[Warning] graph FLOPs failed: {exc}")
        flops, mflops = -1, -1.0

    latency_ms, fps, median_ms, p95_ms = benchmark_graph_inference(model, val_loader)
    pd.DataFrame(per_class_accuracy(val_targets, val_preds)).to_csv(
        os.path.join(result_dir, f"Per_Class_Accuracy_{safe_name}.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    save_confusion_matrix(val_targets, val_preds, method_name, result_dir)

    return {
        "Model": method_name,
        **spec,
        "Trainable Parameters": params,
        "FLOPs": flops,
        "MFLOPs": mflops,
        "Best Epoch": best_epoch,
        "Checkpoint Criterion": CHECKPOINT_CRITERION,
        "Accuracy (%)": val_acc * 100,
        "Macro-Precision (%)": val_precision * 100,
        "Macro-Recall (%)": val_recall * 100,
        "Macro-F1 (%)": val_f1 * 100,
        "Latency (ms/sample)": latency_ms,
        "Median Latency (ms/sample)": median_ms,
        "P95 Latency (ms/sample)": p95_ms,
        "FPS": fps,
        "Device": f"PyTorch {DEVICE}",
        "Seed": SEED,
        "Epochs": EPOCHS,
        "Batch Size": GRAPH_BATCH_SIZE,
        "Learning Rate": LR,
        "Optimizer": "AdamW",
        "Scheduler": "ReduceLROnPlateau(mode=min)",
    }


@torch.no_grad()
def benchmark_graph_inference(model, loader, repeats=BENCHMARK_REPEATS):
    model.eval()
    elapsed_times = []
    per_sample_ms = []
    total_samples = 0
    for repeat_idx in range(repeats):
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(DEVICE)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(batch)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            n = int(batch.y.numel())
            elapsed = time.perf_counter() - start
            is_warmup = repeat_idx == 0 and batch_idx < BENCHMARK_WARMUP_BATCHES
            if not is_warmup:
                elapsed_times.append(elapsed)
                per_sample_ms.append(elapsed * 1000.0 / max(1, n))
                total_samples += n
    if not elapsed_times or total_samples == 0:
        return 0.0, 0.0, 0.0, 0.0
    total_time = sum(elapsed_times)
    return (
        (total_time / total_samples) * 1000.0,
        total_samples / total_time,
        float(np.median(per_sample_ms)),
        float(np.percentile(per_sample_ms, 95)),
    )


# ---------------------------------------------------------------------------
# Standard method runner
# ---------------------------------------------------------------------------

def run_one_standard_method(method_name, dataset, shared_split, result_dir, memory_adjusted=False):
    print("\n" + "=" * 100)
    print(f"Running adapted reproduction: {method_name}")
    print("=" * 100)
    spec = METHOD_SPECS[method_name]
    run_note = spec["Note"]
    if method_name == "Ref31-P4Transformer":
        run_note = f"{run_note} P4DConv backend: {P4_BACKEND_NAME} ({P4_BACKEND_DETAIL})."
        if memory_adjusted:
            run_note = f"{run_note} Retried with memory-adjusted batch size."
    print(run_note)

    model, input_type = build_model(method_name)
    train_idx, val_idx = shared_split
    batch_size = method_batch_size(method_name, memory_adjusted=memory_adjusted)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_idx),
        num_workers=0,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_idx),
        num_workers=0,
        pin_memory=PIN_MEMORY,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_PLATEAU_FACTOR,
        patience=LR_PLATEAU_PATIENCE,
        threshold=LR_PLATEAU_THRESHOLD,
        cooldown=LR_PLATEAU_COOLDOWN,
        min_lr=MIN_LR,
    )

    best_acc = -1.0
    best_state = None
    best_epoch = 0
    best_result = None
    history = []
    logs = []
    no_improve = 0
    safe_name = sanitize_name(method_name)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, input_type, method_name
        )
        val_result = evaluate(model, val_loader, criterion, input_type, method_name)
        scheduler.step(val_result["loss"])

        line = (
            f"[{method_name}] Epoch [{epoch}/{EPOCHS}] - Loss: {train_loss:.4f} - "
            f"Train Acc: {train_acc*100:.2f}% - Val Loss: {val_result['loss']:.4f} - "
            f"Val Acc: {val_result['acc']*100:.2f}% - Val F1: {val_result['macro_f1']*100:.2f}% - "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )
        print(line, flush=True)
        logs.append(line)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc * 100,
            "train_f1": train_f1 * 100,
            "val_loss": val_result["loss"],
            "val_acc": val_result["acc"] * 100,
            "val_f1": val_result["macro_f1"] * 100,
        })

        if val_result["acc"] > best_acc:
            best_acc = val_result["acc"]
            best_epoch = epoch
            best_result = {k: v for k, v in val_result.items()}
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            torch.save(
                {
                    "model_name": method_name,
                    "model_state_dict": best_state,
                    "best_epoch": best_epoch,
                    "checkpoint_criterion": CHECKPOINT_CRITERION,
                    "accuracy": best_acc * 100,
                    "macro_f1": val_result["macro_f1"] * 100,
                    "parameters": count_trainable_params(model),
                },
                os.path.join(result_dir, f"best_model_{safe_name}.pth"),
            )
        else:
            no_improve += 1
            if EARLY_STOPPING_PATIENCE is not None and no_improve >= EARLY_STOPPING_PATIENCE:
                break

    with open(os.path.join(result_dir, f"{safe_name}_training_logs.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(logs))
    pd.DataFrame(history).to_csv(os.path.join(result_dir, f"history_{safe_name}.csv"), index=False)

    if best_state is not None:
        model.load_state_dict(best_state)
    if best_result is None:
        best_result = evaluate(model, val_loader, criterion, input_type, method_name)

    latency_ms, fps, median_ms, p95_ms, latency_details = benchmark_inference(model, val_loader, input_type)
    pd.DataFrame(latency_details).to_csv(
        os.path.join(result_dir, f"Inference_Latency_Details_{safe_name}.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    flops, mflops = estimate_flops_with_thop(model, input_type)
    save_confusion_matrix(best_result["targets"], best_result["preds"], method_name, result_dir)
    pd.DataFrame(per_class_accuracy(best_result["targets"], best_result["preds"])).to_csv(
        os.path.join(result_dir, f"Per_Class_Accuracy_{safe_name}.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "Model": method_name,
        **spec,
        "Note": run_note,
        "Trainable Parameters": count_trainable_params(model),
        "FLOPs": flops,
        "MFLOPs": mflops,
        "Best Epoch": best_epoch,
        "Checkpoint Criterion": CHECKPOINT_CRITERION,
        "Accuracy (%)": best_result["acc"] * 100,
        "Macro-Precision (%)": best_result["macro_precision"] * 100,
        "Macro-Recall (%)": best_result["macro_recall"] * 100,
        "Macro-F1 (%)": best_result["macro_f1"] * 100,
        "Latency (ms/sample)": latency_ms,
        "Median Latency (ms/sample)": median_ms,
        "P95 Latency (ms/sample)": p95_ms,
        "FPS": fps,
        "Device": f"PyTorch {DEVICE}",
        "Seed": SEED,
        "Epochs": EPOCHS,
        "Batch Size": batch_size,
        "Learning Rate": LR,
        "Optimizer": "AdamW",
        "Scheduler": "ReduceLROnPlateau(mode=min)",
    }


def run_one_method(method_name, dataset, shared_split, result_dir):
    if method_name == "Ref42-TAGEGestureNet":
        return run_tage_method(shared_split, result_dir)
    try:
        return run_one_standard_method(method_name, dataset, shared_split, result_dir, memory_adjusted=False)
    except RuntimeError as exc:
        if method_name == "Ref31-P4Transformer" and is_cuda_oom(exc):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[Ref31-P4Transformer] CUDA OOM; retrying with smaller batch size.")
            return run_one_standard_method(method_name, dataset, shared_split, result_dir, memory_adjusted=True)
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_shared_split(dataset, result_dir):
    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=dataset.y,
        random_state=SEED,
    )
    np.savez(
        os.path.join(result_dir, "shared_split_indices.npz"),
        train_idx=train_idx,
        val_idx=val_idx,
    )
    print("=" * 80)
    print("Shared split")
    print("=" * 80)
    print(f"Train samples: {len(train_idx)}")
    print(f"Eval samples : {len(val_idx)}")
    return train_idx, val_idx


def save_paper_table(rows, result_dir):
    preferred_order = [
        "Ref31-P4Transformer",
        "Ref41-TD-CNN-Transformer",
        "Ref42-TAGEGestureNet",
        "DSTG-Net",
    ]
    rank = {name: i for i, name in enumerate(preferred_order)}
    rows = sorted(rows, key=lambda r: rank.get(r.get("Model", ""), len(rank)))
    paper_rows = []
    for row in rows:
        paper_rows.append({
            "Model": row.get("Model", ""),
            "Reference": row.get("Reference", ""),
            "Input": row.get("Input", ""),
            "Trainable Parameters": row.get("Trainable Parameters", ""),
            "MFLOPs": round(row["MFLOPs"], 3) if isinstance(row.get("MFLOPs"), (int, float)) and row["MFLOPs"] >= 0 else "",
            "Best Epoch": row.get("Best Epoch", ""),
            "Accuracy (%)": round(row["Accuracy (%)"], 2) if isinstance(row.get("Accuracy (%)"), (int, float)) else "",
            "Macro-Precision (%)": round(row["Macro-Precision (%)"], 2) if isinstance(row.get("Macro-Precision (%)"), (int, float)) else "",
            "Macro-Recall (%)": round(row["Macro-Recall (%)"], 2) if isinstance(row.get("Macro-Recall (%)"), (int, float)) else "",
            "Macro-F1 (%)": round(row["Macro-F1 (%)"], 2) if isinstance(row.get("Macro-F1 (%)"), (int, float)) else "",
            "Latency (ms/sample)": round(row["Latency (ms/sample)"], 3) if isinstance(row.get("Latency (ms/sample)"), (int, float)) else "",
            "P95 Latency (ms/sample)": round(row["P95 Latency (ms/sample)"], 3) if isinstance(row.get("P95 Latency (ms/sample)"), (int, float)) else "",
            "FPS": round(row["FPS"], 2) if isinstance(row.get("FPS"), (int, float)) else "",
            "Note": row.get("Note", ""),
        })
    pd.DataFrame(paper_rows).to_csv(
        os.path.join(result_dir, "Paper_SoTA_Table.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run Reviewer-2 SoTA comparison for DSTG-Net 260509.")
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument("--result-dir", default=RESULT_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--p4-batch-size", type=int, default=P4_BATCH_SIZE)
    parser.add_argument(
        "--p4-root",
        default=None,
        help=(
            "Path to P4Transformer-main or its modules-pytorch-* directory. "
            "The P4TRANSFORMER_ROOT environment variable is also supported."
        ),
    )
    parser.add_argument("--graph-batch-size", type=int, default=GRAPH_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Optional debug/smoke-test cap per gesture class. Omit for full-data experiments.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["Ref31-P4Transformer", "Ref41-TD-CNN-Transformer", "Ref42-TAGEGestureNet", "DSTG-Net"],
        choices=list(METHOD_SPECS.keys()),
    )
    parser.add_argument("--table-only", action="store_true", help="Regenerate Paper_SoTA_Table.csv from SoTA_Summary.csv.")
    return parser.parse_args()


def main():
    global DATA_PATH, RESULT_DIR, EPOCHS, BATCH_SIZE, P4_BATCH_SIZE, GRAPH_BATCH_SIZE, LR, WEIGHT_DECAY
    global P4DConvBackend, P4_BACKEND_NAME, P4_BACKEND_DETAIL

    args = parse_args()
    DATA_PATH = args.data_path
    RESULT_DIR = args.result_dir
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    P4_BATCH_SIZE = args.p4_batch_size
    GRAPH_BATCH_SIZE = args.graph_batch_size
    LR = args.lr
    WEIGHT_DECAY = args.weight_decay
    if args.p4_root:
        os.environ["P4TRANSFORMER_ROOT"] = args.p4_root
    P4DConvBackend, P4_BACKEND_NAME, P4_BACKEND_DETAIL = _load_p4dconv_backend(args.p4_root)
    ensure_dir(RESULT_DIR)

    if args.table_only:
        summary_path = os.path.join(RESULT_DIR, "SoTA_Summary.csv")
        if not os.path.exists(summary_path):
            raise FileNotFoundError(summary_path)
        rows = pd.read_csv(summary_path).to_dict("records")
        save_paper_table(rows, RESULT_DIR)
        print(f"Saved: {os.path.join(RESULT_DIR, 'Paper_SoTA_Table.csv')}")
        return

    set_seed(SEED)
    print("=" * 100)
    print("Reviewer-2 SoTA comparison for submitted DSTG-Net 260509")
    print("=" * 100)
    print(f"Device     : {DEVICE}")
    print(f"DATA_PATH  : {DATA_PATH}")
    print(f"RESULT_DIR : {RESULT_DIR}")
    print(f"Epochs     : {EPOCHS}")
    print(f"Methods    : {args.methods}")
    print(f"P4 backend : {P4_BACKEND_NAME} ({P4_BACKEND_DETAIL})")

    samples, _ = collect_samples(DATA_PATH, max_samples_per_class=args.max_samples_per_class)
    if args.max_samples_per_class is not None:
        print(f"Debug cap  : first {args.max_samples_per_class} samples per class")
    raw_dataset = SotaRadarDataset(samples)
    shared_split = build_shared_split(raw_dataset, RESULT_DIR)
    norm_dataset = TrainSetNormalizedSotaDataset(raw_dataset, shared_split[0])

    results = []
    for method_name in args.methods:
        set_seed(SEED)
        try:
            row = run_one_method(method_name, norm_dataset, shared_split, RESULT_DIR)
        except Exception as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[ERROR] {method_name} failed: {exc}")
            spec = METHOD_SPECS[method_name]
            row = {
                "Model": method_name,
                **spec,
                "Note": f"Failed: {exc}",
            }
        results.append(row)
        pd.DataFrame(results).to_csv(
            os.path.join(RESULT_DIR, "SoTA_Summary_partial.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    summary_path = os.path.join(RESULT_DIR, "SoTA_Summary.csv")
    pd.DataFrame(results).to_csv(summary_path, index=False, encoding="utf-8-sig")
    save_paper_table(results, RESULT_DIR)

    print("\n" + "=" * 100)
    print("Final SoTA comparison results")
    print("=" * 100)
    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved paper table: {os.path.join(RESULT_DIR, 'Paper_SoTA_Table.csv')}")


if __name__ == "__main__":
    main()
