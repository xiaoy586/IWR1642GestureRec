#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reviewer-2 statistical validation for the submitted 260509 lightweight DSTG-Net.

This script repeats the SoTA comparison over multiple random seeds and produces:
  - repeated_run_results.csv
  - statistical_summary.csv
  - pairwise_significance_tests.csv
  - table_statistical_validation_red.tex

The implementation keeps two protocols explicit:
  - CNN, LSTM, and STGNet follow 对比实验补充260509-深度学习-pytorch.py
    so the repeated runs remain comparable with the existing LaTeX baseline
    results.
  - Ref31/Ref41/Ref42 and DSTG-Net follow run_reviewer2_sota_dstg260509.py.

All methods use the same repeated-run optimizer, scheduler, gradient clipping,
evaluation, and best-validation-accuracy checkpoint logic within their protocol;
only the input adapter differs across sequence, STG, DSTG, point-cloud, and graph
representations.

Default full experiment, using the same 200-epoch training budget as the main
comparison:
  python3 run_reviewer2_statistical_validation_dstg260509.py

Debug smoke test:
  python3 run_reviewer2_statistical_validation_dstg260509.py --epochs 1 --seeds 42,43 --methods CNN LSTM DSTG-Net --max-samples-per-class 10
"""

import argparse
import copy
import os
import time
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from tqdm import tqdm

import run_reviewer2_sota_dstg260509 as sota


DEFAULT_RESULT_DIR = "Result-Reviewer2-StatisticalValidation-DSTG260509"
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_EPOCHS = 200
DEFAULT_METHODS = [
    "CNN",
    "LSTM",
    "STGNet",
    "Ref31-P4Transformer",
    "Ref41-TD-CNN-Transformer",
    "Ref42-TAGEGestureNet",
    "DSTG-Net",
]
REFERENCE_METHOD = "DSTG-Net"
SEQ_LEN = 100
STG_FEATURE_LEN = 32

CLASSIC_METHOD_SPECS = {
    "CNN": {
        "Reference": "Baseline",
        "Family": "1D convolutional network",
        "Input": "Serialized radar sequence [100,4]",
        "Note": "Repeated-run validation using the CNN baseline from the 260509 PyTorch comparison.",
    },
    "LSTM": {
        "Reference": "Ablation baseline",
        "Family": "Single-stream LSTM",
        "Input": "LSTM branch feature [80,240]",
        "Note": "Repeated-run validation of the LSTM spatial-branch baseline.",
    },
    "STGNet": {
        "Reference": "Baseline",
        "Family": "Spatial-temporal gesture network",
        "Input": "Engineered STG feature [32,1]",
        "Note": "Repeated-run validation using the STGNet baseline from the 260509 PyTorch comparison.",
    },
}
ALL_METHOD_SPECS = {**CLASSIC_METHOD_SPECS, **sota.METHOD_SPECS}


def parse_seed_list(text):
    seeds = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            seeds.append(int(item))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_radar_csv(file_path):
    df = pd.read_csv(file_path)
    required_cols = {"FrameNumber", "Velocity", "x", "y"}
    if required_cols.issubset(set(df.columns)):
        return df
    return pd.read_csv(file_path, skiprows=1)


def process_sequence_file(file_path):
    """Serialized CNN input: [100, 4] = Range, Velocity, x, y."""
    df = read_radar_csv(file_path)
    cols = ["Range", "Velocity", "x", "y"]
    if set(cols).issubset(df.columns):
        values = df[cols].values.astype(np.float32)
    else:
        cols_all = df.columns.tolist()
        values = df[[cols_all[2], cols_all[3], cols_all[5], cols_all[6]]].values.astype(np.float32)
    out = np.zeros((SEQ_LEN, 4), dtype=np.float32)
    n = min(SEQ_LEN, len(values))
    out[:n] = values[:n]
    return out


def process_stg_file(file_path):
    """STGNet engineered input: [32, 1] statistics from Range, Velocity, x, and y."""
    df = read_radar_csv(file_path)
    cols = ["Range", "Velocity", "x", "y"]
    if set(cols).issubset(df.columns):
        features = df[cols].values.astype(np.float32)
    else:
        cols_all = df.columns.tolist()
        features = df[[cols_all[2], cols_all[3], cols_all[5], cols_all[6]]].values.astype(np.float32)

    mean_f = np.mean(features, axis=0)
    std_f = np.std(features, axis=0)
    min_f = np.min(features, axis=0)
    max_f = np.max(features, axis=0)
    diff_f = np.diff(features, axis=0)
    mean_diff = np.mean(diff_f, axis=0) if len(diff_f) > 0 else np.zeros(4, dtype=np.float32)
    std_diff = np.std(diff_f, axis=0) if len(diff_f) > 0 else np.zeros(4, dtype=np.float32)
    fft_f = np.abs(np.fft.fft(features, axis=0))
    mean_fft = np.mean(fft_f, axis=0)
    std_fft = np.std(fft_f, axis=0)

    combined = np.concatenate([
        mean_f, std_f, min_f, max_f,
        mean_diff, std_diff, mean_fft, std_fft,
    ]).astype(np.float32)
    return combined.reshape(STG_FEATURE_LEN, 1)


class SingleTensorFeatureDataset(Dataset):
    def __init__(self, samples, processor, description):
        self.X = []
        self.y = []
        print(f"Loading {description}...")
        for file_path, label in tqdm(samples, leave=False):
            self.X.append(processor(file_path))
            self.y.append(label)
        self.X = [np.asarray(x, dtype=np.float32) for x in self.X]
        self.y = np.asarray(self.y, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)


class TrainSetNormalizedTensorDataset(Dataset):
    def __init__(self, source_dataset, train_idx):
        self.y = source_dataset.y
        self.X = [np.asarray(x, dtype=np.float32) for x in source_dataset.X]
        train_x = np.stack([self.X[idx] for idx in train_idx], axis=0)
        self.mean = train_x.mean(axis=0, keepdims=True)
        self.std = train_x.std(axis=0, keepdims=True)
        self.std[self.std < 1e-6] = 1.0

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = (self.X[idx] - self.mean.squeeze(0)) / self.std.squeeze(0)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)


class CNN1DBaseline(nn.Module):
    def __init__(self, input_dim=4, num_classes=12, dropout_rate=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.net(x.transpose(1, 2)))


class STGNetBaseline(nn.Module):
    def __init__(self, input_dim=1, num_classes=12, dropout_rate=0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=3),
            nn.BatchNorm1d(256, momentum=0.01),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 128, kernel_size=3),
            nn.BatchNorm1d(128, momentum=0.01),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm1 = nn.LSTM(128, 256, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(256, 128, num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        feat = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.lstm1(feat)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out[:, -1, :])
        return self.classifier(out)


class LSTMBranchBaseline(nn.Module):
    def __init__(self, input_dim=240, hidden_dim=64, num_layers=2, num_classes=12, dropout_rate=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * sota.MAX_FRAMES, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )

    def forward(self, x_lstm, x_gru=None):
        out, _ = self.lstm(x_lstm)
        return self.classifier(out.reshape(out.shape[0], -1))


def method_row_from_summary(summary_path, method_name):
    if not os.path.exists(summary_path):
        return None
    try:
        df = pd.read_csv(summary_path)
    except Exception:
        return None
    if "Model" not in df.columns:
        return None
    rows = df[df["Model"] == method_name]
    if rows.empty:
        return None
    row = rows.iloc[0].to_dict()
    note = str(row.get("Note", ""))
    if note.startswith("Failed:"):
        return None
    return row


def should_drop_last_train_batch(method_name, train_count, batch_size):
    """Avoid BatchNorm failures from a final singleton training batch."""
    if method_name == "STGNet":
        return True
    if batch_size <= 1:
        return False
    return train_count > batch_size and train_count % batch_size == 1


def save_seed_summary(rows, seed_dir):
    summary_path = os.path.join(seed_dir, "SoTA_Summary.csv")
    pd.DataFrame(rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    sota.save_paper_table(rows, seed_dir)
    return summary_path


def configure_sota(args, seed, seed_dir):
    sota.DATA_PATH = args.data_path
    sota.RESULT_DIR = seed_dir
    sota.SEED = seed
    sota.EPOCHS = args.epochs
    sota.BATCH_SIZE = args.batch_size
    sota.P4_BATCH_SIZE = args.p4_batch_size
    sota.GRAPH_BATCH_SIZE = args.graph_batch_size
    sota.LR = args.lr
    sota.WEIGHT_DECAY = args.weight_decay
    if args.p4_root:
        os.environ["P4TRANSFORMER_ROOT"] = args.p4_root
    sota.P4DConvBackend, sota.P4_BACKEND_NAME, sota.P4_BACKEND_DETAIL = sota._load_p4dconv_backend(args.p4_root)


def build_classic_model(method_name):
    if method_name == "CNN":
        return CNN1DBaseline(num_classes=sota.NUM_CLASSES).to(sota.DEVICE), "classic_single"
    if method_name == "STGNet":
        return STGNetBaseline(num_classes=sota.NUM_CLASSES).to(sota.DEVICE), "classic_single"
    if method_name == "LSTM":
        return LSTMBranchBaseline(num_classes=sota.NUM_CLASSES).to(sota.DEVICE), "classic_dstg"
    raise ValueError(f"Unknown classic method: {method_name}")


def build_non_graph_model(method_name):
    if method_name in CLASSIC_METHOD_SPECS:
        return build_classic_model(method_name)
    if method_name == "Ref42-TAGEGestureNet":
        raise ValueError("Ref42-TAGEGestureNet uses the graph-specific runner.")
    return sota.build_model(method_name)


def forward_stat_model(model, batch, input_type):
    if input_type in {"single", "classic_single"}:
        x, y = batch
        x = x.to(sota.DEVICE, non_blocking=sota.PIN_MEMORY)
        y = y.to(sota.DEVICE, non_blocking=sota.PIN_MEMORY)
        return model(x), y
    if input_type in {"classic_dstg", "lstm_only"}:
        _, _, _, x_lstm, x_gru, y = batch
        x_lstm = x_lstm.to(sota.DEVICE, non_blocking=sota.PIN_MEMORY)
        x_gru = x_gru.to(sota.DEVICE, non_blocking=sota.PIN_MEMORY)
        y = y.to(sota.DEVICE, non_blocking=sota.PIN_MEMORY)
        return model(x_lstm, x_gru), y
    if input_type == "point":
        points, frame_mask, point_mask, _, _, y = batch
        points = points.to(sota.DEVICE, non_blocking=sota.PIN_MEMORY)
        frame_mask = frame_mask.to(sota.DEVICE, non_blocking=sota.PIN_MEMORY)
        point_mask = point_mask.to(sota.DEVICE, non_blocking=sota.PIN_MEMORY)
        y = y.to(sota.DEVICE, non_blocking=sota.PIN_MEMORY)
        return model(points, frame_mask, point_mask), y
    if input_type == "dstg":
        _, _, _, x_lstm, x_gru, y = batch
        x_lstm = x_lstm.to(sota.DEVICE, non_blocking=sota.PIN_MEMORY)
        x_gru = x_gru.to(sota.DEVICE, non_blocking=sota.PIN_MEMORY)
        y = y.to(sota.DEVICE, non_blocking=sota.PIN_MEMORY)
        return model(x_lstm, x_gru), y
    if input_type == "graph":
        batch = batch.to(sota.DEVICE)
        return model(batch), batch.y
    raise ValueError(input_type)


def train_one_epoch_stat(model, loader, criterion, optimizer, input_type, model_name):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    for batch in tqdm(loader, desc=f"{model_name} Train", leave=False):
        optimizer.zero_grad()
        outputs, y = forward_stat_model(model, batch, input_type)
        loss = criterion(outputs, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=sota.GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item()
        all_preds.extend(outputs.argmax(dim=1).detach().cpu().numpy())
        all_targets.extend(y.detach().cpu().numpy())
    return (
        total_loss / max(1, len(loader)),
        accuracy_score(all_targets, all_preds),
        f1_score(all_targets, all_preds, average="macro", zero_division=0),
    )


@torch.no_grad()
def evaluate_stat(model, loader, criterion, input_type, model_name):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    for batch in tqdm(loader, desc=f"{model_name} Eval", leave=False):
        outputs, y = forward_stat_model(model, batch, input_type)
        loss = criterion(outputs, y)
        total_loss += loss.item()
        all_preds.extend(outputs.argmax(dim=1).detach().cpu().numpy())
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
def benchmark_stat(model, loader, input_type):
    if input_type in {"point", "dstg"}:
        latency_ms, fps, _, p95_ms, _ = sota.benchmark_inference(model, loader, input_type)
        return latency_ms, p95_ms, fps
    if input_type == "graph":
        model.eval()
        elapsed_times = []
        total_samples = 0
        per_sample_ms = []
        for repeat_idx in range(sota.BENCHMARK_REPEATS):
            for batch_idx, batch in enumerate(loader):
                batch = batch.to(sota.DEVICE)
                if sota.DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(batch)
                if sota.DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                n = int(batch.y.numel())
                elapsed = time.perf_counter() - start
                if repeat_idx == 0 and batch_idx < sota.BENCHMARK_WARMUP_BATCHES:
                    continue
                elapsed_times.append(elapsed)
                total_samples += n
                per_sample_ms.append(elapsed * 1000.0 / max(1, n))
        if not elapsed_times or total_samples == 0:
            return 0.0, 0.0, 0.0
        total_time = sum(elapsed_times)
        return (total_time / total_samples) * 1000.0, float(np.percentile(per_sample_ms, 95)), total_samples / total_time

    model.eval()
    elapsed_times = []
    total_samples = 0
    per_sample_ms = []
    for repeat_idx in range(sota.BENCHMARK_REPEATS):
        for batch_idx, batch in enumerate(loader):
            if input_type in {"single", "classic_single"}:
                x, _ = batch
                x = x.to(sota.DEVICE, non_blocking=sota.PIN_MEMORY)
                n = x.size(0)
                if sota.DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(x)
                if sota.DEVICE.type == "cuda":
                    torch.cuda.synchronize()
            else:
                _, _, _, x_lstm, x_gru, _ = batch
                x_lstm = x_lstm.to(sota.DEVICE, non_blocking=sota.PIN_MEMORY)
                x_gru = x_gru.to(sota.DEVICE, non_blocking=sota.PIN_MEMORY)
                n = x_lstm.size(0)
                if sota.DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(x_lstm, x_gru)
                if sota.DEVICE.type == "cuda":
                    torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if repeat_idx == 0 and batch_idx < sota.BENCHMARK_WARMUP_BATCHES:
                continue
            elapsed_times.append(elapsed)
            total_samples += n
            per_sample_ms.append(elapsed * 1000.0 / max(1, n))
    if not elapsed_times or total_samples == 0:
        return 0.0, 0.0, 0.0
    total_time = sum(elapsed_times)
    return (total_time / total_samples) * 1000.0, float(np.percentile(per_sample_ms, 95)), total_samples / total_time


def run_legacy_contrast_method(method_name, dataset, shared_split, result_dir):
    print("\n" + "=" * 100)
    print(f"Running repeated-run legacy contrast method: {method_name}")
    print("=" * 100)
    spec = CLASSIC_METHOD_SPECS[method_name]
    sota.set_seed(sota.SEED)
    model, input_type = build_classic_model(method_name)
    train_idx, val_idx = shared_split
    drop_last = should_drop_last_train_batch(method_name, len(train_idx), sota.BATCH_SIZE)
    if drop_last and method_name != "STGNet":
        print(
            f"[{method_name}] Dropping the final singleton training batch "
            f"({len(train_idx)} samples, batch size {sota.BATCH_SIZE}) to keep BatchNorm valid."
        )
    train_loader = DataLoader(
        dataset,
        batch_size=sota.BATCH_SIZE,
        sampler=SubsetRandomSampler(train_idx),
        num_workers=0,
        pin_memory=sota.PIN_MEMORY,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=sota.BATCH_SIZE,
        sampler=SubsetRandomSampler(val_idx),
        num_workers=0,
        pin_memory=sota.PIN_MEMORY,
    )

    # Match the legacy contrast script: reset RNG after model construction and
    # before the first sampler/dropout use during training.
    sota.set_seed(sota.SEED)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=sota.LR, weight_decay=sota.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=sota.LR_PLATEAU_FACTOR,
        patience=sota.LR_PLATEAU_PATIENCE,
        threshold=sota.LR_PLATEAU_THRESHOLD,
        cooldown=sota.LR_PLATEAU_COOLDOWN,
        min_lr=sota.MIN_LR,
    )

    safe_name = sota.sanitize_name(method_name)
    best_acc = -1.0
    best_epoch = 0
    best_state = None
    best_result = None
    logs = []
    history = []

    for epoch in range(1, sota.EPOCHS + 1):
        train_loss, train_acc, train_f1 = train_one_epoch_stat(
            model, train_loader, criterion, optimizer, input_type, method_name
        )
        val_result = evaluate_stat(model, val_loader, criterion, input_type, method_name)
        scheduler.step(val_result["loss"])
        line = (
            f"[{method_name}] Epoch [{epoch}/{sota.EPOCHS}] - Loss: {train_loss:.4f} - "
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
            torch.save(
                {
                    "model_name": method_name,
                    "model_state_dict": best_state,
                    "best_epoch": best_epoch,
                    "checkpoint_criterion": sota.CHECKPOINT_CRITERION,
                    "accuracy": best_acc * 100,
                    "macro_f1": val_result["macro_f1"] * 100,
                    "parameters": sota.count_trainable_params(model),
                },
                os.path.join(result_dir, f"best_model_{safe_name}.pth"),
            )

    with open(os.path.join(result_dir, f"{safe_name}_training_logs.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(logs))
    pd.DataFrame(history).to_csv(os.path.join(result_dir, f"history_{safe_name}.csv"), index=False)

    if best_state is not None:
        model.load_state_dict(best_state)
    if best_result is None:
        best_result = evaluate_stat(model, val_loader, criterion, input_type, method_name)

    latency_ms, p95_ms, fps = benchmark_stat(model, val_loader, input_type)
    flops, mflops = -1, -1.0
    return {
        "Model": method_name,
        **spec,
        "Trainable Parameters": sota.count_trainable_params(model),
        "FLOPs": flops,
        "MFLOPs": mflops,
        "Best Epoch": best_epoch,
        "Checkpoint Criterion": sota.CHECKPOINT_CRITERION,
        "Accuracy (%)": best_result["acc"] * 100,
        "Macro-Precision (%)": best_result["macro_precision"] * 100,
        "Macro-Recall (%)": best_result["macro_recall"] * 100,
        "Macro-F1 (%)": best_result["macro_f1"] * 100,
        "Latency (ms/sample)": latency_ms,
        "Median Latency (ms/sample)": latency_ms,
        "P95 Latency (ms/sample)": p95_ms,
        "FPS": fps,
        "Device": f"PyTorch {sota.DEVICE}",
        "Seed": sota.SEED,
        "Epochs": sota.EPOCHS,
        "Batch Size": sota.BATCH_SIZE,
        "Learning Rate": sota.LR,
        "Optimizer": "AdamW",
        "Scheduler": "ReduceLROnPlateau(mode=min)",
    }


def run_non_graph_sota_method(method_name, dataset, shared_split, result_dir, memory_adjusted=False):
    print("\n" + "=" * 100)
    print(f"Running repeated-run SoTA method: {method_name}")
    print("=" * 100)
    spec = ALL_METHOD_SPECS[method_name]
    sota.set_seed(sota.SEED)
    model, input_type = sota.build_model(method_name)
    train_idx, val_idx = shared_split
    batch_size = sota.method_batch_size(method_name, memory_adjusted=memory_adjusted)
    drop_last = should_drop_last_train_batch(method_name, len(train_idx), batch_size)
    if drop_last:
        print(
            f"[{method_name}] Dropping the final singleton training batch "
            f"({len(train_idx)} samples, batch size {batch_size}) to keep BatchNorm valid."
        )
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_idx),
        num_workers=0,
        pin_memory=sota.PIN_MEMORY,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_idx),
        num_workers=0,
        pin_memory=sota.PIN_MEMORY,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=sota.LR, weight_decay=sota.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=sota.LR_PLATEAU_FACTOR,
        patience=sota.LR_PLATEAU_PATIENCE,
        threshold=sota.LR_PLATEAU_THRESHOLD,
        cooldown=sota.LR_PLATEAU_COOLDOWN,
        min_lr=sota.MIN_LR,
    )

    safe_name = sota.sanitize_name(method_name)
    best_acc = -1.0
    best_epoch = 0
    best_state = None
    best_result = None
    logs = []
    history = []

    for epoch in range(1, sota.EPOCHS + 1):
        train_loss, train_acc, train_f1 = train_one_epoch_stat(
            model, train_loader, criterion, optimizer, input_type, method_name
        )
        val_result = evaluate_stat(model, val_loader, criterion, input_type, method_name)
        scheduler.step(val_result["loss"])
        line = (
            f"[{method_name}] Epoch [{epoch}/{sota.EPOCHS}] - Loss: {train_loss:.4f} - "
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
            torch.save(
                {
                    "model_name": method_name,
                    "model_state_dict": best_state,
                    "best_epoch": best_epoch,
                    "checkpoint_criterion": sota.CHECKPOINT_CRITERION,
                    "accuracy": best_acc * 100,
                    "macro_f1": val_result["macro_f1"] * 100,
                    "parameters": sota.count_trainable_params(model),
                },
                os.path.join(result_dir, f"best_model_{safe_name}.pth"),
            )

    with open(os.path.join(result_dir, f"{safe_name}_training_logs.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(logs))
    pd.DataFrame(history).to_csv(os.path.join(result_dir, f"history_{safe_name}.csv"), index=False)

    if best_state is not None:
        model.load_state_dict(best_state)
    if best_result is None:
        best_result = evaluate_stat(model, val_loader, criterion, input_type, method_name)

    latency_ms, p95_ms, fps = benchmark_stat(model, val_loader, input_type)
    flops, mflops = sota.estimate_flops_with_thop(model, input_type)
    return {
        "Model": method_name,
        **spec,
        "Trainable Parameters": sota.count_trainable_params(model),
        "FLOPs": flops,
        "MFLOPs": mflops,
        "Best Epoch": best_epoch,
        "Checkpoint Criterion": sota.CHECKPOINT_CRITERION,
        "Accuracy (%)": best_result["acc"] * 100,
        "Macro-Precision (%)": best_result["macro_precision"] * 100,
        "Macro-Recall (%)": best_result["macro_recall"] * 100,
        "Macro-F1 (%)": best_result["macro_f1"] * 100,
        "Latency (ms/sample)": latency_ms,
        "Median Latency (ms/sample)": latency_ms,
        "P95 Latency (ms/sample)": p95_ms,
        "FPS": fps,
        "Device": f"PyTorch {sota.DEVICE}",
        "Seed": sota.SEED,
        "Epochs": sota.EPOCHS,
        "Batch Size": batch_size,
        "Learning Rate": sota.LR,
        "Optimizer": "AdamW",
        "Scheduler": "ReduceLROnPlateau(mode=min)",
    }


def run_graph_method(shared_split, result_dir):
    method_name = "Ref42-TAGEGestureNet"
    spec = ALL_METHOD_SPECS[method_name]
    if not sota.TAGE_AVAILABLE:
        return {
            "Model": method_name,
            **spec,
            "Note": "Skipped: TAGEGestureNet dependencies are unavailable.",
        }

    print("\n" + "=" * 100)
    print(f"Running repeated-run method: {method_name}")
    print("=" * 100)

    graph_constructor = sota.GraphConstructor(
        d_th=0.3,
        min_samples=7,
        alpha1=4.0,
        alpha2=4.0,
        beta=0.8,
        frame_interval=0.033,
    )
    graph_dataset = sota.GestureGraphDataset(
        data_dir=sota.DATA_PATH,
        gesture_classes=sota.GESTURES,
        graph_constructor=graph_constructor,
        transform=None,
    )
    train_idx, val_idx = shared_split
    train_dataset = sota.Subset(graph_dataset, train_idx)
    val_dataset = sota.Subset(graph_dataset, val_idx)
    train_loader = sota.GraphDataLoader(train_dataset, batch_size=sota.GRAPH_BATCH_SIZE, shuffle=True)
    val_loader = sota.GraphDataLoader(val_dataset, batch_size=sota.GRAPH_BATCH_SIZE, shuffle=False)

    model = sota.TAGEGestureNet(
        in_dim=4,
        edge_dim=10,
        hidden_dim=64,
        graph_dim=1024,
        num_layers=2,
        num_classes=sota.NUM_CLASSES,
        descriptor_dim=5,
    ).to(sota.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=sota.LR, weight_decay=sota.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=sota.LR_PLATEAU_FACTOR,
        patience=sota.LR_PLATEAU_PATIENCE,
        threshold=sota.LR_PLATEAU_THRESHOLD,
        cooldown=sota.LR_PLATEAU_COOLDOWN,
        min_lr=sota.MIN_LR,
    )

    safe_name = sota.sanitize_name(method_name)
    best_acc = -1.0
    best_epoch = 0
    best_state = None
    best_result = None
    logs = []
    history = []

    for epoch in range(1, sota.EPOCHS + 1):
        train_loss, train_acc, train_f1 = train_one_epoch_stat(
            model, train_loader, criterion, optimizer, "graph", method_name
        )
        val_result = evaluate_stat(model, val_loader, criterion, "graph", method_name)
        scheduler.step(val_result["loss"])
        line = (
            f"[{method_name}] Epoch [{epoch}/{sota.EPOCHS}] - Loss: {train_loss:.4f} - "
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
            torch.save(
                {
                    "model_name": method_name,
                    "model_state_dict": best_state,
                    "best_epoch": best_epoch,
                    "checkpoint_criterion": sota.CHECKPOINT_CRITERION,
                    "accuracy": best_acc * 100,
                    "macro_f1": val_result["macro_f1"] * 100,
                    "parameters": sota.count_trainable_params(model),
                },
                os.path.join(result_dir, f"best_model_{safe_name}.pth"),
            )

    with open(os.path.join(result_dir, f"{safe_name}_training_logs.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(logs))
    pd.DataFrame(history).to_csv(os.path.join(result_dir, f"history_{safe_name}.csv"), index=False)

    if best_state is not None:
        model.load_state_dict(best_state)
    if best_result is None:
        best_result = evaluate_stat(model, val_loader, criterion, "graph", method_name)

    latency_ms, p95_ms, fps = benchmark_stat(model, val_loader, "graph")
    try:
        flops, mflops = sota.compute_graph_flops(
            model,
            in_dim=4,
            edge_dim=10,
            descriptor_dim=5,
            hidden_dim=64,
            graph_dim=1024,
            num_layers=2,
            num_classes=sota.NUM_CLASSES,
        )
    except Exception:
        flops, mflops = -1, -1.0

    return {
        "Model": method_name,
        **spec,
        "Trainable Parameters": sota.count_trainable_params(model),
        "FLOPs": flops,
        "MFLOPs": mflops,
        "Best Epoch": best_epoch,
        "Checkpoint Criterion": sota.CHECKPOINT_CRITERION,
        "Accuracy (%)": best_result["acc"] * 100,
        "Macro-Precision (%)": best_result["macro_precision"] * 100,
        "Macro-Recall (%)": best_result["macro_recall"] * 100,
        "Macro-F1 (%)": best_result["macro_f1"] * 100,
        "Latency (ms/sample)": latency_ms,
        "Median Latency (ms/sample)": latency_ms,
        "P95 Latency (ms/sample)": p95_ms,
        "FPS": fps,
        "Device": f"PyTorch {sota.DEVICE}",
        "Seed": sota.SEED,
        "Epochs": sota.EPOCHS,
        "Batch Size": sota.GRAPH_BATCH_SIZE,
        "Learning Rate": sota.LR,
        "Optimizer": "AdamW",
        "Scheduler": "ReduceLROnPlateau(mode=min)",
    }


def run_seed(args, seed, raw_dataset, classic_datasets, methods):
    seed_dir = os.path.join(args.result_dir, f"seed_{seed}")
    ensure_dir(seed_dir)
    configure_sota(args, seed, seed_dir)
    sota.set_seed(seed)

    print("\n" + "=" * 100)
    print(f"Statistical validation run: seed={seed}")
    print("=" * 100)
    print(f"Result dir : {seed_dir}")
    print(f"Methods    : {methods}")
    print(f"P4 backend : {sota.P4_BACKEND_NAME} ({sota.P4_BACKEND_DETAIL})")

    shared_split = sota.build_shared_split(raw_dataset, seed_dir)
    norm_dataset = sota.TrainSetNormalizedSotaDataset(raw_dataset, shared_split[0])
    method_datasets = {
        # Match 对比实验补充260509-深度学习-pytorch.py:
        # CNN uses raw serialized [100,4] sequences, without normalization.
        "CNN": classic_datasets["sequence"],
        # STGNet uses engineered [32,1] statistics normalized by train split.
        "STGNet": TrainSetNormalizedTensorDataset(classic_datasets["stg"], shared_split[0]),
        # LSTM uses the train-set-normalized DSTG LSTM branch [80,240].
        "LSTM": norm_dataset,
    }
    for method_name in sota.METHOD_SPECS:
        method_datasets[method_name] = norm_dataset

    seed_rows = []
    summary_path = os.path.join(seed_dir, "SoTA_Summary.csv")
    for method_name in methods:
        existing = None if args.force else method_row_from_summary(summary_path, method_name)
        if existing is not None:
            print(f"[seed={seed}] Reusing existing result: {method_name}")
            seed_rows.append(existing)
            continue

        sota.set_seed(seed)
        try:
            if method_name == "Ref42-TAGEGestureNet":
                row = run_graph_method(shared_split, seed_dir)
            elif method_name in CLASSIC_METHOD_SPECS:
                row = run_legacy_contrast_method(
                    method_name,
                    method_datasets[method_name],
                    shared_split,
                    seed_dir,
                )
            else:
                row = run_non_graph_sota_method(
                    method_name,
                    method_datasets[method_name],
                    shared_split,
                    seed_dir,
                )
        except RuntimeError as exc:
            if method_name == "Ref31-P4Transformer" and sota.is_cuda_oom(exc):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("[Ref31-P4Transformer] CUDA OOM; retrying with smaller batch size.")
                try:
                    row = run_non_graph_sota_method(
                        method_name,
                        method_datasets[method_name],
                        shared_split,
                        seed_dir,
                        memory_adjusted=True,
                    )
                except Exception as retry_exc:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(f"[ERROR] seed={seed} method={method_name} failed after retry: {retry_exc}")
                    traceback.print_exc()
                    spec = ALL_METHOD_SPECS[method_name]
                    row = {
                        "Model": method_name,
                        **spec,
                        "Note": f"Failed: {retry_exc}",
                    }
            else:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[ERROR] seed={seed} method={method_name} failed: {exc}")
                traceback.print_exc()
                spec = ALL_METHOD_SPECS[method_name]
                row = {
                    "Model": method_name,
                    **spec,
                    "Note": f"Failed: {exc}",
                }
        except Exception as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[ERROR] seed={seed} method={method_name} failed: {exc}")
            traceback.print_exc()
            spec = ALL_METHOD_SPECS[method_name]
            row = {
                "Model": method_name,
                **spec,
                "Note": f"Failed: {exc}",
            }
        seed_rows.append(row)
        save_seed_summary(seed_rows, seed_dir)

    return seed_rows


def numeric_series(df, method, metric):
    sub = df[df["Model"] == method].sort_values("Seed")
    values = pd.to_numeric(sub[metric], errors="coerce").dropna().to_numpy(dtype=float)
    return values


def mean_ci(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(np.mean(values))
    if n == 1:
        return mean, np.nan, np.nan, np.nan
    sd = float(np.std(values, ddof=1))
    sem = sd / np.sqrt(n)
    half_width = float(stats.t.ppf((1.0 + confidence) / 2.0, n - 1) * sem)
    return mean, sd, mean - half_width, mean + half_width


def safe_paired_ttest(a, b):
    try:
        stat, p = stats.ttest_rel(a, b, nan_policy="omit")
        return float(stat), float(p)
    except Exception:
        return np.nan, np.nan


def safe_wilcoxon(a, b):
    try:
        stat, p = stats.wilcoxon(a, b, zero_method="wilcox")
        return float(stat), float(p)
    except Exception:
        return np.nan, np.nan


def paired_cohens_d(reference, comparison):
    diff = reference - comparison
    diff = diff[np.isfinite(diff)]
    if len(diff) < 2:
        return np.nan
    sd = np.std(diff, ddof=1)
    if sd < 1e-12:
        return np.nan
    return float(np.mean(diff) / sd)


def p_symbol(p):
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def build_statistics(all_runs, methods, reference=REFERENCE_METHOD):
    df = pd.DataFrame(all_runs)
    metrics = ["Accuracy (%)", "Macro-F1 (%)"]

    summary_rows = []
    for method in methods:
        row = {"Model": method}
        for metric in metrics:
            values = numeric_series(df, method, metric)
            mean, sd, ci_low, ci_high = mean_ci(values)
            row[f"{metric} Mean"] = mean
            row[f"{metric} SD"] = sd
            row[f"{metric} 95CI Low"] = ci_low
            row[f"{metric} 95CI High"] = ci_high
            row[f"{metric} n"] = int(len(values))
        for extra in ["Trainable Parameters", "MFLOPs", "Latency (ms/sample)", "FPS"]:
            values = numeric_series(df, method, extra)
            row[extra] = float(np.nanmean(values)) if len(values) else np.nan
        summary_rows.append(row)

    test_rows = []
    for method in methods:
        if method == reference:
            continue
        for metric in metrics:
            ref = df[df["Model"] == reference][["Seed", metric]].rename(columns={metric: "Reference"})
            comp = df[df["Model"] == method][["Seed", metric]].rename(columns={metric: "Comparison"})
            paired = ref.merge(comp, on="Seed", how="inner")
            paired["Reference"] = pd.to_numeric(paired["Reference"], errors="coerce")
            paired["Comparison"] = pd.to_numeric(paired["Comparison"], errors="coerce")
            paired = paired.dropna()
            if paired.empty:
                t_stat = t_p = w_stat = w_p = d = mean_diff = np.nan
                n = 0
            else:
                ref_v = paired["Reference"].to_numpy(dtype=float)
                comp_v = paired["Comparison"].to_numpy(dtype=float)
                t_stat, t_p = safe_paired_ttest(ref_v, comp_v)
                w_stat, w_p = safe_wilcoxon(ref_v, comp_v)
                d = paired_cohens_d(ref_v, comp_v)
                mean_diff = float(np.mean(ref_v - comp_v))
                n = int(len(paired))
            test_rows.append({
                "Reference Model": reference,
                "Compared Model": method,
                "Metric": metric,
                "Paired n": n,
                "Mean Diff.": mean_diff,
                "Paired t statistic": t_stat,
                "Paired t p-value": t_p,
                "Wilcoxon statistic": w_stat,
                "Wilcoxon p-value": w_p,
                "Cohen d": d,
                "Significance": p_symbol(t_p),
            })

    return pd.DataFrame(summary_rows), pd.DataFrame(test_rows)


def fmt_mean_sd(mean, sd):
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(sd):
        return f"{mean:.2f}"
    return f"{mean:.2f} $\\pm$ {sd:.2f}"


def fmt_ci(low, high):
    if not np.isfinite(low) or not np.isfinite(high):
        return "--"
    return f"[{low:.2f}, {high:.2f}]"


def fmt_p(p):
    if not np.isfinite(p):
        return "--"
    if p < 0.001:
        return "$<0.001$"
    return f"{p:.4f}"


def write_latex_table(summary_df, tests_df, methods, out_path, reference=REFERENCE_METHOD):
    test_acc = tests_df[tests_df["Metric"] == "Accuracy (%)"] if not tests_df.empty else pd.DataFrame()
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\caption{\textcolor{red}{Statistical validation over multiple independent runs on the public dataset.\label{tab:statistical_validation}}}")
    lines.append(r"\begingroup")
    lines.append(r"\color{red}")
    lines.append(r"\begin{tabularx}{\textwidth}{CCCCCC}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Method} & \textbf{Accuracy} & \textbf{Macro-F1} & \textbf{95\% CI} & \textbf{$p$-value} & \textbf{Sig.} \\")
    lines.append(r"\midrule")
    for method in methods:
        row = summary_df[summary_df["Model"] == method]
        if row.empty:
            continue
        row = row.iloc[0]
        acc = fmt_mean_sd(row["Accuracy (%) Mean"], row["Accuracy (%) SD"])
        f1 = fmt_mean_sd(row["Macro-F1 (%) Mean"], row["Macro-F1 (%) SD"])
        ci = fmt_ci(row["Accuracy (%) 95CI Low"], row["Accuracy (%) 95CI High"])
        if method == reference:
            p_text = "--"
            sig = "--"
        else:
            test_row = test_acc[test_acc["Compared Model"] == method]
            if test_row.empty:
                p_text = "--"
                sig = "n/a"
            else:
                p = float(test_row.iloc[0]["Paired t p-value"])
                p_text = fmt_p(p)
                sig = str(test_row.iloc[0]["Significance"])
        display = method.replace("Ref31-", "").replace("Ref41-", "").replace("Ref42-", "")
        lines.append(f"{display} & {acc} & {f1} & {ci} & {p_text} & {sig} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(r"\endgroup")
    lines.append(r"\end{table}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-seed statistical validation for Reviewer 2.")
    parser.add_argument("--data-path", default=sota.DATA_PATH)
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Repeated-run training epochs. Default is 200 to match the main comparison protocol.",
    )
    parser.add_argument("--batch-size", type=int, default=sota.BATCH_SIZE)
    parser.add_argument("--p4-batch-size", type=int, default=sota.P4_BATCH_SIZE)
    parser.add_argument("--graph-batch-size", type=int, default=sota.GRAPH_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=sota.LR)
    parser.add_argument("--weight-decay", type=float, default=sota.WEIGHT_DECAY)
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--p4-root", default=None)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS, choices=list(ALL_METHOD_SPECS.keys()))
    parser.add_argument("--force", action="store_true", help="Rerun methods even if seed-level summaries already exist.")
    parser.add_argument("--aggregate-only", action="store_true", help="Only aggregate existing seed_* results.")
    return parser.parse_args()


def collect_existing_runs(result_dir, seeds, methods):
    rows = []
    for seed in seeds:
        summary_path = os.path.join(result_dir, f"seed_{seed}", "SoTA_Summary.csv")
        if not os.path.exists(summary_path):
            continue
        df = pd.read_csv(summary_path)
        for _, row in df.iterrows():
            if row.get("Model") in methods:
                item = row.to_dict()
                item["Seed"] = seed
                rows.append(item)
    return rows


def main():
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    methods = args.methods
    ensure_dir(args.result_dir)

    all_runs = []
    if not args.aggregate_only:
        configure_sota(args, seeds[0], args.result_dir)
        samples, _ = sota.collect_samples(args.data_path, max_samples_per_class=args.max_samples_per_class)
        if args.max_samples_per_class is not None:
            print(f"Debug cap: first {args.max_samples_per_class} samples per class")
        raw_dataset = sota.SotaRadarDataset(samples)
        classic_datasets = {
            "sequence": SingleTensorFeatureDataset(
                samples,
                process_sequence_file,
                "serialized sequence dataset for CNN [100,4]",
            ),
            "stg": SingleTensorFeatureDataset(
                samples,
                process_stg_file,
                "engineered STGNet feature dataset [32,1]",
            ),
        }

        for seed in seeds:
            seed_rows = run_seed(args, seed, raw_dataset, classic_datasets, methods)
            for row in seed_rows:
                row = dict(row)
                row["Seed"] = seed
                all_runs.append(row)
            pd.DataFrame(all_runs).to_csv(
                os.path.join(args.result_dir, "repeated_run_results_partial.csv"),
                index=False,
                encoding="utf-8-sig",
            )
    else:
        all_runs = collect_existing_runs(args.result_dir, seeds, methods)

    if not all_runs:
        raise RuntimeError("No run results were available for aggregation.")

    all_df = pd.DataFrame(all_runs)
    all_df.to_csv(os.path.join(args.result_dir, "repeated_run_results.csv"), index=False, encoding="utf-8-sig")

    summary_df, tests_df = build_statistics(all_runs, methods, reference=REFERENCE_METHOD)
    summary_path = os.path.join(args.result_dir, "statistical_summary.csv")
    tests_path = os.path.join(args.result_dir, "pairwise_significance_tests.csv")
    latex_path = os.path.join(args.result_dir, "table_statistical_validation_red.tex")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    tests_df.to_csv(tests_path, index=False, encoding="utf-8-sig")
    write_latex_table(summary_df, tests_df, methods, latex_path, reference=REFERENCE_METHOD)

    print("\n" + "=" * 100)
    print("Statistical validation summary")
    print("=" * 100)
    keep_cols = [
        "Model",
        "Accuracy (%) Mean",
        "Accuracy (%) SD",
        "Accuracy (%) 95CI Low",
        "Accuracy (%) 95CI High",
        "Macro-F1 (%) Mean",
        "Macro-F1 (%) SD",
    ]
    print(summary_df[keep_cols].to_string(index=False))
    print("\nPairwise tests versus DSTG-Net:")
    print(tests_df.to_string(index=False))
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {tests_path}")
    print(f"Saved: {latex_path}")


if __name__ == "__main__":
    main()
