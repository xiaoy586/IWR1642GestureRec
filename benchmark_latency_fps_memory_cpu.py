#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import gc
import psutil
import tracemalloc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ============================================================
# 全局配置（适配 electronics/对比实验补充260509-深度学习-electronics.py）
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = 'data'
RESULT_DIR = 'electronics'  # 'Result_Contrast_Study-200-pub'
os.makedirs(RESULT_DIR, exist_ok=True)

GESTURES = ['arm_to_left', 'arm_to_right', 'close_fist_horizontally',
            'close_fist_perpendicularly', 'hand_away', 'hand_closer',
            'hand_down', 'hand_rotation_palm_down', 'hand_rotation_palm_up',
            'hand_to_left', 'hand_to_right', 'hand_up']
LABELS_DICT = {gesture: idx for idx, gesture in enumerate(GESTURES)}
NUM_CLASSES = len(GESTURES)


# ============================================================
# 以下定义从 electronics/对比实验补充260509-深度学习-electronics.py 复制
# 确保模型架构和数据预处理完全一致
# ============================================================

def process_single_file_pt(file_path):
    try:
        df = pd.read_csv(file_path, skiprows=1)
        cols = df.columns.tolist()
        frame_col, vel_col, peak_col, x_col, y_col = cols[0], cols[3], cols[4], cols[5], cols[6]

        frames_data = []
        for frame_id, group in df.groupby(frame_col):
            x_vals = group[x_col].values
            y_vals = group[y_col].values
            v_vals = group[vel_col].values
            p_vals = group[peak_col].values if peak_col in group else np.zeros_like(v_vals)
            frames_data.append(list(zip(x_vals, y_vals, v_vals, p_vals)))

        MAX_FRAMES, MAX_POINTS = 80, 80

        # LSTM Feature
        lstm_data = np.zeros((MAX_FRAMES, 3, MAX_POINTS), dtype=np.float32)
        for frame_idx, frame in enumerate(frames_data):
            if frame_idx >= MAX_FRAMES:
                break
            if frame:
                limited = frame[:MAX_POINTS]
                x_p, y_p, v_p = np.array([p[0] for p in limited]), np.array([p[1] for p in limited]), np.array([p[2] for p in limited])
            else:
                x_p, y_p, v_p = np.array([]), np.array([]), np.array([])
            pw = max(0, MAX_POINTS - len(x_p))
            lstm_data[frame_idx] = np.array([np.pad(x_p, (0, pw)), np.pad(y_p, (0, pw)), np.pad(v_p, (0, pw))])
        lstm_feature = lstm_data.reshape(MAX_FRAMES, 240)

        # GRU Feature
        frame_features = []
        for frame_idx, frame in enumerate(frames_data):
            if frame_idx >= MAX_FRAMES:
                break
            fv = []
            if frame:
                limited = frame[:MAX_POINTS]
                v_arr, p_arr = np.array([p[2] for p in limited]), np.array([p[3] for p in limited])
                x_arr, y_arr = np.array([p[0] for p in limited]), np.array([p[1] for p in limited])

                features_list = [v_arr, p_arr, x_arr, y_arr]
                non_zero_mask_total = np.zeros(len(limited), dtype=bool)
                for feat in features_list:
                    non_zero_mask = feat != 0
                    non_zero_mask_total = non_zero_mask_total | non_zero_mask
                    non_zero_values = feat[non_zero_mask]
                    stats = [np.mean(non_zero_values), np.std(non_zero_values), np.median(non_zero_values),
                             np.max(non_zero_values), np.min(non_zero_values),
                             np.percentile(non_zero_values, 25), np.percentile(non_zero_values, 75)] if len(non_zero_values) > 0 else [0.0]*7
                    fv.extend(stats)
                fv.append(np.sum(non_zero_mask_total))
            else:
                fv = [0.0] * 29
            frame_features.append(fv)

        while len(frame_features) < MAX_FRAMES:
            frame_features.append([0.0]*29)
        ff_arr = np.array(frame_features, dtype=np.float32)

        temporal_features = []
        for i in range(1, len(ff_arr)):
            temporal_features.extend(ff_arr[i] - ff_arr[i-1])
        exp_temp = 79 * 29
        if len(temporal_features) < exp_temp:
            temporal_features.extend([0.0]*(exp_temp - len(temporal_features)))
        else:
            temporal_features = temporal_features[:exp_temp]

        combined = np.concatenate([ff_arr.flatten(), temporal_features])
        tgt_size = 80 * 58
        if len(combined) < tgt_size:
            combined = np.concatenate([combined, np.zeros(tgt_size - len(combined))])
        else:
            combined = combined[:tgt_size]
        gru_feature = combined.reshape(MAX_FRAMES, 58)

        return lstm_feature, gru_feature
    except Exception:
        return None, None


class RadarGestureDataset(Dataset):
    def __init__(self):
        self.X_lstm, self.X_gru, self.y = [], [], []
        print("正在加载 PyTorch 双流特征数据 (DSTG-Net)...")
        for gesture, label in LABELS_DICT.items():
            folder = os.path.join(DATA_PATH, gesture)
            if not os.path.exists(folder):
                continue
            for file_name in os.listdir(folder):
                if file_name.endswith('.csv'):
                    lstm_f, gru_f = process_single_file_pt(os.path.join(folder, file_name))
                    if lstm_f is not None:
                        self.X_lstm.append(lstm_f)
                        self.X_gru.append(gru_f)
                        self.y.append(label)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (torch.tensor(self.X_lstm[idx], dtype=torch.float32),
                torch.tensor(self.X_gru[idx], dtype=torch.float32),
                torch.tensor(self.y[idx], dtype=torch.long))


class FusionGestureModel(nn.Module):
    def __init__(self, lstm_in=240, gru_in=58, lstm_hidden=32, gru_hidden=64, num_layers=2, num_classes=12, dropout_rate=0.4):
        super(FusionGestureModel, self).__init__()
        self.lstm_hidden, self.lstm_layers = lstm_hidden, num_layers
        self.lstm = nn.LSTM(lstm_in, lstm_hidden, num_layers, batch_first=True, dropout=0.1)
        self.lstm_fc = nn.Sequential(nn.Linear(lstm_hidden * 80, 128), nn.BatchNorm1d(128), nn.ReLU())

        self.gru_proj = nn.Sequential(nn.Linear(gru_in, 128), nn.BatchNorm1d(128), nn.ReLU(),
                                      nn.Dropout(dropout_rate), nn.Linear(128, 64), nn.ReLU())
        self.gru = nn.GRU(64, gru_hidden, num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(gru_hidden * 2, 64), nn.Tanh(), nn.Linear(64, 1), nn.Dropout(dropout_rate))
        self.gru_fc = nn.Sequential(nn.Linear(gru_hidden * 2, 128), nn.BatchNorm1d(128), nn.ReLU())

        self.fusion_classifier = nn.Sequential(
            nn.Linear(128 + 128, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(dropout_rate), nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(dropout_rate), nn.Linear(64, num_classes))

    def forward(self, x_lstm, x_gru):
        h0 = torch.zeros(self.lstm_layers, x_lstm.size(0), self.lstm_hidden).to(x_lstm.device)
        c0 = torch.zeros(self.lstm_layers, x_lstm.size(0), self.lstm_hidden).to(x_lstm.device)
        out_lstm, _ = self.lstm(x_lstm, (h0, c0))
        feat_lstm = self.lstm_fc(out_lstm.reshape(out_lstm.shape[0], -1))

        batch_size, seq_len, _ = x_gru.shape
        x_g_proj = self.gru_proj(x_gru.reshape(-1, x_gru.size(-1))).reshape(batch_size, seq_len, -1)
        gru_out, _ = self.gru(x_g_proj)
        attn_w = torch.softmax(self.attention(gru_out).squeeze(-1), dim=1).unsqueeze(-1)
        feat_gru = self.gru_fc(torch.sum(attn_w * gru_out, dim=1))

        return self.fusion_classifier(torch.cat([feat_lstm, feat_gru], dim=1))


# ============================================================
# 1. 加载模型
# ============================================================
def load_fusion_model():
    weight_path = os.path.join(RESULT_DIR, "best_model_DSTG-Net.pth")

    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"未找到模型权重: {weight_path}\n"
            f"请先运行训练代码生成 best_model_DSTG-Net.pth"
        )

    checkpoint = torch.load(weight_path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_config" in checkpoint:
        model = FusionGestureModel(**checkpoint["model_config"]).to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model = FusionGestureModel(num_classes=NUM_CLASSES).to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = FusionGestureModel(num_classes=NUM_CLASSES).to(DEVICE)
        model.load_state_dict(checkpoint)

    model.eval()
    return model


# ============================================================
# 2. 构造验证集
# ============================================================
def build_eval_loader(batch_size=1):
    dataset = RadarGestureDataset()

    indices = np.arange(len(dataset))
    _, val_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=dataset.y,
        random_state=42
    )

    val_dataset = Subset(dataset, val_idx)

    loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return loader


# ============================================================
# 3. 单次推理延迟测试
# ============================================================
@torch.no_grad()
def benchmark_inference_only(model, loader, warmup=30, repeat=300):
    print("\n[Benchmark] Inference-only latency / FPS / CPU / memory")

    process = psutil.Process(os.getpid())

    x_lstm_list = []
    x_gru_list = []

    for x_lstm, x_gru, _ in loader:
        x_lstm_list.append(x_lstm)
        x_gru_list.append(x_gru)
        if len(x_lstm_list) >= repeat:
            break

    if len(x_lstm_list) == 0:
        raise RuntimeError("验证集为空，无法测试。")

    # warmup
    for i in range(min(warmup, len(x_lstm_list))):
        x_lstm = x_lstm_list[i].to(DEVICE)
        x_gru = x_gru_list[i].to(DEVICE)
        _ = model(x_lstm, x_gru)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    gc.collect()
    tracemalloc.start()

    latencies = []
    cpu_usages = []
    rss_memory = []

    process.cpu_percent(interval=None)

    for i in range(min(repeat, len(x_lstm_list))):
        x_lstm = x_lstm_list[i].to(DEVICE)
        x_gru = x_gru_list[i].to(DEVICE)

        start = time.perf_counter()

        _ = model(x_lstm, x_gru)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

        latency_ms = (end - start) * 1000.0
        latencies.append(latency_ms)

        cpu_usages.append(process.cpu_percent(interval=None))
        rss_memory.append(process.memory_info().rss / 1024 / 1024)

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    latencies = np.array(latencies)
    cpu_usages = np.array(cpu_usages)
    rss_memory = np.array(rss_memory)

    results = {
        "Mode": "Inference-only",
        "Samples": len(latencies),
        "Mean Latency (ms)": latencies.mean(),
        "Std Latency (ms)": latencies.std(),
        "Min Latency (ms)": latencies.min(),
        "Max Latency (ms)": latencies.max(),
        "P95 Latency (ms)": np.percentile(latencies, 95),
        "P99 Latency (ms)": np.percentile(latencies, 99),
        "FPS": 1000.0 / latencies.mean(),
        "Mean CPU Usage (%)": cpu_usages.mean(),
        "Peak CPU Usage (%)": cpu_usages.max(),
        "Mean RSS Memory (MB)": rss_memory.mean(),
        "Peak RSS Memory (MB)": rss_memory.max(),
        "Python Peak Allocated Memory (MB)": peak_mem / 1024 / 1024
    }

    return results, latencies, cpu_usages, rss_memory


# ============================================================
# 4. 批量推理吞吐率测试
# ============================================================
@torch.no_grad()
def benchmark_batch_throughput(model, batch_sizes=(1, 2, 4, 8, 16, 32), repeat=100):
    print("\n[Benchmark] Batch throughput")

    throughput_results = []

    for bs in batch_sizes:
        loader = build_eval_loader(batch_size=bs)

        total_samples = 0
        total_time = 0.0

        for idx, (x_lstm, x_gru, _) in enumerate(loader):
            if idx >= repeat:
                break

            x_lstm = x_lstm.to(DEVICE)
            x_gru = x_gru.to(DEVICE)

            start = time.perf_counter()
            _ = model(x_lstm, x_gru)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            total_time += end - start
            total_samples += x_lstm.size(0)

        fps = total_samples / total_time if total_time > 0 else 0

        throughput_results.append({
            "Batch Size": bs,
            "Total Samples": total_samples,
            "Total Time (s)": total_time,
            "Throughput FPS": fps,
            "Avg Latency per Batch (ms)": total_time / max(1, idx + 1) * 1000
        })

        print(f"Batch={bs:<3d} | FPS={fps:.2f}")

    return throughput_results


# ============================================================
# 5. 绘图
# ============================================================
def plot_latency_distribution(latencies):
    plt.figure(figsize=(8, 5))
    plt.hist(latencies, bins=30, edgecolor="black", alpha=0.8)
    plt.axvline(np.mean(latencies), linestyle="--", linewidth=2,
                label=f"Mean = {np.mean(latencies):.2f} ms")
    plt.axvline(np.percentile(latencies, 95), linestyle="--", linewidth=2,
                label=f"P95 = {np.percentile(latencies, 95):.2f} ms")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Frequency")
    plt.title("Inference Latency Distribution")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_path = os.path.join(RESULT_DIR, "Latency_Distribution.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


def plot_resource_usage(cpu_usages, rss_memory):
    x = np.arange(len(cpu_usages))

    plt.figure(figsize=(9, 5))
    plt.plot(x, cpu_usages, linewidth=1.8, label="CPU Usage (%)")
    plt.xlabel("Inference Step")
    plt.ylabel("CPU Usage (%)")
    plt.title("CPU Usage During Inference")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(RESULT_DIR, "CPU_Usage.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")

    plt.figure(figsize=(9, 5))
    plt.plot(x, rss_memory, linewidth=1.8, label="RSS Memory (MB)")
    plt.xlabel("Inference Step")
    plt.ylabel("Memory (MB)")
    plt.title("Memory Usage During Inference")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(RESULT_DIR, "Memory_Usage.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


def plot_batch_fps(throughput_results):
    batch_sizes = [r["Batch Size"] for r in throughput_results]
    fps_values = [r["Throughput FPS"] for r in throughput_results]

    plt.figure(figsize=(7, 5))
    plt.plot(batch_sizes, fps_values, marker="o", linewidth=2)
    plt.xlabel("Batch Size")
    plt.ylabel("Throughput FPS")
    plt.title("Batch Inference Throughput")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_path = os.path.join(RESULT_DIR, "Batch_Throughput_FPS.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


# ============================================================
# 6. 主函数
# ============================================================
def main():
    print("=" * 80)
    print("FusionGestureModel Robustness / Real-time Efficiency Benchmark")
    print("=" * 80)
    print(f"Device: {DEVICE}")

    model = load_fusion_model()

    eval_loader = build_eval_loader(batch_size=1)

    results, latencies, cpu_usages, rss_memory = benchmark_inference_only(
        model,
        eval_loader,
        warmup=30,
        repeat=300
    )

    throughput_results = benchmark_batch_throughput(
        model,
        batch_sizes=(1, 2, 4, 8, 16, 32),
        repeat=100
    )

    # 保存 summary
    summary_path = os.path.join(RESULT_DIR, "Realtime_Benchmark_Summary.csv")
    pd.DataFrame([results]).to_csv(summary_path, index=False)

    detail_path = os.path.join(RESULT_DIR, "Realtime_Benchmark_Details.csv")
    pd.DataFrame({
        "Latency_ms": latencies,
        "CPU_usage_percent": cpu_usages,
        "RSS_memory_MB": rss_memory
    }).to_csv(detail_path, index=False)

    throughput_path = os.path.join(RESULT_DIR, "Batch_Throughput_FPS.csv")
    pd.DataFrame(throughput_results).to_csv(throughput_path, index=False)

    print("\n================ Benchmark Summary ================")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    print(f"\nSaved: {summary_path}")
    print(f"Saved: {detail_path}")
    print(f"Saved: {throughput_path}")

    plot_latency_distribution(latencies)
    plot_resource_usage(cpu_usages, rss_memory)
    plot_batch_fps(throughput_results)

    print("\nDone.")


if __name__ == "__main__":
    main()