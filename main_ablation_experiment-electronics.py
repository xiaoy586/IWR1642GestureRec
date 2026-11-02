# 毫米波雷达手势识别 - 单独 LSTM、Attention-BiGRU 与 DSTG-Net 综合消融实验代码
import copy
import os
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm

# 尝试导入 scienceplots 风格
try:
    import scienceplots

    plt.style.use(['science', 'no-latex'])
except ImportError:
    print("未安装 scienceplots，将使用 seaborn 默认科研风格。建议运行: pip install scienceplots")
    plt.style.use('seaborn-v0_8-paper')

warnings.filterwarnings('ignore')

# 解决中文乱码与负号问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 全局配置与标签映射 ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = 'data'
RESULT_DIR = 'Result_Ablation_Study-200-pub-electronics'
SEED = 42
BATCH_SIZE = 32
GLOBAL_EPOCHS = 200
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
SCHEDULER_T_MAX = 50
os.makedirs(RESULT_DIR, exist_ok=True)

# 统一使用12种手势进行消融实验（与DSTG_Net_Plus_note.py一致，使用缩写标签用于可视化）
# 注意：LABELS_DICT的键是用于显示的缩写标签，实际文件夹使用完整名称
LABELS_DICT = {
    "AL": 0, "AR": 1, "HA": 2, "HC": 3,
    "CFH": 4, "CFP": 5, "HR": 6,
    "hL": 7, "HRPD": 8, "HRPU": 9,
    "HU": 10, "HD": 11
}
# 完整标签名称到缩写的映射（用于数据文件夹访问）
LABEL_FULL_NAMES = {
    0: "arm_to_left", 1: "arm_to_right", 2: "hand_away", 3: "hand_closer",
    4: "close_fist_horizontally", 5: "close_fist_perpendicularly", 6: "hand_to_right",
    7: "hand_to_left", 8: "hand_rotation_palm_down", 9: "hand_rotation_palm_up",
    10: "hand_up", 11: "hand_down"
}
REVERSE_LABELS = {v: k for k, v in LABELS_DICT.items()}
NUM_CLASSES = len(LABELS_DICT)


def set_seed(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_inference(model, val_loader, warmup_batches=2):
    model.eval()
    elapsed_times = []
    total_samples = 0

    with torch.no_grad():
        for idx, (b_lstm, b_gru, _) in enumerate(val_loader):
            b_lstm, b_gru = b_lstm.to(DEVICE), b_gru.to(DEVICE)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(b_lstm, b_gru)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            if idx >= warmup_batches:
                elapsed_times.append(elapsed)
                total_samples += b_lstm.size(0)

    if not elapsed_times or total_samples == 0:
        return 0.0, 0.0

    total_time = sum(elapsed_times)
    avg_latency_ms = (total_time / total_samples) * 1000.0
    fps = total_samples / total_time
    return avg_latency_ms, fps


# ==================== 2. 模型定义 (无任何修改) ====================

# --- 模型 A: 仅 LSTM ---
class LSTMOnlyModel(nn.Module):
    def __init__(self, lstm_in=240, lstm_hidden=32, num_layers=2, num_classes=12, dropout_rate=0.4):
        super(LSTMOnlyModel, self).__init__()
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = num_layers

        self.lstm = nn.LSTM(lstm_in, lstm_hidden, num_layers, batch_first=True, dropout=0.1)
        self.lstm_fc = nn.Sequential(
            nn.Linear(lstm_hidden * 80, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_lstm, x_gru):
        h0 = torch.zeros(self.lstm_layers, x_lstm.size(0), self.lstm_hidden).to(x_lstm.device)
        c0 = torch.zeros(self.lstm_layers, x_lstm.size(0), self.lstm_hidden).to(x_lstm.device)
        out_lstm, _ = self.lstm(x_lstm, (h0, c0))
        out_lstm = out_lstm.reshape(out_lstm.shape[0], -1)
        feat_lstm = self.lstm_fc(out_lstm)
        return self.classifier(feat_lstm)


# --- 模型 B: 仅 Attention-BiGRU ---
class AttBiGRUOnlyModel(nn.Module):
    def __init__(self, gru_in=58, gru_hidden=64, num_layers=2, num_classes=12, dropout_rate=0.4):
        super(AttBiGRUOnlyModel, self).__init__()
        self.gru_proj = nn.Sequential(
            nn.Linear(gru_in, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(dropout_rate), nn.Linear(128, 64), nn.ReLU()
        )
        self.gru = nn.GRU(64, gru_hidden, num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(gru_hidden * 2, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.gru_fc = nn.Sequential(
            nn.Linear(gru_hidden * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_lstm, x_gru):
        batch_size, seq_len, _ = x_gru.shape
        x_g_proj = self.gru_proj(x_gru.reshape(-1, x_gru.size(-1))).reshape(batch_size, seq_len, -1)
        gru_out, _ = self.gru(x_g_proj)

        attn_w = torch.softmax(self.attention(gru_out).squeeze(-1), dim=1).unsqueeze(-1)
        weighted_gru = torch.sum(attn_w * gru_out, dim=1)
        feat_gru = self.gru_fc(weighted_gru)
        return self.classifier(feat_gru)


# --- 模型 C: 双流融合网络 (DSTG-Net) ---
class FusionGestureModel(nn.Module):
    def __init__(self, lstm_in=240, gru_in=58, lstm_hidden=32, gru_hidden=64, num_layers=2, num_classes=12,
                 dropout_rate=0.4):
        super(FusionGestureModel, self).__init__()
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = num_layers
        self.lstm = nn.LSTM(lstm_in, lstm_hidden, num_layers, batch_first=True, dropout=0.1)
        self.lstm_fc = nn.Sequential(
            nn.Linear(lstm_hidden * 80, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.gru_proj = nn.Sequential(
            nn.Linear(gru_in, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(dropout_rate), nn.Linear(128, 64), nn.ReLU()
        )
        self.gru = nn.GRU(64, gru_hidden, num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(gru_hidden * 2, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.gru_fc = nn.Sequential(
            nn.Linear(gru_hidden * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.fusion_classifier = nn.Sequential(
            nn.Linear(128 + 128, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_lstm, x_gru):
        h0 = torch.zeros(self.lstm_layers, x_lstm.size(0), self.lstm_hidden).to(x_lstm.device)
        c0 = torch.zeros(self.lstm_layers, x_lstm.size(0), self.lstm_hidden).to(x_lstm.device)
        out_lstm, _ = self.lstm(x_lstm, (h0, c0))
        out_lstm = out_lstm.reshape(out_lstm.shape[0], -1)
        feat_lstm = self.lstm_fc(out_lstm)

        batch_size, seq_len, _ = x_gru.shape
        x_g_proj = self.gru_proj(x_gru.reshape(-1, x_gru.size(-1))).reshape(batch_size, seq_len, -1)
        gru_out, _ = self.gru(x_g_proj)

        attn_w = torch.softmax(self.attention(gru_out).squeeze(-1), dim=1).unsqueeze(-1)
        weighted_gru = torch.sum(attn_w * gru_out, dim=1)
        feat_gru = self.gru_fc(weighted_gru)

        fused_features = torch.cat([feat_lstm, feat_gru], dim=1)
        output = self.fusion_classifier(fused_features)
        return output


class FusionGestureNoAttentionModel(nn.Module):
    """DSTG-Net variant without temporal attention; BiGRU states are mean-pooled."""
    def __init__(self, lstm_in=240, gru_in=58, lstm_hidden=32, gru_hidden=64, num_layers=2, num_classes=12,
                 dropout_rate=0.4):
        super(FusionGestureNoAttentionModel, self).__init__()
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = num_layers
        self.lstm = nn.LSTM(lstm_in, lstm_hidden, num_layers, batch_first=True, dropout=0.1)
        self.lstm_fc = nn.Sequential(
            nn.Linear(lstm_hidden * 80, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.gru_proj = nn.Sequential(
            nn.Linear(gru_in, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(dropout_rate), nn.Linear(128, 64), nn.ReLU()
        )
        self.gru = nn.GRU(64, gru_hidden, num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)
        self.gru_fc = nn.Sequential(
            nn.Linear(gru_hidden * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.fusion_classifier = nn.Sequential(
            nn.Linear(128 + 128, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_lstm, x_gru):
        h0 = torch.zeros(self.lstm_layers, x_lstm.size(0), self.lstm_hidden).to(x_lstm.device)
        c0 = torch.zeros(self.lstm_layers, x_lstm.size(0), self.lstm_hidden).to(x_lstm.device)
        out_lstm, _ = self.lstm(x_lstm, (h0, c0))
        out_lstm = out_lstm.reshape(out_lstm.shape[0], -1)
        feat_lstm = self.lstm_fc(out_lstm)

        batch_size, seq_len, _ = x_gru.shape
        x_g_proj = self.gru_proj(x_gru.reshape(-1, x_gru.size(-1))).reshape(batch_size, seq_len, -1)
        gru_out, _ = self.gru(x_g_proj)
        mean_gru = torch.mean(gru_out, dim=1)
        feat_gru = self.gru_fc(mean_gru)

        fused_features = torch.cat([feat_lstm, feat_gru], dim=1)
        return self.fusion_classifier(fused_features)


# ==================== 3. 核心数据预处理流水线 ====================
def read_radar_csv(file_path):
    """兼容一行表头CSV和额外元数据表头CSV。"""
    df = pd.read_csv(file_path)
    required_cols = {'FrameNumber', 'Velocity', 'x', 'y'}
    if required_cols.issubset(set(df.columns)):
        return df
    return pd.read_csv(file_path, skiprows=1)


def build_temporal_feature(ff_arr, diff_order=1):
    temporal_parts = [ff_arr]
    feat_dim = ff_arr.shape[1]
    if diff_order >= 1:
        diff_arr = np.diff(ff_arr, axis=0)
        diff_arr = np.vstack([np.zeros((1, feat_dim), dtype=np.float32), diff_arr])
        temporal_parts.append(diff_arr)
    if diff_order >= 2:
        diff2_arr = np.diff(ff_arr, n=2, axis=0)
        diff2_arr = np.vstack([np.zeros((2, feat_dim), dtype=np.float32), diff2_arr])
        temporal_parts.append(diff2_arr)
    return np.concatenate(temporal_parts, axis=1)


def process_single_file(file_path, augment=False, diff_order=1):
    try:
        df = read_radar_csv(file_path)
        cols = df.columns.tolist()
        frame_col = cols[0]
        vel_col = cols[3] if len(cols) > 3 else 'Velocity'
        peak_col = 'PeakValue' if 'PeakValue' in cols else (cols[4] if len(cols) > 4 else 'PeakVal')
        x_col = cols[5] if len(cols) > 5 else 'x'
        y_col = cols[6] if len(cols) > 6 else 'y'

        frames_data = []
        for frame_id, group in df.groupby(frame_col):
            x_vals = group[x_col].values
            y_vals = group[y_col].values
            v_vals = group[vel_col].values
            p_vals = group[peak_col].values if peak_col in group else np.zeros_like(v_vals)

            if augment:
                x_vals += np.random.normal(0, 0.01, size=x_vals.shape)
                y_vals += np.random.normal(0, 0.01, size=y_vals.shape)

            pts = list(zip(x_vals, y_vals, v_vals, p_vals))
            frames_data.append(pts)

        MAX_FRAMES, MAX_POINTS = 80, 80

        # LSTM Feature: x, y, v (3 channels)
        lstm_data = np.zeros((MAX_FRAMES, 3, MAX_POINTS), dtype=np.float32)
        for frame_idx, frame in enumerate(frames_data):
            if frame_idx >= MAX_FRAMES: break
            if frame:
                limited = frame[:MAX_POINTS]
                x_p = np.array([p[0] for p in limited])
                y_p = np.array([p[1] for p in limited])
                v_p = np.array([p[2] for p in limited])
            else:
                x_p, y_p, v_p = np.array([]), np.array([]), np.array([])
            pw = max(0, MAX_POINTS - len(x_p))
            lstm_data[frame_idx] = np.array([
                np.pad(x_p, (0, pw)), np.pad(y_p, (0, pw)), np.pad(v_p, (0, pw))
            ])
        lstm_feature = lstm_data.reshape(MAX_FRAMES, 240)

        # GRU Feature: v, p, x, y (4 physical quantities)
        frame_features = []
        for frame_idx, frame in enumerate(frames_data):
            if frame_idx >= MAX_FRAMES: break
            fv = []
            if frame:
                limited = frame[:MAX_POINTS]
                v_arr = np.array([p[2] for p in limited])
                p_arr = np.array([p[3] for p in limited])
                x_arr = np.array([p[0] for p in limited])
                y_arr = np.array([p[1] for p in limited])
                
                features_list = [v_arr, p_arr, x_arr, y_arr]
                non_zero_mask_total = np.zeros(len(limited), dtype=bool)
                for feat in features_list:
                    non_zero_mask = feat != 0
                    non_zero_mask_total = non_zero_mask_total | non_zero_mask
                    non_zero_values = feat[non_zero_mask]
                    if len(non_zero_values) > 0:
                        stats = [np.mean(non_zero_values), np.std(non_zero_values), np.median(non_zero_values),
                                 np.max(non_zero_values), np.min(non_zero_values), np.percentile(non_zero_values, 25),
                                 np.percentile(non_zero_values, 75)]
                    else:
                        stats = [0.0] * 7
                    fv.extend(stats)
                fv.append(np.sum(non_zero_mask_total))
            else:
                fv = [0.0] * 29
            frame_features.append(fv)

        while len(frame_features) < MAX_FRAMES:
            frame_features.append([0.0] * 29)

        ff_arr = np.array(frame_features, dtype=np.float32)
        
        gru_feature = build_temporal_feature(ff_arr, diff_order=diff_order)

        return lstm_feature, gru_feature
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None


# ==================== 4. 数据集构造 ====================
class RadarGestureDataset(Dataset):
    def __init__(self, data_path, is_train=True, diff_order=1):
        self.X_lstm = []
        self.X_gru = []
        self.y = []
        self.diff_order = diff_order

        print(f"正在加载 {'训练集' if is_train else '验证集'} 数据，请稍候... diff_order={diff_order}")
        for label, gesture_name in LABEL_FULL_NAMES.items():
            # 使用完整标签名称访问文件夹（而非缩写标签）
            folder = os.path.join(data_path, gesture_name)
            if not os.path.exists(folder):
                continue

            files = sorted([f for f in os.listdir(folder) if f.endswith('.csv')])
            for file_name in files:
                file_path = os.path.join(folder, file_name)
                lstm_f, gru_f = process_single_file(file_path, augment=is_train, diff_order=diff_order)
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


# ==================== 5. 单模型基础绘图与训练逻辑 ====================
def plot_loss_curve(loss, output_folder, filename='loss_curve.pdf'):
    epochs = range(1, len(loss) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss, label='Training Loss', color='#1f77b4', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_folder, filename), format='pdf', bbox_inches='tight')
    plt.close()


def plot_accuracy_curve(val_acc, output_folder, filename='accuracy_curve.pdf'):
    epochs = range(1, len(val_acc) + 1)
    val_acc_percent = [acc * 100 for acc in val_acc]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_acc_percent, label='Validation Accuracy', color='#ff7f0e', linewidth=2, marker='s',
             markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Validation Accuracy Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_folder, filename), format='pdf', bbox_inches='tight')
    plt.close()


def train_model(model, model_name, train_loader, val_loader, epochs=30):
    print(f"\n{'=' * 20} 开始训练: {model_name} {'=' * 20}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SCHEDULER_T_MAX)

    best_acc = -1.0
    best_metrics = {}
    best_model_state = None

    loss_history = []
    val_acc_history = []
    val_f1_history = []
    terminal_logs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}] [Train]", colour='green', leave=False)
        for b_lstm, b_gru, b_y in train_pbar:
            b_lstm, b_gru, b_y = b_lstm.to(DEVICE), b_gru.to(DEVICE), b_y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(b_lstm, b_gru)
            loss = criterion(outputs, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for b_lstm, b_gru, b_y in val_loader:
                b_lstm, b_gru = b_lstm.to(DEVICE), b_gru.to(DEVICE)
                outputs = model(b_lstm, b_gru)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(b_y.numpy())

        acc = accuracy_score(all_targets, all_preds)
        macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        avg_loss = total_loss / len(train_loader)

        loss_history.append(avg_loss)
        val_acc_history.append(acc)
        val_f1_history.append(macro_f1)

        log_str = f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.4f} - Val Acc: {acc * 100:.2f}% - Val F1: {macro_f1:.4f}"
        print(log_str)
        terminal_logs.append(log_str)

        if acc > best_acc:
            best_acc = acc
            best_model_state = copy.deepcopy(model.state_dict())
            best_metrics = {
                'model_name': model_name,
                'best_epoch': epoch + 1,
                'acc': acc,
                'macro_f1': macro_f1,
                'preds': all_preds.copy(),
                'targets': all_targets.copy()
            }

    print(f"\n{model_name} 训练完成! 最佳整体准确率: {best_metrics['acc'] * 100:.2f}%")

    torch.save({
        'model_name': model_name,
        'model_state_dict': best_model_state,
        'best_epoch': best_metrics['best_epoch'],
        'macro_f1': best_metrics['macro_f1'] * 100,
        'accuracy': best_metrics['acc'] * 100,
        'parameters': count_trainable_params(model)
    }, os.path.join(RESULT_DIR, f'best_{model_name}.pth'))

    log_file_path = os.path.join(RESULT_DIR, f'{model_name}_training_logs.txt')
    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(terminal_logs))

    # 单模型绘图
    plot_loss_curve(loss_history, RESULT_DIR, f'{model_name}_loss_curve.pdf')
    plot_accuracy_curve(val_acc_history, RESULT_DIR, f'{model_name}_accuracy_curve.pdf')

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    avg_latency_ms, fps = benchmark_inference(model, val_loader)
    best_metrics['parameters'] = count_trainable_params(model)
    best_metrics['avg_inference_latency_ms'] = avg_latency_ms
    best_metrics['fps'] = fps

    return best_metrics, loss_history, val_acc_history


def visualize_single_result(metrics, model_name):
    """保存原有的单模型表现（矩阵、柱状图、表格）并返回类准确率用于消融绘图"""
    if not metrics: return None
    class_names = [REVERSE_LABELS[i] for i in range(NUM_CLASSES)]
    targets = metrics['targets']
    preds = metrics['preds']

    overall_acc = accuracy_score(targets, preds) * 100
    cm = confusion_matrix(targets, preds)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        os.path.join(RESULT_DIR, f'Confusion_Matrix_{model_name}.csv'), encoding='utf-8-sig'
    )
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_perc, annot=True, fmt='.1f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix (%) - {model_name}', fontsize=14)
    plt.ylabel('True Gesture', fontsize=12)
    plt.xlabel('Predicted Gesture', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{RESULT_DIR}/Confusion_Matrix_{model_name}.pdf', format='pdf', dpi=300)
    plt.close()

    report_data = []
    class_accs = []
    for i in range(NUM_CLASSES):
        idx = np.where(np.array(targets) == i)[0]
        acc_i = accuracy_score(np.array(targets)[idx], np.array(preds)[idx]) if len(idx) > 0 else 0
        class_accs.append(acc_i * 100)

    precision = precision_score(targets, preds, average=None, zero_division=0)
    recall = recall_score(targets, preds, average=None, zero_division=0)
    f1 = f1_score(targets, preds, average=None, zero_division=0)

    for i, name in enumerate(class_names):
        report_data.append([name, f"{class_accs[i]:.2f}%", f"{precision[i]:.4f}", f"{recall[i]:.4f}", f"{f1[i]:.4f}"])

    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, class_accs, color='skyblue', edgecolor='black')
    plt.axhline(overall_acc, color='red', linestyle='--', label=f'Overall Acc: {overall_acc:.2f}%')
    plt.ylim(0, 110)
    plt.title(f'Accuracy per Gesture Class - {model_name}', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=9)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{RESULT_DIR}/Accuracy_Histogram_{model_name}.pdf', format='pdf', dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')
    columns = ('Gesture Class', 'Accuracy', 'Precision', 'Recall', 'F1-Score')
    table = ax.table(cellText=report_data, colLabels=columns, cellLoc='center', loc='center', bbox=[0, 0, 1, 0.85])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.title(f'Performance Metrics Table - {model_name}', fontsize=16, y=0.95, fontweight='bold')
    plt.savefig(f'{RESULT_DIR}/Metrics_Table_{model_name}.pdf', format='pdf', bbox_inches='tight')
    plt.close()

    return class_accs


# ==================== 6. 综合消融实验绘图 (三合一) ====================
def plot_ablation_loss(losses_dict, output_folder):
    """绘制三模型 Loss 对比图"""
    plt.figure(figsize=(10, 6))
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for idx, (name, loss_data) in enumerate(losses_dict.items()):
        epochs = range(1, len(loss_data) + 1)
        plt.plot(epochs, loss_data, label=f'{name} Loss', color=color_list[idx % len(color_list)],
                 linewidth=1.5)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Ablation Study: Training Loss Comparison', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_folder, 'Ablation_Loss_Curve.pdf'), format='pdf', bbox_inches='tight')
    plt.close()


def plot_ablation_accuracy(accs_dict, output_folder):
    """绘制三模型 Accuracy 对比图"""
    plt.figure(figsize=(10, 6))
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    all_min = 100.0
    for idx, (name, acc_data) in enumerate(accs_dict.items()):
        epochs = range(1, len(acc_data) + 1)
        acc_percent = [a * 100 for a in acc_data]
        plt.plot(epochs, acc_percent, label=f'{name} Val Acc', color=color_list[idx % len(color_list)],
                 linewidth=1.5)
        all_min = min(all_min, min(acc_percent))

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Accuracy (%)', fontsize=12)
    plt.title('Ablation Study: Validation Accuracy Comparison', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(60, 100)
    plt.savefig(os.path.join(output_folder, 'Ablation_Accuracy_Curve.pdf'), format='pdf', bbox_inches='tight')
    plt.close()


def plot_ablation_histogram(class_accs_dict, output_folder):
    """绘制三模型十二分类准确率对比柱状图"""
    class_names = [REVERSE_LABELS[i] for i in range(NUM_CLASSES)]
    x = np.arange(len(class_names))
    n_models = len(class_accs_dict)
    width = min(0.22, 0.78 / max(1, n_models))
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    fig, ax = plt.subplots(figsize=(16, 7))

    rect_groups = []
    start = -width * (n_models - 1) / 2
    for idx, (name, accs) in enumerate(class_accs_dict.items()):
        rects = ax.bar(x + start + idx * width, accs, width, label=name,
                       color=color_list[idx % len(color_list)], edgecolor='black', alpha=0.9)
        rect_groups.append(rects)

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Ablation Study: Accuracy per Gesture Class (12 Classes)', fontsize=15, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=6, fontsize=11)

    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    # 可以在柱子上添加具体数值，此处避免重叠适当留白
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.0f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, rotation=90)

    for rects in rect_groups:
        autolabel(rects)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'Ablation_Class_Accuracy_Histogram.pdf'), format='pdf', dpi=300)
    plt.close()


# ==================== 7. 主程序入口 (消融实验流) ====================
if __name__ == '__main__':
    set_seed(SEED)
    print("=" * 60)
    print(" 毫米波雷达手势识别 - 消融实验综合对比管线启动 ")
    print(" (LSTM vs Attention-BiGRU vs DSTG-Net) ")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"当前目录下未找到 '{DATA_PATH}' 文件夹，请先准备数据集。")
    else:
        # 1. 准备全局数据加载器 (保证所有模型的数据拆分严格一致)
        full_dataset = RadarGestureDataset(DATA_PATH, is_train=False, diff_order=1)
        indices = np.arange(len(full_dataset))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=full_dataset.y, random_state=SEED)

        def make_loaders(dataset):
            train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
            val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=torch.utils.data.SubsetRandomSampler(val_idx))
            return train_loader, val_loader

        train_loader, val_loader = make_loaders(full_dataset)

        epochs_to_train = GLOBAL_EPOCHS
        summary_rows = []
        ablation_losses = {}
        ablation_accs = {}
        ablation_class_accs = {}

        def register_result(model_name, metrics, loss_history, acc_history, class_accs, variant_desc):
            summary_rows.append({
                'Model': model_name,
                'Variant': variant_desc,
                'Best Epoch': metrics['best_epoch'],
                'Accuracy (%)': metrics['acc'] * 100,
                'Macro F1 (%)': metrics['macro_f1'] * 100,
                'Trainable Parameters': metrics['parameters'],
                'Average Inference Latency (ms/sample)': metrics['avg_inference_latency_ms'],
                'FPS': metrics['fps'],
                'Device': str(DEVICE),
                'Seed': SEED,
                'Epochs': GLOBAL_EPOCHS,
                'Batch Size': BATCH_SIZE,
                'Learning Rate': LEARNING_RATE,
                'Optimizer': 'AdamW',
                'Scheduler': f'CosineAnnealingLR(T_max={SCHEDULER_T_MAX})'
            })
            ablation_losses[model_name] = loss_history
            ablation_accs[model_name] = acc_history
            ablation_class_accs[model_name] = class_accs

        # ----------------- 实验A: LSTM -----------------
        set_seed(SEED)
        model_lstm = LSTMOnlyModel(num_classes=NUM_CLASSES).to(DEVICE)
        metrics_lstm, loss_lstm, acc_lstm = train_model(model_lstm, "LSTM", train_loader, val_loader,
                                                        epochs=epochs_to_train)
        class_acc_lstm = visualize_single_result(metrics_lstm, "LSTM")
        register_result("LSTM", metrics_lstm, loss_lstm, acc_lstm, class_acc_lstm, "Only spatial point-cloud branch")

        # ----------------- 实验B: Attention-BiGRU -----------------
        set_seed(SEED)
        model_gru = AttBiGRUOnlyModel(num_classes=NUM_CLASSES).to(DEVICE)
        metrics_gru, loss_gru, acc_gru = train_model(model_gru, "Attention-BiGRU", train_loader, val_loader,
                                                     epochs=epochs_to_train)
        class_acc_gru = visualize_single_result(metrics_gru, "Attention-BiGRU")
        register_result("Attention-BiGRU", metrics_gru, loss_gru, acc_gru, class_acc_gru,
                        "Only Attention-BiGRU temporal branch")

        # ----------------- 实验D: 无一阶差分 -----------------
        set_seed(SEED)
        dataset_no_diff = RadarGestureDataset(DATA_PATH, is_train=False, diff_order=0)
        train_loader_no_diff, val_loader_no_diff = make_loaders(dataset_no_diff)
        set_seed(SEED)
        model_no_diff = FusionGestureModel(gru_in=29, num_classes=NUM_CLASSES).to(DEVICE)
        metrics_no_diff, loss_no_diff, acc_no_diff = train_model(model_no_diff, "DSTG-NoDiff",
                                                                  train_loader_no_diff, val_loader_no_diff,
                                                                  epochs=epochs_to_train)
        class_acc_no_diff = visualize_single_result(metrics_no_diff, "DSTG-NoDiff")
        register_result("DSTG-NoDiff", metrics_no_diff, loss_no_diff, acc_no_diff, class_acc_no_diff,
                        "Full model without temporal difference features")

        # ----------------- 实验E: 二阶差分 -----------------
        set_seed(SEED)
        dataset_second_diff = RadarGestureDataset(DATA_PATH, is_train=False, diff_order=2)
        train_loader_second_diff, val_loader_second_diff = make_loaders(dataset_second_diff)
        set_seed(SEED)
        model_second_diff = FusionGestureModel(gru_in=87, num_classes=NUM_CLASSES).to(DEVICE)
        metrics_second_diff, loss_second_diff, acc_second_diff = train_model(model_second_diff, "DSTG-SecondDiff",
                                                                             train_loader_second_diff,
                                                                             val_loader_second_diff,
                                                                             epochs=epochs_to_train)
        class_acc_second_diff = visualize_single_result(metrics_second_diff, "DSTG-SecondDiff")
        register_result("DSTG-SecondDiff", metrics_second_diff, loss_second_diff, acc_second_diff,
                        class_acc_second_diff, "Full model with first- and second-order difference features")

        # ----------------- 实验F: 无注意力 -----------------
        set_seed(SEED)
        model_no_attention = FusionGestureNoAttentionModel(num_classes=NUM_CLASSES).to(DEVICE)
        metrics_no_attention, loss_no_attention, acc_no_attention = train_model(model_no_attention,
                                                                                "DSTG-NoAttention",
                                                                                train_loader, val_loader,
                                                                                epochs=epochs_to_train)
        class_acc_no_attention = visualize_single_result(metrics_no_attention, "DSTG-NoAttention")
        register_result("DSTG-NoAttention", metrics_no_attention, loss_no_attention, acc_no_attention,
                        class_acc_no_attention, "Full model with mean pooling instead of temporal attention")

        # ----------------- 实验C: DSTG-Net -----------------
        set_seed(SEED)
        model_fusion = FusionGestureModel(num_classes=NUM_CLASSES).to(DEVICE)
        metrics_fusion, loss_fusion, acc_fusion = train_model(model_fusion, "DSTG-Net", train_loader, val_loader,
                                                              epochs=epochs_to_train)
        class_acc_fusion = visualize_single_result(metrics_fusion, "DSTG-Net")
        register_result("DSTG-Net", metrics_fusion, loss_fusion, acc_fusion, class_acc_fusion,
                        "Full dual-stream model with first-order difference and attention")


        # ----------------- 生成综合对比图表 -----------------
        print("\n正在生成三模型对比消融图表 (Loss / Acc / 12类别柱状图)...")

        plot_ablation_loss(ablation_losses, RESULT_DIR)
        plot_ablation_accuracy(ablation_accs, RESULT_DIR)
        plot_ablation_histogram(ablation_class_accs, RESULT_DIR)

        summary_path = os.path.join(RESULT_DIR, 'Ablation_Summary.csv')
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"消融实验汇总表已保存: {summary_path}")

        print(f"\n🎉 恭喜！综合消融实验已全部执行完毕！")
        print(f"所有单模型的评估图表及【三线合一对比图】已保存至 '{RESULT_DIR}' 文件夹中！")
