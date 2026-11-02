import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm
from scipy.fftpack import fft

# ----------------- 框架级依赖 -----------------
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Flatten, Dense, Dropout, 
                                     BatchNormalization, Input, Add, Activation, LSTM)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix)
from imblearn.over_sampling import SMOTE

# ----------------- 解决 GPU 显存冲突 -----------------
# 使得 TensorFlow 按需申请显存，防止霸占全部显存导致 PyTorch 崩溃
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# ----------------- 全局配置 -----------------
# 样式设置
try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
except ImportError:
    print("未安装 scienceplots，将使用 seaborn 默认科研风格。")
    plt.style.use('seaborn-v0_8-paper')
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
plt.rcParams['axes.unicode_minus'] = False    # 负号支持

# 核心全局变量
DATA_PATH = 'data'
RESULT_DIR = 'Result_Contrast_Study-200-pub'
GLOBAL_EPOCHS = 200  # <--- 可供自由设置的全局迭代次数
os.makedirs(RESULT_DIR, exist_ok=True)

# 统一 12 种手势标签字典，确保所有模型类别对齐
GESTURES = ['arm_to_left', 'arm_to_right', 'close_fist_horizontally',
            'close_fist_perpendicularly', 'hand_away', 'hand_closer',
            'hand_down', 'hand_rotation_palm_down', 'hand_rotation_palm_up',
            'hand_to_left', 'hand_to_right', 'hand_up']
LABELS_DICT = {gesture: idx for idx, gesture in enumerate(GESTURES)}
REVERSE_LABELS = {v: k for k, v in LABELS_DICT.items()}
NUM_CLASSES = len(GESTURES)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =====================================================================
# 模块 A：统一的可视化引擎 (参考 DSTG-Net-pro_V2)
# =====================================================================

def plot_single_curves(loss, val_acc, output_folder, model_name):
    epochs = range(1, len(loss) + 1)
    
    # Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss, label='Training Loss', color='#1f77b4', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Curve - {model_name}') 
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_folder, f'{model_name}_loss_curve.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
    
    # Acc Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_acc, label='Validation Accuracy', color='#ff7f0e', linewidth=2, marker='s', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Validation Accuracy Curve - {model_name}') 
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_folder, f'{model_name}_accuracy_curve.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

def visualize_single_model_results(targets, preds, model_name):
    overall_acc = accuracy_score(targets, preds) * 100
    cm = confusion_matrix(targets, preds)
    
    # 1. 混淆矩阵 (带数量)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=GESTURES, yticklabels=GESTURES)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
    plt.ylabel('True Gesture')
    plt.xlabel('Predicted Gesture')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f'Confusion_Matrix_Count_{model_name}.pdf'), format='pdf', dpi=300)
    plt.close()

    # 2. 混淆矩阵 (带百分比)
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    cm_percent_str = np.array([[f"{value:.1f}%" for value in row] for row in cm_perc])
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_perc, annot=cm_percent_str, fmt='', cmap='Blues', xticklabels=GESTURES, yticklabels=GESTURES)
    plt.title(f'Confusion Matrix (%) - {model_name}', fontsize=14)
    plt.ylabel('True Gesture')
    plt.xlabel('Predicted Gesture')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f'Confusion_Matrix_Percent_{model_name}.pdf'), format='pdf', dpi=300)
    plt.close()

    # 指标计算
    class_accs = cm.diagonal() / cm.sum(axis=1) * 100
    precision = precision_score(targets, preds, average=None, zero_division=0)
    recall = recall_score(targets, preds, average=None, zero_division=0)
    f1 = f1_score(targets, preds, average=None, zero_division=0)

    # 3. 各类别准确率柱状图
    plt.figure(figsize=(12, 6))
    bars = plt.bar(GESTURES, class_accs, color='skyblue', edgecolor='black')
    plt.axhline(overall_acc, color='red', linestyle='--', label=f'Overall Acc: {overall_acc:.2f}%')
    plt.ylim(0, 110)
    plt.title(f'Accuracy per Gesture Class - {model_name}', fontsize=14)
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=9)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f'Accuracy_Histogram_{model_name}.pdf'), format='pdf', dpi=300)
    plt.close()

    # 4. 指标表格 (PDF)
    report_data = []
    for i, name in enumerate(GESTURES):
        report_data.append([name, f"{class_accs[i]:.2f}%", f"{precision[i]:.4f}", f"{recall[i]:.4f}", f"{f1[i]:.4f}"])
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')
    columns = ('Gesture Class', 'Accuracy', 'Precision', 'Recall', 'F1-Score')
    table = ax.table(cellText=report_data, colLabels=columns, cellLoc='center', loc='center', bbox=[0, 0, 1, 0.85])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.title(f'Performance Metrics Table - {model_name}', fontsize=16, y=0.95, fontweight='bold')
    plt.savefig(os.path.join(RESULT_DIR, f'Metrics_Table_{model_name}.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
    
    return class_accs # 返回各类别准确率供联合图表使用

# =====================================================================
# 模块 B：TensorFlow 模型群 (单CNN, VGGNet, ResNet, STGNet)
# =====================================================================

# TF 自定义回调：接管打印，计算验证集F1并保存TXT日志
class TF_CustomLogger(Callback):
    def __init__(self, validation_data, log_file_path, total_epochs):
        super(TF_CustomLogger, self).__init__()
        self.validation_data = validation_data
        self.log_file_path = log_file_path
        self.total_epochs = total_epochs
        self.loss_history = []
        self.val_acc_history = []
        
        with open(self.log_file_path, 'w', encoding='utf-8') as f:
            f.write("Training Logs\n")
            f.write("=" * 60 + "\n")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss', 0.0)
        val_acc = logs.get('val_accuracy', 0.0)
        
        self.loss_history.append(loss)
        self.val_acc_history.append(val_acc)

        X_val, y_val = self.validation_data
        y_pred = self.model.predict(X_val, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_val, axis=1)
        val_f1 = f1_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
        
        log_str = f"Epoch [{epoch + 1}/{self.total_epochs}] - Loss: {loss:.4f} - Val Acc: {val_acc * 100:.2f}% - Val F1: {val_f1:.4f}"
        print(log_str)
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(log_str + '\n')

# 1. 序列化数据预处理 (CNN, VGGNet, ResNet 通用)
def load_tf_seq_data():
    print("正在加载 TF 序列化数据 (CNN/VGGNet/ResNet)...")
    X, y = [], []
    for gesture_id, gesture in enumerate(GESTURES):
        gesture_folder = os.path.join(DATA_PATH, gesture)
        if not os.path.exists(gesture_folder): continue
        for file_name in os.listdir(gesture_folder):
            if file_name.endswith('.csv'):
                data = pd.read_csv(os.path.join(gesture_folder, file_name))
                features = data[['Range', 'Velocity', 'x', 'y']].values
                X.append(features)
                y.append(gesture_id)
    X = pad_sequences(X, maxlen=100, dtype='float32', padding='post', truncating='post')
    y = to_categorical(np.array(y), num_classes=NUM_CLASSES)
    
    # 划分数据集 (严格参照原代码: 80% Train, 20% Test. Train 再分 80/20 给 Val)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
    return X_train, y_train, X_val, y_val, X_test, y_test

# 2. STGNet 数据预处理 (特征提取 + SMOTE)
def load_tf_stgnet_data():
    print("正在加载 STGNet 数据 (特征工程 + SMOTE)...")
    X, y = [], []
    for gesture_id, gesture in enumerate(GESTURES):
        gesture_folder = os.path.join(DATA_PATH, gesture)
        if not os.path.exists(gesture_folder): continue
        for file_name in os.listdir(gesture_folder):
            if file_name.endswith('.csv'):
                data = pd.read_csv(os.path.join(gesture_folder, file_name))
                features = data[['Range', 'Velocity', 'x', 'y']].values
                
                mean_f = np.mean(features, axis=0)
                std_f = np.std(features, axis=0)
                min_f = np.min(features, axis=0)
                max_f = np.max(features, axis=0)
                diff_f = np.diff(features, axis=0)
                mean_diff = np.mean(diff_f, axis=0) if len(diff_f) > 0 else np.zeros(4)
                std_diff = np.std(diff_f, axis=0) if len(diff_f) > 0 else np.zeros(4)
                fft_f = np.abs(fft(features, axis=0))
                mean_fft = np.mean(fft_f, axis=0)
                std_fft = np.std(fft_f, axis=0)
                
                comb = np.concatenate([mean_f, std_f, min_f, max_f, mean_diff, std_diff, mean_fft, std_fft])
                X.append(comb)
                y.append(gesture_id)
                
    X, y = np.array(X), np.array(y)
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)
    
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_full = scaler.fit_transform(X_train_full)
    X_test = scaler.transform(X_test)
    
    y_train_full = to_categorical(y_train_full, num_classes=NUM_CLASSES)
    y_test = to_categorical(y_test, num_classes=NUM_CLASSES)
    
    X_train_full = X_train_full.reshape(X_train_full.shape[0], X_train_full.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
    return X_train, y_train, X_val, y_val, X_test, y_test

# --- 定义模型结构 ---
def build_cnn_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_vggnet_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(64, 3, activation='relu', padding='same'),
        Conv1D(64, 3, activation='relu', padding='same'),
        MaxPooling1D(2, strides=2),
        Conv1D(128, 3, activation='relu', padding='same'),
        Conv1D(128, 3, activation='relu', padding='same'),
        MaxPooling1D(2, strides=2),
        Conv1D(256, 3, activation='relu', padding='same'),
        Conv1D(256, 3, activation='relu', padding='same'),
        Conv1D(256, 3, activation='relu', padding='same'),
        MaxPooling1D(2, strides=2),
        Conv1D(512, 3, activation='relu', padding='same'),
        Conv1D(512, 3, activation='relu', padding='same'),
        Conv1D(512, 3, activation='relu', padding='same'),
        MaxPooling1D(2, strides=2),
        Conv1D(512, 3, activation='relu', padding='same'),
        Conv1D(512, 3, activation='relu', padding='same'),
        Conv1D(512, 3, activation='relu', padding='same'),
        MaxPooling1D(2, strides=2),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_resnet_model(input_shape):
    def residual_block(x, filters, kernel_size=3, stride=1):
        shortcut = x
        x = Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv1D(filters, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        if shortcut.shape[-1] != filters:
            shortcut = Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x

    input_layer = Input(shape=input_shape)
    x = Conv1D(64, 7, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(3)(x)
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_stgnet_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=256, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        LSTM(256, return_sequences=True),
        Dropout(0.5),
        LSTM(128),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 统一 TF 训练器
def train_evaluate_tf(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test):
    print(f"\n{'='*20} 开始训练 {model_name} {'='*20}")
    log_file = os.path.join(RESULT_DIR, f'{model_name}_training_logs.txt')
    ckpt_path = os.path.join(RESULT_DIR, f'best_model_{model_name}.keras')
    
    custom_logger = TF_CustomLogger(validation_data=(X_val, y_val), log_file_path=log_file, total_epochs=GLOBAL_EPOCHS)
    checkpoint = ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=0)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=8000, restore_best_weights=True, mode='max', verbose=0)

    model.fit(X_train, y_train, epochs=GLOBAL_EPOCHS, batch_size=128, 
              validation_data=(X_val, y_val), 
              callbacks=[checkpoint, early_stopping, custom_logger], verbose=0)

    # 评估与绘图
    model.load_weights(ckpt_path)
    y_pred = model.predict(X_test, verbose=0)
    preds = np.argmax(y_pred, axis=1)
    targets = np.argmax(y_test, axis=1)
    
    plot_single_curves(custom_logger.loss_history, custom_logger.val_acc_history, RESULT_DIR, model_name)
    class_accs = visualize_single_model_results(targets, preds, model_name)
    
    return {
        'loss': custom_logger.loss_history,
        'val_acc': custom_logger.val_acc_history,
        'class_accs': class_accs
    }

# =====================================================================
# 模块 C：PyTorch 模型群 (DSTG-Net-pro)
# =====================================================================

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
            if frame_idx >= MAX_FRAMES: break
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
            if frame_idx >= MAX_FRAMES: break
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
            
        while len(frame_features) < MAX_FRAMES: frame_features.append([0.0]*29)
        ff_arr = np.array(frame_features, dtype=np.float32)
        
        temporal_features = []
        for i in range(1, len(ff_arr)): temporal_features.extend(ff_arr[i] - ff_arr[i-1])
        exp_temp = 79 * 29 
        if len(temporal_features) < exp_temp: temporal_features.extend([0.0]*(exp_temp - len(temporal_features)))
        else: temporal_features = temporal_features[:exp_temp]
        
        combined = np.concatenate([ff_arr.flatten(), temporal_features])
        tgt_size = 80 * 58 
        if len(combined) < tgt_size: combined = np.concatenate([combined, np.zeros(tgt_size - len(combined))])
        else: combined = combined[:tgt_size]
        gru_feature = combined.reshape(MAX_FRAMES, 58)
        
        return lstm_feature, gru_feature
    except Exception: return None, None

class RadarGestureDataset(Dataset):
    def __init__(self):
        self.X_lstm, self.X_gru, self.y = [], [], []
        print("正在加载 PyTorch 双流特征数据 (DSTG-Net)...")
        for gesture, label in LABELS_DICT.items():
            folder = os.path.join(DATA_PATH, gesture)
            if not os.path.exists(folder): continue
            for file_name in os.listdir(folder):
                if file_name.endswith('.csv'):
                    lstm_f, gru_f = process_single_file_pt(os.path.join(folder, file_name))
                    if lstm_f is not None:
                        self.X_lstm.append(lstm_f)
                        self.X_gru.append(gru_f)
                        self.y.append(label)

    def __len__(self): return len(self.y)
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

def train_evaluate_pytorch():
    print(f"\n{'='*20} 开始训练 DSTG-Net {'='*20}")
    full_dataset = RadarGestureDataset()
    indices = np.arange(len(full_dataset))
    # 按照原文划分: 80% 训练, 20% 测试验证
    train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=full_dataset.y, random_state=42)
    
    train_loader = DataLoader(full_dataset, batch_size=32, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
    val_loader = DataLoader(full_dataset, batch_size=32, sampler=torch.utils.data.SubsetRandomSampler(val_idx))

    model = FusionGestureModel(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    loss_history, val_acc_history, terminal_logs = [], [], []
    best_macro_f1, best_targets, best_preds = 0.0, [], []

    for epoch in range(GLOBAL_EPOCHS):
        model.train()
        total_loss = 0
        for b_lstm, b_gru, b_y in train_loader:
            b_lstm, b_gru, b_y = b_lstm.to(DEVICE), b_gru.to(DEVICE), b_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(b_lstm, b_gru)
            loss = criterion(outputs, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for b_lstm, b_gru, b_y in val_loader:
                b_lstm, b_gru = b_lstm.to(DEVICE), b_gru.to(DEVICE)
                _, preds = torch.max(model(b_lstm, b_gru), 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(b_y.numpy())

        acc = accuracy_score(all_targets, all_preds)
        macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        avg_loss = total_loss / len(train_loader)
        
        loss_history.append(avg_loss)
        val_acc_history.append(acc)
        
        log_str = f"Epoch [{epoch+1}/{GLOBAL_EPOCHS}] - Loss: {avg_loss:.4f} - Val Acc: {acc*100:.2f}% - Val F1: {macro_f1:.4f}"
        print(log_str)
        terminal_logs.append(log_str)

        if macro_f1 >= best_macro_f1:
            best_macro_f1 = macro_f1
            best_targets, best_preds = all_targets.copy(), all_preds.copy()
            torch.save(model.state_dict(), os.path.join(RESULT_DIR, 'best_model_DSTG-Net.pth'))

    with open(os.path.join(RESULT_DIR, 'DSTG-Net_training_logs.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(terminal_logs))
    
    plot_single_curves(loss_history, val_acc_history, RESULT_DIR, "DSTG-Net")
    class_accs = visualize_single_model_results(best_targets, best_preds, "DSTG-Net")
    
    return {'loss': loss_history, 'val_acc': val_acc_history, 'class_accs': class_accs}

# =====================================================================
# 模块 D：综合实验对比图表绘制
# =====================================================================

def plot_contrast_charts(results_dict):
    print("\n正在生成综合对比图表...")
    models = list(results_dict.keys())
    epochs = range(1, GLOBAL_EPOCHS + 1)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # 1. 综合验证集准确率曲线
    plt.figure(figsize=(10, 6))
    for idx, model in enumerate(models):
        plt.plot(epochs, results_dict[model]['val_acc'], label=model, color=colors[idx], linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Comparison of Validation Accuracy Across Models')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(RESULT_DIR, 'Combined_Accuracy_Curve.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

    # 2. 综合训练损失曲线
    plt.figure(figsize=(10, 6))
    for idx, model in enumerate(models):
        plt.plot(epochs, results_dict[model]['loss'], label=model, color=colors[idx], linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Comparison of Training Loss Across Models')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(RESULT_DIR, 'Combined_Loss_Curve.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

    # 3. 12种手势准确率综合直方图
    plt.figure(figsize=(20, 8))
    x = np.arange(NUM_CLASSES)
    width = 0.15
    offsets = [-2*width, -width, 0, width, 2*width]
    
    for idx, model in enumerate(models):
        class_accs = results_dict[model]['class_accs']
        bars = plt.bar(x + offsets[idx], class_accs, width, label=model, color=colors[idx], edgecolor='black')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}', ha='center', va='bottom', fontsize=8, rotation=90)
            
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xlabel('Gesture Class', fontsize=12)
    plt.title('Gesture Recognition Accuracy Comparison Across Models', fontsize=16)
    plt.xticks(x, GESTURES, rotation=45, ha="right")
    plt.ylim(0, 115)  # 留出空间显示文字
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'Combined_Class_Accuracy_Histogram.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
    print(f"全部综合实验对比图表已保存至 {RESULT_DIR} 文件夹。")


# =====================================================================
# 模块 E：主程序执行枢纽
# =====================================================================
if __name__ == '__main__':
    print("=" * 60)
    print(" 多模型毫米波雷达手势识别 - 综合消融对比实验框架启动 ")
    print(f" 目标轮数: {GLOBAL_EPOCHS} Epochs | 数据集路径: {DATA_PATH} ")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        raise ValueError(f"未找到数据集路径 '{DATA_PATH}'，请先准备数据。")

    all_results = {}

    # ----------- 1. TF 序列模型 (CNN, VGGNet, ResNet) -----------
    X_tr_seq, y_tr_seq, X_va_seq, y_va_seq, X_te_seq, y_te_seq = load_tf_seq_data()
    seq_input_shape = (X_tr_seq.shape[1], X_tr_seq.shape[2])

    # 1.1 Single CNN
    model_cnn = build_cnn_model(seq_input_shape)
    all_results['only-CNN'] = train_evaluate_tf(model_cnn, 'only-CNN', X_tr_seq, y_tr_seq, X_va_seq, y_va_seq, X_te_seq, y_te_seq)

    # 1.2 VGGNet
    model_vgg = build_vggnet_model(seq_input_shape)
    all_results['VGGNet'] = train_evaluate_tf(model_vgg, 'VGGNet', X_tr_seq, y_tr_seq, X_va_seq, y_va_seq, X_te_seq, y_te_seq)

    # 1.3 ResNet
    model_resnet = build_resnet_model(seq_input_shape)
    all_results['ResNet'] = train_evaluate_tf(model_resnet, 'ResNet', X_tr_seq, y_tr_seq, X_va_seq, y_va_seq, X_te_seq, y_te_seq)

    # 释放 TF 会话资源，为后续模型腾出内存
    tf.keras.backend.clear_session()

    # ----------- 2. TF STGNet (带特征工程与 SMOTE) -----------
    X_tr_stg, y_tr_stg, X_va_stg, y_va_stg, X_te_stg, y_te_stg = load_tf_stgnet_data()
    stg_input_shape = (X_tr_stg.shape[1], X_tr_stg.shape[2])
    model_stg = build_stgnet_model(stg_input_shape)
    all_results['STGNet'] = train_evaluate_tf(model_stg, 'STGNet', X_tr_stg, y_tr_stg, X_va_stg, y_va_stg, X_te_stg, y_te_stg)
    tf.keras.backend.clear_session()

    # ----------- 3. PyTorch DSTG-Net (双流网络) -----------
    all_results['DSTG-Net'] = train_evaluate_pytorch()

    # ----------- 4. 汇总绘图 -----------
    plot_contrast_charts(all_results)
    
    print("\n恭喜！所有五个模型的训练、测试及全面对比可视化均已圆满完成！")