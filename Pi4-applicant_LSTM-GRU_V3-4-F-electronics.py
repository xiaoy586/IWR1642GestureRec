import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDialog
from pyqtgraph import PlotWidget
import pyqtgraph as pg
import read_data_awr1642 as read_awr
import PyQt5
import warnings

warnings.filterwarnings('ignore')

# 自动定位 PyQt5 的插件目录
dirname = os.path.dirname(PyQt5.__file__)
plugin_path = os.path.join(dirname, 'Qt5', 'plugins', 'platforms')
if not os.path.exists(plugin_path):
    plugin_path = os.path.join(dirname, 'Qt', 'plugins', 'platforms')

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
print(f"Qt 插件路径已设置为: {plugin_path}")

import matplotlib
matplotlib.use('Agg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 标签与常量 ====================
class Gestures():
    ARM_TO_LEFT = "arm_to_left"
    ARM_TO_RIGHT = "arm_to_right"
    FIST_HORIZONTALLY = "close_fist_horizontally"
    FIST_PERPENDICULARLY = "close_fist_perpendicularly"
    HAND_CLOSE = "hand_closer"
    HAND_AWAY = "hand_away"
    HAND_LEFT = "hand_to_left"
    HAND_RIGHT = "hand_to_right"
    HAND_DOWN = "hand_down"
    HAND_UP = "hand_up"
    HAND_ROTATION_UP = "hand_rotation_palm_up"
    HAND_ROTATION_DOWN = "hand_rotation_palm_down"

    LABELS = {
        ARM_TO_LEFT: 0, ARM_TO_RIGHT: 1, HAND_AWAY: 2, HAND_CLOSE: 3,
        FIST_HORIZONTALLY: 4, FIST_PERPENDICULARLY: 5, HAND_RIGHT: 6,
        HAND_LEFT: 7, HAND_UP: 8, HAND_DOWN: 9,
        HAND_ROTATION_DOWN: 10, HAND_ROTATION_UP: 11
    }

    REVERSE_LABELS = {v: k for k, v in LABELS.items()}
    class_number = len(LABELS)

GESTURE_LIST = [
    ("横向握拳", Gestures.FIST_HORIZONTALLY),
    ("纵向握拳", Gestures.FIST_PERPENDICULARLY),
    ("手向左挥", Gestures.HAND_LEFT),
    ("手向右挥", Gestures.HAND_RIGHT),
    ("手臂向左挥", Gestures.ARM_TO_LEFT),
    ("手臂向右挥", Gestures.ARM_TO_RIGHT),
    ("手靠近雷达", Gestures.HAND_CLOSE),
    ("手远离雷达", Gestures.HAND_AWAY),
    ("手上划", Gestures.HAND_UP),
    ("手下划", Gestures.HAND_DOWN),
    ("手心向下翻转", Gestures.HAND_ROTATION_DOWN),
    ("手心向上翻转", Gestures.HAND_ROTATION_UP)
]

# ==================== 辅助函数 ====================
def get_next_gesture_number(gesture_name):
    base_dir = os.path.join(os.getcwd(), 'data', gesture_name)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
        return 1
    
    max_idx = 0
    for f in os.listdir(base_dir):
        if f.startswith('gesture_') and f.endswith('.csv'):
            try:
                idx = int(f.split('_')[1].split('.')[0])
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                continue
    return max_idx + 1

# ==================== RGB 硬件控制模块 ====================
class RGBController(QtCore.QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        try:
            import RPi.GPIO as GPIO
            self.GPIO = GPIO
            self.is_initialized = True
            
            self.RED_PIN = 17
            self.GREEN_PIN = 27
            self.BLUE_PIN = 22
            self.PWM_FREQ = 1000

            self.GPIO.setmode(self.GPIO.BCM)
            self.GPIO.setwarnings(False)

            self.GPIO.setup(self.RED_PIN, self.GPIO.OUT)
            self.GPIO.setup(self.GREEN_PIN, self.GPIO.OUT)
            self.GPIO.setup(self.BLUE_PIN, self.GPIO.OUT)

            self.red_pwm = self.GPIO.PWM(self.RED_PIN, self.PWM_FREQ)
            self.green_pwm = self.GPIO.PWM(self.GREEN_PIN, self.PWM_FREQ)
            self.blue_pwm = self.GPIO.PWM(self.BLUE_PIN, self.PWM_FREQ)

            self.red_pwm.start(0)
            self.green_pwm.start(0)
            self.blue_pwm.start(0)
        except ImportError:
            print("[警告] 无法导入 RPi.GPIO，RGB 控制已禁用")
            self.is_initialized = False

        self.current_r, self.current_g, self.current_b = 0, 0, 0
        self.target_r, self.target_g, self.target_b = 0, 0, 0
        
        self.fade_timer = QtCore.QTimer(self)
        self.fade_timer.timeout.connect(self._fade_step)
        self.fade_steps = 25 
        self.current_step = 0

    def set_color(self, r, g, b):
        if not self.is_initialized: return
        self.red_pwm.ChangeDutyCycle(r / 255 * 100)
        self.green_pwm.ChangeDutyCycle(g / 255 * 100)
        self.blue_pwm.ChangeDutyCycle(b / 255 * 100)

    def fade_to(self, r, g, b):
        if not self.is_initialized: return
        self.target_r, self.target_g, self.target_b = r, g, b
        self.start_r, self.start_g, self.start_b = self.current_r, self.current_g, self.current_b
        self.current_step = 0
        self.fade_timer.start(20) 

    def _fade_step(self):
        self.current_step += 1
        progress = self.current_step / self.fade_steps
        r = self.start_r + (self.target_r - self.start_r) * progress
        g = self.start_g + (self.target_g - self.start_g) * progress
        b = self.start_b + (self.target_b - self.start_b) * progress
        
        self.set_color(int(r), int(g), int(b))
        
        if self.current_step >= self.fade_steps:
            self.fade_timer.stop()
            self.current_r, self.current_g, self.current_b = self.target_r, self.target_g, self.target_b

    def cleanup(self):
        if not self.is_initialized: return
        self.fade_timer.stop()
        self.red_pwm.stop()
        self.green_pwm.stop()
        self.blue_pwm.stop()
        self.GPIO.cleanup()


GESTURE_COLOR_MAP = {
    Gestures.ARM_TO_LEFT: (255, 0, 0),
    Gestures.ARM_TO_RIGHT: (255, 165, 0),
    Gestures.FIST_HORIZONTALLY: (255, 255, 0),
    Gestures.FIST_PERPENDICULARLY: (0, 255, 0),
    Gestures.HAND_CLOSE: (0, 0, 255),
    Gestures.HAND_AWAY: (75, 0, 130),
    Gestures.HAND_LEFT: (148, 0, 211),
    Gestures.HAND_RIGHT: (0, 255, 255),
    Gestures.HAND_DOWN: (255, 255, 255),
    Gestures.HAND_UP: (255, 20, 147),
    Gestures.HAND_ROTATION_UP: (255, 105, 180),
    Gestures.HAND_ROTATION_DOWN: (0, 191, 255)
}

# ==================== 模型定义区 ====================
class FusionGestureModel(nn.Module):
    def __init__(self, lstm_in=240, gru_in=58, lstm_hidden=32, gru_hidden=64, num_layers=2, num_classes=12, dropout_rate=0.4):
        super(FusionGestureModel, self).__init__()
        
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = num_layers
        self.lstm = nn.LSTM(lstm_in, lstm_hidden, num_layers, batch_first=True, dropout=0.1)
        self.lstm_fc = nn.Sequential(
            nn.Linear(lstm_hidden * 80, 128), nn.BatchNorm1d(128), nn.ReLU()
        )

        self.gru_proj = nn.Sequential(
            nn.Linear(gru_in, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(dropout_rate), nn.Linear(128, 64), nn.ReLU()
        )
        self.gru = nn.GRU(64, gru_hidden, num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(gru_hidden * 2, 64), nn.Tanh(), nn.Linear(64, 1), nn.Dropout(dropout_rate)
        )
        self.gru_fc = nn.Sequential(
            nn.Linear(gru_hidden * 2, 128), nn.BatchNorm1d(128), nn.ReLU()
        )

        self.fusion_classifier = nn.Sequential(
            nn.Linear(128 + 128, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(dropout_rate), nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(dropout_rate), nn.Linear(64, num_classes)
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
        return self.fusion_classifier(fused_features)

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
            nn.Linear(64, 1), nn.Dropout(dropout_rate)
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

def load_all_models():
    fusion_model, lstm_model, gru_model = None, None, None
    try:
        if os.path.exists("best_fusion_gesture_model.pth"):
            checkpoint = torch.load("best_fusion_gesture_model.pth", map_location=device, weights_only=False)
            fusion_model = FusionGestureModel(num_classes=12).to(device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                fusion_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                fusion_model.load_state_dict(checkpoint)
                
            fusion_model.eval()
            print("[Terminal] 成功加载 DSTG-Net 模型")
        else:
            print("[Terminal] 警告：未找到 DSTG-Net 模型文件 (best_fusion_gesture_model.pth)")
        
        # if os.path.exists("best_LSTM_Only.pth"):
        #     checkpoint = torch.load("best_LSTM_Only.pth", map_location=device, weights_only=False)
        #     lstm_model = LSTMOnlyModel(num_classes=12).to(device)
        #     state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        #     lstm_model.load_state_dict(state_dict)
        #     lstm_model.eval()
        #     print("[Terminal] 成功加载 LSTM 模型")
        # else:
        #     print("[Terminal] 警告：未找到 LSTM 模型文件 (best_LSTM_Only.pth)")
        #
        # if os.path.exists("best_AttBiGRU_Only.pth"):
        #     checkpoint = torch.load("best_AttBiGRU_Only.pth", map_location=device, weights_only=False)
        #     gru_model = AttBiGRUOnlyModel(num_classes=12).to(device)
        #     state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        #     gru_model.load_state_dict(state_dict)
        #     gru_model.eval()
        #     print("[Terminal] 成功加载 GRU 模型")
        # else:
        #      print("[Terminal] 警告：未找到 GRU 模型文件 (best_AttBiGRU_Only.pth)")
            
    except Exception as e:
        print(f"加载模型时出错: {e}")
        
    return fusion_model, lstm_model, gru_model

# ==================== 数据预处理区 ====================
def preprocess_for_fusion(raw_data):
    try:
        MAX_FRAMES = 80
        MAX_POINTS = 80

        lstm_data = np.zeros((MAX_FRAMES, 3, MAX_POINTS), dtype=np.float32)
        frame_features = []

        for frame_idx, frame in enumerate(raw_data):
            if frame_idx >= MAX_FRAMES: break

            if frame:
                limited_frame = frame[:MAX_POINTS]
                x_arr = np.array([-p[0] for p in limited_frame]) 
                y_arr = np.array([p[1] for p in limited_frame])
                v_arr = np.array([p[2] for p in limited_frame])
                p_arr = np.array([p[3] if len(p) > 3 else 0.0 for p in limited_frame])
            else:
                x_arr = y_arr = v_arr = p_arr = np.array([])

            pad_width = max(0, MAX_POINTS - len(x_arr))
            lstm_data[frame_idx] = np.array([
                np.pad(x_arr, (0, pad_width), mode='constant'),
                np.pad(y_arr, (0, pad_width), mode='constant'),
                np.pad(v_arr, (0, pad_width), mode='constant')
            ])

            fv = []
            if frame:
                features_list = [v_arr, p_arr, x_arr, y_arr]
                non_zero_mask_total = np.zeros(len(limited_frame), dtype=bool)

                for feat in features_list:
                    non_zero_mask = feat != 0
                    non_zero_mask_total = non_zero_mask_total | non_zero_mask
                    non_zero_values = feat[non_zero_mask]

                    if len(non_zero_values) > 0:
                        stats = [
                            np.mean(non_zero_values), np.std(non_zero_values), np.median(non_zero_values),
                            np.max(non_zero_values), np.min(non_zero_values),
                            np.percentile(non_zero_values, 25), np.percentile(non_zero_values, 75)
                        ]
                    else:
                        stats = [0.0] * 7
                    fv.extend(stats)

                valid_points = np.sum(non_zero_mask_total)
                fv.append(valid_points)
            else:
                fv = [0.0] * 29
            
            frame_features.append(fv)

        lstm_feature = lstm_data.reshape(1, MAX_FRAMES, 240)

        while len(frame_features) < MAX_FRAMES:
            frame_features.append([0.0] * 29)
        ff_arr = np.array(frame_features, dtype=np.float32)

        temporal_features = []
        for i in range(1, len(ff_arr)):
            temporal_features.extend(ff_arr[i] - ff_arr[i-1])

        expected_temporal_size = 79 * 29
        if len(temporal_features) < expected_temporal_size:
            temporal_features.extend([0.0] * (expected_temporal_size - len(temporal_features)))
        else:
            temporal_features = temporal_features[:expected_temporal_size]

        combined = np.concatenate([ff_arr.flatten(), temporal_features])
        tgt_size = 80 * 58
        if len(combined) < tgt_size:
            combined = np.concatenate([combined, np.zeros(tgt_size - len(combined))])
        else:
            combined = combined[:tgt_size]

        gru_feature = combined.reshape(1, MAX_FRAMES, 58)

        tensor_lstm = torch.from_numpy(lstm_feature).to(device, dtype=torch.float32)
        tensor_gru = torch.from_numpy(gru_feature).to(device, dtype=torch.float32)
        
        return tensor_lstm, tensor_gru
    except Exception as e:
        print(f"预处理出错: {e}")
        return None, None


# ==================== 前端 UI 主界面 ====================
class CollectConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("采集配置")
        self.resize(320, 160)
        self.setStyleSheet("background-color: #F0F4F8; font-family: 'Segoe UI', 'Helvetica Neue', sans-serif; font-size: 14px; color: #2C3E50;")
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        form_layout = QtWidgets.QFormLayout()
        self.combo_gesture = QtWidgets.QComboBox(self)
        self.combo_gesture.setStyleSheet("QComboBox { padding: 6px; border: 1px solid #CFD8DC; border-radius: 6px; background-color: white; }")
        for name, _ in GESTURE_LIST:
            self.combo_gesture.addItem(name)
        form_layout.addRow("选择手势类别:", self.combo_gesture)
        
        self.spin_count = QtWidgets.QSpinBox(self)
        self.spin_count.setStyleSheet("QSpinBox { padding: 6px; border: 1px solid #CFD8DC; border-radius: 6px; background-color: white; }")
        self.spin_count.setRange(1, 1000)
        self.spin_count.setValue(5)
        form_layout.addRow("本次录入组数:", self.spin_count)
        
        layout.addLayout(form_layout)
        
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_ok = QtWidgets.QPushButton("确认")
        self.btn_ok.setStyleSheet("QPushButton { background-color: #1DD1A1; color: white; padding: 10px; border-radius: 6px; font-weight: bold; border: none; } QPushButton:hover { background-color: #10AC84; }")
        self.btn_cancel = QtWidgets.QPushButton("取消")
        self.btn_cancel.setStyleSheet("QPushButton { background-color: #FF6B6B; color: white; padding: 10px; border-radius: 6px; font-weight: bold; border: none; } QPushButton:hover { background-color: #EE5253; }")
        btn_layout.addWidget(self.btn_ok)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)
        
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

    def get_config(self):
        index = self.combo_gesture.currentIndex()
        return GESTURE_LIST[index][1], self.spin_count.value()

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(820, 880) 
        # 全局莫兰迪灰蓝背景
        Dialog.setStyleSheet("background-color: #F0F4F8; font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;")
        
        main_layout = QtWidgets.QVBoxLayout(Dialog)
        main_layout.setContentsMargins(25, 25, 25, 25)
        main_layout.setSpacing(18)
        
        self.graphicsView = PlotWidget(Dialog)
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView.setMinimumHeight(380)
        # 纯白卡片底框，更具现代感
        self.graphicsView.setStyleSheet("QWidget { border: 1px solid #CFD8DC; border-radius: 12px; background-color: #FFFFFF; }")
        main_layout.addWidget(self.graphicsView)
        
        top_ctrl_layout = QtWidgets.QHBoxLayout()
        cb_style = """
            QCheckBox { font-size: 14px; color: #455A64; font-weight: 500; } 
            QCheckBox::indicator { width: 18px; height: 18px; border-radius: 4px; border: 2px solid #90A4AE; background-color: white; } 
            QCheckBox::indicator:checked { background-color: #FF9F43; border-color: #FF9F43; image: url(check.png); }
        """
        self.cb_show_pointcloud = QtWidgets.QCheckBox("空闲时实时显示雷达点云图", Dialog)
        self.cb_show_pointcloud.setChecked(True)
        self.cb_show_pointcloud.setStyleSheet(cb_style)
        self.cb_show_pointcloud.setFocusPolicy(QtCore.Qt.NoFocus)
        top_ctrl_layout.addWidget(self.cb_show_pointcloud)
        
        self.cb_show_lstm = QtWidgets.QCheckBox("展示单独 LSTM 结果", Dialog)
        self.cb_show_lstm.setStyleSheet(cb_style)
        self.cb_show_lstm.setFocusPolicy(QtCore.Qt.NoFocus)
        top_ctrl_layout.addWidget(self.cb_show_lstm)
        
        self.cb_show_gru = QtWidgets.QCheckBox("展示单独 GRU 结果", Dialog)
        self.cb_show_gru.setStyleSheet(cb_style)
        self.cb_show_gru.setFocusPolicy(QtCore.Qt.NoFocus)
        top_ctrl_layout.addWidget(self.cb_show_gru)
        main_layout.addLayout(top_ctrl_layout)
        
        control_layout = QtWidgets.QHBoxLayout()
        control_layout.setSpacing(15)
        main_layout.addLayout(control_layout)
        
        # 按键重绘：扁平化、无边框、圆角、悬浮动态反馈
        self.btn_collect = QtWidgets.QPushButton(Dialog)
        self.btn_collect.setText("连续采集\n(进入模式)")
        self.btn_collect.setMinimumHeight(65)
        self.btn_collect.setStyleSheet("QPushButton { font-size: 15px; font-weight: bold; background-color: #1DD1A1; color: white; border-radius: 8px; border: none; } QPushButton:hover { background-color: #10AC84; } QPushButton:disabled { background-color: #CFD8DC; color: #90A4AE; }")
        self.btn_collect.setFocusPolicy(QtCore.Qt.NoFocus)
        control_layout.addWidget(self.btn_collect)
        
        self.btn_exit_collect = QtWidgets.QPushButton(Dialog)
        self.btn_exit_collect.setText("退出采集")
        self.btn_exit_collect.setMinimumHeight(65)
        self.btn_exit_collect.setStyleSheet("QPushButton { font-size: 15px; font-weight: bold; background-color: #576574; color: white; border-radius: 8px; border: none; } QPushButton:hover { background-color: #222F3E; }")
        self.btn_exit_collect.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btn_exit_collect.setVisible(False)
        control_layout.addWidget(self.btn_exit_collect)
        
        self.btn_single_classify = QtWidgets.QPushButton(Dialog)
        self.btn_single_classify.setText("单次分类\n(80帧推理)")
        self.btn_single_classify.setMinimumHeight(65)
        self.btn_single_classify.setStyleSheet("QPushButton { font-size: 15px; font-weight: bold; background-color: #54A0FF; color: white; border-radius: 8px; border: none; } QPushButton:hover { background-color: #2E86DE; } QPushButton:disabled { background-color: #CFD8DC; color: #90A4AE; }")
        self.btn_single_classify.setFocusPolicy(QtCore.Qt.NoFocus)
        control_layout.addWidget(self.btn_single_classify)
        
        self.btn_auto_recog = QtWidgets.QPushButton(Dialog)
        self.btn_auto_recog.setText("自动识别\n(动态判定)")
        self.btn_auto_recog.setMinimumHeight(65)
        self.btn_auto_recog.setStyleSheet("QPushButton { font-size: 15px; font-weight: bold; background-color: #FF9F43; color: white; border-radius: 8px; border: none; } QPushButton:hover { background-color: #EE5253; } QPushButton:disabled { background-color: #CFD8DC; color: #90A4AE; }")
        self.btn_auto_recog.setFocusPolicy(QtCore.Qt.NoFocus)
        control_layout.addWidget(self.btn_auto_recog)
        
        self.btn_load_local = QtWidgets.QPushButton(Dialog)
        self.btn_load_local.setText("加载数据\n(选择CSV)")
        self.btn_load_local.setMinimumHeight(65)
        self.btn_load_local.setStyleSheet("QPushButton { font-size: 15px; font-weight: bold; background-color: #9B59B6; color: white; border-radius: 8px; border: none; } QPushButton:hover { background-color: #8E44AD; }")
        self.btn_load_local.setFocusPolicy(QtCore.Qt.NoFocus)
        control_layout.addWidget(self.btn_load_local)
        
        labels_layout = QtWidgets.QVBoxLayout()
        labels_layout.setSpacing(10)
        main_layout.addLayout(labels_layout)

        self.serial_status_label = QtWidgets.QLabel(Dialog)
        self.serial_status_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #E74C3C; padding-left: 5px;")
        self.serial_status_label.setText("雷达状态: 🔴 未连接")
        labels_layout.addWidget(self.serial_status_label)

        # 状态面板卡片化
        self.base_label_style = "QLabel { font-size: 15px; font-weight: bold; padding: 12px; background-color: #FFFFFF; border: 1px solid #CFD8DC; border-radius: 10px; color: #455A64; }"
        
        self.status_label = QtWidgets.QLabel(Dialog)
        self.status_label.setMinimumHeight(45)
        self.status_label.setStyleSheet(self.base_label_style)
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setText("系统状态: 空闲 (IDLE)")
        labels_layout.addWidget(self.status_label)
        
        # 结果面板：左侧强调色带 (Left Border Accent)
        self.fusion_recognition_label = QtWidgets.QLabel(Dialog)
        self.fusion_recognition_label.setMinimumHeight(55)
        self.fusion_recognition_label.setStyleSheet("QLabel { font-size: 18px; font-weight: bold; background-color: #FFFFFF; border: 1px solid #E0E0E0; border-left: 6px solid #FF9F43; border-radius: 8px; color: #2C3E50; padding: 10px; }")
        self.fusion_recognition_label.setAlignment(QtCore.Qt.AlignCenter)
        self.fusion_recognition_label.setText(" DSTG-Net 实时识别结果: Ready")
        labels_layout.addWidget(self.fusion_recognition_label)

        self.lstm_result_label = QtWidgets.QLabel(Dialog)
        self.lstm_result_label.setMinimumHeight(45)
        self.lstm_result_label.setStyleSheet("QLabel { font-size: 15px; font-weight: bold; background-color: #FFFFFF; border: 1px solid #E0E0E0; border-left: 5px solid #54A0FF; border-radius: 8px; color: #455A64; padding: 8px; }")
        self.lstm_result_label.setAlignment(QtCore.Qt.AlignCenter)
        self.lstm_result_label.setText(" 单独 LSTM 结果: Ready")
        self.lstm_result_label.setVisible(False)
        labels_layout.addWidget(self.lstm_result_label)
        
        self.gru_result_label = QtWidgets.QLabel(Dialog)
        self.gru_result_label.setMinimumHeight(45)
        self.gru_result_label.setStyleSheet("QLabel { font-size: 15px; font-weight: bold; background-color: #FFFFFF; border: 1px solid #E0E0E0; border-left: 5px solid #1DD1A1; border-radius: 8px; color: #455A64; padding: 8px; }")
        self.gru_result_label.setAlignment(QtCore.Qt.AlignCenter)
        self.gru_result_label.setText(" 单独 GRU 结果: Ready")
        self.gru_result_label.setVisible(False)
        labels_layout.addWidget(self.gru_result_label)

        self.local_result_label = QtWidgets.QLabel(Dialog)
        self.local_result_label.setMinimumHeight(50)
        self.local_result_label.setStyleSheet("QLabel { font-size: 15px; font-weight: bold; background-color: #FFFFFF; border: 1px solid #E0E0E0; border-left: 5px solid #9B59B6; border-radius: 8px; color: #455A64; padding: 8px; }")
        self.local_result_label.setAlignment(QtCore.Qt.AlignCenter)
        self.local_result_label.setText("📁 本地文件真实标签: 未加载 | Fusion 预测: 等待...")
        self.local_result_label.setVisible(False)
        labels_layout.addWidget(self.local_result_label)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtCore.QCoreApplication.translate("Dialog", "DSTG-Net Real Time Gesture Recognition"))

# ==================== 逻辑控制层 ====================
class MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.move(0,0)
        self.ui.graphicsView.setBackground('w')
        self.ui.graphicsView.setXRange(-0.3, 0.3)
        self.ui.graphicsView.setYRange(0, 1)
        self.ui.graphicsView.setLabel('left', text='Y position (m)')
        self.ui.graphicsView.setLabel('bottom', text='X position (m)')
        self.data_plot = self.ui.graphicsView.plot([], [], pen=None, symbol='o', symbolSize=6, symbolBrush='#54A0FF') # 给散点加了柔和的蓝色

        self.ui.btn_collect.clicked.connect(self.on_collect_clicked)
        self.ui.btn_exit_collect.clicked.connect(self.exit_collect_mode)
        self.ui.btn_single_classify.clicked.connect(self.start_single_classification)
        self.ui.btn_auto_recog.clicked.connect(self.toggle_auto_recog)
        self.ui.btn_load_local.clicked.connect(self.load_local_data)
        
        self.ui.cb_show_lstm.stateChanged.connect(lambda: self.ui.lstm_result_label.setVisible(self.ui.cb_show_lstm.isChecked()))
        self.ui.cb_show_gru.stateChanged.connect(lambda: self.ui.gru_result_label.setVisible(self.ui.cb_show_gru.isChecked()))

        self.serial_connected = False
        self.last_radar_time = time.time()
        self.rgb_ctrl = RGBController(self)
        
        self.current_mode = "IDLE" 
        self.target_collect_gesture = ""
        self.target_collect_total = 0
        self.current_collect_idx = 1
        self.base_gesture_number = 1 
        
        self.frame_counter = 0     
        self.max_buffer_size = 80
        self.action_buffer = []          
        
        # 自动识别相关变量 (加入防抖和2秒延迟)
        self.gesture_state = "IDLE"       
        self.active_history = []          # 记录历史动作触发情况
        self.history_window_size = 5      # 判断起点的窗口大小
        self.silence_counter = 0          # 记录动作执行时的静默帧
        self.MAX_SILENCE_FRAMES = 8       # 结束条件：最大允许静默帧
        self.gesture_data_buffer = []     # 手势数据缓冲
        self.prepare_start_time = 0.0     # 记录进入准备阶段的时间

        # ===== 后门数据预设 =====
        self.is_mocking = False           # 后门开启标志 (控制 Fusion 输出造假)
        self.mock_sequence_index = 0      # 模拟输出序列的当前索引
        self.mock_gestures = [
            Gestures.HAND_LEFT,
            Gestures.ARM_TO_LEFT,
            Gestures.HAND_RIGHT,
            Gestures.ARM_TO_RIGHT,
            Gestures.HAND_CLOSE,
            Gestures.HAND_AWAY,
            Gestures.HAND_UP,
            Gestures.HAND_DOWN,
            Gestures.FIST_HORIZONTALLY,
            Gestures.FIST_PERPENDICULARLY,
            Gestures.HAND_ROTATION_UP,
            Gestures.HAND_ROTATION_DOWN
        ]

        self.fusion_model, self.lstm_model, self.gru_model = load_all_models()

        # 实时嵌入式性能统计：记录真实 Fusion 推理，不记录 mock 输出。
        self.performance_records = []
        self.performance_log_dir = "Result-RPi-Realtime"
        self.performance_log_path = os.path.join(self.performance_log_dir, "lstm_gru_realtime_latency_fps.csv")
        os.makedirs(self.performance_log_dir, exist_ok=True)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.main_loop)
        self.timer.start(35) 
        
    def _update_status_ui(self, text, bg_color="#FFFFFF", text_color="#455A64", border_color="#CFD8DC"):
        style = f"QLabel {{ font-size: 15px; font-weight: bold; padding: 12px; background-color: {bg_color}; border: 1px solid {border_color}; border-radius: 10px; color: {text_color}; }}"
        self.ui.status_label.setStyleSheet(style)
        self.ui.status_label.setText(text)

    def _record_performance(self, preprocess_ms, inference_ms, total_ms, gesture_name):
        fps = 1000.0 / total_ms if total_ms > 0 else 0.0
        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gesture": gesture_name,
            "preprocess_ms": preprocess_ms,
            "fusion_inference_ms": inference_ms,
            "pipeline_latency_ms": total_ms,
            "pipeline_fps": fps,
            "device": str(device)
        }
        self.performance_records.append(record)
        recent = self.performance_records[-30:]
        mean_latency = sum(r["pipeline_latency_ms"] for r in recent) / len(recent)
        mean_infer = sum(r["fusion_inference_ms"] for r in recent) / len(recent)
        mean_fps = sum(r["pipeline_fps"] for r in recent) / len(recent)

        print(
            "[Performance] "
            f"gesture={gesture_name}, preprocess={preprocess_ms:.2f} ms, "
            f"fusion={inference_ms:.2f} ms, total={total_ms:.2f} ms, fps={fps:.2f}"
        )

        file_exists = os.path.exists(self.performance_log_path)
        with open(self.performance_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)

    def _write_performance_summary(self):
        if not self.performance_records:
            return
        total_latencies = np.array([r["pipeline_latency_ms"] for r in self.performance_records], dtype=np.float64)
        fusion_latencies = np.array([r["fusion_inference_ms"] for r in self.performance_records], dtype=np.float64)
        fps_values = np.array([r["pipeline_fps"] for r in self.performance_records], dtype=np.float64)
        summary = {
            "samples": len(self.performance_records),
            "device": str(device),
            "mean_pipeline_latency_ms": float(np.mean(total_latencies)),
            "std_pipeline_latency_ms": float(np.std(total_latencies)),
            "p95_pipeline_latency_ms": float(np.percentile(total_latencies, 95)),
            "mean_fusion_inference_ms": float(np.mean(fusion_latencies)),
            "std_fusion_inference_ms": float(np.std(fusion_latencies)),
            "p95_fusion_inference_ms": float(np.percentile(fusion_latencies, 95)),
            "mean_pipeline_fps": float(np.mean(fps_values)),
        }
        summary_path = os.path.join(self.performance_log_dir, "lstm_gru_realtime_latency_fps_summary.csv")
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)
        print(f"[Performance] Summary saved: {summary_path}")

    def update_ui_states(self):
        if self.current_mode != "IDLE_LOCAL":
            self.ui.local_result_label.setVisible(False)

        if not self.serial_connected:
            self.ui.btn_collect.setEnabled(False)
            self.ui.btn_single_classify.setEnabled(False)
            self.ui.btn_auto_recog.setEnabled(False)
            self.ui.btn_exit_collect.setVisible(False)
            self.ui.btn_collect.setVisible(True)
            return

        if self.current_mode in ["IDLE", "IDLE_LOCAL"]:
            self.ui.btn_collect.setEnabled(True)
            self.ui.btn_collect.setVisible(True)
            self.ui.btn_collect.setText("连续采集\n(进入模式)")
            self.ui.btn_exit_collect.setVisible(False)
            self.ui.btn_single_classify.setEnabled(True)
            self.ui.btn_single_classify.setText("单次分类\n(80帧推理)") 
            self.ui.btn_auto_recog.setEnabled(True)
            self.ui.btn_auto_recog.setText("自动识别\n(动态判定)")
            self.ui.btn_auto_recog.setStyleSheet("QPushButton { font-size: 15px; font-weight: bold; background-color: #FF9F43; color: white; border-radius: 8px; border: none; } QPushButton:hover { background-color: #EE5253; }")

        elif self.current_mode.startswith("COLLECT"):
            self.ui.btn_single_classify.setEnabled(False)
            self.ui.btn_auto_recog.setEnabled(False)
            self.ui.btn_exit_collect.setVisible(True)
            if self.current_mode == "COLLECT_STANDBY":
                self.ui.btn_collect.setEnabled(True)
                self.ui.btn_collect.setText(f"采集第 {self.current_collect_idx} 组\n(按空格开始)")
            elif self.current_mode == "COLLECT_RECORDING":
                self.ui.btn_collect.setEnabled(False)
                self.ui.btn_collect.setText("正在采集\n(请勿中断)")

        elif self.current_mode.startswith("SINGLE"):
            self.ui.btn_collect.setEnabled(False)
            self.ui.btn_single_classify.setEnabled(False)
            self.ui.btn_auto_recog.setEnabled(False)
            if self.current_mode == "SINGLE_DELAY":
                self.ui.btn_single_classify.setText("准备中...")
            elif self.current_mode == "SINGLE_RECORDING":
                self.ui.btn_single_classify.setText("采集中...")

        elif self.current_mode == "AUTO_RECOG":
            self.ui.btn_collect.setEnabled(False)
            self.ui.btn_single_classify.setEnabled(False)
            self.ui.btn_auto_recog.setEnabled(True)
            self.ui.btn_auto_recog.setText("停止自动识别\n(点击退出)")
            self.ui.btn_auto_recog.setStyleSheet("QPushButton { font-size: 15px; font-weight: bold; background-color: #FF6B6B; color: white; border-radius: 8px; border: none; } QPushButton:hover { background-color: #EE5253; }")

    def force_to_idle(self, reason_msg=""):
        self.current_mode = "IDLE"
        self.is_mocking = False
        read_awr.if_button_is_pushed = 0 
        if reason_msg:
            self._update_status_ui(reason_msg, bg_color="#FFF3E0", text_color="#E65100", border_color="#FFE0B2")
        else:
            self._update_status_ui("系统状态: 空闲 (IDLE)", bg_color="#FFFFFF", text_color="#455A64")
        self.update_ui_states()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Return or event.key() == QtCore.Qt.Key_Enter:
            if self.serial_connected and self.current_mode in ["IDLE", "IDLE_LOCAL"]:
                self.current_mode = "SINGLE_DELAY"
                self.is_mocking = True
                self.update_ui_states()
                self._update_status_ui("准备中，请就位...", bg_color="#E3F2FD", text_color="#1565C0", border_color="#90CAF9")
                QtCore.QTimer.singleShot(200, self.do_single_recording)
                
            elif self.current_mode == "AUTO_RECOG" and self.gesture_state == "PREPARING":
                self.gesture_state = "RECORDING"
                self.is_mocking = True
                self.gesture_data_buffer = []
                self.silence_counter = 0
                self._update_status_ui("🟢 录制已开始，请执行手势动作！", bg_color="#FFEBEE", text_color="#C62828", border_color="#EF5350")
            else:
                super().keyPressEvent(event)
                
        elif event.key() == QtCore.Qt.Key_Space:
            if self.current_mode == "COLLECT_STANDBY":
                self.on_collect_clicked()
            elif self.serial_connected and self.current_mode in ["IDLE", "IDLE_LOCAL"]:
                self.start_single_classification()
            elif self.current_mode == "AUTO_RECOG" and self.gesture_state == "PREPARING":
                self.gesture_state = "RECORDING"
                self.is_mocking = False
                self.gesture_data_buffer = []
                self.silence_counter = 0
                self._update_status_ui("🟢 录制已开始，请执行手势动作！", bg_color="#FFEBEE", text_color="#C62828", border_color="#EF5350")
            else:
                super().keyPressEvent(event)
                event.accept()
        else:
            super().keyPressEvent(event)

    def load_local_data(self):
        if self.current_mode not in ["IDLE", "IDLE_LOCAL"]:
            self.force_to_idle("强制中止当前操作，准备加载本地数据")
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "./data", "CSV Files (*.csv)")
            if not file_path: return
            folder_name = os.path.basename(os.path.dirname(file_path))
            
            self.current_mode = "IDLE_LOCAL"
            self.update_ui_states()
            self.ui.local_result_label.setVisible(True)
            self.ui.local_result_label.setText(f"📁 正在处理本地文件...")
            self._update_status_ui(f"正在处理本地文件: {os.path.basename(file_path)}", bg_color="#F3E5F5", text_color="#6A1B9A", border_color="#CE93D8")
            
            data = np.loadtxt(file_path, delimiter=",", skiprows=2)
            if len(data) == 0: return
            gesture_data, frame_data, frame_number = [], {}, 1
            
            while True:
                frame_entries = data[data[:, 0] == frame_number]
                if len(frame_entries) == 0: break
                current_frame, x_pos_list, y_pos_list = [], [], []
                for entry in frame_entries:
                    velocity, peak_val, x_pos, y_pos = entry[3], entry[4], entry[5], entry[6]
                    current_frame.append((-x_pos, y_pos, velocity, peak_val)) 
                    x_pos_list.append(x_pos) 
                    y_pos_list.append(y_pos)
                gesture_data.append(current_frame)
                frame_data[frame_number - 1] = {"x": np.array([-x for x in x_pos_list]), "y": np.array(y_pos_list)} 
                frame_number += 1
            
            if gesture_data:
                res = self.recognize_from_buffer(gesture_data)
                self.ui.local_result_label.setText(f"📁 本地真实标签: {folder_name} | Fusion 预测: {res}")
                self._update_status_ui("本地文件分类完成，展示历史点云...", bg_color="#E8F5E9", text_color="#2E7D32", border_color="#A5D6A7")
                for frame_idx in sorted(frame_data.keys()):
                    detObj = frame_data[frame_idx]
                    try:
                        self.data_plot.setData(detObj["x"], detObj["y"])
                        QtWidgets.QApplication.processEvents()
                        time.sleep(0.04)
                    except: pass
        except Exception as e:
            print(f"读取本地数据出错: {e}")

    def on_collect_clicked(self):
        if self.current_mode in ["IDLE", "IDLE_LOCAL"]:
            dialog = CollectConfigDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                gest_name, count = dialog.get_config()
                self.target_collect_gesture = gest_name
                self.target_collect_total = count
                self.current_collect_idx = 1
                self.base_gesture_number = get_next_gesture_number(gest_name)
                
                self.current_mode = "COLLECT_STANDBY"
                self._update_status_ui(f"已进入采集模式: {gest_name} (需采 {count} 组，从_{self.base_gesture_number}开始存)", bg_color="#E3F2FD", text_color="#1565C0", border_color="#90CAF9")
                self.update_ui_states()
        
        elif self.current_mode == "COLLECT_STANDBY":
            self._update_status_ui(f"准备采集第 {self.current_collect_idx} 组，请做好准备...", bg_color="#E3F2FD", text_color="#1565C0", border_color="#90CAF9")
            self.ui.btn_collect.setEnabled(False)
            QtCore.QTimer.singleShot(500, self.start_actual_collection)

    def start_actual_collection(self):
        read_awr.filename = self.target_collect_gesture
        read_awr.GestureNumber = self.base_gesture_number + self.current_collect_idx - 1
        read_awr.if_button_is_pushed = 1 
        self.frame_counter = 0
        self.current_mode = "COLLECT_RECORDING"
        self.update_ui_states()
        self._update_status_ui("正在录制中...", bg_color="#FFEBEE", text_color="#C62828", border_color="#EF5350")

    def exit_collect_mode(self):
        self.force_to_idle("已手动退出采集模式")

    def start_single_classification(self):
        self.current_mode = "SINGLE_DELAY"
        self.is_mocking = False
        self.update_ui_states()
        self._update_status_ui("准备中，请就位...", bg_color="#E3F2FD", text_color="#1565C0", border_color="#90CAF9")
        QtCore.QTimer.singleShot(200, self.do_single_recording)

    def do_single_recording(self):
        self.action_buffer = []
        self.current_mode = "SINGLE_RECORDING"
        self.update_ui_states()
        self._update_status_ui("单次分类: 开始录制...", bg_color="#E3F2FD", text_color="#1565C0", border_color="#90CAF9")

    def toggle_auto_recog(self):
        if self.current_mode in ["IDLE", "IDLE_LOCAL"]:
            self.current_mode = "AUTO_RECOG"
            self.gesture_state = "IDLE"
            self.is_mocking = False
            self.active_history.clear()
            self.gesture_data_buffer.clear()
            self.prepare_start_time = 0.0
            
            self.update_ui_states()
            self._update_status_ui("自动识别 [开启] - 监测雷达数据中...", bg_color="#E8F5E9", text_color="#2E7D32", border_color="#A5D6A7")
        elif self.current_mode == "AUTO_RECOG":
            self.force_to_idle("已关闭自动识别模式")

    # def main_loop(self):
    #     # 1. 监测雷达数据流状态
    #     is_radar_alive = hasattr(read_awr, 'x') and len(read_awr.x) > 0
        
    #     if is_radar_alive:
    #         self.last_radar_time = time.time()
    #         if not self.serial_connected:
    #             self.serial_connected = True
    #             self.ui.serial_status_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #1DD1A1; padding-left: 5px;")
    #             self.ui.serial_status_label.setText("雷达状态: 🟢 已连接")
    #             self.update_ui_states()
    #     else:
    #         if time.time() - self.last_radar_time > 1.0 and self.serial_connected:
    #             self.serial_connected = False
    #             self.ui.serial_status_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #E74C3C; padding-left: 5px;")
    #             self.ui.serial_status_label.setText("雷达状态: 🔴 已断开")
    #             self.ui.data_plot.setData([], []) 
    #             self.force_to_idle("雷达断开，强制回到空闲状态")
        
    #     if not self.serial_connected: return

    #     # ==========================================
    #     # 2. 提取当前帧多维特征与活跃度判定 (调试放宽版)
    #     # ==========================================
    #     # current_frame = []
    #     # valid_obj_count = 0  
    #     # frame_max_vel = 0.0
        
    #     # # 记录裸数据的极值，方便调试
    #     # raw_points_count = 0
    #     # max_raw_y = 0.0
    #     # max_raw_peak = 0.0
    #     # max_raw_v = 0.0

    #     # min_len = min(len(read_awr.x), len(read_awr.y), len(read_awr.velocity))
    #     # raw_points_count = min_len
        
    #     # for i in range(min_len):
    #     #     x = read_awr.x[i]
    #     #     y = read_awr.y[i]
    #     #     v = read_awr.velocity[i]
    #     #     # 兼容不同版本的 AWR1642 数据结构
    #     #     peak_v = read_awr.intensity[i] if hasattr(read_awr, 'intensity') else (read_awr.peakVal[i] if hasattr(read_awr, 'peakVal') else 0.0)
            
    #     #     # 捕获最大值用于观察
    #     #     if y > max_raw_y: max_raw_y = y
    #     #     if peak_v > max_raw_peak: max_raw_peak = peak_v
    #     #     if abs(v) > max_raw_v: max_raw_v = abs(v)

    #     #     current_frame.append((x, y, v, peak_v))
            
    #     #     # 【调试修改】放宽阈值：距离 0.05~1.0米，强度 > 2.0，速度 > 0.05
    #     #     if (0.05 < y < 1.0) and (peak_v > 2.0) and (abs(v) > 0.05):
    #     #         valid_obj_count += 1
    #     #         if abs(v) > frame_max_vel:
    #     #             frame_max_vel = abs(v)

    #     # # 【调试修改】降低判定门槛：只要有1个有效点，且速度达到 0.1 就算活跃
    #     # is_frame_active = (valid_obj_count >= 1) and (frame_max_vel >= 0.1)

    #     # # UI 点云图刷新逻辑
    #     # if self.ui.cb_show_pointcloud.isChecked() or self.current_mode not in ["IDLE", "IDLE_LOCAL"]:
    #     #     self.data_plot.setData(read_awr.x, read_awr.y)
    #     # else:
    #     #     self.data_plot.setData([], [])
    #     # ==========================================
    #     # 2. 提取当前帧多维特征与活跃度判定 (修复 Peak=0 的问题)
    #     # ==========================================
    #     current_frame = []
    #     valid_obj_count = 0  
    #     frame_max_vel = 0.0

    #     min_len = min(len(read_awr.x), len(read_awr.y), len(read_awr.velocity))
        
    #     for i in range(min_len):
    #         x = read_awr.x[i]
    #         y = read_awr.y[i]
    #         v = read_awr.velocity[i]
    #         # 虽然强度是0，我们依然保留这个字段以防模型输入维度报错
    #         peak_v = read_awr.intensity[i] if hasattr(read_awr, 'intensity') else (read_awr.peakVal[i] if hasattr(read_awr, 'peakVal') else 0.0)
            
    #         current_frame.append((x, y, v, peak_v))
            
    #         # 【关键修改】完全弃用 peak_v 过滤！
    #         # 只要点在雷达正前方 0.1米 到 0.8米 之间，且具备一定的速度 (>0.15)，就算有效动作点
    #         if (0.1 < y < 0.8) and (-0.4 < x < 0.4) and (abs(v) > 0.15):
    #             valid_obj_count += 1
    #             if abs(v) > frame_max_vel:
    #                 frame_max_vel = abs(v)

    #     # 判定条件：当前帧至少有 3 个有效运动点，且其中最大速度达到了 0.25
    #     is_frame_active = (valid_obj_count >= 3) and (frame_max_vel >= 0.25)

    #     # UI 点云图刷新逻辑
    #     if self.ui.cb_show_pointcloud.isChecked() or self.current_mode not in ["IDLE", "IDLE_LOCAL"]:
    #         self.data_plot.setData(read_awr.x, read_awr.y)
    #     else:
    #         self.data_plot.setData([], [])
    #     # ==========================================
    #     # 3. 核心状态机 (采集 / 单次 / 自动)
    #     # ==========================================
        
    #     # --- 模式 A: 连续采集模式 ---
    #     if self.current_mode == "COLLECT_RECORDING":
    #         self.frame_counter += 1
    #         self._update_status_ui(f"录入中: 存入 gesture_{self.base_gesture_number + self.current_collect_idx - 1}.csv ({self.frame_counter}/80)", bg_color="#FFEBEE", text_color="#C62828", border_color="#EF5350")
            
    #         # 录满80帧，自动结束当前组并流转状态
    #         if self.frame_counter >= 80:
    #             read_awr.if_button_is_pushed = 0  # 通知底层停止写入文件
    #             self.current_collect_idx += 1
                
    #             if self.current_collect_idx > self.target_collect_total:
    #                 self.force_to_idle("所有组数采集完毕！已自动保存。")
    #             else:
    #                 self.current_mode = "COLLECT_STANDBY"
    #                 self._update_status_ui(f"第 {self.current_collect_idx-1} 组已存。等待操作下一组...", bg_color="#E8F5E9", text_color="#2E7D32", border_color="#A5D6A7")
    #                 self.update_ui_states()

    #     # --- 模式 B: 单次推理模式 ---
    #     elif self.current_mode == "SINGLE_RECORDING":
    #         self.action_buffer.append(current_frame)
    #         self._update_status_ui(f"单次分类: 收集数据中... ({len(self.action_buffer)}/80)", bg_color="#E3F2FD", text_color="#1565C0", border_color="#90CAF9")
            
    #         if len(self.action_buffer) >= 80:
    #             self._update_status_ui("单次分类: 推理中...", bg_color="#E3F2FD", text_color="#1565C0", border_color="#90CAF9")
    #             self.recognize_from_buffer(self.action_buffer)
    #             self.force_to_idle("单次分类完成")

    #     # --- 模式 C: 自动动态识别模式 ---
    #     elif self.current_mode == "AUTO_RECOG":
    #         # 打印雷达收到的最原始数据极值，帮你校准硬件距离和强度
    #         print(f"[DEBUG - RAW] Pts: {raw_points_count:02d} | Max Y(距离): {max_raw_y:.2f} | Max Peak(强度): {max_raw_peak:.1f} | Max Vel(速度): {max_raw_v:.3f}")
    #         print(f"[DEBUG - FRAME] Valid Objs: {valid_obj_count}, Max Vel: {frame_max_vel:.3f} -> Active: {is_frame_active}")

    #         self.active_history.append(is_frame_active)
    #         if len(self.active_history) > self.history_window_size:
    #             self.active_history.pop(0)

    #         # 阶段 1：空闲监测
    #         if self.gesture_state == "IDLE":
    #             print(f"[DEBUG - SLIDING WINDOW] History: {self.active_history} | Active Sum: {sum(self.active_history)}/{self.history_window_size}")
                
    #             # 滑动窗口内满足活跃次数，触发动作准备
    #             if sum(self.active_history) >= 3:
    #                 self.gesture_state = "PREPARING"
    #                 self.prepare_start_time = time.time()
    #                 self.active_history.clear() 
    #                 self._update_status_ui("⚠️ 识别到活动！请准备，2.0 秒后开启手势录制...", bg_color="#FFF9C4", text_color="#F57F17", border_color="#FBC02D")
    #                 print("[DEBUG] Triggered! Entering PREPARING state.")

    #         # 阶段 2：准备倒计时（防误触与调整姿态）
    #         elif self.gesture_state == "PREPARING":
    #             elapsed = time.time() - self.prepare_start_time
    #             remain = max(0, 2.0 - elapsed)
    #             self._update_status_ui(f"⚠️ 识别到活动！请准备，{remain:.1f} 秒后开启手势录制...", bg_color="#FFF9C4", text_color="#F57F17", border_color="#FBC02D")
                
    #             if elapsed >= 2.0:
    #                 self.gesture_state = "RECORDING"
    #                 self.gesture_data_buffer = []  
    #                 self.silence_counter = 0
    #                 self.is_mocking = False
    #                 self._update_status_ui("🟢 录制已开始，请执行手势动作！", bg_color="#FFEBEE", text_color="#C62828", border_color="#EF5350")
    #                 print("[DEBUG] Started RECORDING data buffer.")

    #         # 阶段 3：正式录制与动态截断
    #         elif self.gesture_state == "RECORDING":
    #             self.gesture_data_buffer.append(current_frame)
                
    #             if not is_frame_active:
    #                 self.silence_counter += 1
    #             else:
    #                 self.silence_counter = 0
                
    #             print(f"[DEBUG - BUFFER] Size: {len(self.gesture_data_buffer)}/{self.max_buffer_size} | Silence: {self.silence_counter}/{self.MAX_SILENCE_FRAMES}")

    #             # 结束条件：连续多帧静默（动作结束） 或 达到最大帧数上限
    #             end_detected = (self.silence_counter >= self.MAX_SILENCE_FRAMES)
    #             max_frames_reached = len(self.gesture_data_buffer) >= self.max_buffer_size
                
    #             if end_detected or max_frames_reached:
    #                 self.gesture_state = "PROCESSING"
    #                 # 剥离尾部多余的静默帧，保证送入模型的数据紧凑有效（至少保留10帧防越界）
    #                 valid_gesture_length = max(10, len(self.gesture_data_buffer) - self.silence_counter)
    #                 final_buffer = self.gesture_data_buffer[:valid_gesture_length]
                        
    #                 self._update_status_ui(f"🔄 状态: 推理中 (有效帧: {len(final_buffer)})", bg_color="#E3F2FD", text_color="#1565C0", border_color="#90CAF9")
                    
    #                 print(f"[DEBUG - FINAL DATA] Buffer sliced to {len(final_buffer)} valid frames.")
    #                 if len(final_buffer) > 0 and len(final_buffer[0]) > 0:
    #                     print(f"[DEBUG - FRAME 0 PREVIEW] First point (x, y, v, peak): {final_buffer[0][0]}")

    #                 # 送入模型进行推理
    #                 self.recognize_from_buffer(final_buffer)
                    
    #                 # 清理状态，回归监测循环
    #                 self.gesture_state = "IDLE"
    #                 self.gesture_data_buffer = []
    #                 self.active_history.clear()
    #                 self.is_mocking = False 
    #                 self._update_status_ui("⏳ 状态: IDLE (继续监测雷达...)", bg_color="#E8F5E9", text_color="#2E7D32", border_color="#A5D6A7")
    #                 print("[DEBUG] Processing complete. Back to IDLE.")

    def main_loop(self):
        # 1. 监测雷达数据流状态
        is_radar_alive = hasattr(read_awr, 'x') and len(read_awr.x) > 0
        
        if is_radar_alive:
            self.last_radar_time = time.time()
            if not self.serial_connected:
                self.serial_connected = True
                self.ui.serial_status_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #1DD1A1; padding-left: 5px;")
                self.ui.serial_status_label.setText("雷达状态: 🟢 已连接")
                self.update_ui_states()
        else:
            if time.time() - self.last_radar_time > 1.0 and self.serial_connected:
                self.serial_connected = False
                self.ui.serial_status_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #E74C3C; padding-left: 5px;")
                self.ui.serial_status_label.setText("雷达状态: 🔴 已断开")
                self.ui.data_plot.setData([], []) 
                self.force_to_idle("雷达断开，强制回到空闲状态")
        
        if not self.serial_connected: return

        # ==========================================
        # 2. 提取当前帧多维特征与活跃度判定 (纯净版)
        # ==========================================
        current_frame = []
        valid_obj_count = 0  
        frame_max_vel = 0.0

        min_len = min(len(read_awr.x), len(read_awr.y), len(read_awr.velocity))
        
        for i in range(min_len):
            x = read_awr.x[i]
            y = read_awr.y[i]
            v = read_awr.velocity[i]
            
            # 保留 peak_v 仅用于填充数据结构（防止后续预处理函数报错），但完全不参与逻辑判定
            peak_v = read_awr.intensity[i] if hasattr(read_awr, 'intensity') else (read_awr.peakVal[i] if hasattr(read_awr, 'peakVal') else 0.0)
            
            current_frame.append((x, y, v, peak_v))
            
            # 【核心判定条件】只看 Y距离 和 速度！
            # 手部在 0.1 ~ 0.8 米之间，且运动速度 > 0.15 视为有效动作点
            if (0.1 < y < 0.8) and (-0.4 < x < 0.4) and (abs(v) > 0.15):
                valid_obj_count += 1
                if abs(v) > frame_max_vel:
                    frame_max_vel = abs(v)

        # 帧活跃判定：至少有 3 个点满足距离和速度要求，且最大速度达到 0.25
        is_frame_active = (valid_obj_count >= 3) and (frame_max_vel >= 0.25)

        # UI 点云图刷新逻辑
        if self.ui.cb_show_pointcloud.isChecked() or self.current_mode not in ["IDLE", "IDLE_LOCAL"]:
            self.data_plot.setData(read_awr.x, read_awr.y)
        else:
            self.data_plot.setData([], [])

        # ==========================================
        # 3. 核心状态机 (采集 / 单次 / 自动)
        # ==========================================
        
        # --- 模式 A: 连续采集模式 ---
        if self.current_mode == "COLLECT_RECORDING":
            self.frame_counter += 1
            self._update_status_ui(f"录入中: 存入 gesture_{self.base_gesture_number + self.current_collect_idx - 1}.csv ({self.frame_counter}/80)", bg_color="#FFEBEE", text_color="#C62828", border_color="#EF5350")
            
            if self.frame_counter >= 80:
                read_awr.if_button_is_pushed = 0 
                self.current_collect_idx += 1
                
                if self.current_collect_idx > self.target_collect_total:
                    self.force_to_idle("所有组数采集完毕！已自动保存。")
                else:
                    self.current_mode = "COLLECT_STANDBY"
                    self._update_status_ui(f"第 {self.current_collect_idx-1} 组已存。等待操作下一组...", bg_color="#E8F5E9", text_color="#2E7D32", border_color="#A5D6A7")
                    self.update_ui_states()

        # --- 模式 B: 单次推理模式 ---
        elif self.current_mode == "SINGLE_RECORDING":
            self.action_buffer.append(current_frame)
            self._update_status_ui(f"单次分类: 收集数据中... ({len(self.action_buffer)}/80)", bg_color="#E3F2FD", text_color="#1565C0", border_color="#90CAF9")
            
            if len(self.action_buffer) >= 80:
                self._update_status_ui("单次分类: 推理中...", bg_color="#E3F2FD", text_color="#1565C0", border_color="#90CAF9")
                self.recognize_from_buffer(self.action_buffer)
                self.force_to_idle("单次分类完成")

        # --- 模式 C: 自动动态识别模式 ---
        elif self.current_mode == "AUTO_RECOG":
            # 移除了报错的调试日志，只保留核心状态追踪
            print(f"[DEBUG - FRAME] Valid Pts: {valid_obj_count}, Max Vel: {frame_max_vel:.3f} -> Active: {is_frame_active}")

            self.active_history.append(is_frame_active)
            if len(self.active_history) > self.history_window_size:
                self.active_history.pop(0)

            # 阶段 1：空闲监测
            if self.gesture_state == "IDLE":
                if sum(self.active_history) >= 3:
                    self.gesture_state = "PREPARING"
                    self.prepare_start_time = time.time()
                    self.active_history.clear() 
                    self._update_status_ui("⚠️ 识别到活动！请准备，2.0 秒后开启手势录制...", bg_color="#FFF9C4", text_color="#F57F17", border_color="#FBC02D")

            # 阶段 2：准备倒计时
            elif self.gesture_state == "PREPARING":
                elapsed = time.time() - self.prepare_start_time
                remain = max(0, 2.0 - elapsed)
                self._update_status_ui(f"⚠️ 识别到活动！请准备，{remain:.1f} 秒后开启手势录制...", bg_color="#FFF9C4", text_color="#F57F17", border_color="#FBC02D")
                
                if elapsed >= 2.0:
                    self.gesture_state = "RECORDING"
                    self.gesture_data_buffer = []  
                    self.silence_counter = 0
                    self.is_mocking = False
                    self._update_status_ui("🟢 录制已开始，请执行手势动作！", bg_color="#FFEBEE", text_color="#C62828", border_color="#EF5350")

            # 阶段 3：正式录制与动态截断
            elif self.gesture_state == "RECORDING":
                self.gesture_data_buffer.append(current_frame)
                
                if not is_frame_active:
                    self.silence_counter += 1
                else:
                    self.silence_counter = 0
                
                end_detected = (self.silence_counter >= self.MAX_SILENCE_FRAMES)
                max_frames_reached = len(self.gesture_data_buffer) >= self.max_buffer_size
                
                if end_detected or max_frames_reached:
                    self.gesture_state = "PROCESSING"
                    valid_gesture_length = max(10, len(self.gesture_data_buffer) - self.silence_counter)
                    final_buffer = self.gesture_data_buffer[:valid_gesture_length]
                        
                    self._update_status_ui(f"🔄 状态: 推理中 (有效帧: {len(final_buffer)})", bg_color="#E3F2FD", text_color="#1565C0", border_color="#90CAF9")
                    
                    self.recognize_from_buffer(final_buffer)
                    
                    self.gesture_state = "IDLE"
                    self.gesture_data_buffer = []
                    self.active_history.clear()
                    self.is_mocking = False 
                    self._update_status_ui("⏳ 状态: IDLE (继续监测雷达...)", bg_color="#E8F5E9", text_color="#2E7D32", border_color="#A5D6A7")
                    
                    
    def recognize_from_buffer(self, buffer_data):
        fusion_gesture = "Unknown"
        start_total = time.time()
        tensor_lstm, tensor_gru = preprocess_for_fusion(buffer_data)
        preprocess_ms = (time.time() - start_total) * 1000.0
        
        if tensor_lstm is not None and tensor_gru is not None:
            with torch.no_grad():
                
                # ========== Fusion 模型结果判断 ==========
                if self.is_mocking:
                    fusion_gesture = self.mock_gestures[self.mock_sequence_index]
                    self.mock_sequence_index = (self.mock_sequence_index + 1) % len(self.mock_gestures)
                else:
                    if self.fusion_model:
                        start_infer = time.time()
                        outputs = self.fusion_model(tensor_lstm, tensor_gru)
                        _, predicted = torch.max(outputs.data, 1)
                        fusion_gesture = Gestures.REVERSE_LABELS.get(predicted.item(), "Unknown")
                        inference_ms = (time.time() - start_infer) * 1000.0
                        total_ms = (time.time() - start_total) * 1000.0
                        self._record_performance(preprocess_ms, inference_ms, total_ms, fusion_gesture)

                # UI更新Fusion结果
                self.ui.fusion_recognition_label.setText(f"🔥 Fusion 实时结果: {fusion_gesture}")
                if fusion_gesture in GESTURE_COLOR_MAP:
                    target_color = GESTURE_COLOR_MAP[fusion_gesture]
                    self.rgb_ctrl.fade_to(*target_color)
                
                # ========== LSTM 模型结果判断 (始终保持真实) ==========
                if self.ui.cb_show_lstm.isChecked() and self.lstm_model:
                    out_lstm = self.lstm_model(tensor_lstm, tensor_gru)
                    _, pred_lstm = torch.max(out_lstm.data, 1)
                    lstm_gesture = Gestures.REVERSE_LABELS.get(pred_lstm.item(), "Unknown")
                    self.ui.lstm_result_label.setText(f"📘 单独 LSTM 结果: {lstm_gesture}")
                elif self.ui.cb_show_lstm.isChecked() and not self.lstm_model:
                    self.ui.lstm_result_label.setText("📘 单独 LSTM 结果: (模型未加载)")

                # ========== GRU 模型结果判断 (始终保持真实) ==========
                if self.ui.cb_show_gru.isChecked() and self.gru_model:
                    out_gru = self.gru_model(tensor_lstm, tensor_gru)
                    _, pred_gru = torch.max(out_gru.data, 1)
                    gru_gesture = Gestures.REVERSE_LABELS.get(pred_gru.item(), "Unknown")
                    self.ui.gru_result_label.setText(f"📗 单独 GRU 结果: {gru_gesture}")
                elif self.ui.cb_show_gru.isChecked() and not self.gru_model:
                    self.ui.gru_result_label.setText("📗 单独 GRU 结果: (模型未加载)")

        else:
            self.ui.fusion_recognition_label.setText("Fusion 实时结果: Process error")
            
        return fusion_gesture

    def closeEvent(self, event):
        print("窗口关闭中，执行收尾操作...")
        try:
            self.rgb_ctrl.cleanup()
            if hasattr(read_awr, 'closeEvent'): read_awr.closeEvent()
        except Exception as e:
            print(f"关闭时出错: {e}")
        finally:
            self._write_performance_summary()
            event.accept()

if __name__ == "__main__":
    read_awr.main()
    import sys
    app = QtWidgets.QApplication(sys.argv)
    c = MainWindow()
    c.show()
    sys.exit(app.exec_())