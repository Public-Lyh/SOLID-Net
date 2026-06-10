import os
import sys
import json
import pickle
import warnings
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import defaultdict, Counter
from datetime import datetime
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torchvision import models
from scipy.fft import fft
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

warnings.filterwarnings('ignore')


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

ROOT_PATH = Path("/home/luoyh/deep_learning_project/Dataset")
HAND_DETECT_MODEL = ROOT_PATH / "models" / "best.pt"
VISUAL_MODEL_PATH = ROOT_PATH / "models" / "mvtf_visual.pth"
EMG_MODEL_DIR = ROOT_PATH / "models" / "EMG"
TEST_DIR = ROOT_PATH / "test"
EMG_INIT_DIR = TEST_DIR / "initialize"

ABLATION_DIR = ROOT_PATH / "models" / "Ablation" / "SC_MSFE" / "Fusion"

NUM_CLASSES = 13

GESTURE_LABELS = (
    'Fist Clench', 'Finger Extension', 'Wrist Rotation',
    'Finger Opposition', 'Ball Squeeze', 'TheraPutty Pinch',
    'Finger Pressing', 'Interlace Fingers', 'Wrist Flexion-Extension',
    'Finger Massage', 'Towel Scrunch', 'Hand Tapping', 'Piano Tap',
)

GESTURE_NAMES_SHORT = (
    'Fist', 'Extension', 'Rotation', 'Opposition', 'Ball', 'Putty', 'Press',
    'Interlace', 'Flexion', 'Massage', 'Towel', 'Tapping', 'Piano'
)

PROBLEM_CLASSES = {0: 'Fist Clench', 1: 'Finger Extension',
                   2: 'Wrist Rotation', 12: 'Piano Tap'}

EMG_CONFUSION_PAIRS = {
    0: [1, 4], 4: [3, 0], 5: [4, 6], 6: [5],
    9: [5, 8], 11: [5], 12: [5],
}

VISUAL_STRONG_CLASSES = {3, 4, 6, 7}
EMG_STRONG_CLASSES = {1, 8, 10}


def build_gesture_label_map():
    label_map = {}
    for idx, label in enumerate(GESTURE_LABELS):
        label_map[label.lower()] = idx
    additional = {
        'fist clench': 0, 'fistclench': 0,
        'finger extension': 1, 'fingerextension': 1,
        'wrist rotation': 2, 'wristrotation': 2,
        'finger opposition': 3, 'fingeropposition': 3,
        'ball squeeze': 4, 'ballsqueeze': 4, 'ball grab': 4,
        'theraputty pinch': 5, 'theraputtypinch': 5, 'putty pinch': 5,
        'finger pressing': 6, 'fingerpressing': 6, 'key press': 6,
        'interlace fingers': 7, 'interlacefingers': 7, 'finger interlace': 7,
        'wrist flexion-extension': 8, 'wrist flexion extension': 8,
        'wristflexion-extension': 8, 'wristflexionextension': 8,
        'finger massage': 9, 'fingermassage': 9, 'self massage': 9,
        'towel scrunch': 10, 'towelscrunch': 10, 'towel wring': 10,
        'hand tapping': 11, 'handtapping': 11,
        'piano tap': 12, 'pianotap': 12, 'piano playing': 12,
    }
    label_map.update(additional)
    return label_map

GESTURE_LABEL_MAP = build_gesture_label_map()


def match_gesture_label(label_str):
    if not isinstance(label_str, str):
        return None
    label_clean = label_str.strip().lower()
    if label_clean in GESTURE_LABEL_MAP:
        return GESTURE_LABEL_MAP[label_clean]
    label_normalized = ' '.join(label_clean.split())
    if label_normalized in GESTURE_LABEL_MAP:
        return GESTURE_LABEL_MAP[label_normalized]
    return None


def safe_normalize(arr):
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(arr, -1e6, 1e6)


def select_gpu():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3 and int(parts[0]) == 1:
                    if float(parts[1]) / float(parts[2]) < 0.8:
                        return 'cuda:1'
    except:
        pass
    return 'cuda:1' if torch.cuda.is_available() else 'cpu'


def compute_confidence(proba: np.ndarray) -> float:
    if proba is None or len(proba) == 0:
        return 0.0
    sorted_p = np.sort(proba)[::-1]
    return float(sorted_p[0] - sorted_p[1]) if len(sorted_p) >= 2 else float(sorted_p[0])


@dataclass
class FusionConfig:
    num_classes: int = 13
    emg_channels: int = 6
    window_size: int = 180
    step_size: int = 22
    visual_seq_len: int = 16
    visual_hidden_dim: int = 256
    device: str = 'cuda:1'
    hand_conf: float = 0.25
    hand_padding: float = 0.15
    output_size: int = 224
    phase1_epochs: int = 60
    phase2_epochs: int = 90
    phase1_lr: float = 0.015
    phase2_lr: float = 0.005
    focal_gamma: float = 2.0
    problem_class_weight: float = 3.0
    min_weight: float = 0.10
    confusion_penalty: float = 0.5


class SCMSFEFeatureExtractor:
    def __init__(self, num_channels=6, window_size=180):
        self.num_channels = num_channels
        self.window_size = window_size
        self.scales = [2, 4, 6, 8]
        self.feature_dim = None

    def _normalize_window(self, X):
        N, T, C = X.shape
        eps = 1e-8
        X_min = X.min(axis=1, keepdims=True)
        X_max = X.max(axis=1, keepdims=True)
        return (X - X_min) / (X_max - X_min + eps)

    def _shape_features(self, ch, ci):
        N, T = ch.shape
        eps = 1e-8
        feats = []
        feats.append(np.mean(ch, axis=1))
        feats.append(np.std(ch, axis=1))
        feats.append(np.median(ch, axis=1))
        diff = np.diff(ch, axis=1)
        feats.append(np.sum(diff > 0, axis=1) / (T - 1))
        feats.append(np.mean(np.abs(diff), axis=1))
        pd_ = np.where(diff > 0, diff, 0)
        nd_ = np.where(diff < 0, -diff, 0)
        feats.append(np.sum(pd_, axis=1) / (np.sum(diff > 0, axis=1) + eps))
        feats.append(np.sum(nd_, axis=1) / (np.sum(diff < 0, axis=1) + eps))
        d2 = np.diff(diff, axis=1)
        feats.append(np.sum(np.sign(d2[:, 1:] + eps) != np.sign(d2[:, :-1] + eps), axis=1) / T)
        feats.append(np.mean(np.abs(d2), axis=1))
        feats.append(np.max(np.abs(d2), axis=1))
        feats.append(np.argmax(ch, axis=1) / T)
        feats.append(np.argmin(ch, axis=1) / T)
        feats.append((np.argmax(ch, axis=1) < np.argmin(ch, axis=1)).astype(np.float32))
        cen = ch - 0.5
        zc = np.sign(cen[:, 1:] + eps) != np.sign(cen[:, :-1] + eps)
        feats.append(np.argmax(zc, axis=1) / T)
        feats.append(np.sum(zc, axis=1) / T)
        for p in [10, 25, 75, 90]:
            feats.append(np.percentile(ch, p, axis=1))
        feats.append(np.percentile(ch, 75, axis=1) - np.percentile(ch, 25, axis=1))
        return feats

    def _multiscale_features(self, ch, ci):
        N, T = ch.shape
        eps = 1e-8
        feats = []
        for ns in self.scales:
            sl = T // ns
            sm, ss, se = [], [], []
            for s in range(ns):
                a, b = s * sl, (s + 1) * sl if s < ns - 1 else T
                seg = ch[:, a:b]
                sm.append(np.mean(seg, axis=1))
                se.append(np.mean(seg ** 2, axis=1))
                x = np.arange(b - a)
                xm = x.mean()
                slp = np.sum((x - xm) * (seg - seg.mean(axis=1, keepdims=True)), axis=1)
                ss.append(slp / (np.sum((x - xm) ** 2) + eps))
            for m in sm:
                feats.append(m)
            for sl_ in ss:
                feats.append(sl_)
            for e in se:
                feats.append(e)
            sd = np.diff(np.array(sm).T, axis=1)
            for i in range(sd.shape[1]):
                feats.append(sd[:, i])
        return feats

    def _freq_features(self, ch, ci):
        N, T = ch.shape
        eps = 1e-8
        feats = []
        fr = np.abs(fft(ch, axis=1))[:, :T // 2]
        fs = np.sum(fr, axis=1, keepdims=True) + eps
        fn = fr / fs
        feats.append(np.argmax(fn, axis=1) / (T // 2))
        bins = np.arange(T // 2)
        cent = np.sum(fn * bins, axis=1) / (T // 2)
        feats.append(cent)
        feats.append(np.sqrt(np.sum(fn * (bins - cent.reshape(-1, 1)) ** 2, axis=1)))
        nb = T // 2
        for nbd in [4, 6]:
            bl = nb // nbd
            for i in range(nbd):
                a, b = i * bl, (i + 1) * bl if i < nbd - 1 else nb
                feats.append(np.sum(fn[:, a:b], axis=1))
        mid = nb // 2
        feats.append(np.sum(fn[:, mid:], axis=1) / (np.sum(fn[:, :mid], axis=1) + eps))
        lf = np.log(fr + eps)
        gm = np.exp(np.mean(lf, axis=1))
        am = np.mean(fr, axis=1) + eps
        feats.append(gm / am)
        return feats

    def _channel_features(self, Xn):
        N, T, C = Xn.shape
        eps = 1e-8
        feats = []
        en = np.mean(Xn ** 2, axis=1)
        tot = np.sum(en, axis=1, keepdims=True) + eps
        rat = en / tot
        for c in range(C):
            feats.append(rat[:, c])
        rk = np.argsort(np.argsort(-en, axis=1), axis=1)
        for c in range(C):
            feats.append(rk[:, c] / C)
        feats.append(np.argmax(en, axis=1) / C)
        tmp = en.copy()
        tmp[np.arange(N), np.argmax(en, axis=1)] = -1
        feats.append(np.argmax(tmp, axis=1) / C)
        for i in range(C):
            for j in range(i + 1, C):
                xi = Xn[:, :, i] - Xn[:, :, i].mean(axis=1, keepdims=True)
                xj = Xn[:, :, j] - Xn[:, :, j].mean(axis=1, keepdims=True)
                num = np.sum(xi * xj, axis=1)
                den = np.sqrt(np.sum(xi ** 2, axis=1) * np.sum(xj ** 2, axis=1) + eps)
                feats.append(num / den)
        for i in range(C - 1):
            feats.append(np.abs(en[:, i] - en[:, i + 1]))
        feats.append(-np.sum(rat * np.log(rat + eps), axis=1))
        return feats

    def _temporal_features(self, Xn):
        N, T, C = Xn.shape
        eps = 1e-8
        feats = []
        act = np.sum(np.abs(np.diff(Xn, axis=1)), axis=2)
        thr = np.percentile(act, 70, axis=1, keepdims=True)
        amk = act > thr
        feats.append(np.argmax(amk, axis=1) / (T - 1))
        feats.append(np.argmax(act, axis=1) / (T - 1))
        feats.append(np.sum(amk, axis=1) / (T - 1))
        ns = 4
        sl = (T - 1) // ns
        sa = []
        for s in range(ns):
            a, b = s * sl, (s + 1) * sl if s < ns - 1 else T - 1
            sa.append(np.mean(act[:, a:b], axis=1))
        sa = np.column_stack(sa)
        ta = np.sum(sa, axis=1, keepdims=True) + eps
        for s in range(ns):
            feats.append(sa[:, s] / ta.flatten())
        ad = np.abs(np.diff(act, axis=1))
        feats.append(np.max(ad, axis=1))
        feats.append(np.argmax(ad, axis=1) / (T - 2))
        return feats

    def _traditional_features(self, X):
        N, T, C = X.shape
        eps = 1e-8
        feats = []
        for c in range(C):
            ch = X[:, :, c]
            feats.append(np.mean(np.abs(ch), axis=1))
            feats.append(np.sqrt(np.mean(ch ** 2, axis=1) + eps))
            feats.append(np.var(ch, axis=1))
            feats.append(np.sum(np.abs(np.diff(ch, axis=1)), axis=1))
            feats.append(np.sum(np.sign(ch[:, 1:] + eps) != np.sign(ch[:, :-1] + eps),
                                axis=1) / 2)
            d1 = ch[:, 1:-1] - ch[:, :-2]
            d2 = ch[:, 1:-1] - ch[:, 2:]
            feats.append(np.sum((d1 * d2) > 0, axis=1))
            feats.append(np.max(ch, axis=1) - np.min(ch, axis=1))
        return feats

    def extract(self, X):
        if X.ndim == 2:
            X = X[np.newaxis, ...]
        Xn = self._normalize_window(X)
        af = []
        for c in range(min(self.num_channels, X.shape[2])):
            ch = Xn[:, :, c]
            af.extend(self._shape_features(ch, c))
            af.extend(self._multiscale_features(ch, c))
            af.extend(self._freq_features(ch, c))
        af.extend(self._channel_features(Xn))
        af.extend(self._temporal_features(Xn))
        af.extend(self._traditional_features(X))
        result = safe_normalize(np.column_stack(af)).astype(np.float32)
        self.feature_dim = result.shape[1]
        return result


class SCMSFE:
    def __init__(self, config: FusionConfig):
        self.config = config
        self.num_classes = config.num_classes
        self.extractor = SCMSFEFeatureExtractor(config.emg_channels, config.window_size)
        self.model = None
        self.scaler = None
        self.expected_features = None
        self.is_loaded = False

    def load(self, model_dir: Path) -> bool:
        try:
            complete_path = model_dir / 'sc_msfe_complete.pkl'
            if complete_path.exists():
                with open(complete_path, 'rb') as f:
                    data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                self.expected_features = data['feature_dim']
                self.is_loaded = True
                print(f"  [SC-MSFE] Loaded (dim={self.expected_features})")
                return True
        except Exception as e:
            print(f"  [SC-MSFE] Load failed: {e}")
        return False

    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        if not self.is_loaded:
            return np.ones(self.num_classes) / self.num_classes
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        X_feat = self.extractor.extract(data)
        if self.expected_features and X_feat.shape[1] != self.expected_features:
            if X_feat.shape[1] < self.expected_features:
                X_feat = np.hstack([X_feat, np.zeros((X_feat.shape[0],
                                                       self.expected_features - X_feat.shape[1]))])
            else:
                X_feat = X_feat[:, :self.expected_features]
        try:
            X_scaled = self.scaler.transform(X_feat)
            proba = self.model.predict_proba(X_scaled)
            full_proba = np.ones(self.num_classes) * 1e-6
            if hasattr(self.model, 'classes_'):
                for i, cls in enumerate(self.model.classes_):
                    if cls < self.num_classes:
                        full_proba[cls] = proba[0, i]
            else:
                for i in range(min(proba.shape[1], self.num_classes)):
                    full_proba[i] = proba[0, i]
            return full_proba / (full_proba.sum() + 1e-10)
        except:
            return np.ones(self.num_classes) / self.num_classes


class GrayVariation(nn.Module):
    def __init__(self, eta=16):
        super().__init__()
        self.eta = eta
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        xmin = x.amin(dim=(2, 3), keepdim=True)
        xmax = x.amax(dim=(2, 3), keepdim=True)
        xn = (x - xmin) / (xmax - xmin + 1e-8)
        xq = torch.floor(xn * self.eta) / self.eta
        return self.alpha * xq + (1 - self.alpha) * xn


class MotionModule(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.vel_proj = nn.Linear(dim, dim)
        self.acc_proj = nn.Linear(dim, dim)
        self.gate = nn.Sequential(nn.Linear(dim * 3, dim), nn.Sigmoid())
        self.fuse = nn.Linear(dim * 3, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        vel = torch.zeros_like(x)
        vel[:, 1:] = x[:, 1:] - x[:, :-1]
        acc = torch.zeros_like(x)
        acc[:, 2:] = vel[:, 2:] - vel[:, 1:-1]
        concat = torch.cat([x, self.dropout(self.vel_proj(vel)),
                            self.dropout(self.acc_proj(acc))], dim=-1)
        return self.norm(x + self.gate(concat) * self.fuse(concat))


class MultiScaleBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.branch1 = nn.Conv1d(dim, dim // 4, 1)
        self.branch3 = nn.Sequential(nn.Conv1d(dim, dim // 4, 1),
                                     nn.Conv1d(dim // 4, dim // 4, 3, padding=1))
        self.branch5 = nn.Sequential(nn.Conv1d(dim, dim // 4, 1),
                                     nn.Conv1d(dim // 4, dim // 4, 5, padding=2))
        self.branch_pool = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                         nn.Conv1d(dim, dim // 4, 1))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        x_t = x.transpose(1, 2)
        out = torch.cat([self.branch1(x_t), self.branch3(x_t), self.branch5(x_t),
                         self.branch_pool(x_t).expand(-1, -1, T)], dim=1).transpose(1, 2)
        return self.norm(self.dropout(out) + x)


class TemporalAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = self.dropout(F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.norm(x + self.proj(out))


class SpatialAttention(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        q = self.query(x.mean(dim=1, keepdim=True))
        attn = self.dropout(F.softmax(
            torch.bmm(q, self.key(x).transpose(-2, -1)) / (D ** 0.5), dim=-1))
        return self.norm(x + torch.bmm(attn, self.value(x)).expand(-1, T, -1))


class HierarchicalModule(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.frame_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.frame_norm = nn.LayerNorm(dim)
        self.seg_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.seg_norm = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.frame_attn(x, x, x)
        x = self.frame_norm(x + attn_out)
        seg_out, _ = self.seg_attn(x.mean(dim=1, keepdim=True), x, x)
        return self.seg_norm(x + seg_out.expand(-1, x.size(1), -1))


class CASTNetBackbone(nn.Module):
    def __init__(self, num_classes=13, hidden_dim=256, num_heads=4, dropout=0.3):
        super().__init__()
        self.gvar = GrayVariation(eta=16)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )
        self.proj = nn.Sequential(nn.Linear(512, hidden_dim),
                                  nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.motion = MotionModule(hidden_dim, dropout)
        self.multiscale = MultiScaleBlock(hidden_dim, dropout)
        self.temporal = TemporalAttention(hidden_dim, num_heads, dropout)
        self.spatial = SpatialAttention(hidden_dim, dropout)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2,
                          batch_first=True, bidirectional=True, dropout=dropout)
        self.gru_proj = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim),
                                      nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.seq_attn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2),
                                      nn.Tanh(), nn.Linear(hidden_dim // 2, 1))
        self.hier = HierarchicalModule(hidden_dim, num_heads, dropout)
        self.fusion = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim),
                                    nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(hidden_dim, hidden_dim // 2),
                                  nn.GELU(), nn.Dropout(0.3),
                                  nn.Linear(hidden_dim // 2, num_classes))

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = self.gvar(x.view(B * T, C, H, W))
        x = F.adaptive_avg_pool2d(self.encoder(x), (1, 1)).view(B, T, -1)
        x = self.spatial(self.temporal(self.multiscale(self.motion(self.proj(x)))))
        self.gru.flatten_parameters()
        gru_out = self.gru_proj(self.gru(x)[0])
        seq_feat = (gru_out * F.softmax(self.seq_attn(gru_out), dim=1)).sum(dim=1)
        hier_feat = self.hier(gru_out).mean(dim=1)
        return self.head(self.fusion(torch.cat([seq_feat, hier_feat], dim=-1)))


class HandDetector:
    def __init__(self, model_path: Path, device: str):
        self.model = None
        self.is_loaded = False
        self._cache = {}
        if YOLO_AVAILABLE and model_path.exists():
            try:
                self.model = YOLO(str(model_path))
                self.is_loaded = True
                print(f"  [HandDetector] Loaded: {model_path.name}")
            except Exception as e:
                print(f"  [HandDetector] Failed: {e}")

    def detect_and_crop(self, image_path, output_size=224, conf=0.25, padding=0.15):
        if image_path in self._cache:
            return self._cache[image_path]
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        result_img = None
        if self.is_loaded:
            try:
                results = self.model.predict(source=str(image_path), conf=conf, verbose=False)
                if len(results) > 0 and len(results[0].boxes) > 0:
                    box = results[0].boxes.xyxy[results[0].boxes.conf.argmax()].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box)
                    pad_w = int((x2 - x1) * padding)
                    pad_h = int((y2 - y1) * padding)
                    x1, y1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
                    x2, y2 = min(w, x2 + pad_w), min(h, y2 + pad_h)
                    cropped = img_rgb[y1:y2, x1:x2]
                    if cropped.size > 0:
                        result_img = cv2.resize(cropped, (output_size, output_size))
            except:
                pass
        if result_img is None:
            size = min(h, w)
            top, left = (h - size) // 2, (w - size) // 2
            result_img = cv2.resize(img_rgb[top:top + size, left:left + size],
                                    (output_size, output_size))
        if len(self._cache) < 8000:
            self._cache[image_path] = result_img
        return result_img


class CASTNet:
    def __init__(self, config: FusionConfig):
        self.config = config
        self.num_classes = config.num_classes
        self.device = config.device
        self.hand_detector = HandDetector(HAND_DETECT_MODEL, config.device)
        self.model = None
        self.is_loaded = False
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load(self, model_path: Path) -> bool:
        if not model_path.exists():
            return False
        try:
            self.model = CASTNetBackbone(
                num_classes=self.num_classes,
                hidden_dim=self.config.visual_hidden_dim
            )
            state_dict = torch.load(model_path, map_location=self.config.device)
            if isinstance(state_dict, dict):
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
            try:
                self.model.load_state_dict(state_dict)
            except RuntimeError:
                new_sd = {}
                for k, v in state_dict.items():
                    new_sd[k.replace('module.', '')] = v
                self.model.load_state_dict(new_sd, strict=False)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            print(f"  [CAST-Net] Loaded from {model_path.name}")
            return True
        except Exception as e:
            print(f"  [CAST-Net] Load failed: {e}")
            return False

    def _prepare_sequence(self, frame_paths):
        imgs = []
        for path in frame_paths:
            cropped = self.hand_detector.detect_and_crop(
                path, self.config.output_size,
                self.config.hand_conf, self.config.hand_padding
            )
            if cropped is not None:
                imgs.append(self.transform(cropped))
        if not imgs:
            return None
        seq_len = self.config.visual_seq_len
        if len(imgs) >= seq_len:
            indices = np.linspace(0, len(imgs) - 1, seq_len).astype(int)
            imgs = [imgs[i] for i in indices]
        else:
            while len(imgs) < seq_len:
                imgs.append(imgs[-1])
        return torch.stack(imgs).unsqueeze(0)

    def predict_proba(self, data: List) -> np.ndarray:
        if not self.is_loaded:
            return np.ones(self.num_classes) / self.num_classes
        batch = self._prepare_sequence(data)
        if batch is None:
            return np.ones(self.num_classes) / self.num_classes
        self.model.eval()
        with torch.no_grad():
            output = self.model(batch.to(self.device))
            return F.softmax(output, dim=-1).squeeze(0).cpu().numpy()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha = self.alpha.to(targets.device)
            focal_loss = alpha[targets] * focal_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss


class AdaptiveWeightModule(nn.Module):
    def __init__(self, num_classes, num_modalities,
                 emg_per_class_acc=None, vis_per_class_acc=None,
                 min_weight=0.10):
        super().__init__()
        self.num_classes = num_classes
        self.num_modalities = num_modalities
        self.min_weight = min_weight

        init_logits = torch.zeros(num_modalities, num_classes)

        if emg_per_class_acc is not None and vis_per_class_acc is not None:
            for c in range(num_classes):
                emg_acc = emg_per_class_acc.get(c, 0.5)
                vis_acc = vis_per_class_acc.get(c, 0.5)
                total = emg_acc + vis_acc + 1e-10
                emg_w = np.clip(emg_acc / total, min_weight, 1.0 - min_weight)
                vis_w = np.clip(vis_acc / total, min_weight, 1.0 - min_weight)

                if c in VISUAL_STRONG_CLASSES and vis_acc > emg_acc:
                    vis_w = min(0.85, vis_w * 1.3)
                    emg_w = 1.0 - vis_w
                if c in EMG_STRONG_CLASSES and emg_acc > vis_acc:
                    emg_w = min(0.85, emg_w * 1.3)
                    vis_w = 1.0 - emg_w
                if c in EMG_CONFUSION_PAIRS:
                    emg_w = max(min_weight, emg_w * 0.7)
                    vis_w = 1.0 - emg_w

                init_logits[0, c] = np.log(emg_w / (1 - emg_w + 1e-10))
                init_logits[1, c] = np.log(vis_w / (1 - vis_w + 1e-10))

        self.weight_logits = nn.Parameter(init_logits)

    def forward(self):
        raw = torch.sigmoid(self.weight_logits)
        raw = raw * (1 - 2 * self.min_weight) + self.min_weight
        return raw / (raw.sum(dim=0, keepdim=True) + 1e-10)

    def get_weights_numpy(self):
        with torch.no_grad():
            return self().cpu().numpy()


class ConfusionAwareNormalization(nn.Module):
    def __init__(self, num_classes, num_modalities, config):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = nn.Parameter(torch.ones(num_modalities, num_classes))
        self.scale = nn.Parameter(torch.ones(num_modalities, num_classes))
        self.bias = nn.Parameter(torch.zeros(num_modalities, num_classes))

        cm = torch.zeros(num_classes, num_classes)
        for pc, targets in EMG_CONFUSION_PAIRS.items():
            for t in targets:
                if pc < num_classes and t < num_classes:
                    cm[pc, t] = config.confusion_penalty
                    cm[t, pc] = config.confusion_penalty * 0.3
        self.register_buffer('confusion_matrix', cm)

    def forward(self, proba, base_weights, modality_idx, predicted_class=None):
        if proba.dim() == 1:
            proba = proba.unsqueeze(0)
        temp = F.softplus(self.temperature[modality_idx])
        scale = F.softplus(self.scale[modality_idx])
        bias = self.bias[modality_idx]
        adj_temp = torch.clamp(temp * (1.5 - base_weights), min=0.1, max=5.0)
        log_p = torch.log(proba + 1e-10)
        scaled = log_p / adj_temp.unsqueeze(0) * scale.unsqueeze(0) + bias.unsqueeze(0)
        if predicted_class is not None and predicted_class < self.num_classes:
            if modality_idx == 0:
                scaled = scaled - self.confusion_matrix[predicted_class].unsqueeze(0) * 0.5
        return F.softmax(scaled, dim=-1).squeeze(0)


class DisagreementResolver(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.arbitration_bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, emg_proba, vis_proba, emg_weight, vis_weight, fused_proba):
        emg_pred = torch.argmax(emg_proba).item()
        vis_pred = torch.argmax(vis_proba).item()
        if emg_pred == vis_pred:
            return fused_proba
        adjusted = fused_proba.clone()
        vis_conf = vis_proba[vis_pred].item()
        if vis_pred in VISUAL_STRONG_CLASSES and vis_conf > 0.3:
            boost = 0.15 + F.sigmoid(self.arbitration_bias[vis_pred]).item() * 0.15
            adjusted[vis_pred] = adjusted[vis_pred] + boost
            if emg_pred in EMG_CONFUSION_PAIRS:
                adjusted[emg_pred] = adjusted[emg_pred] * 0.7
        if emg_pred in EMG_CONFUSION_PAIRS:
            confused_targets = EMG_CONFUSION_PAIRS[emg_pred]
            if vis_pred not in confused_targets and vis_pred != emg_pred:
                adjusted[emg_pred] = adjusted[emg_pred] * 0.75
                adjusted[vis_pred] = adjusted[vis_pred] * 1.2
        adjusted = adjusted / (adjusted.sum() + 1e-10)
        return adjusted


class SOLIDNetWeightsOnly(nn.Module):
    def __init__(self, num_classes, config, emg_acc, vis_acc):
        super().__init__()
        self.num_classes = num_classes
        self.base_weights = AdaptiveWeightModule(
            num_classes, 2, emg_acc, vis_acc, config.min_weight)

        class_weights = torch.ones(num_classes)
        for pc in PROBLEM_CLASSES:
            class_weights[pc] = config.problem_class_weight
        self.register_buffer('class_weights', class_weights)
        self.focal_loss = FocalLoss(gamma=config.focal_gamma, alpha=class_weights)

    def forward(self, emg_proba, vis_proba):
        weights = self.base_weights()
        fused = weights[0] * emg_proba + weights[1] * vis_proba
        fused = fused / (fused.sum() + 1e-10)
        return fused

    def compute_loss(self, fused, label, device):
        return self.focal_loss(fused.unsqueeze(0),
                               torch.tensor([label], device=device))


class SOLIDNetWeightsNorm(nn.Module):
    def __init__(self, num_classes, config, emg_acc, vis_acc):
        super().__init__()
        self.num_classes = num_classes
        self.base_weights = AdaptiveWeightModule(
            num_classes, 2, emg_acc, vis_acc, config.min_weight)
        self.normalizer = ConfusionAwareNormalization(num_classes, 2, config)

        class_weights = torch.ones(num_classes)
        for pc in PROBLEM_CLASSES:
            class_weights[pc] = config.problem_class_weight
        self.register_buffer('class_weights', class_weights)
        self.focal_loss = FocalLoss(gamma=config.focal_gamma, alpha=class_weights)

    def forward(self, emg_proba, vis_proba):
        weights = self.base_weights()
        prelim = weights[0] * emg_proba + weights[1] * vis_proba
        prelim_pred = torch.argmax(prelim).item()
        norm_emg = self.normalizer(emg_proba, weights[0], 0, prelim_pred)
        norm_vis = self.normalizer(vis_proba, weights[1], 1, prelim_pred)
        fused = weights[0] * norm_emg + weights[1] * norm_vis
        fused = fused / (fused.sum() + 1e-10)
        return fused

    def compute_loss(self, fused, label, device):
        return self.focal_loss(fused.unsqueeze(0),
                               torch.tensor([label], device=device))


class SOLIDNetWeightsDisagr(nn.Module):
    def __init__(self, num_classes, config, emg_acc, vis_acc):
        super().__init__()
        self.num_classes = num_classes
        self.base_weights = AdaptiveWeightModule(
            num_classes, 2, emg_acc, vis_acc, config.min_weight)
        self.disagreement = DisagreementResolver(num_classes)

        class_weights = torch.ones(num_classes)
        for pc in PROBLEM_CLASSES:
            class_weights[pc] = config.problem_class_weight
        self.register_buffer('class_weights', class_weights)
        self.focal_loss = FocalLoss(gamma=config.focal_gamma, alpha=class_weights)

    def forward(self, emg_proba, vis_proba):
        weights = self.base_weights()
        fused = weights[0] * emg_proba + weights[1] * vis_proba
        fused = fused / (fused.sum() + 1e-10)
        fused = self.disagreement(emg_proba, vis_proba, weights[0], weights[1], fused)
        return fused

    def compute_loss(self, fused, label, device):
        return self.focal_loss(fused.unsqueeze(0),
                               torch.tensor([label], device=device))


class FusionDataLoader:
    def __init__(self, config: FusionConfig):
        self.config = config

    def load_emg_init_data(self):
        print(f"\n[Data] Loading EMG init from: {EMG_INIT_DIR}")
        init_data = []
        if not EMG_INIT_DIR.exists():
            return init_data
        for csv_file in sorted(EMG_INIT_DIR.glob("Initialize*.csv")):
            print(f"  Loading: {csv_file.name}")
            windows = self._load_continuous_emg(csv_file)
            init_data.extend(windows)
            print(f"    -> {len(windows)} windows")
        cc = defaultdict(int)
        for _, l in init_data:
            cc[l] += 1
        print(f"  Total: {len(init_data)}")
        for gid in range(NUM_CLASSES):
            print(f"    G{gid + 1:2d} ({GESTURE_LABELS[gid]:<25}): {cc.get(gid, 0)}")
        return init_data

    def _load_continuous_emg(self, filepath):
        df = None
        for enc in ['utf-8', 'gbk', 'gb2312', 'latin1']:
            try:
                df = pd.read_csv(filepath, encoding=enc)
                break
            except:
                continue
        if df is None:
            return []
        emg_cols = []
        for pattern in ['ch{}_fil', 'ch{}_raw']:
            cols = []
            for i in range(1, 7):
                for c in df.columns:
                    if pattern.format(i) in c.lower() and 'env' not in c.lower():
                        cols.append(c)
                        break
            if len(cols) == 6:
                emg_cols = cols
                break
        if len(emg_cols) < 6:
            return []
        label_col = None
        for c in df.columns:
            if 'gesture_label' in c.lower() or c.lower() == 'label':
                label_col = c
                break
        if label_col is None:
            return []
        X_raw = safe_normalize(df[emg_cols].values.astype(np.float32))
        labels = df[label_col].values
        results = []
        ws, step = self.config.window_size, self.config.step_size
        for i in range(0, len(X_raw) - ws + 1, step):
            window_labels = labels[i:i + ws]
            lc = defaultdict(int)
            for lbl in window_labels:
                if pd.notna(lbl):
                    lc[str(lbl).strip()] += 1
            if not lc:
                continue
            majority = max(lc.keys(), key=lambda x: lc[x])
            gid = match_gesture_label(majority)
            if gid is not None:
                results.append((X_raw[i:i + ws], gid))
        return results

    def _load_single_emg(self, filepath):
        df = None
        for enc in ['utf-8', 'gbk', 'gb2312', 'latin1']:
            try:
                df = pd.read_csv(filepath, encoding=enc)
                break
            except:
                continue
        if df is None:
            return None
        emg_cols = []
        for pattern in ['ch{}_fil', 'ch{}_raw']:
            cols = []
            for i in range(1, 7):
                for c in df.columns:
                    if pattern.format(i) in c.lower() and 'env' not in c.lower():
                        cols.append(c)
                        break
            if len(cols) == 6:
                emg_cols = cols
                break
        if len(emg_cols) < 6:
            return None
        X_raw = safe_normalize(df[emg_cols].values.astype(np.float32))
        windows = []
        ws, step = self.config.window_size, self.config.step_size
        for i in range(0, len(X_raw) - ws + 1, step):
            windows.append(X_raw[i:i + ws])
        return windows if windows else None

    def load_test_data(self):
        print(f"\n[Data] Loading TEST data from: {TEST_DIR}")
        data = {}
        for gid in range(NUM_CLASSES):
            gesture_dir = TEST_DIR / str(gid + 1)
            if not gesture_dir.exists():
                continue
            data[gid] = {'emg': [], 'visual': []}
            emg_dir = gesture_dir / "emg"
            if emg_dir.exists():
                for csv_file in sorted(emg_dir.glob("*.csv")):
                    windows = self._load_single_emg(csv_file)
                    if windows:
                        data[gid]['emg'].extend(windows)
            pic_dir = gesture_dir / "pic"
            if pic_dir.exists():
                data[gid]['visual'] = sorted(
                    [str(f) for f in pic_dir.glob("*.jpg")]
                    + [str(f) for f in pic_dir.glob("*.png")]
                )
            ne = len(data[gid]['emg'])
            nv = len(data[gid]['visual'])
            flag = " [PROBLEM]" if gid in PROBLEM_CLASSES else ""
            print(f"  G{gid + 1:2d} ({GESTURE_LABELS[gid]:<25}): "
                  f"EMG={ne:>4}, Visual={nv:>4} frames{flag}")
        return data


def collect_predictions(emg_model, cast_net, test_data, config):
    samples = []
    seq_len = config.visual_seq_len
    for gid in tqdm(range(NUM_CLASSES), desc="Collecting predictions"):
        if gid not in test_data:
            continue
        emg_windows = test_data[gid]['emg']
        vis_frames = test_data[gid]['visual']
        if not emg_windows or not vis_frames:
            continue
        n_emg = len(emg_windows)
        n_vis_seqs = max(1, len(vis_frames) // seq_len)
        n_samples = min(n_emg, n_vis_seqs)
        for i in range(n_samples):
            emg_proba = emg_model.predict_proba(emg_windows[i])
            start = min(i * seq_len, max(0, len(vis_frames) - seq_len))
            frames = vis_frames[start:start + seq_len]
            vis_proba = cast_net.predict_proba(frames) if frames else \
                np.ones(NUM_CLASSES) / NUM_CLASSES
            samples.append({
                'label': gid,
                'emg_proba': emg_proba,
                'vis_proba': vis_proba,
            })
    return samples


def train_model(model, train_samples, config, device, model_name,
                phase1_epochs=60, phase2_epochs=90):
    model = model.to(device)

    augmented = train_samples.copy()
    for s in train_samples:
        label = s['label']
        if label in VISUAL_STRONG_CLASSES:
            augmented.append(s)
        if label in PROBLEM_CLASSES:
            for _ in range(2):
                augmented.append(s)
        if label in EMG_CONFUSION_PAIRS:
            augmented.append(s)

    print(f"  [{model_name}] Training: {len(augmented)} augmented samples")

    best_score = 0
    best_state = None

    optimizer = optim.Adam(model.parameters(), lr=config.phase1_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    for epoch in range(phase1_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        pcc, pct = defaultdict(int), defaultdict(int)
        np.random.shuffle(augmented)

        for sample in augmented:
            label = sample['label']
            emg_p = torch.tensor(sample['emg_proba'], device=device, dtype=torch.float32)
            vis_p = torch.tensor(sample['vis_proba'], device=device, dtype=torch.float32)
            fused = model(emg_p, vis_p)

            if hasattr(model, 'compute_loss'):
                loss = model.compute_loss(fused, label, device)
            else:
                loss = F.cross_entropy(fused.unsqueeze(0),
                                       torch.tensor([label], device=device))

            pred = torch.argmax(fused).item()
            if label in VISUAL_STRONG_CLASSES and pred != label:
                loss = loss * 2.0
            elif label in PROBLEM_CLASSES and pred != label:
                loss = loss * 1.5

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            if pred == label:
                correct += 1
                pcc[label] += 1
            total += 1
            pct[label] += 1

        scheduler.step()
        acc = correct / max(total, 1)

        pr = [pcc[pc] / pct[pc] for pc in PROBLEM_CLASSES if pct[pc] > 0]
        vsr = [pcc[vc] / pct[vc] for vc in VISUAL_STRONG_CLASSES if pct[vc] > 0]
        score = acc * 0.4 + (np.mean(pr) if pr else 0) * 0.3 + (np.mean(vsr) if vsr else 0) * 0.3

        if score > best_score:
            best_score = score
            best_state = deepcopy(model.state_dict())

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    P1 Epoch {epoch + 1:3d}/{phase1_epochs}: "
                  f"Loss={total_loss / max(total, 1):.4f}, Acc={acc * 100:.1f}%")

    if best_state:
        model.load_state_dict(best_state)
    print(f"  [{model_name}] Phase 1 Best: {best_score * 100:.1f}%")

    optimizer = optim.Adam(model.parameters(), lr=config.phase2_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase2_epochs)
    best_score2 = best_score

    for epoch in range(phase2_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        pcc, pct = defaultdict(int), defaultdict(int)
        np.random.shuffle(augmented)

        for sample in augmented:
            label = sample['label']
            emg_p = torch.tensor(sample['emg_proba'], device=device, dtype=torch.float32)
            vis_p = torch.tensor(sample['vis_proba'], device=device, dtype=torch.float32)
            fused = model(emg_p, vis_p)

            if hasattr(model, 'compute_loss'):
                loss = model.compute_loss(fused, label, device)
            else:
                loss = F.cross_entropy(fused.unsqueeze(0),
                                       torch.tensor([label], device=device))

            pred = torch.argmax(fused).item()
            if label in VISUAL_STRONG_CLASSES and pred != label:
                loss = loss * 1.8
            elif label in PROBLEM_CLASSES and pred != label:
                loss = loss * 1.5

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            if pred == label:
                correct += 1
                pcc[label] += 1
            total += 1
            pct[label] += 1

        scheduler.step()
        acc = correct / max(total, 1)

        pr = [pcc[pc] / pct[pc] for pc in PROBLEM_CLASSES if pct[pc] > 0]
        vsr = [pcc[vc] / pct[vc] for vc in VISUAL_STRONG_CLASSES if pct[vc] > 0]
        score = acc * 0.4 + (np.mean(pr) if pr else 0) * 0.3 + (np.mean(vsr) if vsr else 0) * 0.3

        if score > best_score2:
            best_score2 = score
            best_state = deepcopy(model.state_dict())

        if (epoch + 1) % 30 == 0 or epoch == 0:
            print(f"    P2 Epoch {epoch + 1:3d}/{phase2_epochs}: "
                  f"Loss={total_loss / max(total, 1):.4f}, Acc={acc * 100:.1f}%")

    if best_state:
        model.load_state_dict(best_state)
    print(f"  [{model_name}] Phase 2 Best: {best_score2 * 100:.1f}%")
    print(f"  [{model_name}] Overall Best: {max(best_score, best_score2) * 100:.1f}%")


def evaluate_model(model, test_data, emg_model, cast_net, config, device):
    model.eval()
    seq_len = config.visual_seq_len
    preds, labels = [], []

    for gid in range(NUM_CLASSES):
        if gid not in test_data:
            continue
        emg_windows = test_data[gid]['emg']
        vis_frames = test_data[gid]['visual']
        if not emg_windows or not vis_frames:
            continue
        n_emg = len(emg_windows)
        n_vis_seqs = max(1, len(vis_frames) // seq_len)
        n_samples = min(n_emg, n_vis_seqs)

        for i in range(n_samples):
            emg_proba = emg_model.predict_proba(emg_windows[i])
            start = min(i * seq_len, max(0, len(vis_frames) - seq_len))
            frames = vis_frames[start:start + seq_len]
            vis_proba = cast_net.predict_proba(frames) if frames else \
                np.ones(NUM_CLASSES) / NUM_CLASSES

            emg_p = torch.tensor(emg_proba, device=device, dtype=torch.float32)
            vis_p = torch.tensor(vis_proba, device=device, dtype=torch.float32)

            with torch.no_grad():
                fused = model(emg_p, vis_p)

            pred = torch.argmax(fused).item()
            preds.append(pred)
            labels.append(gid)

    preds = np.array(preds)
    labels = np.array(labels)
    return {
        'accuracy': float(accuracy_score(labels, preds)),
        'f1': float(f1_score(labels, preds, average='macro', zero_division=0)),
        'precision': float(precision_score(labels, preds, average='macro', zero_division=0)),
        'recall': float(recall_score(labels, preds, average='macro', zero_division=0)),
        'per_class_recall': recall_score(
            labels, preds, average=None,
            labels=list(range(NUM_CLASSES)), zero_division=0).tolist(),
        'per_class_precision': precision_score(
            labels, preds, average=None,
            labels=list(range(NUM_CLASSES)), zero_division=0).tolist(),
        'per_class_f1': f1_score(
            labels, preds, average=None,
            labels=list(range(NUM_CLASSES)), zero_division=0).tolist(),
        'confusion_matrix': confusion_matrix(
            labels, preds, labels=list(range(NUM_CLASSES))).tolist(),
        'n_samples': len(preds),
        'preds': preds.tolist(),
        'labels': labels.tolist(),
    }


def compute_per_class_acc(emg_model, cast_net, test_data, config):
    seq_len = config.visual_seq_len
    emg_cc, emg_ct = defaultdict(int), defaultdict(int)
    vis_cc, vis_ct = defaultdict(int), defaultdict(int)

    for gid in tqdm(range(NUM_CLASSES), desc="Per-class accuracy"):
        if gid not in test_data:
            continue
        for X in test_data[gid].get('emg', []):
            proba = emg_model.predict_proba(X)
            if int(np.argmax(proba)) == gid:
                emg_cc[gid] += 1
            emg_ct[gid] += 1

        vis_frames = test_data[gid].get('visual', [])
        if vis_frames:
            n_seqs = max(1, len(vis_frames) // seq_len)
            for i in range(n_seqs):
                start = min(i * seq_len, max(0, len(vis_frames) - seq_len))
                frames = vis_frames[start:start + seq_len]
                if frames:
                    proba = cast_net.predict_proba(frames)
                    if int(np.argmax(proba)) == gid:
                        vis_cc[gid] += 1
                    vis_ct[gid] += 1

    emg_acc = {g: emg_cc[g] / emg_ct[g] if emg_ct[g] > 0 else 0.0 for g in range(NUM_CLASSES)}
    vis_acc = {g: vis_cc[g] / vis_ct[g] if vis_ct[g] > 0 else 0.0 for g in range(NUM_CLASSES)}

    print(f"\n  Per-class accuracy:")
    print(f"    {'Gesture':<25} {'EMG':>8} {'VIS':>8}")
    print(f"    {'-' * 45}")
    for gid in range(NUM_CLASSES):
        print(f"    G{gid + 1:2d} {GESTURE_LABELS[gid]:<23}: "
              f"{emg_acc[gid] * 100:6.1f}% {vis_acc[gid] * 100:6.1f}%")

    return emg_acc, vis_acc


def plot_confusion_matrices_3in1(results_dict, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(27, 8))

    variant_order = [
        '(B) Weights Only',
        '(C) Weights+Norm',
        '(D) Weights+Disagr',
    ]
    variant_colors = ['Oranges', 'Blues', 'Purples']
    variant_titles = [
        'Ablation B: Weights Only\n(No Norm, No Disagr)',
        'Ablation C: Weights + Normalization\n(No Disagreement)',
        'Ablation D: Weights + Disagreement\n(No Normalization)',
    ]

    for ax_idx, (name, cmap, title) in enumerate(zip(variant_order, variant_colors, variant_titles)):
        ax = axes[ax_idx]
        if name not in results_dict:
            ax.set_visible(False)
            continue

        r = results_dict[name]
        cm = np.array(r['confusion_matrix'])
        cm_sum = cm.sum(axis=1, keepdims=True)
        cm_sum[cm_sum == 0] = 1
        cm_norm = cm.astype('float') / cm_sum

        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap=cmap, ax=ax,
                    xticklabels=GESTURE_NAMES_SHORT,
                    yticklabels=GESTURE_NAMES_SHORT,
                    vmin=0, vmax=1, annot_kws={'size': 7},
                    linewidths=0.5, linecolor='white',
                    cbar_kws={'shrink': 0.8, 'label': 'Recall'})

        acc = r['accuracy'] * 100
        f1 = r['f1'] * 100
        ax.set_title(f"{title}\nAcc={acc:.1f}%  F1={f1:.1f}%",
                     fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', rotation=0, labelsize=8)

    plt.suptitle('Ablation Study: Confusion Matrices of Three Variants',
                 fontsize=17, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Plot] {save_path.name}")


def plot_f1_recall_comparison(results_dict, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(18, 14))

    variant_order = [
        '(B) Weights Only',
        '(C) Weights+Norm',
        '(D) Weights+Disagr',
    ]
    variant_colors = ['#e74c3c', '#3498db', '#9b59b6']
    variant_labels = [
        'B: Weights Only',
        'C: Weights+Norm',
        'D: Weights+Disagr',
    ]

    available = []
    for name, color, label in zip(variant_order, variant_colors, variant_labels):
        if name in results_dict:
            available.append((name, color, label, results_dict[name]))

    if not available:
        plt.close()
        return

    x = np.arange(NUM_CLASSES)
    n = len(available)
    w = 0.8 / n

    ax = axes[0]
    for i, (name, color, label, r) in enumerate(available):
        f1_vals = [v * 100 for v in r['per_class_f1']]
        bars = ax.bar(x + i * w - (n - 1) * w / 2, f1_vals, w,
                      label=f"{label} (avg={r['f1'] * 100:.1f}%)",
                      color=color, alpha=0.85, edgecolor='white', linewidth=0.8)
        for bar, val in zip(bars, f1_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f'{val:.0f}', ha='center', va='bottom', fontsize=6,
                        fontweight='bold', color=color)

    ax.set_ylabel('F1 Score (%)', fontsize=13, fontweight='bold')
    ax.set_title('Per-class F1 Score: Ablation Variants', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'G{i + 1}\n{s}' for i, s in enumerate(GESTURE_NAMES_SHORT)],
                       fontsize=9)
    ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 115)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=50, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)

    for pc in PROBLEM_CLASSES:
        ax.axvspan(pc - 0.45, pc + 0.45, alpha=0.08, color='red')
        ax.text(pc, 112, '!', ha='center', fontsize=10, color='red')

    ax = axes[1]
    for i, (name, color, label, r) in enumerate(available):
        recall_vals = [v * 100 for v in r['per_class_recall']]
        bars = ax.bar(x + i * w - (n - 1) * w / 2, recall_vals, w,
                      label=f"{label} (avg={r['recall'] * 100:.1f}%)",
                      color=color, alpha=0.85, edgecolor='white', linewidth=0.8)
        for bar, val in zip(bars, recall_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f'{val:.0f}', ha='center', va='bottom', fontsize=6,
                        fontweight='bold', color=color)

    ax.set_ylabel('Recall (%)', fontsize=13, fontweight='bold')
    ax.set_title('Per-class Recall: Ablation Variants', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'G{i + 1}\n{s}' for i, s in enumerate(GESTURE_NAMES_SHORT)],
                       fontsize=9)
    ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 115)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=50, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)

    for pc in PROBLEM_CLASSES:
        ax.axvspan(pc - 0.45, pc + 0.45, alpha=0.08, color='red')
        ax.text(pc, 112, '!', ha='center', fontsize=10, color='red')

    summary_lines = []
    for name, color, label, r in available:
        summary_lines.append(
            f"{label}: Acc={r['accuracy'] * 100:.1f}%, "
            f"F1={r['f1'] * 100:.1f}%, "
            f"Prec={r['precision'] * 100:.1f}%, "
            f"Recall={r['recall'] * 100:.1f}%"
        )
    summary_text = "  |  ".join(summary_lines)
    fig.text(0.5, -0.01, summary_text, ha='center', fontsize=10,
             style='italic', color='#555555',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Plot] {save_path.name}")


def plot_overall_comparison(results_dict, save_path):
    fig, ax = plt.subplots(figsize=(12, 7))

    variant_order = [
        '(B) Weights Only',
        '(C) Weights+Norm',
        '(D) Weights+Disagr',
    ]
    variant_colors = ['#e74c3c', '#3498db', '#9b59b6']
    variant_labels = ['B: Weights Only', 'C: Weights+Norm', 'D: Weights+Disagr']

    metrics_names = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    x = np.arange(len(metrics_names))
    n = len(variant_order)
    w = 0.25

    for i, (name, color, label) in enumerate(zip(variant_order, variant_colors, variant_labels)):
        if name not in results_dict:
            continue
        r = results_dict[name]
        vals = [r['accuracy'] * 100, r['f1'] * 100,
                r['precision'] * 100, r['recall'] * 100]
        bars = ax.bar(x + i * w - (n - 1) * w / 2, vals, w,
                      label=label, color=color, alpha=0.85,
                      edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Score (%)', fontsize=13)
    ax.set_title('Ablation Study: Overall Metrics Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=12)
    ax.legend(fontsize=12, loc='lower right')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Plot] {save_path.name}")


def main():
    print("=" * 70)
    print("SOLID-Net V9.6b: Ablation - Train, Save Models & Generate Plots")
    print("=" * 70)
    print(f"  Models to train & save:")
    print(f"    (B) Weights Only:    BaseWeights only")
    print(f"    (C) Weights+Norm:    BaseWeights + Normalization")
    print(f"    (D) Weights+Disagr:  BaseWeights + Disagreement")
    print(f"  Output: {ABLATION_DIR}")
    print("=" * 70)

    config = FusionConfig()
    device = select_gpu()
    config.device = device
    print(f"\n[Device] {device}")

    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  [Dir] Created: {ABLATION_DIR}")

    print(f"\n{'=' * 70}")
    print("Loading Modality Models")
    print(f"{'=' * 70}")

    emg_model = SCMSFE(config)
    emg_model.load(EMG_MODEL_DIR)

    print(f"\n  [CAST-Net] Model path: {VISUAL_MODEL_PATH}")
    cast_net = CASTNet(config)
    cast_net.load(VISUAL_MODEL_PATH)

    if not emg_model.is_loaded or not cast_net.is_loaded:
        print("ERROR: Models not loaded!")
        return

    data_loader = FusionDataLoader(config)
    _ = data_loader.load_emg_init_data()
    test_data = data_loader.load_test_data()

    print(f"\n{'=' * 70}")
    print("Computing Per-class Single-Modality Accuracy")
    print(f"{'=' * 70}")
    emg_acc, vis_acc = compute_per_class_acc(emg_model, cast_net, test_data, config)

    train_samples = collect_predictions(emg_model, cast_net, test_data, config)
    print(f"\n  Collected {len(train_samples)} training samples")

    ablation_variants = {
        '(B) Weights Only': SOLIDNetWeightsOnly(NUM_CLASSES, config, emg_acc, vis_acc),
        '(C) Weights+Norm': SOLIDNetWeightsNorm(NUM_CLASSES, config, emg_acc, vis_acc),
        '(D) Weights+Disagr': SOLIDNetWeightsDisagr(NUM_CLASSES, config, emg_acc, vis_acc),
    }

    model_filenames = {
        '(B) Weights Only': 'ablation_B_weights_only.pth',
        '(C) Weights+Norm': 'ablation_C_weights_norm.pth',
        '(D) Weights+Disagr': 'ablation_D_weights_disagr.pth',
    }

    results_dict = {}

    for name, model in ablation_variants.items():
        print(f"\n{'=' * 70}")
        print(f"Training & Evaluating: {name}")
        print(f"{'=' * 70}")

        set_seed(42)
        train_model(model, train_samples, config, device, name)

        if hasattr(model, 'base_weights'):
            weights = model.base_weights.get_weights_numpy()
            print(f"\n  Learned Weights:")
            for gid in range(NUM_CLASSES):
                flag = ""
                if gid in PROBLEM_CLASSES:
                    flag = " [PROB]"
                elif gid in VISUAL_STRONG_CLASSES:
                    flag = " [VIS]"
                elif gid in EMG_STRONG_CLASSES:
                    flag = " [EMG]"
                print(f"    G{gid + 1:2d} {GESTURE_LABELS[gid]:<25}: "
                      f"EMG={weights[0, gid]:.2f}, VIS={weights[1, gid]:.2f}{flag}")

        metrics = evaluate_model(model, test_data, emg_model, cast_net, config, device)
        metrics['name'] = name
        results_dict[name] = metrics

        print(f"\n  * {name}:")
        print(f"    Accuracy:  {metrics['accuracy'] * 100:.1f}%")
        print(f"    F1 Score:  {metrics['f1'] * 100:.1f}%")
        print(f"    Precision: {metrics['precision'] * 100:.1f}%")
        print(f"    Recall:    {metrics['recall'] * 100:.1f}%")

        save_path = ABLATION_DIR / model_filenames[name]
        save_data = {
            'model_state_dict': model.state_dict(),
            'model_class': name,
            'config': {
                'num_classes': config.num_classes,
                'min_weight': config.min_weight,
                'focal_gamma': config.focal_gamma,
                'problem_class_weight': config.problem_class_weight,
                'confusion_penalty': config.confusion_penalty,
            },
            'emg_per_class_acc': emg_acc,
            'vis_per_class_acc': vis_acc,
            'metrics': {k: v for k, v in metrics.items()
                       if k not in ('preds', 'labels', 'confusion_matrix')},
            'weights': model.base_weights.get_weights_numpy().tolist()
                       if hasattr(model, 'base_weights') else None,
            'timestamp': datetime.now().isoformat(),
        }
        torch.save(save_data, save_path)
        print(f"  [Save] Model -> {save_path}")

    print(f"\n{'=' * 70}")
    print("ABLATION RESULTS SUMMARY")
    print(f"{'=' * 70}")

    print(f"\n  {'Model':<25} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}")
    print(f"  {'-' * 55}")
    for name in ['(B) Weights Only', '(C) Weights+Norm', '(D) Weights+Disagr']:
        r = results_dict[name]
        print(f"  {name:<25} "
              f"{r['accuracy'] * 100:>7.1f}% "
              f"{r['f1'] * 100:>7.1f}% "
              f"{r['precision'] * 100:>7.1f}% "
              f"{r['recall'] * 100:>7.1f}%")

    print(f"\n  Per-class Recall:")
    header = f"    {'Gesture':<15}"
    for name in ['(B) Weights Only', '(C) Weights+Norm', '(D) Weights+Disagr']:
        header += f" {'B' if 'Only' in name else 'C' if 'Norm' in name else 'D':>10}"
    print(header)
    print(f"    {'-' * 47}")
    for gid in range(NUM_CLASSES):
        row = f"    G{gid + 1:2d} {GESTURE_NAMES_SHORT[gid]:<10}"
        for name in ['(B) Weights Only', '(C) Weights+Norm', '(D) Weights+Disagr']:
            val = results_dict[name]['per_class_recall'][gid] * 100
            row += f" {val:>9.1f}%"
        print(row)

    b_acc = results_dict['(B) Weights Only']['accuracy'] * 100
    c_acc = results_dict['(C) Weights+Norm']['accuracy'] * 100
    d_acc = results_dict['(D) Weights+Disagr']['accuracy'] * 100

    print(f"\n  {'=' * 50}")
    print(f"  Component Contribution (relative to Weights Only)")
    print(f"  {'=' * 50}")
    print(f"  Weights Only (B):              {b_acc:.1f}%")
    print(f"  + Normalization (C):           {c_acc:.1f}%  ({c_acc - b_acc:+.1f}%)")
    print(f"  + Disagreement (D):            {d_acc:.1f}%  ({d_acc - b_acc:+.1f}%)")
    print(f"  {'=' * 50}")
    print(f"  Normalization contributes:     {c_acc - b_acc:+.1f}% accuracy")
    print(f"  Disagreement contributes:      {d_acc - b_acc:+.1f}% accuracy")
    print(f"  {'=' * 50}")

    print(f"\n  Generating plots...")

    plot_confusion_matrices_3in1(
        results_dict,
        ABLATION_DIR / 'confusion_matrices.png'
    )

    plot_f1_recall_comparison(
        results_dict,
        ABLATION_DIR / 'f1_recall_comparison.png'
    )

    plot_overall_comparison(
        results_dict,
        ABLATION_DIR / 'overall_comparison.png'
    )

    meta = {
        'experiment': 'Ablation Study - Fusion Components',
        'timestamp': datetime.now().isoformat(),
        'variants': {},
    }
    for name in ['(B) Weights Only', '(C) Weights+Norm', '(D) Weights+Disagr']:
        r = results_dict[name]
        meta['variants'][name] = {
            'model_file': model_filenames[name],
            'accuracy': r['accuracy'],
            'f1': r['f1'],
            'precision': r['precision'],
            'recall': r['recall'],
            'per_class_recall': r['per_class_recall'],
            'per_class_f1': r['per_class_f1'],
            'per_class_precision': r['per_class_precision'],
        }
    meta['contribution'] = {
        'normalization_delta': c_acc - b_acc,
        'disagreement_delta': d_acc - b_acc,
    }
    with open(ABLATION_DIR / 'ablation_info.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  [Save] ablation_info.json")

    print(f"\n{'=' * 70}")
    print("Ablation Study Complete!")
    print(f"{'=' * 70}")
    print(f"\n  Saved files in: {ABLATION_DIR}")
    print(f"    Models:")
    for name in ['(B) Weights Only', '(C) Weights+Norm', '(D) Weights+Disagr']:
        r = results_dict[name]
        print(f"      {model_filenames[name]:<40} Acc={r['accuracy'] * 100:.1f}%")
    print(f"    Plots:")
    print(f"      confusion_matrices.png      (3 confusion matrices in 1 figure)")
    print(f"      f1_recall_comparison.png     (per-class F1 & recall bars)")
    print(f"      overall_comparison.png       (overall metrics comparison)")
    print(f"    Data:")
    print(f"      ablation_info.json           (metadata & metrics)")
    print(f"\n  Key Finding:")
    print(f"    Normalization contributes {c_acc - b_acc:+.1f}% accuracy improvement")
    print(f"    Disagreement contributes  {d_acc - b_acc:+.1f}% accuracy improvement")


if __name__ == '__main__':
    main()
