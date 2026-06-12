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
from abc import ABC, abstractmethod
import threading
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torchvision import models
from scipy.fft import fft
from scipy.special import softmax
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_PATH = Path(os.getenv("FUSION_DATASET_ROOT", PROJECT_ROOT / "Dataset")).expanduser().resolve()
HAND_DETECT_MODEL = ROOT_PATH / "models" / "best.pt"
VISUAL_MODEL_PATH = ROOT_PATH / "models" / "mvtf_visual.pth"
EMG_MODEL_DIR = ROOT_PATH / "models" / "EMG"
TEST_DIR = ROOT_PATH / "test"
EMG_INIT_DIR = TEST_DIR / "initialize"
OUTPUT_DIR = ROOT_PATH / "models" / "Fusion" / "V8"

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

PROBLEM_CLASSES = {
    0: 'Fist Clench',
    1: 'Finger Extension',
    2: 'Wrist Rotation',
    12: 'Piano Tap',
}

EMG_CONFUSION_PAIRS = {
    0: [1, 4],
    4: [3, 0],
    5: [4, 6],
    6: [5],
    9: [5, 8],
    11: [5],
    12: [5],
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
    except Exception:
        pass
    return 'cuda:1' if torch.cuda.is_available() else 'cpu'


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
    fusion_lr: float = 0.008
    fusion_epochs: int = 150
    phase1_epochs: int = 60
    phase2_epochs: int = 90
    phase1_lr: float = 0.015
    phase2_lr: float = 0.005
    power_save_threshold: int = 5
    power_save_step_multiplier: int = 3
    focal_gamma: float = 2.0
    problem_class_weight: float = 3.0
    weak_class_weight: float = 2.5
    min_weight: float = 0.10
    confusion_penalty: float = 0.5
    trust_momentum: float = 0.3
    train_fraction: float = 0.7
    random_seed: int = 42


class BaseModalityModel(ABC):
    def __init__(self, model_name: str, num_classes: int):
        self.model_name = model_name
        self.num_classes = num_classes
        self.is_loaded = False

    @abstractmethod
    def predict_proba(self, data: Any) -> np.ndarray:
        pass

    @abstractmethod
    def load(self, path: Path) -> bool:
        pass

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
            feats.append(np.sum(np.sign(ch[:, 1:] + eps) != np.sign(ch[:, :-1] + eps), axis=1) / 2)
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


class SCMSFE(BaseModalityModel):
    def __init__(self, config: FusionConfig):
        super().__init__("SC-MSFE", config.num_classes)
        self.config = config
        self.extractor = SCMSFEFeatureExtractor(config.emg_channels, config.window_size)
        self.model = None
        self.scaler = None
        self.expected_features = None
        self.missing_classes: Set[int] = set()

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
                print(f"  [{self.model_name}] Loaded complete model (dim={self.expected_features})")
                return True
            scaler_path = model_dir / 'emg_scaler.pkl'
            model_path = model_dir / 'emg_ensemble.pkl'
            if scaler_path.exists() and model_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                if hasattr(self.scaler, 'n_features_in_'):
                    self.expected_features = self.scaler.n_features_in_
                self.is_loaded = True
                print(f"  [{self.model_name}] Loaded from scaler+ensemble (dim={self.expected_features})")
                return True
        except Exception as e:
            print(f"  [{self.model_name}] Load failed: {e}")
        return False

    def verify(self, init_data=None):
        if not self.is_loaded or not init_data:
            return {}
        print(f"\n[{self.model_name}] Verifying with {len(init_data)} samples...")
        correct, total = 0, 0
        cc, ct = defaultdict(int), defaultdict(int)
        for window, label in init_data:
            proba = self.predict_proba(window)
            if np.argmax(proba) == label:
                correct += 1
                cc[label] += 1
            total += 1
            ct[label] += 1
        per_class_acc = {}
        print(f"  Init accuracy: {correct / max(total, 1) * 100:.1f}%")
        for gid in range(self.config.num_classes):
            if ct[gid] > 0:
                acc = cc[gid] / ct[gid]
                per_class_acc[gid] = acc
                print(f"    G{gid + 1:2d}: {acc * 100:.1f}% ({ct[gid]})")
        return per_class_acc

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
        except Exception:
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
        self._cache_lock = threading.Lock()
        if YOLO_AVAILABLE and model_path.exists():
            try:
                self.model = YOLO(str(model_path))
                self.is_loaded = True
                print(f"  [HandDetector] Loaded: {model_path.name}")
            except Exception as e:
                print(f"  [HandDetector] Failed: {e}")

    def detect_and_crop(self, image_path, output_size=224, conf=0.25, padding=0.15):
        with self._cache_lock:
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
            except Exception:
                pass
        if result_img is None:
            size = min(h, w)
            top, left = (h - size) // 2, (w - size) // 2
            result_img = cv2.resize(img_rgb[top:top + size, left:left + size],
                                    (output_size, output_size))
        with self._cache_lock:
            if len(self._cache) < 8000:
                self._cache[image_path] = result_img
        return result_img


class CASTNet(BaseModalityModel):
    def __init__(self, config: FusionConfig):
        super().__init__("CAST-Net", config.num_classes)
        self.config = config
        self.device = config.device
        self.hand_detector = HandDetector(HAND_DETECT_MODEL, config.device)
        self.model = None
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load(self, model_path: Path) -> bool:
        if not model_path.exists():
            print(f"  [{self.model_name}] File not found: {model_path}")
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
            self.model = self.model.to(self.device).eval()
            self.is_loaded = True
            print(f"  [{self.model_name}] Loaded from {model_path.name}")
            return True
        except Exception as e:
            print(f"  [{self.model_name}] Load failed: {e}")
            return False

    def predict_proba(self, data: List[str]) -> np.ndarray:
        if not self.is_loaded:
            return np.ones(self.num_classes) / self.num_classes
        imgs = []
        for path in data:
            cropped = self.hand_detector.detect_and_crop(
                path, self.config.output_size,
                self.config.hand_conf, self.config.hand_padding
            )
            if cropped is not None:
                imgs.append(self.transform(cropped))
        if not imgs:
            return np.ones(self.num_classes) / self.num_classes
        seq_len = self.config.visual_seq_len
        if len(imgs) >= seq_len:
            indices = np.linspace(0, len(imgs) - 1, seq_len).astype(int)
            imgs = [imgs[i] for i in indices]
        else:
            while len(imgs) < seq_len:
                imgs.append(imgs[-1])
        with torch.no_grad():
            batch = torch.stack(imgs).unsqueeze(0).to(self.device)
            output = self.model(batch)
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


class StackingMetaLearner(nn.Module):
    def __init__(self, num_modalities, num_classes, hidden_dim=64):
        super().__init__()
        input_dim = num_modalities * num_classes
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, modality_probas):
        concat = torch.cat(modality_probas, dim=-1)
        return self.net(concat)


class MixtureOfExpertsGate(nn.Module):
    def __init__(self, num_modalities, num_classes, hidden_dim=32):
        super().__init__()
        self.num_modalities = num_modalities
        self.num_classes = num_classes
        self.gate = nn.Sequential(
            nn.Linear(num_modalities * num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_modalities * num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, modality_probas):
        concat = torch.cat(modality_probas, dim=-1)
        weights = self.gate(concat).view(-1, self.num_modalities, self.num_classes)
        weighted = torch.stack(modality_probas, dim=1) * weights
        return weighted.sum(dim=1)


class TemperatureScalingCalibrator:
    def __init__(self, num_modalities):
        self.temperatures = [nn.Parameter(torch.ones(1)) for _ in range(num_modalities)]

    def calibrate_proba(self, proba, modality_idx):
        temp = F.softplus(self.temperatures[modality_idx]) + 0.1
        log_proba = torch.log(proba + 1e-10)
        calibrated = F.softmax(log_proba / temp, dim=-1)
        return calibrated

    def parameters(self):
        return self.temperatures


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
            print("\n  [Adaptive Weights] Initializing from per-class accuracy:")
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

                emg_tag = "*" if emg_acc > vis_acc else " "
                vis_tag = "*" if vis_acc > emg_acc else " "
                print(f"    G{c + 1:2d} {GESTURE_LABELS[c]:<25}: "
                      f"EMG_acc={emg_acc * 100:5.1f}%{emg_tag} "
                      f"VIS_acc={vis_acc * 100:5.1f}%{vis_tag} "
                      f"-> EMG_w={emg_w:.2f}, VIS_w={vis_w:.2f}")

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

        emg_conf = emg_proba[emg_pred].item()
        vis_conf = vis_proba[vis_pred].item()

        adjusted = fused_proba.clone()

        if vis_pred in VISUAL_STRONG_CLASSES and vis_conf > 0.3:
            boost = 0.15 + F.sigmoid(self.arbitration_bias[vis_pred]).item() * 0.15
            adjusted[vis_pred] = adjusted[vis_pred] + boost
            if emg_pred in EMG_CONFUSION_PAIRS:
                adjusted[emg_pred] = adjusted[emg_pred] * 0.7

        if emg_pred in EMG_CONFUSION_PAIRS:
            confused_targets = EMG_CONFUSION_PAIRS[emg_pred]
            if vis_pred in confused_targets or vis_pred == emg_pred:
                pass
            else:
                adjusted[emg_pred] = adjusted[emg_pred] * 0.75
                adjusted[vis_pred] = adjusted[vis_pred] * 1.2

        adjusted = adjusted / (adjusted.sum() + 1e-10)
        return adjusted


class SOLIDNet(nn.Module):
    def __init__(self, num_classes, modality_names, device='cuda', config=None,
                 emg_per_class_acc=None, vis_per_class_acc=None):
        super().__init__()
        self.num_classes = num_classes
        self.modality_names = modality_names
        self.num_modalities = len(modality_names)
        self.device = device
        self.config = config or FusionConfig()

        self.base_weights = AdaptiveWeightModule(
            num_classes, self.num_modalities,
            emg_per_class_acc=emg_per_class_acc,
            vis_per_class_acc=vis_per_class_acc,
            min_weight=self.config.min_weight
        )
        self.normalizer = ConfusionAwareNormalization(
            num_classes, self.num_modalities, self.config)
        self.disagreement = DisagreementResolver(num_classes)

        self.stacking = StackingMetaLearner(self.num_modalities, num_classes)
        self.moe_gate = MixtureOfExpertsGate(self.num_modalities, num_classes)
        self.calibrator = TemperatureScalingCalibrator(self.num_modalities)

        self.modality_to_idx = {name: i for i, name in enumerate(modality_names)}

        class_weights = torch.ones(num_classes)
        for pc in PROBLEM_CLASSES:
            class_weights[pc] = self.config.problem_class_weight
        self.register_buffer('class_weights', class_weights)
        self.focal_loss = FocalLoss(gamma=self.config.focal_gamma, alpha=class_weights)

    def forward(self, modality_probas, fusion_method='adaptive'):
        weights = self.base_weights()
        fused = torch.zeros(self.num_classes, device=self.device)

        modal_proba_tensors = {}
        for name, proba in modality_probas.items():
            if name not in self.modality_to_idx:
                continue
            pt = proba if isinstance(proba, torch.Tensor) else \
                torch.tensor(proba, device=self.device, dtype=torch.float32)
            modal_proba_tensors[name] = pt

        if fusion_method == 'stacking' and len(modal_proba_tensors) >= 2:
            proba_list = [
                modal_proba_tensors[name]
                for name in self.modality_names
                if name in modal_proba_tensors
            ]
            if len(proba_list) == self.num_modalities:
                logits = self.stacking(proba_list)
                fused = F.softmax(logits, dim=-1)
                sorted_fused = torch.sort(fused, descending=True)[0]
                result = {
                    'fused_proba': fused,
                    'prediction': torch.argmax(fused).item(),
                    'confidence': (sorted_fused[0] - sorted_fused[1]).item(),
                    'weights': weights.detach(),
                    'method': 'stacking'
                }
                return result

        elif fusion_method == 'moe' and len(modal_proba_tensors) >= 2:
            proba_list = [
                modal_proba_tensors[name]
                for name in self.modality_names
                if name in modal_proba_tensors
            ]
            if len(proba_list) == self.num_modalities:
                fused = self.moe_gate(proba_list)
                sorted_fused = torch.sort(fused, descending=True)[0]
                result = {
                    'fused_proba': fused,
                    'prediction': torch.argmax(fused).item(),
                    'confidence': (sorted_fused[0] - sorted_fused[1]).item(),
                    'weights': weights.detach(),
                    'method': 'moe'
                }
                return result

        elif fusion_method == 'calibrated':
            for name, pt in modal_proba_tensors.items():
                idx = self.modality_to_idx[name]
                calibrated = self.calibrator.calibrate_proba(pt, idx)
                fused += weights[idx] * calibrated
            fused = fused / (fused.sum() + 1e-10)

        else:
            prelim = torch.zeros(self.num_classes, device=self.device)
            for name, pt in modal_proba_tensors.items():
                idx = self.modality_to_idx[name]
                prelim += weights[idx] * pt
            prelim_pred = torch.argmax(prelim).item()

            for name, pt in modal_proba_tensors.items():
                idx = self.modality_to_idx[name]
                norm_proba = self.normalizer(pt, weights[idx], idx, prelim_pred)
                fused += weights[idx] * norm_proba

            fused = fused / (fused.sum() + 1e-10)

            if 'SC-MSFE' in modal_proba_tensors and 'CAST-Net' in modal_proba_tensors:
                emg_idx = self.modality_to_idx['SC-MSFE']
                vis_idx = self.modality_to_idx['CAST-Net']
                fused = self.disagreement(
                    modal_proba_tensors['SC-MSFE'],
                    modal_proba_tensors['CAST-Net'],
                    weights[emg_idx],
                    weights[vis_idx],
                    fused
                )

        sorted_p, _ = torch.sort(fused, descending=True)
        confidence = (sorted_p[0] - sorted_p[1]).item()

        return {
            'fused_proba': fused,
            'prediction': torch.argmax(fused).item(),
            'confidence': confidence,
            'weights': weights.detach(),
            'method': fusion_method
        }

    def compute_loss(self, fused_proba, label):
        return self.focal_loss(fused_proba.unsqueeze(0),
                               torch.tensor([label], device=self.device))

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
            except Exception:
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

    def _load_single_emg(self, filepath):
        df = None
        for enc in ['utf-8', 'gbk', 'gb2312', 'latin1']:
            try:
                df = pd.read_csv(filepath, encoding=enc)
                break
            except Exception:
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


class FusionSystemManager:
    def __init__(self, config: FusionConfig):
        self.config = config
        self.device = config.device
        self.modality_models: Dict[str, BaseModalityModel] = {}
        self.fusion_model: Optional[SOLIDNet] = None
        self.emg_per_class_acc: Dict[int, float] = {}
        self.vis_per_class_acc: Dict[int, float] = {}
        self.training_stats = {'losses': [], 'accuracies': [], 'confidences': [],
                               'problem_recalls': []}

        self.power_save_mode = False
        self.consecutive_same_results = 0
        self.last_prediction = None

    def register_modality(self, name, model):
        self.modality_models[name] = model
        print(f"  [HotPlug] Registered: {name} ({'OK' if model.is_loaded else 'FAILED'})")

    def compute_visual_accuracy(self, data):
        cast_net = self.modality_models.get('CAST-Net')
        if not cast_net or not cast_net.is_loaded:
            return {}

        print(f"\n[CAST-Net] Pre-evaluating visual accuracy for weight init...")
        seq_len = self.config.visual_seq_len
        cc, ct = defaultdict(int), defaultdict(int)

        for gid in tqdm(range(NUM_CLASSES), desc="  Visual pre-eval"):
            if gid not in data:
                continue
            vis_frames = data[gid]['visual']
            if not vis_frames:
                continue
            n_seqs = max(1, len(vis_frames) // seq_len)
            for i in range(n_seqs):
                start = min(i * seq_len, max(0, len(vis_frames) - seq_len))
                frames = vis_frames[start:start + seq_len]
                if frames:
                    proba = cast_net.predict_proba(frames)
                    pred = int(np.argmax(proba))
                    if pred == gid:
                        cc[gid] += 1
                    ct[gid] += 1

        per_class_acc = {}
        print(f"  Visual per-class accuracy:")
        for gid in range(NUM_CLASSES):
            if ct[gid] > 0:
                acc = cc[gid] / ct[gid]
            else:
                acc = 0.0
            per_class_acc[gid] = acc
            tag = " [high]" if acc >= 0.8 else ""
            print(f"    G{gid + 1:2d} {GESTURE_LABELS[gid]:<25}: "
                  f"{acc * 100:5.1f}% ({cc[gid]}/{ct[gid]}){tag}")

        overall = sum(cc.values()) / max(sum(ct.values()), 1)
        print(f"  Overall visual: {overall * 100:.1f}%")
        return per_class_acc

    def compute_emg_accuracy(self, data):
        emg_model = self.modality_models.get('SC-MSFE')
        if not emg_model or not emg_model.is_loaded:
            return {}

        print(f"\n[SC-MSFE] Pre-evaluating EMG accuracy for weight init...")
        cc, ct = defaultdict(int), defaultdict(int)

        for gid in tqdm(range(NUM_CLASSES), desc="  EMG pre-eval"):
            if gid not in data:
                continue
            windows = data[gid]['emg']
            for X in windows:
                proba = emg_model.predict_proba(X)
                pred = int(np.argmax(proba))
                if pred == gid:
                    cc[gid] += 1
                ct[gid] += 1

        per_class_acc = {}
        print(f"  EMG per-class accuracy:")
        for gid in range(NUM_CLASSES):
            if ct[gid] > 0:
                acc = cc[gid] / ct[gid]
            else:
                acc = 0.0
            per_class_acc[gid] = acc
            tag = " [high]" if acc >= 0.8 else ""
            print(f"    G{gid + 1:2d} {GESTURE_LABELS[gid]:<25}: "
                  f"{acc * 100:5.1f}% ({cc[gid]}/{ct[gid]}){tag}")

        overall = sum(cc.values()) / max(sum(ct.values()), 1)
        print(f"  Overall EMG: {overall * 100:.1f}%")
        return per_class_acc

    def init_fusion_model(self):
        names = list(self.modality_models.keys())
        self.fusion_model = SOLIDNet(
            num_classes=self.config.num_classes,
            modality_names=names,
            device=self.device,
            config=self.config,
            emg_per_class_acc=self.emg_per_class_acc if self.emg_per_class_acc else None,
            vis_per_class_acc=self.vis_per_class_acc if self.vis_per_class_acc else None,
        ).to(self.device)
        print(f"\n  [SOLID-Net] Initialized with modalities: {names}")

    def predict(self, modality_data):
        modality_probas = {}
        for name, data in modality_data.items():
            if name in self.modality_models and self.modality_models[name].is_loaded:
                proba = self.modality_models[name].predict_proba(data)
                modality_probas[name] = torch.tensor(proba, device=self.device,
                                                     dtype=torch.float32)
        if not modality_probas:
            return {'prediction': -1, 'confidence': 0, 'power_save_mode': False,
                    'step_multiplier': 1}

        self.fusion_model.eval()
        with torch.no_grad():
            result = self.fusion_model(modality_probas)

        pred = result['prediction']
        if self.last_prediction == pred:
            self.consecutive_same_results += 1
        else:
            self.consecutive_same_results = 0
        self.last_prediction = pred

        ps_threshold = self.config.power_save_threshold
        if self.consecutive_same_results >= ps_threshold:
            result['power_save_mode'] = True
            result['step_multiplier'] = self.config.power_save_step_multiplier
        else:
            result['power_save_mode'] = False
            result['step_multiplier'] = 1
        return result

    def train(self, train_samples, epochs=150, lr=0.008):
        if not train_samples:
            print("[ERROR] No training samples!")
            return

        label_counts = Counter([s['label'] for s in train_samples])
        has_emg = any('emg_proba' in s for s in train_samples)
        has_vis = any('vis_proba' in s for s in train_samples)
        n_both = sum(1 for s in train_samples if 'emg_proba' in s and 'vis_proba' in s)

        print(f"\n{'=' * 70}")
        print(f"Training SOLID-Net (Two-Phase)")
        print(f"  Samples: {len(train_samples)} (dual-modal: {n_both})")
        print(f"  Has EMG: {has_emg}, Has Visual: {has_vis}")
        print(f"  Phase 1: {self.config.phase1_epochs} epochs @ lr={self.config.phase1_lr}")
        print(f"  Phase 2: {self.config.phase2_epochs} epochs @ lr={self.config.phase2_lr}")
        print(f"{'=' * 70}")

        augmented = train_samples.copy()
        weak_fusion_classes = set()
        for s in train_samples:
            label = s['label']
            if label in VISUAL_STRONG_CLASSES:
                augmented.append(s)
                weak_fusion_classes.add(label)
            if label in PROBLEM_CLASSES:
                for _ in range(2):
                    augmented.append(s)
            if label in EMG_CONFUSION_PAIRS:
                augmented.append(s)

        print(f"  Augmented: {len(augmented)} samples")
        if weak_fusion_classes:
            print(f"  Visual-strong boosted: {sorted(weak_fusion_classes)}")

        best_score = 0
        best_state = None

        print(f"\n  --- Phase 1: Weight Discovery ---")
        optimizer = optim.Adam(self.fusion_model.parameters(), lr=self.config.phase1_lr)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

        for epoch in range(self.config.phase1_epochs):
            self.fusion_model.train()
            total_loss, correct, total = 0, 0, 0
            confidences = []
            pcc = defaultdict(int)
            pct = defaultdict(int)
            np.random.shuffle(augmented)

            for sample in augmented:
                label = sample['label']
                mp = {}
                if 'emg_proba' in sample:
                    mp['SC-MSFE'] = torch.tensor(sample['emg_proba'],
                                                 device=self.device, dtype=torch.float32)
                if 'vis_proba' in sample:
                    mp['CAST-Net'] = torch.tensor(sample['vis_proba'],
                                                  device=self.device, dtype=torch.float32)
                if not mp:
                    continue

                result = self.fusion_model(mp)
                fp = result['fused_proba']
                loss = self.fusion_model.compute_loss(fp, label)

                pred = torch.argmax(fp).item()
                if label in VISUAL_STRONG_CLASSES and pred != label:
                    loss = loss * 2.0
                elif label in PROBLEM_CLASSES and pred != label:
                    loss = loss * 1.5

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.fusion_model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                if pred == label:
                    correct += 1
                    pcc[label] += 1
                total += 1
                pct[label] += 1
                confidences.append(result['confidence'])

            scheduler.step()
            acc = correct / max(total, 1)
            pr = [pcc[pc] / pct[pc] for pc in PROBLEM_CLASSES if pct[pc] > 0]
            apr = np.mean(pr) if pr else 0
            vsr = [pcc[vc] / pct[vc] for vc in VISUAL_STRONG_CLASSES if pct[vc] > 0]
            avsr = np.mean(vsr) if vsr else 0

            self.training_stats['losses'].append(total_loss / max(total, 1))
            self.training_stats['accuracies'].append(acc)
            self.training_stats['confidences'].append(np.mean(confidences) if confidences else 0)
            self.training_stats['problem_recalls'].append(apr)

            score = acc * 0.4 + apr * 0.3 + avsr * 0.3
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"    Epoch {epoch + 1:3d}/{self.config.phase1_epochs}: "
                      f"Loss={total_loss / max(total, 1):.4f}, "
                      f"Acc={acc * 100:.1f}%, ProbRecall={apr * 100:.1f}%, "
                      f"VisStrongRecall={avsr * 100:.1f}%")
            if score > best_score:
                best_score = score
                best_state = deepcopy(self.fusion_model.state_dict())

        if best_state:
            self.fusion_model.load_state_dict(best_state)
        print(f"  Phase 1 Best: {best_score * 100:.1f}%")

        print(f"\n  --- Phase 2: Fine-tune ---")
        optimizer = optim.Adam(self.fusion_model.parameters(), lr=self.config.phase2_lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.phase2_epochs)
        best_score2 = best_score

        for epoch in range(self.config.phase2_epochs):
            self.fusion_model.train()
            total_loss, correct, total = 0, 0, 0
            confidences = []
            pcc = defaultdict(int)
            pct = defaultdict(int)
            np.random.shuffle(augmented)

            for sample in augmented:
                label = sample['label']
                mp = {}
                if 'emg_proba' in sample:
                    mp['SC-MSFE'] = torch.tensor(sample['emg_proba'],
                                                 device=self.device, dtype=torch.float32)
                if 'vis_proba' in sample:
                    mp['CAST-Net'] = torch.tensor(sample['vis_proba'],
                                                  device=self.device, dtype=torch.float32)
                if not mp:
                    continue

                result = self.fusion_model(mp)
                fp = result['fused_proba']
                loss = self.fusion_model.compute_loss(fp, label)

                pred = torch.argmax(fp).item()
                if label in VISUAL_STRONG_CLASSES and pred != label:
                    loss = loss * 1.8
                elif label in PROBLEM_CLASSES and pred != label:
                    loss = loss * 1.5

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.fusion_model.parameters(), 0.5)
                optimizer.step()

                total_loss += loss.item()
                if pred == label:
                    correct += 1
                    pcc[label] += 1
                total += 1
                pct[label] += 1
                confidences.append(result['confidence'])

            scheduler.step()
            acc = correct / max(total, 1)
            pr = [pcc[pc] / pct[pc] for pc in PROBLEM_CLASSES if pct[pc] > 0]
            apr = np.mean(pr) if pr else 0
            vsr = [pcc[vc] / pct[vc] for vc in VISUAL_STRONG_CLASSES if pct[vc] > 0]
            avsr = np.mean(vsr) if vsr else 0

            self.training_stats['losses'].append(total_loss / max(total, 1))
            self.training_stats['accuracies'].append(acc)
            self.training_stats['confidences'].append(np.mean(confidences) if confidences else 0)
            self.training_stats['problem_recalls'].append(apr)

            score = acc * 0.4 + apr * 0.3 + avsr * 0.3
            if (epoch + 1) % 15 == 0 or epoch == 0:
                print(f"    Epoch {epoch + 1:3d}/{self.config.phase2_epochs}: "
                      f"Loss={total_loss / max(total, 1):.4f}, "
                      f"Acc={acc * 100:.1f}%, ProbRecall={apr * 100:.1f}%, "
                      f"VisStrongRecall={avsr * 100:.1f}%")
            if score > best_score2:
                best_score2 = score
                best_state = deepcopy(self.fusion_model.state_dict())

        if best_state:
            self.fusion_model.load_state_dict(best_state)
        print(f"  Phase 2 Best: {best_score2 * 100:.1f}%")
        print(f"  Overall Best: {max(best_score, best_score2) * 100:.1f}%")
        self._print_weights()

    def _print_weights(self):
        print(f"\n  Learned Fusion Weights:")
        weights = self.fusion_model.base_weights.get_weights_numpy()
        for i, label in enumerate(GESTURE_LABELS):
            parts = []
            for j, name in enumerate(self.fusion_model.modality_names):
                parts.append(f"{name}={weights[j, i]:.2f}")
            flag = ""
            if i in PROBLEM_CLASSES:
                flag = " [PROBLEM]"
            elif i in VISUAL_STRONG_CLASSES:
                flag = " [VIS]"
            elif i in EMG_STRONG_CLASSES:
                flag = " [EMG]"
            print(f"    G{i + 1:2d} {label:<25}: {', '.join(parts)}{flag}")

    def save(self, save_dir):
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.fusion_model.state_dict(), save_dir / 'solid_net.pth')
        config_data = {
            'model_name': 'SOLID-Net',
            'version': 'V8.1',
            'modalities': self.fusion_model.modality_names,
            'gesture_labels': list(GESTURE_LABELS),
            'problem_classes': {str(k): v for k, v in PROBLEM_CLASSES.items()},
            'visual_strong_classes': list(VISUAL_STRONG_CLASSES),
            'emg_strong_classes': list(EMG_STRONG_CLASSES),
            'learned_weights': self.fusion_model.base_weights.get_weights_numpy().tolist(),
            'emg_per_class_acc': self.emg_per_class_acc,
            'vis_per_class_acc': self.vis_per_class_acc,
            'training_stats': {
                'total_epochs': len(self.training_stats['losses']),
                'final_loss': self.training_stats['losses'][-1]
                if self.training_stats['losses'] else None,
                'final_acc': self.training_stats['accuracies'][-1]
                if self.training_stats['accuracies'] else None,
            },
            'timestamp': datetime.now().isoformat()
        }
        with open(save_dir / 'solid_net_config.json', 'w') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        print(f"  [Save] SOLID-Net -> {save_dir}")

class ResultVisualizer:
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
        })

    def plot_all(self, metrics: Dict, training_stats: Dict):
        self.plot_confusion_matrices(metrics)
        self.plot_metrics_comparison(metrics)
        self.plot_radar(metrics)
        self.plot_training_curves(training_stats)
        self.plot_per_class_detail(metrics)
        self.plot_weight_heatmap(metrics)

    def plot_confusion_matrices(self, metrics):
        method_keys = ['emg_only', 'visual_only', 'fusion']
        titles = ['SC-MSFE (EMG)', 'CAST-Net (Visual)', 'SOLID-Net (Fusion)']
        available = [(k, t) for k, t in zip(method_keys, titles) if k in metrics]

        if not available:
            return

        n = len(available)
        fig, axes = plt.subplots(1, n, figsize=(7.5 * n, 7))
        if n == 1:
            axes = [axes]

        for ax, (key, title) in zip(axes, available):
            cm = np.array(metrics[key]['confusion_matrix'])
            cm_sum = cm.sum(axis=1, keepdims=True)
            cm_sum[cm_sum == 0] = 1
            cm_norm = cm.astype('float') / cm_sum
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                        xticklabels=GESTURE_NAMES_SHORT,
                        yticklabels=GESTURE_NAMES_SHORT,
                        cbar_kws={'shrink': 0.8}, annot_kws={'size': 7},
                        vmin=0, vmax=1)
            acc = metrics[key]['accuracy'] * 100
            ax.set_title(f'{title}\nAccuracy: {acc:.1f}%', fontsize=14, fontweight='bold')
            ax.set_xlabel('Predicted', fontsize=11)
            ax.set_ylabel('True', fontsize=11)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.tick_params(axis='y', rotation=0, labelsize=8)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [Plot] confusion_matrices.png")

    def plot_metrics_comparison(self, metrics):
        method_keys = ['emg_only', 'visual_only', 'fusion']
        display_names = ['SC-MSFE', 'CAST-Net', 'SOLID-Net']
        colors = ['#2E86AB', '#A23B72', '#27AE60']
        metric_names = ['accuracy', 'f1', 'precision', 'recall']
        metric_display = ['Accuracy', 'F1 Score', 'Precision', 'Recall']

        available = [(mk, dn, cl) for mk, dn, cl in
                     zip(method_keys, display_names, colors) if mk in metrics]
        if not available:
            return

        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(len(metric_names))
        n = len(available)
        width = 0.8 / n

        for i, (mk, dn, cl) in enumerate(available):
            vals = [metrics[mk].get(m, 0) * 100 for m in metric_names]
            bars = ax.bar(x + i * width - (n - 1) * width / 2, vals, width,
                          label=dn, color=cl, alpha=0.85, edgecolor='white', linewidth=1)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_ylabel('Score (%)', fontsize=13, fontweight='bold')
        ax.set_title('Model Performance Comparison',
                     fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_display, fontsize=12)
        ax.legend(fontsize=11, loc='upper right')
        ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [Plot] metrics_comparison.png")

    def plot_radar(self, metrics):
        method_keys = ['emg_only', 'visual_only', 'fusion']
        display_names = ['SC-MSFE', 'CAST-Net', 'SOLID-Net']
        colors = ['#2E86AB', '#A23B72', '#27AE60']

        available = [(mk, dn, cl) for mk, dn, cl in
                     zip(method_keys, display_names, colors)
                     if mk in metrics and 'per_class_recall' in metrics[mk]]
        if not available:
            return

        labels = list(GESTURE_NAMES_SHORT)
        n_cats = len(labels)
        angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(11, 11), subplot_kw=dict(polar=True))

        for mk, dn, cl in available:
            vals = metrics[mk]['per_class_recall'][:n_cats]
            while len(vals) < n_cats:
                vals.append(0)
            vals_plot = [v * 100 for v in vals] + [vals[0] * 100]
            ax.plot(angles, vals_plot, 'o-', linewidth=2.2, label=dn, color=cl, markersize=5)
            ax.fill(angles, vals_plot, alpha=0.08, color=cl)

        ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9)
        ax.set_ylim(0, 110)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=8)
        ax.set_title('Per-class Recall Radar Chart', fontsize=15, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=11)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'radar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [Plot] radar_chart.png")

    def plot_training_curves(self, training_stats):
        if not training_stats.get('losses'):
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        epochs = range(1, len(training_stats['losses']) + 1)

        axes[0, 0].plot(epochs, training_stats['losses'], 'b-', linewidth=1.5)
        axes[0, 0].set_title('Training Loss', fontsize=13, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].plot(epochs, [a * 100 for a in training_stats['accuracies']],
                        'g-', linewidth=1.5)
        axes[0, 1].set_title('Training Accuracy', fontsize=13, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].grid(alpha=0.3)

        axes[1, 0].plot(epochs, training_stats['confidences'], 'r-', linewidth=1.5)
        axes[1, 0].set_title('Average Confidence', fontsize=13, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Confidence')
        axes[1, 0].grid(alpha=0.3)

        if training_stats.get('problem_recalls'):
            axes[1, 1].plot(epochs[:len(training_stats['problem_recalls'])],
                            [r * 100 for r in training_stats['problem_recalls']],
                            'm-', linewidth=1.5)
            axes[1, 1].set_title('Problem Class Recall', fontsize=13, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall (%)')
            axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [Plot] training_curves.png")

    def plot_per_class_detail(self, metrics):
        method_keys = ['emg_only', 'visual_only', 'fusion']
        display_names = ['SC-MSFE', 'CAST-Net', 'SOLID-Net']
        colors = ['#2E86AB', '#A23B72', '#27AE60']

        available = [(mk, dn, cl) for mk, dn, cl in
                     zip(method_keys, display_names, colors)
                     if mk in metrics and 'per_class_recall' in metrics[mk]]
        if not available:
            return

        fig, ax = plt.subplots(figsize=(18, 8))
        x = np.arange(NUM_CLASSES)
        n = len(available)
        width = 0.8 / n

        for i, (mk, dn, cl) in enumerate(available):
            recalls = metrics[mk]['per_class_recall'][:NUM_CLASSES]
            while len(recalls) < NUM_CLASSES:
                recalls.append(0)
            bars = ax.bar(x + i * width - (n - 1) * width / 2,
                          [r * 100 for r in recalls], width,
                          label=dn, color=cl, alpha=0.85, edgecolor='white')

        ax.set_ylabel('Recall (%)', fontsize=13, fontweight='bold')
        ax.set_title('Per-class Recall Comparison',
                     fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(GESTURE_NAMES_SHORT, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=11, loc='upper right')
        ax.set_ylim(0, 120)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
        ax.grid(axis='y', alpha=0.3)

        for pc in PROBLEM_CLASSES:
            ax.axvspan(pc - 0.45, pc + 0.45, alpha=0.08, color='red')
            ax.text(pc, 115, 'PROB', ha='center', fontsize=8, color='red')
        for vc in VISUAL_STRONG_CLASSES:
            ax.text(vc, 112, 'VIS', ha='center', fontsize=7, color='purple')

        plt.tight_layout()
        plt.savefig(self.save_dir / 'per_class_recall.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [Plot] per_class_recall.png")

    def plot_weight_heatmap(self, metrics):
        if 'fusion_weights' not in metrics:
            return
        weights = np.array(metrics['fusion_weights'])
        if weights.ndim != 2 or weights.shape[0] < 2:
            return

        fig, ax = plt.subplots(figsize=(14, 4))
        sns.heatmap(weights, annot=True, fmt='.2f', cmap='RdYlGn',
                    xticklabels=GESTURE_NAMES_SHORT,
                    yticklabels=['SC-MSFE', 'CAST-Net'],
                    ax=ax, vmin=0, vmax=1, annot_kws={'size': 10})
        ax.set_title('Learned Fusion Weights', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', rotation=0, labelsize=11)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'weight_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [Plot] weight_heatmap.png")


class MultimodalFusionSystem:
    def __init__(self, config=None):
        self.config = config or FusionConfig()
        self.manager = FusionSystemManager(self.config)
        self.data_loader = FusionDataLoader(self.config)
        self.visualizer = ResultVisualizer(OUTPUT_DIR)

    def setup(self):
        print(f"\n{'=' * 70}")
        print("Loading Modality Models")
        print(f"{'=' * 70}")

        emg_model = SCMSFE(self.config)
        if emg_model.load(EMG_MODEL_DIR):
            self.manager.register_modality("SC-MSFE", emg_model)
        else:
            print("  [WARN] SC-MSFE not loaded!")

        print(f"\n  [CAST-Net] Model path: {VISUAL_MODEL_PATH}")
        cast_net = CASTNet(self.config)
        if cast_net.load(VISUAL_MODEL_PATH):
            self.manager.register_modality("CAST-Net", cast_net)
        else:
            print("  [WARN] CAST-Net not loaded!")

        print(f"\n  Model Status:")
        for name, model in self.manager.modality_models.items():
            status = "OK" if model.is_loaded else "FAILED"
            print(f"    {name:<15}: {status}")

        n_loaded = sum(1 for m in self.manager.modality_models.values() if m.is_loaded)
        if n_loaded == 0:
            print("\n  [FATAL] No modality loaded!")
            sys.exit(1)

    def verify_emg(self):
        print(f"\n{'=' * 70}")
        print("Verifying SC-MSFE Model")
        print(f"{'=' * 70}")
        init_data = self.data_loader.load_emg_init_data()
        if 'SC-MSFE' in self.manager.modality_models:
            self.manager.modality_models['SC-MSFE'].verify(init_data)

    def collect_predictions(self, data):
        samples = []
        seq_len = self.config.visual_seq_len
        has_emg = 'SC-MSFE' in self.manager.modality_models and \
                  self.manager.modality_models['SC-MSFE'].is_loaded
        has_vis = 'CAST-Net' in self.manager.modality_models and \
                  self.manager.modality_models['CAST-Net'].is_loaded

        for gid in tqdm(range(NUM_CLASSES), desc="Collecting predictions"):
            if gid not in data:
                continue
            emg_windows = data[gid]['emg']
            vis_frames = data[gid]['visual']
            n_emg = len(emg_windows) if emg_windows else 0
            n_vis_seqs = max(1, len(vis_frames) // seq_len) if vis_frames else 0

            if has_emg and has_vis and emg_windows and vis_frames:
                n_samples = min(n_emg, n_vis_seqs)
                for i in range(n_samples):
                    sample = {'label': gid}
                    sample['emg_proba'] = self.manager.modality_models['SC-MSFE'].predict_proba(
                        emg_windows[i])
                    start = min(i * seq_len, max(0, len(vis_frames) - seq_len))
                    frames = vis_frames[start:start + seq_len]
                    if len(frames) > 0:
                        sample['vis_proba'] = self.manager.modality_models['CAST-Net'].predict_proba(
                            frames)
                    samples.append(sample)
            elif has_emg and emg_windows:
                for i in range(min(n_emg, 20)):
                    sample = {'label': gid}
                    sample['emg_proba'] = self.manager.modality_models['SC-MSFE'].predict_proba(
                        emg_windows[i])
                    samples.append(sample)
            elif has_vis and vis_frames:
                for i in range(n_vis_seqs):
                    sample = {'label': gid}
                    start = min(i * seq_len, max(0, len(vis_frames) - seq_len))
                    frames = vis_frames[start:start + seq_len]
                    if frames:
                        sample['vis_proba'] = self.manager.modality_models['CAST-Net'].predict_proba(
                            frames)
                    samples.append(sample)

        return samples

    def pre_evaluate_modalities(self, data):
        print(f"\n{'=' * 70}")
        print("Pre-evaluating Single-Modality Performance")
        print(f"{'=' * 70}")

        self.manager.emg_per_class_acc = self.manager.compute_emg_accuracy(data)
        self.manager.vis_per_class_acc = self.manager.compute_visual_accuracy(data)

        print(f"\n  [Complementarity Analysis]")
        for gid in range(NUM_CLASSES):
            ea = self.manager.emg_per_class_acc.get(gid, 0)
            va = self.manager.vis_per_class_acc.get(gid, 0)
            if ea > 0 or va > 0:
                better = "EMG" if ea > va else "VIS" if va > ea else "TIE"
                gap = abs(ea - va) * 100
                marker = ""
                if gid in PROBLEM_CLASSES:
                    marker = " PROB"
                elif gid in VISUAL_STRONG_CLASSES:
                    marker = " VIS"
                elif gid in EMG_STRONG_CLASSES:
                    marker = " EMG"
                print(f"    G{gid + 1:2d} {GESTURE_LABELS[gid]:<25}: "
                      f"EMG={ea * 100:5.1f}% vs VIS={va * 100:5.1f}% "
                      f"-> {better} (+{gap:.1f}%){marker}")

    def train(self, data):
        self.pre_evaluate_modalities(data)
        self.manager.init_fusion_model()
        train_samples = self.collect_predictions(data)
        print(f"\n  Collected {len(train_samples)} training samples")
        n_both = sum(1 for s in train_samples if 'emg_proba' in s and 'vis_proba' in s)
        n_emg_only = sum(1 for s in train_samples if 'emg_proba' in s and 'vis_proba' not in s)
        n_vis_only = sum(1 for s in train_samples if 'vis_proba' in s and 'emg_proba' not in s)
        print(f"    Both modalities: {n_both}")
        print(f"    EMG only:        {n_emg_only}")
        print(f"    Visual only:     {n_vis_only}")

        if train_samples:
            self.manager.train(
                train_samples,
                epochs=self.config.fusion_epochs,
                lr=self.config.fusion_lr
            )

    def evaluate(self, data):
        print(f"\n{'=' * 70}")
        print("Final Evaluation")
        print(f"{'=' * 70}")

        results = {
            'emg_only': {'preds': [], 'labels': []},
            'visual_only': {'preds': [], 'labels': []},
            'fusion': {'preds': [], 'labels': [], 'confidences': [],
                       'power_saves': []}
        }
        seq_len = self.config.visual_seq_len
        self.manager.power_save_mode = False
        self.manager.consecutive_same_results = 0
        self.manager.last_prediction = None

        has_emg = 'SC-MSFE' in self.manager.modality_models and \
                  self.manager.modality_models['SC-MSFE'].is_loaded
        has_vis = 'CAST-Net' in self.manager.modality_models and \
                  self.manager.modality_models['CAST-Net'].is_loaded

        for gid in tqdm(range(NUM_CLASSES), desc="Evaluating"):
            if gid not in data:
                continue
            emg_windows = data[gid]['emg']
            vis_frames = data[gid]['visual']

            if has_emg and emg_windows:
                for X in emg_windows:
                    proba = self.manager.modality_models['SC-MSFE'].predict_proba(X)
                    results['emg_only']['preds'].append(int(np.argmax(proba)))
                    results['emg_only']['labels'].append(gid)

            if has_vis and vis_frames:
                n_seqs = max(1, len(vis_frames) // seq_len)
                for i in range(n_seqs):
                    start = min(i * seq_len, max(0, len(vis_frames) - seq_len))
                    frames = vis_frames[start:start + seq_len]
                    if frames:
                        proba = self.manager.modality_models['CAST-Net'].predict_proba(frames)
                        results['visual_only']['preds'].append(int(np.argmax(proba)))
                        results['visual_only']['labels'].append(gid)

            n_emg = len(emg_windows) if emg_windows else 0
            n_vis_seqs = max(1, len(vis_frames) // seq_len) if vis_frames else 0

            if has_emg and has_vis and emg_windows and vis_frames:
                n_samples = min(n_emg, n_vis_seqs)
            elif has_emg and emg_windows:
                n_samples = n_emg
            elif has_vis and vis_frames:
                n_samples = n_vis_seqs
            else:
                continue

            for i in range(n_samples):
                md = {}
                if has_emg and emg_windows:
                    md['SC-MSFE'] = emg_windows[min(i, n_emg - 1)]
                if has_vis and vis_frames:
                    start = min(i * seq_len, max(0, len(vis_frames) - seq_len))
                    md['CAST-Net'] = vis_frames[start:start + seq_len]

                result = self.manager.predict(md)
                results['fusion']['preds'].append(result['prediction'])
                results['fusion']['labels'].append(gid)
                results['fusion']['confidences'].append(result['confidence'])
                results['fusion']['power_saves'].append(
                    result.get('power_save_mode', False))

        metrics = {}
        for method in ['emg_only', 'visual_only', 'fusion']:
            if results[method]['preds']:
                preds = np.array(results[method]['preds'])
                labels = np.array(results[method]['labels'])
                metrics[method] = {
                    'accuracy': float(accuracy_score(labels, preds)),
                    'f1': float(f1_score(labels, preds, average='macro', zero_division=0)),
                    'precision': float(precision_score(labels, preds, average='macro',
                                                       zero_division=0)),
                    'recall': float(recall_score(labels, preds, average='macro',
                                                 zero_division=0)),
                    'per_class_recall': recall_score(
                        labels, preds, average=None,
                        labels=list(range(NUM_CLASSES)), zero_division=0).tolist(),
                    'confusion_matrix': confusion_matrix(
                        labels, preds, labels=list(range(NUM_CLASSES))).tolist(),
                    'n_samples': len(preds)
                }

        if results['fusion']['confidences']:
            metrics['fusion']['avg_confidence'] = float(np.mean(results['fusion']['confidences']))
            metrics['fusion']['power_save_rate'] = float(np.mean(results['fusion']['power_saves']))

        if self.manager.fusion_model:
            metrics['fusion_weights'] = self.manager.fusion_model.base_weights.get_weights_numpy().tolist()

        return metrics

    def print_results(self, metrics):
        print(f"\n{'=' * 70}")
        print("Results Summary")
        print(f"{'=' * 70}")
        names = {
            'emg_only': 'SC-MSFE (EMG)',
            'visual_only': 'CAST-Net (Visual)',
            'fusion': 'SOLID-Net (Fusion)'
        }
        print(f"\n  {'Method':<25} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'N':>6}")
        print(f"  {'-' * 65}")
        for method in ['emg_only', 'visual_only', 'fusion']:
            if method in metrics:
                m = metrics[method]
                print(f"  {names[method]:<25} "
                      f"{m['accuracy'] * 100:>7.1f}% "
                      f"{m['f1'] * 100:>7.1f}% "
                      f"{m['precision'] * 100:>7.1f}% "
                      f"{m['recall'] * 100:>7.1f}% "
                      f"{m['n_samples']:>6}")

        if 'fusion' in metrics:
            print(f"\n  Per-class Recall (SOLID-Net):")
            for i, label in enumerate(GESTURE_LABELS):
                r = metrics['fusion']['per_class_recall'][i] \
                    if i < len(metrics['fusion']['per_class_recall']) else 0
                emg_r = metrics.get('emg_only', {}).get('per_class_recall', [0] * 13)
                vis_r = metrics.get('visual_only', {}).get('per_class_recall', [0] * 13)
                er = emg_r[i] if i < len(emg_r) else 0
                vr = vis_r[i] if i < len(vis_r) else 0

                bar = '=' * int(r * 20) + ' ' * (20 - int(r * 20))
                flag = ""
                if i in PROBLEM_CLASSES:
                    flag = " PROB"
                elif i in VISUAL_STRONG_CLASSES:
                    flag = " VIS"

                best_single = max(er, vr)
                improvement = "UP" if r > best_single else "DN" if r < best_single else "EQ"

                print(f"    G{i + 1:2d} {label:<25}: {r * 100:5.1f}% {bar} "
                      f"(E:{er * 100:4.0f}% V:{vr * 100:4.0f}% {improvement}){flag}")

            if 'avg_confidence' in metrics['fusion']:
                print(f"\n  Avg Confidence: {metrics['fusion']['avg_confidence']:.3f}")
            if 'power_save_rate' in metrics['fusion']:
                print(f"  Power Save Rate: {metrics['fusion']['power_save_rate'] * 100:.1f}%")

    def save_results(self, metrics):
        save_dir = OUTPUT_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        self.manager.save(save_dir)
        with open(save_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  Generating plots...")
        self.visualizer.plot_all(metrics, self.manager.training_stats)
        return save_dir


def main():
    print("=" * 70)
    print("SOLID-Net: Multimodal Fusion System")
    print(f"  SC-MSFE:    Six-Channel Multi-Scale Feature Extraction (EMG)")
    print(f"  CAST-Net:   Contextual Attention Sequential Temporal Network (Visual)")
    print(f"  SOLID-Net:  Stable Online Learning Independent Decision-fusion Network")
    print(f"  Output:     {OUTPUT_DIR}")
    print("=" * 70)

    print(f"\n[Fusion Methods Available]")
    print(f"  - Adaptive weighted fusion with confusion awareness")
    print(f"  - Stacking meta-learner")
    print(f"  - Mixture of Experts (MoE)")
    print(f"  - Temperature scaling calibration")

    device = select_gpu()
    print(f"\n[Device] {device}")

    config = FusionConfig(device=device)
    system = MultimodalFusionSystem(config)

    system.setup()
    system.verify_emg()

    test_data = system.data_loader.load_test_data()

    system.train(test_data)

    metrics = system.evaluate(test_data)
    system.print_results(metrics)

    save_dir = system.save_results(metrics)

    print(f"\n{'=' * 70}")
    print("Complete!")
    print(f"{'=' * 70}")
    for key, name in [('emg_only', 'SC-MSFE (EMG)'),
                      ('visual_only', 'CAST-Net (Visual)'),
                      ('fusion', 'SOLID-Net (Fusion)')]:
        if key in metrics:
            print(f"  {name:<25}: {metrics[key]['accuracy'] * 100:.1f}%")
    print(f"\n  Output: {save_dir}")


if __name__ == '__main__':
    main()
