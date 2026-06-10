import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.fft import fft
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import warnings
import pickle
import json

warnings.filterwarnings('ignore')

ROOT_PATH = Path("/yourpath/Dataset")
TRAIN_EMG_SINGLE = ROOT_PATH / "origin_emg" / "Single gesture"
TRAIN_EMG_CONTINUOUS = ROOT_PATH / "origin_emg" / "Continuous gesture"
TEST_DIR = ROOT_PATH / "test"
MODEL_DIR = ROOT_PATH / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

GESTURE_NAMES = (
    'Fist', 'Extension', 'Rotation', 'Opposition', 'Ball', 'Putty', 'Press',
    'Interlace', 'Flexion', 'Massage', 'Towel', 'Tapping', 'Piano'
)

GESTURE_NAME_MAP = {i: name for i, name in enumerate(GESTURE_NAMES)}
GESTURE_ID_MAP = {name: i for i, name in enumerate(GESTURE_NAMES)}

CONFUSION_GROUPS = [
    [0, 4],
    [1, 3],
    [2, 5],
    [6, 12],
]


def get_group_for_class(cls_id):
    for group in CONFUSION_GROUPS:
        if cls_id in group:
            return group
    return [cls_id]


def safe_normalize(arr):
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(arr, -1e6, 1e6)


@dataclass
class EMGConfig:
    num_classes: int = 13
    emg_channels: int = 6
    window_size: int = 180
    step_size: int = 22
    num_augments: int = 4

    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.4
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 100
    patience: int = 25

    device: str = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    label_map: dict = field(default_factory=lambda: {
        'Fist Clench': 0, 'Finger Extension': 1, 'Wrist Rotation': 2,
        'Finger Opposition': 3, 'Ball Squeeze': 4, 'TheraPutty Pinch': 5,
        'Finger Pressing': 6, 'Interlace Fingers': 7, 'Wrist Flexion-Extension': 8,
        'Finger Massage': 9, 'Towel Scrunch': 10, 'Hand Tapping': 11, 'Piano Tap': 12,
    })


class EMGDataLoader:
    def __init__(self, config: EMGConfig):
        self.config = config

    def load_csv(self, filepath):
        for enc in ['utf-8', 'gbk', 'gb2312', 'latin1']:
            try:
                return pd.read_csv(filepath, encoding=enc)
            except:
                continue
        return None

    def get_emg_columns(self, df):
        columns = df.columns.tolist()
        for pattern in ['ch{}_fil', 'ch{}']:
            cols = []
            for i in range(1, 7):
                for c in columns:
                    if pattern.format(i) in c.lower() and 'env' not in c.lower():
                        cols.append(c)
                        break
            if len(cols) == 6:
                return cols
        return []

    def parse_label(self, val):
        if pd.isna(val):
            return None
        try:
            num = int(float(val))
            if 1 <= num <= 13:
                return num - 1
            elif 0 <= num <= 12:
                return num
        except:
            pass
        text = str(val).strip()
        if text in self.config.label_map:
            return self.config.label_map[text]
        return None

    def load_single_gesture(self, filepath, gesture_id):
        df = self.load_csv(filepath)
        if df is None:
            return None, None

        emg_cols = self.get_emg_columns(df)
        if len(emg_cols) < 6:
            return None, None

        X_raw = safe_normalize(df[emg_cols].values.astype(np.float32))
        windows = []
        ws, step = self.config.window_size, self.config.step_size

        for i in range(0, len(X_raw) - ws + 1, step):
            windows.append(X_raw[i:i+ws])

        if not windows:
            return None, None

        return np.array(windows, dtype=np.float32), np.full(len(windows), gesture_id, dtype=np.int32)

    def load_continuous_gesture(self, filepath):
        df = self.load_csv(filepath)
        if df is None:
            return None, None

        emg_cols = self.get_emg_columns(df)
        if len(emg_cols) < 6:
            return None, None

        label_col = None
        for c in df.columns:
            if 'label' in c.lower():
                label_col = c
                break
        if label_col is None:
            return None, None

        trans_col = next((c for c in df.columns if 'transition' in c.lower()), None)
        X_raw = safe_normalize(df[emg_cols].values.astype(np.float32))
        labels = df[label_col].values
        trans = df[trans_col].values if trans_col else np.zeros(len(df))

        windows, window_labels = [], []
        ws, step = self.config.window_size, self.config.step_size

        for i in range(0, len(X_raw) - ws + 1, step):
            mid_s, mid_e = ws // 4, ws * 3 // 4
            votes = Counter()
            trans_count = 0

            for j in range(i + mid_s, min(i + mid_e, len(X_raw))):
                if trans_col:
                    t = trans[j]
                    if (isinstance(t, (int, float, np.integer, np.floating)) and int(t) == 1) or \
                       (isinstance(t, str) and t.lower() in ['true', '1']):
                        trans_count += 1
                        continue
                lbl = self.parse_label(labels[j])
                if lbl is not None:
                    votes[lbl] += 1

            total = mid_e - mid_s
            if total > 0 and trans_count / total < 0.5 and votes:
                windows.append(X_raw[i:i+ws])
                window_labels.append(votes.most_common(1)[0][0])

        if not windows:
            return None, None
        return np.array(windows, dtype=np.float32), np.array(window_labels, dtype=np.int32)


class EMGFeatureExtractor:
    def __init__(self, config: EMGConfig):
        self.config = config
        self.scales = [2, 4, 6, 8]
        self.feature_dim = None

    def _normalize_window(self, X):
        N, T, C = X.shape
        eps = 1e-8
        X_min = X.min(axis=1, keepdims=True)
        X_max = X.max(axis=1, keepdims=True)
        return (X - X_min) / (X_max - X_min + eps)

    def _shape_features(self, ch):
        N, T = ch.shape
        eps = 1e-8
        feats = []

        feats.append(np.mean(ch, axis=1))
        feats.append(np.std(ch, axis=1))
        feats.append(np.median(ch, axis=1))

        diff = np.diff(ch, axis=1)
        feats.append(np.sum(diff > 0, axis=1) / (T - 1))
        feats.append(np.mean(np.abs(diff), axis=1))

        pos_diff = np.where(diff > 0, diff, 0)
        neg_diff = np.where(diff < 0, -diff, 0)
        feats.append(np.sum(pos_diff, axis=1) / (np.sum(diff > 0, axis=1) + eps))
        feats.append(np.sum(neg_diff, axis=1) / (np.sum(diff < 0, axis=1) + eps))

        diff2 = np.diff(diff, axis=1)
        feats.append(np.sum(np.sign(diff2[:, 1:] + eps) != np.sign(diff2[:, :-1] + eps), axis=1) / T)
        feats.append(np.mean(np.abs(diff2), axis=1))
        feats.append(np.max(np.abs(diff2), axis=1))

        feats.append(np.argmax(ch, axis=1) / T)
        feats.append(np.argmin(ch, axis=1) / T)
        feats.append((np.argmax(ch, axis=1) < np.argmin(ch, axis=1)).astype(np.float32))

        centered = ch - 0.5
        zc = np.sign(centered[:, 1:] + eps) != np.sign(centered[:, :-1] + eps)
        feats.append(np.argmax(zc, axis=1) / T)
        feats.append(np.sum(zc, axis=1) / T)

        for p in [10, 25, 75, 90]:
            feats.append(np.percentile(ch, p, axis=1))
        feats.append(np.percentile(ch, 75, axis=1) - np.percentile(ch, 25, axis=1))

        return feats

    def _multiscale_features(self, ch):
        N, T = ch.shape
        eps = 1e-8
        feats = []

        for n_seg in self.scales:
            seg_len = T // n_seg
            seg_means = []
            seg_slopes = []
            seg_energies = []

            for s in range(n_seg):
                start = s * seg_len
                end = start + seg_len if s < n_seg - 1 else T
                seg = ch[:, start:end]

                seg_means.append(np.mean(seg, axis=1))
                seg_energies.append(np.mean(seg**2, axis=1))

                x = np.arange(end - start)
                x_mean = x.mean()
                slope = np.sum((x - x_mean) * (seg - seg.mean(axis=1, keepdims=True)), axis=1)
                slope = slope / (np.sum((x - x_mean)**2) + eps)
                seg_slopes.append(slope)

            feats.extend(seg_means)
            feats.extend(seg_slopes)
            feats.extend(seg_energies)

            seg_means_arr = np.array(seg_means).T
            seg_diff = np.diff(seg_means_arr, axis=1)
            for i in range(seg_diff.shape[1]):
                feats.append(seg_diff[:, i])

        return feats

    def _freq_features(self, ch):
        N, T = ch.shape
        eps = 1e-8
        feats = []

        fft_r = np.abs(fft(ch, axis=1))[:, :T//2]
        fft_sum = np.sum(fft_r, axis=1, keepdims=True) + eps
        fft_n = fft_r / fft_sum

        feats.append(np.argmax(fft_n, axis=1) / (T//2))

        bins = np.arange(T//2)
        centroid = np.sum(fft_n * bins, axis=1) / (T//2)
        feats.append(centroid)
        feats.append(np.sqrt(np.sum(fft_n * (bins - centroid.reshape(-1, 1))**2, axis=1)))

        n_bins = T // 2
        for n_bands in [4, 6]:
            band_len = n_bins // n_bands
            for i in range(n_bands):
                s, e = i * band_len, (i + 1) * band_len if i < n_bands - 1 else n_bins
                feats.append(np.sum(fft_n[:, s:e], axis=1))

        mid = n_bins // 2
        feats.append(np.sum(fft_n[:, mid:], axis=1) / (np.sum(fft_n[:, :mid], axis=1) + eps))

        log_fft = np.log(fft_r + eps)
        geo_mean = np.exp(np.mean(log_fft, axis=1))
        arith_mean = np.mean(fft_r, axis=1) + eps
        feats.append(geo_mean / arith_mean)

        return feats

    def _channel_features(self, X_norm):
        N, T, C = X_norm.shape
        eps = 1e-8
        feats = []

        energy = np.mean(X_norm**2, axis=1)
        total = np.sum(energy, axis=1, keepdims=True) + eps
        ratio = energy / total

        for c in range(C):
            feats.append(ratio[:, c])

        rank = np.argsort(np.argsort(-energy, axis=1), axis=1)
        for c in range(C):
            feats.append(rank[:, c] / C)

        feats.append(np.argmax(energy, axis=1) / C)

        temp = energy.copy()
        temp[np.arange(N), np.argmax(energy, axis=1)] = -1
        feats.append(np.argmax(temp, axis=1) / C)

        for i in range(C):
            for j in range(i+1, C):
                xi = X_norm[:, :, i] - X_norm[:, :, i].mean(axis=1, keepdims=True)
                xj = X_norm[:, :, j] - X_norm[:, :, j].mean(axis=1, keepdims=True)
                num = np.sum(xi * xj, axis=1)
                den = np.sqrt(np.sum(xi**2, axis=1) * np.sum(xj**2, axis=1) + eps)
                feats.append(num / den)

        for i in range(C-1):
            feats.append(np.abs(energy[:, i] - energy[:, i+1]))

        feats.append(-np.sum(ratio * np.log(ratio + eps), axis=1))

        return feats

    def _temporal_features(self, X_norm):
        N, T, C = X_norm.shape
        eps = 1e-8
        feats = []

        activity = np.sum(np.abs(np.diff(X_norm, axis=1)), axis=2)

        threshold = np.percentile(activity, 70, axis=1, keepdims=True)
        active_mask = activity > threshold
        feats.append(np.argmax(active_mask, axis=1) / (T - 1))
        feats.append(np.argmax(activity, axis=1) / (T - 1))
        feats.append(np.sum(active_mask, axis=1) / (T - 1))

        n_segs = 4
        seg_len = (T - 1) // n_segs
        seg_acts = []
        for s in range(n_segs):
            start = s * seg_len
            end = start + seg_len if s < n_segs - 1 else T - 1
            seg_acts.append(np.mean(activity[:, start:end], axis=1))

        seg_acts = np.column_stack(seg_acts)
        total_act = np.sum(seg_acts, axis=1, keepdims=True) + eps
        for s in range(n_segs):
            feats.append(seg_acts[:, s] / total_act.flatten())

        act_diff = np.abs(np.diff(activity, axis=1))
        feats.append(np.max(act_diff, axis=1))
        feats.append(np.argmax(act_diff, axis=1) / (T - 2))

        return feats

    def _traditional_features(self, X):
        N, T, C = X.shape
        eps = 1e-8
        feats = []

        for c in range(C):
            ch = X[:, :, c]

            feats.append(np.mean(np.abs(ch), axis=1))
            feats.append(np.sqrt(np.mean(ch**2, axis=1) + eps))
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

        X_norm = self._normalize_window(X)
        all_feats = []

        for c in range(self.config.emg_channels):
            ch = X_norm[:, :, c]
            all_feats.extend(self._shape_features(ch))
            all_feats.extend(self._multiscale_features(ch))
            all_feats.extend(self._freq_features(ch))

        all_feats.extend(self._channel_features(X_norm))
        all_feats.extend(self._temporal_features(X_norm))
        all_feats.extend(self._traditional_features(X))

        result = safe_normalize(np.column_stack(all_feats)).astype(np.float32)
        self.feature_dim = result.shape[1]
        return result


class EMGNet(nn.Module):
    def __init__(self, input_dim, num_classes=13, hidden_dim=256, num_layers=3, dropout=0.4):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ))

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        x = self.input_layer(x)

        for layer in self.hidden_layers:
            residual = x
            x = layer(x) + residual

        return self.output_layer(x)

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)


class LifelongEMGClassifier:
    def __init__(self, config: EMGConfig):
        self.config = config
        self.extractor = EMGFeatureExtractor(config)
        self.scaler = StandardScaler()

        self.nn_model = None

        self.ensemble_model = None

        self.memory_buffer_X = []
        self.memory_buffer_y = []
        self.buffer_size = 1000

        self.base_X = None
        self.base_y = None

        self.is_trained = False
        self.feature_dim = None

    def _save_base_samples(self, X_feat, y):
        base_X, base_y = [], []
        for cls_id in range(self.config.num_classes):
            mask = y == cls_id
            if np.sum(mask) > 0:
                cls_X = X_feat[mask]
                n_keep = min(30, len(cls_X))
                indices = np.random.choice(len(cls_X), n_keep, replace=False)
                base_X.append(cls_X[indices])
                base_y.extend([cls_id] * n_keep)

        self.base_X = np.concatenate(base_X) if base_X else None
        self.base_y = np.array(base_y) if base_y else None

    def fit(self, X_windows: np.ndarray, y: np.ndarray):
        print(f"[SC-MSFE] Extracting features from {len(X_windows)} windows...")
        X_feat = self.extractor.extract(X_windows)
        self.feature_dim = X_feat.shape[1]
        print(f"[SC-MSFE] Feature dimension: {self.feature_dim}")

        X_scaled = self.scaler.fit_transform(X_feat)

        self._save_base_samples(X_scaled, y)

        print("[SC-MSFE] Training ExtraTreesClassifier...")
        self.ensemble_model = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1,
            warm_start=True
        )
        self.ensemble_model.fit(X_scaled, y)

        print("[SC-MSFE] Training Neural Network...")
        self._train_nn(X_scaled, y)

        self.is_trained = True
        print("[SC-MSFE] Training complete!")

    def _train_nn(self, X_scaled: np.ndarray, y: np.ndarray):
        device = self.config.device

        indices = np.random.permutation(len(X_scaled))
        n_val = int(len(X_scaled) * 0.15)
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        X_train = torch.tensor(X_scaled[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y[train_idx], dtype=torch.long)
        X_val = torch.tensor(X_scaled[val_idx], dtype=torch.float32)
        y_val = torch.tensor(y[val_idx], dtype=torch.long)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)

        self.nn_model = EMGNet(
            input_dim=self.feature_dim,
            num_classes=self.config.num_classes,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        ).to(device)

        optimizer = torch.optim.AdamW(self.nn_model.parameters(), lr=self.config.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        best_val_acc = 0
        patience_counter = 0

        for epoch in range(self.config.epochs):
            self.nn_model.train()
            train_loss, train_correct, train_total = 0, 0, 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                outputs = self.nn_model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.nn_model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += y_batch.size(0)
                train_correct += predicted.eq(y_batch).sum().item()

            scheduler.step()

            self.nn_model.eval()
            with torch.no_grad():
                X_val_d = X_val.to(device)
                y_val_d = y_val.to(device)
                outputs = self.nn_model(X_val_d)
                _, predicted = outputs.max(1)
                val_acc = predicted.eq(y_val_d).sum().item() / len(y_val)

            train_acc = train_correct / train_total

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: Train={train_acc*100:.1f}%, Val={val_acc*100:.1f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.best_nn_state = self.nn_model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        if hasattr(self, 'best_nn_state'):
            self.nn_model.load_state_dict(self.best_nn_state)

        print(f"  Best Val Acc: {best_val_acc*100:.1f}%")

    def predict_proba_raw(self, X_feat: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X_feat)
        proba = self.ensemble_model.predict_proba(X_scaled)

        if proba.shape[1] != self.config.num_classes:
            full_proba = np.zeros((proba.shape[0], self.config.num_classes))
            for i, cls in enumerate(self.ensemble_model.classes_):
                if cls < self.config.num_classes:
                    full_proba[:, cls] = proba[:, i]
            proba = full_proba / (full_proba.sum(axis=1, keepdims=True) + 1e-10)

        return proba

    def predict_proba_nn(self, X_feat: np.ndarray) -> np.ndarray:
        if self.nn_model is None:
            return self.predict_proba_raw(X_feat)

        X_scaled = self.scaler.transform(X_feat)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.config.device)

        self.nn_model.eval()
        with torch.no_grad():
            proba = self.nn_model.predict_proba(X_tensor)

        return proba.cpu().numpy()

    def predict_proba_grouped(self, X_feat: np.ndarray) -> np.ndarray:
        raw_proba = self.predict_proba_raw(X_feat)
        N = len(X_feat)
        grouped_proba = np.zeros((N, self.config.num_classes))

        for i in range(N):
            pred_class = np.argmax(raw_proba[i])
            group = get_group_for_class(pred_class)

            if len(group) > 1:
                group_total_prob = sum(raw_proba[i, g] for g in group)
                shared_prob = group_total_prob / len(group)
                for g in group:
                    grouped_proba[i, g] = shared_prob
            else:
                grouped_proba[i, pred_class] = raw_proba[i, pred_class]

            for c in range(self.config.num_classes):
                if c not in group:
                    grouped_proba[i, c] = raw_proba[i, c]

        return grouped_proba

    def predict(self, X_window: np.ndarray):
        if X_window.ndim == 2:
            X_window = X_window[np.newaxis, ...]

        X_feat = self.extractor.extract(X_window)
        raw_proba = self.predict_proba_raw(X_feat)
        grouped_proba = self.predict_proba_grouped(X_feat)

        pred_class = np.argmax(raw_proba[0])
        group = get_group_for_class(pred_class)
        hint_labels = [GESTURE_NAME_MAP[g] for g in group]

        return grouped_proba[0], hint_labels

    def get_hint_for_visual(self, X_window: np.ndarray):
        _, hint_labels = self.predict(X_window)
        return hint_labels

    def lifelong_update(self, X_window: np.ndarray, label):
        if isinstance(label, str):
            if label in GESTURE_ID_MAP:
                label_id = GESTURE_ID_MAP[label]
            else:
                for name, idx in self.config.label_map.items():
                    if label.lower() in name.lower() or name.lower() in label.lower():
                        label_id = idx
                        break
                else:
                    return False
        else:
            label_id = int(label)
            if not 0 <= label_id < self.config.num_classes:
                return False

        if X_window.ndim == 2:
            X_window = X_window[np.newaxis, ...]

        X_feat = self.extractor.extract(X_window)
        X_scaled = self.scaler.transform(X_feat)

        self.memory_buffer_X.append(X_scaled[0])
        self.memory_buffer_y.append(label_id)

        if len(self.memory_buffer_X) > self.buffer_size:
            self.memory_buffer_X.pop(0)
            self.memory_buffer_y.pop(0)

        if len(self.memory_buffer_X) >= 50 and len(self.memory_buffer_X) % 50 == 0:
            self._incremental_train()

        return True

    def _incremental_train(self):
        if self.base_X is not None:
            X_all = np.concatenate([self.base_X, np.array(self.memory_buffer_X)])
            y_all = np.concatenate([self.base_y, np.array(self.memory_buffer_y)])
        else:
            X_all = np.array(self.memory_buffer_X)
            y_all = np.array(self.memory_buffer_y)

        self.ensemble_model.n_estimators += 20
        self.ensemble_model.fit(X_all, y_all)

    def save(self, save_dir: Path):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.nn_model is not None:
            torch.save({
                'model_state_dict': self.nn_model.state_dict(),
                'input_dim': self.feature_dim,
                'num_classes': self.config.num_classes,
                'hidden_dim': self.config.hidden_dim,
                'num_layers': self.config.num_layers,
                'dropout': self.config.dropout
            }, save_dir / 'emg_model.pth')
            print(f"  Saved: emg_model.pth")

        with open(save_dir / 'emg_ensemble.pkl', 'wb') as f:
            pickle.dump({
                'model': self.ensemble_model,
                'base_X': self.base_X,
                'base_y': self.base_y,
                'memory_buffer_X': self.memory_buffer_X[-100:],
                'memory_buffer_y': self.memory_buffer_y[-100:]
            }, f)
        print(f"  Saved: emg_ensemble.pkl")

        with open(save_dir / 'emg_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"  Saved: emg_scaler.pkl")

        config_dict = {
            'num_classes': self.config.num_classes,
            'emg_channels': self.config.emg_channels,
            'window_size': self.config.window_size,
            'step_size': self.config.step_size,
            'feature_dim': self.feature_dim,
            'hidden_dim': self.config.hidden_dim,
            'num_layers': self.config.num_layers,
            'dropout': self.config.dropout,
            'confusion_groups': CONFUSION_GROUPS,
            'gesture_names': list(GESTURE_NAMES)
        }
        with open(save_dir / 'emg_config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"  Saved: emg_config.json")

    def load(self, save_dir: Path):
        save_dir = Path(save_dir)

        with open(save_dir / 'emg_config.json', 'r') as f:
            config_dict = json.load(f)
        self.feature_dim = config_dict['feature_dim']

        with open(save_dir / 'emg_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)

        with open(save_dir / 'emg_ensemble.pkl', 'rb') as f:
            data = pickle.load(f)
        self.ensemble_model = data['model']
        self.base_X = data.get('base_X')
        self.base_y = data.get('base_y')
        self.memory_buffer_X = data.get('memory_buffer_X', [])
        self.memory_buffer_y = data.get('memory_buffer_y', [])

        pth_path = save_dir / 'emg_model.pth'
        if pth_path.exists():
            checkpoint = torch.load(pth_path, map_location=self.config.device)
            self.nn_model = EMGNet(
                input_dim=checkpoint['input_dim'],
                num_classes=checkpoint['num_classes'],
                hidden_dim=checkpoint['hidden_dim'],
                num_layers=checkpoint['num_layers'],
                dropout=checkpoint['dropout']
            ).to(self.config.device)
            self.nn_model.load_state_dict(checkpoint['model_state_dict'])
            self.nn_model.eval()

        self.is_trained = True
        print(f"[SC-MSFE] Model loaded from {save_dir}")


def augment_data(X, y, num_augments=4):
    X_aug, y_aug = [X], [y]

    for _ in range(num_augments):
        X_new = X.copy()
        scales = np.random.uniform(0.85, 1.15, size=(len(X), 1, 1))
        X_new = X_new * scales
        noise = np.random.randn(*X_new.shape) * 0.03 * np.std(X_new, axis=(1, 2), keepdims=True)
        X_new = X_new + noise
        X_aug.append(X_new.astype(np.float32))
        y_aug.append(y.copy())

    return np.concatenate(X_aug), np.concatenate(y_aug)


def evaluate_model(classifier, test_data, config):
    preds_all, labels_all = [], []
    group_correct, total = 0, 0

    for cls_id, X_test in test_data.items():
        grouped_proba, _ = [], []
        for X_window in X_test:
            proba, hints = classifier.predict(X_window)
            grouped_proba.append(proba)

        grouped_proba = np.array(grouped_proba)
        preds = np.argmax(grouped_proba, axis=1)

        for pred in preds:
            group = get_group_for_class(pred)
            if cls_id in group:
                group_correct += 1
            total += 1

        gt = np.full(len(preds), cls_id)
        preds_all.extend(preds)
        labels_all.extend(gt)

    acc = accuracy_score(labels_all, preds_all)
    prec = precision_score(labels_all, preds_all, average='macro', zero_division=0)
    rec = recall_score(labels_all, preds_all, average='macro', zero_division=0)
    f1 = f1_score(labels_all, preds_all, average='macro', zero_division=0)
    group_acc = group_correct / total if total > 0 else 0

    class_recall = {}
    for i in range(config.num_classes):
        mask = np.array(labels_all) == i
        if np.sum(mask) > 0:
            correct = np.sum(np.array(preds_all)[mask] == i)
            class_recall[i] = correct / np.sum(mask)
        else:
            class_recall[i] = 0

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'group_accuracy': group_acc,
        'class_recall': class_recall,
        'preds': preds_all,
        'labels': labels_all
    }


def plot_results(results, config, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    labels = results['labels']
    preds = results['preds']
    cm = confusion_matrix(labels, preds)
    cm_pct = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8) * 100

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_pct, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=[n[:6] for n in GESTURE_NAMES],
                yticklabels=[n[:6] for n in GESTURE_NAMES])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'SC-MSFE Confusion Matrix\nAccuracy: {results["accuracy"]*100:.1f}%, Group Acc: {results["group_accuracy"]*100:.1f}%')
    plt.tight_layout()
    plt.savefig(save_dir / 'emg_confusion_matrix.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    recalls = [results['class_recall'].get(i, 0) * 100 for i in range(config.num_classes)]
    bars = plt.bar(range(config.num_classes), recalls, color=plt.cm.Set3(np.linspace(0, 1, config.num_classes)))
    plt.xlabel('Gesture')
    plt.ylabel('Recall (%)')
    plt.title('Per-Class Recall')
    plt.xticks(range(config.num_classes), [n[:8] for n in GESTURE_NAMES], rotation=45, ha='right')
    plt.ylim(0, 110)
    for bar, val in zip(bars, recalls):
        plt.annotate(f'{val:.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_dir / 'emg_class_recall.png', dpi=300)
    plt.close()

    print(f"  Saved figures to {save_dir}")


def main():
    print("=" * 60)
    print("SC-MSFE Training")
    print("=" * 60)

    config = EMGConfig()
    loader = EMGDataLoader(config)

    print("\n[1/6] Loading training data...")
    train_X, train_y = [], []

    for gid in range(1, 14):
        gdir = TRAIN_EMG_SINGLE / str(gid)
        if not gdir.exists():
            continue
        count = 0
        for f in gdir.glob("Person*.csv"):
            X, y = loader.load_single_gesture(str(f), gid - 1)
            if X is not None:
                train_X.append(X)
                train_y.append(y)
                count += len(X)
        print(f"  Gesture {gid}: {count} windows")

    if TRAIN_EMG_CONTINUOUS.exists():
        cont_count = 0
        for f in tqdm(list(TRAIN_EMG_CONTINUOUS.glob("Person*.csv")), desc="  Continuous"):
            X, y = loader.load_continuous_gesture(str(f))
            if X is not None:
                train_X.append(X)
                train_y.append(y)
                cont_count += len(X)
        print(f"  Continuous: {cont_count} windows")

    X_train = np.concatenate(train_X)
    y_train = np.concatenate(train_y)
    print(f"  Total: {len(X_train)} windows")

    print("\n[2/6] Loading personalization data...")
    init_dir = TEST_DIR / "initialize"
    for init_file in ['Initialize1.csv', 'Initialize2.csv']:
        f = init_dir / init_file
        if f.exists():
            X, y = loader.load_continuous_gesture(str(f))
            if X is not None:
                X_train = np.concatenate([X_train, X])
                y_train = np.concatenate([y_train, y])
                print(f"  Added {len(X)} from {init_file}")

    print("\n[3/6] Loading test data...")
    test_data = {}
    for gid in range(1, 14):
        emg_dir = TEST_DIR / str(gid) / "emg"
        if emg_dir.exists():
            for f in emg_dir.glob("*.csv"):
                X, _ = loader.load_single_gesture(str(f), gid - 1)
                if X is not None:
                    test_data[gid - 1] = X
                    print(f"  Gesture {gid}: {len(X)} windows")
                    break

    print("\n[4/6] Augmenting data...")
    X_aug, y_aug = augment_data(X_train, y_train, config.num_augments)
    print(f"  {len(X_train)} -> {len(X_aug)} windows")

    print("\n[5/6] Training model...")
    classifier = LifelongEMGClassifier(config)
    classifier.fit(X_aug, y_aug)

    print("\n[6/6] Evaluating model...")
    results = evaluate_model(classifier, test_data, config)

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Accuracy:       {results['accuracy']*100:.1f}%")
    print(f"  Group Accuracy: {results['group_accuracy']*100:.1f}%")
    print(f"  Precision:      {results['precision']*100:.1f}%")
    print(f"  Recall:         {results['recall']*100:.1f}%")
    print(f"  F1 Score:       {results['f1']*100:.1f}%")

    print("\nSaving models...")
    save_dir = MODEL_DIR / "EMG"
    classifier.save(save_dir)

    print("\nGenerating figures...")
    plot_results(results, config, save_dir)

    with open(save_dir / 'emg_results.json', 'w') as f:
        json.dump({
            'accuracy': float(results['accuracy']),
            'group_accuracy': float(results['group_accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1': float(results['f1']),
            'class_recall': {str(k): float(v) for k, v in results['class_recall'].items()}
        }, f, indent=2)

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print(f"\nSaved files:")
    print(f"  - {save_dir / 'emg_model.pth'} (PyTorch model)")
    print(f"  - {save_dir / 'emg_ensemble.pkl'} (sklearn model)")
    print(f"  - {save_dir / 'emg_scaler.pkl'} (scaler)")
    print(f"  - {save_dir / 'emg_config.json'} (config)")
    print(f"  - {save_dir / 'emg_results.json'} (results)")
    print(f"  - {save_dir / 'emg_confusion_matrix.png'}")
    print(f"  - {save_dir / 'emg_class_recall.png'}")

    print("\nConfusion Groups for visual model hints:")
    for group in CONFUSION_GROUPS:
        names = [GESTURE_NAME_MAP[g] for g in group]
        print(f"  {names}")

    print("\nUsage in fusion model:")
    print("  classifier = LifelongEMGClassifier(config)")
    print("  classifier.load(MODEL_DIR / 'EMG')")
    print("  proba, hints = classifier.predict(X_window)")

    return classifier, results


if __name__ == '__main__':
    classifier, results = main()
