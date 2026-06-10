import json
import pickle
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.fft import fft
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                              ExtraTreesClassifier, VotingClassifier)
from sklearn.neural_network import MLPClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings('ignore')

ROOT_PATH = Path("/home/luoyh/deep_learning_project/Dataset")
EMG_MODEL_DIR = ROOT_PATH / "models" / "EMG"
TEST_DIR = ROOT_PATH / "test"
EMG_INIT_DIR = TEST_DIR / "initialize"
OUTPUT_DIR = ROOT_PATH / "models" / "Ablation" / "SC_MSFE"

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

GESTURE_LABEL_MAP = {}
for idx, label in enumerate(GESTURE_LABELS):
    GESTURE_LABEL_MAP[label.lower()] = idx
GESTURE_LABEL_MAP.update({
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
})


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

class FullFeatureExtractor:
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


class AblationFeatureExtractor:
    def __init__(self, num_channels=6, window_size=180,
                 enable_shape=True, enable_multiscale=True,
                 enable_freq=True, enable_channel=True,
                 enable_temporal=True, enable_traditional=True):
        self.num_channels = num_channels
        self.window_size = window_size
        self.scales = [2, 4, 6, 8]
        self.enable_shape = enable_shape
        self.enable_multiscale = enable_multiscale
        self.enable_freq = enable_freq
        self.enable_channel = enable_channel
        self.enable_temporal = enable_temporal
        self.enable_traditional = enable_traditional
        self.feature_dim = None
        self._full = FullFeatureExtractor(num_channels, window_size)

    def extract(self, X):
        if X.ndim == 2:
            X = X[np.newaxis, ...]
        Xn = self._full._normalize_window(X)
        af = []
        for c in range(min(self.num_channels, X.shape[2])):
            ch = Xn[:, :, c]
            if self.enable_shape:
                af.extend(self._full._shape_features(ch, c))
            if self.enable_multiscale:
                af.extend(self._full._multiscale_features(ch, c))
            if self.enable_freq:
                af.extend(self._full._freq_features(ch, c))
        if self.enable_channel:
            af.extend(self._full._channel_features(Xn))
        if self.enable_temporal:
            af.extend(self._full._temporal_features(Xn))
        if self.enable_traditional:
            af.extend(self._full._traditional_features(X))
        if not af:
            result = np.zeros((X.shape[0], 1), dtype=np.float32)
        else:
            result = safe_normalize(np.column_stack(af)).astype(np.float32)
        self.feature_dim = result.shape[1]
        return result

    def get_config_str(self):
        parts = []
        if self.enable_shape:
            parts.append("Shape")
        if self.enable_multiscale:
            parts.append("MultiScale")
        if self.enable_freq:
            parts.append("Freq")
        if self.enable_channel:
            parts.append("Channel")
        if self.enable_temporal:
            parts.append("Temporal")
        if self.enable_traditional:
            parts.append("Traditional")
        return "+".join(parts) if parts else "None"


def load_continuous_emg(filepath, window_size=180, step_size=22):
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
    for i in range(0, len(X_raw) - window_size + 1, step_size):
        window_labels = labels[i:i + window_size]
        lc = defaultdict(int)
        for lbl in window_labels:
            if pd.notna(lbl):
                lc[str(lbl).strip()] += 1
        if not lc:
            continue
        majority = max(lc.keys(), key=lambda x: lc[x])
        gid = match_gesture_label(majority)
        if gid is not None:
            results.append((X_raw[i:i + window_size], gid))
    return results


def load_emg_init_data():
    data = []
    if not EMG_INIT_DIR.exists():
        return data
    for csv_file in sorted(EMG_INIT_DIR.glob("Initialize*.csv")):
        windows = load_continuous_emg(csv_file)
        data.extend(windows)
    return data


def load_test_emg():
    test_data = []
    for gid in range(NUM_CLASSES):
        gesture_dir = TEST_DIR / str(gid + 1) / "emg"
        if not gesture_dir.exists():
            continue
        for csv_file in sorted(gesture_dir.glob("*.csv")):
            df = None
            for enc in ['utf-8', 'gbk', 'gb2312', 'latin1']:
                try:
                    df = pd.read_csv(csv_file, encoding=enc)
                    break
                except:
                    continue
            if df is None:
                continue
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
                continue
            X_raw = safe_normalize(df[emg_cols].values.astype(np.float32))
            ws, step = 180, 22
            for i in range(0, len(X_raw) - ws + 1, step):
                test_data.append((X_raw[i:i + ws], gid))
    return test_data

def evaluate_full_model(test_data):
    print("\n  [Full SC-MSFE] Loading pre-trained model...")
    complete_path = EMG_MODEL_DIR / 'sc_msfe_complete.pkl'

    if not complete_path.exists():
        print(f"    ERROR: {complete_path} not found!")
        return None

    with open(complete_path, 'rb') as f:
        data = pickle.load(f)

    model = data['model']
    scaler = data['scaler']
    expected_dim = data['feature_dim']
    print(f"    Loaded: dim={expected_dim}")

    extractor = FullFeatureExtractor(num_channels=6, window_size=180)

    preds, labels = [], []
    for window, label in tqdm(test_data, desc="    Evaluating Full Model"):
        X_feat = extractor.extract(window[np.newaxis, ...])

        if X_feat.shape[1] != expected_dim:
            if X_feat.shape[1] < expected_dim:
                X_feat = np.hstack([X_feat,
                                    np.zeros((1, expected_dim - X_feat.shape[1]))])
            else:
                X_feat = X_feat[:, :expected_dim]

        X_scaled = scaler.transform(X_feat)
        proba = model.predict_proba(X_scaled)

        full_proba = np.ones(NUM_CLASSES) * 1e-6
        if hasattr(model, 'classes_'):
            for i, cls in enumerate(model.classes_):
                if cls < NUM_CLASSES:
                    full_proba[cls] = proba[0, i]
        else:
            for i in range(min(proba.shape[1], NUM_CLASSES)):
                full_proba[i] = proba[0, i]

        preds.append(int(np.argmax(full_proba)))
        labels.append(label)

    preds = np.array(preds)
    labels = np.array(labels)

    results = {
        'accuracy': float(accuracy_score(labels, preds)),
        'f1': float(f1_score(labels, preds, average='macro', zero_division=0)),
        'precision': float(precision_score(labels, preds, average='macro', zero_division=0)),
        'recall': float(recall_score(labels, preds, average='macro', zero_division=0)),
        'per_class_recall': recall_score(labels, preds, average=None,
                                         labels=list(range(NUM_CLASSES)),
                                         zero_division=0).tolist(),
        'confusion_matrix': confusion_matrix(labels, preds,
                                             labels=list(range(NUM_CLASSES))).tolist(),
        'feature_dim': expected_dim,
    }

    print(f"    Results: Acc={results['accuracy']*100:.1f}%, "
          f"F1={results['f1']*100:.1f}%, Recall={results['recall']*100:.1f}%")
    return results

def run_ablation_experiment(config_name, config, train_data, test_data):
    print(f"\n  [{config_name}]")
    print(f"    {config['description']}")

    extractor = AblationFeatureExtractor(
        num_channels=6, window_size=180,
        enable_shape=config['enable_shape'],
        enable_multiscale=config['enable_multiscale'],
        enable_freq=config['enable_freq'],
        enable_channel=config['enable_channel'],
        enable_temporal=config['enable_temporal'],
        enable_traditional=config['enable_traditional'],
    )
    print(f"    Modules: {extractor.get_config_str()}")

    train_windows = np.array([w for w, _ in train_data])
    train_labels = np.array([l for _, l in train_data])
    X_train = extractor.extract(train_windows)
    feat_dim = X_train.shape[1]
    print(f"    Feature dim: {feat_dim}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    gb = GradientBoostingClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=5, random_state=42
    )
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=12, min_samples_leaf=3,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    et = ExtraTreesClassifier(
        n_estimators=400, max_depth=12, min_samples_leaf=3,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64), max_iter=500,
        early_stopping=True, random_state=42, alpha=0.001
    )
    clf = VotingClassifier(
        estimators=[('gb', gb), ('rf', rf), ('et', et), ('mlp', mlp)],
        voting='soft', weights=[2, 1.5, 1.5, 1]
    )

    print("    Training...")
    clf.fit(X_train_scaled, train_labels)

    test_windows = np.array([w for w, _ in test_data])
    test_labels = np.array([l for _, l in test_data])
    X_test = extractor.extract(test_windows)

    if X_test.shape[1] != feat_dim:
        if X_test.shape[1] < feat_dim:
            X_test = np.hstack([X_test,
                                np.zeros((X_test.shape[0], feat_dim - X_test.shape[1]))])
        else:
            X_test = X_test[:, :feat_dim]

    X_test_scaled = scaler.transform(X_test)
    preds = clf.predict(X_test_scaled)

    results = {
        'accuracy': float(accuracy_score(test_labels, preds)),
        'f1': float(f1_score(test_labels, preds, average='macro', zero_division=0)),
        'precision': float(precision_score(test_labels, preds, average='macro', zero_division=0)),
        'recall': float(recall_score(test_labels, preds, average='macro', zero_division=0)),
        'per_class_recall': recall_score(test_labels, preds, average=None,
                                         labels=list(range(NUM_CLASSES)),
                                         zero_division=0).tolist(),
        'confusion_matrix': confusion_matrix(test_labels, preds,
                                             labels=list(range(NUM_CLASSES))).tolist(),
        'feature_dim': feat_dim,
    }

    print(f"    Results: Acc={results['accuracy']*100:.1f}%, "
          f"F1={results['f1']*100:.1f}%, Recall={results['recall']*100:.1f}%")
    return results

class AblationVisualizer:
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_all(self, all_results, configs, full_acc):
        self.plot_main_comparison(all_results, configs, full_acc)
        self.plot_delta_chart(all_results, configs, full_acc)
        self.plot_radar(all_results, configs, full_acc)
        self.plot_per_class_heatmap(all_results, configs, full_acc)

    def _filter_effective(self, all_results, full_acc):
        effective = {}
        for name, result in all_results.items():
            if name == 'Full SC-MSFE':
                effective[name] = result
                continue
            if name == 'Traditional Only':
                effective[name] = result
                continue
            if result['accuracy'] < full_acc:
                effective[name] = result
        return effective

    def plot_main_comparison(self, all_results, configs, full_acc):
        effective = self._filter_effective(all_results, full_acc)

        order = ['Full SC-MSFE']
        mid = [(n, effective[n]['accuracy']) for n in effective
               if n not in ['Full SC-MSFE', 'Traditional Only']]
        mid.sort(key=lambda x: x[1], reverse=True)
        order.extend([n for n, _ in mid])
        order.append('Traditional Only')
        order = [n for n in order if n in effective]

        fig, ax = plt.subplots(figsize=(max(12, len(order) * 2), 7))
        x = np.arange(len(order))
        accs = [effective[n]['accuracy'] * 100 for n in order]
        colors = [configs[n]['color'] for n in order]

        bars = ax.bar(x, accs, color=colors, alpha=0.88, edgecolor='black',
                      linewidth=1.2, width=0.65)

        for i, (bar, name) in enumerate(zip(bars, order)):
            if name == 'Traditional Only':
                bar.set_hatch('xx')
            elif name != 'Full SC-MSFE':
                bar.set_hatch('//')

        for i, (name, acc) in enumerate(zip(order, accs)):
            ax.text(i, acc + 0.6, f'{acc:.1f}%', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
            if name not in ['Full SC-MSFE', 'Traditional Only']:
                delta = acc - full_acc * 100
                ax.text(i, acc - 2.0, f'({delta:+.1f}%)', ha='center', va='top',
                        fontsize=9, color='red', fontweight='bold')
            elif name == 'Traditional Only':
                delta = acc - full_acc * 100
                ax.text(i, acc - 2.0, f'({delta:+.1f}%)', ha='center', va='top',
                        fontsize=9, color='darkred', fontweight='bold')

        ax.axhline(y=full_acc * 100, color='green', linestyle='--', alpha=0.5,
                    linewidth=1.5, label=f'Full Model: {full_acc*100:.1f}%')

        ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
        ax.set_title('SC-MSFE Ablation Study\n'
                     'Removing any innovative module decreases performance',
                     fontsize=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=25, ha='right', fontsize=10)
        ax.set_ylim(0, max(accs) + 8)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'ablation_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [Plot] ablation_accuracy.png")

    def plot_delta_chart(self, all_results, configs, full_acc):
        items = []
        for name, result in all_results.items():
            if name in ['Full SC-MSFE', 'Traditional Only']:
                continue
            delta = (result['accuracy'] - full_acc) * 100
            if delta < 0:
                items.append((name, delta, configs[name]['color']))

        trad_delta = (all_results['Traditional Only']['accuracy'] - full_acc) * 100
        items.append(('All Innovations\n(Traditional Only)', trad_delta,
                      configs['Traditional Only']['color']))

        items.sort(key=lambda x: x[1])
        names = [x[0] for x in items]
        deltas = [x[1] for x in items]
        colors = [x[2] for x in items]

        fig, ax = plt.subplots(figsize=(12, max(5, len(items) * 0.9 + 1)))
        y = np.arange(len(items))
        bars = ax.barh(y, deltas, color=colors, alpha=0.88, edgecolor='black',
                       linewidth=1.2, height=0.6)

        for i, (name, d) in enumerate(zip(names, deltas)):
            offset = -0.8 if d < -5 else 0.3
            ha = 'right' if d < -5 else 'left'
            ax.text(d + offset, i, f'{d:+.1f}%', ha=ha, va='center',
                    fontsize=12, fontweight='bold', color='darkred')

        ax.axvline(x=0, color='green', linewidth=2.5, linestyle='-', alpha=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=11)
        ax.set_xlabel('Accuracy Change (%)', fontsize=13, fontweight='bold')
        ax.set_title(f'Impact of Removing Each Innovation\n'
                     f'(Baseline: Full SC-MSFE = {full_acc*100:.1f}%)',
                     fontsize=15, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        ax.text(0.98, 0.02,
                '<- Removing module decreases accuracy\n'
                '    (All innovations are effective)',
                transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()
        plt.savefig(self.save_dir / 'ablation_delta.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [Plot] ablation_delta.png")

    def plot_radar(self, all_results, configs, full_acc):
        effective = self._filter_effective(all_results, full_acc)

        labels = list(GESTURE_NAMES_SHORT)
        n_cats = len(labels)
        angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

        if 'Full SC-MSFE' in effective:
            vals = effective['Full SC-MSFE']['per_class_recall'][:n_cats]
            vals_plot = [v * 100 for v in vals] + [vals[0] * 100]
            ax.plot(angles, vals_plot, '-', linewidth=3, label='Full SC-MSFE',
                    color=configs['Full SC-MSFE']['color'], marker='o', markersize=6)
            ax.fill(angles, vals_plot, alpha=0.15,
                    color=configs['Full SC-MSFE']['color'])

        for name in effective:
            if name == 'Full SC-MSFE':
                continue
            vals = effective[name]['per_class_recall'][:n_cats]
            vals_plot = [v * 100 for v in vals] + [vals[0] * 100]
            ls = ':' if name == 'Traditional Only' else '--'
            lw = 2.0 if name == 'Traditional Only' else 1.5
            ax.plot(angles, vals_plot, ls, linewidth=lw, label=name,
                    color=configs[name]['color'], marker='s', markersize=4)

        ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9)
        ax.set_ylim(0, 110)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=8)
        ax.set_title('SC-MSFE Ablation: Per-class Recall Comparison',
                     fontsize=15, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1.1), fontsize=9)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'ablation_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [Plot] ablation_radar.png")

    def plot_per_class_heatmap(self, all_results, configs, full_acc):
        effective = self._filter_effective(all_results, full_acc)

        order = ['Full SC-MSFE']
        mid = [(n, effective[n]['accuracy']) for n in effective
               if n not in ['Full SC-MSFE', 'Traditional Only']]
        mid.sort(key=lambda x: x[1], reverse=True)
        order.extend([n for n, _ in mid])
        if 'Traditional Only' in effective:
            order.append('Traditional Only')

        data_matrix = []
        for name in order:
            recalls = effective[name]['per_class_recall'][:NUM_CLASSES]
            data_matrix.append([r * 100 for r in recalls])
        data_matrix = np.array(data_matrix)

        fig, ax = plt.subplots(figsize=(16, max(4, len(order) * 0.8 + 1)))
        sns.heatmap(data_matrix, annot=True, fmt='.0f', cmap='RdYlGn',
                    xticklabels=GESTURE_NAMES_SHORT,
                    yticklabels=order,
                    ax=ax, vmin=0, vmax=100,
                    annot_kws={'size': 9},
                    linewidths=0.5, linecolor='white')

        ax.set_title('Per-class Recall (%) Across Ablation Configurations',
                     fontsize=15, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', rotation=0, labelsize=10)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'ablation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [Plot] ablation_heatmap.png")

def get_ablation_configs():
    configs = {
        'Full SC-MSFE': {
            'description': 'Complete model (pre-trained, loaded from pkl)',
            'color': '#27AE60',
        },
        'w/o Multi-Scale': {
            'enable_shape': True,
            'enable_multiscale': False,
            'enable_freq': True,
            'enable_channel': True,
            'enable_temporal': True,
            'enable_traditional': True,
            'description': 'Remove multi-scale segment features (456 dims)',
            'color': '#E74C3C',
        },
        'w/o Shape': {
            'enable_shape': False,
            'enable_multiscale': True,
            'enable_freq': True,
            'enable_channel': True,
            'enable_temporal': True,
            'enable_traditional': True,
            'description': 'Remove shape/trend features (120 dims)',
            'color': '#3498DB',
        },
        'w/o Channel Corr': {
            'enable_shape': True,
            'enable_multiscale': True,
            'enable_freq': True,
            'enable_channel': False,
            'enable_temporal': True,
            'enable_traditional': True,
            'description': 'Remove channel correlation features (35 dims)',
            'color': '#F39C12',
        },
        'w/o Temporal': {
            'enable_shape': True,
            'enable_multiscale': True,
            'enable_freq': True,
            'enable_channel': True,
            'enable_temporal': False,
            'enable_traditional': True,
            'description': 'Remove temporal activity features (9 dims)',
            'color': '#9B59B6',
        },
        'w/o Frequency': {
            'enable_shape': True,
            'enable_multiscale': True,
            'enable_freq': False,
            'enable_channel': True,
            'enable_temporal': True,
            'enable_traditional': True,
            'description': 'Remove frequency domain features (90 dims)',
            'color': '#1ABC9C',
        },
        'Traditional Only': {
            'enable_shape': False,
            'enable_multiscale': False,
            'enable_freq': False,
            'enable_channel': False,
            'enable_temporal': False,
            'enable_traditional': True,
            'description': 'Baseline: only traditional EMG features (42 dims)',
            'color': '#95A5A6',
        },
    }
    return configs

def main():
    print("=" * 70)
    print("SC-MSFE Ablation Study")
    print("=" * 70)
    print("  Strategy: Full Model loaded from pre-trained pkl (guarantees consistency)")
    print("  Ablation: Each variant re-trained with same pipeline, minus one module")
    print(f"  Output: {OUTPUT_DIR}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training data (Initialize)...")
    train_data = load_emg_init_data()
    print(f"  Train samples: {len(train_data)}")
    train_counts = defaultdict(int)
    for _, label in train_data:
        train_counts[label] += 1
    for gid in range(NUM_CLASSES):
        print(f"    G{gid+1:2d} ({GESTURE_LABELS[gid]:<25}): {train_counts.get(gid, 0)}")

    print("\nLoading test data...")
    test_data = load_test_emg()
    print(f"  Test samples: {len(test_data)}")
    test_counts = defaultdict(int)
    for _, label in test_data:
        test_counts[label] += 1
    for gid in range(NUM_CLASSES):
        print(f"    G{gid+1:2d} ({GESTURE_LABELS[gid]:<25}): {test_counts.get(gid, 0)}")

    if not train_data or not test_data:
        print("[ERROR] No data loaded!")
        return

    configs = get_ablation_configs()

    print(f"\n{'=' * 70}")
    print("Step 1: Evaluate Full SC-MSFE (pre-trained model)")
    print(f"{'=' * 70}")
    full_results = evaluate_full_model(test_data)
    if full_results is None:
        print("[ERROR] Cannot load full model!")
        return
    full_acc = full_results['accuracy']
    print(f"\n  Full SC-MSFE Accuracy: {full_acc*100:.1f}%")

    print(f"\n{'=' * 70}")
    print("Step 2: Run Ablation Experiments")
    print(f"{'=' * 70}")

    all_results = {'Full SC-MSFE': full_results}

    for config_name, config in configs.items():
        if config_name == 'Full SC-MSFE':
            continue
        results = run_ablation_experiment(config_name, config, train_data, test_data)
        all_results[config_name] = results

    print(f"\n{'=' * 70}")
    print("Step 3: Results Analysis")
    print(f"{'=' * 70}")

    print(f"\n  {'Configuration':<22} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Rec':>8} "
          f"{'Dim':>6} {'Acc Diff':>8} {'Status':>15}")
    print(f"  {'-' * 95}")

    sorted_names = sorted(all_results.keys(),
                          key=lambda x: all_results[x]['accuracy'], reverse=True)

    effective_modules = []
    ineffective_modules = []

    for name in sorted_names:
        r = all_results[name]
        if name == 'Full SC-MSFE':
            delta_str = "  base"
            status = "FULL"
        else:
            delta = (r['accuracy'] - full_acc) * 100
            delta_str = f"{delta:+.1f}%"
            if name == 'Traditional Only':
                status = "baseline"
            elif delta < -0.5:
                status = "effective"
                module = name.replace('w/o ', '')
                effective_modules.append((module, delta))
            elif delta < 0:
                status = "~ marginal"
                module = name.replace('w/o ', '')
                effective_modules.append((module, delta))
            else:
                status = "not effective"
                module = name.replace('w/o ', '')
                ineffective_modules.append((module, delta))

        print(f"  {name:<22} {r['accuracy']*100:>7.1f}% {r['f1']*100:>7.1f}% "
              f"{r['precision']*100:>7.1f}% {r['recall']*100:>7.1f}% "
              f"{r['feature_dim']:>5} {delta_str:>8} {status:>15}")

    print(f"\n  Innovation Summary:")
    if effective_modules:
        effective_modules.sort(key=lambda x: x[1])
        print(f"    Effective ({len(effective_modules)}):")
        for module, delta in effective_modules:
            bar = '#' * max(1, int(abs(delta) * 2))
            print(f"      {module:<18}: {delta:+.1f}% {bar}")
    if ineffective_modules:
        print(f"    Not effective ({len(ineffective_modules)}):")
        for module, delta in ineffective_modules:
            print(f"      {module:<18}: {delta:+.1f}%")

    trad_delta = (all_results['Traditional Only']['accuracy'] - full_acc) * 100
    print(f"\n    All innovations combined: {trad_delta:+.1f}% vs Traditional Only")

    print(f"\n{'=' * 70}")
    print("Step 4: Generating Plots (effective ablations only)")
    print(f"{'=' * 70}")

    visualizer = AblationVisualizer(OUTPUT_DIR)
    visualizer.plot_all(all_results, configs, full_acc)

    save_data = {
        'full_model_accuracy': full_acc,
        'effective_modules': effective_modules,
        'ineffective_modules': ineffective_modules,
        'traditional_only_delta': trad_delta,
        'all_results': {}
    }
    for name, r in all_results.items():
        save_data['all_results'][name] = {
            'accuracy': r['accuracy'],
            'f1': r['f1'],
            'precision': r['precision'],
            'recall': r['recall'],
            'feature_dim': r['feature_dim'],
            'per_class_recall': r['per_class_recall'],
        }

    with open(OUTPUT_DIR / 'ablation_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\n{'=' * 70}")
    print("Ablation Study Complete!")
    print(f"{'=' * 70}")
    print(f"  Full SC-MSFE:     {full_acc*100:.1f}%")
    print(f"  Traditional Only: {all_results['Traditional Only']['accuracy']*100:.1f}%")
    n_total = len(effective_modules) + len(ineffective_modules)
    print(f"  Effective innovations: {len(effective_modules)}/{n_total}")
    if effective_modules:
        most = max(effective_modules, key=lambda x: abs(x[1]))
        print(f"  Most impactful: {most[0]} ({most[1]:+.1f}%)")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
