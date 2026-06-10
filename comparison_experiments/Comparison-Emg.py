import pickle
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.stats import skew, kurtosis, binomtest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

ROOT = Path("/home/luoyh/deep_learning_project")
DATASET_ROOT = ROOT / "Dataset"
OUR_EMG_SCALER = DATASET_ROOT / "models" / "emg_scaler.pkl"
OUR_EMG_MODEL = DATASET_ROOT / "models" / "emg_ensemble.pkl"
COMPARISON_DIR = DATASET_ROOT / "Model_others" / "EMG"
TEST_DIR = DATASET_ROOT / "test"
OUTPUT_DIR = ROOT / "data" / "Comparative experiment" / "Single Emg"

NUM_CLASSES = 13
WINDOW_SIZE = 180
STEP_SIZE = 22
BOOTSTRAP_ROUNDS = 1000
BOOTSTRAP_SEED = 2026

GESTURE_LABELS = (
    'Fist Clench', 'Finger Extension', 'Wrist Rotation', 'Finger Opposition',
    'Ball Squeeze', 'TheraPutty Pinch', 'Finger Pressing', 'Interlace Fingers',
    'Wrist Flexion-Extension', 'Finger Massage', 'Towel Scrunch', 'Hand Tapping', 'Piano Tap'
)

GESTURE_SHORT = (
    'Fist', 'Extension', 'Rotation', 'Opposition', 'Ball', 'Putty', 'Press',
    'Interlace', 'Flexion', 'Massage', 'Towel', 'Tapping', 'Piano'
)


def safe_normalize(arr):
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(arr, -1e6, 1e6)


def align_features(X_feat, expected):
    if X_feat.shape[1] < expected:
        return np.hstack([X_feat, np.zeros((X_feat.shape[0], expected - X_feat.shape[1]))])
    if X_feat.shape[1] > expected:
        return X_feat[:, :expected]
    return X_feat


def clean_metric_dict(res):
    out = {}
    for k, v in res.items():
        if k in {'y_pred'}:
            continue
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


class OurFeatureExtractor:
    def __init__(self, num_channels=6):
        self.num_channels = num_channels
        self.scales = [2, 4, 6, 8]

    def extract(self, X):
        if X.ndim == 2:
            X = X[np.newaxis, ...]
        N, T, C = X.shape
        eps = 1e-8
        X_min = X.min(axis=1, keepdims=True)
        X_max = X.max(axis=1, keepdims=True)
        X_norm = (X - X_min) / (X_max - X_min + eps)

        all_feats = []
        for c in range(min(self.num_channels, C)):
            ch = X_norm[:, :, c]
            feats = []
            feats.append(np.mean(ch, axis=1))
            feats.append(np.std(ch, axis=1))
            feats.append(np.median(ch, axis=1))
            feats.append(np.max(ch, axis=1))
            feats.append(np.min(ch, axis=1))
            feats.append(np.max(ch, axis=1) - np.min(ch, axis=1))
            for p in [10, 25, 75, 90]:
                feats.append(np.percentile(ch, p, axis=1))
            diff = np.diff(ch, axis=1)
            feats.append(np.mean(np.abs(diff), axis=1))
            feats.append(np.std(diff, axis=1))
            feats.append(np.sum(np.diff(np.sign(ch), axis=1) != 0, axis=1) / T)
            feats.append(np.sqrt(np.mean(ch ** 2, axis=1)))
            feats.append(np.sum(np.abs(diff), axis=1))
            feats.append(np.sum(np.diff(np.sign(diff), axis=1) != 0, axis=1) / T)

            fft_result = np.abs(fft(ch, axis=1))[:, :T // 2]
            fft_sum = np.sum(fft_result, axis=1, keepdims=True) + eps
            fft_norm = fft_result / fft_sum
            feats.append(np.argmax(fft_norm, axis=1) / (T // 2))
            n_bins = T // 2
            band_len = n_bins // 5
            for i in range(5):
                start = i * band_len
                end = (i + 1) * band_len if i < 4 else n_bins
                feats.append(np.sum(fft_norm[:, start:end], axis=1))
            freqs = np.arange(n_bins)
            feats.append(np.sum(fft_norm * freqs, axis=1))
            feats.append(-np.sum(fft_norm * np.log(fft_norm + eps), axis=1))

            for scale in self.scales:
                if T >= scale:
                    ds = ch[:, ::scale]
                    feats.append(np.mean(ds, axis=1))
                    feats.append(np.std(ds, axis=1))
            all_feats.extend(feats)

        return safe_normalize(np.column_stack(all_feats)).astype(np.float32)


class ComprehensiveExtractor:
    def __init__(self, num_channels=6):
        self.num_channels = num_channels

    def extract(self, X):
        if X.ndim == 2:
            X = X[np.newaxis, ...]
        N, T, C = X.shape
        all_features = []

        for c in range(min(self.num_channels, C)):
            ch = X[:, :, c]
            feats = []
            feats.append(np.mean(np.abs(ch), axis=1))
            feats.append(np.sqrt(np.mean(ch ** 2, axis=1)))
            feats.append(np.sum(np.abs(np.diff(ch, axis=1)), axis=1))
            feats.append(np.sum(np.diff(np.sign(ch), axis=1) != 0, axis=1) / T)
            diff = np.diff(ch, axis=1)
            feats.append(np.sum(np.diff(np.sign(diff), axis=1) != 0, axis=1) / T)
            feats.append(np.var(ch, axis=1))
            feats.append(np.sum(np.abs(ch), axis=1))
            feats.append(np.sqrt(np.mean(diff ** 2, axis=1)))
            feats.append(np.exp(np.mean(np.log(np.abs(ch) + 1e-10), axis=1)))
            threshold = np.mean(np.abs(ch)) * 0.5
            feats.append(np.mean(np.abs(ch) > threshold, axis=1))
            feats.append(np.sum(np.abs(diff) > 0.01, axis=1) / T)
            feats.append(np.sum(ch ** 2, axis=1))
            feats.append(np.abs(np.mean(ch ** 3, axis=1)))
            feats.append(np.mean(ch ** 4, axis=1))
            feats.append(np.abs(np.mean(ch ** 5, axis=1)))
            feats.append(np.mean(np.abs(diff), axis=1))
            feats.append(np.mean(np.abs(diff), axis=1))
            feats.append(np.log10(np.sum(np.abs(diff), axis=1) + 1e-10))
            feats.append(np.mean(ch, axis=1))
            feats.append(np.std(ch, axis=1))
            feats.append(np.median(ch, axis=1))
            feats.append(np.max(ch, axis=1))
            feats.append(np.min(ch, axis=1))
            feats.append(np.max(ch, axis=1) - np.min(ch, axis=1))
            feats.append(np.percentile(ch, 10, axis=1))
            feats.append(np.percentile(ch, 25, axis=1))
            feats.append(np.percentile(ch, 75, axis=1))
            feats.append(np.percentile(ch, 90, axis=1))
            feats.append(np.array([skew(ch[i]) for i in range(N)]))
            feats.append(np.array([kurtosis(ch[i]) for i in range(N)]))

            fft_result = np.abs(fft(ch, axis=1))[:, :T // 2]
            freqs = np.arange(T // 2)
            fft_sum = np.sum(fft_result, axis=1, keepdims=True) + 1e-10
            fft_norm = fft_result / fft_sum
            feats.append(np.sum(fft_norm * freqs, axis=1))
            cumsum = np.cumsum(fft_norm, axis=1)
            feats.append(np.argmax(cumsum >= 0.5, axis=1) / (T // 2))
            feats.append(np.argmax(fft_result, axis=1) / (T // 2))
            feats.append(np.sum(fft_result ** 2, axis=1))
            n_bins = T // 2
            band_len = n_bins // 5
            for i in range(5):
                start = i * band_len
                end = (i + 1) * band_len if i < 4 else n_bins
                feats.append(np.sum(fft_norm[:, start:end], axis=1))
            feats.append(-np.sum(fft_norm * np.log(fft_norm + 1e-10), axis=1))
            all_features.extend(feats)

        return safe_normalize(np.column_stack(all_features)).astype(np.float32)


def load_test_data():
    all_windows, all_labels = [], []
    for gid in range(NUM_CLASSES):
        emg_dir = TEST_DIR / str(gid + 1) / "emg"
        if not emg_dir.exists():
            continue
        for csv_path in emg_dir.glob("*.csv"):
            df = None
            for enc in ['utf-8', 'gbk', 'gb2312', 'latin1']:
                try:
                    df = pd.read_csv(csv_path, encoding=enc)
                    break
                except Exception:
                    continue
            if df is None:
                continue

            emg_cols = []
            for pattern in ['ch{}_fil', 'ch{}_raw']:
                cols = []
                for i in range(1, 7):
                    for c in df.columns:
                        if pattern.format(i) in c.lower():
                            cols.append(c)
                            break
                if len(cols) == 6:
                    emg_cols = cols
                    break

            if len(emg_cols) < 6:
                continue

            X_raw = safe_normalize(df[emg_cols].values.astype(np.float32))
            for i in range(0, len(X_raw) - WINDOW_SIZE + 1, STEP_SIZE):
                all_windows.append(X_raw[i:i + WINDOW_SIZE])
                all_labels.append(gid)

    return np.array(all_windows), np.array(all_labels)


def compute_metrics(y_true, y_pred):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'per_class_f1': f1_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0).tolist(),
        'per_class_precision': precision_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0).tolist(),
        'per_class_recall': recall_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0).tolist(),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES))).tolist(),
        'n_samples': int(len(y_pred)),
        'y_pred': np.asarray(y_pred, dtype=int)
    }


def bootstrap_ci(y_true, y_pred, rounds=BOOTSTRAP_ROUNDS, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    metrics = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}

    for _ in range(rounds):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        yp = y_pred[idx]
        metrics['accuracy'].append(accuracy_score(yt, yp))
        metrics['f1'].append(f1_score(yt, yp, average='macro', zero_division=0))
        metrics['precision'].append(precision_score(yt, yp, average='macro', zero_division=0))
        metrics['recall'].append(recall_score(yt, yp, average='macro', zero_division=0))

    out = {}
    for k, values in metrics.items():
        arr = np.asarray(values)
        out[k] = {
            'mean': float(np.mean(arr)),
            'low95': float(np.percentile(arr, 2.5)),
            'high95': float(np.percentile(arr, 97.5))
        }
    return out


def add_bootstrap_to_result(res, y_true):
    if res is None or 'y_pred' not in res:
        return res
    res['bootstrap_ci'] = bootstrap_ci(y_true, res['y_pred'])
    return res


def evaluate_our_model(X_test, y_test):
    try:
        with open(OUR_EMG_SCALER, 'rb') as f:
            scaler = pickle.load(f)
        with open(OUR_EMG_MODEL, 'rb') as f:
            data = pickle.load(f)
            model = data['model']

        extractor = OurFeatureExtractor(num_channels=6)
        X_feat = extractor.extract(X_test)
        expected = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else 192
        X_feat = align_features(X_feat, expected)
        X_scaled = scaler.transform(X_feat)
        y_pred = model.predict(X_scaled)
        res = compute_metrics(y_test, y_pred)
        res['n_features'] = int(expected)
        return add_bootstrap_to_result(res, y_test)
    except Exception:
        return None


def evaluate_comparison_model(model_path, X_test, y_test):
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            model = data.get('model')
            scaler = data.get('scaler')
        else:
            model, scaler = data, None

        if model is None:
            return None

        extractor = ComprehensiveExtractor(num_channels=6)
        X_feat = extractor.extract(X_test)
        expected = model.n_features_in_ if hasattr(model, 'n_features_in_') else X_feat.shape[1]
        X_feat = align_features(X_feat, expected)

        if scaler is not None:
            X_feat = scaler.transform(X_feat)

        y_pred = model.predict(X_feat)
        res = compute_metrics(y_test, y_pred)
        res['n_features'] = int(expected)
        return add_bootstrap_to_result(res, y_test)
    except Exception:
        return None


def get_clean_model_name(filename):
    name = filename.replace('.pkl', '').replace('_Paper3', '').replace('_Paper4', '')
    name = name.replace('_BVNet', '').replace('Paper', '')
    return name


def run_channel_ablation(X_test, y_test):
    results = {}
    base = evaluate_our_model(X_test, y_test)
    if base is not None:
        results['Full_SC-MSFE'] = base

    for ch in range(6):
        X_masked = X_test.copy()
        X_masked[:, :, ch] = 0.0
        res = evaluate_our_model(X_masked, y_test)
        if res is not None:
            results[f'w/o_CH{ch + 1}'] = res

    return results


def run_mcnemar_tests(results, y_true):
    our_name = None
    for name in results:
        if 'Ours' in name or 'SC-MSFE' in name:
            our_name = name
            break

    if our_name is None:
        return pd.DataFrame()

    rows = []
    y_true = np.asarray(y_true)
    our_pred = np.asarray(results[our_name]['y_pred'])
    our_correct = our_pred == y_true

    for name, res in results.items():
        if name == our_name:
            continue
        pred = np.asarray(res['y_pred'])
        other_correct = pred == y_true
        b = int(np.sum(our_correct & ~other_correct))
        c = int(np.sum(~our_correct & other_correct))
        n = b + c
        p_value = 1.0 if n == 0 else float(binomtest(min(b, c), n=n, p=0.5, alternative='two-sided').pvalue)
        rows.append({
            'Reference': our_name,
            'Compared model': name,
            'SC-MSFE correct / Other wrong': b,
            'SC-MSFE wrong / Other correct': c,
            'McNemar p-value': p_value,
            'Significant at 0.05': p_value < 0.05
        })

    return pd.DataFrame(rows)


def compute_class_sample_table(y_true):
    rows = []
    for i in range(NUM_CLASSES):
        rows.append({
            'Class ID': i,
            'Gesture': GESTURE_LABELS[i],
            'Short name': GESTURE_SHORT[i],
            'Test samples': int(np.sum(y_true == i))
        })
    return pd.DataFrame(rows)


def compute_per_class_table(results):
    rows = []
    for name, res in results.items():
        for i in range(NUM_CLASSES):
            rows.append({
                'Model': name,
                'Class ID': i,
                'Gesture': GESTURE_LABELS[i],
                'Short name': GESTURE_SHORT[i],
                'Precision (%)': res['per_class_precision'][i] * 100,
                'Recall (%)': res['per_class_recall'][i] * 100,
                'F1 (%)': res['per_class_f1'][i] * 100
            })
    return pd.DataFrame(rows)


def compute_top_confusions(results, topk=20):
    rows = []
    for name, res in results.items():
        cm = np.asarray(res['confusion_matrix'])
        for i in range(NUM_CLASSES):
            row_sum = np.sum(cm[i])
            if row_sum == 0:
                continue
            for j in range(NUM_CLASSES):
                if i == j:
                    continue
                count = int(cm[i, j])
                if count > 0:
                    rows.append({
                        'Model': name,
                        'True class': GESTURE_LABELS[i],
                        'Predicted class': GESTURE_LABELS[j],
                        'True short': GESTURE_SHORT[i],
                        'Pred short': GESTURE_SHORT[j],
                        'Count': count,
                        'Row-normalized ratio (%)': count / row_sum * 100
                    })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    return df.sort_values(['Model', 'Count', 'Row-normalized ratio (%)'], ascending=[True, False, False]).groupby('Model').head(topk)


class Visualizer:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 300
        self.ours_color = '#E74C3C'
        self.other_colors = plt.cm.Set2(np.linspace(0, 1, 8))

    def _get_color(self, name, idx):
        if 'Ours' in name or 'SC-MSFE' in name:
            return self.ours_color
        return self.other_colors[idx % len(self.other_colors)]

    def plot_metrics_comparison(self, results):
        metrics = ['accuracy', 'f1', 'precision', 'recall']
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        names = list(results.keys())

        for ax, metric in zip(axes, metrics):
            values = [results[n][metric] * 100 for n in names]
            colors = [self._get_color(n, i) for i, n in enumerate(names)]
            bars = ax.bar(range(len(names)), values, color=colors, edgecolor='black', linewidth=0.5)

            for i, n in enumerate(names):
                if 'Ours' in n or 'SC-MSFE' in n:
                    bars[i].set_edgecolor('darkred')
                    bars[i].set_linewidth(2)

            ax.set_ylabel(f'{metric.capitalize()} (%)', fontweight='bold')
            ax.set_title(f'{metric.capitalize()} Comparison', fontweight='bold')
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels([n.replace('_', '\n') for n in names], rotation=45, ha='right')
            ax.set_ylim(0, 100)
            ax.grid(axis='y', linestyle='--', alpha=0.3)

            for i, v in enumerate(values):
                ax.text(i, v + 1.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'comparison_metrics.png', bbox_inches='tight')
        plt.close()

    def plot_radar(self, results):
        metrics = ['accuracy', 'f1', 'precision', 'recall']
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        for i, (name, res) in enumerate(results.items()):
            values = [res[m] * 100 for m in metrics] + [res[metrics[0]] * 100]
            color = self._get_color(name, i)
            lw = 3 if ('Ours' in name or 'SC-MSFE' in name) else 1.5
            ax.plot(angles, values, 'o-', linewidth=lw, label=name, color=color)
            ax.fill(angles, values, alpha=0.25 if ('Ours' in name or 'SC-MSFE' in name) else 0.15, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.capitalize() for m in metrics], fontsize=12, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_title('SC-MSFE Model Performance Radar Chart', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.savefig(self.output_dir / 'radar_chart.png', bbox_inches='tight')
        plt.close()

    def plot_per_class(self, results):
        fig, axes = plt.subplots(3, 1, figsize=(18, 15))
        data_keys = [
            ('per_class_f1', 'Per-class F1-Score'),
            ('per_class_precision', 'Per-class Precision'),
            ('per_class_recall', 'Per-class Recall')
        ]

        names = list(results.keys())
        x = np.arange(NUM_CLASSES)
        width = 0.8 / len(names)

        for ax, (key, title) in zip(axes, data_keys):
            for i, name in enumerate(names):
                values = [v * 100 for v in results[name][key]]
                offset = (i - len(names) / 2 + 0.5) * width
                color = self._get_color(name, i)
                alpha = 0.9 if ('Ours' in name or 'SC-MSFE' in name) else 0.7
                ax.bar(x + offset, values, width, label=name, color=color, alpha=alpha)

            ax.set_ylabel(title.split()[-1] + ' (%)', fontweight='bold')
            ax.set_title(title, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(GESTURE_SHORT, rotation=45, ha='right')
            ax.legend(loc='upper right', fontsize=8)
            ax.set_ylim(0, 110)
            ax.grid(axis='y', linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'per_class_performance.png', bbox_inches='tight')
        plt.close()

    def plot_confusion_matrices(self, results):
        n = len(results)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 7 * rows))

        if n == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)

        for idx, (name, res) in enumerate(results.items()):
            r, c = idx // cols, idx % cols
            ax = axes[r, c]
            cm = np.array(res['confusion_matrix'])
            cm_sum = cm.sum(axis=1, keepdims=True)
            cm_sum[cm_sum == 0] = 1
            cm_norm = cm.astype('float') / cm_sum
            cmap = 'Reds' if ('Ours' in name or 'SC-MSFE' in name) else 'Blues'

            sns.heatmap(
                cm_norm,
                annot=True,
                fmt='.2f',
                cmap=cmap,
                ax=ax,
                xticklabels=GESTURE_SHORT,
                yticklabels=GESTURE_SHORT,
                cbar_kws={'shrink': 0.8},
                annot_kws={'size': 8}
            )

            color = 'darkred' if ('Ours' in name or 'SC-MSFE' in name) else 'black'
            ax.set_title(
                f'{name}\nAcc: {res["accuracy"] * 100:.1f}% | F1: {res["f1"] * 100:.1f}%',
                fontsize=12,
                fontweight='bold',
                color=color
            )
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.tick_params(axis='y', rotation=0, labelsize=8)

        for idx in range(n, rows * cols):
            axes[idx // cols, idx % cols].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrices.png', bbox_inches='tight')
        plt.close()

    def plot_improvement(self, results):
        our_name = None
        for name in results:
            if 'Ours' in name or 'SC-MSFE' in name:
                our_name = name
                break

        if not our_name:
            return

        our = results[our_name]
        others = {k: v for k, v in results.items() if k != our_name}

        if not others:
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        names = list(others.keys())
        x = np.arange(len(names))
        width = 0.35
        acc_imp = [(our['accuracy'] - others[n]['accuracy']) / (others[n]['accuracy'] + 1e-10) * 100 for n in names]
        f1_imp = [(our['f1'] - others[n]['f1']) / (others[n]['f1'] + 1e-10) * 100 for n in names]

        ax.bar(x - width / 2, acc_imp, width, label='Accuracy Improvement', color='#3498DB', alpha=0.8)
        ax.bar(x + width / 2, f1_imp, width, label='F1 Improvement', color='#E74C3C', alpha=0.8)
        ax.set_ylabel('Relative Improvement (%)', fontweight='bold')
        ax.set_title(f'Performance Improvement of {our_name} over Other Models', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace('_', '\n') for n in names], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        for i, (a, f) in enumerate(zip(acc_imp, f1_imp)):
            ax.text(i - width / 2, a + 2, f'{a:.1f}%', ha='center', fontsize=8)
            ax.text(i + width / 2, f + 2, f'{f:.1f}%', ha='center', fontsize=8)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'improvement_chart.png', bbox_inches='tight')
        plt.close()

    def plot_channel_ablation(self, channel_results):
        if len(channel_results) == 0:
            return

        names = list(channel_results.keys())
        acc = [channel_results[n]['accuracy'] * 100 for n in names]
        f1 = [channel_results[n]['f1'] * 100 for n in names]
        x = np.arange(len(names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width / 2, acc, width, label='Accuracy', color='#3498DB', alpha=0.85)
        ax.bar(x + width / 2, f1, width, label='F1-score', color='#E74C3C', alpha=0.85)
        ax.set_ylabel('Performance (%)', fontweight='bold')
        ax.set_title('Test-time Channel Masking Analysis of SC-MSFE', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.legend()

        for i, v in enumerate(acc):
            ax.text(i - width / 2, v + 1.5, f'{v:.1f}', ha='center', fontsize=8)
        for i, v in enumerate(f1):
            ax.text(i + width / 2, v + 1.5, f'{v:.1f}', ha='center', fontsize=8)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'channel_ablation.png', bbox_inches='tight')
        plt.close()

    def save_results(self, results, y_true, channel_results=None):
        rows = []
        for name, res in results.items():
            ci = res.get('bootstrap_ci', {})
            rows.append({
                'Model': name,
                'Accuracy (%)': f"{res['accuracy'] * 100:.2f}",
                'Accuracy 95% CI (%)': f"[{ci.get('accuracy', {}).get('low95', np.nan) * 100:.2f}, {ci.get('accuracy', {}).get('high95', np.nan) * 100:.2f}]",
                'F1 (%)': f"{res['f1'] * 100:.2f}",
                'F1 95% CI (%)': f"[{ci.get('f1', {}).get('low95', np.nan) * 100:.2f}, {ci.get('f1', {}).get('high95', np.nan) * 100:.2f}]",
                'Precision (%)': f"{res['precision'] * 100:.2f}",
                'Recall (%)': f"{res['recall'] * 100:.2f}",
                'Features': res.get('n_features', '-'),
                'Samples': res['n_samples']
            })

        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / 'results_summary.csv', index=False)

        with open(self.output_dir / 'results_table.tex', 'w') as f:
            f.write(df.to_latex(index=False, caption='SC-MSFE EMG Model Comparison with Bootstrap Confidence Intervals', label='tab:sc_msfe_emg'))

        json_out = {k: clean_metric_dict(v) for k, v in results.items()}
        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(json_out, f, indent=2)

        np.savez(
            self.output_dir / 'predictions.npz',
            y_true=np.asarray(y_true),
            **{f'pred_{k}': np.asarray(v['y_pred']) for k, v in results.items()}
        )

        class_samples = compute_class_sample_table(y_true)
        class_samples.to_csv(self.output_dir / 'class_sample_counts.csv', index=False)

        per_class_table = compute_per_class_table(results)
        per_class_table.to_csv(self.output_dir / 'per_class_metrics.csv', index=False)

        top_confusions = compute_top_confusions(results)
        top_confusions.to_csv(self.output_dir / 'top_confusions.csv', index=False)

        mcnemar_df = run_mcnemar_tests(results, y_true)
        mcnemar_df.to_csv(self.output_dir / 'mcnemar_tests.csv', index=False)

        if channel_results is not None and len(channel_results) > 0:
            channel_rows = []
            full_acc = channel_results.get('Full_SC-MSFE', {}).get('accuracy', np.nan)
            full_f1 = channel_results.get('Full_SC-MSFE', {}).get('f1', np.nan)

            for name, res in channel_results.items():
                channel_rows.append({
                    'Variant': name,
                    'Accuracy (%)': f"{res['accuracy'] * 100:.2f}",
                    'F1 (%)': f"{res['f1'] * 100:.2f}",
                    'Precision (%)': f"{res['precision'] * 100:.2f}",
                    'Recall (%)': f"{res['recall'] * 100:.2f}",
                    'Accuracy drop (%)': f"{(full_acc - res['accuracy']) * 100:.2f}" if not np.isnan(full_acc) else '-',
                    'F1 drop (%)': f"{(full_f1 - res['f1']) * 100:.2f}" if not np.isnan(full_f1) else '-'
                })

            channel_df = pd.DataFrame(channel_rows)
            channel_df.to_csv(self.output_dir / 'channel_ablation_results.csv', index=False)

            with open(self.output_dir / 'channel_ablation_table.tex', 'w') as f:
                f.write(channel_df.to_latex(index=False, caption='Test-time Channel Masking Analysis of SC-MSFE', label='tab:sc_msfe_channel_ablation'))

        print(df.to_string(index=False))
        print(str(self.output_dir))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_test, y_test = load_test_data()

    if len(X_test) == 0:
        print("No test data found.")
        return

    results = {}

    res = evaluate_our_model(X_test, y_test)
    if res:
        results['Ours_SC-MSFE'] = res

    if COMPARISON_DIR.exists():
        for model_path in sorted(COMPARISON_DIR.glob("*.pkl")):
            if 'scaler' in model_path.name.lower():
                continue
            clean_name = get_clean_model_name(model_path.name)
            res = evaluate_comparison_model(model_path, X_test, y_test)
            if res:
                results[clean_name] = res

    if not results:
        print("No models evaluated successfully.")
        return

    channel_results = run_channel_ablation(X_test, y_test)

    viz = Visualizer(OUTPUT_DIR)
    viz.plot_metrics_comparison(results)
    viz.plot_radar(results)
    viz.plot_per_class(results)
    viz.plot_confusion_matrices(results)
    viz.plot_improvement(results)
    viz.plot_channel_ablation(channel_results)
    viz.save_results(results, y_test, channel_results)


if __name__ == '__main__':
    main()
