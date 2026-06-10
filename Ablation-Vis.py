import gc
import json
import re
import shutil
import threading
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "DejaVu Sans"

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class Config:
    DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
    DATA_DIR = Path("/home/luoyh/deep_learning_project/Dataset/origin_pic")
    TEST_DIR = Path("/home/luoyh/deep_learning_project/Dataset/test")
    TRAINED_DIR = Path("/home/luoyh/deep_learning_project/Dataset/ablation_study")
    OUTPUT_DIR = Path("/home/luoyh/deep_learning_project/data/Ablation experiment/EMG/VIS")
    VISUAL_MODEL_PATH = Path("/home/luoyh/deep_learning_project/Dataset/Model_others/Visual/True-Use/mvtf_best.pth")
    HAND_DETECT_PATH = Path("/home/luoyh/deep_learning_project/Dataset/models/best.pt")

    VAL_PERSONS = [5]
    NUM_CLASSES = 13
    SEQ_LEN = 16
    HIDDEN_DIM = 256
    BATCH_SIZE = 32
    WORKERS = 4
    PREFETCH = 3
    HAND_CONF = 0.25
    HAND_PADDING = 0.15
    OUTPUT_SIZE = 224

    GESTURE_NAMES = [
        "Fist",
        "Extension",
        "Rotation",
        "Opposition",
        "Ball",
        "Putty",
        "Press",
        "Interlace",
        "Flexion",
        "Massage",
        "Towel",
        "Tapping",
        "Piano",
    ]

    ABLATION_ORDER = [
        "Full",
        "wo_GrayVar",
        "wo_Motion",
        "wo_MultiScale",
        "wo_CausalAttn",
        "wo_HierAttn",
        "wo_FocalLoss",
        "wo_WeightAdj",
    ]

    ABLATION_DISPLAY = {
        "Full": "CAST-Net (Full)",
        "wo_GrayVar": "w/o GrayVariation",
        "wo_Motion": "w/o MotionModule",
        "wo_MultiScale": "w/o MultiScale",
        "wo_CausalAttn": "w/o CausalAttention",
        "wo_HierAttn": "w/o HierAttention",
        "wo_FocalLoss": "w/o FocalLoss",
        "wo_WeightAdj": "w/o WeightAdjust",
    }

    ABLATION_CONFIGS = {
        "Full": {
            "use_gray_variation": True,
            "use_motion_module": True,
            "use_multiscale": True,
            "use_causal_attention": True,
            "use_hier_attention": True,
        },
        "wo_GrayVar": {
            "use_gray_variation": False,
            "use_motion_module": True,
            "use_multiscale": True,
            "use_causal_attention": True,
            "use_hier_attention": True,
        },
        "wo_Motion": {
            "use_gray_variation": True,
            "use_motion_module": False,
            "use_multiscale": True,
            "use_causal_attention": True,
            "use_hier_attention": True,
        },
        "wo_MultiScale": {
            "use_gray_variation": True,
            "use_motion_module": True,
            "use_multiscale": False,
            "use_causal_attention": True,
            "use_hier_attention": True,
        },
        "wo_CausalAttn": {
            "use_gray_variation": True,
            "use_motion_module": True,
            "use_multiscale": True,
            "use_causal_attention": False,
            "use_hier_attention": True,
        },
        "wo_HierAttn": {
            "use_gray_variation": True,
            "use_motion_module": True,
            "use_multiscale": True,
            "use_causal_attention": True,
            "use_hier_attention": False,
        },
        "wo_FocalLoss": {
            "use_gray_variation": True,
            "use_motion_module": True,
            "use_multiscale": True,
            "use_causal_attention": True,
            "use_hier_attention": True,
        },
        "wo_WeightAdj": {
            "use_gray_variation": True,
            "use_motion_module": True,
            "use_multiscale": True,
            "use_causal_attention": True,
            "use_hier_attention": True,
        },
    }


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class HandDetector:
    def __init__(self, model_path, device):
        self.model = None
        self.loaded = False
        self.cache = {}
        self.lock = threading.Lock()

        if YOLO is None or not Path(model_path).exists():
            return

        try:
            self.model = YOLO(str(model_path))
            self.loaded = True
            print(f"  [HandDetector] loaded: {model_path}")
        except RuntimeError as exc:
            print(f"  [HandDetector] unavailable: {exc}")
        except Exception as exc:
            print(f"  [HandDetector] unavailable: {exc}")

    def detect_and_crop(self, image_path, output_size=224, conf=0.25, padding=0.15):
        with self.lock:
            cached = self.cache.get(image_path)
            if cached is not None:
                return cached

        image = cv2.imread(str(image_path))
        if image is None:
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        crop = None

        if self.loaded:
            try:
                results = self.model.predict(source=str(image_path), conf=conf, verbose=False)
                if results and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    box = boxes.xyxy[boxes.conf.argmax()].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box)
                    pad_w = int((x2 - x1) * padding)
                    pad_h = int((y2 - y1) * padding)
                    x1 = max(0, x1 - pad_w)
                    y1 = max(0, y1 - pad_h)
                    x2 = min(width, x2 + pad_w)
                    y2 = min(height, y2 + pad_h)
                    candidate = image[y1:y2, x1:x2]
                    if candidate.size:
                        crop = cv2.resize(candidate, (output_size, output_size))
            except (RuntimeError, ValueError, cv2.error):
                crop = None

        if crop is None:
            side = min(height, width)
            top = (height - side) // 2
            left = (width - side) // 2
            crop = cv2.resize(image[top : top + side, left : left + side], (output_size, output_size))

        with self.lock:
            if len(self.cache) < 5000:
                self.cache[image_path] = crop

        return crop

    def clear_cache(self):
        with self.lock:
            self.cache.clear()


class GrayVariation(nn.Module):
    def __init__(self, eta=16):
        super().__init__()
        self.eta = eta
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        xmin = x.amin(dim=(2, 3), keepdim=True)
        xmax = x.amax(dim=(2, 3), keepdim=True)
        x_norm = (x - xmin) / (xmax - xmin + 1e-8)
        x_quant = torch.floor(x_norm * self.eta) / self.eta
        return self.alpha * x_quant + (1 - self.alpha) * x_norm


class IdentityModule(nn.Module):
    def forward(self, x):
        return x


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
        velocity = torch.zeros_like(x)
        acceleration = torch.zeros_like(x)
        velocity[:, 1:] = x[:, 1:] - x[:, :-1]
        acceleration[:, 2:] = velocity[:, 2:] - velocity[:, 1:-1]
        features = torch.cat(
            [x, self.dropout(self.vel_proj(velocity)), self.dropout(self.acc_proj(acceleration))], dim=-1
        )
        return self.norm(x + self.gate(features) * self.fuse(features))


class MultiScaleBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        branch_dim = dim // 4
        self.branch1 = nn.Conv1d(dim, branch_dim, 1)
        self.branch3 = nn.Sequential(nn.Conv1d(dim, branch_dim, 1), nn.Conv1d(branch_dim, branch_dim, 3, padding=1))
        self.branch5 = nn.Sequential(nn.Conv1d(dim, branch_dim, 1), nn.Conv1d(branch_dim, branch_dim, 5, padding=2))
        self.branch_pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(dim, branch_dim, 1))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.size(1)
        x_t = x.transpose(1, 2)
        features = torch.cat(
            [
                self.branch1(x_t),
                self.branch3(x_t),
                self.branch5(x_t),
                self.branch_pool(x_t).expand(-1, -1, seq_len),
            ],
            dim=1,
        ).transpose(1, 2)
        return self.norm(self.dropout(features) + x)


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
        batch, seq_len, dim = x.shape
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention = F.softmax((query @ key.transpose(-2, -1)) * self.scale, dim=-1)
        attention = self.dropout(attention)
        features = (attention @ value).transpose(1, 2).reshape(batch, seq_len, dim)
        return self.norm(x + self.proj(features))


class SpatialAttention(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.size(1)
        dim = x.size(2)
        query = self.query(x.mean(dim=1, keepdim=True))
        scores = torch.bmm(query, self.key(x).transpose(-2, -1)) / (dim ** 0.5)
        attention = self.dropout(F.softmax(scores, dim=-1))
        features = torch.bmm(attention, self.value(x)).expand(-1, seq_len, -1)
        return self.norm(x + features)


class HierarchicalModule(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.frame_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.frame_norm = nn.LayerNorm(dim)
        self.seg_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.seg_norm = nn.LayerNorm(dim)

    def forward(self, x):
        frame_features, _ = self.frame_attn(x, x, x)
        x = self.frame_norm(x + frame_features)
        segment_features, _ = self.seg_attn(x.mean(dim=1, keepdim=True), x, x)
        return self.seg_norm(x + segment_features.expand(-1, x.size(1), -1))


class VisualBackbone(nn.Module):
    def __init__(self, num_classes=13, hidden_dim=256, num_heads=4, dropout=0.3):
        super().__init__()
        self.gvar = GrayVariation(eta=16)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.proj = nn.Sequential(nn.Linear(512, hidden_dim), nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.motion = MotionModule(hidden_dim, dropout)
        self.multiscale = MultiScaleBlock(hidden_dim, dropout)
        self.temporal = TemporalAttention(hidden_dim, num_heads, dropout)
        self.spatial = SpatialAttention(hidden_dim, dropout)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.gru_proj = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.seq_attn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1))
        self.hier = HierarchicalModule(hidden_dim, num_heads, dropout)
        self.fusion = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        batch, seq_len, channels, height, width = x.shape
        x = self.gvar(x.view(batch * seq_len, channels, height, width))
        x = F.adaptive_avg_pool2d(self.encoder(x), (1, 1)).view(batch, seq_len, -1)
        x = self.proj(x)
        x = self.spatial(self.temporal(self.multiscale(self.motion(x))))
        self.gru.flatten_parameters()
        recurrent = self.gru_proj(self.gru(x)[0])
        sequence_features = (recurrent * F.softmax(self.seq_attn(recurrent), dim=1)).sum(dim=1)
        hierarchy_features = self.hier(recurrent).mean(dim=1)
        return self.head(self.fusion(torch.cat([sequence_features, hierarchy_features], dim=-1)))


class SimpleNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


class SingleConv(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, padding=1)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        features = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(self.dropout(features) + x)


class SimpleHier(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


class AblationModel(VisualBackbone):
    def __init__(self, num_classes=13, hidden_dim=256, num_heads=4, dropout=0.3, cfg=None):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        cfg = cfg or {}
        self.gvar = GrayVariation(eta=16) if cfg.get("use_gray_variation", True) else IdentityModule()
        self.motion = MotionModule(hidden_dim, dropout) if cfg.get("use_motion_module", True) else SimpleNorm(hidden_dim)
        self.multiscale = MultiScaleBlock(hidden_dim, dropout) if cfg.get("use_multiscale", True) else SingleConv(hidden_dim, dropout)
        self.temporal = TemporalAttention(hidden_dim, num_heads, dropout) if cfg.get("use_causal_attention", True) else SimpleNorm(hidden_dim)
        self.hier = HierarchicalModule(hidden_dim, num_heads, dropout) if cfg.get("use_hier_attention", True) else SimpleHier(hidden_dim)


class SequenceDataset(Dataset):
    def __init__(self, root, persons, seq_len=16, transform=None):
        self.seq_len = seq_len
        self.transform = transform
        self.samples = []
        root = Path(root)

        for class_id in range(1, Config.NUM_CLASSES + 1):
            class_dir = root / str(class_id)
            if not class_dir.exists():
                continue

            grouped_frames = defaultdict(list)
            for image_path in sorted(class_dir.glob("*.jpg")):
                match = re.match(r"(\d+)_person(\d+)_(\w+)_(\d+)\.jpg", image_path.name)
                if match is None:
                    continue

                _, person_id, view, frame_id = match.groups()
                if int(person_id) in persons:
                    grouped_frames[(int(person_id), view)].append((int(frame_id), str(image_path)))

            for frames in grouped_frames.values():
                frames = sorted(frames, key=lambda item: item[0])
                if len(frames) < seq_len:
                    continue

                for start in range(0, len(frames) - seq_len + 1, seq_len):
                    self.samples.append({"label": class_id - 1, "frames": [path for _, path in frames[start : start + seq_len]]})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        frames = []

        for frame_path in sample["frames"]:
            try:
                image = Image.open(frame_path).convert("RGB")
                frames.append(self.transform(image) if self.transform else T.ToTensor()(image))
            except (OSError, RuntimeError, ValueError):
                frames.append(torch.zeros(3, Config.OUTPUT_SIZE, Config.OUTPUT_SIZE))

        while len(frames) < self.seq_len:
            frames.append(frames[-1] if frames else torch.zeros(3, Config.OUTPUT_SIZE, Config.OUTPUT_SIZE))

        return torch.stack(frames[: self.seq_len]), sample["label"]


def compute_metrics(predictions, labels):
    class_ids = list(range(Config.NUM_CLASSES))
    return {
        "acc": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "precision": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "per_class_recall": [float(v) for v in recall_score(labels, predictions, average=None, labels=class_ids, zero_division=0)],
        "per_class_precision": [float(v) for v in precision_score(labels, predictions, average=None, labels=class_ids, zero_division=0)],
        "per_class_f1": [float(v) for v in f1_score(labels, predictions, average=None, labels=class_ids, zero_division=0)],
        "confusion_matrix": confusion_matrix(labels, predictions, labels=class_ids).tolist(),
    }


@torch.no_grad()
def evaluate_test_set(model, detector, device):
    model.eval()
    transform = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    predictions = []
    labels = []

    for gesture_id in range(Config.NUM_CLASSES):
        image_dir = Config.TEST_DIR / str(gesture_id + 1) / "pic"
        if not image_dir.exists():
            continue

        frame_paths = sorted(str(path) for path in image_dir.glob("*.jpg"))
        if not frame_paths:
            continue

        segment_count = max(1, len(frame_paths) // Config.SEQ_LEN)
        for segment_index in range(segment_count):
            start = min(segment_index * Config.SEQ_LEN, max(0, len(frame_paths) - Config.SEQ_LEN))
            selected_paths = frame_paths[start : start + Config.SEQ_LEN]

            if len(selected_paths) >= Config.SEQ_LEN:
                indices = np.linspace(0, len(selected_paths) - 1, Config.SEQ_LEN).astype(int)
                selected_paths = [selected_paths[index] for index in indices]
            else:
                while len(selected_paths) < Config.SEQ_LEN:
                    selected_paths.append(selected_paths[-1])

            frames = []
            for frame_path in selected_paths:
                crop = detector.detect_and_crop(frame_path, Config.OUTPUT_SIZE, Config.HAND_CONF, Config.HAND_PADDING)
                frames.append(transform(crop) if crop is not None else torch.zeros(3, Config.OUTPUT_SIZE, Config.OUTPUT_SIZE))

            batch = torch.stack(frames).unsqueeze(0).to(device)
            predictions.append(model(batch).argmax(1).item())
            labels.append(gesture_id)
            del batch

    metrics = compute_metrics(np.array(predictions), np.array(labels))
    metrics["n_samples"] = len(predictions)
    return metrics


@torch.no_grad()
def evaluate_loader(model, loader, device):
    model.eval()
    predictions = []
    labels = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        predictions.extend(model(images).argmax(1).cpu().numpy())
        labels.extend(targets.numpy())
        del images

    return compute_metrics(np.array(predictions), np.array(labels))


def find_trained_model(name):
    candidates = [
        Config.TRAINED_DIR / f"{name}_v6.pth",
        Config.TRAINED_DIR / f"{name}_v5.pth",
        Config.TRAINED_DIR / f"{name}.pth",
    ]

    if name == "Full":
        candidates.append(Config.VISUAL_MODEL_PATH)

    for path in candidates:
        if path.exists():
            return path

    return None


def load_checkpoint(path, model, device):
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: {path}")

    state = {key.removeprefix("module."): value for key, value in state.items()}
    model_state = model.state_dict()

    if set(model_state) == set(state):
        model.load_state_dict(state)
        return len(model_state), len(model_state)

    matched = 0
    for key, value in state.items():
        if key in model_state and model_state[key].shape == value.shape:
            model_state[key] = value
            matched += 1

    model.load_state_dict(model_state)
    return matched, len(model_state)


def build_model(name, device):
    if name == "Full":
        return VisualBackbone(Config.NUM_CLASSES, Config.HIDDEN_DIM).to(device)

    config = Config.ABLATION_CONFIGS[name]
    return AblationModel(Config.NUM_CLASSES, Config.HIDDEN_DIM, cfg=config).to(device)


def plot_results(results, save_dir):
    names = [name for name in Config.ABLATION_ORDER if name in results]
    display_names = [Config.ABLATION_DISPLAY.get(name, name) for name in names]
    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for axis, metric, title in zip(axes, ["acc", "f1", "precision", "recall"], ["Accuracy", "F1 Score", "Precision", "Recall"]):
        values = [results[name][metric] * 100 for name in names]
        bars = axis.bar(range(len(names)), values, color=colors, edgecolor="black", linewidth=0.5)
        axis.set_xticks(range(len(names)))
        axis.set_xticklabels(display_names, rotation=45, ha="right", fontsize=7)
        axis.set_ylabel(f"{title} (%)")
        axis.set_title(f"Test {title}")
        axis.set_ylim(0, (max(values) if values else 50) * 1.3 + 5)

        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{value:.1f}", ha="center", va="bottom", fontsize=6)

        if "Full" in results:
            axis.axhline(y=results["Full"][metric] * 100, color="red", ls="--", alpha=0.5, lw=0.8)

    plt.tight_layout()
    plt.savefig(save_dir / "ablation_metrics.png", dpi=200, bbox_inches="tight")
    plt.close()

    if "Full" in results:
        fig, axis = plt.subplots(figsize=(10, 5))
        full_f1 = results["Full"]["f1"] * 100
        drops = [(Config.ABLATION_DISPLAY.get(name, name), full_f1 - results[name]["f1"] * 100) for name in names if name != "Full"]
        drops.sort(key=lambda item: item[1], reverse=True)

        if drops:
            labels, values = zip(*drops)
            bar_colors = ["#e74c3c" if value > 0 else "#27ae60" for value in values]
            bars = axis.barh(range(len(labels)), values, color=bar_colors, edgecolor="black")
            axis.set_yticks(range(len(labels)))
            axis.set_yticklabels(labels, fontsize=9)
            axis.set_xlabel("Test F1 Drop (%)")
            axis.set_title("CAST-Net Component Contribution")
            axis.axvline(x=0, color="black", lw=0.5)

            for bar, value in zip(bars, values):
                offset = 0.3 if value >= 0 else -0.3
                axis.text(value + offset, bar.get_y() + bar.get_height() / 2, f"{value:+.1f}%", ha="left" if value >= 0 else "right", va="center", fontsize=8)

        plt.tight_layout()
        plt.savefig(save_dir / "ablation_contribution.png", dpi=200, bbox_inches="tight")
        plt.close()

    fig, axis = plt.subplots(figsize=(14, 6))
    recall_matrix = np.array([np.array(results[name]["per_class_recall"]) * 100 for name in names])
    sns.heatmap(
        recall_matrix,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        xticklabels=Config.GESTURE_NAMES,
        yticklabels=display_names,
        ax=axis,
        cbar_kws={"label": "Recall (%)"},
    )
    axis.set_title("Per-Class Test Recall")
    plt.tight_layout()
    plt.savefig(save_dir / "ablation_recall_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()

    if "Full" in results and "confusion_matrix" in results["Full"]:
        fig, axis = plt.subplots(figsize=(10, 8))
        matrix = np.array(results["Full"]["confusion_matrix"])
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        sns.heatmap(
            matrix.astype(float) / row_sums,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            ax=axis,
            xticklabels=Config.GESTURE_NAMES,
            yticklabels=Config.GESTURE_NAMES,
            annot_kws={"size": 7},
        )
        axis.set_title(f"CAST-Net Test Confusion Matrix\nAcc: {results['Full']['acc'] * 100:.1f}%, F1: {results['Full']['f1'] * 100:.1f}%")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        plt.tight_layout()
        plt.savefig(save_dir / "ablation_confusion_matrix.png", dpi=200, bbox_inches="tight")
        plt.close()

    print(f"  Plots saved to {save_dir}")


def print_table(results):
    print("\n" + "=" * 115)
    print("CAST-Net Ablation Study Summary")
    print("=" * 115)
    print(f'{"Experiment":<25} {"TestAcc%":<10} {"TestF1%":<10} {"TestPrec%":<10} {"TestRec%":<10} {"ValAcc%":<10} {"ValF1%":<10} {"DeltaF1%":<10}')
    print("-" * 115)

    full_f1 = results.get("Full", {}).get("f1", 0) * 100
    for name in Config.ABLATION_ORDER:
        if name not in results:
            continue

        result = results[name]
        display_name = Config.ABLATION_DISPLAY.get(name, name)
        delta = result["f1"] * 100 - full_f1
        delta_text = f"{delta:+.2f}" if name != "Full" else "  -"
        print(
            f'{display_name:<25} {result["acc"] * 100:<10.2f} {result["f1"] * 100:<10.2f} '
            f'{result["precision"] * 100:<10.2f} {result["recall"] * 100:<10.2f} '
            f'{result.get("val_acc", 0) * 100:<10.2f} {result.get("val_f1", 0) * 100:<10.2f} {delta_text:<10}'
        )

    print("=" * 115)

    if "Full" in results:
        importance = [(Config.ABLATION_DISPLAY.get(name, name), full_f1 - results[name]["f1"] * 100) for name in results if name != "Full"]
        importance.sort(key=lambda item: item[1], reverse=True)
        print("\nComponent Importance (F1 drop when removed):")
        for index, (component, value) in enumerate(importance, 1):
            print(f"  {index}. {component}: {value:+.2f}%")

    print("=" * 115)


def save_results(results):
    payload = {}
    for name, result in results.items():
        payload[name] = {
            "display_name": result.get("display_name", name),
            "test_acc": result["acc"],
            "test_f1": result["f1"],
            "test_precision": result["precision"],
            "test_recall": result["recall"],
            "val_acc": result.get("val_acc", 0),
            "val_f1": result.get("val_f1", 0),
            "params": result.get("params", 0),
            "per_class_recall": result["per_class_recall"],
            "per_class_f1": result["per_class_f1"],
            "per_class_precision": result["per_class_precision"],
            "n_test_samples": result.get("n_samples", 0),
        }

    output_path = Config.OUTPUT_DIR / "ablation_results.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    print(f"\nResults saved to {output_path}")


def list_output_files():
    print(f"\nOutput files in {Config.OUTPUT_DIR}:")
    for path in sorted(Config.OUTPUT_DIR.iterdir()):
        size = path.stat().st_size
        if size > 1024 * 1024:
            print(f"  {path.name:<40} ({size / 1024 / 1024:.1f} MB)")
        else:
            print(f"  {path.name:<40} ({size / 1024:.1f} KB)")


def main():
    device = Config.DEVICE
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CAST-Net Ablation Study")
    print(f"  Device: {device}")
    print(f"  Model directory: {Config.TRAINED_DIR}")
    print(f"  Output directory: {Config.OUTPUT_DIR}")
    print("=" * 70)

    print("\nSearching for trained models...")
    model_paths = {}
    for name in Config.ABLATION_ORDER:
        path = find_trained_model(name)
        if path is None:
            print(f"  {name:<20} -> not found")
            continue
        model_paths[name] = path
        print(f"  {name:<20} -> {path}")

    if not model_paths:
        print("\nNo trained models were found.")
        return

    print("\nInitializing hand detector...")
    detector = HandDetector(Config.HAND_DETECT_PATH, device)

    eval_transform = T.Compose(
        [
            T.Resize((Config.OUTPUT_SIZE, Config.OUTPUT_SIZE)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_dataset = SequenceDataset(Config.DATA_DIR, Config.VAL_PERSONS, Config.SEQ_LEN, eval_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.WORKERS,
        pin_memory=True,
        prefetch_factor=Config.PREFETCH,
        persistent_workers=False,
    )
    print(f"Validation samples: {len(val_dataset)}")

    print(f"\nTest set: {Config.TEST_DIR}")
    frame_total = 0
    for gesture_id, gesture_name in enumerate(Config.GESTURE_NAMES, 1):
        image_dir = Config.TEST_DIR / str(gesture_id) / "pic"
        frame_count = len(list(image_dir.glob("*.jpg"))) if image_dir.exists() else 0
        frame_total += frame_count
        print(f"  G{gesture_id:02d} ({gesture_name:<12}): {frame_count} frames")
    print(f"  Total: {frame_total} frames")

    results = {}
    for name in Config.ABLATION_ORDER:
        if name not in model_paths:
            continue

        source_path = model_paths[name]
        display_name = Config.ABLATION_DISPLAY.get(name, name)

        print(f"\n{'=' * 70}")
        print(f"Evaluating {display_name} ({name})")
        print(f"  Model: {source_path}")
        print(f"{'=' * 70}")

        model = build_model(name, device)
        loaded, total = load_checkpoint(source_path, model, device)
        status = "complete" if loaded == total else "partial"
        print(f"  Checkpoint load: {status} ({loaded}/{total} tensors)")

        params = sum(parameter.numel() for parameter in model.parameters()) / 1e6
        print(f"  Parameters: {params:.2f}M")

        print("  Evaluating validation set...")
        val_metrics = evaluate_loader(model, val_loader, device)
        print(f'  Validation -> Acc: {val_metrics["acc"] * 100:.2f}%, F1: {val_metrics["f1"] * 100:.2f}%')

        print("  Evaluating test set...")
        test_metrics = evaluate_test_set(model, detector, device)
        print(f'  Test -> Acc: {test_metrics["acc"] * 100:.2f}%, F1: {test_metrics["f1"] * 100:.2f}%')
        print(f'  Test -> Prec: {test_metrics["precision"] * 100:.2f}%, Rec: {test_metrics["recall"] * 100:.2f}%')
        print(f'  Test -> Samples: {test_metrics["n_samples"]}')

        test_metrics["params"] = params
        test_metrics["display_name"] = display_name
        test_metrics["val_acc"] = val_metrics["acc"]
        test_metrics["val_f1"] = val_metrics["f1"]
        results[name] = test_metrics

        destination_path = Config.OUTPUT_DIR / f"{name}.pth"
        shutil.copy2(source_path, destination_path)
        print(f"  Copied model to {destination_path}")

        del model
        clear_memory()
        detector.clear_cache()

    save_results(results)
    print("\nGenerating plots...")
    plot_results(results, Config.OUTPUT_DIR)
    print_table(results)
    list_output_files()
    print("\nFinished.")


if __name__ == "__main__":
    main()

