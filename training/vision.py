import re
import json
import warnings
import subprocess
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T
from torchvision import models
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] ultralytics is not installed.")

warnings.filterwarnings('ignore')


class Config:
    DATA_DIR = Path('/yourpath/Dataset/origin_pic')
    HAND_MODEL_PATH = Path('/yourpath/Dataset/models/best.pt')
    SAVE_DIR = Path('/yourpath/Dataset/models/Visual/True-Use')
    CROP_CACHE_DIR = Path('/yourpath/Dataset/cropped_hands_cache')

    TRAIN_PERSONS = [1, 2, 3, 4]
    VAL_PERSONS = [5]

    NUM_CLASSES = 13
    SEQ_LEN = 16
    HIDDEN_DIM = 256
    NUM_HEADS = 4

    BATCH_SIZE = 16
    EPOCHS = 150
    LR = 5e-5
    WEIGHT_DECAY = 0.05
    PATIENCE = 35
    NUM_WORKERS = 4

    HAND_CONF = 0.25
    PADDING_RATIO = 0.15
    OUTPUT_SIZE = 224

    DEVICE = None

    GESTURE_NAMES = [
        'Fist', 'Extension', 'Rotation', 'Opposition', 'Ball', 'Putty', 'Press',
        'Interlace', 'Flexion', 'Massage', 'Towel', 'Tapping', 'Piano'
    ]


def select_gpu():
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=index,utilization.gpu,memory.used,memory.total',
                '--format=csv,noheader,nounits'
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    gpu_info.append({
                        'id': int(parts[0]),
                        'util': float(parts[1]),
                        'mem_ratio': float(parts[2]) / float(parts[3])
                    })
            if gpu_info:
                best = min(gpu_info, key=lambda x: (x['util'], x['mem_ratio']))
                print(f"[GPU] Selected GPU {best['id']} with utilization {best['util']:.1f}%.")
                return f'cuda:{best["id"]}'
    except Exception:
        pass

    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


class HandDetector:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_path: Path = None, device: str = 'cuda:0'):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.device = device
        self.model = None
        self.is_loaded = False

        if model_path and YOLO_AVAILABLE:
            self.load(model_path)

        self._initialized = True

    def load(self, model_path: Path) -> bool:
        try:
            if not model_path.exists():
                print(f"[HandDetector] Model does not exist: {model_path}")
                return False

            self.model = YOLO(str(model_path))
            self.is_loaded = True
            print(f"[HandDetector] Model loaded successfully: {model_path}")
            return True
        except Exception as e:
            print(f"[HandDetector] Failed to load model: {e}")
            return False

    def detect_and_crop(
        self,
        image_path: str,
        output_size: int = 224,
        conf: float = 0.25,
        padding: float = 0.15
    ) -> np.ndarray:
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        if self.is_loaded and self.model is not None:
            try:
                results = self.model.predict(
                    source=str(image_path),
                    conf=conf,
                    verbose=False,
                    device=self.device
                )

                if len(results) > 0 and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    best_idx = boxes.conf.argmax()
                    box = boxes.xyxy[best_idx].cpu().numpy()

                    x1, y1, x2, y2 = map(int, box)

                    box_w, box_h = x2 - x1, y2 - y1
                    pad_w = int(box_w * padding)
                    pad_h = int(box_h * padding)

                    x1 = max(0, x1 - pad_w)
                    y1 = max(0, y1 - pad_h)
                    x2 = min(w, x2 + pad_w)
                    y2 = min(h, y2 + pad_h)

                    cropped = img_rgb[y1:y2, x1:x2]

                    if cropped.size > 0:
                        if cropped.shape[0] > output_size or cropped.shape[1] > output_size:
                            interp = cv2.INTER_AREA
                        else:
                            interp = cv2.INTER_LINEAR

                        resized = cv2.resize(
                            cropped,
                            (output_size, output_size),
                            interpolation=interp
                        )
                        return resized
            except Exception:
                pass

        size = min(h, w)
        top = (h - size) // 2
        left = (w - size) // 2
        cropped = img_rgb[top:top + size, left:left + size]
        resized = cv2.resize(cropped, (output_size, output_size))
        return resized


def preprocess_hand_crops(config: Config, detector: HandDetector):
    cache_dir = config.CROP_CACHE_DIR

    if cache_dir.exists():
        existing_count = sum(1 for _ in cache_dir.rglob('*.jpg'))
        if existing_count > 1000:
            print(f"[Cache] Existing cache found: {existing_count} cropped images.")
            return True

    print("\n[Preprocess] Starting hand crop preprocessing...")
    cache_dir.mkdir(parents=True, exist_ok=True)

    total_success = 0
    total_failed = 0

    for cls_id in range(1, 14):
        cls_dir = config.DATA_DIR / str(cls_id)
        if not cls_dir.exists():
            continue

        out_cls_dir = cache_dir / str(cls_id)
        out_cls_dir.mkdir(parents=True, exist_ok=True)

        images = list(cls_dir.glob('*.jpg'))

        for img_path in tqdm(images, desc=f"  Gesture {cls_id}", leave=False):
            out_path = out_cls_dir / img_path.name

            if out_path.exists():
                total_success += 1
                continue

            cropped = detector.detect_and_crop(
                str(img_path),
                output_size=config.OUTPUT_SIZE,
                conf=config.HAND_CONF,
                padding=config.PADDING_RATIO
            )

            if cropped is not None:
                cv2.imwrite(str(out_path), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
                total_success += 1
            else:
                total_failed += 1

    print(f"[Preprocess] Done. Success: {total_success}, Failed: {total_failed}")
    return True


class GestureSequenceDataset(Dataset):
    def __init__(
        self,
        root: Path,
        persons: list,
        seq_len: int = 16,
        transform=None,
        is_training: bool = True
    ):
        self.root = Path(root)
        self.seq_len = seq_len
        self.transform = transform
        self.is_training = is_training
        self.samples = []

        print("\n[Dataset] Loading data...")
        print(f"  Path: {self.root}")
        print(f"  Persons: {persons}")
        print(f"  Mode: {'train' if is_training else 'validation'}")

        for cls_id in range(1, 14):
            cls_dir = self.root / str(cls_id)
            if not cls_dir.exists():
                continue

            groups = defaultdict(list)

            for img_file in cls_dir.glob('*.jpg'):
                match = re.match(r'(\d+)_person(\d+)_(\w+)_(\d+)\.jpg', img_file.name)
                if not match:
                    continue

                gesture_id, person_id, view, frame_idx = match.groups()
                person_id = int(person_id)

                if person_id not in persons:
                    continue

                groups[(person_id, view)].append((int(frame_idx), img_file))

            for (person_id, view), frames in groups.items():
                frames = sorted(frames, key=lambda x: x[0])

                if len(frames) < seq_len:
                    continue

                stride = seq_len // 2 if is_training else seq_len

                for i in range(0, len(frames) - seq_len + 1, stride):
                    self.samples.append({
                        'label': cls_id - 1,
                        'frames': [f for _, f in frames[i:i + seq_len]],
                        'person': person_id,
                        'view': view
                    })

        label_counts = Counter(s['label'] for s in self.samples)
        print(f"  Total samples: {len(self.samples)}")
        for gid in sorted(label_counts.keys()):
            print(f"    Gesture {gid + 1} ({Config.GESTURE_NAMES[gid]}): {label_counts[gid]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        imgs = []

        for frame_path in sample['frames']:
            try:
                img = Image.open(frame_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                imgs.append(img)
            except Exception:
                imgs.append(torch.zeros(3, 224, 224))

        while len(imgs) < self.seq_len:
            imgs.append(imgs[-1] if imgs else torch.zeros(3, 224, 224))

        return torch.stack(imgs[:self.seq_len]), sample['label']


def get_weighted_sampler(dataset):
    labels = [s['label'] for s in dataset.samples]
    counts = Counter(labels)
    weights = [1.0 / counts[l] for l in labels]
    return WeightedRandomSampler(weights, len(weights), replacement=True)


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

        vel_f = self.dropout(self.vel_proj(vel))
        acc_f = self.dropout(self.acc_proj(acc))

        concat = torch.cat([x, vel_f, acc_f], dim=-1)
        gate = self.gate(concat)
        fused = self.fuse(concat)

        return self.norm(x + gate * fused)


class MultiScaleBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.branch1 = nn.Conv1d(dim, dim // 4, 1)
        self.branch3 = nn.Sequential(
            nn.Conv1d(dim, dim // 4, 1),
            nn.Conv1d(dim // 4, dim // 4, 3, padding=1)
        )
        self.branch5 = nn.Sequential(
            nn.Conv1d(dim, dim // 4, 1),
            nn.Conv1d(dim // 4, dim // 4, 5, padding=2)
        )
        self.branch_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(dim, dim // 4, 1)
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        x_t = x.transpose(1, 2)

        b1 = self.branch1(x_t)
        b3 = self.branch3(x_t)
        b5 = self.branch5(x_t)
        bp = self.branch_pool(x_t).expand(-1, -1, T)

        out = torch.cat([b1, b3, b5, bp], dim=1).transpose(1, 2)
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

        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.dropout(F.softmax(attn, dim=-1))

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
        k = self.key(x)
        v = self.value(x)

        attn = F.softmax(torch.bmm(q, k.transpose(-2, -1)) / (D ** 0.5), dim=-1)
        attn = self.dropout(attn)
        out = torch.bmm(attn, v)

        return self.norm(x + out.expand(-1, T, -1))


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

        global_feat = x.mean(dim=1, keepdim=True)
        seg_out, _ = self.seg_attn(global_feat, x, x)
        x = self.seg_norm(x + seg_out.expand(-1, x.size(1), -1))

        return x


class CASTNet(nn.Module):
    def __init__(self, num_classes=13, hidden_dim=256, num_heads=4, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.gvar = GrayVariation(eta=16)

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for name, param in resnet.named_parameters():
            if 'layer3' not in name and 'layer4' not in name:
                param.requires_grad = False

        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        self.proj = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.motion = MotionModule(hidden_dim, dropout)
        self.multiscale = MultiScaleBlock(hidden_dim, dropout)
        self.temporal = TemporalAttention(hidden_dim, num_heads, dropout)
        self.spatial = SpatialAttention(hidden_dim, dropout)

        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.gru_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.seq_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.hier = HierarchicalModule(hidden_dim, num_heads, dropout)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        x = self.gvar(x)
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(B, T, -1)
        x = self.proj(x)

        x = self.motion(x)
        x = self.multiscale(x)
        x = self.temporal(x)
        x = self.spatial(x)

        self.gru.flatten_parameters()
        gru_out, _ = self.gru(x)
        gru_out = self.gru_proj(gru_out)

        attn_weights = F.softmax(self.seq_attn(gru_out), dim=1)
        seq_feat = (gru_out * attn_weights).sum(dim=1)

        hier_out = self.hier(gru_out)
        hier_feat = hier_out.mean(dim=1)

        fused = self.fusion(torch.cat([seq_feat, hier_feat], dim=-1))
        logits = self.head(fused)

        return logits

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            proba = F.softmax(logits, dim=-1)
        return proba


MVTFNet = CASTNet


class LabelSmoothingFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, smoothing=0.1, class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.class_weights = class_weights

    def forward(self, pred, target):
        n_classes = pred.size(-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        log_prob = F.log_softmax(pred, dim=-1)
        prob = torch.exp(log_prob)

        focal_weight = (1 - prob) ** self.gamma

        if self.class_weights is not None:
            class_weight = self.class_weights[target].unsqueeze(1)
            loss = -class_weight * focal_weight * true_dist * log_prob
        else:
            loss = -focal_weight * true_dist * log_prob

        return loss.sum(dim=-1).mean()


def compute_class_weights(dataset, device, power=0.5):
    counts = np.zeros(Config.NUM_CLASSES)
    for s in dataset.samples:
        counts[s['label']] += 1

    weights = 1.0 / (counts ** power + 1e-8)
    weights = weights / weights.sum() * Config.NUM_CLASSES

    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'preds': all_preds,
        'labels': all_labels,
        'per_class_recall': recall_score(all_labels, all_preds, average=None, zero_division=0),
        'per_class_f1': f1_score(all_labels, all_preds, average=None, zero_division=0)
    }


def train_model(model, train_loader, val_loader, train_ds, config):
    device = config.DEVICE
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY
    )

    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return (epoch + 1) / warmup
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup) / (config.EPOCHS - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    class_weights = compute_class_weights(train_ds, device)
    criterion = LabelSmoothingFocalLoss(gamma=2.0, smoothing=0.1, class_weights=class_weights)

    scaler = GradScaler()

    best_f1 = 0
    best_acc = 0
    no_improve = 0
    best_state = None
    history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}

    save_path = config.SAVE_DIR / 'mvtf_best.pth'

    print("\n" + "=" * 60)
    print("Training started")
    print("=" * 60)

    pbar = tqdm(range(config.EPOCHS), desc='Training')

    for epoch in pbar:
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device
        )

        val_result = evaluate(model, val_loader, device)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_result['accuracy'])
        history['val_f1'].append(val_result['f1'])

        pbar.set_postfix({
            'loss': f'{train_loss:.3f}',
            'train': f'{train_acc * 100:.1f}%',
            'val': f'{val_result["accuracy"] * 100:.1f}%',
            'f1': f'{val_result["f1"] * 100:.1f}%',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })

        if val_result['f1'] > best_f1:
            best_f1 = val_result['f1']
            best_acc = val_result['accuracy']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            torch.save(best_state, save_path)
        else:
            no_improve += 1

        if no_improve >= config.PATIENCE:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    final_result = evaluate(model, val_loader, device)
    final_result['history'] = history
    final_result['best_f1'] = best_f1
    final_result['best_acc'] = best_acc

    return final_result, model


def plot_results(result, save_dir):
    cm = confusion_matrix(result['labels'], result['preds'])
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        ax=ax,
        xticklabels=Config.GESTURE_NAMES,
        yticklabels=Config.GESTURE_NAMES
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(
        f'CAST-Net + Hand Detection\nAccuracy: {result["accuracy"] * 100:.1f}%, F1: {result["f1"] * 100:.1f}%',
        fontsize=14
    )
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix.png', dpi=300)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(Config.NUM_CLASSES)

    axes[0].bar(x, result['per_class_recall'], color='steelblue', edgecolor='black')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(Config.GESTURE_NAMES, rotation=45, ha='right', fontsize=8)
    axes[0].set_ylabel('Recall')
    axes[0].set_title('Per-Class Recall')
    axes[0].set_ylim(0, 1.1)
    axes[0].axhline(
        y=result['recall'],
        color='r',
        linestyle='--',
        label=f'Avg: {result["recall"]:.2f}'
    )
    axes[0].legend()

    axes[1].bar(x, result['per_class_f1'], color='coral', edgecolor='black')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(Config.GESTURE_NAMES, rotation=45, ha='right', fontsize=8)
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Per-Class F1 Score')
    axes[1].set_ylim(0, 1.1)
    axes[1].axhline(
        y=result['f1'],
        color='r',
        linestyle='--',
        label=f'Avg: {result["f1"]:.2f}'
    )
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_dir / 'per_class_metrics.png', dpi=300)
    plt.close()

    if 'history' in result:
        history = result['history']
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(history['train_loss'], 'b-', label='Train Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history['train_acc'], 'b-', label='Train Acc')
        axes[1].plot(history['val_acc'], 'r-', label='Val Acc')
        axes[1].plot(history['val_f1'], 'g--', label='Val F1')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Training Progress')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / 'training_curves.png', dpi=300)
        plt.close()


def main():
    print("=" * 70)
    print("CAST-Net visual model training with hand detection")
    print("=" * 70)

    Config.DEVICE = select_gpu()
    print(f"Device: {Config.DEVICE}")

    Config.SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Save directory: {Config.SAVE_DIR}")

    print(f"\n[Hand Detector] Loading model: {Config.HAND_MODEL_PATH}")
    detector = HandDetector(Config.HAND_MODEL_PATH, Config.DEVICE)

    if not detector.is_loaded:
        print("[ERROR] Failed to load the hand detection model.")
        return

    preprocess_hand_crops(Config, detector)

    train_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(10),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.1, scale=(0.02, 0.1))
    ])

    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = GestureSequenceDataset(
        Config.CROP_CACHE_DIR,
        Config.TRAIN_PERSONS,
        Config.SEQ_LEN,
        train_transform,
        is_training=True
    )

    val_ds = GestureSequenceDataset(
        Config.CROP_CACHE_DIR,
        Config.VAL_PERSONS,
        Config.SEQ_LEN,
        val_transform,
        is_training=False
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        print("[ERROR] Empty dataset. Please check the crop cache.")
        return

    train_sampler = get_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )

    print(f"\nTraining set: {len(train_ds)} samples")
    print(f"Validation set: {len(val_ds)} samples")

    model = CASTNet(
        num_classes=Config.NUM_CLASSES,
        hidden_dim=Config.HIDDEN_DIM,
        num_heads=Config.NUM_HEADS,
        dropout=0.3
    )

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Model parameters: {n_params:.2f}M, trainable: {n_trainable:.2f}M")

    result, model = train_model(model, train_loader, val_loader, train_ds, Config)

    print("\n" + "=" * 60)
    print("Final results")
    print("=" * 60)
    print(f"  Accuracy:  {result['accuracy'] * 100:.2f}%")
    print(f"  F1 Score:  {result['f1'] * 100:.2f}%")
    print(f"  Precision: {result['precision'] * 100:.2f}%")
    print(f"  Recall:    {result['recall'] * 100:.2f}%")

    print("\nPer-class recall:")
    for i, name in enumerate(Config.GESTURE_NAMES):
        print(f"  {name}: {result['per_class_recall'][i] * 100:.1f}%")

    if result['accuracy'] >= 0.40:
        print(f"\nTarget reached. Accuracy {result['accuracy'] * 100:.1f}% >= 40%")
    else:
        print(f"\nTarget not reached. Accuracy {result['accuracy'] * 100:.1f}% < 40%")

    plot_results(result, Config.SAVE_DIR)

    with open(Config.SAVE_DIR / 'training_results.json', 'w') as f:
        json.dump({
            'accuracy': float(result['accuracy']),
            'f1': float(result['f1']),
            'precision': float(result['precision']),
            'recall': float(result['recall']),
            'per_class_recall': result['per_class_recall'].tolist(),
            'per_class_f1': result['per_class_f1'].tolist(),
            'best_f1': float(result['best_f1']),
            'best_acc': float(result['best_acc']),
            'config': {
                'train_persons': Config.TRAIN_PERSONS,
                'val_persons': Config.VAL_PERSONS,
                'seq_len': Config.SEQ_LEN,
                'batch_size': Config.BATCH_SIZE,
                'epochs': Config.EPOCHS,
                'lr': Config.LR,
                'weight_decay': Config.WEIGHT_DECAY,
                'hand_detection': True
            },
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\nModel saved to: {Config.SAVE_DIR / 'mvtf_best.pth'}")
    print(f"Results saved to: {Config.SAVE_DIR}")


if __name__ == '__main__':
    main()
