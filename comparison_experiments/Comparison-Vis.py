#!/usr/bin/env python3

import gc
import json
import re
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import torchvision.transforms as T
from torchvision import models
from PIL import Image
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from tqdm import tqdm
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "Dataset"
TRAIN_DIR = DATASET_DIR / "origin_pic"
TEST_DIR = DATASET_DIR / "test"
OUTPUT_DIR = PROJECT_ROOT / "data" / "Comparative experiment" / "Single Vis"
CHECKPOINT_DIR = DATASET_DIR / "Model_others" / "Visual_final"

PRETRAINED_WEIGHT_PATH = DATASET_DIR / "models" / "mvtf_visual.pth"
YOLO_WEIGHT_PATH = DATASET_DIR / "models" / "best.pt"

NUM_CLASSES = 13
SEQUENCE_LENGTH = 16
BATCH_SIZE = 4
NUM_EPOCHS = 35
NUM_WORKERS = 4
LEARNING_RATE = 2e-4
EARLY_STOP_PATIENCE = 10
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

GESTURE_NAMES = [
    "Fist", "Extension", "Rotation", "Opposition", "Ball", "Putty",
    "Press", "Interlace", "Flexion", "Massage", "Towel", "Tapping", "Piano",
]


def amp_enabled() -> bool:
    return DEVICE.type == "cuda"


def autocast_context():
    return torch.cuda.amp.autocast(enabled=amp_enabled())


class HandCropper:
    def __init__(self):
        self.model = None
        if HAS_YOLO and YOLO_WEIGHT_PATH.exists():
            try:
                self.model = YOLO(str(YOLO_WEIGHT_PATH))
                dummy = np.zeros((224, 224, 3), dtype=np.uint8)
                self.model.predict(source=dummy, conf=0.25, verbose=False)
                print("  YOLO hand detector loaded")
            except Exception as exc:
                print(f"  YOLO initialization failed: {exc}")

    def crop_pil(self, img_path, size=224):
        img = cv2.imread(str(img_path))
        if img is None:
            return Image.new('RGB', (size, size))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if self.model is not None:
            try:
                res = self.model.predict(source=img, conf=0.25, verbose=False)
                if len(res) > 0 and len(res[0].boxes) > 0:
                    boxes = res[0].boxes
                    best = boxes.conf.argmax()
                    x1, y1, x2, y2 = boxes.xyxy[best].cpu().numpy().astype(int)
                    pad = 0.15
                    pw = int((x2 - x1) * pad)
                    ph = int((y2 - y1) * pad)
                    x1 = max(0, x1 - pw)
                    y1 = max(0, y1 - ph)
                    x2 = min(w, x2 + pw)
                    y2 = min(h, y2 + ph)
                    crop = rgb[y1:y2, x1:x2]
                    if crop.size > 0:
                        return Image.fromarray(cv2.resize(crop, (size, size)))
            except Exception:
                pass
        side = min(h, w)
        top = (h - side) // 2
        left = (w - side) // 2
        return Image.fromarray(cv2.resize(rgb[top:top + side, left:left + side], (size, size)))

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
        self.gate = nn.Sequential(nn.Linear(dim*3, dim), nn.Sigmoid())
        self.fuse = nn.Linear(dim*3, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        vel = torch.zeros_like(x)
        vel[:, 1:] = x[:, 1:] - x[:, :-1]
        acc = torch.zeros_like(x)
        acc[:, 2:] = vel[:, 2:] - vel[:, 1:-1]
        c = torch.cat([x, self.dropout(self.vel_proj(vel)),
                       self.dropout(self.acc_proj(acc))], dim=-1)
        return self.norm(x + self.gate(c) * self.fuse(c))

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
        _, time_steps, _ = x.shape
        xt = x.transpose(1, 2)
        out = torch.cat([self.branch1(xt), self.branch3(xt), self.branch5(xt),
                         self.branch_pool(xt).expand(-1, -1, time_steps)], dim=1).transpose(1, 2)
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
        batch_size, time_steps, hidden_dim = x.shape
        qkv = self.qkv(x).reshape(batch_size, time_steps, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = self.dropout(F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(batch_size, time_steps, hidden_dim)
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
        _, time_steps, hidden_dim = x.shape
        q = self.query(x.mean(dim=1, keepdim=True))
        attn = self.dropout(F.softmax(
            torch.bmm(q, self.key(x).transpose(-2, -1)) / (hidden_dim ** 0.5), dim=-1))
        return self.norm(x + torch.bmm(attn, self.value(x)).expand(-1, time_steps, -1))

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

class CASTNet(nn.Module):
    def __init__(self, num_classes=13, hidden_dim=256, num_heads=4, dropout=0.3):
        super().__init__()
        self.gvar = GrayVariation(eta=16)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.proj = nn.Sequential(
            nn.Linear(512, hidden_dim), nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.motion = MotionModule(hidden_dim, dropout)
        self.multiscale = MultiScaleBlock(hidden_dim, dropout)
        self.temporal = TemporalAttention(hidden_dim, num_heads, dropout)
        self.spatial = SpatialAttention(hidden_dim, dropout)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2,
                          batch_first=True, bidirectional=True, dropout=dropout)
        self.gru_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.seq_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1))
        self.hier = HierarchicalModule(hidden_dim, num_heads, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(), nn.Dropout(0.3), nn.Linear(hidden_dim // 2, num_classes))

    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.shape
        x = self.gvar(x.view(batch_size * time_steps, channels, height, width))
        x = F.adaptive_avg_pool2d(self.encoder(x), (1, 1)).view(batch_size, time_steps, -1)
        x = self.spatial(self.temporal(self.multiscale(self.motion(self.proj(x)))))
        self.gru.flatten_parameters()
        gru_out = self.gru_proj(self.gru(x)[0])
        seq_feat = (gru_out * F.softmax(self.seq_attn(gru_out), dim=1)).sum(dim=1)
        hier_feat = self.hier(gru_out).mean(dim=1)
        return self.head(self.fusion(torch.cat([seq_feat, hier_feat], dim=-1)))


def _resnet_enc():
    r = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool,
                         r.layer1, r.layer2, r.layer3, r.layer4)

class BaselineNet(nn.Module):
    def __init__(self, nc=13):
        super().__init__()
        self.enc = _resnet_enc()
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, nc))

    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.shape
        f = F.adaptive_avg_pool2d(self.enc(x.view(batch_size * time_steps, channels, height, width)), (1, 1)).view(batch_size, time_steps, -1)
        return self.fc(f.mean(1))

class ResNetLSTMNet(nn.Module):
    def __init__(self, nc=13, hd=256):
        super().__init__()
        self.enc = _resnet_enc()
        self.lstm = nn.LSTM(512, hd, 2, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(hd * 2, nc))

    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.shape
        f = F.adaptive_avg_pool2d(self.enc(x.view(batch_size * time_steps, channels, height, width)), (1, 1)).view(batch_size, time_steps, -1)
        self.lstm.flatten_parameters()
        return self.fc(self.lstm(f)[0][:, -1])

class ConvGRUNet(nn.Module):
    def __init__(self, nc=13, hd=256):
        super().__init__()
        self.enc = _resnet_enc()
        self.gru = nn.GRU(512, hd, 2, batch_first=True, bidirectional=True, dropout=0.3)
        self.attn = nn.Sequential(nn.Linear(hd * 2, 128), nn.Tanh(), nn.Linear(128, 1))
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(hd * 2, nc))

    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.shape
        f = F.adaptive_avg_pool2d(self.enc(x.view(batch_size * time_steps, channels, height, width)), (1, 1)).view(batch_size, time_steps, -1)
        self.gru.flatten_parameters()
        output, _ = self.gru(f)
        return self.fc((output * F.softmax(self.attn(output), 1)).sum(1))

class TemporalConvNet(nn.Module):
    def __init__(self, nc=13, hd=256):
        super().__init__()
        self.enc = _resnet_enc()
        self.tcn = nn.Sequential(
            nn.Conv1d(512, hd, 3, padding=1), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(hd, hd, 3, padding=2, dilation=2), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(hd, hd, 3, padding=4, dilation=4), nn.ReLU())
        self.fc = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                nn.Dropout(0.5), nn.Linear(hd, nc))

    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.shape
        f = F.adaptive_avg_pool2d(self.enc(x.view(batch_size * time_steps, channels, height, width)), (1, 1)).view(batch_size, time_steps, -1)
        return self.fc(self.tcn(f.transpose(1, 2)))

class DepthCRNNet(nn.Module):
    def __init__(self, nc=13, hd=256):
        super().__init__()
        self.enc = _resnet_enc()
        self.conv = nn.Sequential(
            nn.Conv1d(512, hd, 3, padding=1), nn.BatchNorm1d(hd), nn.ReLU(),
            nn.Conv1d(hd, hd, 3, padding=1), nn.BatchNorm1d(hd), nn.ReLU())
        self.gru = nn.GRU(hd, hd, 1, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(hd * 2, nc))

    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.shape
        f = F.adaptive_avg_pool2d(self.enc(x.view(batch_size * time_steps, channels, height, width)), (1, 1)).view(batch_size, time_steps, -1)
        f = self.conv(f.transpose(1, 2)).transpose(1, 2)
        self.gru.flatten_parameters()
        return self.fc(self.gru(f)[0][:, -1])

class TestDatasetYOLO(Dataset):
    def __init__(self, test_dir, seq_len, cropper):
        self.seq_len = seq_len
        self.cropper = cropper
        self.tf = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.samples = []
        for gid in range(1, 14):
            pic = Path(test_dir) / str(gid) / "pic"
            if not pic.exists():
                continue
            frames = sorted(pic.glob("frame_*.jpg"))
            if not frames:
                continue
            stride = max(1, seq_len // 2)
            for i in range(0, max(1, len(frames) - seq_len + 1), stride):
                self.samples.append((gid - 1, [str(f) for f in frames[i:i + seq_len]]))
            if len(frames) < seq_len:
                self.samples.append((gid - 1, [str(f) for f in frames]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, paths = self.samples[idx]
        imgs = []
        for p in paths:
            imgs.append(self.tf(self.cropper.crop_pil(p, 224)))
        if len(imgs) >= self.seq_len:
            sel = np.linspace(0, len(imgs) - 1, self.seq_len).astype(int)
            imgs = [imgs[i] for i in sel]
        while len(imgs) < self.seq_len:
            imgs.append(imgs[-1] if imgs else torch.zeros(3, 224, 224))
        return torch.stack(imgs[:self.seq_len]), label

class TestDatasetPlain(Dataset):
    def __init__(self, test_dir, seq_len):
        self.seq_len = seq_len
        self.tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.samples = []
        for gid in range(1, 14):
            pic = Path(test_dir) / str(gid) / "pic"
            if not pic.exists():
                continue
            frames = sorted(pic.glob("frame_*.jpg"))
            if not frames:
                continue
            stride = max(1, seq_len // 2)
            for i in range(0, max(1, len(frames) - seq_len + 1), stride):
                self.samples.append((gid - 1, [str(f) for f in frames[i:i + seq_len]]))
            if len(frames) < seq_len:
                self.samples.append((gid - 1, [str(f) for f in frames]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, paths = self.samples[idx]
        imgs = []
        for p in paths:
            try:
                imgs.append(self.tf(Image.open(p).convert('RGB')))
            except Exception:
                imgs.append(torch.zeros(3, 224, 224))
        if len(imgs) >= self.seq_len:
            sel = np.linspace(0, len(imgs) - 1, self.seq_len).astype(int)
            imgs = [imgs[i] for i in sel]
        while len(imgs) < self.seq_len:
            imgs.append(imgs[-1] if imgs else torch.zeros(3, 224, 224))
        return torch.stack(imgs[:self.seq_len]), label

class TrainDataset(Dataset):
    def __init__(self, root, persons, seq_len, training=True):
        self.seq_len = seq_len
        if training:
            self.tf = T.Compose([
                T.Resize((224, 224)), T.RandomHorizontalFlip(),
                T.RandomRotation(10), T.ColorJitter(.2,.2,.2),
                T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        else:
            self.tf = T.Compose([
                T.Resize((224, 224)), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.samples = []
        for cid in range(1, 14):
            cd = Path(root) / str(cid)
            if not cd.exists():
                continue
            grps = defaultdict(list)
            for f in cd.glob('*.jpg'):
                m = re.match(r'(\d+)_person(\d+)_(\w+)_(\d+)\.jpg', f.name)
                if m:
                    _, pid, view, si = m.groups()
                    if int(pid) in persons:
                        grps[(int(pid), view)].append((int(si), str(f)))
            for _, frames in grps.items():
                frames = sorted(frames, key=lambda x: x[0])
                if len(frames) < seq_len:
                    continue
                stride = seq_len // 3 if training else seq_len
                for i in range(0, len(frames) - seq_len + 1, stride):
                    self.samples.append((cid - 1, [p for _, p in frames[i:i + seq_len]]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, paths = self.samples[idx]
        imgs = []
        for p in paths:
            try:
                imgs.append(self.tf(Image.open(p).convert('RGB')))
            except Exception:
                imgs.append(torch.zeros(3, 224, 224))
        while len(imgs) < self.seq_len:
            imgs.append(imgs[-1] if imgs else torch.zeros(3, 224, 224))
        return torch.stack(imgs[:self.seq_len]), label


def train_model(model, train_loader, val_loader, name, lr=LEARNING_RATE):
    model.to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, NUM_EPOCHS)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled())
    best_val_score = 0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        correct = 0
        total = 0
        for x, y in tqdm(train_loader, desc=f"  {name} E{epoch + 1}", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with autocast_context():
                o = model(x)
                loss = crit(o, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            correct += (o.argmax(1) == y).sum().item()
            total += y.size(0)
            del x, y, o, loss
        sch.step()

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                with autocast_context():
                    val_correct += (model(x).argmax(1) == y).sum().item()
                val_total += y.size(0)
                del x, y

        val_score = val_correct / max(val_total, 1)
        train_score = 100 * correct / max(total, 1)
        print(f"    {name} E{epoch + 1}: Tr={train_score:.1f}% Val={100 * val_score:.1f}%")

        if val_score > best_val_score:
            best_val_score = val_score
            epochs_without_improvement = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"    Early stop E{epoch + 1}, best={100 * best_val_score:.1f}%")
                break

        gc.collect()
        torch.cuda.empty_cache()

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val_score

@torch.no_grad()
def evaluate(model, loader, name=""):
    model.eval()
    preds, labels = [], []
    for x, y in tqdm(loader, desc=f"  Test {name}", leave=False):
        x = x.to(DEVICE)
        with autocast_context():
            o = model(x)
        preds.extend(o.argmax(1).cpu().numpy())
        labels.extend(y.numpy())
        del x, o
    preds, labels = np.array(preds), np.array(labels)
    return {
        'accuracy':  float(accuracy_score(labels, preds)),
        'f1':        float(f1_score(labels, preds, average='macro', zero_division=0)),
        'precision': float(precision_score(labels, preds, average='macro', zero_division=0)),
        'recall':    float(recall_score(labels, preds, average='macro', zero_division=0)),
        'per_class_f1': f1_score(labels, preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0).tolist(),
        'per_class_recall': recall_score(labels, preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0).tolist(),
        'confusion_matrix': confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES))).tolist(),
        'n_seq': len(preds),
    }


class Visualizer:
    def __init__(self, out_dir):
        self.d = Path(out_dir)
        self.d.mkdir(parents=True, exist_ok=True)
        self.ours_c = '#E74C3C'
        self.others_c = ['#3498DB', '#2ECC71', '#9B59B6', '#F39C12', '#1ABC9C', '#E67E22']

    def _c(self, name, idx):
        return self.ours_c if 'CAST' in name else self.others_c[idx % len(self.others_c)]

    def plot_all(self, R):
        self._bar(R)
        self._radar(R)
        self._pcr(R)
        self._cm(R)
        self._save(R)

    def _bar(self, R):
        ms = ['accuracy', 'f1', 'precision', 'recall']
        ts = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        ns = list(R.keys())
        for ax, m, t in zip(axes.flat, ms, ts):
            vs = [R[n][m] * 100 for n in ns]
            cs = [self._c(n, i) for i, n in enumerate(ns)]
            bars = ax.bar(range(len(ns)), vs, color=cs, edgecolor='black', linewidth=0.5)
            for i, n in enumerate(ns):
                if 'CAST' in n:
                    bars[i].set_edgecolor('darkred')
                    bars[i].set_linewidth(2.5)
            ax.set_ylabel(f'{t} (%)', fontweight='bold', fontsize=11)
            ax.set_title(f'{t} Comparison', fontweight='bold', fontsize=13)
            ax.set_xticks(range(len(ns)))
            ax.set_xticklabels(ns, rotation=25, ha='right', fontsize=10)
            ax.set_ylim(0, max(vs) + 15)
            ax.grid(axis='y', ls='--', alpha=0.3)
            for i, v in enumerate(vs):
                ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')
        plt.suptitle('Visual Model Performance Comparison', fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(self.d / 'comparison_metrics.png', bbox_inches='tight', dpi=200)
        plt.close()
        print("  Saved: comparison_metrics.png")

    def _radar(self, R):
        ms = ['accuracy', 'f1', 'precision', 'recall']
        ls = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='polar'))
        angs = np.linspace(0, 2 * np.pi, 4, endpoint=False).tolist() + [0]
        for i, (n, r) in enumerate(R.items()):
            vs = [r[m] * 100 for m in ms] + [r[ms[0]] * 100]
            c = self._c(n, i)
            lw = 3.5 if 'CAST' in n else 1.5
            ax.plot(angs, vs, 'o-', lw=lw, label=n, color=c, ms=7)
            ax.fill(angs, vs, alpha=0.2 if 'CAST' in n else 0.05, color=c)
        ax.set_xticks(angs[:-1])
        ax.set_xticklabels(ls, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_title('Multi-metric Radar Chart', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=10)
        plt.tight_layout()
        plt.savefig(self.d / 'radar_chart.png', bbox_inches='tight', dpi=200)
        plt.close()
        print("  Saved: radar_chart.png")

    def _pcr(self, R):
        fig, ax = plt.subplots(figsize=(18, 8))
        ns = list(R.keys())
        x = np.arange(NUM_CLASSES)
        w = 0.8 / len(ns)
        for i, n in enumerate(ns):
            vs = [v * 100 for v in R[n]['per_class_recall']]
            ax.bar(x + (i - len(ns) / 2 + 0.5) * w, vs, w, label=n, color=self._c(n, i),
                   alpha=0.9 if 'CAST' in n else 0.7,
                   edgecolor='darkred' if 'CAST' in n else 'gray',
                   linewidth=1.5 if 'CAST' in n else 0.5)
        ax.set_ylabel('Recall (%)', fontweight='bold', fontsize=12)
        ax.set_title('Per-class Recall Comparison', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(GESTURE_NAMES, rotation=45, ha='right', fontsize=10)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_ylim(0, 115)
        ax.grid(axis='y', ls='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.d / 'per_class_recall.png', bbox_inches='tight', dpi=200)
        plt.close()
        print("  Saved: per_class_recall.png")

    def _cm(self, R):
        n = len(R)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
        if n == 1:
            axes = np.array([[axes]])
        axes = np.atleast_2d(axes)
        for idx, (nm, r) in enumerate(R.items()):
            ax = axes[idx // cols, idx % cols]
            cm = np.array(r['confusion_matrix'], dtype=float)
            s = cm.sum(1, keepdims=True)
            s[s == 0] = 1
            cmap = 'Reds' if 'CAST' in nm else 'Blues'
            sns.heatmap(cm/s, annot=True, fmt='.2f', cmap=cmap, ax=ax,
                        xticklabels=GESTURE_NAMES, yticklabels=GESTURE_NAMES, annot_kws={'size': 7})
            c = 'darkred' if 'CAST' in nm else 'black'
            ax.set_title(f'{nm}\nAcc={r["accuracy"]*100:.1f}% F1={r["f1"]*100:.1f}%',
                         fontsize=11, fontweight='bold', color=c)
            ax.set_xlabel('Predicted', fontsize=9)
            ax.set_ylabel('Actual', fontsize=9)
            ax.tick_params(axis='x', rotation=45, labelsize=7)
            ax.tick_params(axis='y', rotation=0, labelsize=7)
        for idx in range(n, rows * cols):
            axes[idx // cols, idx % cols].axis('off')
        plt.suptitle('Confusion Matrices (Normalized)', fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(self.d / 'confusion_matrices.png', bbox_inches='tight', dpi=200)
        plt.close()
        print("  Saved: confusion_matrices.png")

    def _save(self, R):
        rows = []
        for n, r in R.items():
            rows.append({'Model': n, 'Accuracy (%)': f"{r['accuracy']*100:.2f}",
                         'F1 (%)': f"{r['f1']*100:.2f}",
                          'Precision (%)': f"{r['precision']*100:.2f}",
                          'Recall (%)': f"{r['recall']*100:.2f}"})
        df = pd.DataFrame(rows)
        df.to_csv(self.d / 'results_summary.csv', index=False)
        with open(self.d / 'results.json', 'w', encoding='utf-8') as f:
            json.dump(R, f, indent=2)
        print("\n" + "=" * 80)
        print("  FINAL RESULTS - Visual Model Comparison")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)

def main():
    print("=" * 70)
    print("  Visual Model Comparison - FAIR")
    print("  CAST-Net: pretrained init + finetune + YOLO test")
    print("  Others: trained from scratch, plain test")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    print("\n[0] Building train/val datasets...")
    tr_ds = TrainDataset(TRAIN_DIR, [1, 2, 3, 4], SEQUENCE_LENGTH, True)
    va_ds = TrainDataset(TRAIN_DIR, [5], SEQUENCE_LENGTH, False)
    print(f"  Train={len(tr_ds)} Val={len(va_ds)}")
    tr_ld = TorchDataLoader(tr_ds, BATCH_SIZE, True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    va_ld = TorchDataLoader(va_ds, BATCH_SIZE, False, num_workers=NUM_WORKERS, pin_memory=True)

    print(f"\n{'='*60}")
    print(f"  [1] CAST-Net (Ours) - pretrained init + finetune")
    print(f"{'='*60}")

    cast_pth = CHECKPOINT_DIR / "CAST-Net_finetuned.pth"

    if cast_pth.exists():
        print(f"  Loading finetuned: {cast_pth}")
        cast_model = CASTNet(NUM_CLASSES).to(DEVICE)
        cast_model.load_state_dict(torch.load(cast_pth, map_location=DEVICE))
    else:
        cast_model = CASTNet(NUM_CLASSES)
        if PRETRAINED_WEIGHT_PATH.exists():
            state = torch.load(PRETRAINED_WEIGHT_PATH, map_location='cpu')
            model_keys = set(cast_model.state_dict().keys())
            state_keys = set(state.keys())
            print(f"  Pretrained init: {len(model_keys & state_keys)}/{len(model_keys)} keys matched")
            cast_model.load_state_dict(state, strict=False)
        else:
            print("  Pretrained weights not found, training from scratch")
        npar = sum(p.numel() for p in cast_model.parameters()) / 1e6
        print(f"  Params: {npar:.1f}M")
        cast_model, best_va = train_model(cast_model, tr_ld, va_ld, "CAST-Net", lr=5e-5)
        torch.save(cast_model.state_dict(), cast_pth)
        print(f"  Saved: {cast_pth} (Val={100*best_va:.1f}%)")

    print("  Testing with YOLO hand crop...")
    cropper = HandCropper()
    test_yolo = TestDatasetYOLO(TEST_DIR, SEQUENCE_LENGTH, cropper)
    loader_yolo = TorchDataLoader(test_yolo, BATCH_SIZE, False, num_workers=0, pin_memory=True)
    cast_model.to(DEVICE)
    res = evaluate(cast_model, loader_yolo, "CAST-Net")
    results['CAST-Net'] = res
    print(f"  => Acc={res['accuracy']*100:.1f}%  F1={res['f1']*100:.1f}%")
    del cast_model, loader_yolo, test_yolo, cropper
    gc.collect()
    torch.cuda.empty_cache()

    comp = {
        'Baseline':     BaselineNet,
        'ResNet-LSTM':  ResNetLSTMNet,
        'ConvGRU-Attn': ConvGRUNet,
        'TCN':          TemporalConvNet,
        'DepthCRNN':    DepthCRNNet,
    }

    for nm, model_cls in comp.items():
        pth = CHECKPOINT_DIR / f"{nm}.pth"
        if pth.exists():
            print(f"\n  {nm}: already trained")
            continue
        print(f"\n{'='*60}")
        print(f"  Training {nm}...")
        print(f"{'='*60}")
        m = model_cls(NUM_CLASSES)
        print(f"  Params: {sum(p.numel() for p in m.parameters()) / 1e6:.1f}M")
        m, bv = train_model(m, tr_ld, va_ld, nm, lr=LEARNING_RATE)
        torch.save(m.state_dict(), pth)
        print(f"  Saved: {pth} (Val={100*bv:.1f}%)")
        del m
        gc.collect()
        torch.cuda.empty_cache()

    del tr_ds, va_ds, tr_ld, va_ld
    gc.collect()

    test_plain = TestDatasetPlain(TEST_DIR, SEQUENCE_LENGTH)
    loader_plain = TorchDataLoader(test_plain, BATCH_SIZE, False, num_workers=NUM_WORKERS, pin_memory=True)

    print(f"\n{'='*60}")
    print(f"  [3] Evaluating comparison models")
    print(f"{'='*60}")
    for nm, model_cls in comp.items():
        pth = CHECKPOINT_DIR / f"{nm}.pth"
        if not pth.exists():
            continue
        m = model_cls(NUM_CLASSES).to(DEVICE)
        m.load_state_dict(torch.load(pth, map_location=DEVICE))
        res = evaluate(m, loader_plain, nm)
        results[nm] = res
        print(f"  {nm}: Acc={res['accuracy']*100:.1f}%  F1={res['f1']*100:.1f}%")
        del m
        gc.collect()
        torch.cuda.empty_cache()
    del test_plain, loader_plain
    gc.collect()

    print(f"\n{'='*60}")
    print(f"  [4] Charts")
    print(f"{'='*60}")
    Visualizer(OUTPUT_DIR).plot_all(results)

    print("\n  RANKING:")
    for rank, (n, r) in enumerate(
        sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True), 1):
        flag = " ★" if 'CAST' in n else ""
        print(f"    #{rank} {n:<15} Acc={r['accuracy']*100:.1f}%  F1={r['f1']*100:.1f}%{flag}")
    print(f"\n  Output: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
