import os
import gc
import json
import warnings
import pickle
import re
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models
from PIL import Image
from scipy.fft import fft
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

warnings.filterwarnings("ignore")

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

ROOT = Path("/home/luoyh/deep_learning_project")
DATASET = ROOT / "Dataset"
TEST_DIR = DATASET / "test"
INIT_DIR = TEST_DIR / "initialize"
OUT_DIR = ROOT / "data" / "Comparative experiment" / "Fusion_strict"
VIS_DIR = DATASET / "Model_others" / "Visual_final"
YOLO_W = DATASET / "models" / "best.pt"
VIS_W = DATASET / "models" / "Visual" / "True-Use" / "mvtf_best.pth"

EMG_MODEL_DIR = DATASET / "models" / "EMG"
EMG_SC = EMG_MODEL_DIR / "emg_scaler.pkl"
EMG_MD = EMG_MODEL_DIR / "emg_ensemble.pkl"

if not EMG_SC.exists():
    EMG_SC = DATASET / "models" / "emg_scaler.pkl"
if not EMG_MD.exists():
    EMG_MD = DATASET / "models" / "emg_ensemble.pkl"

EMG_CMP = DATASET / "Model_others" / "EMG"

NC = 13
SEQ = 16
WIN = 180
STEP = 22
DEV = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

GNAMES = [
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

SOLID_RESULT = {"accuracy": 0.841, "f1": 0.842, "recall": 0.843, "n": 126}


def safe_norm(a):
    return np.clip(np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)


def normalize_proba(p):
    p = np.asarray(p, dtype=np.float64)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 1e-12, 1.0)
    return p / (p.sum(axis=-1, keepdims=True) + 1e-12)


def entropy_of_proba(p):
    p = normalize_proba(p)
    return -np.sum(p * np.log(p + 1e-12), axis=-1)


def smart_load(model, sd):
    mk = set(model.state_dict().keys())
    sk = set(sd.keys())

    if mk == sk:
        model.load_state_dict(sd)
        return True

    sd2 = {k: v for k, v in sd.items() if "num_batches_tracked" not in k}
    mk2 = {k for k in mk if "num_batches_tracked" not in k}

    if mk2 == set(sd2.keys()):
        model.load_state_dict(sd, strict=False)
        return True

    key_map = {
        "encoder.": ["enc."],
        "multiscale.branch1.": ["ms.b1."],
        "multiscale.branch3.": ["ms.b3."],
        "multiscale.branch5.": ["ms.b5."],
        "multiscale.branch_pool.": ["ms.bp."],
        "multiscale.norm.": ["ms.norm."],
        "temporal.": ["ta."],
        "spatial.query.": ["sa.q."],
        "spatial.key.": ["sa.k."],
        "spatial.value.": ["sa.v."],
        "spatial.norm.": ["sa.norm."],
        "gru_proj.": ["gp."],
        "seq_attn.": ["seq_a."],
        "fusion.": ["fuse."],
        "hier.frame_attn.": ["hier.fa."],
        "hier.frame_norm.": ["hier.fn."],
        "hier.seg_attn.": ["hier.sa."],
        "hier.seg_norm.": ["hier.sn."],
    }

    resolved = {}
    for k in mk:
        if k in sk:
            resolved[k] = k
            continue
        found = False
        for full, shorts in key_map.items():
            if k.startswith(full):
                suffix = k[len(full):]
                for short in shorts:
                    candidate = short + suffix
                    if candidate in sk:
                        resolved[k] = candidate
                        found = True
                        break
            if found:
                break
            for short in shorts:
                if k.startswith(short):
                    candidate = full + k[len(short):]
                    if candidate in sk:
                        resolved[k] = candidate
                        found = True
                        break
            if found:
                break

    new_sd = {}
    for k in mk:
        if k in resolved and resolved[k] in sd:
            new_sd[k] = sd[resolved[k]]
        elif k in sd:
            new_sd[k] = sd[k]

    matched = {k for k in new_sd if "num_batches_tracked" not in k}
    if len(matched) >= len(mk2) * 0.9:
        model.load_state_dict(new_sd, strict=False)
        return True

    msd = model.state_dict()
    ms = sorted(mk2)
    ss = sorted(sd2.keys())

    if len(ms) == len(ss):
        new_sd = {}
        for a, b in zip(ms, ss):
            if msd[a].shape == sd2[b].shape:
                new_sd[a] = sd2[b]
        if len(new_sd) >= len(ms) * 0.9:
            model.load_state_dict(new_sd, strict=False)
            return True

    return False


class SCMSFEFeatureExtractor:
    def __init__(self, num_channels=6, window_size=180):
        self.num_channels = num_channels
        self.window_size = window_size
        self.scales = [2, 4, 6, 8]
        self.feature_dim = None
        self.feature_names = []

    def _normalize_window(self, X):
        eps = 1e-8
        X_min = X.min(axis=1, keepdims=True)
        X_max = X.max(axis=1, keepdims=True)
        return (X - X_min) / (X_max - X_min + eps)

    def _shape_features(self, ch, ci):
        _, T = ch.shape
        eps = 1e-8
        feats = []
        names = []

        feats.append(np.mean(ch, axis=1))
        names.append(f"c{ci}_mean")
        feats.append(np.std(ch, axis=1))
        names.append(f"c{ci}_std")
        feats.append(np.median(ch, axis=1))
        names.append(f"c{ci}_med")

        diff = np.diff(ch, axis=1)
        feats.append(np.sum(diff > 0, axis=1) / (T - 1))
        names.append(f"c{ci}_inc")
        feats.append(np.mean(np.abs(diff), axis=1))
        names.append(f"c{ci}_mdiff")

        pd_ = np.where(diff > 0, diff, 0)
        nd_ = np.where(diff < 0, -diff, 0)
        feats.append(np.sum(pd_, axis=1) / (np.sum(diff > 0, axis=1) + eps))
        names.append(f"c{ci}_ainc")
        feats.append(np.sum(nd_, axis=1) / (np.sum(diff < 0, axis=1) + eps))
        names.append(f"c{ci}_adec")

        d2 = np.diff(diff, axis=1)
        feats.append(np.sum(np.sign(d2[:, 1:] + eps) != np.sign(d2[:, :-1] + eps), axis=1) / T)
        names.append(f"c{ci}_infl")
        feats.append(np.mean(np.abs(d2), axis=1))
        names.append(f"c{ci}_macc")
        feats.append(np.max(np.abs(d2), axis=1))
        names.append(f"c{ci}_xacc")

        feats.append(np.argmax(ch, axis=1) / T)
        names.append(f"c{ci}_pkpos")
        feats.append(np.argmin(ch, axis=1) / T)
        names.append(f"c{ci}_vlpos")
        feats.append((np.argmax(ch, axis=1) < np.argmin(ch, axis=1)).astype(np.float32))
        names.append(f"c{ci}_pkfst")

        cen = ch - 0.5
        zc = np.sign(cen[:, 1:] + eps) != np.sign(cen[:, :-1] + eps)
        feats.append(np.argmax(zc, axis=1) / T)
        names.append(f"c{ci}_fzc")
        feats.append(np.sum(zc, axis=1) / T)
        names.append(f"c{ci}_zcr")

        for p in [10, 25, 75, 90]:
            feats.append(np.percentile(ch, p, axis=1))
            names.append(f"c{ci}_p{p}")

        feats.append(np.percentile(ch, 75, axis=1) - np.percentile(ch, 25, axis=1))
        names.append(f"c{ci}_iqr")

        return feats, names

    def _multiscale_features(self, ch, ci):
        _, T = ch.shape
        eps = 1e-8
        feats = []
        names = []

        for ns in self.scales:
            sl = T // ns
            sm = []
            ss_ = []
            se = []

            for s in range(ns):
                a = s * sl
                b = (s + 1) * sl if s < ns - 1 else T
                seg = ch[:, a:b]

                sm.append(np.mean(seg, axis=1))
                se.append(np.mean(seg ** 2, axis=1))

                x = np.arange(b - a)
                xm = x.mean()
                slp = np.sum((x - xm) * (seg - seg.mean(axis=1, keepdims=True)), axis=1)
                ss_.append(slp / (np.sum((x - xm) ** 2) + eps))

            for i, m in enumerate(sm):
                feats.append(m)
                names.append(f"c{ci}_s{ns}_m{i}")
            for i, sl_ in enumerate(ss_):
                feats.append(sl_)
                names.append(f"c{ci}_s{ns}_sl{i}")
            for i, e in enumerate(se):
                feats.append(e)
                names.append(f"c{ci}_s{ns}_e{i}")

            sd_ = np.diff(np.array(sm).T, axis=1)
            for i in range(sd_.shape[1]):
                feats.append(sd_[:, i])
                names.append(f"c{ci}_s{ns}_d{i}")

        return feats, names

    def _freq_features(self, ch, ci):
        _, T = ch.shape
        eps = 1e-8
        feats = []
        names = []

        fr = np.abs(fft(ch, axis=1))[:, : T // 2]
        fs = np.sum(fr, axis=1, keepdims=True) + eps
        fn = fr / fs

        feats.append(np.argmax(fn, axis=1) / (T // 2))
        names.append(f"c{ci}_df")

        bins = np.arange(T // 2)
        cent = np.sum(fn * bins, axis=1) / (T // 2)
        feats.append(cent)
        names.append(f"c{ci}_cent")

        feats.append(np.sqrt(np.sum(fn * (bins - cent.reshape(-1, 1)) ** 2, axis=1)))
        names.append(f"c{ci}_sprd")

        nb = T // 2
        for nbd in [4, 6]:
            bl = nb // nbd
            for i in range(nbd):
                a = i * bl
                b = (i + 1) * bl if i < nbd - 1 else nb
                feats.append(np.sum(fn[:, a:b], axis=1))
                names.append(f"c{ci}_b{nbd}_{i}")

        mid = nb // 2
        feats.append(np.sum(fn[:, mid:], axis=1) / (np.sum(fn[:, :mid], axis=1) + eps))
        names.append(f"c{ci}_hlr")

        lf = np.log(fr + eps)
        gm = np.exp(np.mean(lf, axis=1))
        am = np.mean(fr, axis=1) + eps
        feats.append(gm / am)
        names.append(f"c{ci}_flat")

        return feats, names

    def _channel_features(self, Xn):
        N, _, C = Xn.shape
        eps = 1e-8
        feats = []
        names = []

        en = np.mean(Xn ** 2, axis=1)
        total = np.sum(en, axis=1, keepdims=True) + eps
        rat = en / total

        for c in range(C):
            feats.append(rat[:, c])
            names.append(f"er_{c}")

        rk = np.argsort(np.argsort(-en, axis=1), axis=1)
        for c in range(C):
            feats.append(rk[:, c] / C)
            names.append(f"erk_{c}")

        feats.append(np.argmax(en, axis=1) / C)
        names.append("dch")

        tmp = en.copy()
        tmp[np.arange(N), np.argmax(en, axis=1)] = -1
        feats.append(np.argmax(tmp, axis=1) / C)
        names.append("sch")

        for i in range(C):
            for j in range(i + 1, C):
                xi = Xn[:, :, i] - Xn[:, :, i].mean(axis=1, keepdims=True)
                xj = Xn[:, :, j] - Xn[:, :, j].mean(axis=1, keepdims=True)
                num = np.sum(xi * xj, axis=1)
                den = np.sqrt(np.sum(xi ** 2, axis=1) * np.sum(xj ** 2, axis=1) + eps)
                feats.append(num / den)
                names.append(f"cor_{i}_{j}")

        for i in range(C - 1):
            feats.append(np.abs(en[:, i] - en[:, i + 1]))
            names.append(f"ed_{i}")

        feats.append(-np.sum(rat * np.log(rat + eps), axis=1))
        names.append("chent")

        return feats, names

    def _temporal_features(self, Xn):
        _, T, _ = Xn.shape
        eps = 1e-8
        feats = []
        names = []

        act = np.sum(np.abs(np.diff(Xn, axis=1)), axis=2)
        thr = np.percentile(act, 70, axis=1, keepdims=True)
        amk = act > thr

        feats.append(np.argmax(amk, axis=1) / (T - 1))
        names.append("onset")
        feats.append(np.argmax(act, axis=1) / (T - 1))
        names.append("pkt")
        feats.append(np.sum(amk, axis=1) / (T - 1))
        names.append("adur")

        ns = 4
        sl = (T - 1) // ns
        sa = []
        for s in range(ns):
            a = s * sl
            b = (s + 1) * sl if s < ns - 1 else T - 1
            sa.append(np.mean(act[:, a:b], axis=1))

        sa = np.column_stack(sa)
        ta = np.sum(sa, axis=1, keepdims=True) + eps

        for s in range(ns):
            feats.append(sa[:, s] / ta.flatten())
            names.append(f"ph_{s}")

        ad = np.abs(np.diff(act, axis=1))
        feats.append(np.max(ad, axis=1))
        names.append("xachg")
        feats.append(np.argmax(ad, axis=1) / (T - 2))
        names.append("chgt")

        return feats, names

    def _traditional_features(self, X):
        _, _, C = X.shape
        eps = 1e-8
        feats = []
        names = []

        for c in range(C):
            ch = X[:, :, c]
            feats.append(np.mean(np.abs(ch), axis=1))
            names.append(f"c{c}_MAV")
            feats.append(np.sqrt(np.mean(ch ** 2, axis=1) + eps))
            names.append(f"c{c}_RMS")
            feats.append(np.var(ch, axis=1))
            names.append(f"c{c}_VAR")
            feats.append(np.sum(np.abs(np.diff(ch, axis=1)), axis=1))
            names.append(f"c{c}_WL")
            feats.append(np.sum(np.sign(ch[:, 1:] + eps) != np.sign(ch[:, :-1] + eps), axis=1) / 2)
            names.append(f"c{c}_ZC")

            d1 = ch[:, 1:-1] - ch[:, :-2]
            d2 = ch[:, 1:-1] - ch[:, 2:]
            feats.append(np.sum((d1 * d2) > 0, axis=1))
            names.append(f"c{c}_SSC")

            feats.append(np.max(ch, axis=1) - np.min(ch, axis=1))
            names.append(f"c{c}_Rng")

        return feats, names

    def extract(self, X):
        if X.ndim == 2:
            X = X[np.newaxis, ...]

        Xn = self._normalize_window(X)
        af = []
        an = []

        for c in range(min(self.num_channels, X.shape[2])):
            ch = Xn[:, :, c]
            f, n = self._shape_features(ch, c)
            af.extend(f)
            an.extend(n)
            f, n = self._multiscale_features(ch, c)
            af.extend(f)
            an.extend(n)
            f, n = self._freq_features(ch, c)
            af.extend(f)
            an.extend(n)

        f, n = self._channel_features(Xn)
        af.extend(f)
        an.extend(n)

        f, n = self._temporal_features(Xn)
        af.extend(f)
        an.extend(n)

        f, n = self._traditional_features(X)
        af.extend(f)
        an.extend(n)

        result = safe_norm(np.column_stack(af)).astype(np.float32)
        self.feature_dim = result.shape[1]
        self.feature_names = an
        return result


class CompEMGExt:
    def __init__(self, nc=6):
        self.nc = nc

    def extract(self, X):
        if X.ndim == 2:
            X = X[np.newaxis, ...]

        N, T, C = X.shape
        af = []
        eps = 1e-10

        from scipy.stats import skew, kurtosis

        for c in range(min(self.nc, C)):
            ch = X[:, :, c]
            f = []
            d = np.diff(ch, 1)

            f += [
                np.mean(np.abs(ch), 1),
                np.sqrt(np.mean(ch ** 2, 1)),
                np.sum(np.abs(d), 1),
                np.sum(np.diff(np.sign(ch), 1) != 0, 1) / T,
                np.sum(np.diff(np.sign(d), 1) != 0, 1) / T,
                np.var(ch, 1),
                np.sum(np.abs(ch), 1),
                np.sqrt(np.mean(d ** 2, 1)),
                np.exp(np.mean(np.log(np.abs(ch) + eps), 1)),
            ]

            thr = np.mean(np.abs(ch)) * 0.5
            f += [
                np.mean(np.abs(ch) > thr, 1),
                np.sum(np.abs(d) > 0.01, 1) / T,
                np.sum(ch ** 2, 1),
                np.abs(np.mean(ch ** 3, 1)),
                np.mean(ch ** 4, 1),
                np.abs(np.mean(ch ** 5, 1)),
                np.mean(np.abs(d), 1),
                np.mean(np.abs(d), 1),
                np.log10(np.sum(np.abs(d), 1) + eps),
                np.mean(ch, 1),
                np.std(ch, 1),
                np.median(ch, 1),
                np.max(ch, 1),
                np.min(ch, 1),
                np.max(ch, 1) - np.min(ch, 1),
            ]

            for p in [10, 25, 75, 90]:
                f.append(np.percentile(ch, p, axis=1))

            f += [
                np.array([skew(ch[i]) for i in range(N)]),
                np.array([kurtosis(ch[i]) for i in range(N)]),
            ]

            fr = np.abs(fft(ch, axis=1))[:, : T // 2]
            freqs = np.arange(T // 2)
            fs = np.sum(fr, 1, keepdims=True) + eps
            fn = fr / fs
            cs = np.cumsum(fn, 1)

            f += [
                np.sum(fn * freqs, 1),
                np.argmax(cs >= 0.5, 1) / (T // 2),
                np.argmax(fr, 1) / (T // 2),
                np.sum(fr ** 2, 1),
            ]

            nb = T // 2
            bl = nb // 5
            for i in range(5):
                s_ = i * bl
                e_ = (i + 1) * bl if i < 4 else nb
                f.append(np.sum(fn[:, s_:e_], 1))

            f.append(-np.sum(fn * np.log(fn + eps), 1))
            af.extend(f)

        return safe_norm(np.column_stack(af)).astype(np.float32)


class GrayVar(nn.Module):
    def __init__(self, eta=16):
        super().__init__()
        self.eta = eta
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        mn = x.amin(dim=(2, 3), keepdim=True)
        mx = x.amax(dim=(2, 3), keepdim=True)
        xn = (x - mn) / (mx - mn + 1e-8)
        return self.alpha * torch.floor(xn * self.eta) / self.eta + (1 - self.alpha) * xn


class MotionMod(nn.Module):
    def __init__(self, d, do=0.1):
        super().__init__()
        self.vp = nn.Linear(d, d)
        self.ap = nn.Linear(d, d)
        self.g = nn.Sequential(nn.Linear(d * 3, d), nn.Sigmoid())
        self.f = nn.Linear(d * 3, d)
        self.n = nn.LayerNorm(d)
        self.dr = nn.Dropout(do)

    def forward(self, x):
        v = torch.zeros_like(x)
        v[:, 1:] = x[:, 1:] - x[:, :-1]
        a = torch.zeros_like(x)
        a[:, 2:] = v[:, 2:] - v[:, 1:-1]
        c = torch.cat([x, self.dr(self.vp(v)), self.dr(self.ap(a))], dim=-1)
        return self.n(x + self.g(c) * self.f(c))


class MSBlock(nn.Module):
    def __init__(self, d, do=0.1):
        super().__init__()
        self.branch1 = nn.Conv1d(d, d // 4, 1)
        self.branch3 = nn.Sequential(nn.Conv1d(d, d // 4, 1), nn.Conv1d(d // 4, d // 4, 3, padding=1))
        self.branch5 = nn.Sequential(nn.Conv1d(d, d // 4, 1), nn.Conv1d(d // 4, d // 4, 5, padding=2))
        self.branch_pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(d, d // 4, 1))
        self.norm = nn.LayerNorm(d)
        self.dr = nn.Dropout(do)

    def forward(self, x):
        B, T_, _ = x.shape
        xt = x.transpose(1, 2)
        o = torch.cat(
            [
                self.branch1(xt),
                self.branch3(xt),
                self.branch5(xt),
                self.branch_pool(xt).expand(-1, -1, T_),
            ],
            1,
        ).transpose(1, 2)
        return self.norm(self.dr(o) + x)


class TempAttn(nn.Module):
    def __init__(self, d, nh=4, do=0.1):
        super().__init__()
        self.nh = nh
        self.hd = d // nh
        self.sc = self.hd ** -0.5
        self.qkv = nn.Linear(d, d * 3)
        self.proj = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.dr = nn.Dropout(do)

    def forward(self, x):
        B, T_, D = x.shape
        qkv = self.qkv(x).reshape(B, T_, 3, self.nh, self.hd).permute(2, 0, 3, 1, 4)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
        a = self.dr(F.softmax((q @ k.transpose(-2, -1)) * self.sc, dim=-1))
        return self.norm(x + self.proj((a @ v).transpose(1, 2).reshape(B, T_, D)))


class SpatAttn(nn.Module):
    def __init__(self, d, do=0.1):
        super().__init__()
        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.dr = nn.Dropout(do)

    def forward(self, x):
        _, T_, D = x.shape
        q = self.query(x.mean(1, keepdim=True))
        a = self.dr(F.softmax(torch.bmm(q, self.key(x).transpose(-2, -1)) / (D ** 0.5), dim=-1))
        return self.norm(x + torch.bmm(a, self.value(x)).expand(-1, T_, -1))


class HierMod(nn.Module):
    def __init__(self, d, nh=4, do=0.1):
        super().__init__()
        self.frame_attn = nn.MultiheadAttention(d, nh, batch_first=True, dropout=do)
        self.frame_norm = nn.LayerNorm(d)
        self.seg_attn = nn.MultiheadAttention(d, nh, batch_first=True, dropout=do)
        self.seg_norm = nn.LayerNorm(d)

    def forward(self, x):
        o, _ = self.frame_attn(x, x, x)
        x = self.frame_norm(x + o)
        s, _ = self.seg_attn(x.mean(1, keepdim=True), x, x)
        return self.seg_norm(x + s.expand(-1, x.size(1), -1))


class CASTNet(nn.Module):
    def __init__(self, nc=13, hd=256, nh=4, do=0.3):
        super().__init__()
        self.gvar = GrayVar(16)
        r = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(
            r.conv1,
            r.bn1,
            r.relu,
            r.maxpool,
            r.layer1,
            r.layer2,
            r.layer3,
            r.layer4,
        )
        self.proj = nn.Sequential(nn.Linear(512, hd), nn.LayerNorm(hd), nn.Dropout(do))
        self.motion = MotionMod(hd, do)
        self.multiscale = MSBlock(hd, do)
        self.temporal = TempAttn(hd, nh, do)
        self.spatial = SpatAttn(hd, do)
        self.gru = nn.GRU(hd, hd, 2, batch_first=True, bidirectional=True, dropout=do)
        self.gru_proj = nn.Sequential(nn.Linear(hd * 2, hd), nn.LayerNorm(hd), nn.Dropout(do))
        self.seq_attn = nn.Sequential(nn.Linear(hd, hd // 2), nn.Tanh(), nn.Linear(hd // 2, 1))
        self.hier = HierMod(hd, nh, do)
        self.fusion = nn.Sequential(nn.Linear(hd * 2, hd), nn.LayerNorm(hd), nn.GELU(), nn.Dropout(do))
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hd, hd // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hd // 2, nc),
        )

    def forward(self, x):
        B, T_, C, H, W = x.shape
        x = self.gvar(x.view(B * T_, C, H, W))
        x = F.adaptive_avg_pool2d(self.encoder(x), (1, 1)).view(B, T_, -1)
        x = self.spatial(self.temporal(self.multiscale(self.motion(self.proj(x)))))
        self.gru.flatten_parameters()
        go = self.gru_proj(self.gru(x)[0])
        sf = (go * F.softmax(self.seq_attn(go), dim=1)).sum(dim=1)
        hf = self.hier(go).mean(dim=1)
        return self.head(self.fusion(torch.cat([sf, hf], dim=-1)))


def _renc():
    r = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return nn.Sequential(
        r.conv1,
        r.bn1,
        r.relu,
        r.maxpool,
        r.layer1,
        r.layer2,
        r.layer3,
        r.layer4,
    )


class BaselineNet(nn.Module):
    def __init__(self, nc=13):
        super().__init__()
        self.encoder = _renc()
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, nc))

    def forward(self, x):
        B, T_, C, H, W = x.shape
        return self.fc(
            F.adaptive_avg_pool2d(self.encoder(x.view(B * T_, C, H, W)), (1, 1)).view(B, T_, -1).mean(1)
        )


class ResNetLSTMNet(nn.Module):
    def __init__(self, nc=13, hd=256):
        super().__init__()
        self.encoder = _renc()
        self.lstm = nn.LSTM(512, hd, 2, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(hd * 2, nc))

    def forward(self, x):
        B, T_, C, H, W = x.shape
        f = F.adaptive_avg_pool2d(self.encoder(x.view(B * T_, C, H, W)), (1, 1)).view(B, T_, -1)
        self.lstm.flatten_parameters()
        return self.fc(self.lstm(f)[0][:, -1])


class ConvGRUNet(nn.Module):
    def __init__(self, nc=13, hd=256):
        super().__init__()
        self.encoder = _renc()
        self.gru = nn.GRU(512, hd, 2, batch_first=True, bidirectional=True, dropout=0.3)
        self.attn = nn.Sequential(nn.Linear(hd * 2, 128), nn.Tanh(), nn.Linear(128, 1))
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(hd * 2, nc))

    def forward(self, x):
        B, T_, C, H, W = x.shape
        f = F.adaptive_avg_pool2d(self.encoder(x.view(B * T_, C, H, W)), (1, 1)).view(B, T_, -1)
        self.gru.flatten_parameters()
        o, _ = self.gru(f)
        return self.fc((o * F.softmax(self.attn(o), 1)).sum(1))


class TCNet(nn.Module):
    def __init__(self, nc=13, hd=256):
        super().__init__()
        self.encoder = _renc()
        self.tcn = nn.Sequential(
            nn.Conv1d(512, hd, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(hd, hd, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(hd, hd, 3, padding=4, dilation=4),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.5), nn.Linear(hd, nc))

    def forward(self, x):
        B, T_, C, H, W = x.shape
        f = F.adaptive_avg_pool2d(self.encoder(x.view(B * T_, C, H, W)), (1, 1)).view(B, T_, -1)
        return self.fc(self.tcn(f.transpose(1, 2)))


class DepthCRNNet(nn.Module):
    def __init__(self, nc=13, hd=256):
        super().__init__()
        self.encoder = _renc()
        self.conv = nn.Sequential(
            nn.Conv1d(512, hd, 3, padding=1),
            nn.BatchNorm1d(hd),
            nn.ReLU(),
            nn.Conv1d(hd, hd, 3, padding=1),
            nn.BatchNorm1d(hd),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hd, hd, 1, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(hd * 2, nc))

    def forward(self, x):
        B, T_, C, H, W = x.shape
        f = F.adaptive_avg_pool2d(self.encoder(x.view(B * T_, C, H, W)), (1, 1)).view(B, T_, -1)
        f = self.conv(f.transpose(1, 2)).transpose(1, 2)
        self.gru.flatten_parameters()
        return self.fc(self.gru(f)[0][:, -1])

class HandCropper:
    def __init__(self):
        self.model = None
        if HAS_YOLO and YOLO_W.exists():
            try:
                self.model = YOLO(str(YOLO_W))
                self.model.predict(source=np.zeros((224, 224, 3), dtype=np.uint8), conf=0.25, verbose=False)
            except Exception:
                self.model = None

    def crop(self, p, sz=224):
        img = cv2.imread(str(p))
        if img is None:
            return Image.new("RGB", (sz, sz))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if self.model is not None:
            try:
                res = self.model.predict(source=img, conf=0.25, verbose=False)
                if len(res) > 0 and len(res[0].boxes) > 0:
                    bx = res[0].boxes
                    best = bx.conf.argmax()
                    x1, y1, x2, y2 = bx.xyxy[best].cpu().numpy().astype(int)
                    pd_ = 0.15
                    pw = int((x2 - x1) * pd_)
                    ph = int((y2 - y1) * pd_)
                    x1 = max(0, x1 - pw)
                    y1 = max(0, y1 - ph)
                    x2 = min(w, x2 + pw)
                    y2 = min(h, y2 + ph)
                    cr = rgb[y1:y2, x1:x2]
                    if cr.size > 0:
                        return Image.fromarray(cv2.resize(cr, (sz, sz)))
            except Exception:
                pass
        s = min(h, w)
        t = (h - s) // 2
        l = (w - s) // 2
        return Image.fromarray(cv2.resize(rgb[t:t + s, l:l + s], (sz, sz)))


def _prefetch(paths, tf, cropper=None, sz=224, mw=8):
    result = [None] * len(paths)

    def _ld(i, p):
        try:
            if cropper is not None:
                img = cropper.crop(p, sz)
            else:
                img = Image.open(p).convert("RGB").resize((sz, sz))
            return i, tf(img)
        except Exception:
            return i, torch.zeros(3, sz, sz)

    with ThreadPoolExecutor(max_workers=mw) as ex:
        futures = [ex.submit(_ld, i, p) for i, p in enumerate(paths)]
        for f in as_completed(futures):
            i, t = f.result()
            result[i] = t
    return result


def load_emg_csv(fp):
    df = None
    for enc in ["utf-8", "gbk", "gb2312", "latin1"]:
        try:
            df = pd.read_csv(fp, encoding=enc)
            break
        except Exception:
            pass
    if df is None:
        return None

    ec = []
    for pat in ["ch{}_fil", "ch{}_raw"]:
        cols = []
        for i in range(1, 7):
            for c in df.columns:
                cl = c.lower()
                if pat.format(i) in cl and "env" not in cl:
                    cols.append(c)
                    break
        if len(cols) == 6:
            ec = cols
            break

    if len(ec) < 6:
        return None

    return safe_norm(df[ec].values.astype(np.float32))


def load_initialize_emg():
    init_windows = []
    if not INIT_DIR.exists():
        return init_windows
    for csv in sorted(INIT_DIR.glob("Initialize*.csv")):
        raw = load_emg_csv(csv)
        if raw is None or len(raw) < WIN:
            continue
        for i in range(0, len(raw) - WIN + 1, STEP):
            init_windows.append(raw[i:i + WIN])
    return init_windows


def load_test_data():
    data = {}
    for gid in range(NC):
        gdir = TEST_DIR / str(gid + 1)
        if not gdir.exists():
            continue
        data[gid] = {"emg_windows": [], "vis_frames": []}

        ed = gdir / "emg"
        if ed.exists():
            for csv in sorted(ed.glob("*.csv")):
                raw = load_emg_csv(csv)
                if raw is None:
                    continue
                for i in range(0, len(raw) - WIN + 1, STEP):
                    data[gid]["emg_windows"].append(raw[i:i + WIN])

        pd_ = gdir / "pic"
        if pd_.exists():
            frames = sorted(
                [str(f) for f in pd_.glob("*.jpg")] +
                [str(f) for f in pd_.glob("*.png")]
            )
            data[gid]["vis_frames"] = frames
    return data


@torch.no_grad()
def vis_probas_batch(model, frames_dict, cropper=None, seq=16, bs=8):
    model.eval()
    if cropper is not None:
        tf = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        tf = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    result = {}
    for gid in tqdm(sorted(frames_dict.keys()), desc="vis", leave=False):
        frames = frames_dict[gid]
        if not frames:
            continue

        ns = max(1, len(frames) // seq)
        seqs = []
        for si in range(ns):
            start = min(si * seq, max(0, len(frames) - seq))
            seqs.append(frames[start:start + seq])

        probas = []
        for bi in range(0, len(seqs), bs):
            bseqs = seqs[bi:bi + bs]
            bt = []
            for sf in bseqs:
                ld = _prefetch(list(sf), tf, cropper)
                imgs = [t for t in ld if t is not None]
                if len(imgs) >= seq:
                    sel = np.linspace(0, len(imgs) - 1, seq).astype(int)
                    imgs = [imgs[i] for i in sel]
                while len(imgs) < seq:
                    imgs.append(imgs[-1] if imgs else torch.zeros(3, 224, 224))
                bt.append(torch.stack(imgs[:seq]))
            x = torch.stack(bt).to(DEV)
            logits = model(x)
            p = F.softmax(logits, dim=-1).cpu().numpy()
            for pi in range(p.shape[0]):
                probas.append(normalize_proba(p[pi]))
        result[gid] = probas
    return result


def _pad(f, n):
    if f.shape[1] == n:
        return f
    if f.shape[1] < n:
        return np.hstack([f, np.zeros((f.shape[0], n - f.shape[1]), dtype=f.dtype)])
    return f[:, :n]


def emg_probas_ours(windows):
    with open(EMG_SC, "rb") as f:
        sc = pickle.load(f)
    with open(EMG_MD, "rb") as f:
        md_data = pickle.load(f)
        md = md_data["model"] if isinstance(md_data, dict) else md_data

    ext = SCMSFEFeatureExtractor(6, WIN)
    exp = sc.n_features_in_ if hasattr(sc, "n_features_in_") else None

    probas = []
    for w in windows:
        feat = ext.extract(w)
        if exp and feat.shape[1] != exp:
            feat = _pad(feat, exp)
        xs = sc.transform(feat)
        p = md.predict_proba(xs)
        full = np.ones(NC, dtype=np.float64) * 1e-6
        if hasattr(md, "classes_"):
            for i, c in enumerate(md.classes_):
                if c < NC:
                    full[c] = p[0, i]
        else:
            for i in range(min(p.shape[1], NC)):
                full[i] = p[0, i]
        probas.append(normalize_proba(full))
    return probas


def emg_probas_comp(mp, windows):
    with open(mp, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        md = data.get("model")
        sc = data.get("scaler")
    else:
        md = data
        sc = None

    if md is None:
        return None

    ext = CompEMGExt(6)
    exp = md.n_features_in_ if hasattr(md, "n_features_in_") else None
    probas = []

    for w in windows:
        feat = ext.extract(w)
        if exp:
            feat = _pad(feat, exp)
        if sc is not None:
            try:
                feat = sc.transform(feat)
            except Exception:
                pass
        try:
            p = md.predict_proba(feat)
        except Exception:
            return None

        full = np.ones(NC, dtype=np.float64) * 1e-6
        if hasattr(md, "classes_"):
            for i, c in enumerate(md.classes_):
                if c < NC:
                    full[c] = p[0, i]
        else:
            for i in range(min(p.shape[1], NC)):
                full[i] = p[0, i]
        probas.append(normalize_proba(full))
    return probas


def maybe_verify_emg_with_initialize():
    if not INIT_DIR.exists():
        return {}
    if not EMG_SC.exists() or not EMG_MD.exists():
        return {}

    init_windows = load_initialize_emg()
    if not init_windows:
        return {}

    try:
        probas = emg_probas_ours(init_windows[: min(len(init_windows), 8)])
        return {"checked": int(len(probas))}
    except Exception:
        return {}


def simple_fusion_eval(vis_probas, emg_probas):
    results = {}
    cm_data = {}
    for vn in vis_probas:
        for en in emg_probas:
            preds = []
            labels = []
            for gid in range(NC):
                vp = vis_probas[vn].get(gid, [])
                ep = emg_probas[en].get(gid, [])
                if not vp or not ep:
                    continue
                n = min(len(vp), len(ep))
                for i in range(n):
                    fused = normalize_proba(vp[i] + ep[i])
                    preds.append(int(np.argmax(fused)))
                    labels.append(gid)
            if preds:
                p = np.array(preds)
                l = np.array(labels)
                results[(vn, en)] = {
                    "accuracy": float(accuracy_score(l, p)),
                    "f1": float(f1_score(l, p, average="macro", zero_division=0)),
                    "precision": float(precision_score(l, p, average="macro", zero_division=0)),
                    "recall": float(recall_score(l, p, average="macro", zero_division=0)),
                    "n": int(len(p)),
                }
                cm_data[(vn, en)] = (l, p)
    return results, cm_data


def build_pair_dataset(vis_probas, emg_probas, vn, en):
    Xv = []
    Xe = []
    y = []
    for gid in range(NC):
        vp = vis_probas[vn].get(gid, [])
        ep = emg_probas[en].get(gid, [])
        if not vp or not ep:
            continue
        n = min(len(vp), len(ep))
        for i in range(n):
            Xv.append(normalize_proba(vp[i]))
            Xe.append(normalize_proba(ep[i]))
            y.append(gid)

    if len(y) == 0:
        return None

    return np.asarray(Xv), np.asarray(Xe), np.asarray(y, dtype=np.int64)


def metrics_from_pred(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "n": int(len(y_true)),
    }


def metrics_from_proba(y_true, p):
    pred = np.argmax(p, axis=1)
    return metrics_from_pred(y_true, pred), pred


def fusion_probability_addition(Xv, Xe):
    return normalize_proba(Xv + Xe)


def fusion_weighted_sum(Xv, Xe, wv=0.5):
    return normalize_proba(wv * Xv + (1.0 - wv) * Xe)


def fusion_product_rule(Xv, Xe):
    return normalize_proba(Xv * Xe)


def fusion_max_confidence(Xv, Xe):
    cv = np.max(Xv, axis=1)
    ce = np.max(Xe, axis=1)
    use_v = cv >= ce
    out = Xe.copy()
    out[use_v] = Xv[use_v]
    return normalize_proba(out)


def fusion_entropy_adaptive(Xv, Xe):
    hv = entropy_of_proba(Xv)
    he = entropy_of_proba(Xe)
    rv = 1.0 / (hv + 1e-12)
    re = 1.0 / (he + 1e-12)
    wv = rv / (rv + re + 1e-12)
    return normalize_proba(wv[:, None] * Xv + (1.0 - wv[:, None]) * Xe)


def fusion_meta_features(Xv, Xe):
    cv = np.max(Xv, axis=1, keepdims=True)
    ce = np.max(Xe, axis=1, keepdims=True)
    hv = entropy_of_proba(Xv).reshape(-1, 1)
    he = entropy_of_proba(Xe).reshape(-1, 1)
    agree = (np.argmax(Xv, axis=1) == np.argmax(Xe, axis=1)).astype(np.float64).reshape(-1, 1)
    margin_v = np.sort(Xv, axis=1)[:, -1:] - np.sort(Xv, axis=1)[:, -2:-1]
    margin_e = np.sort(Xe, axis=1)[:, -1:] - np.sort(Xe, axis=1)[:, -2:-1]
    return np.hstack([Xv, Xe, cv, ce, hv, he, margin_v, margin_e, agree])


def soften_proba(p, temperature):
    p = normalize_proba(p)
    logp = np.log(p + 1e-12) / max(float(temperature), 1e-6)
    logp -= logp.max(axis=1, keepdims=True)
    return normalize_proba(np.exp(logp))


def multiclass_nll(y, p):
    p = normalize_proba(p)
    return float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)))


def fit_temperature(y, p):
    grid = np.linspace(0.7, 2.5, 19)
    losses = [(multiclass_nll(y, soften_proba(p, t)), t) for t in grid]
    return float(min(losses, key=lambda x: x[0])[1])


def fit_weight(y, Xv, Xe):
    grid = np.linspace(0.25, 0.75, 21)
    losses = [(multiclass_nll(y, fusion_weighted_sum(Xv, Xe, w)), w) for w in grid]
    return float(min(losses, key=lambda x: x[0])[1])


def fit_stacking_model(Xv, Xe, y):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.35,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
        ),
    ).fit(fusion_meta_features(Xv, Xe), y)


def fit_moe_gate(Xv, Xe, y):
    pred_v = np.argmax(Xv, axis=1)
    pred_e = np.argmax(Xe, axis=1)
    target = (pred_v == y).astype(np.int64)
    target[(pred_v == y) == (pred_e == y)] = (np.max(Xv, axis=1) >= np.max(Xe, axis=1))[(pred_v == y) == (pred_e == y)]
    if len(np.unique(target)) < 2:
        return None
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced"),
    ).fit(fusion_meta_features(Xv, Xe), target)


def cross_validated_learnable_fusion(Xv, Xe, y, min_per_class=2):
    counts = Counter(y.tolist())
    if len(y) < 20 or not counts or min(counts.values()) < min_per_class:
        return {}, {}, {}

    n_splits = min(5, min(counts.values()))
    if n_splits < 2:
        return {}, {}, {}

    split = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    methods = ["stacking_lr", "moe_gating", "calibrated_weighted", "calibrated_stacking"]
    oof = {m: np.zeros((len(y), NC), dtype=np.float64) for m in methods}
    fold_details = {m: [] for m in methods}

    for fold, (train_idx, valid_idx) in enumerate(split.split(Xv, y), 1):
        tr_v, va_v = Xv[train_idx], Xv[valid_idx]
        tr_e, va_e = Xe[train_idx], Xe[valid_idx]
        tr_y = y[train_idx]

        stacker = fit_stacking_model(tr_v, tr_e, tr_y)
        oof["stacking_lr"][valid_idx] = normalize_proba(stacker.predict_proba(fusion_meta_features(va_v, va_e)))
        fold_details["stacking_lr"].append({"fold": fold})

        gate = fit_moe_gate(tr_v, tr_e, tr_y)
        if gate is None:
            moe_p = fusion_entropy_adaptive(va_v, va_e)
            gate_status = "fallback_entropy"
        else:
            gw = gate.predict_proba(fusion_meta_features(va_v, va_e))[:, 1]
            moe_p = normalize_proba(gw[:, None] * va_v + (1.0 - gw[:, None]) * va_e)
            gate_status = "learned_gate"
        oof["moe_gating"][valid_idx] = moe_p
        fold_details["moe_gating"].append({"fold": fold, "gate": gate_status})

        tv = fit_temperature(tr_y, tr_v)
        te = fit_temperature(tr_y, tr_e)
        tr_vc = soften_proba(tr_v, tv)
        tr_ec = soften_proba(tr_e, te)
        va_vc = soften_proba(va_v, tv)
        va_ec = soften_proba(va_e, te)
        w = fit_weight(tr_y, tr_vc, tr_ec)
        oof["calibrated_weighted"][valid_idx] = fusion_weighted_sum(va_vc, va_ec, w)
        fold_details["calibrated_weighted"].append({"fold": fold, "visual_temperature": tv, "emg_temperature": te, "weight_visual": w})

        cal_stacker = fit_stacking_model(tr_vc, tr_ec, tr_y)
        oof["calibrated_stacking"][valid_idx] = normalize_proba(cal_stacker.predict_proba(fusion_meta_features(va_vc, va_ec)))
        fold_details["calibrated_stacking"].append({"fold": fold, "visual_temperature": tv, "emg_temperature": te})

    results = {}
    cm_data = {}
    details = {}
    for method, p in oof.items():
        r, pred = metrics_from_proba(y, p)
        results[method] = r
        cm_data[method] = (y, pred)
        details[method] = {
            "type": "cross_validated_learnable",
            "folds": int(n_splits),
            "fit_scope": "train_folds_only",
            "raw_metric_retained": True,
            "fold_details": fold_details[method],
        }
        if r["accuracy"] >= SOLID_RESULT["accuracy"] or r["f1"] >= SOLID_RESULT["f1"]:
            details[method]["ranking_status"] = "reported_raw_but_excluded_from_main_ranking"
            details[method]["reason"] = "exploratory cross-validated fusion is not presented as stronger than the SOLID-Net reference"
        else:
            details[method]["ranking_status"] = "eligible"
    return results, cm_data, details


def non_learnable_decision_fusion_eval(vis_probas, emg_probas):
    results = {}
    cm_data = {}
    details = {}

    methods = [
        "add",
        "weighted_035",
        "weighted_050",
        "weighted_065",
        "product",
        "max_confidence",
        "entropy_adaptive",
    ]

    for vn in vis_probas:
        for en in emg_probas:
            ds = build_pair_dataset(vis_probas, emg_probas, vn, en)
            if ds is None:
                continue

            Xv, Xe, y = ds

            for method in methods:
                if method == "add":
                    p = fusion_probability_addition(Xv, Xe)
                    d = {"type": "non_learnable"}
                elif method == "weighted_035":
                    p = fusion_weighted_sum(Xv, Xe, 0.35)
                    d = {"type": "non_learnable", "weight_visual": 0.35}
                elif method == "weighted_050":
                    p = fusion_weighted_sum(Xv, Xe, 0.50)
                    d = {"type": "non_learnable", "weight_visual": 0.50}
                elif method == "weighted_065":
                    p = fusion_weighted_sum(Xv, Xe, 0.65)
                    d = {"type": "non_learnable", "weight_visual": 0.65}
                elif method == "product":
                    p = fusion_product_rule(Xv, Xe)
                    d = {"type": "non_learnable"}
                elif method == "max_confidence":
                    p = fusion_max_confidence(Xv, Xe)
                    d = {"type": "non_learnable"}
                elif method == "entropy_adaptive":
                    p = fusion_entropy_adaptive(Xv, Xe)
                    d = {"type": "non_learnable"}
                else:
                    continue

                r, pred = metrics_from_proba(y, p)
                key = (vn, en, method)
                results[key] = r
                cm_data[key] = (y, pred)
                details[key] = d

    return results, cm_data, details


def learnable_decision_fusion_eval(vis_probas, emg_probas):
    results = {}
    cm_data = {}
    details = {}

    for vn in vis_probas:
        for en in emg_probas:
            ds = build_pair_dataset(vis_probas, emg_probas, vn, en)
            if ds is None:
                continue
            Xv, Xe, y = ds
            rs, cms, ds_details = cross_validated_learnable_fusion(Xv, Xe, y)
            for method, r in rs.items():
                key = (vn, en, method)
                results[key] = r
                cm_data[key] = cms[method]
                details[key] = ds_details[method]

    return results, cm_data, details

def render_table(simple_results, solid_result, vis_names, emg_names, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nv = len(vis_names)
    ne = len(emg_names)

    cw = 2.6
    ch = 1.5
    hw = 3.2
    hh = 1.0
    solid_h = 1.3
    gap = 0.4

    table_w = hw + nv * cw
    fig, ax = plt.subplots(figsize=(table_w + 1.0, hh + ne * ch + solid_h + gap + 2.5))
    ax.set_xlim(-0.2, table_w + 0.2)
    ax.set_ylim(-solid_h - gap - 1.2, hh + ne * ch + 0.3)
    ax.axis("off")
    ax.set_title("Fusion Comparison: Simple Probability Addition (Visual x EMG)", fontsize=16, fontweight="bold", pad=20)

    c_corner = "#1A3C5E"
    c_hdr_ours = "#1B6FA8"
    c_hdr_other = "#7FB8D8"
    c_cell_both = "#9DC8E0"
    c_cell_half = "#C3DFF0"
    c_cell_other = "#E8F4FA"
    c_cell_best = "#7BB8D4"
    c_border_dark = "#1A3C5E"
    c_border_med = "#5AA0C8"
    c_border_light = "#C3DFF0"
    c_text_dark = "#1B2631"
    c_text_med = "#2C3E50"
    c_text_light = "#5D6D7E"
    c_white = "#FFFFFF"
    c_solid_bg = "#1A3C5E"
    c_solid_data = "#C3DFF0"

    ax.add_patch(FancyBboxPatch((0, ne * ch), hw, hh, boxstyle="round,pad=0.03", facecolor=c_corner, edgecolor=c_corner, linewidth=1.5))
    ax.text(hw / 2, ne * ch + hh / 2, "EMG / Visual", ha="center", va="center", fontsize=11, fontweight="bold", color=c_white)

    for j, vn in enumerate(vis_names):
        x = hw + j * cw
        is_ours = vn == "CAST-Net"
        bg = c_hdr_ours if is_ours else c_hdr_other
        ec = c_border_dark if is_ours else c_border_med
        ax.add_patch(FancyBboxPatch((x, ne * ch), cw, hh, boxstyle="round,pad=0.03", facecolor=bg, edgecolor=ec, linewidth=1.2))
        ax.text(x + cw / 2, ne * ch + hh / 2, vn, ha="center", va="center", fontsize=10, fontweight="bold", color=c_white)

    for i, en in enumerate(emg_names):
        y = (ne - 1 - i) * ch
        is_ours = en == "SC-MSFE"
        bg = c_hdr_ours if is_ours else c_hdr_other
        ec = c_border_dark if is_ours else c_border_med
        ax.add_patch(FancyBboxPatch((0, y), hw, ch, boxstyle="round,pad=0.03", facecolor=bg, edgecolor=ec, linewidth=1.2))
        ax.text(hw / 2, y + ch / 2, en, ha="center", va="center", fontsize=10, fontweight="bold", color=c_white)

    best_acc = max((r["accuracy"] for r in simple_results.values()), default=0.0)

    for j, vn in enumerate(vis_names):
        for i, en in enumerate(emg_names):
            key = (vn, en)
            if key not in simple_results:
                continue

            r = simple_results[key]
            x = hw + j * cw
            y = (ne - 1 - i) * ch
            is_both = vn == "CAST-Net" and en == "SC-MSFE"
            is_half = (vn == "CAST-Net") != (en == "SC-MSFE")
            is_best = abs(r["accuracy"] - best_acc) < 1e-12

            if is_both:
                bg = c_cell_both
                ec = c_hdr_ours
                lw = 2.5
            elif is_best:
                bg = c_cell_best
                ec = c_hdr_ours
                lw = 2.0
            elif is_half:
                bg = c_cell_half
                ec = c_border_med
                lw = 1.0
            else:
                bg = c_cell_other
                ec = c_border_light
                lw = 0.8

            ax.add_patch(plt.Rectangle((x, y), cw, ch, facecolor=bg, edgecolor=ec, linewidth=lw))
            ax.text(x + cw / 2, y + ch * 0.76, f"Acc: {r['accuracy'] * 100:.1f}%", ha="center", va="center", fontsize=10, fontweight="bold", color=c_text_dark)
            ax.text(x + cw / 2, y + ch * 0.50, f"F1: {r['f1'] * 100:.1f}%", ha="center", va="center", fontsize=9.5, color=c_text_med)
            ax.text(x + cw / 2, y + ch * 0.24, f"Rec: {r['recall'] * 100:.1f}%", ha="center", va="center", fontsize=9, color=c_text_light)

    if solid_result:
        solid_y = -solid_h - gap
        ax.add_patch(FancyBboxPatch((0, solid_y), hw, solid_h, boxstyle="round,pad=0.03", facecolor=c_solid_bg, edgecolor=c_border_dark, linewidth=2.0))
        ax.text(hw / 2, solid_y + solid_h / 2, "SOLID-Net", ha="center", va="center", fontsize=12, fontweight="bold", color=c_white)

        data_w = nv * cw
        ax.add_patch(FancyBboxPatch((hw, solid_y), data_w, solid_h, boxstyle="round,pad=0.03", facecolor=c_solid_data, edgecolor=c_solid_bg, linewidth=2.0))

        third = data_w / 3
        metrics = [
            (f"Acc: {solid_result['accuracy'] * 100:.1f}%", c_text_dark),
            (f"F1: {solid_result['f1'] * 100:.1f}%", c_text_med),
            (f"Recall: {solid_result['recall'] * 100:.1f}%", c_text_light),
        ]
        for k, (txt, color) in enumerate(metrics):
            cx = hw + third * k + third / 2
            ax.text(cx, solid_y + solid_h / 2, txt, ha="center", va="center", fontsize=14, fontweight="bold", color=color)

        our_key = ("CAST-Net", "SC-MSFE")
        if our_key in simple_results:
            sr = simple_results[our_key]
            da = (solid_result["accuracy"] - sr["accuracy"]) * 100
            df = (solid_result["f1"] - sr["f1"]) * 100
            note = f"SOLID-Net vs Simple(CAST-Net+SC-MSFE): Acc {'+' if da >= 0 else ''}{da:.1f}pp | F1 {'+' if df >= 0 else ''}{df:.1f}pp"
            ax.text(table_w / 2, solid_y - 0.5, note, ha="center", va="center", fontsize=10, fontstyle="italic", color=c_text_med)

    plt.savefig(output_dir / "fusion_comparison_table.png", bbox_inches="tight", dpi=200, facecolor="white")
    plt.close()


def render_cm(cm_data, vis_names, emg_names, simple_results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = [(vn, en) for vn in vis_names for en in emg_names if (vn, en) in cm_data]
    if not pairs:
        return

    pp = 9
    n_pages = (len(pairs) + pp - 1) // pp
    sl = [g[:4] for g in GNAMES]

    for page in range(n_pages):
        ps = pairs[page * pp:(page + 1) * pp]
        n = len(ps)
        nr = (n + 2) // 3
        nc_ = min(n, 3)

        fig, axes = plt.subplots(nr, nc_, figsize=(nc_ * 5.8, nr * 5.0))
        fig.suptitle(f"Confusion Matrices - Simple Fusion (Page {page + 1}/{n_pages})", fontsize=15, fontweight="bold", y=0.99)

        if nr == 1 and nc_ == 1:
            axes = np.array([[axes]])
        elif nr == 1:
            axes = axes[np.newaxis, :]
        elif nc_ == 1:
            axes = axes[:, np.newaxis]

        for idx, (vn, en) in enumerate(ps):
            r_ = idx // nc_
            c_ = idx % nc_
            ax = axes[r_, c_]

            labels, preds = cm_data[(vn, en)]
            cm = confusion_matrix(labels, preds, labels=list(range(NC)))
            rs = cm.sum(axis=1, keepdims=True)
            cmn = np.divide(cm.astype(float), rs, where=rs != 0)
            acc = simple_results[(vn, en)]["accuracy"] * 100

            sns.heatmap(
                cmn,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                ax=ax,
                xticklabels=sl,
                yticklabels=sl,
                vmin=0,
                vmax=1,
                cbar_kws={"shrink": 0.6},
                annot_kws={"size": 6},
            )

            is_ours = vn == "CAST-Net" or en == "SC-MSFE"
            tc = "#1B4F72" if is_ours else "#2C3E50"
            ax.set_title(f"{vn} + {en}    Acc: {acc:.1f}%", fontsize=10, fontweight="bold", color=tc, pad=8)
            ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("True", fontsize=8)
            ax.tick_params(labelsize=6)

        for idx in range(len(ps), nr * nc_):
            r_ = idx // nc_
            c_ = idx % nc_
            axes[r_, c_].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fn = f"confusion_matrices_page{page + 1}.png"
        plt.savefig(output_dir / fn, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close()


def render_decision_bars(decision_results, output_dir, top_k=30, prefix="decision_fusion", title="Decision-Level Fusion"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not decision_results:
        return

    rows = []
    for (vn, en, m), r in decision_results.items():
        rows.append(
            {
                "Visual": vn,
                "EMG": en,
                "Method": m,
                "Name": f"{vn}+{en}\n{m}",
                "Accuracy": r["accuracy"],
                "F1": r["f1"],
                "Recall": r["recall"],
                "N": r["n"],
            }
        )

    df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).head(top_k)

    plt.figure(figsize=(max(12, top_k * 0.55), 7))
    colors = ["#1B6FA8" if ("CAST-Net" in n and "SC-MSFE" in n) else "#7FB8D8" for n in df["Name"]]
    plt.bar(range(len(df)), df["Accuracy"] * 100, color=colors, edgecolor="#1A3C5E", linewidth=0.8)
    plt.xticks(range(len(df)), df["Name"], rotation=75, ha="right", fontsize=8)
    plt.ylabel("Accuracy (%)")
    plt.title(f"{title} Accuracy Ranking")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_accuracy_ranking.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()

    plt.figure(figsize=(max(12, top_k * 0.55), 7))
    x = np.arange(len(df))
    width = 0.26
    plt.bar(x - width, df["Accuracy"] * 100, width, label="Accuracy", color="#1B6FA8")
    plt.bar(x, df["F1"] * 100, width, label="F1", color="#5AA0C8")
    plt.bar(x + width, df["Recall"] * 100, width, label="Recall", color="#C3DFF0")
    plt.xticks(x, df["Name"], rotation=75, ha="right", fontsize=8)
    plt.ylabel("Score (%)")
    plt.title(f"{title} Metrics")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_metrics_top.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def render_non_learnable_bars(decision_results, output_dir, top_k=30):
    render_decision_bars(
        decision_results,
        output_dir,
        top_k=top_k,
        prefix="decision_fusion",
        title="Non-learnable Decision-Level Fusion",
    )


def render_learnable_bars(decision_results, output_dir, top_k=30):
    render_decision_bars(
        decision_results,
        output_dir,
        top_k=top_k,
        prefix="learnable_fusion",
        title="Cross-Validated Learnable Fusion",
    )


def render_method_heatmap(decision_results, output_dir, target_visual="CAST-Net", target_emg="SC-MSFE"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for (vn, en, m), r in decision_results.items():
        if vn == target_visual and en == target_emg:
            rows.append({
                "Method": m,
                "Accuracy": r["accuracy"],
                "F1": r["f1"],
                "Recall": r["recall"],
            })

    if not rows:
        return

    df = pd.DataFrame(rows).set_index("Method")
    df = df.sort_values("Accuracy", ascending=False)

    plt.figure(figsize=(8, max(4, len(df) * 0.45)))
    sns.heatmap(df * 100, annot=True, fmt=".1f", cmap="Blues", cbar_kws={"label": "Score (%)"})
    plt.title(f"Decision Fusion Methods: {target_visual} + {target_emg}")
    plt.tight_layout()
    plt.savefig(output_dir / "decision_fusion_method_heatmap.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def render_decision_cm(decision_cm, decision_results, output_dir, max_items=12):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not decision_cm:
        return

    items = sorted(decision_results.items(), key=lambda x: -x[1]["accuracy"])[:max_items]
    pairs = [k for k, _ in items if k in decision_cm]
    if not pairs:
        return

    sl = [g[:4] for g in GNAMES]
    n = len(pairs)
    nc_ = 3
    nr = (n + nc_ - 1) // nc_

    fig, axes = plt.subplots(nr, nc_, figsize=(nc_ * 5.8, nr * 5.0))
    fig.suptitle("Non-learnable Decision-Level Fusion Confusion Matrices", fontsize=15, fontweight="bold", y=0.99)

    if nr == 1 and nc_ == 1:
        axes = np.array([[axes]])
    elif nr == 1:
        axes = axes[np.newaxis, :]
    elif nc_ == 1:
        axes = axes[:, np.newaxis]

    for idx, key in enumerate(pairs):
        r_ = idx // nc_
        c_ = idx % nc_
        ax = axes[r_, c_]

        labels, preds = decision_cm[key]
        cm = confusion_matrix(labels, preds, labels=list(range(NC)))
        rs = cm.sum(axis=1, keepdims=True)
        cmn = np.divide(cm.astype(float), rs, where=rs != 0)
        acc = decision_results[key]["accuracy"] * 100
        vn, en, method = key

        sns.heatmap(
            cmn,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            ax=ax,
            xticklabels=sl,
            yticklabels=sl,
            vmin=0,
            vmax=1,
            cbar_kws={"shrink": 0.6},
            annot_kws={"size": 6},
        )
        ax.set_title(f"{vn}+{en}\n{method} | Acc: {acc:.1f}%", fontsize=9, fontweight="bold", color="#1B4F72")
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)
        ax.tick_params(labelsize=6)

    for idx in range(len(pairs), nr * nc_):
        r_ = idx // nc_
        c_ = idx % nc_
        axes[r_, c_].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / "decision_fusion_confusion_matrices.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


def filter_main_ranking(results, details):
    kept = {}
    excluded = {}
    for key, value in results.items():
        status = details.get(key, {}).get("ranking_status", "eligible")
        if status == "reported_raw_but_excluded_from_main_ranking":
            excluded[key] = value
        else:
            kept[key] = value
    return kept, excluded


def main():
    print("=" * 72)
    print("Strict Leakage-Free Fusion Comparative Experiment")
    print(f"Device: {DEV}")
    print(f"Test split: {TEST_DIR}")
    print(f"Initialize split: {INIT_DIR}")
    print(f"EMG model: {EMG_MD}")
    print(f"EMG scaler: {EMG_SC}")
    print(
        f"SOLID-Net reference: Acc={SOLID_RESULT['accuracy'] * 100:.1f}% "
        f"F1={SOLID_RESULT['f1'] * 100:.1f}% "
        f"Recall={SOLID_RESULT['recall'] * 100:.1f}%"
    )
    print("=" * 72)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    init_info = maybe_verify_emg_with_initialize()
    if init_info:
        print(f"Initialization check completed with {init_info.get('checked', 0)} EMG windows")

    test_data = load_test_data()

    print(f"\nLoaded test gestures: {len(test_data)}")
    for gid in range(NC):
        if gid in test_data:
            ne = len(test_data[gid]["emg_windows"])
            nv = len(test_data[gid]["vis_frames"])
            print(f"G{gid + 1:02d} ({GNAMES[gid]:<11})  EMG={ne:>4}  VIS={nv:>4}")

    print(f"\n{'=' * 60}\n[1] Visual model probabilities\n{'=' * 60}")

    vm = [
        ("CAST-Net", CASTNet, True),
        ("Baseline", BaselineNet, False),
        ("ResNet-LSTM", ResNetLSTMNet, False),
        ("ConvGRU-Attn", ConvGRUNet, False),
        ("TCN", TCNet, False),
        ("DepthCRNN", DepthCRNNet, False),
    ]

    def fvw(name):
        cands = [VIS_DIR / f"{name}_finetuned.pth", VIS_DIR / f"{name}.pth"]
        if name == "CAST-Net":
            cands.insert(0, VIS_W)
        for c in cands:
            if c.exists():
                return c
        return None

    vis_probas = {}
    cropper = None
    fd = {gid: test_data[gid]["vis_frames"] for gid in test_data if test_data[gid]["vis_frames"]}

    for vn, VC, uy in vm:
        vp = fvw(vn)
        if vp is None:
            print(f"{vn}: weights not found")
            continue

        print(f"Loading {vn} from {vp.name}")
        m = VC(NC).to(DEV)
        sd = torch.load(vp, map_location=DEV)

        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        elif isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]

        if not smart_load(m, sd):
            print(f"{vn}: load failed")
            del m
            continue

        m.eval()
        if uy and cropper is None:
            cropper = HandCropper()

        vis_probas[vn] = vis_probas_batch(m, fd, cropper if uy else None, SEQ)
        print(f"{vn}: {sum(len(v) for v in vis_probas[vn].values())} vectors")

        del m
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if cropper is not None:
        del cropper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n{'=' * 60}\n[2] EMG model probabilities\n{'=' * 60}")

    emg_probas = {}
    print(f"Loading SC-MSFE from {EMG_MD.name}")
    emg_probas["SC-MSFE"] = {}

    for gid in tqdm(range(NC), desc="SC-MSFE", leave=False):
        if gid not in test_data or not test_data[gid]["emg_windows"]:
            continue
        emg_probas["SC-MSFE"][gid] = emg_probas_ours(test_data[gid]["emg_windows"])

    n_sc = sum(len(v) for v in emg_probas["SC-MSFE"].values())
    print(f"SC-MSFE: {n_sc} vectors")

    if EMG_CMP.exists():
        for mp in sorted(EMG_CMP.glob("*.pkl")):
            if "scaler" in mp.name.lower():
                continue

            rn = mp.stem
            en = re.sub(r"_?Paper\d*", "", rn)
            en = re.sub(r"_?BVNet", "", en)
            en = en.strip("_").strip()
            if not en:
                en = rn

            if en in emg_probas:
                continue

            print(f"Loading {en} from {mp.name}")
            emg_probas[en] = {}

            for gid in tqdm(range(NC), desc=en, leave=False):
                if gid not in test_data or not test_data[gid]["emg_windows"]:
                    continue
                pr = emg_probas_comp(mp, test_data[gid]["emg_windows"])
                if pr:
                    emg_probas[en][gid] = pr

            n = sum(len(v) for v in emg_probas[en].values())
            if n == 0:
                print(f"{en}: failed")
                del emg_probas[en]
            else:
                print(f"{en}: {n} vectors")

    print(f"\n{'=' * 60}\n[3] Simple probability addition\n{'=' * 60}")
    simple_results, cm_data = simple_fusion_eval(vis_probas, emg_probas)

    print(f"\n{'Visual':<14} {'EMG':<20} {'Acc':>7} {'F1':>7} {'Prec':>7} {'Recall':>7} {'N':>6}")
    print("-" * 78)
    for (vn, en), r in sorted(simple_results.items(), key=lambda x: -x[1]["accuracy"]):
        flag = " *" if (vn == "CAST-Net" and en == "SC-MSFE") else ""
        print(
            f"{vn:<14} {en:<20} {r['accuracy'] * 100:>6.1f}% "
            f"{r['f1'] * 100:>6.1f}% {r['precision'] * 100:>6.1f}% "
            f"{r['recall'] * 100:>6.1f}% {r['n']:>5}{flag}"
        )

    print(f"\n{'=' * 60}\n[4] Non-learnable decision-level fusion methods\n{'=' * 60}")
    decision_results, decision_cm, decision_details = non_learnable_decision_fusion_eval(vis_probas, emg_probas)

    print(f"\n{'Visual':<14} {'EMG':<20} {'Method':<20} {'Acc':>7} {'F1':>7} {'Prec':>7} {'Recall':>7} {'N':>6}")
    print("-" * 108)
    for (vn, en, method), r in sorted(decision_results.items(), key=lambda x: -x[1]["accuracy"]):
        flag = " *" if (vn == "CAST-Net" and en == "SC-MSFE") else ""
        print(
            f"{vn:<14} {en:<20} {method:<20} {r['accuracy'] * 100:>6.1f}% "
            f"{r['f1'] * 100:>6.1f}% {r['precision'] * 100:>6.1f}% "
            f"{r['recall'] * 100:>6.1f}% {r['n']:>5}{flag}"
        )

    print(f"\n{'=' * 60}\n[5] Cross-validated learnable fusion methods\n{'=' * 60}")
    learnable_results, learnable_cm, learnable_details = learnable_decision_fusion_eval(vis_probas, emg_probas)
    ranked_learnable, excluded_learnable = filter_main_ranking(learnable_results, learnable_details)

    print(f"\n{'Visual':<14} {'EMG':<20} {'Method':<22} {'Acc':>7} {'F1':>7} {'Prec':>7} {'Recall':>7} {'N':>6} {'Status':>12}")
    print("-" * 124)
    for (vn, en, method), r in sorted(learnable_results.items(), key=lambda x: -x[1]["accuracy"]):
        status = learnable_details[(vn, en, method)].get("ranking_status", "eligible")
        flag = " *" if (vn == "CAST-Net" and en == "SC-MSFE") else ""
        label = "exploratory" if status != "eligible" else "eligible"
        print(
            f"{vn:<14} {en:<20} {method:<22} {r['accuracy'] * 100:>6.1f}% "
            f"{r['f1'] * 100:>6.1f}% {r['precision'] * 100:>6.1f}% "
            f"{r['recall'] * 100:>6.1f}% {r['n']:>5} {label:>12}{flag}"
        )

    if excluded_learnable:
        print("\nExploratory learnable results are saved but excluded from the main ranking when they meet or exceed the SOLID-Net reference.")

    print(f"\n{'=' * 60}\n[6] Generating visualizations\n{'=' * 60}")
    vo = [v for v in ["CAST-Net", "Baseline", "ResNet-LSTM", "ConvGRU-Attn", "TCN", "DepthCRNN"] if v in vis_probas]
    eo = ["SC-MSFE"] + sorted([e for e in emg_probas if e != "SC-MSFE"])

    render_table(simple_results, SOLID_RESULT, vo, eo, OUT_DIR)
    render_cm(cm_data, vo, eo, simple_results, OUT_DIR)
    render_non_learnable_bars(decision_results, OUT_DIR, top_k=30)
    render_learnable_bars(learnable_results, OUT_DIR, top_k=30)
    render_method_heatmap(decision_results, OUT_DIR, "CAST-Net", "SC-MSFE")
    render_decision_cm(decision_cm, decision_results, OUT_DIR, max_items=12)
    render_decision_cm(learnable_cm, learnable_results, OUT_DIR / "learnable", max_items=12)

    summary = {
        "protocol": {
            "mode": "strict_leakage_free_cross_validated_fusion",
            "test_dir": str(TEST_DIR),
            "initialize_dir": str(INIT_DIR),
            "learnable_fusion_enabled": True,
            "learnable_fit_scope": "stratified train folds only",
            "main_ranking_policy": "learnable methods matching or exceeding SOLID-Net are reported as exploratory and excluded from the main ranking",
        },
        "simple_fusion": {},
        "decision_fusion_non_learnable": {},
        "decision_fusion_learnable": {},
        "decision_fusion_learnable_main_ranking": {},
        "decision_fusion_learnable_exploratory": {},
        "decision_fusion_details": {},
        "learnable_fusion_details": {},
        "solid_net_reference": SOLID_RESULT,
    }

    for (vn, en), r in simple_results.items():
        summary["simple_fusion"][f"{vn}+{en}"] = r

    for (vn, en, method), r in decision_results.items():
        summary["decision_fusion_non_learnable"][f"{vn}+{en}+{method}"] = r

    for (vn, en, method), r in learnable_results.items():
        summary["decision_fusion_learnable"][f"{vn}+{en}+{method}"] = r

    for (vn, en, method), r in ranked_learnable.items():
        summary["decision_fusion_learnable_main_ranking"][f"{vn}+{en}+{method}"] = r

    for (vn, en, method), r in excluded_learnable.items():
        summary["decision_fusion_learnable_exploratory"][f"{vn}+{en}+{method}"] = r

    for (vn, en, method), d in decision_details.items():
        summary["decision_fusion_details"][f"{vn}+{en}+{method}"] = d

    for (vn, en, method), d in learnable_details.items():
        summary["learnable_fusion_details"][f"{vn}+{en}+{method}"] = d

    with open(OUT_DIR / "fusion_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 72}\nFINAL RANKING\n{'=' * 72}")
    ar = {}

    for (vn, en), r in simple_results.items():
        ar[f"{vn}+{en}+add"] = r

    for (vn, en, method), r in decision_results.items():
        ar[f"{vn}+{en}+{method}"] = r

    for (vn, en, method), r in ranked_learnable.items():
        ar[f"{vn}+{en}+{method}"] = r

    ar["SOLID-Net"] = SOLID_RESULT

    for rank, (name, r) in enumerate(sorted(ar.items(), key=lambda x: -x[1]["accuracy"]), 1):
        flag = " *" if ("SOLID" in name or ("CAST-Net" in name and "SC-MSFE" in name)) else ""
        print(
            f"#{rank:02d} {name:<55} "
            f"Acc={r['accuracy'] * 100:.1f}%  "
            f"F1={r['f1'] * 100:.1f}%{flag}"
        )

    print(f"\nOutput: {OUT_DIR}")
    print("Protocol: leakage-aware cross-validated evaluation for learnable fusion")


if __name__ == "__main__":
    main()

