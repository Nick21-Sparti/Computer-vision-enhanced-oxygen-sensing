# Repository edition: see docs/CHANGES.md and docs/REPRODUCIBILITY.md.
# -*- coding: utf-8 -*-
"""
==============================================================================
PROJECT: PINN-IDC (Inverse Dynamic Compensation) Framework
VERSION: 11.1.0-Local-Classical - Fixed-Epoch Benchmark Suite, V14 T90
==============================================================================

--- SCIENTIFIC CONTEXT & GOAL ---
The fundamental challenge is the "Response Time Problem" in chemical sensing.
Diffusion creates a physical lag where the optode/film state trails the true bulk
concentration.

This script benchmarks the local dynamic classical models against the lag:
   1. Two-site Stern-Volmer steady-state calibration (SV)
   2. First-order causal inverse model with experimentally fitted time constant tau
   3. Regularized frequency-domain inverse/deconvolution (offline Tikhonov/Wiener)
   4. Causal two-state Kalman filter for bulk + surface oxygen

(Neural baseline architectures - CNN-LSTM, CNN-TCN, TViT - are retained in the
 codebase but disabled by default in this local classical-only iteration).

All dynamic parameters are fit ONLY on the outer training experiments.

--- EVOLUTION OF THE FRAMEWORK ---
- V11.0.0: Reviewer-proof benchmark suite with dynamic inverse, deconvolution,
  Kalman filter, CNN-LSTM, CNN-TCN, nested epoch selection, and resume caching.
- V11.1.0-Cluster: HPC deployment, fixed-epoch neural parity, V14 T90 imports.
- V11.1.0-Local-Classical (CURRENT):
    1. LOCAL DEPLOYMENT: Restored local Windows paths.
    2. CLASSICAL FOCUS: Neural network execution toggled off in `RUN_METHODS`
       to allow rapid testing and generation of classical physical baselines.
    3. WINDOWS-SAFE MULTIPROCESSING: Disabled DataLoader parallel workers
       (`NUM_WORKERS = 0`) to prevent local Windows spawn crashes.
==============================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
# Portable repository paths; edit scripts/repo_config.py for your computer.
from repo_config import data_path, output_path, cache_path, external_path

import os
import sys
import glob
import re
import json
import math
import time
import hashlib
import warnings
import random
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')  # Headless-safe for background execution
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit, minimize_scalar
from scipy.signal import savgol_filter
from scipy.stats import t as student_t

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

# ==============================================================================
# 1. REPRODUCIBILITY & PERFORMANCE CONFIGURATION
# ==============================================================================
SEED = 42

def set_global_seed(seed: int) -> None:
    """Set all stochastic generators used by this script."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_global_seed(SEED)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

# ==============================================================================
# 2. USER CONFIGURATION & PATHS (LOCAL CONFIGURATION)
# ==============================================================================
BASE_PROJECT_DIR = data_path('aqueous')
EXPERIMENTS_ROOT_DIR = BASE_PROJECT_DIR

# Fail-fast check for local mount availability
if not os.path.exists(BASE_PROJECT_DIR):
    print(f"CRITICAL ERROR: Root project directory not found: {BASE_PROJECT_DIR}")
    print("Please verify your local storage path or external drive.")
    sys.exit(1)

# Known equilibrium/calibration windows used by the existing framework.
ANALYSIS_INTERVALS_S = [
    (1450, 1650),
    (2300, 2500),
    (3100, 3300),
    (3900, 4100),
    (4700, 4900),
]

UNSTRUCTURED_CUTOFF_S = 1000
N_EXPERIMENTS_CAP = None

# Image / model configuration.
N_STACKED_FRAMES = 3
FRAME_STACK_STEP = 10
RESIZE_DIM = (224, 224)
STORAGE_DIM = (48, 48)
BATCH_SIZE = 32

# --- LOCAL RESOURCE MANAGEMENT ---
# 0 is recommended for Windows to avoid multiprocessing spawn crashes.
NUM_WORKERS = 0

# --- FIXED-EPOCH TRAINING BUDGET (V10.9 PARITY) ---
N_EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# First-order dynamic inversion tuning.
TAU_MIN_S = 0.05
TAU_MAX_S = 1000.0
DERIVATIVE_EWM_ALPHA_GRID = [0.10, 0.20, 0.35, 0.50, 0.75, 1.00]

# Regularized deconvolution tuning.
REG_DECONV_LAMBDAS = np.logspace(-8, 2, 15)

# Kalman tuning grids.
KALMAN_Q_BULK_FACTORS = [0.01, 0.03, 0.10, 0.30, 1.0, 3.0, 10.0]
KALMAN_Q_SURFACE_FACTORS = [0.001, 0.01, 0.10, 1.0]
KALMAN_R_FACTORS = [0.5, 1.0, 2.0]

CLIP_DYNAMIC_PREDICTIONS = False

# --- METHOD TOGGLES (CLASSICAL ONLY FOR LOCAL RUN) ---
RUN_METHODS = {
    "SV": True,
    "FirstOrderInverse": True,
    "RegularizedDeconv": True,
    "CausalKalman": True,
    "CNNLSTM": False,  # DISABLED
    "CNNTCN": False,   # DISABLED
    "TViT": False,     # DISABLED
}

# --- V14 T90 METHODOLOGY CONSTANTS ---
T90_MIN_STEP_UMOL = 5.0
T90_SAVGOL_WINDOW = 11
T90_SAVGOL_POLYORDER = 2
T90_SAVGOL_MIN_SAMPLES = 20
T90_RAW_ALIGNED_MIN_PLATEAU = 3

# --- OPTIONAL IMPORT OF PRE-COMPUTED V14 T90 ROWS ---
V14_T90_IMPORT_ENABLED = True
V14_T90_CSV_DIR = external_path("aqueous_t90_reports")
V14_T90_RAW_ALIGNED_FILE = "Extracted_T90_RawAligned_Metrics.csv"
V14_T90_SAVGOL_FILE = "Extracted_T90_Raw_Metrics.csv"
V14_IMPORT_BASE_KEY = "o2_static_vit_10"
V14_IMPORT_METHOD_LABEL = "TViT_V10.9_legacy"

OXYGEN_UNIT = "umol/L"

# Output / cache directories.
OUTPUT_DIR = output_path('aqueous_dynamic_benchmarks')
CSV_EXPORT_DIR = os.path.join(OUTPUT_DIR, "csv_exports")
FEATURE_CACHE_DIR = cache_path('aqueous_dynamic_benchmarks', "features")
EXPERIMENT_FEATURE_CACHE_DIR = os.path.join(FEATURE_CACHE_DIR, "per_experiment")
CV_CACHE_DIR = cache_path('aqueous_dynamic_benchmarks', "folds")

SUBDIRS = {
    "preflight": os.path.join(OUTPUT_DIR, "0_Preflight_Diagnostics"),
    "training": os.path.join(OUTPUT_DIR, "1_Training_Convergence"),
    "inference": os.path.join(OUTPUT_DIR, "2_Inference_Results"),
    "chemometrics": os.path.join(OUTPUT_DIR, "3_Chemometrics"),
    "t90": os.path.join(OUTPUT_DIR, "4_Response_Time_Reports"),
    "params": os.path.join(OUTPUT_DIR, "5_Method_Parameters"),
    "logs": os.path.join(OUTPUT_DIR, "6_Run_Logs"),
}

for d in [
    OUTPUT_DIR,
    CSV_EXPORT_DIR,
    FEATURE_CACHE_DIR,
    EXPERIMENT_FEATURE_CACHE_DIR,
    CV_CACHE_DIR,
    *SUBDIRS.values(),
]:
    os.makedirs(d, exist_ok=True)

# ==============================================================================
# 3. CACHE VERSIONING, HASHING, AND ATOMIC FILE WRITES
# ==============================================================================
SCRIPT_VERSION = "11.1.0-Local-Classical"
CACHE_SCHEMA_VERSION = "IDC_V11_1_SCHEMA_LOCAL"


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def atomic_write_json(data: dict, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def atomic_to_csv(df: pd.DataFrame, path: str, index: bool = False) -> None:
    tmp = path + ".tmp"
    df.to_csv(tmp, index=index)
    os.replace(tmp, path)


def atomic_to_parquet(df: pd.DataFrame, path: str, index: bool = False) -> None:
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=index)
    os.replace(tmp, path)


def atomic_torch_save(payload: dict, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def safe_torch_load(path: str, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))


def build_config_payload() -> dict:
    return {
        "script_version": SCRIPT_VERSION,
        "cache_schema": CACHE_SCHEMA_VERSION,
        "analysis_intervals": ANALYSIS_INTERVALS_S,
        "unstructured_cutoff": UNSTRUCTURED_CUTOFF_S,
        "stacked_frames": N_STACKED_FRAMES,
        "frame_stack_step": FRAME_STACK_STEP,
        "resize_dim": RESIZE_DIM,
        "storage_dim": STORAGE_DIM,
        "batch_size": BATCH_SIZE,
        "epochs": N_EPOCHS,
        "training_protocol": "fixed_budget_final_epoch",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "tau_bounds": [TAU_MIN_S, TAU_MAX_S],
        "derivative_alpha_grid": DERIVATIVE_EWM_ALPHA_GRID,
        "reg_deconv_lambdas": REG_DECONV_LAMBDAS.tolist(),
        "kalman_q_bulk": KALMAN_Q_BULK_FACTORS,
        "kalman_q_surface": KALMAN_Q_SURFACE_FACTORS,
        "kalman_r": KALMAN_R_FACTORS,
        "seed": SEED,
    }


CONFIG_PAYLOAD = build_config_payload()
CONFIG_HASH = hashlib.sha256(
    json.dumps(CONFIG_PAYLOAD, sort_keys=True, default=_json_default).encode("utf-8")
).hexdigest()[:16]

atomic_write_json(
    {"config_hash": CONFIG_HASH, "config": CONFIG_PAYLOAD},
    os.path.join(OUTPUT_DIR, "Run_Configuration.json"),
)


def prediction_cache_paths(fold_dir: str, method: str) -> Tuple[str, str]:
    method_safe = safe_name(method)
    return (
        os.path.join(fold_dir, f"predictions_{method_safe}.parquet"),
        os.path.join(fold_dir, f"manifest_{method_safe}.json"),
    )


def fold_data_signature(test_df: pd.DataFrame) -> str:
    arr = np.column_stack([
        test_df["sample_id"].to_numpy(dtype=np.int64),
        np.round(test_df["oxygen"].to_numpy(dtype=float), 8),
        np.round(test_df["elapsed"].to_numpy(dtype=float), 8),
    ])
    return hashlib.sha256(arr.tobytes()).hexdigest()[:20]


def save_prediction_cache(
    fold_dir: str,
    method: str,
    test_df: pd.DataFrame,
    preds: np.ndarray,
    extra_manifest: Optional[dict] = None,
) -> None:
    pred_path, manifest_path = prediction_cache_paths(fold_dir, method)
    cache_df = pd.DataFrame({
        "sample_id": test_df["sample_id"].to_numpy(),
        "prediction": np.asarray(preds, dtype=float),
    })
    atomic_to_parquet(cache_df, pred_path)
    manifest = {
        "script_version": SCRIPT_VERSION,
        "config_hash": CONFIG_HASH,
        "fold_signature": fold_data_signature(test_df),
        "method": method,
        "n_samples": len(test_df),
        "completed_at": datetime.now().isoformat(),
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    atomic_write_json(manifest, manifest_path)


def load_prediction_cache(
    fold_dir: str,
    method: str,
    test_df: pd.DataFrame,
) -> Optional[np.ndarray]:
    pred_path, manifest_path = prediction_cache_paths(fold_dir, method)
    if not (os.path.exists(pred_path) and os.path.exists(manifest_path)):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("config_hash") != CONFIG_HASH:
            return None
        if manifest.get("fold_signature") != fold_data_signature(test_df):
            return None
        cached = pd.read_parquet(pred_path)
        if len(cached) != len(test_df):
            return None
        merged = test_df[["sample_id"]].merge(cached, on="sample_id", how="left")
        if merged["prediction"].isna().any():
            return None
        return merged["prediction"].to_numpy(dtype=float)
    except Exception:
        return None

# ==============================================================================
# 4. ROBUST DATA DISCOVERY & PARSING
# ==============================================================================
def find_experiment_files_v11(root_dir: str) -> List[dict]:
    print("--- Discovery: Searching for Experiment Directories ---")
    all_dirs = [d for d in glob.glob(os.path.join(root_dir, "*-*")) if os.path.isdir(d)]
    all_dirs.sort()
    valid_exps = []

    for exp_dir in tqdm(all_dirs, desc="Scanning for Valid Data"):
        exp_id = os.path.basename(exp_dir)
        roi_folders = [
            d for d in glob.glob(os.path.join(exp_dir, "*"))
            if os.path.isdir(d) and d.endswith("_ROI_Output")
        ]
        arduino_logs = glob.glob(
            os.path.join(exp_dir, "**", "*_arduino_*.txt"), recursive=True
        )
        temp_logs = glob.glob(
            os.path.join(exp_dir, "**", "*temperature*.csv"), recursive=True
        )

        if roi_folders and arduino_logs and temp_logs:
            vids = glob.glob(os.path.join(roi_folders[0], "*.mp4"))
            if vids:
                valid_exps.append({
                    "id": exp_id,
                    "video_path": vids[0],
                    "arduino_path": arduino_logs[0],
                    "temp_path": temp_logs[0],
                })

    print(f"  > Found {len(valid_exps)} valid experiments.")
    return valid_exps


def parse_logs(exp):
    o2_data = []
    try:
        with open(exp["arduino_path"], "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "MEA" not in line:
                    continue
                p = line.split()
                try:
                    ts = datetime.strptime(f"{p[0]} {p[1]}", "%Y-%m-%d %H:%M:%S")
                    o2_val = float(p[p.index("MEA") + 5]) / 1000.0
                    o2_data.append([ts, o2_val])
                except Exception:
                    continue
    except Exception:
        pass

    df_o2 = pd.DataFrame(o2_data, columns=["timestamp", "oxygen"])
    if df_o2.empty:
        return df_o2
    df_o2 = df_o2.sort_values("timestamp")
    df_o2["oxygen"] = pd.to_numeric(df_o2["oxygen"], errors="coerce")

    try:
        df_temp = pd.read_csv(exp["temp_path"]).iloc[:, :2].copy()
        df_temp.columns = ["timestamp", "temp"]
        df_temp["timestamp"] = pd.to_datetime(df_temp["timestamp"], errors="coerce")
        df_temp["temp"] = pd.to_numeric(df_temp["temp"], errors="coerce")
        df_temp = df_temp.dropna(subset=["timestamp"]).sort_values("timestamp")
    except Exception:
        df_temp = pd.DataFrame({
            "timestamp": df_o2["timestamp"],
            "temp": 20.0,
        })

    df = pd.merge_asof(
        df_o2,
        df_temp,
        on="timestamp",
        direction="nearest",
    )
    df["oxygen"] = df["oxygen"].interpolate().bfill().ffill()
    df["temp"] = df["temp"].interpolate().bfill().ffill().fillna(20.0)
    return df


def process_single_experiment(exp: dict, cache_path: str) -> pd.DataFrame:
    if os.path.exists(cache_path):
        try:
            cached = pd.read_parquet(cache_path)
            if not cached.empty:
                return cached
        except Exception:
            print(f"  ! Corrupt feature cache detected, rebuilding: {cache_path}")

    df_logs = parse_logs(exp)
    if df_logs.empty:
        return pd.DataFrame()

    cap = cv2.VideoCapture(exp["video_path"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid FPS for {exp['video_path']}: {fps}")

    match = re.search(r"_(\d{8}_\d{6})", os.path.basename(exp["video_path"]))
    start_ts = (
        datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
        if match else df_logs["timestamp"].iloc[0].to_pydatetime()
    )

    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = int(max(1, round(fps / 2.0)))
    rows = []

    iterator = range(0, total_f, step)
    for f_idx in tqdm(iterator, desc=f"Frames {exp['id']}", leave=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            break

        ts = start_ts + timedelta(seconds=f_idx / fps)

        idx = df_logs["timestamp"].searchsorted(pd.Timestamp(ts))
        if idx <= 0:
            nearest_idx = 0
        elif idx >= len(df_logs):
            nearest_idx = len(df_logs) - 1
        else:
            prev_dt = abs((df_logs.iloc[idx - 1]["timestamp"] - pd.Timestamp(ts)).total_seconds())
            next_dt = abs((df_logs.iloc[idx]["timestamp"] - pd.Timestamp(ts)).total_seconds())
            nearest_idx = idx - 1 if prev_dt <= next_dt else idx

        resized = cv2.resize(frame, STORAGE_DIM)
        red = resized[:, :, 2].astype(float)
        intensity = np.mean(red) + 1e-6

        rows.append({
            "experiment_id": exp["id"],
            "timestamp": ts,
            "elapsed": f_idx / fps,
            "oxygen": float(df_logs.iloc[nearest_idx]["oxygen"]),
            "temp": float(df_logs.iloc[nearest_idx]["temp"]),
            "inv_i": 1.0 / intensity,
            "inv_std": 1.0 / (np.std(red) + 1e-6),
            "pixels": resized.flatten().astype(np.uint8),
        })

    cap.release()
    exp_df = pd.DataFrame(rows)
    if not exp_df.empty:
        atomic_to_parquet(exp_df, cache_path)
    return exp_df


def get_master_df(experiments: List[dict]) -> pd.DataFrame:
    master_path = os.path.join(FEATURE_CACHE_DIR, "idc_master_v11.parquet")

    all_frames = []
    for exp in tqdm(experiments, desc="Loading/Building Experiment Feature Caches"):
        cache_path = os.path.join(
            EXPERIMENT_FEATURE_CACHE_DIR,
            f"features_{safe_name(exp['id'])}.parquet",
        )
        exp_df = process_single_experiment(exp, cache_path)
        if not exp_df.empty:
            all_frames.append(exp_df)

    if not all_frames:
        raise RuntimeError("No valid frames were extracted from the experiments.")

    master_df = pd.concat(all_frames, ignore_index=True)
    master_df = master_df.sort_values(["experiment_id", "elapsed"]).reset_index(drop=True)
    master_df["sample_id"] = np.arange(len(master_df), dtype=np.int64)
    atomic_to_parquet(master_df, master_path)
    return master_df

# ==============================================================================
# 5. DATASET AND NEURAL ARCHITECTURES (RETAINED FOR STRUCTURE)
# ==============================================================================
class IDCImageTemporalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, stack_size: int, step: int):
        self.df = df.sort_values(["experiment_id", "elapsed"]).reset_index(drop=True).copy()
        self.stack_size = int(stack_size)
        self.step = int(step)
        self.resize = transforms.Resize(RESIZE_DIM)

        self.exp_ids = self.df["experiment_id"].astype(str).to_numpy()
        self.targets = self.df["oxygen"].to_numpy(dtype=np.float32)
        self.sample_ids = self.df["sample_id"].to_numpy(dtype=np.int64)
        self.pixels = self.df["pixels"].to_numpy()

        self.exp_start = np.zeros(len(self.df), dtype=np.int64)
        start = 0
        for i in range(len(self.df)):
            if i == 0 or self.exp_ids[i] != self.exp_ids[i - 1]:
                start = i
            self.exp_start[i] = start

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        frames = []
        start = int(self.exp_start[idx])

        for j in range(self.stack_size - 1, -1, -1):
            lookback = max(start, idx - j * self.step)
            pix = self.pixels[lookback].reshape(STORAGE_DIM[0], STORAGE_DIM[1], 3)
            img_t = torch.from_numpy(pix).permute(2, 0, 1).float() / 255.0
            frames.append(self.resize(img_t))

        x = torch.cat(frames, dim=0)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        sid = torch.tensor(self.sample_ids[idx], dtype=torch.long)
        return x, y, sid


class TViTDecoupler(nn.Module):
    def __init__(self, stack_size: int = 3):
        super().__init__()
        self.stack_size = stack_size
        self.vit = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0)

        orig_proj = self.vit.patch_embed.proj
        self.vit.patch_embed.proj = nn.Conv2d(
            3 * stack_size,
            orig_proj.out_channels,
            kernel_size=orig_proj.kernel_size,
            stride=orig_proj.stride,
            padding=orig_proj.padding,
        )

        with torch.no_grad():
            repeated = orig_proj.weight.repeat(1, stack_size, 1, 1) / stack_size
            self.vit.patch_embed.proj.weight.copy_(repeated)
            if orig_proj.bias is not None:
                self.vit.patch_embed.proj.bias.copy_(orig_proj.bias)

        self.head = nn.Sequential(
            nn.Linear(self.vit.embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        features = self.vit.forward_features(x)[:, 0]
        return self.head(features).squeeze(-1)


class CNNLSTMDecoupler(nn.Module):
    def __init__(self, stack_size: int = 3, hidden_size: int = 128):
        super().__init__()
        self.stack_size = stack_size
        self.encoder = timm.create_model("resnet18", pretrained=True, num_classes=0)
        feat_dim = self.encoder.num_features
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        seq = x.view(b, self.stack_size, 3, h, w)
        enc = self.encoder(seq.reshape(b * self.stack_size, 3, h, w))
        enc = enc.view(b, self.stack_size, -1)
        out, _ = self.lstm(enc)
        return self.head(out[:, -1, :]).squeeze(-1)


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size=kernel_size, dilation=dilation, padding=0
        )

    def forward(self, x):
        x = nn.functional.pad(x, (self.left_padding, 0))
        return self.conv(x)


class CNNTCNDecoupler(nn.Module):
    def __init__(self, stack_size: int = 3, hidden_size: int = 128):
        super().__init__()
        self.stack_size = stack_size
        self.encoder = timm.create_model("resnet18", pretrained=True, num_classes=0)
        feat_dim = self.encoder.num_features

        self.temporal = nn.Sequential(
            CausalConv1d(feat_dim, hidden_size, kernel_size=2, dilation=1),
            nn.GELU(),
            CausalConv1d(hidden_size, hidden_size, kernel_size=2, dilation=2),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        seq = x.view(b, self.stack_size, 3, h, w)
        enc = self.encoder(seq.reshape(b * self.stack_size, 3, h, w))
        enc = enc.view(b, self.stack_size, -1).transpose(1, 2)
        z = self.temporal(enc)
        return self.head(z[:, :, -1]).squeeze(-1)


def make_neural_model(method: str) -> nn.Module:
    if method == "TViT":
        return TViTDecoupler(N_STACKED_FRAMES)
    if method == "CNNLSTM":
        return CNNLSTMDecoupler(N_STACKED_FRAMES)
    if method == "CNNTCN":
        return CNNTCNDecoupler(N_STACKED_FRAMES)
    raise ValueError(f"Unknown neural method: {method}")

# ==============================================================================
# 6. STEADY-STATE TWO-SITE STERN-VOLMER CALIBRATION
# ==============================================================================
def stable_mask(df: pd.DataFrame) -> np.ndarray:
    mask = np.zeros(len(df), dtype=bool)
    elapsed = df["elapsed"].to_numpy(dtype=float)
    for start, end in ANALYSIS_INTERVALS_S:
        mask |= (elapsed >= start) & (elapsed <= end)
    return mask


def two_site_stern_volmer_forward(o2, I0, f, K1, K2):
    return I0 * (
        f / (1.0 + K1 * o2)
        + (1.0 - f) / (1.0 + K2 * o2)
    )


def fit_steady_calibration(train_df: pd.DataFrame) -> dict:
    mask = stable_mask(train_df)
    stable_train = train_df.loc[mask].copy()
    if stable_train.empty:
        stable_train = train_df.copy()

    intensity = 1.0 / stable_train["inv_i"].to_numpy(dtype=float)
    oxygen = stable_train["oxygen"].to_numpy(dtype=float)

    valid = np.isfinite(intensity) & np.isfinite(oxygen) & (intensity > 0)
    intensity = intensity[valid]
    oxygen = oxygen[valid]

    if len(oxygen) < 20:
        raise RuntimeError("Insufficient stable training data for calibration.")

    I0_guess = float(np.percentile(intensity, 99))
    p0_grid = [
        [I0_guess, 0.80, 0.05, 0.005],
        [I0_guess, 0.50, 0.02, 0.002],
        [I0_guess, 0.20, 0.10, 0.005],
        [I0_guess, 0.70, 0.01, 0.0005],
        [I0_guess, 0.30, 0.005, 0.05],
    ]
    bounds = ([1e-9, 0.0, 0.0, 0.0], [np.inf, 1.0, np.inf, np.inf])

    best = None
    for p0 in p0_grid:
        try:
            popt, pcov = curve_fit(
                two_site_stern_volmer_forward,
                oxygen,
                intensity,
                p0=p0,
                bounds=bounds,
                maxfev=50000,
            )
            fitted = two_site_stern_volmer_forward(oxygen, *popt)
            sse = float(np.sum((fitted - intensity) ** 2))
            if best is None or sse < best["sse"]:
                best = {"popt": popt, "pcov": pcov, "sse": sse}
        except Exception:
            continue

    if best is not None:
        I0_fit, f_fit, K1_fit, K2_fit = [float(x) for x in best["popt"]]
        return {
            "model_type": "two_site_sv",
            "I0": I0_fit,
            "f": f_fit,
            "K1": K1_fit,
            "K2": K2_fit,
            "training_sse": float(best["sse"]),
            "n_calibration_points": int(len(oxygen)),
        }

    x = (1.0 / intensity).reshape(-1, 1)
    reg = LinearRegression().fit(x, oxygen)
    return {
        "model_type": "linear_inverse_intensity_fallback",
        "coef": float(reg.coef_[0]),
        "intercept": float(reg.intercept_),
        "n_calibration_points": int(len(oxygen)),
    }


def apply_steady_calibration(calib: dict, df: pd.DataFrame) -> np.ndarray:
    if calib["model_type"] == "linear_inverse_intensity_fallback":
        return calib["coef"] * df["inv_i"].to_numpy(dtype=float) + calib["intercept"]

    I0 = calib["I0"]
    f = calib["f"]
    K1 = calib["K1"]
    K2 = calib["K2"]

    I = 1.0 / df["inv_i"].to_numpy(dtype=float)
    y = I / I0

    A = y * K1 * K2
    B = y * (K1 + K2) - (f * (K2 - K1) + K1)
    C = y - 1.0

    delta = np.maximum(B ** 2 - 4.0 * A * C, 0.0)
    sqrt_delta = np.sqrt(delta)

    denom = 2.0 * A
    root1 = (-B - sqrt_delta) / np.where(np.abs(denom) < 1e-12, np.nan, denom)
    root2 = (-B + sqrt_delta) / np.where(np.abs(denom) < 1e-12, np.nan, denom)

    pred = np.where(
        np.isfinite(root1) & (root1 >= 0),
        root1,
        root2,
    )
    linear_root = -C / np.where(np.abs(B) < 1e-12, np.nan, B)
    pred = np.where(np.isfinite(pred), pred, linear_root)

    return pred.astype(float)

# ==============================================================================
# 7. DYNAMIC CLASSICAL BASELINES
# ==============================================================================
def causal_ewm(values: np.ndarray, alpha: float) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return x.copy()
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def backward_derivative(values: np.ndarray, elapsed: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    t = np.asarray(elapsed, dtype=float)
    d = np.zeros_like(x)
    if len(x) <= 1:
        return d
    dt = np.diff(t)
    dx = np.diff(x)
    valid_dt = np.where(np.abs(dt) < 1e-9, np.nan, dt)
    d[1:] = dx / valid_dt
    d[~np.isfinite(d)] = 0.0
    return d


def sanitize_prediction_series(pred: np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(pred, dtype=float))
    s = s.replace([np.inf, -np.inf], np.nan).interpolate().bfill().ffill()
    return s.to_numpy(dtype=float)


def clip_if_enabled(pred: np.ndarray, train_df: pd.DataFrame) -> np.ndarray:
    if not CLIP_DYNAMIC_PREDICTIONS:
        return pred
    lo = float(train_df["oxygen"].min())
    hi = float(train_df["oxygen"].max())
    return np.clip(pred, lo, hi)


def first_order_inverse_single(
    surface_o2: np.ndarray,
    elapsed: np.ndarray,
    tau_s: float,
    ewm_alpha: float,
) -> np.ndarray:
    s = sanitize_prediction_series(surface_o2)
    s_sm = causal_ewm(s, ewm_alpha)
    dsdt = backward_derivative(s_sm, elapsed)
    return s_sm + tau_s * dsdt


def group_apply_first_order(
    df: pd.DataFrame,
    surface_col: str,
    tau_s: float,
    ewm_alpha: float,
) -> np.ndarray:
    out = np.full(len(df), np.nan, dtype=float)
    df_reset = df.reset_index(drop=True)
    for _, g in df_reset.groupby("experiment_id", sort=False):
        idx = g.index.to_numpy()
        out[idx] = first_order_inverse_single(
            g[surface_col].to_numpy(dtype=float),
            g["elapsed"].to_numpy(dtype=float),
            tau_s=tau_s,
            ewm_alpha=ewm_alpha,
        )
    return sanitize_prediction_series(out)


def fit_first_order_parameters(train_df: pd.DataFrame) -> dict:
    best = None

    for alpha in DERIVATIVE_EWM_ALPHA_GRID:
        exp_cache = []
        for exp_id, g in train_df.groupby("experiment_id", sort=False):
            surface = g["surface_o2"].to_numpy(dtype=float)
            elapsed = g["elapsed"].to_numpy(dtype=float)
            gt = g["oxygen"].to_numpy(dtype=float)
            s_sm = causal_ewm(sanitize_prediction_series(surface), alpha)
            dsdt = backward_derivative(s_sm, elapsed)
            exp_cache.append((gt, s_sm, dsdt))

        def objective(tau):
            maes = []
            for gt, s_sm, dsdt in exp_cache:
                pred = s_sm + tau * dsdt
                maes.append(mean_absolute_error(gt, pred))
            return float(np.median(maes))

        result = minimize_scalar(
            objective,
            bounds=(TAU_MIN_S, TAU_MAX_S),
            method="bounded",
            options={"xatol": 1e-3, "maxiter": 200},
        )
        candidate = {
            "tau_s": float(result.x),
            "ewm_alpha": float(alpha),
            "training_median_mae": float(result.fun),
            "optimizer_success": bool(result.success),
        }
        if best is None or candidate["training_median_mae"] < best["training_median_mae"]:
            best = candidate

    if best is None:
        raise RuntimeError("First-order parameter fitting failed.")
    return best


def regularized_frequency_deconv_single(
    surface_o2: np.ndarray,
    elapsed: np.ndarray,
    tau_s: float,
    lam: float,
) -> np.ndarray:
    s = sanitize_prediction_series(surface_o2)
    t = np.asarray(elapsed, dtype=float)
    n = len(s)
    if n < 4:
        return s.copy()

    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        dt = 1.0

    pad = int(min(max(round(0.10 * n), 16), 1024))
    if n <= 2:
        pad = 0
    s_pad = np.pad(s, (pad, pad), mode="reflect") if pad > 0 else s.copy()

    S = np.fft.rfft(s_pad)
    freq = np.fft.rfftfreq(len(s_pad), d=dt)
    omega = 2.0 * np.pi * freq

    H = 1.0 / (1.0 + 1j * omega * tau_s)
    omega_max = max(float(np.max(omega)), 1e-12)
    L2 = (omega / omega_max) ** 4

    G = np.conj(H) / (np.abs(H) ** 2 + lam * L2 + 1e-12)
    c_pad = np.fft.irfft(G * S, n=len(s_pad))

    if pad > 0:
        c = c_pad[pad:-pad]
    else:
        c = c_pad
    return np.asarray(c, dtype=float)


def group_apply_regularized_deconv(
    df: pd.DataFrame,
    surface_col: str,
    tau_s: float,
    lam: float,
) -> np.ndarray:
    out = np.full(len(df), np.nan, dtype=float)
    df_reset = df.reset_index(drop=True)
    for _, g in df_reset.groupby("experiment_id", sort=False):
        idx = g.index.to_numpy()
        out[idx] = regularized_frequency_deconv_single(
            g[surface_col].to_numpy(dtype=float),
            g["elapsed"].to_numpy(dtype=float),
            tau_s=tau_s,
            lam=lam,
        )
    return sanitize_prediction_series(out)


def fit_regularized_deconv_lambda(train_df: pd.DataFrame, tau_s: float) -> dict:
    best = None
    for lam in REG_DECONV_LAMBDAS:
        maes = []
        for _, g in train_df.groupby("experiment_id", sort=False):
            pred = regularized_frequency_deconv_single(
                g["surface_o2"].to_numpy(dtype=float),
                g["elapsed"].to_numpy(dtype=float),
                tau_s=tau_s,
                lam=float(lam),
            )
            maes.append(mean_absolute_error(g["oxygen"], pred))
        score = float(np.median(maes))
        if best is None or score < best["training_median_mae"]:
            best = {
                "lambda": float(lam),
                "tau_s": float(tau_s),
                "training_median_mae": score,
                "causal": False,
            }
    return best


def estimate_sensor_noise_variance(train_df: pd.DataFrame) -> float:
    variances = []
    mask = stable_mask(train_df)
    stable = train_df.loc[mask].copy()
    if stable.empty:
        stable = train_df.copy()

    for _, g in stable.groupby("experiment_id", sort=False):
        s = sanitize_prediction_series(g["surface_o2"].to_numpy(dtype=float))
        if len(s) > 3:
            d = np.diff(s)
            if len(d) > 1:
                var = np.nanvar(d, ddof=1) / 2.0
                if np.isfinite(var) and var > 0:
                    variances.append(var)

    if not variances:
        return 1.0
    return float(max(np.median(variances), 1e-8))


def causal_kalman_bulk_surface_single(
    surface_o2: np.ndarray,
    elapsed: np.ndarray,
    tau_s: float,
    q_bulk_factor: float,
    q_surface_factor: float,
    r_factor: float,
    base_noise_var: float,
) -> np.ndarray:
    z = sanitize_prediction_series(surface_o2)
    t = np.asarray(elapsed, dtype=float)
    n = len(z)
    if n == 0:
        return z.copy()

    x = np.array([z[0], z[0]], dtype=float)
    P = np.eye(2, dtype=float) * max(base_noise_var, 1e-6) * 10.0
    H = np.array([[0.0, 1.0]], dtype=float)
    I2 = np.eye(2, dtype=float)
    R = np.array([[max(base_noise_var * r_factor, 1e-10)]], dtype=float)

    out = np.empty(n, dtype=float)
    out[0] = x[0]

    dt_default = float(np.median(np.diff(t))) if n > 1 else 1.0
    if not np.isfinite(dt_default) or dt_default <= 0:
        dt_default = 1.0

    tau_s = max(float(tau_s), TAU_MIN_S)

    for k in range(1, n):
        dt = float(t[k] - t[k - 1])
        if not np.isfinite(dt) or dt <= 0:
            dt = dt_default

        a = math.exp(-dt / tau_s)
        F = np.array([
            [1.0, 0.0],
            [1.0 - a, a],
        ], dtype=float)

        scale = max(base_noise_var, 1e-10)
        Q = np.diag([
            q_bulk_factor * scale * max(dt, 1e-6),
            q_surface_factor * scale * max(dt, 1e-6),
        ])

        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        innovation = float(z[k] - (H @ x_pred)[0])
        S_cov = float((H @ P_pred @ H.T)[0, 0] + R[0, 0])
        K = (P_pred @ H.T) / max(S_cov, 1e-12)
        x = x_pred + K[:, 0] * innovation

        KH = K @ H
        P = (I2 - KH) @ P_pred @ (I2 - KH).T + K @ R @ K.T

        out[k] = x[0]

    return out


def fit_kalman_parameters(train_df: pd.DataFrame, tau_s: float) -> dict:
    base_noise_var = estimate_sensor_noise_variance(train_df)
    best = None

    for q_bulk in KALMAN_Q_BULK_FACTORS:
        for q_surface in KALMAN_Q_SURFACE_FACTORS:
            for r_factor in KALMAN_R_FACTORS:
                maes = []
                for _, g in train_df.groupby("experiment_id", sort=False):
                    pred = causal_kalman_bulk_surface_single(
                        g["surface_o2"].to_numpy(dtype=float),
                        g["elapsed"].to_numpy(dtype=float),
                        tau_s=tau_s,
                        q_bulk_factor=q_bulk,
                        q_surface_factor=q_surface,
                        r_factor=r_factor,
                        base_noise_var=base_noise_var,
                    )
                    maes.append(mean_absolute_error(g["oxygen"], pred))

                score = float(np.median(maes))
                if best is None or score < best["training_median_mae"]:
                    best = {
                        "tau_s": float(tau_s),
                        "q_bulk_factor": float(q_bulk),
                        "q_surface_factor": float(q_surface),
                        "r_factor": float(r_factor),
                        "base_noise_var": float(base_noise_var),
                        "training_median_mae": score,
                        "causal": True,
                    }
    return best


def group_apply_kalman(df: pd.DataFrame, params: dict) -> np.ndarray:
    out = np.full(len(df), np.nan, dtype=float)
    df_reset = df.reset_index(drop=True)
    for _, g in df_reset.groupby("experiment_id", sort=False):
        idx = g.index.to_numpy()
        out[idx] = causal_kalman_bulk_surface_single(
            g["surface_o2"].to_numpy(dtype=float),
            g["elapsed"].to_numpy(dtype=float),
            tau_s=params["tau_s"],
            q_bulk_factor=params["q_bulk_factor"],
            q_surface_factor=params["q_surface_factor"],
            r_factor=params["r_factor"],
            base_noise_var=params["base_noise_var"],
        )
    return sanitize_prediction_series(out)

# ==============================================================================
# 8. FIXED-EPOCH NEURAL TRAINING WITH PER-EPOCH RESUME
# ==============================================================================
def build_loader(df: pd.DataFrame, shuffle: bool) -> DataLoader:
    dataset = IDCImageTemporalDataset(
        df,
        stack_size=N_STACKED_FRAMES,
        step=FRAME_STACK_STEP,
    )
    dl_kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": shuffle,
        "num_workers": NUM_WORKERS,
        "pin_memory": (torch.cuda.is_available()),
    }
    if NUM_WORKERS > 0:
        dl_kwargs["persistent_workers"] = True
    return DataLoader(dataset, **dl_kwargs)


def infer_neural(model: nn.Module, df: pd.DataFrame) -> np.ndarray:
    loader = build_loader(df, shuffle=False)
    model.eval()
    pred_by_id = {}

    with torch.no_grad():
        for x, _, sid in loader:
            x = x.to(DEVICE, non_blocking=True)
            pred = model(x).detach().cpu().numpy().reshape(-1)
            for s, p in zip(sid.numpy().reshape(-1), pred):
                pred_by_id[int(s)] = float(p)

    preds = np.array([pred_by_id[int(s)] for s in df["sample_id"].to_numpy()], dtype=float)
    return preds


def _train_one_epoch(model, loader, optimizer, criterion) -> float:
    model.train()
    loss_sum = 0.0
    n_batches = 0

    for x, y, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        loss_sum += float(loss.item())
        n_batches += 1

    return loss_sum / max(n_batches, 1)


def evaluate_mae(model, df: pd.DataFrame) -> float:
    pred = infer_neural(model, df)
    return float(mean_absolute_error(df["oxygen"], pred))


def save_training_plot(history_df: pd.DataFrame, method: str, fold_id: str, stage: str) -> None:
    stem = f"{safe_name(method)}_{safe_name(fold_id)}_{safe_name(stage)}"
    csv_path = os.path.join(SUBDIRS["training"], f"History_{stem}.csv")
    png_path = os.path.join(SUBDIRS["training"], f"Convergence_{stem}.png")
    atomic_to_csv(history_df, csv_path, index=False)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(history_df["epoch"], history_df["train_loss"], label="Train MSE")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training MSE")

    if "val_mae" in history_df.columns and history_df["val_mae"].notna().any():
        ax2 = ax1.twinx()
        ax2.plot(history_df["epoch"], history_df["val_mae"], marker="o", label="Held-out MAE")
        ax2.set_ylabel(f"Held-out MAE ({OXYGEN_UNIT})")
        ax2.legend(loc="upper right")

    ax1.legend(loc="upper left")
    ax1.set_title(f"{method} - {fold_id} - {stage}")
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def train_fixed_epochs(
    method: str,
    outer_train_df: pd.DataFrame,
    fold_id: str,
    fold_index: int,
    method_cache_dir: str,
) -> nn.Module:
    complete_path = os.path.join(method_cache_dir, "fixed_epoch_complete.json")
    checkpoint_path = os.path.join(method_cache_dir, "fixed_epoch_checkpoint.pt")
    final_model_path = os.path.join(method_cache_dir, "final_model.pt")

    seed_offset = {"CNNLSTM": 1101, "CNNTCN": 1202, "TViT": 1303}[method]
    set_global_seed(SEED + fold_index * 1000 + seed_offset)

    model = make_neural_model(method).to(DEVICE)

    if os.path.exists(complete_path) and os.path.exists(final_model_path):
        ckpt = safe_torch_load(final_model_path, map_location=DEVICE)
        if (
            ckpt.get("config_hash") == CONFIG_HASH
            and int(ckpt.get("n_epochs", -1)) == int(N_EPOCHS)
        ):
            model.load_state_dict(ckpt["model_state"])
            print(f"    > {method}: loaded completed fixed-epoch model.")
            return model

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    train_loader = build_loader(outer_train_df, shuffle=True)

    history = []
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        ckpt = safe_torch_load(checkpoint_path, map_location=DEVICE)
        if (
            ckpt.get("config_hash") == CONFIG_HASH
            and int(ckpt.get("n_epochs", -1)) == int(N_EPOCHS)
        ):
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            history = ckpt.get("history", [])
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            print(f"    > Resuming {method} fixed-epoch training at epoch {start_epoch}/{N_EPOCHS}")

    for epoch in range(start_epoch, N_EPOCHS):
        train_loss = _train_one_epoch(model, train_loader, optimizer, criterion)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mae": np.nan,
        })

        atomic_torch_save(
            {
                "config_hash": CONFIG_HASH,
                "n_epochs": N_EPOCHS,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history": history,
            },
            checkpoint_path,
        )

        print(
            f"    > {method} epoch {epoch+1:02d}/{N_EPOCHS}: "
            f"train MSE={train_loss:.5f}"
        )

    history_df = pd.DataFrame(history)
    save_training_plot(history_df, method, fold_id, "FixedEpochTrain")

    atomic_torch_save(
        {
            "config_hash": CONFIG_HASH,
            "n_epochs": N_EPOCHS,
            "model_state": model.state_dict(),
        },
        final_model_path,
    )
    atomic_write_json(
        {
            "method": method,
            "n_epochs": N_EPOCHS,
            "protocol": "fixed_budget_final_epoch",
            "completed_at": datetime.now().isoformat(),
        },
        complete_path,
    )
    return model


def run_neural_method(
    method: str,
    outer_train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fold_id: str,
    fold_index: int,
    fold_dir: str,
) -> np.ndarray:
    cached = load_prediction_cache(fold_dir, method, test_df)
    if cached is not None:
        print(f"  > {method}: loaded cached predictions.")
        return cached

    method_cache_dir = os.path.join(fold_dir, f"neural_{safe_name(method)}")
    os.makedirs(method_cache_dir, exist_ok=True)

    model = train_fixed_epochs(
        method,
        outer_train_df,
        fold_id,
        fold_index,
        method_cache_dir=method_cache_dir,
    )

    preds = infer_neural(model, test_df)
    save_prediction_cache(
        fold_dir,
        method,
        test_df,
        preds,
        extra_manifest={
            "n_epochs": int(N_EPOCHS),
            "protocol": "fixed_budget_final_epoch",
        },
    )
    return preds

# ==============================================================================
# 9. METRICS, T90 (V14 METHODOLOGY), PLOTS, AND REPRODUCIBLE CSV EXPORTS
# ==============================================================================
def safe_r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 2 or np.nanstd(y_true[valid]) < 1e-12:
        return np.nan
    return float(r2_score(y_true[valid], y_pred[valid]))


def metric_block(y_true, y_pred, prefix: str) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() == 0:
        return {
            f"MAE_{prefix}": np.nan,
            f"RMSE_{prefix}": np.nan,
            f"R2_{prefix}": np.nan,
            f"N_{prefix}": 0,
        }
    yt = y_true[valid]
    yp = y_pred[valid]
    return {
        f"MAE_{prefix}": float(mean_absolute_error(yt, yp)),
        f"RMSE_{prefix}": float(np.sqrt(mean_squared_error(yt, yp))),
        f"R2_{prefix}": safe_r2(yt, yp),
        f"N_{prefix}": int(len(yt)),
    }


def regime_masks(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    elapsed = df["elapsed"].to_numpy(dtype=float)
    stable = stable_mask(df)
    unstructured = elapsed < UNSTRUCTURED_CUTOFF_S
    structured = ~unstructured
    transient = structured & (~stable)
    return {
        "Global": np.ones(len(df), dtype=bool),
        "Stable": stable,
        "Transient": transient,
        "Unstructured": unstructured,
    }


def compute_method_metrics(df: pd.DataFrame, pred_col: str, method: str, exp_id: str) -> dict:
    row = {"experiment_id": exp_id, "method": method}
    masks = regime_masks(df)
    for regime, mask in masks.items():
        row.update(metric_block(
            df.loc[mask, "oxygen"].to_numpy(dtype=float),
            df.loc[mask, pred_col].to_numpy(dtype=float),
            regime,
        ))
    return row

# ------------------------------------------------------------------------------
# 9a. V14 T90 EXTRACTOR (A): SAVGOL-BASED, SIGNED LAG
# ------------------------------------------------------------------------------
def compute_t90_events_savgol(
    df: pd.DataFrame,
    prediction_columns: Dict[str, str],
    exp_id: str,
) -> pd.DataFrame:
    d = df.sort_values("elapsed").reset_index(drop=True)
    records = []

    for i in range(len(ANALYSIS_INTERVALS_S)):
        target_s, target_e = ANALYSIS_INTERVALS_S[i]
        step_start_time = (
            ANALYSIS_INTERVALS_S[i - 1][1] if i > 0 else max(0, target_s - 200)
        )

        sub = d[(d["elapsed"] >= step_start_time) & (d["elapsed"] <= target_e)].copy()
        if len(sub) < T90_SAVGOL_MIN_SAMPLES:
            continue

        def get_plat_med(col):
            plat = d[(d["elapsed"] >= target_s) & (d["elapsed"] <= target_e)]
            if col not in plat.columns or plat[col].isna().all():
                return np.nan
            return plat[col].median()

        gt_target = get_plat_med("oxygen")
        start_val = sub["oxygen"].iloc[0]
        delta = gt_target - start_val

        if not np.isfinite(delta) or abs(delta) < T90_MIN_STEP_UMOL:
            continue
        direction = 1 if delta > 0 else -1

        def get_t90_duration(col):
            if col not in sub.columns or sub[col].isna().all():
                return np.nan
            target_plat = get_plat_med(col)
            if np.isnan(target_plat):
                return np.nan
            thresh = start_val + 0.9 * (target_plat - start_val)
            try:
                sig = savgol_filter(
                    sub[col].values,
                    window_length=T90_SAVGOL_WINDOW,
                    polyorder=T90_SAVGOL_POLYORDER,
                )
                mask = (sig >= thresh) if direction > 0 else (sig <= thresh)
                if mask.any():
                    t90_timestamp = float(sub.iloc[np.argmax(mask)]["elapsed"])
                    return t90_timestamp - step_start_time
            except Exception:
                pass
            return np.nan

        gt_t90_dur = get_t90_duration("oxygen")
        if np.isnan(gt_t90_dur):
            continue

        for method, col in prediction_columns.items():
            mod_t90_dur = get_t90_duration(col)
            lag = mod_t90_dur - gt_t90_dur if not np.isnan(mod_t90_dur) else np.nan

            records.append({
                "exp_id": exp_id,
                "step_idx": i + 1,
                "target_conc": gt_target,
                "step_magnitude": delta,
                "model_key": method,
                "base_key": method,
                "method": method,
                "is_rts": False,
                "gt_t90_duration": gt_t90_dur,
                "model_t90_duration": mod_t90_dur,
                "lag_s": lag,
            })

    return pd.DataFrame(records)

# ------------------------------------------------------------------------------
# 9b. V14 T90 EXTRACTOR (B): RAW-SIGNAL ALIGNED, ABSOLUTE LAG
# ------------------------------------------------------------------------------
def compute_t90_events_raw_aligned(
    df: pd.DataFrame,
    prediction_columns: Dict[str, str],
    exp_id: str,
) -> pd.DataFrame:
    d = df.sort_values("elapsed").reset_index(drop=True)
    records = []

    intervals = ANALYSIS_INTERVALS_S
    if len(intervals) < 2:
        return pd.DataFrame(records)

    for i in range(1, len(intervals)):
        prev_s, prev_e = intervals[i - 1]
        curr_s, curr_e = intervals[i]

        base_df = d[(d["elapsed"] >= prev_s) & (d["elapsed"] <= prev_e)].dropna(subset=["oxygen"])
        curr_df = d[(d["elapsed"] >= curr_s) & (d["elapsed"] <= curr_e)].dropna(subset=["oxygen"])
        if len(base_df) < T90_RAW_ALIGNED_MIN_PLATEAU or len(curr_df) < T90_RAW_ALIGNED_MIN_PLATEAU:
            continue

        base_o2 = base_df["oxygen"].mean()
        target_o2 = curr_df["oxygen"].mean()
        delta = target_o2 - base_o2
        if not np.isfinite(delta) or abs(delta) < T90_MIN_STEP_UMOL:
            continue
        sign = np.sign(delta)

        t_win = d[(d["elapsed"] >= prev_e) & (d["elapsed"] <= curr_e)].copy()
        t_win = t_win.dropna(subset=["oxygen"])
        if t_win.empty:
            continue

        v5 = base_o2 + 0.05 * delta
        v90 = base_o2 + 0.90 * delta

        gt_start_mask = (t_win["oxygen"] - v5) * sign >= 0
        t_start = t_win.loc[gt_start_mask, "elapsed"].min() if gt_start_mask.any() else np.nan

        gt_90_mask = (t_win["oxygen"] - v90) * sign >= 0
        t90_gt = t_win.loc[gt_90_mask, "elapsed"].min() if gt_90_mask.any() else np.nan

        t90_true_abs = (t90_gt - t_start) if pd.notna(t90_gt) and pd.notna(t_start) else np.nan

        for method, col in prediction_columns.items():
            if col not in t_win.columns:
                continue
            t_win_m = t_win.dropna(subset=[col])
            if t_win_m.empty:
                continue

            model_90_mask = (t_win_m[col] - v90) * sign >= 0
            t90_m = t_win_m.loc[model_90_mask, "elapsed"].min() if model_90_mask.any() else np.nan

            t90_lag = abs(t90_m - t90_gt) if pd.notna(t90_m) and pd.notna(t90_gt) else np.nan
            t90_abs = (t90_true_abs + t90_lag) if pd.notna(t90_true_abs) and pd.notna(t90_lag) else np.nan

            records.append({
                "exp_id": exp_id,
                "step_idx": i,
                "target_conc": target_o2,
                "base_conc": base_o2,
                "step_magnitude": delta,
                "model_key": method,
                "base_key": method,
                "method": method,
                "is_rts": False,
                "gt_t90_true": t90_true_abs,
                "T90_Lag": t90_lag,
                "T90_Abs": t90_abs,
            })

    return pd.DataFrame(records)

# ------------------------------------------------------------------------------
# 9c. OPTIONAL IMPORT OF PRE-COMPUTED V14 T90 ROWS
# ------------------------------------------------------------------------------
def load_v14_t90_imports() -> Dict[str, pd.DataFrame]:
    empty = {"raw_aligned": pd.DataFrame(), "savgol": pd.DataFrame()}

    if not V14_T90_IMPORT_ENABLED:
        return empty

    if not os.path.isdir(V14_T90_CSV_DIR):
        print(f"  ! V14 T90 import skipped: directory not found: {V14_T90_CSV_DIR}")
        return empty

    out = {}
    for key, filename in [
        ("raw_aligned", V14_T90_RAW_ALIGNED_FILE),
        ("savgol", V14_T90_SAVGOL_FILE),
    ]:
        path = os.path.join(V14_T90_CSV_DIR, filename)
        if not os.path.exists(path):
            print(f"  ! V14 T90 import skipped ({key}): file not found: {path}")
            out[key] = pd.DataFrame()
            continue
        try:
            raw = pd.read_csv(path)
            if "base_key" not in raw.columns:
                print(f"  ! V14 T90 import skipped ({key}): no 'base_key' column in {filename}")
                out[key] = pd.DataFrame()
                continue

            sel = raw[raw["base_key"].astype(str) == V14_IMPORT_BASE_KEY].copy()
            if "is_rts" in sel.columns:
                sel = sel[~sel["is_rts"].astype(bool)]

            if sel.empty:
                print(
                    f"  ! V14 T90 import ({key}): no rows for base_key="
                    f"'{V14_IMPORT_BASE_KEY}'."
                )
                out[key] = pd.DataFrame()
                continue

            sel["method"] = V14_IMPORT_METHOD_LABEL
            sel["model_key"] = V14_IMPORT_METHOD_LABEL
            sel["base_key"] = V14_IMPORT_METHOD_LABEL
            sel["imported_from_v14"] = True
            out[key] = sel
            print(f"  > V14 T90 import ({key}): {len(sel)} rows as '{V14_IMPORT_METHOD_LABEL}'.")
        except Exception as exc:
            print(f"  ! V14 T90 import failed ({key}): {exc}")
            out[key] = pd.DataFrame()

    return out


def plot_all_method_timeseries(
    df: pd.DataFrame,
    prediction_columns: Dict[str, str],
    exp_id: str,
) -> None:
    export_cols = [
        "sample_id", "experiment_id", "timestamp", "elapsed", "oxygen",
        "surface_o2", "inv_i", "inv_std", "temp",
    ] + list(prediction_columns.values())
    export_cols = [c for c in export_cols if c in df.columns]

    csv_path = os.path.join(CSV_EXPORT_DIR, f"Inference_{safe_name(exp_id)}_AllMethods.csv")
    atomic_to_csv(df[export_cols].copy(), csv_path, index=False)

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.plot(df["elapsed"], df["oxygen"], label="Ground Truth", linewidth=3)

    for method, col in prediction_columns.items():
        ax.plot(df["elapsed"], df[col], label=method, linewidth=1.4, alpha=0.9)

    for start, end in ANALYSIS_INTERVALS_S:
        ax.axvspan(start, end, alpha=0.08)

    ax.set_title(f"Dynamic Oxygen Reconstruction - {exp_id}")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel(f"Oxygen ({OXYGEN_UNIT})")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(
        os.path.join(SUBDIRS["inference"], f"Inference_{safe_name(exp_id)}_AllMethods.png"),
        dpi=180,
    )
    plt.close(fig)


def plot_chemometrics_for_method(
    df: pd.DataFrame,
    pred_col: str,
    method: str,
    exp_id: str,
) -> None:
    y_true = df["oxygen"].to_numpy(dtype=float)
    y_pred = df[pred_col].to_numpy(dtype=float)
    residual = y_pred - y_true

    method_dir = os.path.join(SUBDIRS["chemometrics"], safe_name(exp_id))
    os.makedirs(method_dir, exist_ok=True)

    export = pd.DataFrame({
        "sample_id": df["sample_id"].to_numpy(),
        "elapsed": df["elapsed"].to_numpy(),
        "ground_truth": y_true,
        "prediction": y_pred,
        "residual": residual,
    })
    atomic_to_csv(
        export,
        os.path.join(CSV_EXPORT_DIR, f"Chemometrics_{safe_name(exp_id)}_{safe_name(method)}.csv"),
        index=False,
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].scatter(y_true, y_pred, alpha=0.30, s=15)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if finite.any():
        lo = min(np.min(y_true[finite]), np.min(y_pred[finite]))
        hi = max(np.max(y_true[finite]), np.max(y_pred[finite]))
        axes[0].plot([lo, hi], [lo, hi], "k--", linewidth=2)
    axes[0].set_xlabel(f"Ground truth ({OXYGEN_UNIT})")
    axes[0].set_ylabel(f"Prediction ({OXYGEN_UNIT})")
    axes[0].set_title(f"{method} parity | R2={safe_r2(y_true, y_pred):.3f}")

    axes[1].hist(residual[np.isfinite(residual)], bins=50, alpha=0.8)
    axes[1].set_xlabel(f"Residual: prediction - truth ({OXYGEN_UNIT})")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Residuals | MAE={mean_absolute_error(y_true, y_pred):.3f}")

    fig.suptitle(f"{exp_id} - {method}")
    fig.tight_layout()
    fig.savefig(
        os.path.join(method_dir, f"Chemometrics_{safe_name(method)}.png"),
        dpi=180,
    )
    plt.close(fig)


def export_method_parameters(exp_id: str, method: str, params: dict) -> None:
    json_path = os.path.join(
        SUBDIRS["params"],
        f"Params_{safe_name(exp_id)}_{safe_name(method)}.json",
    )
    csv_path = os.path.join(
        SUBDIRS["params"],
        f"Params_{safe_name(exp_id)}_{safe_name(method)}.csv",
    )
    atomic_write_json(params, json_path)
    flat = {k: v for k, v in params.items() if np.isscalar(v) or isinstance(v, str)}
    atomic_to_csv(pd.DataFrame([flat]), csv_path, index=False)


def aggregate_metric_summary(metrics_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = [c for c in metrics_long.columns if c.startswith(("MAE_", "RMSE_", "R2_"))]

    for method, g in metrics_long.groupby("method"):
        row = {"method": method, "n_folds": int(g["experiment_id"].nunique())}
        for col in metric_cols:
            vals = g[col].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_median"] = np.nan
                row[f"{col}_std"] = np.nan
                row[f"{col}_ci95_low"] = np.nan
                row[f"{col}_ci95_high"] = np.nan
                continue

            mean_v = float(np.mean(vals))
            median_v = float(np.median(vals))
            std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            if len(vals) > 1:
                sem = std_v / math.sqrt(len(vals))
                crit = float(student_t.ppf(0.975, df=len(vals) - 1))
                low, high = mean_v - crit * sem, mean_v + crit * sem
            else:
                low = high = mean_v

            row[f"{col}_mean"] = mean_v
            row[f"{col}_median"] = median_v
            row[f"{col}_std"] = std_v
            row[f"{col}_ci95_low"] = low
            row[f"{col}_ci95_high"] = high
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_t90_summary_raw_aligned(t90_df: pd.DataFrame) -> pd.DataFrame:
    if t90_df.empty:
        return pd.DataFrame()

    rows = []
    sv_lag = t90_df.loc[t90_df["method"] == "SV", "T90_Lag"].dropna()
    sv_mean = float(sv_lag.mean()) if not sv_lag.empty else np.nan

    for method, g in t90_df.groupby("method"):
        lag_vals = g["T90_Lag"].dropna().to_numpy(dtype=float)
        abs_vals = g["T90_Abs"].dropna().to_numpy(dtype=float)

        lag_mean = float(np.mean(lag_vals)) if len(lag_vals) else np.nan
        reduction = (
            100.0 * (sv_mean - lag_mean) / sv_mean
            if np.isfinite(sv_mean) and sv_mean > 0 and np.isfinite(lag_mean)
            else np.nan
        )

        rows.append({
            "method": method,
            "n_events": int(len(lag_vals)),
            "T90_Lag_mean_s": lag_mean,
            "T90_Lag_median_s": float(np.median(lag_vals)) if len(lag_vals) else np.nan,
            "T90_Lag_std_s": float(np.std(lag_vals, ddof=1)) if len(lag_vals) > 1 else np.nan,
            "T90_Abs_mean_s": float(np.mean(abs_vals)) if len(abs_vals) else np.nan,
            "T90_Abs_std_s": float(np.std(abs_vals, ddof=1)) if len(abs_vals) > 1 else np.nan,
            "T90_Abs_n": int(len(abs_vals)),
            "percent_lag_reduction_vs_SV": reduction,
        })
    return pd.DataFrame(rows)


def aggregate_t90_summary_savgol(t90_df: pd.DataFrame) -> pd.DataFrame:
    if t90_df.empty:
        return pd.DataFrame()

    rows = []
    for method, g in t90_df.groupby("method"):
        vals = g["lag_s"].dropna().to_numpy(dtype=float)
        rows.append({
            "method": method,
            "n_events": int(len(vals)),
            "signed_lag_mean_s": float(np.mean(vals)) if len(vals) else np.nan,
            "signed_lag_median_s": float(np.median(vals)) if len(vals) else np.nan,
            "signed_lag_std_s": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
            "abs_lag_mean_s": float(np.mean(np.abs(vals))) if len(vals) else np.nan,
            "abs_lag_std_s": float(np.std(np.abs(vals), ddof=1)) if len(vals) > 1 else np.nan,
        })
    return pd.DataFrame(rows)

# ==============================================================================
# 10. PREFLIGHT REPORTING
# ==============================================================================
def export_preflight_summary(df: pd.DataFrame) -> None:
    rows = []
    for exp_id, g in df.groupby("experiment_id"):
        rows.append({
            "experiment_id": exp_id,
            "n_frames": len(g),
            "duration_s": float(g["elapsed"].max() - g["elapsed"].min()),
            "oxygen_min": float(g["oxygen"].min()),
            "oxygen_max": float(g["oxygen"].max()),
            "oxygen_mean": float(g["oxygen"].mean()),
            "temperature_min": float(g["temp"].min()),
            "temperature_max": float(g["temp"].max()),
            "timestamp_start": str(g["timestamp"].min()),
            "timestamp_end": str(g["timestamp"].max()),
        })
    summary = pd.DataFrame(rows)
    atomic_to_csv(summary, os.path.join(SUBDIRS["preflight"], "Dataset_Summary.csv"), index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(summary["experiment_id"].astype(str), summary["n_frames"])
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Number of cached frames")
    ax.set_title("Per-experiment dataset size")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(os.path.join(SUBDIRS["preflight"], "Dataset_Size_By_Experiment.png"), dpi=180)
    plt.close(fig)

# ==============================================================================
# 11. FOLD-LEVEL METHOD ORCHESTRATION
# ==============================================================================
def load_or_fit_json(path: str, fit_fn):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    params = fit_fn()
    atomic_write_json(params, path)
    return params


def run_fold(
    df: pd.DataFrame,
    test_day: str,
    fold_index: int,
) -> Tuple[pd.DataFrame, List[dict], pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 100)
    print(f"Processing Fold {fold_index + 1}: OUTER TEST = {test_day}")
    print("=" * 100)

    outer_train_df = df[df["experiment_id"] != test_day].copy()
    test_df = df[df["experiment_id"] == test_day].copy()

    outer_train_df = outer_train_df.sort_values(["experiment_id", "elapsed"]).reset_index(drop=True)
    test_df = test_df.sort_values(["experiment_id", "elapsed"]).reset_index(drop=True)

    fold_dir = os.path.join(
        CV_CACHE_DIR,
        f"fold_{fold_index:03d}_{safe_name(test_day)}_{CONFIG_HASH}",
    )
    os.makedirs(fold_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # STEP 1: Training-only steady-state calibration.
    # --------------------------------------------------------------------------
    calibration_path = os.path.join(fold_dir, "steady_calibration.json")
    calibration = load_or_fit_json(
        calibration_path,
        lambda: fit_steady_calibration(outer_train_df),
    )
    export_method_parameters(test_day, "SV_Calibration", calibration)

    outer_train_df["surface_o2"] = apply_steady_calibration(calibration, outer_train_df)
    test_df["surface_o2"] = apply_steady_calibration(calibration, test_df)

    prediction_columns = {}

    if RUN_METHODS.get("SV", False):
        cached = load_prediction_cache(fold_dir, "SV", test_df)
        if cached is None:
            cached = test_df["surface_o2"].to_numpy(dtype=float)
            save_prediction_cache(
                fold_dir,
                "SV",
                test_df,
                cached,
                extra_manifest={"calibration": calibration},
            )
        test_df["pred_SV"] = cached
        prediction_columns["SV"] = "pred_SV"

    # --------------------------------------------------------------------------
    # STEP 2: Fit dynamic first-order parameters using OUTER TRAINING only.
    # --------------------------------------------------------------------------
    dynamic_params_path = os.path.join(fold_dir, "first_order_params.json")
    first_order_params = load_or_fit_json(
        dynamic_params_path,
        lambda: fit_first_order_parameters(outer_train_df),
    )
    export_method_parameters(test_day, "FirstOrderInverse", first_order_params)

    if RUN_METHODS.get("FirstOrderInverse", False):
        method = "FirstOrderInverse"
        pred = load_prediction_cache(fold_dir, method, test_df)
        if pred is None:
            pred = group_apply_first_order(
                test_df,
                "surface_o2",
                tau_s=first_order_params["tau_s"],
                ewm_alpha=first_order_params["ewm_alpha"],
            )
            pred = clip_if_enabled(pred, outer_train_df)
            save_prediction_cache(fold_dir, method, test_df, pred, first_order_params)
        test_df[f"pred_{method}"] = pred
        prediction_columns[method] = f"pred_{method}"

    # --------------------------------------------------------------------------
    # STEP 3: Offline regularized deconvolution baseline.
    # --------------------------------------------------------------------------
    if RUN_METHODS.get("RegularizedDeconv", False):
        method = "RegularizedDeconv"
        params_path = os.path.join(fold_dir, "regularized_deconv_params.json")
        reg_params = load_or_fit_json(
            params_path,
            lambda: fit_regularized_deconv_lambda(
                outer_train_df,
                tau_s=first_order_params["tau_s"],
            ),
        )
        export_method_parameters(test_day, method, reg_params)

        pred = load_prediction_cache(fold_dir, method, test_df)
        if pred is None:
            pred = group_apply_regularized_deconv(
                test_df,
                "surface_o2",
                tau_s=reg_params["tau_s"],
                lam=reg_params["lambda"],
            )
            pred = clip_if_enabled(pred, outer_train_df)
            save_prediction_cache(fold_dir, method, test_df, pred, reg_params)
        test_df[f"pred_{method}"] = pred
        prediction_columns[method] = f"pred_{method}"

    # --------------------------------------------------------------------------
    # STEP 4: Causal two-state Kalman baseline.
    # --------------------------------------------------------------------------
    if RUN_METHODS.get("CausalKalman", False):
        method = "CausalKalman"
        params_path = os.path.join(fold_dir, "causal_kalman_params.json")
        kalman_params = load_or_fit_json(
            params_path,
            lambda: fit_kalman_parameters(
                outer_train_df,
                tau_s=first_order_params["tau_s"],
            ),
        )
        export_method_parameters(test_day, method, kalman_params)

        pred = load_prediction_cache(fold_dir, method, test_df)
        if pred is None:
            pred = group_apply_kalman(test_df, kalman_params)
            pred = clip_if_enabled(pred, outer_train_df)
            save_prediction_cache(fold_dir, method, test_df, pred, kalman_params)
        test_df[f"pred_{method}"] = pred
        prediction_columns[method] = f"pred_{method}"

    # --------------------------------------------------------------------------
    # STEP 5: Fair temporal neural baselines (DISABLED FOR CLASSICAL RUN).
    # --------------------------------------------------------------------------
    for method in ["CNNLSTM", "CNNTCN", "TViT"]:
        if not RUN_METHODS.get(method, False):
            continue
        print(f"  > Running neural method: {method}")
        pred = run_neural_method(
            method,
            outer_train_df,
            test_df,
            fold_id=test_day,
            fold_index=fold_index,
            fold_dir=fold_dir,
        )
        test_df[f"pred_{method}"] = pred
        prediction_columns[method] = f"pred_{method}"

    # --------------------------------------------------------------------------
    # STEP 6: Reproducible fold-level exports and figures.
    # --------------------------------------------------------------------------
    plot_all_method_timeseries(test_df, prediction_columns, test_day)

    metric_rows = []
    for method, col in prediction_columns.items():
        metric_rows.append(compute_method_metrics(test_df, col, method, test_day))
        plot_chemometrics_for_method(test_df, col, method, test_day)

    metrics_df = pd.DataFrame(metric_rows)
    atomic_to_csv(
        metrics_df,
        os.path.join(CSV_EXPORT_DIR, f"Metrics_{safe_name(test_day)}.csv"),
        index=False,
    )

    t90_raw_aligned = compute_t90_events_raw_aligned(test_df, prediction_columns, test_day)
    atomic_to_csv(
        t90_raw_aligned,
        os.path.join(SUBDIRS["t90"], f"T90_RawAligned_{safe_name(test_day)}.csv"),
        index=False,
    )

    t90_savgol = compute_t90_events_savgol(test_df, prediction_columns, test_day)
    atomic_to_csv(
        t90_savgol,
        os.path.join(SUBDIRS["t90"], f"T90_Savgol_{safe_name(test_day)}.csv"),
        index=False,
    )

    atomic_write_json(
        {
            "fold": test_day,
            "fold_index": fold_index,
            "config_hash": CONFIG_HASH,
            "methods": list(prediction_columns.keys()),
            "training_protocol": "fixed_budget_final_epoch",
            "n_epochs": N_EPOCHS,
            "completed_at": datetime.now().isoformat(),
        },
        os.path.join(fold_dir, "FOLD_COMPLETE.json"),
    )

    return test_df, metric_rows, t90_raw_aligned, t90_savgol

# ==============================================================================
# 12. GLOBAL SUMMARY UPDATES
# ==============================================================================
def write_global_summaries(
    metric_rows: List[dict],
    t90_raw_aligned_frames: List[pd.DataFrame],
    t90_savgol_frames: List[pd.DataFrame],
    v14_imports: Optional[Dict[str, pd.DataFrame]] = None,
) -> None:
    if not metric_rows:
        return

    metrics_long = pd.DataFrame(metric_rows)
    atomic_to_csv(
        metrics_long,
        os.path.join(OUTPUT_DIR, "Global_Performance_Summary_Long.csv"),
        index=False,
    )

    base = metrics_long[["experiment_id"]].drop_duplicates().set_index("experiment_id")
    wide = base.copy()
    for method, g in metrics_long.groupby("method"):
        g2 = g.set_index("experiment_id")
        for metric in ["MAE_Global", "MAE_Stable", "MAE_Transient", "MAE_Unstructured", "R2_Global"]:
            if metric in g2.columns:
                wide[f"{metric}_{method}"] = g2[metric]
    wide = wide.reset_index()
    atomic_to_csv(
        wide,
        os.path.join(OUTPUT_DIR, "Global_Performance_Summary.csv"),
        index=False,
    )

    aggregate = aggregate_metric_summary(metrics_long)
    atomic_to_csv(
        aggregate,
        os.path.join(OUTPUT_DIR, "Aggregate_Performance_Summary.csv"),
        index=False,
    )

    v14_imports = v14_imports or {"raw_aligned": pd.DataFrame(), "savgol": pd.DataFrame()}

    valid_ra = [x for x in t90_raw_aligned_frames if x is not None and not x.empty]
    imported_ra = v14_imports.get("raw_aligned", pd.DataFrame())
    if imported_ra is not None and not imported_ra.empty:
        valid_ra = valid_ra + [imported_ra]
    if valid_ra:
        ra_all = pd.concat(valid_ra, ignore_index=True)
        atomic_to_csv(
            ra_all,
            os.path.join(OUTPUT_DIR, "Global_T90_Events_RawAligned.csv"),
            index=False,
        )
        atomic_to_csv(
            aggregate_t90_summary_raw_aligned(ra_all),
            os.path.join(OUTPUT_DIR, "Aggregate_T90_Summary_RawAligned.csv"),
            index=False,
        )

    valid_sg = [x for x in t90_savgol_frames if x is not None and not x.empty]
    imported_sg = v14_imports.get("savgol", pd.DataFrame())
    if imported_sg is not None and not imported_sg.empty:
        valid_sg = valid_sg + [imported_sg]
    if valid_sg:
        sg_all = pd.concat(valid_sg, ignore_index=True)
        atomic_to_csv(
            sg_all,
            os.path.join(OUTPUT_DIR, "Global_T90_Events_Savgol.csv"),
            index=False,
        )
        atomic_to_csv(
            aggregate_t90_summary_savgol(sg_all),
            os.path.join(OUTPUT_DIR, "Aggregate_T90_Summary_Savgol.csv"),
            index=False,
        )

# ==============================================================================
# 13. EXECUTION ORCHESTRATOR
# ==============================================================================
def main():
    print("=" * 100)
    print(f"PINN-IDC Dynamic Benchmark Suite V{SCRIPT_VERSION}")
    print(f"Device: {DEVICE}")
    print(f"Config hash: {CONFIG_HASH}")
    print(f"Local CPU Workers: {NUM_WORKERS}")
    print(f"Training protocol: fixed {N_EPOCHS} epochs, final epoch retained (V10.9 parity)")
    print(f"T90 methodology: V14 (both raw-aligned and Savgol extractors)")
    print(f"Active methods: {[k for k, v in RUN_METHODS.items() if v]}")
    print("=" * 100)

    experiments = find_experiment_files_v11(EXPERIMENTS_ROOT_DIR)
    if N_EXPERIMENTS_CAP:
        experiments = experiments[:N_EXPERIMENTS_CAP]

    if not experiments:
        raise RuntimeError("No experiments discovered. Check EXPERIMENTS_ROOT_DIR.")

    df = get_master_df(experiments)
    export_preflight_summary(df)

    exp_list = df["experiment_id"].astype(str).unique().tolist()
    print(f"\nOuter LOOCV experiments ({len(exp_list)}): {exp_list}")

    print("\n--- Optional V14 T90 import ---")
    v14_imports = load_v14_t90_imports()

    all_metric_rows = []
    all_t90_raw_aligned = []
    all_t90_savgol = []

    failure_log_path = os.path.join(SUBDIRS["logs"], "Fold_Failures.log")

    for fold_index, test_day in enumerate(exp_list):
        try:
            _, metric_rows, t90_ra, t90_sg = run_fold(df, test_day, fold_index)
            all_metric_rows.extend(metric_rows)
            all_t90_raw_aligned.append(t90_ra)
            all_t90_savgol.append(t90_sg)

            write_global_summaries(
                all_metric_rows,
                all_t90_raw_aligned,
                all_t90_savgol,
                v14_imports,
            )

        except KeyboardInterrupt:
            print("\nExecution interrupted by user. Cached progress has been preserved.")
            write_global_summaries(
                all_metric_rows,
                all_t90_raw_aligned,
                all_t90_savgol,
                v14_imports,
            )
            raise
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"\n! Fold FAILED: {test_day}\n{tb}")
            with open(failure_log_path, "a", encoding="utf-8") as f:
                f.write("\n" + "=" * 100 + "\n")
                f.write(f"TIME: {datetime.now().isoformat()}\n")
                f.write(f"FOLD: {test_day}\n")
                f.write(tb + "\n")
            continue

    write_global_summaries(
        all_metric_rows,
        all_t90_raw_aligned,
        all_t90_savgol,
        v14_imports,
    )

    print("\n" + "=" * 100)
    print("Execution Finished.")
    print(f"Results: {OUTPUT_DIR}")
    print(f"Persistent fold caches: {CV_CACHE_DIR}")
    print("Restarting this script will resume cached work rather than starting over.")
    print("=" * 100)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
