# Repository edition: see docs/CHANGES.md and docs/REPRODUCIBILITY.md.
"""
==============================================================================
PROJECT: PINN-IDC (Inverse Dynamic Compensation) Framework
VERSION: 10.9.4 - Physics-Informed Kinetic MTL (Gradient & Future)
==============================================================================

--- SCIENTIFIC CONTEXT & GOAL ---
The fundamental challenge is the "Response Time Problem" in chemical sensing.
Diffusion creates a physical "lag" where the sensor reading trails the true
concentration.

Previous Bayesian approaches (V10.9.3) focused on quantifying uncertainty.
However, diffusion is a deterministic physical process governed by Fick's Laws.
Therefore, uncertainty is less important than KINETICS (Velocity and Momentum).

In V10.9.4, we implement a "Kinetic Multi-Task Learning" approach.
The model is forced to learn three simultaneous tasks:
1. PREDICT PRESENT: Current Oxygen (t).
2. PREDICT FLUX: The Gradient/Derivative (dC/dt) over the last 1 second.
3. PREDICT TRAJECTORY: The Future Oxygen (t + 5 seconds).

By sharing the Vision Transformer backbone across these tasks, the embeddings
become "Physically Informed." The model cannot minimize the loss for Task 2
and Task 3 without learning to extract temporal features (deltas) from the
frame stack.

--- EVOLUTION OF THE FRAMEWORK (V1.0 -> V10.9.4) ---
- V1-V7: Static PINNs.
- V8-V10.8: Dynamic sensing, T90 Benchmarking.
- V10.9.0-2: MTL (Classification) & Checkpointing.
- V10.9.3: Bayesian Uncertainty (Good for reliability, bad for lag correction).
- V10.9.4 (CURRENT):
    1. KINETIC MTL HEADS: Architecture split into Head_Curr, Head_Grad, Head_Fut.
    2. TARGET ENGINEERING: Pre-calculates derivatives and look-ahead targets
       using Polars shifts during the caching phase.
    3. NEW DIAGNOSTICS: Plots "Predicted vs True Derivative" to validate
       physical learning.

--- FOLDER STRUCTURE & OUTPUTS ---
- Dynamic_IDC_Analysis_V10/
    - ... (Standard Folders)
    - 6_Kinetic_Diagnostics/ (NEW: Derivative & Future analysis)
    - 8_Optuna_Optimization/
    - csv_exports/
    - Global_Performance_Summary.csv

==============================================================================
"""

# Portable repository paths; edit scripts/repo_config.py for your computer.
from repo_config import data_path, output_path, cache_path, external_path

import os
import sys
import glob
import re
import json
import warnings
import random
import traceback
import functools
from datetime import datetime, timedelta

# Data Handling
import pandas as pd
import polars as pl  # High-performance Dataframe library
import numpy as np
import cv2

# Visualization
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# SciPy & Sklearn
from scipy.signal import medfilt, savgol_filter
from scipy.interpolate import interp1d
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold

# PyTorch & AI
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
from torchvision import transforms

# Optimization
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances

# ==============================================================================
# 1. REPRODUCIBILITY & PERFORMANCE CONFIG
# ==============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

matplotlib_backend = plt.get_backend()
plt.style.use('seaborn-v0_8-whitegrid')
warnings.filterwarnings("ignore")

# ==============================================================================
# 2. CONFIGURATION & PATHS
# ==============================================================================
# Update this path to your specific dataset location
BASE_PROJECT_DIR = data_path('aqueous')
EXPERIMENTS_ROOT_DIR = BASE_PROJECT_DIR

# Manual Definition of Equilibrium States (Ground Truth for Phase Classifier)
ANALYSIS_INTERVALS_S = [(1450, 1650), (2300, 2500), (3100, 3300), (3900, 4100), (4700, 4900)]
UNSTRUCTURED_CUTOFF_S = 1000
N_EXPERIMENTS_CAP = None

# Output Directories
OUTPUT_DIR = output_path('aqueous_tvit_kinetic')
CSV_EXPORT_DIR = os.path.join(OUTPUT_DIR, "csv_exports")
FEATURE_CACHE_DIR = cache_path('aqueous_tvit_kinetic', "features")
CV_CACHE_DIR = cache_path('aqueous_tvit_kinetic', "folds")
MODEL_CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "7_Model_Checkpoints")
OPTUNA_DIR = os.path.join(OUTPUT_DIR, "8_Optuna_Optimization")

# Hyperparameters
N_STACKED_FRAMES = 3  # Number of historical frames to feed (t, t-1, t-2)
FRAME_STACK_STEP = 10  # Gap between stacked frames (approx 0.33s at 30fps)
RESIZE_DIM = (224, 224)
STORAGE_DIM = (48, 48)
BATCH_SIZE = 32
N_EPOCHS = 20
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Kinetic Physics Parameters (V10.9.4 NEW)
# --------------------------------------------------------------------------
# We define physics targets based on time deltas.
# If FPS is approx 30, and we want a 1-second derivative, we look back ~30 frames.
DERIVATIVE_WINDOW_S = 1.0  # Calculate slope over last 1 second
FUTURE_PRED_S = 5.0  # Predict oxygen 5 seconds into the future
# --------------------------------------------------------------------------

# Optuna Config
OPTUNA_TRIALS = 20
OPTUNA_SUBSET_FRAC = 0.05
OPTUNA_DB_PATH = f"sqlite:///{os.path.join(OPTUNA_DIR, 'optuna_kinetic.db')}"

# Create Directory Structure
directories_to_make = [
    OUTPUT_DIR, CSV_EXPORT_DIR, FEATURE_CACHE_DIR, CV_CACHE_DIR, MODEL_CHECKPOINT_DIR, OPTUNA_DIR,
    os.path.join(OUTPUT_DIR, "0_Preflight_Diagnostics"),
    os.path.join(OUTPUT_DIR, "1_Training_Convergence"),
    os.path.join(OUTPUT_DIR, "2_Inference_Results"),
    os.path.join(OUTPUT_DIR, "3_Chemometrics"),
    os.path.join(OUTPUT_DIR, "5_Uncertainty_Analysis"),
    os.path.join(OUTPUT_DIR, "6_Kinetic_Diagnostics")  # NEW
]
for d in directories_to_make: os.makedirs(d, exist_ok=True)


# ==============================================================================
# 3. ROBUST DATA DISCOVERY & PARSING (POLARS ENHANCED)
# ==============================================================================
def find_experiment_files_v10(root_dir):
    """
    Scans the root directory for experimental subfolders.
    """
    print("--- Discovery: Searching for Experiment Directories ---")
    all_dirs = [d for d in glob.glob(os.path.join(root_dir, '*-*')) if os.path.isdir(d)]
    all_dirs.sort()
    valid_exps = []
    for exp_dir in tqdm(all_dirs, desc="Scanning for Valid Data"):
        exp_id = os.path.basename(exp_dir)
        roi_folders = [d for d in glob.glob(os.path.join(exp_dir, '*')) if d.endswith('_ROI_Output')]
        arduino_logs = glob.glob(os.path.join(exp_dir, "**", "*_arduino_*.txt"), recursive=True)
        temp_logs = glob.glob(os.path.join(exp_dir, "**", "*temperature*.csv"), recursive=True)

        if roi_folders and arduino_logs and temp_logs:
            vids = glob.glob(os.path.join(roi_folders[0], '*.mp4'))
            if vids:
                valid_exps.append({
                    "id": exp_id,
                    "video_path": vids[0],
                    "arduino_path": arduino_logs[0],
                    "temp_path": temp_logs[0]
                })
    return valid_exps


def parse_logs_polars(exp):
    """
    Parses Arduino and Temperature logs with Polars.
    """
    # 1. Parse Arduino Log
    o2_data = []
    try:
        with open(exp['arduino_path'], 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if "MEA" in line:
                    p = line.split()
                    try:
                        ts_str = f"{p[0]} {p[1]}"
                        o2_val = float(p[p.index("MEA") + 5]) / 1000.0
                        o2_data.append({"timestamp_str": ts_str, "oxygen": o2_val})
                    except:
                        continue
    except Exception as e:
        print(f"Error reading arduino log for {exp['id']}: {e}")
        return pd.DataFrame()

    if not o2_data: return pd.DataFrame()

    df_o2 = pl.DataFrame(o2_data)
    df_o2 = df_o2.with_columns(pl.col("timestamp_str").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("timestamp"))
    df_o2 = df_o2.sort("timestamp")

    # 2. Parse Temperature Log
    try:
        df_temp = pl.read_csv(exp['temp_path'], has_header=True, infer_schema_length=10000,
                              null_values=["NaN", "nan", "NA", "null"])
        df_temp = df_temp.select([
            pl.col(df_temp.columns[0]).alias("timestamp"),
            pl.col(df_temp.columns[1]).cast(pl.Float64, strict=False).alias("temp")
        ])
        if df_temp.schema["timestamp"] == pl.Utf8:
            df_temp = df_temp.with_columns(pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False))

    except Exception as e:
        df_temp = pl.DataFrame({"timestamp": df_o2["timestamp"], "temp": 20.0})

    # 3. Merge
    df_o2 = df_o2.sort("timestamp")
    df_temp = df_temp.sort("timestamp").drop_nulls(subset=["timestamp"])
    df_merged = df_o2.join_asof(df_temp, on="timestamp", strategy="nearest")

    # 4. Interpolate via Pandas
    pdf = df_merged.to_pandas()
    pdf['temp'] = pd.to_numeric(pdf['temp'], errors='coerce')
    pdf['oxygen'] = pdf['oxygen'].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    pdf['temp'] = pdf['temp'].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    pdf['temp'] = pdf['temp'].fillna(20.0)

    # 5. Handle NaNs
    if pdf.isnull().values.any():
        pdf = pdf.fillna(method='ffill').fillna(0.0)

    return pdf


def get_master_df(experiments, cache_path):
    """
    Extracts features and calculates KINETIC TARGETS (Derivative + Future).
    """
    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Cache corrupted ({e}), rebuilding...")

    all_frames = []

    for exp in tqdm(experiments, desc="Processing Videos"):
        df_logs = parse_logs_polars(exp)
        if df_logs.empty: continue

        cap = cv2.VideoCapture(exp['video_path'])
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        match = re.search(r'_(\d{8}_\d{6})', os.path.basename(exp['video_path']))
        start_ts = datetime.strptime(match.group(1), '%Y%m%d_%H%M%S') if match else df_logs['timestamp'].iloc[0]

        step = int(max(1, fps / 2))  # Approx 2 samples per second
        log_timestamps = df_logs['timestamp'].values

        # Determine indices for Kinetic Calculations
        # Note: 'step' determines the sampling rate of 'all_frames'.
        # We need to approximate how many 'steps' equal 1 second and 5 seconds.
        # fps / step = samples_per_second
        samples_per_sec = fps / step
        deriv_lookback = int(DERIVATIVE_WINDOW_S * samples_per_sec)
        future_lookahead = int(FUTURE_PRED_S * samples_per_sec)

        temp_exp_data = []

        for f_idx in range(0, total_f, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: break

            ts = start_ts + timedelta(seconds=f_idx / fps)
            elapsed = f_idx / fps

            idx = np.searchsorted(log_timestamps, np.datetime64(ts))
            if idx >= len(df_logs): idx = len(df_logs) - 1
            if idx > 0:
                if abs(log_timestamps[idx] - np.datetime64(ts)) > abs(log_timestamps[idx - 1] - np.datetime64(ts)):
                    idx -= 1

            resized = cv2.resize(frame, STORAGE_DIM)

            temp_exp_data.append({
                'experiment_id': exp['id'],
                'timestamp': ts,
                'elapsed': elapsed,
                'oxygen': df_logs.iloc[idx]['oxygen'],
                'temp': df_logs.iloc[idx]['temp'],
                'inv_i': 1.0 / (np.mean(resized[:, :, 2].astype(float)) + 1e-6),
                'pixels': resized.flatten().astype(np.uint8)
            })
        cap.release()

        # --- CALCULATE KINETIC TARGETS USING POLARS ---
        # We convert the list to Polars to fast-calc shifts
        df_pl = pl.DataFrame(temp_exp_data)

        # 1. Derivative: Current - Value_1s_Ago
        # We use a shift. If current index is i, we want i - deriv_lookback
        df_pl = df_pl.with_columns([
            (pl.col("oxygen") - pl.col("oxygen").shift(deriv_lookback)).fill_null(0.0).alias("target_deriv"),
            pl.col("oxygen").shift(-future_lookahead).alias("target_future")
        ])

        # Drop rows where future target is NaN (end of video)
        # We only drop them for training purposes, or we can fill them with current value
        # (Clamping future to present at end of exp). Clamping is safer for array sizes.
        df_pl = df_pl.with_columns(pl.col("target_future").fill_null(pl.col("oxygen")))

        all_frames.extend(df_pl.to_dicts())

    master_df = pd.DataFrame(all_frames)

    # Type casting
    master_df['temp'] = master_df['temp'].astype(float)
    master_df['oxygen'] = master_df['oxygen'].astype(float)
    master_df['elapsed'] = master_df['elapsed'].astype(float)
    master_df['target_deriv'] = master_df['target_deriv'].astype(float)
    master_df['target_future'] = master_df['target_future'].astype(float)

    master_df.to_parquet(cache_path)
    return master_df


# ==============================================================================
# 4. KINETIC NEURAL NETWORK ARCHITECTURE
# ==============================================================================

class IDC_KineticDataset(Dataset):
    def __init__(self, df, exp_id_map, stack_size=3, step=5):
        self.df = df.reset_index(drop=True)
        self.exp_id_map = exp_id_map
        self.stack_size = stack_size
        self.step = step
        self.resize = transforms.Resize(RESIZE_DIM)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        exp_id = row['experiment_id']

        # Build Stack (Past context)
        frames = []
        for i in range(self.stack_size):
            lookback = max(0, idx - (i * self.step))
            # Prevent crossing experiment boundaries
            if self.df.iloc[lookback]['experiment_id'] != exp_id:
                lookback = idx
            pix = np.asarray(self.df.iloc[lookback]['pixels'], dtype=np.uint8).reshape(STORAGE_DIM[0], STORAGE_DIM[1], 3)
            img_t = torch.from_numpy(pix).permute(2, 0, 1).float() / 255.0
            frames.append(self.resize(img_t))

        stack = torch.cat(frames, dim=0)

        # Targets
        y_curr = torch.tensor(row['oxygen'], dtype=torch.float32)
        y_grad = torch.tensor(row['target_deriv'], dtype=torch.float32)
        y_fut = torch.tensor(row['target_future'], dtype=torch.float32)
        dom_id = torch.tensor(self.exp_id_map[exp_id], dtype=torch.long)

        return stack, y_curr, y_grad, y_fut, dom_id


class TViT_Kinetic_MTL(nn.Module):
    """
    Vision Transformer with Three Kinetic Heads:
    1. Present (Main Task)
    2. Gradient (Physics Task)
    3. Future (Trajectory Task)
    """

    def __init__(self, num_domains, stack_size=3):
        super().__init__()
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0)
        orig_proj = self.vit.patch_embed.proj
        self.vit.patch_embed.proj = nn.Conv2d(
            3 * stack_size, orig_proj.out_channels,
            kernel_size=orig_proj.kernel_size,
            stride=orig_proj.stride, padding=orig_proj.padding
        )
        with torch.no_grad():
            self.vit.patch_embed.proj.weight = nn.Parameter(orig_proj.weight.repeat(1, stack_size, 1, 1) / stack_size)
            self.vit.patch_embed.proj.bias = orig_proj.bias

        embed_dim = self.vit.embed_dim

        # Shared Feature Extractor ^

        # --- HEAD 1: CURRENT STATE ---
        self.head_curr = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

        # --- HEAD 2: DERIVATIVE (Velocity) ---
        self.head_grad = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.GELU(),
            nn.Linear(64, 1)
        )

        # --- HEAD 3: FUTURE (Momentum) ---
        self.head_fut = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        features = self.vit.forward_features(x)[:, 0]  # CLS token

        curr_pred = self.head_curr(features).squeeze(-1)
        grad_pred = self.head_grad(features).squeeze(-1)
        fut_pred = self.head_fut(features).squeeze(-1)

        return curr_pred, grad_pred, fut_pred


# ==============================================================================
# 5. OPTUNA OPTIMIZATION (Weighted Loss Balancing)
# ==============================================================================
def objective_optuna(trial, df_subset, exp_map, num_domains):
    """
    Optimize Learning Rate and Task Weights.
    """
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    w_grad = trial.suggest_float("w_grad", 0.1, 2.0)
    w_fut = trial.suggest_float("w_fut", 0.1, 2.0)

    gkf = GroupKFold(n_splits=3)
    train_idx, val_idx = next(gkf.split(df_subset, groups=df_subset['experiment_id']))

    train_loader = DataLoader(IDC_KineticDataset(df_subset.iloc[train_idx], exp_map), batch_size=16, shuffle=True)
    val_loader = DataLoader(IDC_KineticDataset(df_subset.iloc[val_idx], exp_map), batch_size=16, shuffle=False)

    model = TViT_Kinetic_MTL(num_domains, stack_size=N_STACKED_FRAMES).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Short Training
    for epoch in range(3):
        model.train()
        for x, y_c, y_g, y_f, _ in train_loader:
            x, y_c, y_g, y_f = x.to(DEVICE), y_c.to(DEVICE), y_g.to(DEVICE), y_f.to(DEVICE)
            optimizer.zero_grad()
            p_c, p_g, p_f = model(x)

            loss = criterion(p_c, y_c) + (w_grad * criterion(p_g, y_g)) + (w_fut * criterion(p_f, y_f))
            loss.backward()
            optimizer.step()

    # Validation (Metric: MAE of Current State)
    model.eval()
    val_mae = []
    with torch.no_grad():
        for x, y_c, _, _, _ in val_loader:
            x, y_c = x.to(DEVICE), y_c.to(DEVICE)
            p_c, _, _ = model(x)
            val_mae.append(mean_absolute_error(y_c.cpu().numpy(), p_c.cpu().numpy()))

    return np.mean(val_mae)


def run_optuna_optimization(df, exp_map, num_domains):
    print("\n[OPTUNA] Preparing 5% Data Subset...")
    df_pl = pl.from_pandas(df)
    subset_pl = df_pl.sample(fraction=OPTUNA_SUBSET_FRAC, seed=SEED, shuffle=True)
    df_subset = subset_pl.to_pandas()

    study = optuna.create_study(
        study_name="idc_kinetic_opt", storage=OPTUNA_DB_PATH,
        load_if_exists=True, direction="minimize"
    )

    if len(study.trials) < OPTUNA_TRIALS:
        print(f"[OPTUNA] Starting optimization (Target: {OPTUNA_TRIALS} trials)...")
        study.optimize(
            lambda trial: objective_optuna(trial, df_subset, exp_map, num_domains),
            n_trials=OPTUNA_TRIALS - len(study.trials)
        )
    else:
        print(f"[OPTUNA] Study complete ({len(study.trials)} trials). Loading best params.")

    print(f"[OPTUNA] Best Params: {study.best_params}")
    return study.best_params


# ==============================================================================
# 6. ANALYTICS & PLOTTING
# ==============================================================================

def fit_classical_baseline(train_df, test_df):
    stable_train = pd.concat(
        [train_df[(train_df['elapsed'] >= s) & (train_df['elapsed'] <= e)] for s, e in ANALYSIS_INTERVALS_S])
    if stable_train.empty: stable_train = train_df
    poly = make_pipeline(PolynomialFeatures(2), LinearRegression())
    poly.fit(stable_train[['inv_i']].values, stable_train['oxygen'].values)
    return poly.predict(test_df[['inv_i']].values)


def plot_chemometrics(df, exp_id):
    y_true, y_pred = df['oxygen'], df['o2_pred']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    ax1.scatter(y_true, y_pred, alpha=0.3, s=15, c='blue')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax1.plot(lims, lims, 'k--', lw=2)
    ax1.set_title(f"Parity Plot: {exp_id} ($R^2={r2_score(y_true, y_pred):.3f}$)")
    residuals = y_pred - y_true
    sns.histplot(residuals, kde=True, ax=ax2, color='purple')
    ax2.set_title("Residual Distribution")
    plt.savefig(os.path.join(OUTPUT_DIR, "3_Chemometrics", f"Chemometrics_{exp_id}.png"))
    plt.close()
    pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'residual': residuals}).to_csv(
        os.path.join(CSV_EXPORT_DIR, f"Chemometrics_{exp_id}.csv"), index=False)


def plot_kinetic_diagnostics(df, exp_id):
    """
    NEW: Validates if the model learned the Physics (Derivative and Future).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    t = df['elapsed']

    # 1. Derivative Comparison
    # Smooth the true derivative for better plotting visualization
    true_grad_smooth = savgol_filter(df['target_deriv'], window_length=51, polyorder=3)
    ax1.plot(t, true_grad_smooth, 'k-', alpha=0.6, label='True Derivative (Smoothed)')
    ax1.plot(t, df['grad_pred'], 'r--', label='Predicted Derivative', lw=1.5)
    ax1.set_title(f"Kinetic Learning: Derivative Prediction ({exp_id})")
    ax1.set_ylabel("Delta Oxygen / 1 sec")
    ax1.legend()

    # 2. Future Prediction vs Actual Future
    # We shift the "Future Pred" BACK by 5 seconds to overlay it with the moment it predicted.
    # i.e., at t=100, we predicted t=105. We plot that prediction at t=105 to see if it matches.
    # Actually, simpler visual: Plot (t) vs Future_Pred(t). It should look like the curve shifted left.
    ax2.plot(t, df['oxygen'], 'k-', label='True Oxygen', alpha=0.3)
    ax2.plot(t, df['fut_pred'], 'g--', label=f'Predicted Future (+{FUTURE_PRED_S}s)', lw=2)
    ax2.set_title("Future Trajectory Prediction")
    ax2.legend()

    plt.savefig(os.path.join(OUTPUT_DIR, "6_Kinetic_Diagnostics", f"Kinetics_{exp_id}.png"))
    plt.close()

    df[['elapsed', 'target_deriv', 'grad_pred', 'target_future', 'fut_pred']].to_csv(
        os.path.join(OUTPUT_DIR, "6_Kinetic_Diagnostics", f"Kinetics_Data_{exp_id}.csv"), index=False)


def plot_training_dashboard(history, exp_id):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    ax1.plot(history['epoch'], history['loss_total'], label='Total Loss', color='black')
    ax1.plot(history['epoch'], history['loss_curr'], label='Curr Loss', color='blue', linestyle='--')
    ax1.plot(history['epoch'], history['loss_grad'], label='Grad Loss', color='red', linestyle=':')
    ax1.plot(history['epoch'], history['loss_fut'], label='Fut Loss', color='green', linestyle=':')
    ax1.legend()
    ax2.plot(history['epoch'], history['val_mae'], label='Validation MAE (Current)', color='blue', marker='o')
    ax1.set_title(f"Training Dynamics: {exp_id}")
    plt.savefig(os.path.join(OUTPUT_DIR, "1_Training_Convergence", f"Convergence_{exp_id}.png"))
    plt.close()
    pd.DataFrame(history).to_csv(os.path.join(OUTPUT_DIR, "1_Training_Convergence", f"History_{exp_id}.csv"),
                                 index=False)


def analyze_dynamics_v10(df, exp_id):
    plot_df = df.copy()
    fig, ax = plt.subplots(figsize=(16, 9))
    t = plot_df['elapsed']
    ax.plot(t, plot_df['oxygen'], label='Ground Truth', color='black', lw=3)
    if 'o2_classical' in plot_df.columns: ax.plot(t, plot_df['o2_classical'], label='Classical', color='gray', ls='--')
    if 'o2_pred' in plot_df.columns: ax.plot(t, plot_df['o2_pred'], label='AI Prediction', color='blue', lw=2)
    for (start, end) in ANALYSIS_INTERVALS_S:
        ax.axvspan(start, end, color='green', alpha=0.1)
    ax.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_Inference_Results", f"Inference_{exp_id}.png"))
    plt.close()
    plot_df.to_csv(os.path.join(CSV_EXPORT_DIR, f"Inference_{exp_id}.csv"), index=False)


# ==============================================================================
# 7. EXECUTION ORCHESTRATOR
# ==============================================================================
def main():
    experiments = find_experiment_files_v10(EXPERIMENTS_ROOT_DIR)
    if N_EXPERIMENTS_CAP: experiments = experiments[:N_EXPERIMENTS_CAP]

    # Load Data (Now includes 'target_deriv' and 'target_future')
    df = get_master_df(experiments, os.path.join(FEATURE_CACHE_DIR, "idc_kinetic_v10.9.4.parquet"))

    exp_list = df['experiment_id'].unique()
    exp_map = {name: i for i, name in enumerate(exp_list)}
    num_domains = len(exp_list)

    print(f"Loaded {len(df)} frames. Kinetic Targets Prepared.")

    # --------------------------------------------------------------------------
    # OPTUNA HPO PHASE
    # --------------------------------------------------------------------------
    print("\n--- Phase 1: Optuna Kinetic Optimization ---")
    best_params = run_optuna_optimization(df, exp_map, num_domains)

    p_lr = best_params.get("lr", LR)
    p_w_grad = best_params.get("w_grad", 1.0)
    p_w_fut = best_params.get("w_fut", 1.0)

    print(f"Using Params -> LR: {p_lr:.2e} | W_Grad: {p_w_grad:.2f} | W_Fut: {p_w_fut:.2f}")

    # --------------------------------------------------------------------------
    # MAIN TRAINING LOOP
    # --------------------------------------------------------------------------
    summary_path = os.path.join(OUTPUT_DIR, "Global_Performance_Summary.csv")
    completed_exps = []
    if os.path.exists(summary_path):
        try:
            existing_df = pd.read_csv(summary_path)
            completed_exps = existing_df['experiment_id'].unique().tolist()
            print(f"Resuming... Skipping {len(completed_exps)} experiments.")
        except:
            pass
    else:
        headers = ['experiment_id', 'MAE_AI', 'MAE_Unstruct', 'MAE_Gradient_Pred', 'MAE_Future_Pred']
        pd.DataFrame(columns=headers).to_csv(summary_path, index=False)

    for test_idx, test_day in enumerate(exp_list):
        if test_day in completed_exps:
            continue

        print(f"\nProcessing Fold {test_idx + 1}/{len(exp_list)}: {test_day}")

        train_df = df[df['experiment_id'] != test_day]
        test_df = df[df['experiment_id'] == test_day].copy()

        test_df['o2_classical'] = fit_classical_baseline(train_df, test_df)

        # Datasets
        train_loader = DataLoader(IDC_KineticDataset(train_df, exp_map), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(IDC_KineticDataset(test_df, exp_map), batch_size=BATCH_SIZE, shuffle=False)

        # Init Model
        model = TViT_Kinetic_MTL(num_domains, stack_size=N_STACKED_FRAMES).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=p_lr)
        criterion = nn.MSELoss()

        history = {'epoch': [], 'loss_total': [], 'loss_curr': [], 'loss_grad': [], 'loss_fut': [], 'val_mae': []}

        for epoch in range(N_EPOCHS):
            model.train()
            l_tot, l_c, l_g, l_f = 0, 0, 0, 0

            for x, y_c, y_g, y_f, _ in train_loader:
                x, y_c, y_g, y_f = x.to(DEVICE), y_c.to(DEVICE), y_g.to(DEVICE), y_f.to(DEVICE)
                optimizer.zero_grad()

                # Forward
                pred_c, pred_g, pred_f = model(x)

                # Kinetic Loss Combination
                loss_c = criterion(pred_c, y_c)
                loss_g = criterion(pred_g, y_g)
                loss_f = criterion(pred_f, y_f)

                loss = loss_c + (p_w_grad * loss_g) + (p_w_fut * loss_f)
                loss.backward()
                optimizer.step()

                l_tot += loss.item()
                l_c += loss_c.item()
                l_g += loss_g.item()
                l_f += loss_f.item()

            # Validation (Current State)
            model.eval()
            val_preds = []
            with torch.no_grad():
                for x, y_c, _, _, _ in val_loader:
                    x = x.to(DEVICE)
                    p_c, _, _ = model(x)
                    val_preds.extend(p_c.cpu().numpy())

            current_mae = mean_absolute_error(test_df['oxygen'], val_preds)

            history['epoch'].append(epoch)
            history['loss_total'].append(l_tot / len(train_loader))
            history['loss_curr'].append(l_c / len(train_loader))
            history['loss_grad'].append(l_g / len(train_loader))
            history['loss_fut'].append(l_f / len(train_loader))
            history['val_mae'].append(current_mae)

        plot_training_dashboard(history, test_day)
        torch.save(model.state_dict(), os.path.join(MODEL_CHECKPOINT_DIR, f"model_fold_{test_day}.pth"))

        # Final Inference for Plots
        model.eval()
        preds_c, preds_g, preds_f = [], [], []
        with torch.no_grad():
            for x, _, _, _, _ in val_loader:
                x = x.to(DEVICE)
                p_c, p_g, p_f = model(x)
                preds_c.extend(p_c.cpu().numpy())
                preds_g.extend(p_g.cpu().numpy())
                preds_f.extend(p_f.cpu().numpy())

        test_df['o2_pred'] = preds_c
        test_df['grad_pred'] = preds_g
        test_df['fut_pred'] = preds_f

        analyze_dynamics_v10(test_df, test_day)
        plot_chemometrics(test_df, test_day)
        plot_kinetic_diagnostics(test_df, test_day)  # The new diagnostic

        # Metrics
        unstruct_mask = test_df['elapsed'] < UNSTRUCTURED_CUTOFF_S
        metrics_dict = {
            'experiment_id': test_day,
            'MAE_AI': mean_absolute_error(test_df['oxygen'], test_df['o2_pred']),
            'MAE_Unstruct': mean_absolute_error(test_df[unstruct_mask]['oxygen'], test_df[unstruct_mask]['o2_pred']),
            'MAE_Gradient_Pred': mean_absolute_error(test_df['target_deriv'], test_df['grad_pred']),
            'MAE_Future_Pred': mean_absolute_error(test_df['target_future'], test_df['fut_pred'])
        }

        pd.DataFrame([metrics_dict]).to_csv(summary_path, mode='a', header=False, index=False)

    print("Execution Finished. Physics-Informed Gradient Learning Complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("CRITICAL FAILURE:")
        traceback.print_exc()
        raise
