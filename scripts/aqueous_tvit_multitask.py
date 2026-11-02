# Repository edition: see docs/CHANGES.md and docs/REPRODUCIBILITY.md.
"""
==============================================================================
PROJECT: PINN-IDC (Inverse Dynamic Compensation) Framework
VERSION: 10.9.2 - Crash-Proof Resumable Workflow
==============================================================================

--- SCIENTIFIC CONTEXT & GOAL ---
The fundamental challenge is the "Response Time Problem" in chemical sensing.
Diffusion creates a physical "lag" where the sensor reading trails the true
concentration. We use a Temporal Vision Transformer (TViT) to invert this
diffusion and "predict the present."

--- EVOLUTION OF THE FRAMEWORK (V1.0 -> V10.9.2) ---
- V1-V7: Static PINNs.
- V8-V10.8: Dynamic sensing, T90 Benchmarking, Black Box Training.
- V10.9.0: Multi-Task Learning (MTL) & Uncertainty Quantification (UQ).
- V10.9.1: Fixed PyArrow/Parquet Type Errors.
- V10.9.2 (CURRENT):
    1. CRASH RECOVERY: Implemented "Smart Resume". The script now checks
       'Global_Performance_Summary.csv' before starting a fold. If the
       experiment ID exists, it SKIPS training for that fold.
    2. INCREMENTAL SAVING: Performance metrics are appended to the CSV
       line-by-line immediately after each fold, preventing data loss
       if the script crashes halfway.
    3. MODEL CHECKPOINTING: Saves the trained PyTorch model (.pth) for
       each fold, allowing for future inference without retraining.

--- FOLDER STRUCTURE & OUTPUTS ---
- Dynamic_IDC_Analysis_V10/
    - 0_Preflight_Diagnostics/
    - 1_Training_Convergence/
    - 2_Inference_Results/
    - 3_Chemometrics/
    - 5_Uncertainty_Analysis/
    - 6_MultiTask_Diagnostics/
    - 7_Model_Checkpoints/ (NEW: Saved .pth models)
    - csv_exports/
    - Global_Performance_Summary.csv (Updated incrementally)

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
from scipy.signal import medfilt
from scipy.interpolate import interp1d
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# PyTorch & AI
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader
import timm
from torchvision import transforms

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
OUTPUT_DIR = output_path('aqueous_tvit_multitask')
CSV_EXPORT_DIR = os.path.join(OUTPUT_DIR, "csv_exports")
FEATURE_CACHE_DIR = cache_path('aqueous_tvit_multitask', "features")
CV_CACHE_DIR = cache_path('aqueous_tvit_multitask', "folds")
MODEL_CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "7_Model_Checkpoints")  # NEW

# Hyperparameters
N_STACKED_FRAMES = 3
FRAME_STACK_STEP = 10
RESIZE_DIM = (224, 224)
STORAGE_DIM = (48, 48)
BATCH_SIZE = 32
N_EPOCHS = 20
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Adversarial & Multi-Task Weights
ALPHA_GRL = 1.0
LAMBDA_PHASE = 0.5
LAMBDA_DOMAIN = 0.5

# Create Directory Structure
directories_to_make = [
    OUTPUT_DIR, CSV_EXPORT_DIR, FEATURE_CACHE_DIR, CV_CACHE_DIR, MODEL_CHECKPOINT_DIR,
    os.path.join(OUTPUT_DIR, "0_Preflight_Diagnostics"),
    os.path.join(OUTPUT_DIR, "1_Training_Convergence"),
    os.path.join(OUTPUT_DIR, "2_Inference_Results"),
    os.path.join(OUTPUT_DIR, "3_Chemometrics"),
    os.path.join(OUTPUT_DIR, "4_Response_Time_Reports"),
    os.path.join(OUTPUT_DIR, "5_Uncertainty_Analysis"),
    os.path.join(OUTPUT_DIR, "6_MultiTask_Diagnostics")
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
    Includes Type-Safety for Parquet export.
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

        # STRICT CASTING to Float64 to avoid PyArrow Object Error
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

    return pdf


def get_master_df(experiments, cache_path):
    """
    Extracts features or loads from Parquet cache.
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

        step = int(max(1, fps / 2))
        log_timestamps = df_logs['timestamp'].values

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

            is_stable = 0
            for (s, e) in ANALYSIS_INTERVALS_S:
                if s <= elapsed <= e:
                    is_stable = 1
                    break

            all_frames.append({
                'experiment_id': exp['id'],
                'timestamp': ts,
                'elapsed': elapsed,
                'oxygen': df_logs.iloc[idx]['oxygen'],
                'temp': df_logs.iloc[idx]['temp'],
                'phase_label': is_stable,
                'inv_i': 1.0 / (np.mean(resized[:, :, 2].astype(float)) + 1e-6),
                'inv_std': 1.0 / (np.std(resized[:, :, 2].astype(float)) + 1e-6),
                'pixels': resized.flatten().astype(np.uint8)
            })
        cap.release()

    master_df = pd.DataFrame(all_frames)

    # EXPLICIT TYPE CASTING for Parquet Stability
    master_df['temp'] = master_df['temp'].astype(float)
    master_df['oxygen'] = master_df['oxygen'].astype(float)
    master_df['elapsed'] = master_df['elapsed'].astype(float)

    master_df.to_parquet(cache_path)
    return master_df


# ==============================================================================
# 4. NEURAL NETWORK COMPONENTS (MTL + GRL + UQ)
# ==============================================================================

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFn.apply(x, self.alpha)


class IDC_TemporalDataset(Dataset):
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
        frames = []
        for i in range(self.stack_size):
            lookback = max(0, idx - (i * self.step))
            if self.df.iloc[lookback]['experiment_id'] != exp_id:
                lookback = idx
            pix = self.df.iloc[lookback]['pixels'].reshape(STORAGE_DIM[0], STORAGE_DIM[1], 3)
            img_t = torch.from_numpy(pix).permute(2, 0, 1).float() / 255.0
            frames.append(self.resize(img_t))
        stack = torch.cat(frames, dim=0)
        oxygen = torch.tensor(row['oxygen'], dtype=torch.float32)
        domain_label = torch.tensor(self.exp_id_map[exp_id], dtype=torch.long)
        phase_label = torch.tensor(row['phase_label'], dtype=torch.long)
        return stack, oxygen, domain_label, phase_label


class TViT_MultiTask_IDC(nn.Module):
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
        self.regressor = nn.Sequential(nn.Linear(embed_dim, 128), nn.GELU(), nn.Linear(128, 2))
        self.phase_classifier = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 2))
        self.grl = GradientReversalLayer(alpha=ALPHA_GRL)
        self.domain_classifier = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, num_domains))

    def forward(self, x):
        features = self.vit.forward_features(x)[:, 0]
        reg_out = self.regressor(features)
        mu = reg_out[:, 0]
        log_var = reg_out[:, 1]
        phase_logits = self.phase_classifier(features)
        rev_features = self.grl(features)
        domain_logits = self.domain_classifier(rev_features)
        return mu, log_var, phase_logits, domain_logits


# ==============================================================================
# 5. ANALYTICS: CHEMOMETRICS, UQ, & ADVERSARIAL METRICS
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


def plot_uncertainty_analysis(df, exp_id):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    t = df['elapsed']
    sigma = np.sqrt(np.exp(df['o2_logvar']))
    ax1.plot(t, df['oxygen'], 'k-', label='Ground Truth', lw=2)
    ax1.plot(t, df['o2_pred'], 'b--', label='AI Prediction', lw=1.5)
    ax1.fill_between(t, df['o2_pred'] - 2 * sigma, df['o2_pred'] + 2 * sigma, color='blue', alpha=0.2,
                     label='95% Uncertainty')
    ax1.set_title(f"Uncertainty Quantification: {exp_id}")
    ax2.plot(t, sigma, color='orange', lw=2)
    ax2.set_ylabel("Predicted Sigma")
    for (s, e) in ANALYSIS_INTERVALS_S:
        ax1.axvspan(s, e, color='green', alpha=0.05)
        ax2.axvspan(s, e, color='green', alpha=0.05)
    plt.savefig(os.path.join(OUTPUT_DIR, "5_Uncertainty_Analysis", f"Uncertainty_{exp_id}.png"))
    plt.close()
    df[['elapsed', 'oxygen', 'o2_pred', 'o2_logvar']].to_csv(
        os.path.join(OUTPUT_DIR, "5_Uncertainty_Analysis", f"Uncertainty_Data_{exp_id}.csv"), index=False)


def plot_multitask_diagnostics(df, exp_id):
    y_true_phase = df['phase_label']
    y_pred_phase = df['phase_pred']
    cm = confusion_matrix(y_true_phase, y_pred_phase)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Transient', 'Stable'],
                yticklabels=['Transient', 'Stable'], ax=ax)
    ax.set_title(f"Phase Classification Confusion: {exp_id}")
    plt.savefig(os.path.join(OUTPUT_DIR, "6_MultiTask_Diagnostics", f"Phase_CM_{exp_id}.png"))
    plt.close()
    report = classification_report(y_true_phase, y_pred_phase, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(
        os.path.join(OUTPUT_DIR, "6_MultiTask_Diagnostics", f"Phase_Metrics_{exp_id}.csv"))


def plot_training_dashboard(history, exp_id):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    ax1.plot(history['epoch'], history['loss_total'], label='Total Loss', color='black')
    ax1.plot(history['epoch'], history['loss_reg'], label='Reg Loss', color='blue', linestyle='--')
    ax1.plot(history['epoch'], history['loss_phase'], label='Phase Loss', color='green', linestyle=':')
    ax1.plot(history['epoch'], history['loss_domain'], label='Domain Loss', color='red', linestyle=':')
    ax1.legend()
    ax2.plot(history['epoch'], history['val_mae'], label='O2 MAE', color='blue', marker='o')
    ax2_r = ax2.twinx()
    ax2_r.plot(history['epoch'], history['domain_acc'], label='Domain Acc', color='red', marker='x')
    ax2.legend(loc='upper left');
    ax2_r.legend(loc='upper right')
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
# 6. EXECUTION ORCHESTRATOR
# ==============================================================================
def main():
    experiments = find_experiment_files_v10(EXPERIMENTS_ROOT_DIR)
    if N_EXPERIMENTS_CAP: experiments = experiments[:N_EXPERIMENTS_CAP]

    # Load Data (Cached)
    df = get_master_df(experiments, os.path.join(FEATURE_CACHE_DIR, "idc_master_v10.9.1.parquet"))

    exp_list = df['experiment_id'].unique()
    exp_map = {name: i for i, name in enumerate(exp_list)}
    num_domains = len(exp_list)

    print(f"Loaded {len(df)} frames across {num_domains} experiments.")

    # RESUME LOGIC: Check which experiments are already in the Summary CSV
    summary_path = os.path.join(OUTPUT_DIR, "Global_Performance_Summary.csv")
    completed_exps = []
    if os.path.exists(summary_path):
        try:
            existing_df = pd.read_csv(summary_path)
            completed_exps = existing_df['experiment_id'].unique().tolist()
            print(f"Found existing summary with {len(completed_exps)} completed experiments. Resuming...")
        except:
            print("Summary file unreadable. Starting fresh.")
    else:
        # Create empty CSV with headers if it doesn't exist
        headers = ['experiment_id', 'MAE_AI', 'MAE_Unstruct', 'Phase_Accuracy', 'Final_Domain_Leakage_Acc']
        pd.DataFrame(columns=headers).to_csv(summary_path, index=False)

    for test_idx, test_day in enumerate(exp_list):
        if test_day in completed_exps:
            print(f"Skipping {test_day} (Already Completed).")
            continue

        print(f"\n========================================================")
        print(f"Processing Fold {test_idx + 1}/{len(exp_list)}: {test_day}")
        print(f"========================================================")

        train_df = df[df['experiment_id'] != test_day]
        test_df = df[df['experiment_id'] == test_day].copy()

        test_df['o2_classical'] = fit_classical_baseline(train_df, test_df)

        train_loader = DataLoader(IDC_TemporalDataset(train_df, exp_map), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(IDC_TemporalDataset(test_df, exp_map), batch_size=BATCH_SIZE, shuffle=False)

        model = TViT_MultiTask_IDC(num_domains, stack_size=N_STACKED_FRAMES).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LR)
        criterion_ce = nn.CrossEntropyLoss()

        history = {'epoch': [], 'loss_total': [], 'loss_reg': [], 'loss_phase': [], 'loss_domain': [], 'val_mae': [],
                   'domain_acc': []}

        for epoch in range(N_EPOCHS):
            model.train()
            l_total_sum = 0
            l_reg_sum = 0
            l_phase_sum = 0
            l_dom_sum = 0

            for x, y_o2, y_domain, y_phase in train_loader:
                x, y_o2, y_domain, y_phase = x.to(DEVICE), y_o2.to(DEVICE), y_domain.to(DEVICE), y_phase.to(DEVICE)
                optimizer.zero_grad()
                mu, log_var, phase_logits, domain_logits = model(x)

                precision = torch.exp(-log_var)
                loss_reg = torch.mean(0.5 * precision * (y_o2 - mu) ** 2 + 0.5 * log_var)
                loss_phase = criterion_ce(phase_logits, y_phase)
                loss_domain = criterion_ce(domain_logits, y_domain)

                loss = loss_reg + (LAMBDA_PHASE * loss_phase) + (LAMBDA_DOMAIN * loss_domain)
                loss.backward()
                optimizer.step()

                l_total_sum += loss.item()
                l_reg_sum += loss_reg.item()
                l_phase_sum += loss_phase.item()
                l_dom_sum += loss_domain.item()

            model.eval()
            val_preds_o2 = []
            val_preds_dom = []
            val_targets_dom = []

            with torch.no_grad():
                for x, y_o2, y_domain, y_phase in val_loader:
                    x = x.to(DEVICE)
                    mu, _, _, domain_logits = model(x)
                    val_preds_o2.extend(mu.cpu().numpy())
                    val_preds_dom.extend(torch.argmax(domain_logits, dim=1).cpu().numpy())
                    val_targets_dom.extend(y_domain.numpy())

            current_mae = mean_absolute_error(test_df['oxygen'], val_preds_o2)
            dom_acc = accuracy_score(val_targets_dom, val_preds_dom)

            history['epoch'].append(epoch)
            history['loss_total'].append(l_total_sum / len(train_loader))
            history['loss_reg'].append(l_reg_sum / len(train_loader))
            history['loss_phase'].append(l_phase_sum / len(train_loader))
            history['loss_domain'].append(l_dom_sum / len(train_loader))
            history['val_mae'].append(current_mae)
            history['domain_acc'].append(dom_acc)

        plot_training_dashboard(history, test_day)

        # SAVE MODEL CHECKPOINT (NEW)
        torch.save(model.state_dict(), os.path.join(MODEL_CHECKPOINT_DIR, f"model_fold_{test_day}.pth"))

        model.eval()
        o2_preds = []
        o2_logvars = []
        phase_preds = []

        with torch.no_grad():
            for x, _, _, _ in val_loader:
                x = x.to(DEVICE)
                mu, log_var, phase_logits, _ = model(x)
                o2_preds.extend(mu.cpu().numpy())
                o2_logvars.extend(log_var.cpu().numpy())
                phase_preds.extend(torch.argmax(phase_logits, dim=1).cpu().numpy())

        test_df['o2_pred'] = o2_preds
        test_df['o2_logvar'] = o2_logvars
        test_df['phase_pred'] = phase_preds

        analyze_dynamics_v10(test_df, test_day)
        plot_chemometrics(test_df, test_day)
        plot_uncertainty_analysis(test_df, test_day)
        plot_multitask_diagnostics(test_df, test_day)

        # INCREMENTAL METRICS SAVING (NEW)
        unstruct_mask = test_df['elapsed'] < UNSTRUCTURED_CUTOFF_S
        metrics_dict = {
            'experiment_id': test_day,
            'MAE_AI': mean_absolute_error(test_df['oxygen'], test_df['o2_pred']),
            'MAE_Unstruct': mean_absolute_error(test_df[unstruct_mask]['oxygen'], test_df[unstruct_mask]['o2_pred']),
            'Phase_Accuracy': accuracy_score(test_df['phase_label'], test_df['phase_pred']),
            'Final_Domain_Leakage_Acc': history['domain_acc'][-1]
        }

        # Append single row to CSV immediately
        pd.DataFrame([metrics_dict]).to_csv(summary_path, mode='a', header=False, index=False)

    print("Execution Finished. See OUTPUT_DIR for high-impact results.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("CRITICAL FAILURE:")
        traceback.print_exc()
        raise
