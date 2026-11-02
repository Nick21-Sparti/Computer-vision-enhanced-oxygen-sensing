# Repository edition: see docs/CHANGES.md and docs/REPRODUCIBILITY.md.
"""
==============================================================================
PROJECT: PINN-IDC (Inverse Dynamic Compensation) Framework
VERSION: 11.0 - High-Impact Journal Edition (Validation & XAI)
==============================================================================

--- SCIENTIFIC CONTEXT & GOAL ---
The fundamental challenge is the "Response Time Problem" in chemical sensing.
Diffusion creates a physical "lag" where the sensor reading trails the true
concentration. We use a Temporal Vision Transformer (TViT) constrained by
a Physics-Informed Loss (PINN) to invert this diffusion.

To satisfy high-impact publication standards, we must demonstrate:
1. Generalization (Validation Curves).
2. Precision (Parity Plots & R^2).
3. Unbiasedness (Residual Histograms).
4. Physical Interpretability (Evolution of Mass Transfer Coefficients).

--- EVOLUTION OF THE FRAMEWORK (V1.0 -> V11.0) ---
- V1-V8: Basics & Data Discovery.
- V10.0-10.8: Dynamic Sensing, Classical Baselines, Black Box Logic.
- V10.9: Introduction of PINN Loss.
- V11.0 (CURRENT):
    1. VALIDATION LOOP: Calculates Test MAE at every epoch to prove
       convergence without overfitting.
    2. PARITY PLOTS: Generates Predicted vs GT scatter plots (The "Gold Standard"
       chemometric visualization).
    3. RESIDUAL ANALYSIS: Plots error histograms to check for systematic bias.
    4. PHYSICS DASHBOARD: Tracks the evolution of 'k' (Diffusion) and
       'alpha' (Stern-Volmer) alongside the Loss components.
    5. MECHANISM SUBPLOT: Time-series plots now show the calculated dS/dt,
       visually explaining HOW the physics term corrects the lag.

--- FOLDER STRUCTURE & OUTPUTS ---
- Dynamic_IDC_Analysis_V11/
    - 0_Preflight_Diagnostics/
    - 1_Training_Dynamics/ (Loss Curves, Parameter Evolution, Validation MAE)
    - 2_Inference_Results/ (Time Series with Physics Subplots)
    - 3_Chemometrics/ (Parity Plots, Residual Histograms)
    - 4_Response_Time_Reports/
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

import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import medfilt
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
from torchvision import transforms
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

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
BASE_PROJECT_DIR = data_path('aqueous')
EXPERIMENTS_ROOT_DIR = BASE_PROJECT_DIR

# --- MANUAL INTERVAL CONFIGURATION ---
ANALYSIS_INTERVALS_S = [
    (1450, 1650),
    (2300, 2500),
    (3100, 3300),
    (3900, 4100),
    (4700, 4900)
]

# Time split for "Structured" vs "Unstructured" analysis
UNSTRUCTURED_CUTOFF_S = 1000

# Cap number of experiments (None = All)
N_EXPERIMENTS_CAP = None

# PINN Configuration
PINN_PHYSICS_WEIGHT = 0.5  # Weight (Lambda) for the Physics Residual Term

# Output Directory Hierarchy
OUTPUT_DIR = output_path('aqueous_tvit_mass_transfer')
CSV_EXPORT_DIR = os.path.join(OUTPUT_DIR, "csv_exports")
FEATURE_CACHE_DIR = cache_path('aqueous_tvit_mass_transfer', "features")
CV_CACHE_DIR = cache_path('aqueous_tvit_mass_transfer', "folds")

# Hyperparameters
N_STACKED_FRAMES = 3
FRAME_STACK_STEP = 10
RESIZE_DIM = (224, 224)
STORAGE_DIM = (48, 48)
BATCH_SIZE = 32
N_EPOCHS = 20
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Folder Structure
directories_to_make = [
    OUTPUT_DIR,
    CSV_EXPORT_DIR,
    FEATURE_CACHE_DIR,
    CV_CACHE_DIR,
    os.path.join(OUTPUT_DIR, "0_Preflight_Diagnostics"),
    os.path.join(OUTPUT_DIR, "1_Training_Dynamics"),
    os.path.join(OUTPUT_DIR, "2_Inference_Results"),
    os.path.join(OUTPUT_DIR, "3_Chemometrics"),
    os.path.join(OUTPUT_DIR, "4_Response_Time_Reports")
]

for d in directories_to_make:
    os.makedirs(d, exist_ok=True)


# ==============================================================================
# 3. ROBUST DATA DISCOVERY
# ==============================================================================

def find_experiment_files_v10(root_dir):
    """Scans for valid experiment folders (V8.2 Logic)."""
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

    print(f"Found {len(valid_exps)} complete experiments.")
    return valid_exps


def parse_logs(exp):
    """Parses and synchronizes logs."""
    o2_data = []
    try:
        f = open(exp['arduino_path'], 'r', encoding='utf-8', errors='ignore')
        with f:
            for line in f:
                if "MEA" in line:
                    p = line.split()
                    try:
                        ts = datetime.strptime(f"{p[0]} {p[1]}", '%Y-%m-%d %H:%M:%S')
                        o2_val = float(p[p.index("MEA") + 5]) / 1000.0
                        o2_data.append([ts, o2_val])
                    except: continue
    except: pass
    df_o2 = pd.DataFrame(o2_data, columns=['timestamp', 'oxygen']).sort_values('timestamp')

    try:
        df_temp = pd.read_csv(exp['temp_path']).iloc[:, :2]
        df_temp.columns = ['timestamp', 'temp']
        df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'])
    except:
        df_temp = pd.DataFrame({'timestamp': df_o2['timestamp'], 'temp': 20.0})

    df = pd.merge_asof(df_o2, df_temp.sort_values('timestamp'), on='timestamp', direction='nearest')
    df['oxygen'] = df['oxygen'].interpolate().fillna(method='bfill').fillna(method='ffill')
    df['temp'] = df['temp'].interpolate().fillna(method='bfill').fillna(method='ffill')
    return df


# ==============================================================================
# 4. FEATURE EXTRACTION & CACHING
# ==============================================================================

def get_master_df(experiments, cache_path):
    """Extracts features and caches to Parquet."""
    if os.path.exists(cache_path):
        print(f"--- Cache Found: Loading {cache_path} ---")
        return pd.read_parquet(cache_path)

    print("--- Feature Extraction: Processing Video Pixels and Proxies ---")
    all_frames = []

    for exp in tqdm(experiments, desc="Processing Videos"):
        try:
            df_logs = parse_logs(exp)
            if df_logs.empty: continue

            cap = cv2.VideoCapture(exp['video_path'])
            fps = cap.get(cv2.CAP_PROP_FPS)
            match = re.search(r'_(\d{8}_\d{6})', os.path.basename(exp['video_path']))
            start_ts = datetime.strptime(match.group(1), '%Y%m%d_%H%M%S') if match else df_logs['timestamp'].iloc[0]

            total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = int(max(1, fps / 2)) # 2 FPS

            for f_idx in range(0, total_f, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if not ret: break

                elapsed = f_idx / fps
                ts = start_ts + timedelta(seconds=elapsed)

                closest_idx = df_logs['timestamp'].searchsorted(ts)
                if closest_idx >= len(df_logs): closest_idx = len(df_logs) - 1

                gt_o2 = df_logs.iloc[closest_idx]['oxygen']
                gt_temp = df_logs.iloc[closest_idx]['temp']

                # Resize and Extract Red Channel
                resized = cv2.resize(frame, STORAGE_DIM)
                red = resized[:, :, 2].astype(float)

                mean_i = np.mean(red)
                std_i = np.std(red)

                all_frames.append({
                    'experiment_id': exp['id'],
                    'timestamp': ts,
                    'elapsed': elapsed,
                    'oxygen': gt_o2,
                    'temp': gt_temp,
                    'inv_i': 1.0 / (mean_i + 1e-6),
                    'inv_std': 1.0 / (std_i + 1e-6),
                    'pixels': resized.flatten().astype(np.uint8)
                })
            cap.release()
        except Exception as e:
            print(f"Skipping {exp['id']}: {e}")

    master_df = pd.DataFrame(all_frames)
    master_df.to_parquet(cache_path)
    return master_df


# ==============================================================================
# 5. DATASET & TViT MODEL
# ==============================================================================

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
            lookback = idx - (i * self.step)
            if lookback < 0 or self.df.iloc[lookback]['experiment_id'] != exp_id:
                lookback = idx
            pix = self.df.iloc[lookback]['pixels'].reshape(STORAGE_DIM[0], STORAGE_DIM[1], 3)
            img_t = torch.from_numpy(pix).permute(2, 0, 1).float() / 255.0
            frames.append(self.resize(img_t))

        x = torch.cat(frames, dim=0)
        y = torch.tensor(row['oxygen'], dtype=torch.float32)
        e = torch.tensor(self.exp_id_map[exp_id], dtype=torch.long)
        return x, y, e

class TViT_Decoupler(nn.Module):
    def __init__(self, num_exps, stack_size=3):
        super().__init__()
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0)
        orig_proj = self.vit.patch_embed.proj
        self.vit.patch_embed.proj = nn.Conv2d(3 * stack_size, orig_proj.out_channels,
                                              kernel_size=orig_proj.kernel_size,
                                              stride=orig_proj.stride, padding=orig_proj.padding)
        with torch.no_grad():
            self.vit.patch_embed.proj.weight = nn.Parameter(
                orig_proj.weight.repeat(1, stack_size, 1, 1) / stack_size)
            self.vit.patch_embed.proj.bias = orig_proj.bias

        self.head = nn.Sequential(
            nn.Linear(self.vit.embed_dim, 128), nn.GELU(), nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.head(self.vit.forward_features(x)[:, 0]).squeeze(-1)


# ==============================================================================
# 6. PHYSICS-INFORMED LOSS MODULE
# ==============================================================================

class SternVolmer_MassTransfer_Loss(nn.Module):
    """
    PINN Loss: dS/dt = k * (O_ext - S)
    """
    def __init__(self, lambda_phy=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_phy = lambda_phy

        # Learnable Physics Parameters
        self.k = nn.Parameter(torch.tensor(0.2)) # Diffusion Rate
        self.alpha = nn.Parameter(torch.tensor(1000.0)) # Stern-Volmer Scale

    def forward(self, pred_o2_ext, input_stack, gt_o2):
        # 1. Data Loss
        loss_data = self.mse(pred_o2_ext, gt_o2)

        # 2. Physics Extraction
        center = slice(80, 144)
        i_t = torch.mean(input_stack[:, 0, center, center], dim=[1,2])
        i_prev = torch.mean(input_stack[:, 3, center, center], dim=[1,2])

        s_t = 1.0 / (i_t + 1e-6)
        s_prev = 1.0 / (i_prev + 1e-6)
        ds_dt = s_t - s_prev

        # 3. Physics Residual
        # Inferred O_ext = alpha*S + (alpha/k)*dS/dt
        phy_inferred_o2 = (self.alpha * s_t) + (self.alpha / (self.k + 1e-6)) * ds_dt
        loss_phy = self.mse(pred_o2_ext, phy_inferred_o2)

        total_loss = loss_data + (self.lambda_phy * loss_phy)
        return total_loss, loss_data, loss_phy, ds_dt

# ==============================================================================
# 7. ANALYTICS & VISUALIZATION (HIGH IMPACT SUITE)
# ==============================================================================

def fit_classical_baseline(train_df, test_df):
    """Fits 2nd-degree polynomial on Stable Intervals."""
    stable_train = pd.DataFrame()
    for (start, end) in ANALYSIS_INTERVALS_S:
        chunk = train_df[(train_df['elapsed'] >= start) & (train_df['elapsed'] <= end)]
        stable_train = pd.concat([stable_train, chunk])
    if stable_train.empty: stable_train = train_df

    poly_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    poly_model.fit(stable_train[['inv_i']].values, stable_train['oxygen'].values)
    return poly_model.predict(test_df[['inv_i']].values)

def analyze_dynamics_v11(df, exp_id, phase="Inference", ds_dt_series=None):
    """
    Main Time-Series Plot.
    V11 Additions: 'Mechanism' Subplot showing dS/dt (The Physics Trigger).
    """
    plot_df = df.copy()
    def norm(s): return (s - s.min()) / (s.max() - s.min() + 1e-8)

    # Create 2 Subplots (Main Signal, Physics Derivative)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    t = plot_df['elapsed']

    # --- PLOT 1: SIGNALS ---
    ax1.plot(t, norm(plot_df['oxygen']), label='Ground Truth', color='black', lw=3, zorder=10)
    ax1.plot(t, norm(plot_df['inv_i']), label='Raw Physics (1/I)', color='red', alpha=0.3, zorder=1)

    if 'o2_classical' in plot_df.columns:
        ax1.plot(t, norm(plot_df['o2_classical']), label='Classical Calibration', color='gray', linestyle='--', lw=2, alpha=0.8)

    if phase == "Inference" and 'o2_pred' in plot_df.columns:
        plot_df['pred_smooth'] = medfilt(plot_df['o2_pred'], 5)
        ax1.plot(t, norm(plot_df['pred_smooth']), label='PINN-IDC Prediction', color='blue', lw=2.5, zorder=8)

    # Zones
    for i, (start, end) in enumerate(ANALYSIS_INTERVALS_S):
        if start < t.max():
            ax1.axvspan(start, min(end, t.max()), color='green', alpha=0.1)
        if i < len(ANALYSIS_INTERVALS_S) - 1:
            trans_start = end
            trans_end = ANALYSIS_INTERVALS_S[i+1][0]
            if trans_start < t.max():
                ax1.axvspan(trans_start, min(trans_end, t.max()), color='red', alpha=0.05)

    ax1.set_title(f"PINN-IDC Analysis: {exp_id} ({phase})", fontsize=14)
    ax1.legend(loc='upper right')
    ax1.set_ylabel("Normalized Concentration")

    # --- PLOT 2: MECHANISM (dS/dt) ---
    # We estimate dS/dt from the dataframe if not provided directly
    if ds_dt_series is None:
        # Approximate from inv_i
        ds_dt_approx = np.gradient(plot_df['inv_i'])
    else:
        ds_dt_approx = ds_dt_series

    ax2.plot(t, ds_dt_approx, color='purple', lw=1.5, label='Physics Term (dS/dt)')
    ax2.axhline(0, color='black', lw=1, linestyle=':')
    ax2.set_ylabel("Rate of Change (dS/dt)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Mechanism: The Physics Derivative (Acceleration of Diffusion)")
    ax2.legend(loc='upper right')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "0_Preflight_Diagnostics" if phase == "Preflight" else "2_Inference_Results", f"IDC_{phase}_{exp_id}.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # CSV Export
    csv_cols = ['elapsed', 'oxygen', 'inv_i']
    if 'o2_pred' in plot_df.columns: csv_cols.append('o2_pred')
    plot_df[csv_cols].to_csv(os.path.join(CSV_EXPORT_DIR, f"Data_{phase}_{exp_id}.csv"), index=False)

def plot_chemometrics(df, exp_id):
    """Generates Parity Plot and Residual Histogram."""
    if 'o2_pred' not in df.columns: return

    y_true = df['oxygen']
    y_pred = df['o2_pred']

    # 1. Parity Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.5, s=10, c='blue')

    # 45-degree line
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'k--', lw=2, label='Ideal')

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    ax.set_title(f"Parity Plot: {exp_id}\nR²={r2:.4f}, MAE={mae:.4f}")
    ax.set_xlabel("Ground Truth Oxygen")
    ax.set_ylabel("PINN Predicted Oxygen")
    ax.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_Chemometrics", f"Parity_{exp_id}.png"))
    plt.close()

    # 2. Residual Histogram
    residuals = y_pred - y_true
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=50, kde=True, color='purple')
    plt.axvline(0, color='k', linestyle='--')
    plt.title(f"Residual Distribution: {exp_id}")
    plt.xlabel("Prediction Error (Predicted - GT)")
    plt.savefig(os.path.join(OUTPUT_DIR, "3_Chemometrics", f"Residuals_{exp_id}.png"))
    plt.close()

def plot_training_dashboard(history, exp_id):
    """
    V11 Dashboard:
    1. Loss Curves (Train vs Validation MAE)
    2. Physics Parameters (k, alpha)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Losses
    ax1.plot(history['epoch'], history['data_loss'], label='Train Data Loss (MSE)', color='blue')
    ax1.plot(history['epoch'], history['phy_loss'], label='Physics Residual', color='green', linestyle=':')
    # Create secondary axis for Test MAE (Scale might differ)
    ax1b = ax1.twinx()
    ax1b.plot(history['epoch'], history['val_mae'], label='Validation MAE', color='red', lw=2)
    ax1b.set_ylabel("Validation MAE (umol/L)")

    ax1.set_title(f"Learning Dynamics: {exp_id}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training MSE")
    ax1.set_yscale('log')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper center')

    # Panel 2: Physics Parameters
    ax2.plot(history['epoch'], history['k_val'], label='k (Diffusion)', color='purple', marker='x')
    # Twin axis for Alpha (Different scale)
    ax2b = ax2.twinx()
    ax2b.plot(history['epoch'], history['alpha_val'], label='alpha (Stern-Volmer)', color='orange', marker='o')

    ax2.set_title("Physics Parameter Discovery")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Mass Transfer Coeff (k)")
    ax2b.set_ylabel("Scaling Factor (alpha)")

    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='center right')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_Training_Dynamics", f"Dashboard_{exp_id}.png"))
    plt.close()

    pd.DataFrame(history).to_csv(os.path.join(OUTPUT_DIR, "1_Training_Dynamics", f"History_{exp_id}.csv"), index=False)


# ==============================================================================
# 9. EXECUTION ORCHESTRATOR
# ==============================================================================

def main():
    print("--- PINN-IDC V11.0: High-Impact Validation Suite ---")

    # 1. Discovery
    experiments = find_experiment_files_v10(EXPERIMENTS_ROOT_DIR)
    if N_EXPERIMENTS_CAP: experiments = experiments[:N_EXPERIMENTS_CAP]
    if not experiments: sys.exit("No experiments found.")

    # 2. Caching
    cache_suffix = f"_n{N_EXPERIMENTS_CAP}" if N_EXPERIMENTS_CAP else "_all"
    master_cache = os.path.join(FEATURE_CACHE_DIR, f"idc_master_v10{cache_suffix}.parquet")
    df = get_master_df(experiments, master_cache)

    exp_list = df['experiment_id'].unique()
    exp_map = {name: i for i, name in enumerate(exp_list)}

    # 3. Preflight
    print("\n--- Running Global Preflight Diagnostics ---")
    for exp_id in tqdm(exp_list, desc="Generating Preflight Plots"):
        analyze_dynamics_v11(df[df['experiment_id'] == exp_id], exp_id, phase="Preflight")

    # 4. PINN LOOCV Loop
    global_metrics = []
    print(f"\n--- Starting PINN Cross-Validation Loop ({len(exp_list)} Folds) ---")

    for test_idx, test_day in enumerate(exp_list):
        print(f"\n[{test_idx+1}/{len(exp_list)}] Processing Target: {test_day}")

        # A. Split Data
        train_df = df[df['experiment_id'] != test_day]
        test_df = df[df['experiment_id'] == test_day]

        train_ds = IDC_TemporalDataset(train_df, exp_map, stack_size=N_STACKED_FRAMES, step=FRAME_STACK_STEP)
        test_ds = IDC_TemporalDataset(test_df, exp_map, stack_size=N_STACKED_FRAMES, step=FRAME_STACK_STEP)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        # Validation Loader (Same as Test in LOOCV)
        val_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        # B. Classical Baseline
        test_df['o2_classical'] = fit_classical_baseline(train_df, test_df)

        # C. Init Model
        model = TViT_Decoupler(len(exp_list), stack_size=N_STACKED_FRAMES).to(DEVICE)
        pinn_criterion = SternVolmer_MassTransfer_Loss(lambda_phy=PINN_PHYSICS_WEIGHT).to(DEVICE)
        optimizer = optim.AdamW(list(model.parameters()) + list(pinn_criterion.parameters()), lr=LR)
        model_path = os.path.join(CV_CACHE_DIR, f"pinn_v11_target_{test_day}.pth")

        # D. Training with Validation Loop
        if os.path.exists(model_path):
            print(f"   > Loading cached weights: {os.path.basename(model_path)}")
            model.load_state_dict(torch.load(model_path))
        else:
            print(f"   > Training PINN (Epochs: {N_EPOCHS})...")
            history = {k: [] for k in ['epoch', 'data_loss', 'phy_loss', 'val_mae', 'k_val', 'alpha_val']}

            for epoch in range(N_EPOCHS):
                # 1. Train
                model.train()
                d_loss, p_loss = 0, 0
                for x, y, _ in train_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    optimizer.zero_grad()
                    pred = model(x)
                    loss, ld, lp, _ = pinn_criterion(pred, x, y)
                    loss.backward()
                    optimizer.step()
                    d_loss += ld.item(); p_loss += lp.item()

                # 2. Validate (Calculate Test MAE)
                model.eval()
                val_preds = []
                val_gt = []
                with torch.no_grad():
                    for x, y, _ in val_loader:
                        val_preds.extend(model(x.to(DEVICE)).cpu().numpy())
                        val_gt.extend(y.numpy())
                val_mae = mean_absolute_error(val_gt, val_preds)

                # 3. Log
                n = len(train_loader)
                history['epoch'].append(epoch)
                history['data_loss'].append(d_loss/n)
                history['phy_loss'].append(p_loss/n)
                history['val_mae'].append(val_mae)
                history['k_val'].append(pinn_criterion.k.item())
                history['alpha_val'].append(pinn_criterion.alpha.item())

            torch.save(model.state_dict(), model_path)
            plot_training_dashboard(history, test_day)
            print(f"   > Learned k: {pinn_criterion.k.item():.4f} | Final Val MAE: {history['val_mae'][-1]:.3f}")

        # E. Final Inference
        model.eval()
        preds = []
        with torch.no_grad():
            for x, _, _ in val_loader:
                preds.extend(model(x.to(DEVICE)).cpu().numpy())
        test_df['o2_pred'] = preds

        # F. High Impact Visuals
        analyze_dynamics_v11(test_df, test_day, phase="Inference")
        plot_chemometrics(test_df, test_day)

        # G. Metrics
        unstruct_mask = test_df['elapsed'] < UNSTRUCTURED_CUTOFF_S
        def calc_mae(mask, col): return mean_absolute_error(test_df[mask]['oxygen'], test_df[mask][col]) if isinstance(mask, slice) or mask.any() else np.nan

        metrics = {
            'experiment_id': test_day,
            'MAE_AI_Total': calc_mae(slice(None), 'o2_pred'),
            'MAE_AI_Unstruct': calc_mae(unstruct_mask, 'o2_pred'),
            'Learned_k': getattr(pinn_criterion, 'k', torch.tensor(0.0)).item()
        }
        global_metrics.append(metrics)

    # 5. Global Summary
    summary_df = pd.DataFrame(global_metrics)
    summary_path = os.path.join(OUTPUT_DIR, "Global_Performance_Summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n--- Final Aggregation ---")
    print(summary_df[['experiment_id', 'MAE_AI_Total', 'MAE_AI_Unstruct', 'Learned_k']].to_string())
    print(f"Avg Unstructured MAE: {summary_df['MAE_AI_Unstruct'].mean():.4f}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
