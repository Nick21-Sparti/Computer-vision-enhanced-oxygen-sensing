# Repository edition: see docs/CHANGES.md and docs/REPRODUCIBILITY.md.
"""
==============================================================================
PROJECT: PINN-IDC (Inverse Dynamic Compensation) Framework
VERSION: 10.9.0 - Physically Informed Classical Baseline (Stern-Volmer)
==============================================================================

--- SCIENTIFIC CONTEXT & GOAL ---
The fundamental challenge is the "Response Time Problem" in chemical sensing.
Diffusion creates a physical "lag" where the sensor reading trails the true
concentration. We use a Temporal Vision Transformer (TViT) to invert this
diffusion and "predict the present."

A critical scientific concern is "Overfitting to Structure." Our datasets
often follow a set calibration schedule (0%, 5%, 10%...). We must ensure the
AI is learning the *Inverse Diffusion Equation*, not just memorizing the
timeline of the pump.

--- EVOLUTION OF THE FRAMEWORK (V1.0 -> V10.9.0) ---
- V1-V7: Static PINNs.
- V8.2: Robust file discovery.
- V10.0-10.6: Dynamic sensing, T90 Benchmarking, LOOCV.
- V10.7: Added Classical Baseline & Segmented Analytics.
- V10.8: Strict "Black Box" Training (Final Epoch Saving).
- V10.8.1: Chemometrics Suite (Parity Plots & Residual Histograms).
- V10.9.0 (CURRENT):
    1. STERN-VOLMER TWO-SITE MODEL: Replaced the generic polynomial baseline
       with the proper physical equation for heterogeneous polymer sensors.
       The code now fits I0, f, K1, and K2, and analytically inverts the
       quadratic relationship to predict Oxygen from Intensity.
    2. ADDED VALUE: This provides the most rigorous possible "Classical"
       benchmark. If the AI beats *this*, it is beating the physics of the
       steady-state sensor, proving it is compensating for dynamics (diffusion),
       not just calibration non-linearity.

--- FOLDER STRUCTURE & OUTPUTS ---
- Dynamic_IDC_Analysis_V10/
    - 0_Preflight_Diagnostics/
    - 1_Training_Convergence/
    - 2_Inference_Results/
    - 3_Chemometrics/
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
from scipy.optimize import curve_fit  # REQUIRED for Stern-Volmer fitting
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import torch
import torch.nn as nn
import torch.optim as optim
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
BASE_PROJECT_DIR = data_path('aqueous')
EXPERIMENTS_ROOT_DIR = BASE_PROJECT_DIR

# Calibration steps usually happen in these windows. Used for Static Baseline fitting.
ANALYSIS_INTERVALS_S = [(1450, 1650), (2300, 2500), (3100, 3300), (3900, 4100), (4700, 4900)]
UNSTRUCTURED_CUTOFF_S = 1000
N_EXPERIMENTS_CAP = None

OUTPUT_DIR = output_path('aqueous_tvit')
CSV_EXPORT_DIR = os.path.join(OUTPUT_DIR, "csv_exports")
FEATURE_CACHE_DIR = cache_path('aqueous_tvit', "features")
CV_CACHE_DIR = cache_path('aqueous_tvit', "folds")

N_STACKED_FRAMES = 3
FRAME_STACK_STEP = 10
RESIZE_DIM = (224, 224)
STORAGE_DIM = (48, 48)
BATCH_SIZE = 32
N_EPOCHS = 20
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

directories_to_make = [
    OUTPUT_DIR, CSV_EXPORT_DIR, FEATURE_CACHE_DIR, CV_CACHE_DIR,
    os.path.join(OUTPUT_DIR, "0_Preflight_Diagnostics"),
    os.path.join(OUTPUT_DIR, "1_Training_Convergence"),
    os.path.join(OUTPUT_DIR, "2_Inference_Results"),
    os.path.join(OUTPUT_DIR, "3_Chemometrics"),
    os.path.join(OUTPUT_DIR, "4_Response_Time_Reports")
]
for d in directories_to_make: os.makedirs(d, exist_ok=True)

# ==============================================================================
# 3. ROBUST DATA DISCOVERY & PARSING
# ==============================================================================
def find_experiment_files_v10(root_dir):
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
                valid_exps.append({"id": exp_id, "video_path": vids[0], "arduino_path": arduino_logs[0], "temp_path": temp_logs[0]})
    return valid_exps

def parse_logs(exp):
    o2_data = []
    try:
        with open(exp['arduino_path'], 'r', encoding='utf-8', errors='ignore') as f:
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
    except: df_temp = pd.DataFrame({'timestamp': df_o2['timestamp'], 'temp': 20.0})
    df = pd.merge_asof(df_o2, df_temp.sort_values('timestamp'), on='timestamp', direction='nearest')
    df['oxygen'] = df['oxygen'].interpolate().fillna(method='bfill').fillna(method='ffill')
    df['temp'] = df['temp'].interpolate().fillna(method='bfill').fillna(method='ffill')
    return df

def get_master_df(experiments, cache_path):
    if os.path.exists(cache_path): return pd.read_parquet(cache_path)
    all_frames = []
    for exp in tqdm(experiments, desc="Processing Videos"):
        df_logs = parse_logs(exp)
        if df_logs.empty: continue
        cap = cv2.VideoCapture(exp['video_path'])
        fps = cap.get(cv2.CAP_PROP_FPS)
        match = re.search(r'_(\d{8}_\d{6})', os.path.basename(exp['video_path']))
        start_ts = datetime.strptime(match.group(1), '%Y%m%d_%H%M%S') if match else df_logs['timestamp'].iloc[0]
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = int(max(1, fps / 2))
        for f_idx in range(0, total_f, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: break
            ts = start_ts + timedelta(seconds=f_idx / fps)
            idx = df_logs['timestamp'].searchsorted(ts)
            if idx >= len(df_logs): idx = len(df_logs) - 1
            resized = cv2.resize(frame, STORAGE_DIM)

            # NOTE: We store 'inv_i' which is 1/Intensity.
            # For classical fitting, we will convert this back to Intensity (I).
            intensity = np.mean(resized[:, :, 2].astype(float)) + 1e-6

            all_frames.append({
                'experiment_id': exp['id'], 'timestamp': ts, 'elapsed': f_idx / fps,
                'oxygen': df_logs.iloc[idx]['oxygen'], 'temp': df_logs.iloc[idx]['temp'],
                'inv_i': 1.0 / intensity,
                'inv_std': 1.0 / (np.std(resized[:, :, 2].astype(float)) + 1e-6),
                'pixels': resized.flatten().astype(np.uint8)
            })
        cap.release()
    master_df = pd.DataFrame(all_frames)
    master_df.to_parquet(cache_path)
    return master_df

class IDC_TemporalDataset(Dataset):
    def __init__(self, df, exp_id_map, stack_size=3, step=5):
        self.df = df.reset_index(drop=True)
        self.exp_id_map = exp_id_map
        self.stack_size = stack_size
        self.step = step
        self.resize = transforms.Resize(RESIZE_DIM)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        exp_id = row['experiment_id']
        frames = []
        for i in range(self.stack_size):
            lookback = max(0, idx - (i * self.step))
            if self.df.iloc[lookback]['experiment_id'] != exp_id: lookback = idx
            pix = self.df.iloc[lookback]['pixels'].reshape(STORAGE_DIM[0], STORAGE_DIM[1], 3)
            img_t = torch.from_numpy(pix).permute(2, 0, 1).float() / 255.0
            frames.append(self.resize(img_t))
        return torch.cat(frames, dim=0), torch.tensor(row['oxygen'], dtype=torch.float32), torch.tensor(self.exp_id_map[exp_id], dtype=torch.long)

class TViT_Decoupler(nn.Module):
    def __init__(self, num_exps, stack_size=3):
        super().__init__()
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0)
        orig_proj = self.vit.patch_embed.proj
        self.vit.patch_embed.proj = nn.Conv2d(3 * stack_size, orig_proj.out_channels, kernel_size=orig_proj.kernel_size, stride=orig_proj.stride, padding=orig_proj.padding)
        with torch.no_grad():
            self.vit.patch_embed.proj.weight = nn.Parameter(orig_proj.weight.repeat(1, stack_size, 1, 1) / stack_size)
            self.vit.patch_embed.proj.bias = orig_proj.bias
        self.head = nn.Sequential(nn.Linear(self.vit.embed_dim, 128), nn.GELU(), nn.Linear(128, 1))
    def forward(self, x): return self.head(self.vit.forward_features(x)[:, 0]).squeeze(-1)

# ==============================================================================
# 7. ANALYTICS: CHEMOMETRICS & STERN-VOLMER FITTING (UPDATED)
# ==============================================================================
def fit_classical_baseline(train_df, test_df):
    """
    Fits the rigorous Two-Site Stern-Volmer model to the training data.
    Model: I = I0 * [ f/(1 + K1*[O2]) + (1-f)/(1 + K2*[O2]) ]

    This accounts for heterogeneity in the polymer matrix where dye molecules
    exist in two distinct environments with different quenching constants (K1, K2).
    """

    # 1. Filter stable regions for calibration
    stable_train = pd.concat([train_df[(train_df['elapsed'] >= s) & (train_df['elapsed'] <= e)] for s, e in ANALYSIS_INTERVALS_S])
    if stable_train.empty: stable_train = train_df

    # 2. Prepare Data: Convert stored 'inv_i' back to Intensity (I)
    # inv_i = 1/I  ->  I = 1/inv_i
    y_train_I = 1.0 / stable_train['inv_i'].values
    x_train_O2 = stable_train['oxygen'].values

    # 3. Define the Physical Model Function (Forward: O2 -> Intensity)
    def two_site_stern_volmer(o2, I0, f, K1, K2):
        # f is fraction of sites type 1
        # (1-f) is fraction of sites type 2
        return I0 * ( (f / (1.0 + K1 * o2)) + ((1.0 - f) / (1.0 + K2 * o2)) )

    # 4. Fit Parameters using Non-Linear Least Squares
    # Initial Guesses: I0=max(I), f=0.8 (mostly type 1), K1=0.01, K2=0.001
    p0 = [np.max(y_train_I), 0.8, 0.05, 0.005]
    # Bounds: I0>0, 0<=f<=1, K1>=0, K2>=0
    bounds = ([0, 0, 0, 0], [np.inf, 1.0, np.inf, np.inf])

    try:
        popt, pcov = curve_fit(two_site_stern_volmer, x_train_O2, y_train_I, p0=p0, bounds=bounds, maxfev=10000)
        I0_fit, f_fit, K1_fit, K2_fit = popt
        # print(f"  > SV Fit: I0={I0_fit:.1f}, f={f_fit:.2f}, K1={K1_fit:.3f}, K2={K2_fit:.3f}")
    except RuntimeError:
        print("  ! Stern-Volmer Fit Failed. Fallback to Linear.")
        # Fallback to simple linear if complex fit fails
        poly = make_pipeline(PolynomialFeatures(1), LinearRegression())
        poly.fit(stable_train[['inv_i']].values, stable_train['oxygen'].values)
        return poly.predict(test_df[['inv_i']].values)

    # 5. Prediction (Inverse: Intensity -> O2)
    # The Two-Site model leads to a Quadratic Equation for [O2] given I/I0.
    # Let y = I / I0.
    # The equation rearranges to: A*[O2]^2 + B*[O2] + C = 0

    I_test = 1.0 / test_df['inv_i'].values
    y_ratio = I_test / I0_fit

    # Coefficients for Ax^2 + Bx + C = 0
    # Derived algebraically from the two-site equation:
    A = y_ratio * K1_fit * K2_fit
    B = y_ratio * (K1_fit + K2_fit) - (f_fit * (K2_fit - K1_fit) + K1_fit)
    C = y_ratio - 1.0

    # Solve Quadratic Formula: x = (-B +/- sqrt(B^2 - 4AC)) / 2A
    # Note: B^2 - 4AC must be positive.
    delta = B**2 - 4*A*C

    # Handle numerical issues (delta < 0 due to noise)
    delta[delta < 0] = 0.0

    # We typically want the positive root that makes physical sense
    # Since y_ratio < 1 (quenching), and K > 0.
    # Usually the form gives one positive and one negative root. We take the positive one.
    o2_pred = (-B - np.sqrt(delta)) / (2 * A + 1e-9)

    # Alternatively check the other root if values are negative (physically impossible)
    mask_neg = o2_pred < 0
    if np.any(mask_neg):
        o2_pred[mask_neg] = (-B[mask_neg] + np.sqrt(delta[mask_neg])) / (2 * A[mask_neg] + 1e-9)

    return o2_pred

def plot_chemometrics(df, exp_id):
    y_true, y_pred = df['oxygen'], df['o2_pred']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Parity Plot
    ax1.scatter(y_true, y_pred, alpha=0.3, s=15, c='blue')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax1.plot(lims, lims, 'k--', lw=2)
    ax1.set_xlabel("Ground Truth Oxygen (%)")
    ax1.set_ylabel("Predicted Oxygen (%)")
    ax1.set_title(f"Parity Plot: {exp_id} ($R^2={r2_score(y_true, y_pred):.3f}$)")

    # Residual Histogram
    residuals = y_pred - y_true
    sns.histplot(residuals, kde=True, ax=ax2, color='purple')
    ax2.set_xlabel("Error (Pred - True)")
    ax2.set_title(f"Residual Distribution (MAE: {mean_absolute_error(y_true, y_pred):.3f})")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_Chemometrics", f"Chemometrics_{exp_id}.png"))
    plt.close()

def plot_training_dashboard(history, exp_id):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(history['epoch'], history['train_loss'], label='Train Loss (MSE)', color='blue')
    ax2 = ax1.twinx()
    ax2.plot(history['epoch'], history['val_mae'], label='Validation MAE', color='red', marker='o')
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Training MSE"); ax2.set_ylabel("Validation MAE")
    ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    plt.savefig(os.path.join(OUTPUT_DIR, "1_Training_Convergence", f"Convergence_{exp_id}.png"))
    plt.close()
    pd.DataFrame(history).to_csv(os.path.join(OUTPUT_DIR, "1_Training_Convergence", f"History_{exp_id}.csv"), index=False)

def analyze_dynamics_v10(df, exp_id, phase="Inference"):
    plot_df = df.copy()
    fig, ax = plt.subplots(figsize=(16, 9))
    t = plot_df['elapsed']
    ax.plot(t, plot_df['oxygen'], label='Ground Truth', color='black', lw=3)

    # Plot Classical Baseline (Two-Site Model)
    if 'o2_classical' in plot_df.columns:
        ax.plot(t, plot_df['o2_classical'], label='Classical (Stern-Volmer)', color='gray', ls='--', alpha=0.8)

    if 'o2_pred' in plot_df.columns:
        ax.plot(t, plot_df['o2_pred'], label='AI Prediction (TViT)', color='blue', lw=2)

    for (start, end) in ANALYSIS_INTERVALS_S:
        ax.axvspan(start, end, color='green', alpha=0.1)

    ax.set_title(f"Sensor Response: {exp_id} ({phase})")
    ax.set_ylabel("Oxygen Concentration")
    ax.set_xlabel("Time (s)")
    ax.legend()

    plt.savefig(os.path.join(OUTPUT_DIR, "2_Inference_Results" if phase=="Inference" else "0_Preflight_Diagnostics", f"{phase}_{exp_id}.png"))
    plt.close()
    plot_df.to_csv(os.path.join(CSV_EXPORT_DIR, f"{phase}_{exp_id}.csv"), index=False)

# ==============================================================================
# 8. EXECUTION ORCHESTRATOR
# ==============================================================================
def main():
    experiments = find_experiment_files_v10(EXPERIMENTS_ROOT_DIR)
    if N_EXPERIMENTS_CAP: experiments = experiments[:N_EXPERIMENTS_CAP]
    df = get_master_df(experiments, os.path.join(FEATURE_CACHE_DIR, "idc_master_v10.parquet"))
    exp_list = df['experiment_id'].unique()
    exp_map = {name: i for i, name in enumerate(exp_list)}
    global_metrics = []

    for test_idx, test_day in enumerate(exp_list):
        print(f"\nProcessing Fold {test_idx+1}/{len(exp_list)}: {test_day}")
        train_df, test_df = df[df['experiment_id'] != test_day], df[df['experiment_id'] == test_day]

        # --- NEW: Fit Physically Informed Classical Baseline ---
        print("  > Fitting Classical Stern-Volmer Two-Site Model...")
        test_df['o2_classical'] = fit_classical_baseline(train_df, test_df)

        train_loader = DataLoader(IDC_TemporalDataset(train_df, exp_map), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(IDC_TemporalDataset(test_df, exp_map), batch_size=BATCH_SIZE, shuffle=False)
        model = TViT_Decoupler(len(exp_list), stack_size=N_STACKED_FRAMES).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LR)
        criterion = nn.MSELoss()

        history = {'epoch': [], 'train_loss': [], 'val_mae': []}
        for epoch in range(N_EPOCHS):
            model.train()
            l_sum = 0
            for x, y, _ in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward(); optimizer.step()
                l_sum += loss.item()
            model.eval()
            val_preds = []
            with torch.no_grad():
                for x, y, _ in val_loader: val_preds.extend(model(x.to(DEVICE)).cpu().numpy())
            history['epoch'].append(epoch); history['train_loss'].append(l_sum/len(train_loader)); history['val_mae'].append(mean_absolute_error(test_df['oxygen'], val_preds))

        plot_training_dashboard(history, test_day)
        test_df['o2_pred'] = val_preds

        # Analyze results with both AI and Stern-Volmer baseline
        analyze_dynamics_v10(test_df, test_day)
        plot_chemometrics(test_df, test_day)

        unstruct_mask = test_df['elapsed'] < UNSTRUCTURED_CUTOFF_S
        global_metrics.append({
            'experiment_id': test_day,
            'MAE_AI': mean_absolute_error(test_df['oxygen'], test_df['o2_pred']),
            'MAE_Classical': mean_absolute_error(test_df['oxygen'], test_df['o2_classical']),
            'MAE_AI_Unstruct': mean_absolute_error(test_df[unstruct_mask]['oxygen'], test_df[unstruct_mask]['o2_pred'])
        })
    pd.DataFrame(global_metrics).to_csv(os.path.join(OUTPUT_DIR, "Global_Performance_Summary.csv"))
    print("Execution Finished. See OUTPUT_DIR for high-impact results.")

if __name__ == "__main__": main()
