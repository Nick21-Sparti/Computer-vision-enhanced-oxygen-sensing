# Repository edition: see docs/CHANGES.md and docs/REPRODUCIBILITY.md.
"""
==============================================================================
PROJECT: PINN-IDC (Inverse Dynamic Compensation) Framework
VERSION: 10.9.3 - Bayesian Logic & Optuna HPO
==============================================================================

--- SCIENTIFIC CONTEXT & GOAL ---
The fundamental challenge is the "Response Time Problem" in chemical sensing.
Diffusion creates a physical "lag" where the sensor reading trails the true
concentration. 

In this version (V10.9.3), we shift from a pure frequentist approach to a 
BAYESIAN DEEP LEARNING approach.
1. We view the weights of the network as distributions, approximated via 
   Monte Carlo Dropout (Gal & Ghahramani, 2016).
2. We treat the output not as a scalar, but as a Gaussian distribution 
   parameterized by (mu, log_var), maximizing the Log-Likelihood.
3. We optimize the priors (hyperparameters) using Optuna on a 5% subset 
   of the data to ensure the Bayesian priors are calibrated.

--- EVOLUTION OF THE FRAMEWORK (V1.0 -> V10.9.3) ---
- V1-V7: Static PINNs.
- V8-V10.8: Dynamic sensing, T90 Benchmarking, Black Box Training.
- V10.9.0: Multi-Task Learning (MTL) & Uncertainty Quantification (UQ).
- V10.9.1: Fixed PyArrow/Parquet Type Errors.
- V10.9.2: Crash Recovery, Incremental Saving, Model Checkpointing.
- V10.9.3 (CURRENT):
    1. BAYESIAN INFERENCE: Implemented MC Dropout during inference to 
       quantify Epistemic (Model) vs Aleatoric (Data) uncertainty.
    2. OPTUNA HPO: Automatic hyperparameter tuning using a persistent 
       SQLite database and a 5% data subset for speed.
    3. UNCERTAINTY DECOMPOSITION: New plots separating 'Sensor Noise' 
       from 'Physics Unknowns'.

--- FOLDER STRUCTURE & OUTPUTS ---
- Dynamic_IDC_Analysis_V10/
    - 0_Preflight_Diagnostics/
    - 1_Training_Convergence/
    - 2_Inference_Results/
    - 3_Chemometrics/
    - 5_Uncertainty_Analysis/ (Enhanced with Bayesian Decomp)
    - 6_MultiTask_Diagnostics/
    - 7_Model_Checkpoints/
    - 8_Optuna_Optimization/ (NEW: HPO results)
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
from scipy.signal import medfilt
from scipy.interpolate import interp1d
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold

# PyTorch & AI
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
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
OUTPUT_DIR = output_path('aqueous_tvit_bayesian')
CSV_EXPORT_DIR = os.path.join(OUTPUT_DIR, "csv_exports")
FEATURE_CACHE_DIR = cache_path('aqueous_tvit_bayesian', "features")
CV_CACHE_DIR = cache_path('aqueous_tvit_bayesian', "folds")
MODEL_CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "7_Model_Checkpoints")
OPTUNA_DIR = os.path.join(OUTPUT_DIR, "8_Optuna_Optimization")

# Bayesian Hyperparameters (Defaults, will be overridden by Optuna if enabled)
N_STACKED_FRAMES = 3
FRAME_STACK_STEP = 10
RESIZE_DIM = (224, 224)
STORAGE_DIM = (48, 48)
BATCH_SIZE = 32
LR = 1e-4  # <--- ADD THIS LINE
N_EPOCHS = 20
MC_SAMPLES = 50  # Number of stochastic forward passes for Bayesian Inference
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Adversarial & Multi-Task Weights
ALPHA_GRL = 1.0
LAMBDA_PHASE = 0.5
LAMBDA_DOMAIN = 0.5

# Optuna Config
OPTUNA_TRIALS = 20  # Number of HPO trials
OPTUNA_SUBSET_FRAC = 0.05  # Use 5% of data for speed
OPTUNA_DB_PATH = f"sqlite:///{os.path.join(OPTUNA_DIR, 'optuna_study.db')}"

# Create Directory Structure
directories_to_make = [
    OUTPUT_DIR, CSV_EXPORT_DIR, FEATURE_CACHE_DIR, CV_CACHE_DIR, MODEL_CHECKPOINT_DIR, OPTUNA_DIR,
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

    # 4. Interpolate via Pandas (Graceful NaN handling as requested)
    pdf = df_merged.to_pandas()
    pdf['temp'] = pd.to_numeric(pdf['temp'], errors='coerce')
    pdf['oxygen'] = pdf['oxygen'].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    pdf['temp'] = pdf['temp'].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    pdf['temp'] = pdf['temp'].fillna(20.0)

    # 5. Sanity Check for NaNs
    if pdf.isnull().values.any():
        pdf = pdf.fillna(method='ffill').fillna(0.0)

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
# 4. BAYESIAN NEURAL NETWORK (GRL + UQ + DROPOUT)
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


class BayesianTViT_IDC(nn.Module):
    """
    Bayesian Temporal Vision Transformer.
    Uses MC Dropout (dropout active during train AND eval) to approximate
    the posterior distribution of the weights.
    """

    def __init__(self, num_domains, stack_size=3, dropout_rate=0.1):
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
        self.dropout = nn.Dropout(p=dropout_rate)  # The key to Bayesian Logic via MC Dropout

        # Regressor: Predicts mean and log_variance (Aleatoric Uncertainty)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 2)
        )

        self.phase_classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        self.grl = GradientReversalLayer(alpha=ALPHA_GRL)
        self.domain_classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_domains)
        )

    def forward(self, x):
        features = self.vit.forward_features(x)[:, 0]
        features = self.dropout(features)  # Apply dropout to features

        reg_out = self.regressor(features)
        mu = reg_out[:, 0]
        log_var = reg_out[:, 1]

        phase_logits = self.phase_classifier(features)

        rev_features = self.grl(features)
        domain_logits = self.domain_classifier(rev_features)

        return mu, log_var, phase_logits, domain_logits


# ==============================================================================
# 5. OPTUNA OPTIMIZATION (Bayesian Prior Calibration)
# ==============================================================================
def objective_optuna(trial, df_subset, exp_map, num_domains):
    """
    Optuna objective function using a small subset of data.
    """
    # Suggest Hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.05, 0.4)
    lambda_p = trial.suggest_float("lambda_phase", 0.1, 1.0)

    # K-Fold within the subset to prevent leakage
    gkf = GroupKFold(n_splits=3)
    train_idx, val_idx = next(gkf.split(df_subset, groups=df_subset['experiment_id']))

    train_sub = df_subset.iloc[train_idx]
    val_sub = df_subset.iloc[val_idx]

    train_loader = DataLoader(IDC_TemporalDataset(train_sub, exp_map), batch_size=16, shuffle=True)
    val_loader = DataLoader(IDC_TemporalDataset(val_sub, exp_map), batch_size=16, shuffle=False)

    model = BayesianTViT_IDC(num_domains, stack_size=N_STACKED_FRAMES, dropout_rate=dropout).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion_ce = nn.CrossEntropyLoss()

    # Short Training Loop
    for epoch in range(5):
        model.train()
        for x, y_o2, y_dom, y_phase in train_loader:
            x, y_o2 = x.to(DEVICE), y_o2.to(DEVICE)
            y_dom, y_phase = y_dom.to(DEVICE), y_phase.to(DEVICE)

            optimizer.zero_grad()
            mu, log_var, phase_logit, dom_logit = model(x)

            precision = torch.exp(-log_var)
            loss_reg = torch.mean(0.5 * precision * (y_o2 - mu) ** 2 + 0.5 * log_var)
            loss_phase = criterion_ce(phase_logit, y_phase)
            loss_dom = criterion_ce(dom_logit, y_dom)

            loss = loss_reg + (lambda_p * loss_phase) + (0.5 * loss_dom)
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    val_mae = []
    with torch.no_grad():
        for x, y_o2, _, _ in val_loader:
            x = x.to(DEVICE)
            mu, _, _, _ = model(x)
            val_mae.append(mean_absolute_error(y_o2.cpu().numpy(), mu.cpu().numpy()))

    return np.mean(val_mae)


def run_optuna_optimization(df, exp_map, num_domains):
    """
    Runs Optuna on a 5% subset. Resumes from DB if available.
    """
    print("\n[OPTUNA] Preparing 5% Data Subset...")
    # Polars-style sampling for speed, then convert to Pandas for DataLoader
    df_pl = pl.from_pandas(df)
    subset_pl = df_pl.sample(fraction=OPTUNA_SUBSET_FRAC, seed=SEED, shuffle=True)
    df_subset = subset_pl.to_pandas()

    print(f"[OPTUNA] subset size: {len(df_subset)}")

    study = optuna.create_study(
        study_name="idc_bayesian_opt",
        storage=OPTUNA_DB_PATH,
        load_if_exists=True,
        direction="minimize"
    )

    # Check if we need more trials
    if len(study.trials) < OPTUNA_TRIALS:
        print(f"[OPTUNA] Starting optimization (Target: {OPTUNA_TRIALS} trials)...")
        study.optimize(
            lambda trial: objective_optuna(trial, df_subset, exp_map, num_domains),
            n_trials=OPTUNA_TRIALS - len(study.trials)
        )
    else:
        print(f"[OPTUNA] Study complete ({len(study.trials)} trials). Loading best params.")

    print(f"[OPTUNA] Best Params: {study.best_params}")

    # Save visualizations
    try:
        fig1 = plot_optimization_history(study)
        fig1.write_image(os.path.join(OPTUNA_DIR, "optuna_history.png"))
        fig2 = plot_param_importances(study)
        fig2.write_image(os.path.join(OPTUNA_DIR, "optuna_importance.png"))

        # CSV Export of trials
        study.trials_dataframe().to_csv(os.path.join(OPTUNA_DIR, "optuna_trials.csv"))
    except:
        print("Could not generate Optuna plots (check plotly dependency).")

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
    y_true, y_pred = df['oxygen'], df['o2_pred_mean']
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


def plot_bayesian_uncertainty_analysis(df, exp_id):
    """
    New Bayesian Visualization: Separates Aleatoric (Data noise) vs Epistemic (Model confusion).
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    t = df['elapsed']

    # 1. Prediction with Total Uncertainty
    sigma_total = df['o2_std_total']
    ax1.plot(t, df['oxygen'], 'k-', label='Ground Truth', lw=2)
    ax1.plot(t, df['o2_pred_mean'], 'b--', label='Bayesian Mean', lw=1.5)
    ax1.fill_between(t, df['o2_pred_mean'] - 2 * sigma_total, df['o2_pred_mean'] + 2 * sigma_total,
                     color='blue', alpha=0.2, label='95% Total Uncertainty')
    ax1.set_title(f"Bayesian Inference: {exp_id}")
    ax1.legend(loc='upper right')

    # 2. Epistemic Uncertainty (Model Knowledge)
    ax2.plot(t, df['o2_std_epistemic'], color='purple', lw=2)
    ax2.set_ylabel("Epistemic Std (Model)")
    ax2.set_title("Epistemic Uncertainty (High = Unknown Physics/Drift)")

    # 3. Aleatoric Uncertainty (Sensor Noise)
    ax3.plot(t, df['o2_std_aleatoric'], color='orange', lw=2)
    ax3.set_ylabel("Aleatoric Std (Noise)")
    ax3.set_title("Aleatoric Uncertainty (High = Noisy Sensor)")
    ax3.set_xlabel("Time (s)")

    # Highlight Stable Regions
    for (s, e) in ANALYSIS_INTERVALS_S:
        ax1.axvspan(s, e, color='green', alpha=0.05)
        ax2.axvspan(s, e, color='green', alpha=0.05)
        ax3.axvspan(s, e, color='green', alpha=0.05)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "5_Uncertainty_Analysis", f"Bayesian_Uncertainty_{exp_id}.png"))
    plt.close()

    # Export Data
    df[['elapsed', 'oxygen', 'o2_pred_mean', 'o2_std_total', 'o2_std_epistemic', 'o2_std_aleatoric']].to_csv(
        os.path.join(OUTPUT_DIR, "5_Uncertainty_Analysis", f"Bayesian_Data_{exp_id}.csv"), index=False)


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
    ax1.plot(history['epoch'], history['loss_reg'], label='Reg Loss (NLL)', color='blue', linestyle='--')
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
    if 'o2_pred_mean' in plot_df.columns: ax.plot(t, plot_df['o2_pred_mean'], label='Bayesian Mean', color='blue', lw=2)
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

    # Load Data (Cached)
    df = get_master_df(experiments, os.path.join(FEATURE_CACHE_DIR, "idc_master_v10.9.1.parquet"))

    exp_list = df['experiment_id'].unique()
    exp_map = {name: i for i, name in enumerate(exp_list)}
    num_domains = len(exp_list)

    print(f"Loaded {len(df)} frames across {num_domains} experiments.")

    # --------------------------------------------------------------------------
    # OPTUNA HPO PHASE
    # --------------------------------------------------------------------------
    print("\n--- Phase 1: Optuna Hyperparameter Optimization ---")
    best_params = run_optuna_optimization(df, exp_map, num_domains)

    # Extract params or use defaults
    p_lr = best_params.get("lr", LR)
    p_dropout = best_params.get("dropout", 0.1)
    p_lambda_phase = best_params.get("lambda_phase", LAMBDA_PHASE)

    print(f"Using Parameters -> LR: {p_lr}, Dropout: {p_dropout}, Lambda_Phase: {p_lambda_phase}")

    # --------------------------------------------------------------------------
    # MAIN TRAINING LOOP (CROSS-VALIDATION)
    # --------------------------------------------------------------------------
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

        # Init Bayesian Model with Optuna Dropout
        model = BayesianTViT_IDC(num_domains, stack_size=N_STACKED_FRAMES, dropout_rate=p_dropout).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=p_lr)
        criterion_ce = nn.CrossEntropyLoss()

        history = {'epoch': [], 'loss_total': [], 'loss_reg': [], 'loss_phase': [], 'loss_domain': [], 'val_mae': [],
                   'domain_acc': []}

        for epoch in range(N_EPOCHS):
            model.train()  # Dropout is ACTIVE
            l_total_sum = 0
            l_reg_sum = 0
            l_phase_sum = 0
            l_dom_sum = 0

            for x, y_o2, y_domain, y_phase in train_loader:
                x, y_o2, y_domain, y_phase = x.to(DEVICE), y_o2.to(DEVICE), y_domain.to(DEVICE), y_phase.to(DEVICE)
                optimizer.zero_grad()
                mu, log_var, phase_logits, domain_logits = model(x)

                # NLL Loss for Gaussian
                precision = torch.exp(-log_var)
                loss_reg = torch.mean(0.5 * precision * (y_o2 - mu) ** 2 + 0.5 * log_var)
                loss_phase = criterion_ce(phase_logits, y_phase)
                loss_domain = criterion_ce(domain_logits, y_domain)

                loss = loss_reg + (p_lambda_phase * loss_phase) + (LAMBDA_DOMAIN * loss_domain)
                loss.backward()
                optimizer.step()

                l_total_sum += loss.item()
                l_reg_sum += loss_reg.item()
                l_phase_sum += loss_phase.item()
                l_dom_sum += loss_domain.item()

            # Validation (Standard pass for dashboard, not full MC yet)
            model.eval()  # Warning: Standard eval turns off dropout. We use this just for quick metrics.
            val_preds_o2 = []
            val_preds_dom = []
            val_targets_dom = []

            with torch.no_grad():
                for x, y_o2, y_domain, y_phase in val_loader:
                    x = x.to(DEVICE)
                    # For quick validation, we run single pass (mean estimation)
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
        torch.save(model.state_dict(), os.path.join(MODEL_CHECKPOINT_DIR, f"model_fold_{test_day}.pth"))

        # ----------------------------------------------------------------------
        # BAYESIAN INFERENCE (MC DROPOUT)
        # ----------------------------------------------------------------------
        print(f"Running Bayesian Inference (MC Samples={MC_SAMPLES})...")
        model.train()  # ENABLE DROPOUT for Inference

        # Storage for MC samples
        mc_means = []
        mc_logvars = []
        mc_phases = []

        with torch.no_grad():
            # Run N forward passes
            for _ in range(MC_SAMPLES):
                batch_means = []
                batch_logvars = []
                batch_phases = []
                for x, _, _, _ in val_loader:
                    x = x.to(DEVICE)
                    mu, log_var, phase_logits, _ = model(x)
                    batch_means.append(mu.cpu().numpy())
                    batch_logvars.append(log_var.cpu().numpy())
                    batch_phases.append(torch.argmax(phase_logits, dim=1).cpu().numpy())

                mc_means.append(np.concatenate(batch_means))
                mc_logvars.append(np.concatenate(batch_logvars))
                mc_phases.append(np.concatenate(batch_phases))

        # Stack: (Samples, N_Data)
        mc_means = np.stack(mc_means)
        mc_logvars = np.stack(mc_logvars)
        mc_phases = np.stack(mc_phases)

        # Bayesian Uncertainty Decomposition
        # 1. Prediction Mean (Average of MC means)
        pred_mean = np.mean(mc_means, axis=0)

        # 2. Epistemic Uncertainty (Variance of the Means) -> "Model Confusion"
        epistemic_var = np.var(mc_means, axis=0)

        # 3. Aleatoric Uncertainty (Average of the output Variances) -> "Data Noise"
        # We predict log_var, so convert to var first: exp(log_var)
        aleatoric_var = np.mean(np.exp(mc_logvars), axis=0)

        # 4. Total Uncertainty
        total_std = np.sqrt(epistemic_var + aleatoric_var)

        # Phase Consensus
        from scipy.stats import mode
        final_phase_pred, _ = mode(mc_phases, axis=0, keepdims=False)

        # Store Results
        test_df['o2_pred_mean'] = pred_mean
        test_df['o2_std_total'] = total_std
        test_df['o2_std_epistemic'] = np.sqrt(epistemic_var)
        test_df['o2_std_aleatoric'] = np.sqrt(aleatoric_var)
        test_df['phase_pred'] = final_phase_pred

        analyze_dynamics_v10(test_df, test_day)
        plot_chemometrics(test_df, test_day)
        plot_bayesian_uncertainty_analysis(test_df, test_day)  # NEW PLOT
        plot_multitask_diagnostics(test_df, test_day)

        # INCREMENTAL METRICS SAVING
        unstruct_mask = test_df['elapsed'] < UNSTRUCTURED_CUTOFF_S
        metrics_dict = {
            'experiment_id': test_day,
            'MAE_AI': mean_absolute_error(test_df['oxygen'], test_df['o2_pred_mean']),
            'MAE_Unstruct': mean_absolute_error(test_df[unstruct_mask]['oxygen'],
                                                test_df[unstruct_mask]['o2_pred_mean']),
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
