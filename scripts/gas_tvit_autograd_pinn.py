# Repository edition: see docs/CHANGES.md and docs/REPRODUCIBILITY.md.
"""
=========================================================================================
PROJECT: PINN-IDC (Inverse Dynamic Compensation) Framework
VERSION: 12.3.0 - Myriad Cluster Headless + Classical Autograd PINN
=========================================================================================

--- ANALYTICAL DOCUMENTATION FOR FUTURE LLMs & DEVELOPERS ---
1. ARCHITECTURE & PURPOSE:
   This framework addresses the "Response Time Problem" in chemical sensing.
   We merge the robust cluster environment handling (from V11.4) with the
   Classical Autograd Physics-Informed Neural Network (PINN) (from V12.2).

2. CLUSTER ADAPTATIONS (Preserved from V11.4.0):
   - Headless Execution: All GUI elements stripped. Strict `matplotlib.use('Agg')`.
   - Constrained Environment: Native numpy polyfit replaces sklearn to prevent crashes.
   - Robust Checkpointing: Folds automatically skip if `.pt` weights and CSVs exist.
   - Forced Logging: `sys.stdout.flush()` is heavily used for Myriad log capture.
   - Auto-ROI & Rotation: Reads JSON configurations seamlessly.
   - Bulletproof Arduino Parsing: Scans logs directly for "MEA", bypassing commas.

3. PINN MATHEMATICS & AUTOGRAD (From V12.2):
   - The network takes two inputs: [Video Frames (x), Continuous Time (t)]
   - Outputs: S_pred (Continuous proxy), P_O2 (Predicted Oxygen)
   - PyTorch Autograd is used to calculate the EXACT analytical derivative dS/dt.
   - Total Loss = MSE(O2) + LAMBDA_S * MSE(S_Proxy) + LAMBDA_PHY * MSE(Physics_Constraint)

=========================================================================================
"""

# --- STRICT LIBRARY IMPORTS ---
# Portable repository paths; edit scripts/repo_config.py for your computer.
from repo_config import data_path, output_path, cache_path, external_path

import os
import sys
import glob
import re
import warnings
import random
import traceback
import json
from datetime import datetime, timedelta
from multiprocessing import cpu_count

import pandas as pd
import polars as pl
import numpy as np
import cv2
from tqdm import tqdm

# --- CLUSTER & VISUALIZATION CONFIG ---
import matplotlib

matplotlib.use('Agg')  # Strictly Headless rendering (No X11 forwarding needed)
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, r2_score

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
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

plt.style.use('seaborn-v0_8-whitegrid')
warnings.filterwarnings("ignore")  # Suppress warnings for cleaner cluster logs


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ==============================================================================
# 2. CONFIGURATION & PATHS (MYRIAD CLUSTER SPECIFIC)
# ==============================================================================
EXPERIMENTS_ROOT_DIR = data_path('gas')
OUTPUT_DIR = output_path('gas_tvit_autograd_pinn')
CSV_EXPORT_DIR = os.path.join(OUTPUT_DIR, "csv_exports")
CACHE_DIR = os.path.join(OUTPUT_DIR, "4_Caches")
SAVED_MODELS_DIR = os.path.join(OUTPUT_DIR, "5_Saved_Models")
ROI_CONFIG_PATH = external_path("gas_roi")

# --- EXECUTION FLAGS & HYPERPARAMETERS ---
FRAME_STEP_RATE = 1  # 1 = keep all frames (10fps)
ASSUMED_FPS = 10.0  # STRICTLY ENFORCED CAMERA FRAMERATE

N_EXPERIMENTS_CAP = None  # Set to an integer to limit for fast testing (e.g., 2)
N_STACKED_FRAMES = 3
STACK_LOOKBACK_STEP = 5  # Steps between frames in a stack
RESIZE_DIM = (224, 224)
STORAGE_DIM = (48, 48)
BATCH_SIZE = 32
N_EPOCHS = 35
LR = 1e-4

# --- PINN SPECIFIC HYPERPARAMETERS ---
LAMBDA_PHY = 0.5  # Weight of the physics constraint
LAMBDA_S = 1.0  # Weight for the proxy smoother constraint

# Auto-detect cluster GPUs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Compute Device Set to: {DEVICE}")
sys.stdout.flush()

# Create structured output directories
directories_to_make = [
    OUTPUT_DIR, CSV_EXPORT_DIR, CACHE_DIR, SAVED_MODELS_DIR,
    os.path.join(OUTPUT_DIR, "1_Training_Dynamics"),
    os.path.join(OUTPUT_DIR, "2_Inference_Results"),
    os.path.join(OUTPUT_DIR, "3_Chemometrics")
]
for d in directories_to_make: os.makedirs(d, exist_ok=True)


# ==============================================================================
# 3. ROBUST DATA DISCOVERY & RAW PROTOCOL PARSING
# ==============================================================================
def find_experiment_files_cluster(root_dir):
    print("\n--- Discovery: Searching for Valid Experiment Folders ---")
    sys.stdout.flush()
    valid_exps = []

    for subdir in os.listdir(root_dir):
        full_dir = os.path.join(root_dir, subdir)
        if not os.path.isdir(full_dir) or "Analysis" in subdir: continue

        match = re.search(r'(\d{8}_\d{6})', subdir)
        if not match: continue

        ts_str = match.group(1)
        video_t0 = datetime.strptime(ts_str, '%Y%m%d_%H%M%S')

        mp4s = glob.glob(os.path.join(full_dir, "*.mp4"))
        csvs = glob.glob(os.path.join(full_dir, "*arduino*.csv"))

        if len(mp4s) >= 1 and len(csvs) >= 1:
            valid_exps.append({
                "id": subdir,
                "video_path": mp4s[0],
                "arduino_path": csvs[0],
                "video_start_time": video_t0
            })
            print(f"  [+] Valid data found: {subdir}")
        else:
            print(f"  [-] Skipped {subdir}: Found {len(mp4s)} MP4s and {len(csvs)} CSVs.")

    sys.stdout.flush()
    return valid_exps


def get_roi_coordinates(path):
    if not os.path.exists(path):
        print("  [!] global_roi_settings.json NOT FOUND. Processing full frames.")
        return None, 0.0
    try:
        with open(path, 'r') as f:
            roi = json.load(f)
            angle = float(roi.get("angle", 0.0))
            if "roi_coords" in roi and isinstance(roi["roi_coords"], list) and len(roi["roi_coords"]) >= 4:
                c = roi["roi_coords"]
                return (int(c[0]), int(c[1]), int(c[2]), int(c[3])), angle
            elif isinstance(roi, dict) and all(k in roi for k in ('x', 'y', 'w', 'h')):
                return (int(roi['x']), int(roi['y']), int(roi['w']), int(roi['h'])), angle
            else:
                return None, 0.0
    except Exception as e:
        print(f"  [!] Error reading ROI file: {e}. Proceeding with full frames.")
        return None, 0.0


def parse_arduino_logs_polars(log_path, exp_id):
    records = []
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            next(f, None)
            for line in f:
                line = line.strip()
                if not line: continue

                ts_str = line.split(',')[0].strip()
                try:
                    ts = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    continue

                last_mea_idx = line.rfind("MEA")
                if last_mea_idx == -1: continue

                mea_portion = line[last_mea_idx:].replace(',', ' ')
                tokens = mea_portion.split()

                if len(tokens) >= 16 and tokens[0] == "MEA":
                    try:
                        o2_pct = float(tokens[15]) / 1000.0
                        records.append({'experiment_id': exp_id, 'timestamp': ts, 'oxygen': o2_pct})
                    except ValueError:
                        continue
    except Exception as e:
        print(f"  [!] Error parsing {log_path}: {e}")

    if not records: return pl.DataFrame()
    df = pl.DataFrame(records)
    df = df.with_columns(pl.col("oxygen").fill_nan(None).interpolate())
    return df


# ==============================================================================
# 4. DATA EXTRACTION, ROI APPLICATION, & TEMPORAL JOINING
# ==============================================================================
def get_master_df(experiments, step_rate):
    cache_path = os.path.join(CACHE_DIR, f"idc_master_v12_3_step{step_rate}.parquet")
    if os.path.exists(cache_path):
        print(f"\n[+] Loading Extracted Video Data from Cache (Step={step_rate})")
        sys.stdout.flush()
        return pl.read_parquet(cache_path).to_pandas()

    print(f"\n--- Extracting Video Frames (Step Rate = {step_rate}) ---")
    sys.stdout.flush()
    all_frames = []
    roi_coords, rotation_angle = get_roi_coordinates(ROI_CONFIG_PATH)

    for exp in tqdm(experiments, desc="Processing Videos (Cluster)", leave=False):
        cap = cv2.VideoCapture(exp['video_path'])
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for f_idx in range(0, total_f, step_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: break

            if rotation_angle != 0.0:
                frame_h, frame_w = frame.shape[:2]
                center = (frame_w // 2, frame_h // 2)
                M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                frame = cv2.warpAffine(frame, M, (frame_w, frame_h))

            if roi_coords:
                x, y, w, h = roi_coords
                frame_h, frame_w = frame.shape[:2]
                if x < frame_w and y < frame_h:
                    x_end, y_end = min(x + w, frame_w), min(y + h, frame_h)
                    cropped = frame[y:y_end, x:x_end]
                    if cropped.size > 0: frame = cropped

            ts = exp['video_start_time'] + timedelta(seconds=f_idx / ASSUMED_FPS)
            resized = cv2.resize(frame, STORAGE_DIM)
            intensity = np.mean(resized[:, :, 2].astype(float)) + 1e-6
            frame_bytes = resized.astype(np.uint8).tobytes()

            all_frames.append({
                'experiment_id': exp['id'],
                'timestamp': ts,
                'elapsed': f_idx / ASSUMED_FPS,  # The continuous time t for Autograd
                'inv_i': 1.0 / intensity,  # Noisy Proxy S
                'pixels': frame_bytes
            })
        cap.release()

    if not all_frames:
        print("\n[!!!] FATAL: No video frames extracted.")
        sys.exit(1)

    print("  > Joining Video frames with Arduino temporal data via Polars join_asof...")
    sys.stdout.flush()
    video_pl = pl.DataFrame(all_frames).sort("timestamp")

    ard_master = [parse_arduino_logs_polars(exp['arduino_path'], exp['id']) for exp in experiments]
    ard_master_clean = [df for df in ard_master if len(df) > 0]

    if not ard_master_clean:
        print("\n[!!!] FATAL: All Arduino files were corrupted or empty.")
        sys.exit(1)

    ard_pl = pl.concat(ard_master_clean).sort("timestamp")
    master_pl = video_pl.join_asof(ard_pl, on="timestamp", by="experiment_id", strategy="nearest")
    master_pl = master_pl.filter(pl.col("oxygen").is_not_null())

    master_pl.write_parquet(cache_path)
    return master_pl.to_pandas()


# ==============================================================================
# 5. CLASSICAL AUTOGRAD PINN ARCHITECTURE
# ==============================================================================
class AutogradPINN_Dataset(Dataset):
    """ Yields the Video Stack, Continuous Time, Oxygen, and the Proxy Intensity. """

    def __init__(self, df, stack_size=3, step=5):
        self.df = df.reset_index(drop=True)
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
            if self.df.iloc[lookback]['experiment_id'] != exp_id: lookback = idx

            raw_bytes = self.df.iloc[lookback]['pixels']
            pix = np.frombuffer(raw_bytes, dtype=np.uint8).copy().reshape(STORAGE_DIM[0], STORAGE_DIM[1], 3)
            img_t = torch.from_numpy(pix).permute(2, 0, 1).float() / 255.0
            frames.append(self.resize(img_t))

        x_stack = torch.cat(frames, dim=0)

        # Return: Images, Continuous Time (t), Ground Truth O2, Noisy Proxy Data (S)
        return (x_stack,
                torch.tensor([row['elapsed']], dtype=torch.float32),
                torch.tensor(row['oxygen'], dtype=torch.float32),
                torch.tensor(row['inv_i'], dtype=torch.float32))


class AutogradPINN_TViT(nn.Module):
    """ Spatio-Temporal PINN: takes frames (x) and time (t). Outputs Proxy & O2. """

    def __init__(self, stack_size=3):
        super().__init__()

        # Vision Backbone
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0)
        orig_proj = self.vit.patch_embed.proj
        self.vit.patch_embed.proj = nn.Conv2d(3 * stack_size, orig_proj.out_channels,
                                              kernel_size=orig_proj.kernel_size,
                                              stride=orig_proj.stride, padding=orig_proj.padding)
        with torch.no_grad():
            self.vit.patch_embed.proj.weight = nn.Parameter(orig_proj.weight.repeat(1, stack_size, 1, 1) / stack_size)
            self.vit.patch_embed.proj.bias = orig_proj.bias

        # Combined Space (Latent features + 1 Time coordinate)
        combined_dim = self.vit.embed_dim + 1

        # Dual Heads
        self.head_S = nn.Sequential(nn.Linear(combined_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.head_O2 = nn.Sequential(nn.Linear(combined_dim, 128), nn.GELU(), nn.Linear(128, 1))

        # Learnable Physics Components
        self.alpha_raw = nn.Parameter(torch.tensor(20.0))
        self.k_raw = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, t):
        feat = self.vit.forward_features(x)[:, 0]
        combined = torch.cat([feat, t], dim=1)  # Inject continuous time

        S_pred = self.head_S(combined).squeeze(-1)
        P_O2 = self.head_O2(combined).squeeze(-1)
        return S_pred, P_O2

    def get_physics_params(self):
        # Softplus strictly bounds constants to positive physical reality
        alpha = torch.nn.functional.softplus(self.alpha_raw)
        k = torch.nn.functional.softplus(self.k_raw) + 1e-6
        return alpha, k


# ==============================================================================
# 6. GLOBAL ANALYTICS & STERN-VOLMER FITTING
# ==============================================================================
def fit_classical_baseline_global(train_df, test_df):
    y_train_I = 1.0 / train_df['inv_i'].values
    x_train_O2 = train_df['oxygen'].values

    def sv_model(o2, I0, f, K1, K2):
        return I0 * ((f / (1.0 + K1 * o2)) + ((1.0 - f) / (1.0 + K2 * o2)))

    try:
        popt, pcov = curve_fit(sv_model, x_train_O2, y_train_I,
                               p0=[np.max(y_train_I), 0.8, 0.05, 0.005],
                               bounds=([0, 0, 0, 0], [np.inf, 1.0, np.inf, np.inf]), maxfev=10000)
        I0_fit, f_fit, K1_fit, K2_fit = popt

        I_test = 1.0 / test_df['inv_i'].values
        y_ratio = I_test / I0_fit
        A = y_ratio * K1_fit * K2_fit
        B = y_ratio * (K1_fit + K2_fit) - (f_fit * (K2_fit - K1_fit) + K1_fit)
        C = y_ratio - 1.0

        delta = B ** 2 - 4 * A * C
        delta[delta < 0] = 0.0
        o2_pred = (-B - np.sqrt(delta)) / (2 * A + 1e-9)

        mask_neg = o2_pred < 0
        if np.any(mask_neg):
            o2_pred[mask_neg] = (-B[mask_neg] + np.sqrt(delta[mask_neg])) / (2 * A[mask_neg] + 1e-9)
        return o2_pred

    except RuntimeError:
        print("  [!] Global Stern-Volmer Fit Failed. Fallback to native numpy polynomial.")
        z = np.polyfit(train_df['inv_i'].values, train_df['oxygen'].values, 2)
        p = np.poly1d(z)
        return p(test_df['inv_i'].values)


def export_plot_and_csv(fig, df_export, path_prefix):
    try:
        fig.tight_layout()
        fig.savefig(path_prefix + ".png", dpi=150)
        plt.close(fig)
        df_export.to_csv(path_prefix + ".csv", index=False)
    except Exception as e:
        print(f"  [!] Error saving visualization {path_prefix}: {e}")


def plot_chemometrics(df, exp_id):
    y_true, y_pred = df['oxygen'], df['o2_pred']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ax1.scatter(y_true, y_pred, alpha=0.3, s=15, c='blue')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax1.plot(lims, lims, 'k--', lw=2)
    ax1.set_xlabel("Ground Truth Oxygen (%)")
    ax1.set_ylabel("Predicted Oxygen (%)")
    ax1.set_title(f"Parity Plot: {exp_id} ($R^2={r2_score(y_true, y_pred):.3f}$)")

    residuals = y_pred - y_true
    sns.histplot(residuals, kde=True, ax=ax2, color='purple')
    ax2.set_xlabel("Error (Pred - True)")
    ax2.set_title(f"Residual Distribution (MAE: {mean_absolute_error(y_true, y_pred):.3f})")

    export_plot_and_csv(fig, pd.DataFrame({'True': y_true, 'Pred': y_pred, 'Residuals': residuals}),
                        os.path.join(OUTPUT_DIR, "3_Chemometrics", f"Chemometrics_{exp_id}"))


def plot_training_dashboard(history, exp_id):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot detailed loss components
    ax1.plot(history['epoch'], history['total_loss'], 'k-', lw=2, label='Total Loss')
    ax1.plot(history['epoch'], history['data_O2_loss'], 'b--', label='Data O2 Loss')
    ax1.plot(history['epoch'], history['data_S_loss'], 'y--', label='Proxy Smoother Loss')
    ax1.plot(history['epoch'], history['physics_loss'], 'g--', label='Physics Loss')

    ax2 = ax1.twinx()
    ax2.plot(history['epoch'], history['val_mae'], label='Validation MAE', color='red', marker='o')

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss Components (MSE)")
    ax2.set_ylabel("Validation MAE (%)")
    ax1.legend(loc='center left');
    ax2.legend(loc='center right')
    ax1.set_title("Autograd PINN Training Landscape")

    export_plot_and_csv(fig, pd.DataFrame(history),
                        os.path.join(OUTPUT_DIR, "1_Training_Dynamics", f"Convergence_{exp_id}"))


def analyze_dynamics_global(df, exp_id):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    t = df['elapsed']

    ax1.plot(t, df['oxygen'], label='Ground Truth O2', color='black', lw=3)
    if 'o2_classical' in df.columns:
        ax1.plot(t, df['o2_classical'], label='Global Classical (Stern-Volmer)', color='gray', ls='--', alpha=0.8)
    if 'physics_target' in df.columns:
        ax1.plot(t, df['physics_target'], 'g:', label='Analytical Physics Target', alpha=0.7, lw=2)
    if 'o2_pred' in df.columns:
        ax1.plot(t, df['o2_pred'], label='Autograd PINN O2', color='blue', lw=2)

    ax1.set_title(f"Autograd Sensor Response Dynamics: {exp_id}")
    ax1.set_ylabel("Oxygen Concentration (%)")
    ax1.legend()

    if 'o2_pred' in df.columns:
        error = df['o2_pred'] - df['oxygen']
        ax2.fill_between(t, 0, error, color='red', alpha=0.3, label='Prediction Error')
        ax2.set_ylabel("Error (%)")
        ax2.legend()

    ax2.set_xlabel("Elapsed Time (s)")

    export_cols = [c for c in ['elapsed', 'timestamp', 'oxygen', 'o2_classical', 'physics_target', 'o2_pred'] if
                   c in df.columns]
    export_plot_and_csv(fig, df[export_cols], os.path.join(OUTPUT_DIR, "2_Inference_Results", f"Inference_{exp_id}"))


# ==============================================================================
# 7. EXECUTION ORCHESTRATOR & AUTOGRAD TRAINING LOOP
# ==============================================================================
def main():
    experiments = find_experiment_files_cluster(EXPERIMENTS_ROOT_DIR)
    if not experiments:
        print("\n[!] No valid experiments found. Terminating.")
        sys.exit(1)

    if N_EXPERIMENTS_CAP: experiments = experiments[:N_EXPERIMENTS_CAP]

    df = get_master_df(experiments, FRAME_STEP_RATE)
    exp_list = df['experiment_id'].unique()

    summary_path = os.path.join(OUTPUT_DIR, "Global_Performance_Summary.csv")
    global_metrics = pd.read_csv(summary_path).to_dict('records') if os.path.exists(summary_path) else []

    # LOOCV Execution Loop
    for test_idx, test_day in enumerate(exp_list):
        print(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing LOOCV Fold {test_idx + 1}/{len(exp_list)}: {test_day}")
        sys.stdout.flush()

        # --- CLUSTER PROGRESSIVE SKIPPING CHECK ---
        model_save_path = os.path.join(SAVED_MODELS_DIR, f"AutogradPINN_{test_day}.pt")
        inference_csv_path = os.path.join(OUTPUT_DIR, "2_Inference_Results", f"Inference_{test_day}.csv")

        if os.path.exists(model_save_path) and os.path.exists(inference_csv_path):
            print(f"  [+] Fold for {test_day} already completed. Checkpoint active. Skipping.")
            continue

        # Split Data (Leave one out)
        train_df = df[df['experiment_id'] != test_day].copy()
        test_df = df[df['experiment_id'] == test_day].copy()

        # Classical Baseline
        test_df['o2_classical'] = fit_classical_baseline_global(train_df, test_df)

        # Datasets & Loaders
        train_loader = DataLoader(AutogradPINN_Dataset(train_df, stack_size=N_STACKED_FRAMES, step=STACK_LOOKBACK_STEP),
                                  batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=min(cpu_count(), 4), worker_init_fn=seed_worker)
        val_loader = DataLoader(AutogradPINN_Dataset(test_df, stack_size=N_STACKED_FRAMES, step=STACK_LOOKBACK_STEP),
                                batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=min(cpu_count(), 4))

        model = AutogradPINN_TViT(stack_size=N_STACKED_FRAMES).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LR)
        criterion = nn.MSELoss()

        history = {"epoch": [], "total_loss": [], "data_O2_loss": [], "data_S_loss": [], "physics_loss": [],
                   "val_mae": []}

        # --- AUTOGRAD PINN TRAINING LOOP ---
        for epoch in range(N_EPOCHS):
            model.train()
            ep_tot, ep_o2, ep_s, ep_phy = 0, 0, 0, 0

            for x, t, y_true, S_data in train_loader:
                x, y_true, S_data = x.to(DEVICE), y_true.to(DEVICE), S_data.to(DEVICE)

                # CRITICAL: Track gradients for continuous variable 't'
                t = t.to(DEVICE).requires_grad_(True)
                optimizer.zero_grad()

                S_pred, P_O2 = model(x, t)

                # EXACT analytical derivative calculation
                dS_dt_raw = torch.autograd.grad(
                    outputs=S_pred, inputs=t,
                    grad_outputs=torch.ones_like(S_pred),
                    create_graph=True, retain_graph=True
                )[0]
                dS_dt = dS_dt_raw.squeeze(-1)

                alpha, k = model.get_physics_params()

                # Multi-objective Loss Calculation
                loss_data_O2 = criterion(P_O2, y_true)
                loss_data_S = criterion(S_pred, S_data)

                physics_target = alpha * S_pred + (alpha / k) * dS_dt
                loss_physics = criterion(P_O2, physics_target)

                loss = loss_data_O2 + (LAMBDA_S * loss_data_S) + (LAMBDA_PHY * loss_physics)

                loss.backward()
                optimizer.step()

                ep_tot += loss.item();
                ep_o2 += loss_data_O2.item()
                ep_s += loss_data_S.item();
                ep_phy += loss_physics.item()

            # Validation (No gradients needed on t here)
            model.eval()
            val_preds = []
            with torch.no_grad():
                for x_val, t_val, _, _ in val_loader:
                    _, P_O2_val = model(x_val.to(DEVICE), t_val.to(DEVICE))
                    val_preds.extend(P_O2_val.cpu().numpy())

            # Record Epoch stats
            valid_mask = test_df['oxygen'].notna()
            epoch_mae = mean_absolute_error(test_df.loc[valid_mask, 'oxygen'], val_preds)

            history["epoch"].append(epoch)
            history["total_loss"].append(ep_tot / len(train_loader))
            history["data_O2_loss"].append(ep_o2 / len(train_loader))
            history["data_S_loss"].append(ep_s / len(train_loader))
            history["physics_loss"].append(ep_phy / len(train_loader))
            history["val_mae"].append(epoch_mae)

        torch.save(model.state_dict(), model_save_path)
        print(f"  [+] Autograd Model checkpointed.")

        # --- FINAL INFERENCE (Requires Gradients for target plotting) ---
        model.eval()
        final_preds, final_physics_targets = [], []

        for x, t, _, _ in val_loader:
            x = x.to(DEVICE)
            t = t.to(DEVICE).requires_grad_(True)
            S_pred, P_O2 = model(x, t)

            dS_dt = torch.autograd.grad(
                outputs=S_pred, inputs=t,
                grad_outputs=torch.ones_like(S_pred),
                create_graph=False
            )[0].squeeze(-1)

            alpha, k = model.get_physics_params()
            phys_targ = alpha * S_pred + (alpha / k) * dS_dt

            final_preds.extend(P_O2.detach().cpu().numpy())
            final_physics_targets.extend(phys_targ.detach().cpu().numpy())

        test_df['o2_pred'] = final_preds
        test_df['physics_target'] = final_physics_targets

        # Plots & Exports
        plot_training_dashboard(history, test_day)
        analyze_dynamics_global(test_df, test_day)
        plot_chemometrics(test_df, test_day)

        a_val, k_val = alpha.item(), k.item()

        # Calculate strict MAE metrics
        valid_mask = test_df['oxygen'].notna()
        global_metrics = [m for m in global_metrics if m['experiment_id'] != test_day]
        global_metrics.append({
            'experiment_id': test_day,
            'MAE_Autograd_PINN': mean_absolute_error(test_df.loc[valid_mask, 'oxygen'],
                                                     test_df.loc[valid_mask, 'o2_pred']),
            'MAE_Classical_SV': mean_absolute_error(test_df.loc[valid_mask, 'oxygen'],
                                                    test_df.loc[valid_mask, 'o2_classical']),
            'Learned_Alpha': a_val,
            'Learned_k': k_val,
            'R2_Score': r2_score(test_df.loc[valid_mask, 'oxygen'], test_df.loc[valid_mask, 'o2_pred'])
        })

        pd.DataFrame(global_metrics).to_csv(summary_path, index=False)
        print(f"  [+] Updated global summary saved. Constants: Alpha={a_val:.4f}, k={k_val:.4f}")
        sys.stdout.flush()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Cluster Execution Finished Successfully.")
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[!!!] FATAL CRASH [!!!]")
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)
