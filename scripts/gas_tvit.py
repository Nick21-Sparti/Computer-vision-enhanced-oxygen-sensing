# Repository edition: see docs/CHANGES.md and docs/REPRODUCIBILITY.md.
"""
==============================================================================
PROJECT: PINN-IDC (Inverse Dynamic Compensation) Framework
VERSION: 11.1.1 - Timestamp-Matched Discovery & Strict FPS Enforcement
==============================================================================

--- CONTEXT AND HISTORY ---
The fundamental challenge is the "Response Time Problem" in chemical sensing.
Diffusion creates a physical "lag" where the sensor reading trails the true
concentration. We use a Temporal Vision Transformer (TViT) to invert this
diffusion and "predict the present."

--- EVOLUTION OF THE FRAMEWORK (V1.0 -> V11.1.1) ---
- V1-V7: Static PINNs.
- V8.2: Robust file discovery.
- V10.0-10.6: Dynamic sensing, T90 Benchmarking, LOOCV.
- V10.7-10.8: Classical Baseline, Chemometrics, Black Box Training.
- V10.9.0: Stern-Volmer Two-Site Physical Model for heterogeneous polymer sensors.
- V11.0.0: Raw Protocol Parsing (MEA string) & Polars Temporal Alignment.
- V11.1.0: Global Physical Baseline & Fast Alignment checks.
- V11.1.1 (CURRENT) ADDED VALUE:
    1. TIMESTAMP-MATCHED DISCOVERY: Shifted from strict file-counting to
       timestamp-matching. Extracts the datetime from the parent folder name
       and selectively targets the specific .mp4 and .csv containing that
       timestamp. Prevents failures caused by auxiliary files in the directory.
    2. STRICT FPS ENFORCEMENT: Hardcoded video frame rate to 10.0 FPS to
       guarantee perfect temporal alignment, bypassing OpenCV metadata read
       errors.

--- DATA PROCESSING PIPELINE & SCIENTIFIC TECHNIQUES ---
1. File Traversal: Extracts T=0 from folder name, matches files by substring.
2. MEA Parsing: Extracts R12 (Oxygen in 0.001 %O2) from the Pyroscience protocol.
3. Fast Alignment Check: 1-FPS scan computes gross intensity to plot against O2.
4. Temporal Joining (Polars): Ultra-fast `join_asof` matches the exact Arduino
   reading to each extracted video frame based on T=0 timestamps.
5. NaN Handling: Missing serial packets are interpolated linearly via Polars.
6. TViT Processing: Decouples spatial variance with temporal stacking (3 frames).
7. Omni-Export: Every matplotlib figure automatically dumps its backing arrays
   to a CSV for full reproducibility.

--- FOLDER STRUCTURE & OUTPUTS ---
- Dynamic_IDC_Analysis_V11_1/
    - 0_Preflight_Diagnostics/     (Alignment visual checks)
    - 1_Training_Convergence/      (Epoch loss curves)
    - 2_Inference_Results/         (Predicted vs True dynamics)
    - 3_Chemometrics/              (Parity plots, Residual distributions)
    - 4_Caches/                    (Parquet DFs, Video hashes)
    - csv_exports/                 (Raw plot data for EVERY figure)
    - Global_Performance_Summary.csv
==============================================================================
"""
"""
==============================================================================
PROJECT: PINN-IDC (Inverse Dynamic Compensation) Framework
VERSION: 11.1.3 - Headless Configuration & LOOCV Execution
==============================================================================

--- CONTEXT AND HISTORY ---
The fundamental challenge is the "Response Time Problem" in chemical sensing.
Diffusion creates a physical "lag" where the sensor reading trails the true
concentration. We use a Temporal Vision Transformer (TViT) to invert this
diffusion and "predict the present."

--- EVOLUTION OF THE FRAMEWORK (V1.0 -> V11.1.3) ---
- V1-V10.9: Static PINNs -> Dynamic sensing -> Classical SV Baseline.
- V11.0.0: Raw Protocol Parsing (MEA string) & Polars Temporal Alignment.
- V11.1.0: Global Physical Baseline & Fast Alignment checks.
- V11.1.1: Timestamp-Matched Discovery & Strict FPS Enforcement.
- V11.1.2: Parquet Binary Serialization Patch for NumPy arrays.
- V11.1.3 (CURRENT) ADDED VALUE:
    1. HEADLESS EXECUTION UX: Moved the frame skip rate (`FRAME_STEP_RATE`) 
       and the visual alignment check toggle (`SHOW_ALIGNMENT_PLOT`) to the 
       global configuration block. This removes the need for blocking CLI 
       inputs, allowing for uninterrupted, headless batch processing.
    2. LOOCV CONFIRMATION: The training loop strictly adheres to Leave-One-Out 
       Cross-Validation at the experiment-level.

--- FOLDER STRUCTURE & OUTPUTS ---
- Dynamic_IDC_Analysis_V11_1/
    - 0_Preflight_Diagnostics/
    - 1_Training_Convergence/
    - 2_Inference_Results/
    - 3_Chemometrics/
    - 4_Caches/
    - csv_exports/
    - Global_Performance_Summary.csv
==============================================================================
"""

# Portable repository paths; edit scripts/repo_config.py for your computer.
from repo_config import data_path, output_path, cache_path, external_path

import os
import re
import glob
from datetime import datetime, timedelta

import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import polars as pl
import pandas as pd

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
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

plt.style.use('seaborn-v0_8-whitegrid')

# ==============================================================================
# 2. CONFIGURATION & PATHS
# ==============================================================================
EXPERIMENTS_ROOT_DIR = data_path('gas')
OUTPUT_DIR = output_path('gas_tvit')
CSV_EXPORT_DIR = os.path.join(OUTPUT_DIR, "csv_exports")
CACHE_DIR = os.path.join(OUTPUT_DIR, "4_Caches")

# --- UX / EXECUTION FLAGS ---
SHOW_ALIGNMENT_PLOT = False  # Set to True to pause and visually check synchronization.
FRAME_STEP_RATE = 1  # 1 = keep all frames (10fps). 5 = keep 1 in 5 frames (2fps).

ASSUMED_FPS = 10.0  # STRICTLY ENFORCED CAMERA FRAMERATE

N_EXPERIMENTS_CAP = None
N_STACKED_FRAMES = 3
RESIZE_DIM = (224, 224)
STORAGE_DIM = (48, 48)
BATCH_SIZE = 32
N_EPOCHS = 20
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

directories_to_make = [
    OUTPUT_DIR, CSV_EXPORT_DIR, CACHE_DIR,
    os.path.join(OUTPUT_DIR, "0_Preflight_Diagnostics"),
    os.path.join(OUTPUT_DIR, "1_Training_Convergence"),
    os.path.join(OUTPUT_DIR, "2_Inference_Results"),
    os.path.join(OUTPUT_DIR, "3_Chemometrics")
]
for d in directories_to_make: os.makedirs(d, exist_ok=True)


# ==============================================================================
# 3. ROBUST DATA DISCOVERY & RAW PROTOCOL PARSING
# ==============================================================================
def find_experiment_files_v11_1(root_dir):
    """
    Extracts timestamp from folder name, and finds files matching that timestamp.
    """
    print("\n--- Discovery: Searching for Valid Experiment Folders ---")
    valid_exps = []

    for subdir in os.listdir(root_dir):
        full_dir = os.path.join(root_dir, subdir)
        if not os.path.isdir(full_dir): continue
        if "Dynamic_IDC_Analysis" in subdir: continue  # Skip output directories

        match = re.search(r'(\d{8}_\d{6})', subdir)
        if not match:
            print(f"  [-] Skipping {subdir}: Could not parse timestamp from folder name.")
            continue

        ts_str = match.group(1)
        video_t0 = datetime.strptime(ts_str, '%Y%m%d_%H%M%S')

        mp4s = [f for f in glob.glob(os.path.join(full_dir, "*.mp4")) if ts_str in os.path.basename(f)]
        csvs = [f for f in glob.glob(os.path.join(full_dir, "*.csv")) if ts_str in os.path.basename(f)]

        if len(mp4s) >= 1 and len(csvs) >= 1:
            valid_exps.append({
                "id": subdir,
                "video_path": mp4s[0],
                "arduino_path": csvs[0],
                "video_start_time": video_t0
            })
            print(f"  [+] Found valid data aligned to {ts_str}: {subdir}")
        else:
            all_mp4s = len(glob.glob(os.path.join(full_dir, "*.mp4")))
            all_csvs = len(glob.glob(os.path.join(full_dir, "*.csv")))
            print(
                f"  [-] Skipping {subdir}: Found {all_mp4s} MP4s & {all_csvs} CSVs overall, but missing files matching '{ts_str}'.")

    return valid_exps


def parse_arduino_logs_polars(log_path):
    """ Reads Pyroscience CSV, handles glued MEA strings, extracts R12 (%O2). """
    records = []
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        next(f, None)  # Skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2: continue

            ts_str, raw_str = parts[0].strip(), parts[1].strip()
            try:
                ts = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                continue

            last_mea_idx = raw_str.rfind("MEA")
            if last_mea_idx == -1: continue
            tokens = raw_str[last_mea_idx:].split()

            if len(tokens) >= 16 and tokens[0] == "MEA":
                try:
                    o2_val = float(tokens[15]) / 1000.0
                    records.append({'timestamp': ts, 'oxygen': o2_val})
                except ValueError:
                    continue

    df = pl.DataFrame(records)
    if len(df) > 0:
        df = df.with_columns(pl.col("oxygen").fill_nan(None).interpolate())
    return df


# ==============================================================================
# 4. VISUAL ALIGNMENT GATE
# ==============================================================================
def do_fast_alignment_scan(video_path, video_t0):
    """ Reads 1 frame approx every 3s based on strictly enforced 10 FPS. """
    cap = cv2.VideoCapture(video_path)
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = int(max(1, ASSUMED_FPS * 3))

    ts_list, int_list = [], []
    for f_idx in range(0, total_f, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret: break
        ts_list.append(video_t0 + timedelta(seconds=f_idx / ASSUMED_FPS))
        int_list.append(np.mean(frame[:, :, 2]))  # Red channel approx
    cap.release()
    return ts_list, int_list


def verify_alignment_visually(experiments):
    """ Plots Video Intensity vs Arduino O2 and pauses to let the user check it. """
    print("\n--- INITIATING VISUAL ALIGNMENT CHECK ---")

    for exp in experiments:
        print(f"  > Scanning {exp['id']} for alignment preview...")
        ard_df = parse_arduino_logs_polars(exp['arduino_path']).to_pandas()
        fast_ts, fast_int = do_fast_alignment_scan(exp['video_path'], exp['video_start_time'])

        fig, ax1 = plt.subplots(figsize=(14, 6))
        ax2 = ax1.twinx()

        ax1.plot(ard_df['timestamp'], ard_df['oxygen'], color='blue', label='Arduino O2 (%)')
        ax2.plot(fast_ts, fast_int, color='orange', alpha=0.7, label='Video Intensity (Fast Scan)')
        ax1.axvline(x=exp['video_start_time'], color='red', linestyle='--', lw=2, label="Video Start (T=0)")

        ax1.set_ylabel('Oxygen (%)', color='blue')
        ax2.set_ylabel('Avg Brightness', color='orange')
        ax1.set_title(f"ALIGNMENT CHECK: {exp['id']}\nReview synchronization, then CLOSE this window to continue.")
        fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))

        plt.show(block=True)

        fig.savefig(os.path.join(OUTPUT_DIR, "0_Preflight_Diagnostics", f"Alignment_{exp['id']}.png"), dpi=150)
        plt.close(fig)

        pd.DataFrame({'fast_video_ts': fast_ts, 'fast_video_int': fast_int}).to_csv(
            os.path.join(CSV_EXPORT_DIR, f"Alignment_Video_Trace_{exp['id']}.csv"), index=False)
        ard_df.to_csv(os.path.join(CSV_EXPORT_DIR, f"Alignment_Arduino_Trace_{exp['id']}.csv"), index=False)


# ==============================================================================
# 5. DATA EXTRACTION & TEMPORAL JOINING (POLARS)
# ==============================================================================
def get_master_df(experiments, step_rate):
    """ Extracts frames. Joins with Arduino via Polars `join_asof`. """
    cache_path = os.path.join(CACHE_DIR, f"idc_master_v11_3_step{step_rate}.parquet")
    if os.path.exists(cache_path):
        print(f"\n  [+] Loading Extracted Video Data from Cache (Step={step_rate})")
        return pl.read_parquet(cache_path).to_pandas()

    print(f"\n--- Extracting Video Frames (Step Rate = {step_rate}) ---")
    all_frames = []

    for exp in tqdm(experiments, desc="Processing Videos"):
        cap = cv2.VideoCapture(exp['video_path'])
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for f_idx in range(0, total_f, step_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: break

            ts = exp['video_start_time'] + timedelta(seconds=f_idx / ASSUMED_FPS)
            resized = cv2.resize(frame, STORAGE_DIM)
            intensity = np.mean(resized[:, :, 2].astype(float)) + 1e-6

            # Convert numpy array to raw bytes so Parquet can serialize it.
            frame_bytes = resized.astype(np.uint8).tobytes()

            all_frames.append({
                'experiment_id': exp['id'],
                'timestamp': ts,
                'elapsed': f_idx / ASSUMED_FPS,
                'inv_i': 1.0 / intensity,
                'pixels': frame_bytes
            })
        cap.release()

    print("  > Joining Video frames with Arduino temporal data via Polars join_asof...")
    video_pl = pl.DataFrame(all_frames).sort("timestamp")

    ard_master = [parse_arduino_logs_polars(exp['arduino_path']) for exp in experiments]
    ard_pl = pl.concat(ard_master).sort("timestamp")

    master_pl = video_pl.join_asof(ard_pl, on="timestamp", strategy="nearest")
    master_pl.write_parquet(cache_path)

    return master_pl.to_pandas()


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
            if self.df.iloc[lookback]['experiment_id'] != exp_id: lookback = idx

            # Read the raw bytes from the dataframe and reconstruct the NumPy array
            raw_bytes = self.df.iloc[lookback]['pixels']
            pix = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(STORAGE_DIM[0], STORAGE_DIM[1], 3)

            img_t = torch.from_numpy(pix).permute(2, 0, 1).float() / 255.0
            frames.append(self.resize(img_t))

        return torch.cat(frames, dim=0), torch.tensor(row['oxygen'], dtype=torch.float32), torch.tensor(
            self.exp_id_map[exp_id], dtype=torch.long)


class TViT_Decoupler(nn.Module):
    def __init__(self, num_exps, stack_size=3):
        super().__init__()
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0)
        orig_proj = self.vit.patch_embed.proj
        self.vit.patch_embed.proj = nn.Conv2d(3 * stack_size, orig_proj.out_channels, kernel_size=orig_proj.kernel_size,
                                              stride=orig_proj.stride, padding=orig_proj.padding)
        with torch.no_grad():
            self.vit.patch_embed.proj.weight = nn.Parameter(orig_proj.weight.repeat(1, stack_size, 1, 1) / stack_size)
            self.vit.patch_embed.proj.bias = orig_proj.bias
        self.head = nn.Sequential(nn.Linear(self.vit.embed_dim, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, x): return self.head(self.vit.forward_features(x)[:, 0]).squeeze(-1)


# ==============================================================================
# 6. GLOBAL ANALYTICS & STERN-VOLMER FITTING
# ==============================================================================
def fit_classical_baseline_global(train_df, test_df):
    """
    Fits the rigorous Two-Site Stern-Volmer physical model globally
    across the entire continuous training set, removing interval bias.
    """
    y_train_I = 1.0 / train_df['inv_i'].values
    x_train_O2 = train_df['oxygen'].values

    def two_site_stern_volmer(o2, I0, f, K1, K2):
        return I0 * ((f / (1.0 + K1 * o2)) + ((1.0 - f) / (1.0 + K2 * o2)))

    p0 = [np.max(y_train_I), 0.8, 0.05, 0.005]
    bounds = ([0, 0, 0, 0], [np.inf, 1.0, np.inf, np.inf])

    try:
        popt, pcov = curve_fit(two_site_stern_volmer, x_train_O2, y_train_I, p0=p0, bounds=bounds, maxfev=10000)
        I0_fit, f_fit, K1_fit, K2_fit = popt
    except RuntimeError:
        print("  ! Global Stern-Volmer Fit Failed. Fallback to global polynomial.")
        poly = make_pipeline(PolynomialFeatures(2), LinearRegression())
        poly.fit(train_df[['inv_i']].values, train_df['oxygen'].values)
        return poly.predict(test_df[['inv_i']].values)

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


def export_plot_and_csv(fig, df_export, path_prefix):
    """ Utility to guarantee ALL plots have an underlying CSV generated. """
    fig.tight_layout()
    fig.savefig(path_prefix + ".png", dpi=150)
    plt.close(fig)
    df_export.to_csv(path_prefix + ".csv", index=False)


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
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(history['epoch'], history['train_loss'], label='Train Loss (MSE)', color='blue')
    ax2 = ax1.twinx()
    ax2.plot(history['epoch'], history['val_mae'], label='Validation MAE', color='red', marker='o')
    ax1.set_xlabel("Epoch");
    ax1.set_ylabel("Training MSE");
    ax2.set_ylabel("Validation MAE")
    ax1.legend(loc='upper left');
    ax2.legend(loc='upper right')

    export_plot_and_csv(fig, pd.DataFrame(history),
                        os.path.join(OUTPUT_DIR, "1_Training_Convergence", f"Convergence_{exp_id}"))


def analyze_dynamics_global(df, exp_id, phase="Inference"):
    plot_df = df.copy()
    fig, ax = plt.subplots(figsize=(16, 9))
    t = plot_df['elapsed']
    ax.plot(t, plot_df['oxygen'], label='Ground Truth', color='black', lw=3)

    if 'o2_classical' in plot_df.columns:
        ax.plot(t, plot_df['o2_classical'], label='Global Classical (Stern-Volmer)', color='gray', ls='--', alpha=0.8)

    if 'o2_pred' in plot_df.columns:
        ax.plot(t, plot_df['o2_pred'], label='AI Prediction (TViT)', color='blue', lw=2)

    ax.set_title(f"Sensor Response Dynamics: {exp_id} ({phase})")
    ax.set_ylabel("Oxygen Concentration (%)")
    ax.set_xlabel("Elapsed Time (s)")
    ax.legend()

    export_plot_and_csv(fig, plot_df[
        ['elapsed', 'timestamp', 'oxygen', 'o2_classical', 'o2_pred'] if 'o2_pred' in plot_df.columns else ['elapsed',
                                                                                                            'timestamp',
                                                                                                            'oxygen']],
                        os.path.join(OUTPUT_DIR,
                                     "2_Inference_Results" if phase == "Inference" else "0_Preflight_Diagnostics",
                                     f"{phase}_{exp_id}"))


# ==============================================================================
# 7. EXECUTION ORCHESTRATOR
# ==============================================================================
def main():
    experiments = find_experiment_files_v11_1(EXPERIMENTS_ROOT_DIR)
    if not experiments:
        print("\n[!] No valid experiments found. Terminating.")
        return

    if N_EXPERIMENTS_CAP: experiments = experiments[:N_EXPERIMENTS_CAP]

    # 1. Visual gate - conditionally executed based on global flag
    if SHOW_ALIGNMENT_PLOT:
        verify_alignment_visually(experiments)

    # 2. Process & Merge (Using global FRAME_STEP_RATE)
    df = get_master_df(experiments, FRAME_STEP_RATE)
    exp_list = df['experiment_id'].unique()
    exp_map = {name: i for i, name in enumerate(exp_list)}
    global_metrics = []

    # LOOCV Execution Loop (Leaves one experiment out for validation/testing)
    for test_idx, test_day in enumerate(exp_list):
        print(f"\nProcessing LOOCV Fold {test_idx + 1}/{len(exp_list)}: {test_day}")

        # Split Data
        train_df = df[df['experiment_id'] != test_day].copy()
        test_df = df[df['experiment_id'] == test_day].copy()

        print("  > Fitting Global Classical Stern-Volmer Two-Site Model...")
        test_df['o2_classical'] = fit_classical_baseline_global(train_df, test_df)

        print("  > Training Dynamic Spatial-Temporal Decoupler (TViT)...")
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
                loss.backward();
                optimizer.step()
                l_sum += loss.item()
            model.eval()
            val_preds = []
            with torch.no_grad():
                for x, y, _ in val_loader: val_preds.extend(model(x.to(DEVICE)).cpu().numpy())
            history['epoch'].append(epoch)
            history['train_loss'].append(l_sum / len(train_loader))
            history['val_mae'].append(
                mean_absolute_error(test_df['oxygen'].fillna(test_df['oxygen'].mean()), val_preds))

        plot_training_dashboard(history, test_day)
        test_df['o2_pred'] = val_preds

        # Visually & Structurally Save Output
        analyze_dynamics_global(test_df, test_day)
        plot_chemometrics(test_df, test_day)

        # Global MAE Metrics (Ignoring NaNs gracefully)
        valid_mask = test_df['oxygen'].notna()
        global_metrics.append({
            'experiment_id': test_day,
            'MAE_AI': mean_absolute_error(test_df.loc[valid_mask, 'oxygen'], test_df.loc[valid_mask, 'o2_pred']),
            'MAE_Classical': mean_absolute_error(test_df.loc[valid_mask, 'oxygen'],
                                                 test_df.loc[valid_mask, 'o2_classical']),
        })

    pd.DataFrame(global_metrics).to_csv(os.path.join(OUTPUT_DIR, "Global_Performance_Summary.csv"), index=False)
    print("\nExecution Finished. All plots and CSV backings exported successfully.")


if __name__ == "__main__":
    main()
