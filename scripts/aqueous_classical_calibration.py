# Repository edition: see docs/CHANGES.md and docs/REPRODUCIBILITY.md.
# Portable repository paths; edit scripts/repo_config.py for your computer.
from repo_config import data_path, output_path, cache_path, external_path

import os
import glob
import re
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
ROOT_DIR = data_path('aqueous')
BASE_OUTPUT = output_path('aqueous_classical_calibration')

ANALYSIS_INTERVALS_S = [(1450, 1650), (2300, 2500), (3100, 3300), (3900, 4100), (4700, 4900)]
# STORAGE_DIM = (128, 128) # original results were for 128x128
STORAGE_DIM = (48, 48)
TOP_N_PIXELS = 10000

# Subfolder Structure
BRANCHES = ["Plateau_Only", "Full_Timeline"]
for b in BRANCHES:
    os.makedirs(os.path.join(BASE_OUTPUT, b, "Calibration_Fits"), exist_ok=True)
    os.makedirs(os.path.join(BASE_OUTPUT, b, "Inference_Plots"), exist_ok=True)
    os.makedirs(os.path.join(BASE_OUTPUT, b, "CSVs"), exist_ok=True)


# ==============================================================================
# 2. MODELS
# ==============================================================================
def two_site_model(o2, S0, f, K1, K2):
    return S0 * ((f / (1.0 + K1 * o2)) + ((1.0 - f) / (1.0 + K2 * o2)))


def inverse_two_site_model(signal, S0, f, K1, K2):
    y = np.clip(signal / (S0 + 1e-9), 1e-5, 0.999)
    A, B, C = y * K1 * K2, y * (K1 + K2) - (f * K2 + (1 - f) * K1), y - 1.0
    delta = np.maximum(B ** 2 - 4 * A * C, 0)
    r1, r2 = (-B + np.sqrt(delta)) / (2 * A + 1e-9), (-B - np.sqrt(delta)) / (2 * A + 1e-9)
    return np.maximum(np.where(r1 > 0, r1, r2), 0)


def poly_model_predict(signal, coeffs):
    return np.polyval(coeffs, signal)


# ==============================================================================
# 3. DATA EXTRACTION
# ==============================================================================
def parse_arduino_log(path):
    data = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if "MEA" in line:
                parts = line.split()
                try:
                    ts = datetime.strptime(f"{parts[0]} {parts[1]}", '%Y-%m-%d %H:%M:%S')
                    o2 = float(parts[parts.index("MEA") + 5]) / 1000.0
                    data.append({'timestamp': ts, 'oxygen': o2})
                except:
                    continue
    return pd.DataFrame(data).sort_values('timestamp')


def extract_signals(video_path, log_df):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    match = re.search(r'_(\d{8}_\d{6})', os.path.basename(video_path))
    start_ts = datetime.strptime(match.group(1), '%Y%m%d_%H%M%S') if match else log_df['timestamp'].iloc[0]

    ret, frame = cap.read()
    if not ret: return None
    red_init = cv2.resize(frame, STORAGE_DIM)[:, :, 2].flatten()
    bright_idx = np.argsort(red_init)[-TOP_N_PIXELS:]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    trace = []
    for f_idx in range(0, total_frames, int(max(1, fps))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret: break
        red_flat = cv2.resize(frame, STORAGE_DIM)[:, :, 2].flatten()
        selected = red_flat[bright_idx]
        elapsed = f_idx / fps
        trace.append({
            'timestamp': start_ts + timedelta(seconds=elapsed),
            'elapsed': elapsed,
            'mean_I': np.mean(selected),
            'is_stable': any(s <= elapsed <= e for s, e in ANALYSIS_INTERVALS_S)
        })
    cap.release()
    df = pd.merge_asof(pd.DataFrame(trace), log_df, on='timestamp', direction='nearest')
    return df.ffill().bfill()


# ==============================================================================
# 4. LOOCV CORE
# ==============================================================================
def main():
    exp_dirs = [d for d in glob.glob(os.path.join(ROOT_DIR, '*-*')) if os.path.isdir(d)]
    all_data = []
    for d in tqdm(exp_dirs, desc="Pre-loading Videos"):
        try:
            v, l = glob.glob(os.path.join(d, "**/*.mp4"), recursive=True)[0], \
            glob.glob(os.path.join(d, "**/*_arduino_*.txt"), recursive=True)[0]
            df = extract_signals(v, parse_arduino_log(l))
            if df is not None:
                df['exp_id'] = os.path.basename(d)
                all_data.append(df)
        except:
            continue

    master_df = pd.concat(all_data)
    ids = master_df['exp_id'].unique()

    for test_id in ids:
        print(f"\nProcessing LOOCV Fold: {test_id}")

        # Define Train Sets
        train_pool = master_df[master_df['exp_id'] != test_id].dropna()
        test_df = master_df[master_df['exp_id'] == test_id].copy()

        modes = [
            ("Plateau_Only", train_pool[train_pool['is_stable']]),
            ("Full_Timeline", train_pool)
        ]

        for mode_name, train_df in modes:
            # 1. Fitting
            x_train, y_train = train_df['oxygen'].values, train_df['mean_I'].values

            # Physics Fit
            max_sig = np.max(y_train)
            p0_ts = [max_sig * 1.5, 0.7, 0.001, 0.0001]
            bounds_ts = ([max_sig * 1.01, 0, 0, 0], [np.inf, 1.0, 1.0, 1.0])
            try:
                popt_ts, _ = curve_fit(two_site_model, x_train, y_train, p0=p0_ts, bounds=bounds_ts)
            except:
                popt_ts = p0_ts

            # Polynomial Fit (3rd Degree)
            popt_poly = np.polyfit(y_train, x_train, 3)

            # 2. Inference
            out_df = test_df.copy()
            out_df['o2_pred_twosite'] = inverse_two_site_model(out_df['mean_I'], *popt_ts)
            out_df['o2_pred_poly'] = poly_model_predict(out_df['mean_I'], popt_poly)

            # 3. Save Fitting Visualization
            # This shows how well the model captures the training cloud
            plt.figure(figsize=(10, 6))
            plt.scatter(y_train, x_train, color='gray', alpha=0.1, s=1, label=f'Train Data ({mode_name})')
            sig_range = np.linspace(y_train.min(), y_train.max(), 100)
            plt.plot(sig_range, inverse_two_site_model(sig_range, *popt_ts), 'r-', lw=2, label='Two-Site Physics')
            plt.plot(sig_range, poly_model_predict(sig_range, popt_poly), 'b--', lw=2, label='3rd-Order Poly')
            plt.title(f"Fit Comparison [{mode_name}]: {test_id}")
            plt.xlabel("Intensity");
            plt.ylabel("Oxygen");
            plt.legend();
            plt.grid(True, alpha=0.2)
            plt.savefig(os.path.join(BASE_OUTPUT, mode_name, "Calibration_Fits", f"Fit_{test_id}.png"))
            plt.close()

            # 4. Save Inference Visualization
            plt.figure(figsize=(12, 6))
            plt.plot(out_df['elapsed'], out_df['oxygen'], 'k-', label='GT', lw=2)
            plt.plot(out_df['elapsed'], out_df['o2_pred_twosite'], 'r--', label='Physics', alpha=0.8)
            plt.plot(out_df['elapsed'], out_df['o2_pred_poly'], 'b:', label='Polynomial', alpha=0.8)
            for (s, e) in ANALYSIS_INTERVALS_S: plt.axvspan(s, e, color='green', alpha=0.08)
            plt.title(f"Inference [{mode_name}]: {test_id}")
            plt.xlabel("Seconds");
            plt.ylabel("Oxygen");
            plt.legend()
            plt.savefig(os.path.join(BASE_OUTPUT, mode_name, "Inference_Plots", f"Inf_{test_id}.png"))
            plt.close()

            # 5. Save CSV
            out_df.to_csv(os.path.join(BASE_OUTPUT, mode_name, "CSVs", f"Result_{test_id}.csv"), index=False)


if __name__ == "__main__":
    main()
