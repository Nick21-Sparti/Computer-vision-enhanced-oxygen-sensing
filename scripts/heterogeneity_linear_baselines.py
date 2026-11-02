# Repository edition: see docs/CHANGES.md and docs/REPRODUCIBILITY.md.
"""
==============================================================================
PROJECT: PINN-IDC (Inverse Dynamic Compensation) Framework
VERSION: 11.2.1 — COMPARATORS ONLY (Linear Ablations & Classical Baselines)
==============================================================================

--- SCIENTIFIC CONTEXT & GOAL ---
This script evaluates the LINEAR and CLASSICAL baselines to act as rigorous
controls against the deep-learning (TViT) approach.
The PyTorch/TViT training code has been completely removed to make this a fast,
CPU-only script. It automatically reads the existing TViT results from the
`Global_Group_Performance.csv` and merges the new baseline metrics alongside
them, outputting combined comparison plots.

ADDED VALUE (V11.2.0):
1. NO PYTORCH DEPENDENCY: Strictly runs the Scikit-Learn ridge regression and
   classical physics models.
2. AI LINEAR DECODER ABLATION: A Ridge Regression model applied to the exact
   same temporally stacked pixel features that the TViT received.
3. CLASSICAL OPTIMAL WEIGHTING: A least-squares fit (Ridge on 1/I) mapped to O2,
   calibrated ONLY on stable intervals, acting as the strongest possible
   classical control (optimal weighting rather than uniform mean).
4. SEAMLESS MERGING: Preserves existing TViT (`MAE_AI`) metrics and overlays
   them in Figures B and C for direct visual comparison of Deep Dynamics vs
   Optimal Linear Spatial Weights.

ADDED VALUE (V11.2.1 - corrections before quoting any number):
1. ROBUST TWO-SITE SV (bug fix): the V10.9 inversion divided by (2A + 1e-9).
   When the fit drives K1 or K2 to its bound of 0, A = 0 exactly and the
   numerator -B + sqrt(B^2) = 0, so EVERY prediction was 0 and MAE_Classical
   collapsed to the mean reference O2 (~173.8 umol/L, identical across groups).
   Replaced by the V11.1.0 calibration: multi-start fit and the analytic
   single-site limit o2 = -C/B when A -> 0.
2. SAME PIXELS AS THE TViT: groups are now LOADED from the saved
   Group_Definitions (the ones the TViT was trained on) instead of rebuilt;
   the rebuilt groups are compared against them and any mismatch is reported.
3. DAY-WISE RIDGE REGULARISATION: alpha is chosen by leave-one-TRAINING-day-out
   CV instead of RidgeCV's leave-one-frame-out, which under-regularises on
   strongly autocorrelated frames. Selected alphas are written to the summary.
   (RIDGE_CV_BY_DAY = False restores the V11.2.0 behaviour.)
4. NO OVERWRITING OF TViT EXPORTS: per-day exports are written as
   Inference_Linear_<day>.csv/.png; Global_Group_Performance.csv is backed up
   (timestamped) before it is modified.
==============================================================================
"""

# Portable repository paths; edit scripts/repo_config.py for your computer.
from repo_config import data_path, output_path, cache_path, external_path

import os
import sys
import glob
import re
import warnings
import random
import traceback
import shutil
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. REPRODUCIBILITY & PERFORMANCE CONFIG
# ==============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['axes.grid'] = False

# ==============================================================================
# 2. CONFIGURATION & PATHS
# ==============================================================================
BASE_PROJECT_DIR = data_path('aqueous')
EXPERIMENTS_ROOT_DIR = BASE_PROJECT_DIR

N_PIXELS_PER_GROUP = 16
N_DAYS_CAP = 13
N_GROUPS = 8

GROUP_VISUAL_TIME_S = 1500.0

W_INTENSITY = 1.0
W_R2 = 1.0

FILTER_BY_SV_R2 = True
SV_R2_POOL_THRESHOLD = 0.95

GROUP_MODE = 'expanding'
INCLUDE_T90_IN_FEATURES = True
T90_PIXEL_CACHE_DIR = external_path("pixel_t90")

# Shared parameters — MUST match previous TViT runs exactly
ANALYSIS_INTERVALS_S = [(1450, 1650), (2300, 2500), (3100, 3300),
                        (3900, 4100), (4700, 4900)]
UNSTRUCTURED_CUTOFF_S = 1000

N_STACKED_FRAMES = 3
FRAME_STACK_STEP = 10
STORAGE_DIM = (48, 48)
N_PIXELS_TOTAL = STORAGE_DIM[0] * STORAGE_DIM[1]
RED_CH = 2
MIN_FRAMES_IN_WINDOW = 5

# ---- V11.2.1 correctness switches ----
USE_SAVED_GROUP_DEFINITIONS = True       # use the groups the TViT was trained on
RIDGE_CV_BY_DAY = True                   # choose ridge alpha by leave-one-training-day-out
RIDGE_ALPHAS = np.logspace(-3, 5, 9)     # wider grid: frames are many and correlated
BACKUP_SUMMARY = True                    # timestamped copy of Global_Group_Performance.csv
LINEAR_EXPORT_PHASE = "Inference_Linear" # per-day export name (TViT exports untouched)

FEATURE_CACHE_DIR = cache_path('aqueous_heterogeneity', "features")
MASTER_PARQUET = os.path.join(FEATURE_CACHE_DIR, "idc_master_v10.parquet")

STUDY_DIR = output_path("heterogeneity_quality_groups")
GROUPDEF_DIR = os.path.join(STUDY_DIR, "Group_Definitions")
SUMMARY_DIR = os.path.join(STUDY_DIR, "Summary")
PIXEL_FEATURES_PATH = os.path.join(GROUPDEF_DIR, "pixel_features.npz")
GLOBAL_SUMMARY_PATH = os.path.join(SUMMARY_DIR, "Global_Group_Performance.csv")

for d in [STUDY_DIR, GROUPDEF_DIR, SUMMARY_DIR, FEATURE_CACHE_DIR]:
    os.makedirs(d, exist_ok=True)


def group_dir(g: int) -> str:
    d = os.path.join(STUDY_DIR, f"G{g:02d}")
    for sub in ["2_Inference_Results", "3_Chemometrics", "csv_exports"]:
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    return d


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
                valid_exps.append({"id": exp_id, "video_path": vids[0],
                                   "arduino_path": arduino_logs[0],
                                   "temp_path": temp_logs[0]})
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
                    except:
                        continue
    except:
        pass
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


def _decode_pixels_series(s: pd.Series) -> pd.Series:
    if len(s) == 0: return s
    first = s.iloc[0]
    if isinstance(first, bytes): return s.apply(lambda x: np.frombuffer(x, dtype=np.uint8))
    if isinstance(first, np.ndarray): return s.apply(lambda x: x.astype(np.uint8, copy=False))
    return s.apply(lambda x: np.asarray(x, dtype=np.uint8))


def _read_parquet_safe(cache_path: str) -> pd.DataFrame:
    try:
        import pyarrow.parquet as pq
    except Exception:
        df = pd.read_parquet(cache_path)
        df['pixels'] = _decode_pixels_series(df['pixels'])
        return df

    pf = pq.ParquetFile(cache_path)
    n_rg = pf.num_row_groups
    print(f"[CACHE] Streaming parquet in {n_rg} row group(s)...")
    parts = []
    for batch in tqdm(pf.iter_batches(batch_size=4096), desc="Reading parquet"):
        pdf = batch.to_pandas()
        pdf['pixels'] = _decode_pixels_series(pdf['pixels'])
        parts.append(pdf)
    return pd.concat(parts, ignore_index=True)


def get_master_df(experiments, cache_path):
    if os.path.exists(cache_path):
        return _read_parquet_safe(cache_path)

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
    save_df = master_df.copy()
    save_df['pixels'] = save_df['pixels'].apply(lambda a: a.tobytes())
    save_df.to_parquet(cache_path, index=False)
    return master_df


# ==============================================================================
# 4. PER-PIXEL FEATURES
# ==============================================================================
def _red_matrix(exp_df: pd.DataFrame) -> tuple:
    exp_df = exp_df.sort_values('elapsed').reset_index(drop=True)
    stacked = np.stack(exp_df['pixels'].values)
    red_idx = np.arange(RED_CH, N_PIXELS_TOTAL * 3, 3)
    return (exp_df['elapsed'].values.astype(np.float32),
            exp_df['oxygen'].values.astype(np.float32),
            stacked[:, red_idx].astype(np.float32))


def _vectorised_sv_fit(interval_I: np.ndarray, interval_O2: np.ndarray) -> tuple:
    i0_idx = int(np.argmin(interval_O2))
    I0 = np.maximum(interval_I[i0_idx], 1.0)
    Y = I0[np.newaxis, :] / np.clip(interval_I, 1.0, None)
    x = interval_O2.astype(np.float64)

    xm = x.mean()
    ym = Y.mean(axis=0)
    Sxx = float(np.sum((x - xm) ** 2))
    Sxy = np.sum((x - xm)[:, None] * (Y - ym[None, :]), axis=0)
    slope = Sxy / max(Sxx, 1e-12)
    intercept = ym - slope * xm
    Y_hat = intercept[None, :] + slope[None, :] * x[:, None]
    ss_res = np.sum((Y - Y_hat) ** 2, axis=0)
    ss_tot = np.sum((Y - ym[None, :]) ** 2, axis=0)
    r2 = np.where(ss_tot > 1e-10, 1.0 - ss_res / ss_tot, 0.0)
    return slope.astype(np.float32), np.clip(r2, 0.0, 1.0).astype(np.float32)


def compute_pixel_features(df: pd.DataFrame, day_list: list) -> dict:
    key = "|".join(day_list) + f"|t90={INCLUDE_T90_IN_FEATURES}"
    feat_keys = ['mean_intensity', 'sv_r2', 'sv_ksv', 't90']
    if os.path.exists(PIXEL_FEATURES_PATH):
        cache = np.load(PIXEL_FEATURES_PATH, allow_pickle=True)
        if str(cache.get('key', '')) == key and all(k in cache for k in feat_keys):
            return {k: cache[k] for k in feat_keys}

    per_day_int, per_day_r2, per_day_ksv = [], [], []
    for day in tqdm(day_list, desc="Per-pixel features"):
        elapsed, oxygen, red = _red_matrix(df[df['experiment_id'] == day])
        interval_I, interval_O2 = [], []
        for (s, e) in ANALYSIS_INTERVALS_S:
            m = (elapsed >= s) & (elapsed <= e)
            if m.sum() < MIN_FRAMES_IN_WINDOW: continue
            interval_I.append(np.median(red[m], axis=0))
            interval_O2.append(float(np.median(oxygen[m])))
        if len(interval_I) < 2: continue
        interval_I = np.array(interval_I)
        interval_O2 = np.array(interval_O2)
        per_day_int.append(interval_I.mean(axis=0))
        ksv, r2 = _vectorised_sv_fit(interval_I, interval_O2)
        per_day_ksv.append(ksv);
        per_day_r2.append(r2)

    t90 = np.full(N_PIXELS_TOTAL, np.nan, dtype=np.float32)
    if INCLUDE_T90_IN_FEATURES:
        per_day_t90 = []
        for day in day_list:
            p = os.path.join(T90_PIXEL_CACHE_DIR, f"t90_cache_{day}.npz")
            if not os.path.exists(p):
                hits = glob.glob(os.path.join(BASE_PROJECT_DIR, "**", f"t90_cache_{day}.npz"), recursive=True)
                if hits:
                    p = hits[0]
                else:
                    continue
            try:
                d = np.load(p, allow_pickle=True)
                per_day_t90.append(np.asarray(d['T90_mean'], dtype=np.float32).flatten())
            except Exception:
                pass
        if per_day_t90: t90 = np.nanmean(np.stack(per_day_t90), axis=0).astype(np.float32)

    features = {
        'mean_intensity': np.nanmean(np.stack(per_day_int), axis=0).astype(np.float32),
        'sv_r2': np.nan_to_num(np.nanmean(np.stack(per_day_r2), axis=0), nan=0.0).astype(np.float32),
        'sv_ksv': np.nan_to_num(np.nanmean(np.stack(per_day_ksv), axis=0), nan=0.0).astype(np.float32),
        't90': t90,
    }
    np.savez_compressed(PIXEL_FEATURES_PATH, key=key, **features)
    return features


# ==============================================================================
# 5. GROUP CONSTRUCTION & HETEROGENEITY METRICS
# ==============================================================================
def _stratified_sample(start: int, end: int, n: int) -> np.ndarray:
    idx = np.round(np.linspace(start, end - 1, n)).astype(int)
    used, out = set(), []
    for v in idx:
        v = int(np.clip(v, start, end - 1))
        step = 0
        while v in used:
            step += 1
            for cand in (v + step, v - step):
                if start <= cand < end and cand not in used:
                    v = cand;
                    break
            else:
                continue
            break
        used.add(v);
        out.append(v)
    return np.array(sorted(out), dtype=int)


def build_groups(features: dict) -> tuple:
    def _z(v):
        s = v.std()
        return (v - v.mean()) / (s if s > 1e-12 else 1.0)

    quality = (W_INTENSITY * _z(features['mean_intensity']) + W_R2 * _z(features['sv_r2']))
    eligible = np.where(features['sv_r2'] >= SV_R2_POOL_THRESHOLD)[0] if FILTER_BY_SV_R2 else np.arange(N_PIXELS_TOTAL)

    rank_order = eligible[np.argsort(-quality[eligible])]
    M = len(eligible);
    N = N_PIXELS_PER_GROUP

    def _z_nan(v):
        v = np.asarray(v, dtype=np.float64)
        mval = np.nanmean(v);
        sval = np.nanstd(v)
        z = (v - mval) / (sval if sval > 1e-12 else 1.0)
        z[~np.isfinite(z)] = 0.0
        return z

    feat_cols = [_z(features['mean_intensity']), _z(features['sv_r2']), _z(features['sv_ksv'])]
    t90_used = False
    if INCLUDE_T90_IN_FEATURES and 't90' in features and np.any(np.isfinite(features['t90'])):
        feat_cols.append(_z_nan(features['t90']));
        t90_used = True
    feat_mat = np.stack(feat_cols, axis=1)

    q_pct_pool = np.empty(N_PIXELS_TOTAL, dtype=np.float32)
    q_pct_pool[:] = np.nan
    order_in_pool = np.argsort(np.argsort(quality[eligible]))
    q_pct_pool[eligible] = 100.0 * order_in_pool / max(M - 1, 1)

    def _mpd_of(pix):
        return float(pdist(feat_mat[pix]).mean()) if len(pix) > 1 else 0.0

    cand_ends = np.unique(np.round(np.linspace(N, M, max(60, N_GROUPS * 8))).astype(int))
    cand_mpds = np.array([_mpd_of(rank_order[_stratified_sample(0, int(e), N)]) for e in cand_ends])
    run_max = np.maximum.accumulate(cand_mpds)
    is_record = cand_mpds >= run_max - 1e-12
    rec_ends = cand_ends[is_record]
    rec_mpds = cand_mpds[is_record]
    mpd_lo, mpd_hi = float(rec_mpds[0]), float(rec_mpds[-1])
    targets = np.linspace(mpd_lo, mpd_hi, N_GROUPS)
    sel, prev = [], -1
    for t in targets:
        i = int(np.argmin(np.abs(rec_mpds - t)))
        if i <= prev: i = min(len(rec_ends) - 1, prev + 1)
        sel.append(i);
        prev = i
    windows = [(0, int(rec_ends[i])) for i in sel]

    groups, rows = [], []
    for g, (start, end) in enumerate(windows, start=1):
        ranks = _stratified_sample(start, end, N)
        pix = rank_order[ranks]
        groups.append(pix)
        row = {
            'group': g, 'window_start_rank': start, 'window_end_rank': end,
            'n_pixels': len(pix), 'mpd': _mpd_of(pix), 'q_sd': float(quality[pix].std()),
            'q_range': float(q_pct_pool[pix].max() - q_pct_pool[pix].min()),
            'q_mean': float(quality[pix].mean()),
            'mean_intensity_mean': float(features['mean_intensity'][pix].mean()),
            'sv_r2_mean': float(features['sv_r2'][pix].mean()),
        }
        if t90_used:
            t90_pix = features['t90'][pix]
            row['t90_mean'] = float(np.nanmean(t90_pix)) if np.any(np.isfinite(t90_pix)) else np.nan
        rows.append(row)
    metrics_df = pd.DataFrame(rows)
    return groups, metrics_df, quality, rank_order


# ==============================================================================
# 6. FEATURE EXTRACTION FOR LINEAR ABLATIONS
# ==============================================================================
def get_red_matrix_and_group_intensities(df: pd.DataFrame, groups: list) -> tuple:
    print("[PRE-COMPUTE] Extracting global RED matrix and uniform group intensities...")
    M = N_PIXELS_TOTAL
    member = np.zeros((M, len(groups)), dtype=np.float32)
    for gi, pix in enumerate(groups):
        member[pix, gi] = 1.0 / len(pix)

    red_idx = np.arange(RED_CH, M * 3, 3)
    red_mat = np.empty((len(df), M), dtype=np.float32)
    group_means = np.empty((len(df), len(groups)), dtype=np.float32)

    CH = 2000
    for s in tqdm(range(0, len(df), CH), desc="Frame Processing"):
        e = min(s + CH, len(df))
        red = np.stack(df['pixels'].iloc[s:e].values)[:, red_idx].astype(np.float32)
        red_mat[s:e] = red
        group_means[s:e] = red @ member

    for gi in range(len(groups)):
        df[f'inv_i_g{gi + 1}'] = 1.0 / (group_means[:, gi] + 1e-6)

    return df, red_mat


def build_stacked_features_fast(features: np.ndarray, exp_ids: np.ndarray,
                                stack_size: int, step: int) -> np.ndarray:
    N_frames, N_pix = features.shape
    out = np.empty((N_frames, stack_size * N_pix), dtype=np.float32)
    idx_base = np.arange(N_frames)

    for s in range(stack_size):
        shift = s * step
        idx = idx_base - shift
        idx[idx < 0] = 0
        valid_mask = exp_ids[idx] == exp_ids[idx_base]
        idx = np.where(valid_mask, idx, idx_base)
        out[:, s * N_pix: (s + 1) * N_pix] = features[idx]
    return out


# ==============================================================================
# 7. CLASSICAL BASELINE (Two-Site SV, uniform group intensity)
# ==============================================================================
def fit_classical_baseline(train_df, test_df, inv_col: str):
    """
    Two-site Stern-Volmer on the group-mean intensity (V11.1.0 robust version).
    Calibrated on the stable plateaus of the TRAINING days only.
      * five starting points, best SSE kept (single starts often hit a bound);
      * analytic inversion of  y = f/(1+K1 o) + (1-f)/(1+K2 o),  y = I/I0:
            A o^2 + B o + C = 0,  A = y K1 K2,
            B = y (K1+K2) - (f (K2-K1) + K1),  C = y - 1;
      * when A -> 0 (K1 or K2 at its bound: single-site behaviour) the root is
        the linear limit o = -C/B. The V10.9 code divided by (2A + 1e-9)
        instead, which returned o = 0 for every frame in exactly this case.
    """
    calib_df = test_df if len(train_df) == 0 else train_df
    stable = pd.concat([calib_df[(calib_df['elapsed'] >= s) & (calib_df['elapsed'] <= e)]
                        for s, e in ANALYSIS_INTERVALS_S])
    if stable.empty:
        stable = calib_df
    I = 1.0 / stable[inv_col].to_numpy(float)
    o2 = stable['oxygen'].to_numpy(float)
    ok = np.isfinite(I) & np.isfinite(o2) & (I > 0)
    I, o2 = I[ok], o2[ok]
    if len(I) == 0:
        return np.full(len(test_df), np.nan, dtype=float)

    def sv(o, I0, f, K1, K2):
        return I0 * (f / (1.0 + K1 * o) + (1.0 - f) / (1.0 + K2 * o))

    I0g, best = float(np.percentile(I, 99)), None
    for p0 in ([I0g, .8, .05, .005], [I0g, .5, .02, .002], [I0g, .2, .1, .005],
               [I0g, .7, .01, .0005], [I0g, .3, .005, .05]):
        try:
            popt, _ = curve_fit(sv, o2, I, p0=p0, bounds=([1e-9, 0, 0, 0], [np.inf, 1, np.inf, np.inf]),
                                maxfev=50000)
            sse = float(np.sum((sv(o2, *popt) - I) ** 2))
            if best is None or sse < best[1]:
                best = (popt, sse)
        except Exception:
            continue
    I_test = 1.0 / test_df[inv_col].to_numpy(float)
    if best is None:   # linear fallback in 1/I
        reg = LinearRegression().fit((1.0 / I).reshape(-1, 1), o2)
        return reg.predict((1.0 / I_test).reshape(-1, 1))
    I0, f, K1, K2 = best[0]
    y = I_test / I0
    A = y * K1 * K2
    B = y * (K1 + K2) - (f * (K2 - K1) + K1)
    C = y - 1.0
    sq = np.sqrt(np.maximum(B ** 2 - 4.0 * A * C, 0.0))
    den = np.where(np.abs(2.0 * A) < 1e-12, np.nan, 2.0 * A)
    r1, r2 = (-B - sq) / den, (-B + sq) / den
    pred = np.where(np.isfinite(r1) & (r1 >= 0), r1, r2)
    lin = -C / np.where(np.abs(B) < 1e-12, np.nan, B)   # A -> 0: single-site limit
    return np.where(np.isfinite(pred), pred, lin).astype(float)


def fit_ridge_day_cv(X: np.ndarray, y: np.ndarray, days: np.ndarray):
    """
    StandardScaler + Ridge with alpha chosen by leave-one-TRAINING-day-out MAE.
    Frames within a day are strongly autocorrelated, so RidgeCV's efficient
    leave-one-frame-out criterion picks too little regularisation. Only the
    training rows passed in are used - the outer test day never enters.
    Returns (fitted pipeline, selected alpha).
    """
    uniq = np.unique(days)
    if not RIDGE_CV_BY_DAY or len(uniq) < 2:
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS)).fit(X, y)
        return m, float(m[-1].alpha_)
    best_a, best_err = None, np.inf
    for a in RIDGE_ALPHAS:
        errs = []
        for d in uniq:
            tr, va = days != d, days == d
            m = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(X[tr], y[tr])
            errs.append(mean_absolute_error(y[va], m.predict(X[va])))
        if np.mean(errs) < best_err:
            best_err, best_a = float(np.mean(errs)), float(a)
    return make_pipeline(StandardScaler(), Ridge(alpha=best_a)).fit(X, y), best_a


def select_group_definitions(groups_rebuilt, metrics_rebuilt):
    """
    Use the group definitions the TViT was trained on (Group_Definitions/),
    so every comparator is evaluated on exactly the same pixels. The groups
    rebuilt from the current features are compared against them and any
    difference is reported - a mismatch means the settings or caches changed
    since the TViT run.
    """
    a_csv = os.path.join(GROUPDEF_DIR, "group_assignments.csv")
    m_csv = os.path.join(GROUPDEF_DIR, "group_metrics.csv")
    if not (USE_SAVED_GROUP_DEFINITIONS and os.path.exists(a_csv) and os.path.exists(m_csv)):
        print("[GROUPS] Using REBUILT groups (saved definitions not used/found).")
        return groups_rebuilt, metrics_rebuilt, "rebuilt"
    assign = pd.read_csv(a_csv)
    metrics = pd.read_csv(m_csv).sort_values('group').reset_index(drop=True)
    groups = [assign.loc[assign['group'] == g, 'pixel_idx'].astype(int).to_numpy()
              for g in metrics['group']]
    same = (len(groups) == len(groups_rebuilt) and
            all(set(a) == set(b) for a, b in zip(groups, groups_rebuilt)))
    print(f"[GROUPS] Loaded {len(groups)} saved groups from {GROUPDEF_DIR}.")
    if same:
        print("[GROUPS] Rebuilt groups are IDENTICAL to the saved (TViT) groups.")
    else:
        n_diff = sum(set(a) != set(b) for a, b in zip(groups, groups_rebuilt))
        print(f"[GROUPS] WARNING: rebuilt groups differ from the saved ones in {n_diff} group(s) "
              f"(or in the number of groups). The SAVED groups are used, so the comparators "
              f"match the TViT pixels - but check why the settings/caches changed.")
    return groups, metrics, "saved"


# ==============================================================================
# 8. PER-FOLD ANALYTICS
# ==============================================================================
def plot_chemometrics(df, exp_id, out_dir, pred_col='o2_ai_linear', title="AI Linear Ablation"):
    y_true, y_pred = df['oxygen'], df[pred_col]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    ax1.scatter(y_true, y_pred, alpha=0.3, s=15, c='blue')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax1.plot(lims, lims, 'k--', lw=2)
    ax1.set_xlabel("Ground Truth Oxygen");
    ax1.set_ylabel("Predicted Oxygen")
    ax1.set_title(f"{title} Parity Plot: {exp_id} ($R^2={r2_score(y_true, y_pred):.3f}$)")
    residuals = y_pred - y_true
    sns.histplot(residuals, kde=True, ax=ax2, color='purple')
    ax2.set_xlabel("Error (Pred - True)")
    ax2.set_title(f"Residual Distribution (MAE: {mean_absolute_error(y_true, y_pred):.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "3_Chemometrics", f"Chemometrics_{exp_id}.png"))
    plt.close()


def analyze_dynamics_v10(df, exp_id, out_dir, phase="Inference"):
    plot_df = df.copy()
    fig, ax = plt.subplots(figsize=(16, 9))
    t = plot_df['elapsed']
    ax.plot(t, plot_df['oxygen'], label='Ground Truth', color='black', lw=3)

    if 'o2_classical' in plot_df.columns:
        ax.plot(t, plot_df['o2_classical'], label='Classical (Uniform Mean SV)', color='gray', ls='-', alpha=0.8)
    if 'o2_classical_opt' in plot_df.columns:
        ax.plot(t, plot_df['o2_classical_opt'], label='Classical (Opt. Linear Weights)', color='saddlebrown', ls=':',
                alpha=0.9, lw=2)
    if 'o2_ai_linear' in plot_df.columns:
        ax.plot(t, plot_df['o2_ai_linear'], label='AI Ablation (Linear Decoder)', color='cyan', ls='--', alpha=0.9,
                lw=2)

    # Optional: If deep learning TViT output exists, plot it too
    if 'o2_pred' in plot_df.columns:
        ax.plot(t, plot_df['o2_pred'], label='AI Prediction (TViT, masked)', color='blue', lw=2)

    for (start, end) in ANALYSIS_INTERVALS_S:
        ax.axvspan(start, end, color='green', alpha=0.1)
    ax.set_title(f"Sensor Response: {exp_id} ({phase})")
    ax.set_ylabel("Oxygen Concentration");
    ax.set_xlabel("Time (s)")
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "2_Inference_Results", f"{phase}_{exp_id}.png"))
    plt.close()

    keep = [c for c in plot_df.columns if c != 'pixels']
    plot_df[keep].to_csv(os.path.join(out_dir, "csv_exports", f"{phase}_{exp_id}.csv"), index=False)


# ==============================================================================
# 9. SUMMARY FIGURES (Auto-merges TViT results if they exist)
# ==============================================================================
def regime_masks(df):
    stable = np.zeros(len(df), dtype=bool)
    el = df['elapsed'].values
    for s, e in ANALYSIS_INTERVALS_S: stable |= (el >= s) & (el <= e)
    unstruct = el < UNSTRUCTURED_CUTOFF_S
    transition = (~stable) & (~unstruct)
    return stable, transition, unstruct


def _safe_mae(y, p, m):
    return float(mean_absolute_error(y[m], p[m])) if m.sum() > 0 else np.nan


def plot_summary(summary_df, metrics_df):
    cols_to_mean = [
        'MAE_AI_Linear', 'MAE_Classical', 'MAE_Classical_OptWeight',
        'MAE_AI_Linear_stable', 'MAE_AI_Linear_transition', 'MAE_AI_Linear_unstructured'
    ]
    # Optionally include TViT columns if they exist in the merged dataframe
    for col in ['MAE_AI', 'MAE_AI_stable', 'MAE_AI_transition', 'MAE_AI_unstructured']:
        if col in summary_df.columns:
            cols_to_mean.append(col)

    agg = (summary_df.groupby('group')[cols_to_mean].mean().reset_index()
           .merge(metrics_df[['group', 'mpd', 'q_sd', 'q_range']], on='group'))
    agg.to_csv(os.path.join(SUMMARY_DIR, "Fig_B_C_source_data.csv"), index=False)

    def _norm(v):
        v = np.asarray(v, float)
        rng = np.nanmax(v) - np.nanmin(v)
        return (v - np.nanmin(v)) / (rng if rng > 1e-12 else 1.0)

    # ---- Figure B: MAE vs heterogeneity (with Ablations) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    lines = [
        ('MAE_AI_Linear', 'cyan', 'AI Ablation (Linear Stack Decoder)', '--'),
        ('MAE_Classical', 'gray', 'Classical (Uniform SV)', '-'),
        ('MAE_Classical_OptWeight', 'saddlebrown', 'Classical (Least-Squares weights)', ':')
    ]
    if 'MAE_AI' in agg.columns:
        lines.insert(0, ('MAE_AI', 'royalblue', 'AI (TViT, Non-linear dynamics)', '-'))

    for col, colr, lbl, ls in lines:
        if col in agg.columns:
            ax1.plot(agg['mpd'], agg[col], marker='o', color=colr, label=lbl, linestyle=ls, lw=2)
            ax2.plot(_norm(agg['mpd']), _norm(agg[col]), marker='o', color=colr, label=lbl, linestyle=ls, lw=2)

    for _, r in agg.iterrows():
        y_anno = r['MAE_AI'] if 'MAE_AI' in r else r['MAE_AI_Linear']
        if not np.isnan(y_anno):
            ax1.annotate(f"G{int(r['group']):02d}", (r['mpd'], y_anno),
                         fontsize=7, xytext=(3, 3), textcoords='offset points')

    ax1.set_xlabel("Heterogeneity — Mean Pairwise Distance (MPD)")
    ax1.set_ylabel("MAE (LOOCV mean)")
    ax1.set_title("Absolute Error");
    ax1.legend()
    ax2.set_xlabel("Normalised MPD (0–1)");
    ax2.set_ylabel("Normalised MAE (0–1)")
    ax2.set_title("Min-max Normalised");
    ax2.legend()
    fig.suptitle(f"Reconstruction error vs Heterogeneity: Deep AI vs Linear & Classical Baselines", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(SUMMARY_DIR, "Fig_B_MAE_vs_Heterogeneity.png"), dpi=250, bbox_inches='tight')
    plt.close()

    # ---- Figure C: regime-decomposed MAE ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

    # Left Plot: Choose to plot TViT regimes (if exist) OR Linear Regimes
    regime_prefix = 'MAE_AI' if 'MAE_AI_stable' in agg.columns else 'MAE_AI_Linear'

    for suffix, colr, lbl in [('_stable', 'green', 'Stable plateaux'),
                              ('_transition', 'crimson', 'Transitions'),
                              ('_unstructured', 'darkorange', 'Unstructured (<1000 s)')]:
        col = regime_prefix + suffix
        ax1.plot(agg['mpd'], agg[col], 'o-', color=colr,
                 label=f"{lbl} ({'TViT' if regime_prefix == 'MAE_AI' else 'Linear'})")
    ax1.set_xlabel("MPD");
    ax1.set_ylabel("MAE (LOOCV mean)")
    ax1.set_title("Error by dynamic regime");
    ax1.legend()

    # Right Plot: Penalty Ratios
    if 'MAE_AI_transition' in agg.columns:
        ratio_ai = agg['MAE_AI_transition'] / agg['MAE_AI_stable']
        ax2.plot(agg['mpd'], ratio_ai, 's-', color='purple', label='TViT (Non-linear)')

    ratio_lin = agg['MAE_AI_Linear_transition'] / agg['MAE_AI_Linear_stable']
    ax2.plot(agg['mpd'], ratio_lin, '^-', color='cyan', label='Linear Decoder')
    ax2.axhline(1.0, color='k', ls=':', lw=1)
    ax2.set_xlabel("MPD");
    ax2.set_ylabel("Transition MAE / Stable MAE")
    ax2.set_title("Dynamics Penalty Ratio (lower = better at transients)")
    ax2.legend()
    fig.suptitle("Dynamics-learning evidence: where does heterogeneity help?", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(SUMMARY_DIR, "Fig_C_Regime_MAE.png"), dpi=250, bbox_inches='tight')
    plt.close()

    # ---- Figure D: heatmap (Linear model as fallback if TViT missing) ----
    hm_col = 'MAE_AI' if 'MAE_AI' in summary_df.columns else 'MAE_AI_Linear'
    piv = summary_df.pivot_table(index='group', columns='fold_day', values=hm_col)
    fig, ax = plt.subplots(figsize=(1.1 * max(len(piv.columns), 4) + 3, 0.55 * N_GROUPS + 2))
    sns.heatmap(piv, annot=True, fmt=".3f", cmap='viridis_r', cbar_kws={'label': f'{hm_col}'}, ax=ax)
    ax.set_title(f"{hm_col} per group × held-out day")
    plt.tight_layout()
    plt.savefig(os.path.join(SUMMARY_DIR, "Fig_D_GroupFold_Heatmap.png"), dpi=250, bbox_inches='tight')
    plt.close()


# ==============================================================================
# 10. EXECUTION ORCHESTRATOR
# ==============================================================================
def main():
    print("=" * 74)
    print("  PIXEL HETEROGENEITY STUDY V11.2.0 — COMPARATORS ONLY")
    print(f"  N_PIXELS_PER_GROUP={N_PIXELS_PER_GROUP}  N_GROUPS={N_GROUPS}")
    print(f"  saved groups={USE_SAVED_GROUP_DEFINITIONS}  ridge alpha by day={RIDGE_CV_BY_DAY}")
    print("=" * 74)

    experiments = find_experiment_files_v10(EXPERIMENTS_ROOT_DIR)
    df = get_master_df(experiments, MASTER_PARQUET)
    day_list = sorted(df['experiment_id'].unique())
    if N_DAYS_CAP: day_list = day_list[:N_DAYS_CAP]
    df = df[df['experiment_id'].isin(day_list)].reset_index(drop=True)

    exp_ids_array = df['experiment_id'].values
    y_all = df['oxygen'].values
    elapsed_all = df['elapsed'].values

    features = compute_pixel_features(df, day_list)
    groups_rebuilt, metrics_rebuilt, quality, rank_order = build_groups(features)
    groups, metrics_df, group_source = select_group_definitions(groups_rebuilt, metrics_rebuilt)

    df, red_mat = get_red_matrix_and_group_intensities(df, groups)

    # --- Load existing CSV to preserve TViT results ---
    summary_dict = {}
    if BACKUP_SUMMARY and os.path.exists(GLOBAL_SUMMARY_PATH):
        bak = GLOBAL_SUMMARY_PATH.replace(".csv", f"_backup_{datetime.now():%Y%m%d_%H%M%S}.csv")
        shutil.copy2(GLOBAL_SUMMARY_PATH, bak)
        print(f"[BACKUP] {bak}")
    if os.path.exists(GLOBAL_SUMMARY_PATH):
        print(f"[MERGE] Loading existing summary file to preserve deep learning metrics...")
        existing_df = pd.read_csv(GLOBAL_SUMMARY_PATH)
        for _, r in existing_df.iterrows():
            summary_dict[(r['group'], r['fold_day'])] = r.to_dict()

    for g, pix in enumerate(groups, start=1):
        out_dir = group_dir(g)
        gm = metrics_df.loc[metrics_df['group'] == g].iloc[0]
        print(f"\n{'=' * 66}\n[GROUP G{g:02d}] MPD={gm['mpd']:.3f}\n{'=' * 66}")

        group_red = red_mat[:, pix]
        group_inv_I = 1.0 / np.clip(group_red, 1.0, None)
        stacked_ai_features = build_stacked_features_fast(group_red / 255.0, exp_ids_array, N_STACKED_FRAMES,
                                                          FRAME_STACK_STEP)

        for fold_idx, test_day in enumerate(day_list):
            print(f"\n  Fold {fold_idx + 1}/{len(day_list)}: hold out {test_day}")

            train_mask = (exp_ids_array != test_day)
            test_mask = (exp_ids_array == test_day)
            train_df = df[train_mask].copy()
            test_df = df[test_mask].copy()

            # --- 1. Classical Uniform Mean ---
            print("  > Fitting Classical Uniform Mean...")
            test_df['o2_classical'] = fit_classical_baseline(train_df, test_df, inv_col=f'inv_i_g{g}')

            # --- 2. Classical Optimal Weights (Ridge on stable intervals) ---
            print("  > Fitting Classical Optimal Weights...")
            stable_train_mask = train_mask.copy()
            stable_train_mask[:] = False
            for st, en in ANALYSIS_INTERVALS_S:
                stable_train_mask |= (train_mask & (elapsed_all >= st) & (elapsed_all <= en))
            if stable_train_mask.sum() == 0: stable_train_mask = test_mask.copy()

            ridge_opt, alpha_opt = fit_ridge_day_cv(group_inv_I[stable_train_mask], y_all[stable_train_mask],
                                                    exp_ids_array[stable_train_mask])
            test_df['o2_classical_opt'] = ridge_opt.predict(group_inv_I[test_mask])

            # --- 3. AI Linear Ablation (Ridge on Stacked Frames) ---
            print("  > Fitting AI Linear Decoder Ablation...")
            train_mask_lin = train_mask if train_mask.sum() > 0 else test_mask
            ridge_ai, alpha_ai = fit_ridge_day_cv(stacked_ai_features[train_mask_lin], y_all[train_mask_lin],
                                                  exp_ids_array[train_mask_lin])
            test_df['o2_ai_linear'] = ridge_ai.predict(stacked_ai_features[test_mask])

            # Check if TViT predictions exist on disk for the dynamics plot
            csv_path = os.path.join(out_dir, "csv_exports", f"Inference_{test_day}.csv")
            if os.path.exists(csv_path):
                old_inf = pd.read_csv(csv_path)
                if 'o2_pred' in old_inf.columns:
                    test_df['o2_pred'] = old_inf['o2_pred'].values

            analyze_dynamics_v10(test_df, test_day, out_dir, phase=LINEAR_EXPORT_PHASE)
            plot_chemometrics(test_df, test_day, out_dir, pred_col='o2_ai_linear', title="AI Linear Ablation")

            y = test_df['oxygen'].values
            pc = test_df['o2_classical'].values
            plin = test_df['o2_ai_linear'].values
            popt = test_df['o2_classical_opt'].values

            m_st, m_tr, m_un = regime_masks(test_df)

            # Fetch existing row to preserve deep learning MAEs if they exist
            row = summary_dict.get((g, test_day), {})
            row.update({
                'group': g, 'fold_day': test_day,
                'mpd': float(gm['mpd']), 'q_sd': float(gm['q_sd']), 'q_range': float(gm['q_range']),
                'group_source': group_source,
                'ridge_alpha_opt_weight': alpha_opt, 'ridge_alpha_ai_linear': alpha_ai,

                'MAE_AI_Linear': float(mean_absolute_error(y, plin)),
                'MAE_Classical': float(mean_absolute_error(y, pc)),
                'MAE_Classical_OptWeight': float(mean_absolute_error(y, popt)),

                'MAE_AI_Linear_stable': _safe_mae(y, plin, m_st),
                'MAE_AI_Linear_transition': _safe_mae(y, plin, m_tr),
                'MAE_AI_Linear_unstructured': _safe_mae(y, plin, m_un),

                'MAE_Classical_stable': _safe_mae(y, pc, m_st),
                'MAE_Classical_transition': _safe_mae(y, pc, m_tr),

                'MAE_Classical_Opt_stable': _safe_mae(y, popt, m_st),
                'MAE_Classical_Opt_transition': _safe_mae(y, popt, m_tr),
            })
            summary_dict[(g, test_day)] = row

            # Save progress iteratively
            pd.DataFrame(list(summary_dict.values())).to_csv(GLOBAL_SUMMARY_PATH, index=False)

    summary_df = pd.DataFrame(list(summary_dict.values()))
    if len(summary_df) > 0:
        print(f"\n[SUMMARY] Generating updated Figures B, C, D...")
        plot_summary(summary_df, metrics_df)

    print("\n" + "=" * 74 + "\n  HETEROGENEITY STUDY (BASELINES) COMPLETE\n" + "=" * 74)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
