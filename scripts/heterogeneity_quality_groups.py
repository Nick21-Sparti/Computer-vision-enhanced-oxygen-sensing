# Repository edition: see docs/CHANGES.md and docs/REPRODUCIBILITY.md.
"""
==============================================================================
PROJECT: PINN-IDC (Inverse Dynamic Compensation) Framework
VERSION: 11.0.0 — PIXEL HETEROGENEITY STUDY ("Coded Chemical Aperture" test)
==============================================================================

--- SCIENTIFIC CONTEXT & GOAL ---
HYPOTHESIS UNDER TEST ("Coded Chemical Aperture"):
A spatially HETEROGENEOUS set of sensor pixels — spanning bright/dim,
fast/slow, well-calibrated/poorly-calibrated regions — carries MORE usable
information for a learned spatiotemporal operator (TViT) than a homogeneous
set of classically "excellent" pixels of the same size.  If true, groups
with maximal internal variety should achieve LOWER O2 reconstruction error
than groups of uniformly elite pixels, despite every individual pixel in
the elite group being classically superior.

EXPERIMENTAL DESIGN:
  • All 2304 pixels (48×48 grid) are ranked by a composite QUALITY SCORE
    built from per-pixel features averaged across experiment days:
        quality = z(mean stable RED intensity) + z(SV linear-fit R²)
    (weights configurable; features cached to disk).
  • N_GROUPS = 13 groups of exactly N_PIXELS_PER_GROUP pixels each are
    constructed along a homogeneity→heterogeneity→homogeneity spectrum:
        Group 1  : top-N contiguous block of the ranking
                   (brightest / best-fitting pixels, minimal variety)
        Groups 2–6 : sampling window grows downward from the top;
                   N pixels stratified-sampled evenly within the window
        Group 7  : window = ENTIRE ranking → maximum variety
                   (evenly spans best → worst, including darkest pixels)
        Groups 8–12: window shrinks from the top toward the bottom
        Group 13 : bottom-N contiguous block (worst pixels, minimal variety)
  • HETEROGENEITY METRICS (computed per group, all exported):
        MPD  : Mean Pairwise Euclidean Distance in z-scored feature space
               [mean_intensity, sv_r2, sv_ksv]  ← PRIMARY metric
               (equivalent to Rao's quadratic entropy with uniform weights)
        Q_SD : standard deviation of the composite quality score in-group
        Q_RANGE : span of global quality percentiles covered by the group
  • For each group, the FULL V10.9.0 pipeline is re-run with the input
    restricted to that group's pixels:
        AI (TViT): frames are spatially MASKED — every pixel outside the
                   group is set to 0 before resizing to 224×224.  The
                   architecture, loss, optimiser, epochs, and LOOCV fold
                   structure are UNCHANGED from V10.9.0.
        Classical: the Two-Site Stern-Volmer baseline is refit using the
                   mean RED intensity of the GROUP pixels only (the group
                   analogue of the whole-ROI inv_i).  The fitting and the
                   analytic quadratic inversion are UNCHANGED.
  • LOOCV as before: each experiment day is held out in turn; the model
    trains on the remaining days.  N_DAYS_CAP limits how many days are
    used (calendar order) to control runtime.

--- EVOLUTION OF THE FRAMEWORK (V1.0 -> V11.0.0) ---
- V1-V7: Static PINNs.
- V8.2: Robust file discovery.
- V10.0-10.6: Dynamic sensing, T90 Benchmarking, LOOCV.
- V10.7: Classical Baseline & Segmented Analytics.
- V10.8: Strict "Black Box" Training.
- V10.8.1: Chemometrics Suite (Parity Plots & Residual Histograms).
- V10.9.0: Two-Site Stern-Volmer classical baseline.
- V11.0.0 (CURRENT):
    ADDED VALUE:
    1. PIXEL-GROUP FRAMEWORK: 13 groups along the homogeneity spectrum,
       exact stratified-window construction, three heterogeneity metrics.
    2. MASKED-INPUT TViT: identical model/training, spatially masked input
       → the group IS the aperture; the TViT IS the reconstruction operator.
    3. GROUP-LEVEL CLASSICAL BASELINE: two-site SV refit on group-mean
       intensity for a like-for-like physics benchmark per group.
    4. FULL CHECKPOINTING: every epoch of every (group, fold) model is
       checkpointed and training resumes automatically after interruption;
       finished (group, fold) runs are skipped entirely on re-run.
    5. MODEL PERSISTENCE: final .pth saved per (group, fold).
    6. SUMMARY ANALYTICS: MAE vs heterogeneity (absolute + min-max
       normalised), regime-decomposed MAE (stable / transition /
       unstructured) as direct evidence of dynamics learning, group×fold
       MAE heatmap, per-group quality distributions, group visual panel.

--- FOLDER STRUCTURE & OUTPUTS ---
- Heterogeneity_Study/                       (inside BASE_PROJECT_DIR)
    - Group_Definitions/
        - pixel_features.npz                 (cached per-pixel features)
        - group_assignments.csv              (pixel→group mapping + features)
        - group_metrics.csv                  (heterogeneity metrics per group)
        - Fig_A_Group_Panels.png             (figure a: 13 groups @ ~1500 s)
        - group_masks.npz                    (48×48 boolean mask per group)
    - checkpoints/  ckpt_G{gg}_{day}.pth     (resumable, every epoch)
    - models/       TViT_G{gg}_{day}.pth     (final weights per group/fold)
    - G01/ ... G13/                          (per-group V10.9.0 outputs)
        - 1_Training_Convergence/            (loss curves + History CSVs)
        - 2_Inference_Results/               (time-series overlays)
        - 3_Chemometrics/                    (parity + residuals)
        - csv_exports/  Inference_{day}.csv
    - Summary/
        - Global_Group_Performance.csv       (every group×fold metric)
        - Fig_B_MAE_vs_Heterogeneity.png     (figure b: abs + normalised)
        - Fig_C_Regime_MAE.png               (dynamics-learning evidence)
        - Fig_D_GroupFold_Heatmap.png
        - Fig_E_Quality_Distributions.png
        (+ a CSV for every figure)
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
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')          # headless-safe (cluster compatible)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist
from scipy.optimize import curve_fit
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

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. REPRODUCIBILITY & PERFORMANCE CONFIG (verbatim from V10.9.0)
# ==============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['axes.grid'] = False          # no grid lines on any plot

# ==============================================================================
# 2. CONFIGURATION & PATHS
# ==============================================================================
BASE_PROJECT_DIR = data_path('aqueous')
EXPERIMENTS_ROOT_DIR = BASE_PROJECT_DIR

# ------------------------------------------------------------------
# ★★★ STUDY-SPECIFIC KNOBS — set these before running ★★★
# ------------------------------------------------------------------
N_PIXELS_PER_GROUP = 16     # pixels per group (start small, e.g. 10; paper: 100)
N_DAYS_CAP         = 5   # None = all experiment days; int = first N (calendar)
N_GROUPS           = 8      # groups along the homogeneity spectrum (odd number
                            #   so the middle group is the max-variety one)

# ------------------------------------------------------------------
# ★★★ SPEED OPTIMISATIONS — toggle these; all default ON ★★★
# ------------------------------------------------------------------
# (1) Automatic Mixed Precision: fp16 compute on CUDA. ~1.5-2x, negligible
#     accuracy impact. Silently ignored on CPU.
USE_AMP = True
# (2) Cache resized 224x224 masked tensors per (group,fold) so the expensive
#     uint8->float->mask->resize is done ONCE, not every epoch. Costs RAM.
USE_TENSOR_CACHE = True
# (4) Freeze the pretrained ViT backbone; train only the (modified) patch-embed
#     projection + regression head. Far fewer params -> faster + fewer epochs
#     to converge. This CHANGES the training regime vs V10.9.0 (opt-in).
FREEZE_BACKBONE = True
# (5) Early stopping: end a fold once val MAE hasn't improved by MIN_DELTA for
#     PATIENCE consecutive epochs. The BEST-MAE weights are restored & saved.
USE_EARLY_STOPPING = True
EARLY_STOP_PATIENCE = 3
EARLY_STOP_MIN_DELTA = 1e-4
# (6) torch.compile the model (PyTorch 2.x). First epoch pays a compile cost,
#     amortised over the run. REQUIRES Triton, which has no official Windows
#     support — so leave this False on Windows. It is verified with a live
#     forward pass at build time and silently falls back to eager if Triton
#     (or anything else) is missing, so turning it True is safe to try.
USE_COMPILE = False

# DataLoader throughput. On Windows, workers>0 requires the __main__ guard
# (present). Set NUM_WORKERS=0 if you hit spawn issues.
# DataLoader throughput. IMPORTANT on Windows: num_workers>0 uses 'spawn',
# which gives EACH worker its own full copy of the (RAM-resident) tensor
# cache — with 4 workers that quadruples memory and exhausts Windows shared
# memory ("Couldn't open shared file mapping, error 1455"). Because
# USE_TENSOR_CACHE already removes the per-epoch CPU cost that workers were
# meant to hide, 0 workers is both safe and nearly as fast here. Only raise
# this on Linux, or set USE_TENSOR_CACHE=False first.
NUM_WORKERS = 0
PIN_MEMORY  = True

# Time (elapsed s) of the frame used to visualise the groups in Figure A.
# 1500 s sits inside the first stable interval (1450–1650 s), the most
# deoxygenated plateau → brightest emission → clearest visual.
GROUP_VISUAL_TIME_S = 1500.0

# Feature weights for the composite QUALITY score used for ranking.
# quality = W_INTENSITY * z(mean_intensity) + W_R2 * z(sv_r2)
W_INTENSITY = 1.0
W_R2        = 1.0

# ------------------------------------------------------------------
# CANDIDATE-POOL FILTER
# Restrict the pixels eligible for grouping to those with a valid Stern-Volmer
# calibration (per-pixel SV fit R² >= SV_R2_POOL_THRESHOLD). This removes
# dead/dark/substrate pixels — the ~64% that fail the SV fit — so the whole
# homogeneity→heterogeneity→homogeneity spectrum is built over GENUINE sensing
# pixels. Without it, the low-quality end of the spectrum is dominated by dark
# pixels whose "response" is meaningless. Reported coverage in the manuscript
# is ~36% of pixels passing at R² >= 0.95, matching the SV sensitivity map.
# Set FILTER_BY_SV_R2 = False to grade over all 2304 pixels (old behaviour).
FILTER_BY_SV_R2   = True
SV_R2_POOL_THRESHOLD = 0.95

# ------------------------------------------------------------------
# GROUP CONSTRUCTION MODE
# 'expanding' (NEW, default): all groups are anchored at the TOP of the
#   quality ranking; each successive group only ADDS diversity by widening
#   the sampling window downward. Window ends are auto-searched so the
#   group MPDs are ~equally spaced from the minimum (contiguous top-N) to
#   the maximum achievable, i.e. G1 = smallest MPD, G{last} = largest MPD,
#   monotonically. This isolates heterogeneity from quality: no group is
#   ever composed of only bad pixels, so MPD and mean quality are no
#   longer confounded (the flaw of 'spectrum' mode past its midpoint).
# 'spectrum' (old behaviour): homogeneous-best -> max-diversity (middle)
#   -> homogeneous-worst. Kept for comparison.
GROUP_MODE = 'expanding'

# ------------------------------------------------------------------
# T90 IN THE HETEROGENEITY FEATURE SPACE
# When True, the per-pixel classical T90 (mean over transitions, averaged
# across days) is loaded from the pixel caches written by the classical
# pixel analysis (response_paper_1_3.py: t90_cache_{exp_id}.npz) and added
# as a 4th dimension to the MPD feature space [intensity, Ksv, R2, T90].
# Rationale: T90 is the DYNAMICS axis — two pixels identical in brightness
# and sensitivity but different in response time are exactly the
# complementary encoders the coded-aperture hypothesis relies on; without
# T90, MPD is blind to that diversity. Pixels lacking a valid T90 are
# z-scored as the pool mean (0) so they neither attract nor repel.
# T90 does NOT enter the quality score Q (ranking is unchanged).
INCLUDE_T90_IN_FEATURES = True
T90_PIXEL_CACHE_DIR = external_path("pixel_t90")

# ------------------------------------------------------------------
# Shared parameters — MUST match V10.9.0 exactly
# ------------------------------------------------------------------
ANALYSIS_INTERVALS_S = [(1450, 1650), (2300, 2500), (3100, 3300),
                        (3900, 4100), (4700, 4900)]
UNSTRUCTURED_CUTOFF_S = 1000

N_STACKED_FRAMES = 3
FRAME_STACK_STEP = 10
RESIZE_DIM  = (224, 224)
STORAGE_DIM = (48, 48)
N_PIXELS_TOTAL = STORAGE_DIM[0] * STORAGE_DIM[1]      # 2304
RED_CH = 2                                             # BGR index of RED
BATCH_SIZE = 32
N_EPOCHS   = 15
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MIN_FRAMES_IN_WINDOW = 5     # frames needed per stable window for SV features

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
FEATURE_CACHE_DIR = cache_path('aqueous_heterogeneity', "features")
MASTER_PARQUET    = os.path.join(FEATURE_CACHE_DIR, "idc_master_v10.parquet")

STUDY_DIR = output_path("heterogeneity_quality_groups")
GROUPDEF_DIR   = os.path.join(STUDY_DIR, "Group_Definitions")
CKPT_DIR       = os.path.join(STUDY_DIR, "checkpoints")
MODELS_DIR     = os.path.join(STUDY_DIR, "models")
SUMMARY_DIR    = os.path.join(STUDY_DIR, "Summary")
PIXEL_FEATURES_PATH  = os.path.join(GROUPDEF_DIR, "pixel_features.npz")
GLOBAL_SUMMARY_PATH  = os.path.join(SUMMARY_DIR, "Global_Group_Performance.csv")

for d in [STUDY_DIR, GROUPDEF_DIR, CKPT_DIR, MODELS_DIR, SUMMARY_DIR,
          FEATURE_CACHE_DIR]:
    os.makedirs(d, exist_ok=True)


def group_dir(g: int) -> str:
    """Per-group output directory (G01 … G13) with V10.9.0 substructure."""
    d = os.path.join(STUDY_DIR, f"G{g:02d}")
    for sub in ["1_Training_Convergence", "2_Inference_Results",
                "3_Chemometrics", "csv_exports"]:
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    return d


# ==============================================================================
# 3. ROBUST DATA DISCOVERY & PARSING (verbatim from V10.9.0)
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


def _decode_pixels_series(s: pd.Series) -> pd.Series:
    """Coerce a pixels column (bytes, list, or ndarray) to uint8 ndarrays."""
    if len(s) == 0:
        return s
    first = s.iloc[0]
    if isinstance(first, bytes):
        return s.apply(lambda x: np.frombuffer(x, dtype=np.uint8))
    if isinstance(first, np.ndarray):
        return s.apply(lambda x: x.astype(np.uint8, copy=False))
    # list / pyarrow list -> ndarray
    return s.apply(lambda x: np.asarray(x, dtype=np.uint8))


def _read_parquet_safe(cache_path: str) -> pd.DataFrame:
    """
    Memory-safe parquet load.

    The V10.9.0 cache stores `pixels` as a nested list column. PyArrow's
    single-shot `read_table` reconstructs ALL of those lists in memory at once,
    which can balloon to many GB and raise ArrowMemoryError (observed: an 8 GB
    realloc failure). We instead stream the file one row group at a time via
    ParquetFile.iter_batches, immediately collapsing each batch's `pixels`
    column to compact uint8 ndarrays, so the inflated list representation never
    exists for the whole file simultaneously. Peak memory is one row group, not
    the entire column.

    Falls back to a plain pd.read_parquet if pyarrow batching is unavailable.
    """
    try:
        import pyarrow.parquet as pq
    except Exception:
        df = pd.read_parquet(cache_path)
        df['pixels'] = _decode_pixels_series(df['pixels'])
        return df

    pf = pq.ParquetFile(cache_path)
    n_rg = pf.num_row_groups
    print(f"[CACHE] Streaming parquet in {n_rg} row group(s) to avoid OOM...")
    parts = []
    # iter_batches yields Arrow RecordBatches; a modest batch size keeps the
    # transient list expansion small regardless of how the file was chunked.
    for batch in tqdm(pf.iter_batches(batch_size=4096), desc="Reading parquet"):
        pdf = batch.to_pandas()
        pdf['pixels'] = _decode_pixels_series(pdf['pixels'])   # collapse now
        parts.append(pdf)
    df = pd.concat(parts, ignore_index=True)
    return df


def get_master_df(experiments, cache_path):
    """Verbatim V10.9.0 pixel extraction, plus a memory-safe cache reader that
    survives the PyArrow list-inflation OOM, and a one-time rewrite of the
    cache into the compact raw-bytes format so later runs load fast and small."""
    if os.path.exists(cache_path):
        df = _read_parquet_safe(cache_path)

        # One-time migration: if the cache still stores pixels as lists, the
        # next load would hit the same OOM risk. Rewrite it once as raw bytes
        # (the compact V10.9.7 format), which PyArrow handles without inflation.
        try:
            import pyarrow.parquet as pq
            schema = pq.ParquetFile(cache_path).schema_arrow
            is_list = 'list' in str(schema.field('pixels').type).lower()
        except Exception:
            is_list = False
        if is_list and len(df) > 0:
            try:
                print("[CACHE] Migrating cache to compact byte format (one-time)...")
                tmp = df.copy()
                tmp['pixels'] = tmp['pixels'].apply(
                    lambda a: np.asarray(a, dtype=np.uint8).tobytes())
                tmp.to_parquet(cache_path, index=False)
                print("[CACHE] Migration complete — future loads will be fast.")
            except Exception as e:
                print(f"[CACHE] Migration skipped ({e}); original cache kept.")
        return df

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
    # Write NEW caches in the compact byte format from the start.
    save_df = master_df.copy()
    save_df['pixels'] = save_df['pixels'].apply(lambda a: a.tobytes())
    save_df.to_parquet(cache_path, index=False)
    return master_df


# ==============================================================================
# 4. PER-PIXEL FEATURES (for ranking + heterogeneity metrics)
# ==============================================================================
def _red_matrix(exp_df: pd.DataFrame) -> tuple:
    """(elapsed, oxygen, red) for one experiment; red shape (N_frames, 2304)."""
    exp_df  = exp_df.sort_values('elapsed').reset_index(drop=True)
    stacked = np.stack(exp_df['pixels'].values)                       # (N, 6912)
    red_idx = np.arange(RED_CH, N_PIXELS_TOTAL * 3, 3)
    return (exp_df['elapsed'].values.astype(np.float32),
            exp_df['oxygen'].values.astype(np.float32),
            stacked[:, red_idx].astype(np.float32))


def _vectorised_sv_fit(interval_I: np.ndarray, interval_O2: np.ndarray) -> tuple:
    """
    Closed-form OLS of the linearised SV model  I0/I = a + Ksv·[O2]  for ALL
    pixels simultaneously (no per-pixel Python loop).

    interval_I  : (k, n_px) median intensity per stable interval per pixel
    interval_O2 : (k,)      median O2 per stable interval
    Returns (ksv, r2) each of shape (n_px,); NaN where the fit is undefined.
    """
    i0_idx = int(np.argmin(interval_O2))
    I0     = np.maximum(interval_I[i0_idx], 1.0)
    Y      = I0[np.newaxis, :] / np.clip(interval_I, 1.0, None)       # (k, n_px)
    x      = interval_O2.astype(np.float64)

    xm  = x.mean()
    ym  = Y.mean(axis=0)
    Sxx = float(np.sum((x - xm) ** 2))
    Sxy = np.sum((x - xm)[:, None] * (Y - ym[None, :]), axis=0)
    slope     = Sxy / max(Sxx, 1e-12)
    intercept = ym - slope * xm
    Y_hat  = intercept[None, :] + slope[None, :] * x[:, None]
    ss_res = np.sum((Y - Y_hat) ** 2, axis=0)
    ss_tot = np.sum((Y - ym[None, :]) ** 2, axis=0)
    r2 = np.where(ss_tot > 1e-10, 1.0 - ss_res / ss_tot, 0.0)
    return slope.astype(np.float32), np.clip(r2, 0.0, 1.0).astype(np.float32)


def compute_pixel_features(df: pd.DataFrame, day_list: list) -> dict:
    """
    For every pixel, averaged across the experiment days in day_list:
      mean_intensity : median RED intensity over stable intervals
      sv_r2          : R² of the linearised SV fit (0 where undefined —
                       so dark/dead pixels remain rankable as "worst")
      sv_ksv         : SV slope (sensitivity)
    Cached to PIXEL_FEATURES_PATH keyed by the day list, so re-runs are free.
    """
    key = "|".join(day_list) + f"|t90={INCLUDE_T90_IN_FEATURES}"
    feat_keys = ['mean_intensity', 'sv_r2', 'sv_ksv', 't90']
    if os.path.exists(PIXEL_FEATURES_PATH):
        cache = np.load(PIXEL_FEATURES_PATH, allow_pickle=True)
        if str(cache.get('key', '')) == key and all(k in cache for k in feat_keys):
            print("[FEATURES] Loaded per-pixel features from cache.")
            return {k: cache[k] for k in feat_keys}

    print(f"[FEATURES] Computing per-pixel features over {len(day_list)} day(s)...")
    per_day_int, per_day_r2, per_day_ksv = [], [], []
    for day in tqdm(day_list, desc="Per-pixel features"):
        elapsed, oxygen, red = _red_matrix(df[df['experiment_id'] == day])
        interval_I, interval_O2 = [], []
        for (s, e) in ANALYSIS_INTERVALS_S:
            m = (elapsed >= s) & (elapsed <= e)
            if m.sum() < MIN_FRAMES_IN_WINDOW: continue
            interval_I.append(np.median(red[m], axis=0))
            interval_O2.append(float(np.median(oxygen[m])))
        if len(interval_I) < 2:
            continue
        interval_I  = np.array(interval_I)
        interval_O2 = np.array(interval_O2)
        per_day_int.append(interval_I.mean(axis=0))
        ksv, r2 = _vectorised_sv_fit(interval_I, interval_O2)
        per_day_ksv.append(ksv)
        per_day_r2.append(r2)

    # --- Per-pixel classical T90, from the caches written by the classical
    # pixel analysis (response_paper_1_3.py): t90_cache_{day}.npz holding
    # 'T90_mean' (48x48, NaN where no valid T90). Averaged across days.
    # The exact folder can vary between machines/runs, so we search
    # T90_PIXEL_CACHE_DIR first and then fall back to a recursive glob under
    # BASE_PROJECT_DIR. Missing files are tolerated (warn + skip); if no day
    # has a cache, t90 is all-NaN and build_groups drops it from the MPD space.
    t90 = np.full(N_PIXELS_TOTAL, np.nan, dtype=np.float32)
    if INCLUDE_T90_IN_FEATURES:
        per_day_t90 = []
        for day in day_list:
            p = os.path.join(T90_PIXEL_CACHE_DIR, f"t90_cache_{day}.npz")
            if not os.path.exists(p):
                hits = glob.glob(os.path.join(BASE_PROJECT_DIR, "**",
                                              f"t90_cache_{day}.npz"),
                                 recursive=True)
                if hits:
                    p = hits[0]
                    print(f"  [T90] found via search: {p}")
                else:
                    print(f"  [T90] cache missing for {day} "
                          f"(looked in {T90_PIXEL_CACHE_DIR} and recursively "
                          f"under BASE_PROJECT_DIR)")
                    continue
            try:
                d = np.load(p, allow_pickle=True)
                per_day_t90.append(np.asarray(d['T90_mean'],
                                              dtype=np.float32).flatten())
            except Exception as e:
                print(f"  [T90] unreadable for {day} ({e}); skipping.")
        if per_day_t90:
            t90 = np.nanmean(np.stack(per_day_t90), axis=0).astype(np.float32)
            n_fin = int(np.sum(np.isfinite(t90)))
            print(f"  [T90] loaded from {len(per_day_t90)}/{len(day_list)} day "
                  f"cache(s); {n_fin}/{N_PIXELS_TOTAL} pixels with finite T90.")
        else:
            print("  [T90] no caches found — MPD will use 3 features only.")

    features = {
        'mean_intensity': np.nanmean(np.stack(per_day_int), axis=0).astype(np.float32),
        'sv_r2':          np.nan_to_num(np.nanmean(np.stack(per_day_r2), axis=0), nan=0.0).astype(np.float32),
        'sv_ksv':         np.nan_to_num(np.nanmean(np.stack(per_day_ksv), axis=0), nan=0.0).astype(np.float32),
        't90':            t90,
    }
    np.savez_compressed(PIXEL_FEATURES_PATH, key=key, **features)
    print(f"[FEATURES] Cached → {PIXEL_FEATURES_PATH}")
    return features


# ==============================================================================
# 5. GROUP CONSTRUCTION & HETEROGENEITY METRICS
# ==============================================================================
def _stratified_sample(start: int, end: int, n: int) -> np.ndarray:
    """
    Return exactly n UNIQUE integer ranks sampled evenly from [start, end).
    linspace rounding can collide when the window is narrow, so collisions
    are repaired by walking to the nearest unused rank inside the window.
    """
    idx = np.round(np.linspace(start, end - 1, n)).astype(int)
    used, out = set(), []
    for v in idx:
        v = int(np.clip(v, start, end - 1))
        step = 0
        while v in used:                       # repair duplicates
            step += 1
            for cand in (v + step, v - step):
                if start <= cand < end and cand not in used:
                    v = cand; break
            else:
                continue
            break
        used.add(v); out.append(v)
    return np.array(sorted(out), dtype=int)


def build_groups(features: dict) -> tuple:
    """
    Construct N_GROUPS groups of N_PIXELS_PER_GROUP pixels along the
    homogeneity → heterogeneity → homogeneity spectrum.

    CANDIDATE POOL: if FILTER_BY_SV_R2 is on, only pixels with a valid
    Stern-Volmer fit (sv_r2 >= SV_R2_POOL_THRESHOLD) are eligible; the
    spectrum is then graded WITHIN those ~36% good pixels, so even the
    lowest-quality group is real sensing pixels, not dead/dark ones.

    RANKING: eligible pixels sorted DESCENDING by composite quality
        quality = W_INTENSITY·z(mean_intensity) + W_R2·z(sv_r2)
    WINDOWS on the ranked pool (M = pool size, N = pixels/group, mid = middle):
        g ≤ mid : window [0, N + (M−N)·(g−1)/(mid−1))          — grows downward
        g ≥ mid : window [(M−N)·(g−mid)/(mid−1), M)            — shrinks from top
    Within each window, N ranks are stratified-sampled evenly, so:
        g=1   → contiguous top-N        (min variety, best pixels)
        g=mid → spans the whole pool    (MAX variety)
        g=N_GROUPS → contiguous bottom-N (min variety, worst *valid* pixels)

    HETEROGENEITY per group (all exported):
        mpd     : mean pairwise Euclidean distance in z-scored
                  [mean_intensity, sv_r2, sv_ksv] space   ← primary
        q_sd    : std of the composite quality score in-group
        q_range : pool-quality percentile span covered by the group

    Returns (groups, metrics_df, quality, rank_order)
        groups : list of N_GROUPS arrays of pixel indices (0..2303)
    """
    def _z(v):
        s = v.std()
        return (v - v.mean()) / (s if s > 1e-12 else 1.0)

    quality = (W_INTENSITY * _z(features['mean_intensity'])
               + W_R2      * _z(features['sv_r2']))

    # ---- CANDIDATE POOL: keep only SV-valid pixels if the filter is on ----
    # eligible = absolute pixel indices allowed into any group. All ranking and
    # window arithmetic below runs over this pool (size M), so the spectrum
    # spans best→worst WITHIN the good pixels, never touching dead/dark ones.
    if FILTER_BY_SV_R2:
        eligible = np.where(features['sv_r2'] >= SV_R2_POOL_THRESHOLD)[0]
        pct = 100.0 * len(eligible) / N_PIXELS_TOTAL
        print(f"[POOL] SV R² ≥ {SV_R2_POOL_THRESHOLD}: {len(eligible)}/"
              f"{N_PIXELS_TOTAL} pixels eligible ({pct:.1f}%). "
              f"Grading spectrum within this pool only.")
        if len(eligible) < N_PIXELS_PER_GROUP:
            raise ValueError(
                f"Only {len(eligible)} pixels pass SV R² ≥ {SV_R2_POOL_THRESHOLD}, "
                f"fewer than N_PIXELS_PER_GROUP={N_PIXELS_PER_GROUP}. "
                f"Lower the threshold, lower N_PIXELS_PER_GROUP, or set "
                f"FILTER_BY_SV_R2=False.")
    else:
        eligible = np.arange(N_PIXELS_TOTAL)
        print(f"[POOL] No SV filter — grading over all {N_PIXELS_TOTAL} pixels.")

    # rank_order: eligible pixel ids sorted best→worst by quality
    rank_order = eligible[np.argsort(-quality[eligible])]
    M   = len(eligible)                       # pool size drives the windows
    N   = N_PIXELS_PER_GROUP
    mid = (N_GROUPS + 1) // 2
    if N > M:
        raise ValueError(f"N_PIXELS_PER_GROUP={N} exceeds pool size {M}")

    # ---- MPD feature space: intensity, R2, Ksv (+ T90 when available) ----
    # T90 can carry NaNs (pixels with no valid transition); a nan-aware
    # z-score is used and NaNs are set to 0 (= pool mean), so such pixels
    # neither inflate nor deflate pairwise distances along the T90 axis.
    def _z_nan(v):
        v = np.asarray(v, dtype=np.float64)
        mval = np.nanmean(v); sval = np.nanstd(v)
        z = (v - mval) / (sval if sval > 1e-12 else 1.0)
        z[~np.isfinite(z)] = 0.0
        return z

    feat_cols = [_z(features['mean_intensity']),
                 _z(features['sv_r2']),
                 _z(features['sv_ksv'])]
    t90_used = False
    if INCLUDE_T90_IN_FEATURES and 't90' in features \
            and np.any(np.isfinite(features['t90'])):
        feat_cols.append(_z_nan(features['t90']))
        t90_used = True
    feat_mat = np.stack(feat_cols, axis=1)    # (2304, 3 or 4), abs-indexed
    print(f"[MPD SPACE] {feat_mat.shape[1]} features "
          f"({'incl.' if t90_used else 'NO'} T90)")

    # global-quality percentile computed WITHIN the pool (0=worst, 100=best pool)
    q_pct_pool = np.empty(N_PIXELS_TOTAL, dtype=np.float32)
    q_pct_pool[:] = np.nan
    order_in_pool = np.argsort(np.argsort(quality[eligible]))
    q_pct_pool[eligible] = 100.0 * order_in_pool / max(M - 1, 1)

    def _mpd_of(pix):
        return float(pdist(feat_mat[pix]).mean()) if len(pix) > 1 else 0.0

    # ---- WINDOW SELECTION -------------------------------------------------
    if GROUP_MODE == 'expanding':
        # All windows anchored at rank 0 (best pixels always included);
        # window ENDS are searched so group MPDs land ~equally spaced from
        # the minimum (contiguous top-N) to the maximum achievable.
        # MPD vs window-end is evaluated on a dense candidate grid and the
        # end nearest each equally-spaced MPD target is chosen; ends are
        # then forced strictly increasing so groups are nested & distinct.
        cand_ends = np.unique(np.round(
            np.linspace(N, M, max(60, N_GROUPS * 8))).astype(int))
        cand_mpds = np.array([_mpd_of(rank_order[_stratified_sample(0, int(e), N)])
                              for e in cand_ends])
        # RECORD-BASED SELECTION: keep only ends where the MPD sets a new
        # running maximum ("record" ends). Records are non-decreasing in MPD
        # by definition, and the stratified sample at a given end is
        # deterministic, so any increasing selection of record ends yields
        # groups whose achieved MPDs are guaranteed monotone — no repair
        # heuristics needed. Targets equally spaced between the smallest and
        # largest record MPD are then matched to the nearest records.
        run_max   = np.maximum.accumulate(cand_mpds)
        is_record = cand_mpds >= run_max - 1e-12
        rec_ends  = cand_ends[is_record]
        rec_mpds  = cand_mpds[is_record]
        if len(rec_ends) < N_GROUPS:
            print(f"    [MODE] only {len(rec_ends)} distinct MPD records for "
                  f"{N_GROUPS} groups — top groups will repeat the widest "
                  f"window. Consider more pixels/group or fewer groups.")
        mpd_lo, mpd_hi = float(rec_mpds[0]), float(rec_mpds[-1])
        targets = np.linspace(mpd_lo, mpd_hi, N_GROUPS)
        sel, prev = [], -1
        for t in targets:
            i = int(np.argmin(np.abs(rec_mpds - t)))
            if i <= prev:                       # keep record indices increasing
                i = min(len(rec_ends) - 1, prev + 1)
            sel.append(i); prev = i
        windows = [(0, int(rec_ends[i])) for i in sel]
        print(f"[MODE] 'expanding': ends={[w[1] for w in windows]} "
              f"(MPD targets {mpd_lo:.3f} → {mpd_hi:.3f})")
    else:                                       # 'spectrum' (legacy)
        windows = []
        for g in range(1, N_GROUPS + 1):
            if g <= mid:
                frac = (g - 1) / max(mid - 1, 1)
                windows.append((0, int(round(N + (M - N) * frac))))
            else:
                frac = (g - mid) / max(mid - 1, 1)
                windows.append((int(round((M - N) * frac)), M))
        print(f"[MODE] 'spectrum' (legacy good→diverse→bad)")

    groups, rows = [], []
    for g, (start, end) in enumerate(windows, start=1):
        ranks  = _stratified_sample(start, end, N)
        pix    = rank_order[ranks]
        groups.append(pix)

        mpd = _mpd_of(pix)
        row = {
            'group': g, 'window_start_rank': start, 'window_end_rank': end,
            'n_pixels': len(pix),
            'mpd': mpd,
            'q_sd': float(quality[pix].std()),
            'q_range': float(q_pct_pool[pix].max() - q_pct_pool[pix].min()),
            'q_mean': float(quality[pix].mean()),
            'mean_intensity_mean': float(features['mean_intensity'][pix].mean()),
            'sv_r2_mean': float(features['sv_r2'][pix].mean()),
        }
        if t90_used:
            t90_pix = features['t90'][pix]
            row['t90_mean'] = (float(np.nanmean(t90_pix))
                               if np.any(np.isfinite(t90_pix)) else np.nan)
        rows.append(row)
    metrics_df = pd.DataFrame(rows)

    # ---- persist definitions for full reproducibility ----
    assign_rows = []
    for g, pix in enumerate(groups, start=1):
        for p in pix:
            r, c = divmod(int(p), STORAGE_DIM[1])
            assign_rows.append({'group': g, 'pixel_idx': int(p),
                                'pixel_row': r, 'pixel_col': c,
                                'quality': float(quality[p]),
                                'mean_intensity': float(features['mean_intensity'][p]),
                                'sv_r2': float(features['sv_r2'][p]),
                                'sv_ksv': float(features['sv_ksv'][p])})
    pd.DataFrame(assign_rows).to_csv(
        os.path.join(GROUPDEF_DIR, "group_assignments.csv"), index=False)
    metrics_df.to_csv(os.path.join(GROUPDEF_DIR, "group_metrics.csv"), index=False)
    np.savez_compressed(os.path.join(GROUPDEF_DIR, "group_masks.npz"),
                        **{f"G{g:02d}": np.isin(np.arange(N_PIXELS_TOTAL), pix).reshape(STORAGE_DIM)
                           for g, pix in enumerate(groups, start=1)})
    print("[GROUPS] Definitions saved:")
    print(metrics_df[['group', 'mpd', 'q_sd', 'q_range', 'q_mean']].to_string(index=False))
    return groups, metrics_df, quality, rank_order


# ==============================================================================
# 6. FIGURE A — GROUP VISUAL PANELS AT THE LOW-O2 PLATEAU
# ==============================================================================
def plot_group_panels(df, groups, metrics_df, day_list):
    """
    One panel per group: the 48×48 frame nearest GROUP_VISUAL_TIME_S from the
    first experiment day (low-O2 plateau, brightest emission), non-group
    pixels dimmed to 20%, group pixels at full colour with a thin outline.
    The primary heterogeneity metric (MPD) is printed under each panel.
    """
    day = day_list[0]
    exp_df = df[df['experiment_id'] == day].sort_values('elapsed')
    idx = (exp_df['elapsed'] - GROUP_VISUAL_TIME_S).abs().idxmin()
    frame = exp_df.loc[idx, 'pixels'].reshape(STORAGE_DIM[0], STORAGE_DIM[1], 3)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t_actual = float(exp_df.loc[idx, 'elapsed'])

    fig, axes = plt.subplots(1, N_GROUPS, figsize=(2.1 * N_GROUPS, 3.3))
    fig.suptitle(f"Pixel groups (N={N_PIXELS_PER_GROUP} each) on low-O₂ frame "
                 f"(t={t_actual:.0f}s, day {day}) — left: homogeneous-best, "
                 f"middle: max variety, right: homogeneous-worst",
                 fontsize=11, y=1.04)
    for g, (ax, pix) in enumerate(zip(np.atleast_1d(axes), groups), start=1):
        mask = np.isin(np.arange(N_PIXELS_TOTAL), pix).reshape(STORAGE_DIM)
        img = frame_rgb * 0.20                              # dim background
        img[mask] = frame_rgb[mask]                         # full-colour group
        ax.imshow(img, origin='upper', aspect='equal')
        yy, xx = np.where(mask)
        ax.scatter(xx, yy, s=6, facecolors='none', edgecolors='cyan',
                   linewidths=0.5)
        m = metrics_df.loc[metrics_df['group'] == g].iloc[0]
        ax.set_title(f"G{g:02d}", fontsize=9)
        ax.set_xlabel(f"MPD={m['mpd']:.2f}\nQ_sd={m['q_sd']:.2f}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    path = os.path.join(GROUPDEF_DIR, "Fig_A_Group_Panels.png")
    plt.savefig(path, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[FIG A] {path}")

    # ------------------------------------------------------------------
    # Fig A2 — compact, tidy true-colour pixel grids (no ROI background).
    # For each group, its N_PIXELS_PER_GROUP pixels are pulled at their ACTUAL
    # RGB colour from the same low-O2 frame, sorted by brightness, and packed
    # into a near-square block. The block has `n_cols` columns; pixels fill it
    # COLUMN-BY-COLUMN with the brightest column on the LEFT and the dimmest on
    # the RIGHT (each column is a brightness tier, filled top→bottom). This is
    # a display re-layout only — the pixels shown are exactly each group's real
    # members; their true 48×48 positions are shown in Fig A.
    # ------------------------------------------------------------------
    # Grid geometry: as square as possible; e.g. 16 → 4×4, 10 → 4×3 (extra
    # cells left blank/white). Columns are the brightness axis.
    n_cols = int(np.ceil(np.sqrt(N_PIXELS_PER_GROUP)))
    n_rows = int(np.ceil(N_PIXELS_PER_GROUP / n_cols))

    fig2, axes2 = plt.subplots(1, N_GROUPS, figsize=(2.1 * N_GROUPS, 3.3))
    fig2.suptitle(f"Pixel groups as tidy true-colour grids (N={N_PIXELS_PER_GROUP} each) "
                  f"— each square = one pixel's actual colour at t={t_actual:.0f}s "
                  f"(day {day}); sorted brightest (left) → dimmest (right)",
                  fontsize=11, y=1.04)

    for g, (ax, pix) in enumerate(zip(np.atleast_1d(axes2), groups), start=1):
        # True RGB colour of each group pixel from the low-O2 frame.
        rows_ = pix // STORAGE_DIM[1]
        cols_ = pix %  STORAGE_DIM[1]
        colours = frame_rgb[rows_, cols_, :]                 # (N, 3) in [0,1]

        # Brightness = perceptual luma; sort DESCENDING (brightest first).
        luma = colours @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        order = np.argsort(-luma)
        colours_sorted = colours[order]

        # Lay out column-by-column: index k → (row = k % n_rows, col = k // n_rows)
        # so the first n_rows brightest pixels fill column 0 (leftmost), etc.
        canvas = np.ones((n_rows, n_cols, 3), dtype=np.float32)   # white padding
        for k in range(len(colours_sorted)):
            r = k %  n_rows
            c = k // n_rows
            canvas[r, c, :] = colours_sorted[k]

        ax.imshow(canvas, origin='upper', aspect='equal', interpolation='nearest')
        # Thin gridlines between cells for a clean tiled look.
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which='minor', color='lightgrey', linewidth=0.6)
        ax.tick_params(which='both', length=0)
        ax.set_xticks([]); ax.set_yticks([])

        m = metrics_df.loc[metrics_df['group'] == g].iloc[0]
        ax.set_title(f"G{g:02d}", fontsize=9)
        ax.set_xlabel(f"MPD={m['mpd']:.2f}\nQ_sd={m['q_sd']:.2f}", fontsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('grey'); spine.set_linewidth(0.6)

    plt.tight_layout()
    path2 = os.path.join(GROUPDEF_DIR, "Fig_A2_Group_Pixel_Grids.png")
    plt.savefig(path2, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[FIG A2] {path2}")


# ==============================================================================
# 7. DATASET & MODEL (V10.9.0 verbatim + spatial input mask)
# ==============================================================================
class IDC_TemporalDataset(Dataset):
    """
    Identical to V10.9.0 except that `mask48` (48×48 bool) zeroes every pixel
    outside the active group BEFORE the resize to 224×224.  The model, frame
    stacking, normalisation and resize are untouched — the group is purely an
    input aperture.

    SPEED (USE_TENSOR_CACHE): the per-frame uint8→float→permute→mask step is
    identical on every epoch, so when enabled we precompute the masked
    3×48×48 float tensor for every frame ONCE at construction (a few hundred
    MB even for tens of thousands of frames) and only the 48→224 resize runs
    per access. Numerically identical to the original; just not recomputed
    each epoch. Set USE_TENSOR_CACHE=False to fall back to lazy per-access.
    """
    def __init__(self, df, exp_id_map, mask48, stack_size=3, step=5,
                 use_cache=True):
        self.df = df.reset_index(drop=True)
        self.exp_id_map = exp_id_map
        self.stack_size = stack_size
        self.step = step
        self.resize = transforms.Resize(RESIZE_DIM)
        self.mask = torch.from_numpy(mask48.astype(np.float32)).unsqueeze(0)  # (1,48,48)
        self.exp_ids = self.df['experiment_id'].values
        self.oxy = self.df['oxygen'].values.astype(np.float32)
        self.use_cache = use_cache
        if use_cache:
            # Precompute masked 3×48×48 float tensors once. This is exactly the
            # tensor that _getitem_ would build per frame, minus the resize.
            m = self.mask                                    # (1,48,48)
            self._cache = []
            for k in range(len(self.df)):
                pix = self.df.iloc[k]['pixels'].reshape(STORAGE_DIM[0], STORAGE_DIM[1], 3)
                t = torch.from_numpy(pix).permute(2, 0, 1).float() / 255.0
                self._cache.append(t * m)                    # apply aperture once
        else:
            self._cache = None

    def __len__(self): return len(self.df)

    def _frame_tensor(self, k):
        """Masked 3×48×48 float tensor for row k (cached or lazy)."""
        if self.use_cache:
            return self._cache[k]
        pix = self.df.iloc[k]['pixels'].reshape(STORAGE_DIM[0], STORAGE_DIM[1], 3)
        t = torch.from_numpy(pix).permute(2, 0, 1).float() / 255.0
        return t * self.mask

    def __getitem__(self, idx):
        exp_id = self.exp_ids[idx]
        frames = []
        for i in range(self.stack_size):
            lookback = max(0, idx - (i * self.step))
            if self.exp_ids[lookback] != exp_id: lookback = idx
            frames.append(self.resize(self._frame_tensor(lookback)))
        return (torch.cat(frames, dim=0),
                torch.tensor(self.oxy[idx], dtype=torch.float32),
                torch.tensor(self.exp_id_map[exp_id], dtype=torch.long))


class TViT_Decoupler(nn.Module):
    """
    V10.9.0 architecture, UNCHANGED in structure and forward pass. The only
    addition is an optional freeze (FREEZE_BACKBONE): when enabled, all ViT
    transformer-block parameters are frozen and only the modified patch-embed
    projection (which must adapt to the 9-channel stacked input) and the
    regression head remain trainable. The computation graph is identical;
    only which parameters receive gradients changes.
    """
    def __init__(self, num_exps, stack_size=3, freeze_backbone=False):
        super().__init__()
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0)
        orig_proj = self.vit.patch_embed.proj
        self.vit.patch_embed.proj = nn.Conv2d(3 * stack_size, orig_proj.out_channels,
                                              kernel_size=orig_proj.kernel_size,
                                              stride=orig_proj.stride,
                                              padding=orig_proj.padding)
        with torch.no_grad():
            self.vit.patch_embed.proj.weight = nn.Parameter(
                orig_proj.weight.repeat(1, stack_size, 1, 1) / stack_size)
            self.vit.patch_embed.proj.bias = orig_proj.bias
        self.head = nn.Sequential(nn.Linear(self.vit.embed_dim, 128),
                                  nn.GELU(), nn.Linear(128, 1))

        if freeze_backbone:
            # Freeze everything in the ViT...
            for p in self.vit.parameters():
                p.requires_grad = False
            # ...except the new input projection, which encodes the stacked
            # frames and MUST train (its weights were only heuristically
            # initialised by repeating the pretrained RGB kernel).
            for p in self.vit.patch_embed.proj.parameters():
                p.requires_grad = True
            # self.head is trainable by default.

    def forward(self, x): return self.head(self.vit.forward_features(x)[:, 0]).squeeze(-1)


# ==============================================================================
# 8. GROUP-LEVEL CLASSICAL BASELINE (V10.9.0 two-site SV, group intensity)
# ==============================================================================
def add_group_intensity_columns(df: pd.DataFrame, groups: list) -> pd.DataFrame:
    """
    Add inv_i_g{g} = 1 / (mean RED intensity over the group's pixels) for
    every frame — the group analogue of V10.9.0's whole-ROI inv_i.  Computed
    in chunks with a single matrix multiply so 13 groups cost one pass.
    """
    print("[CLASSICAL] Computing per-group mean intensities...")
    M = N_PIXELS_TOTAL
    member = np.zeros((M, len(groups)), dtype=np.float32)
    for gi, pix in enumerate(groups):
        member[pix, gi] = 1.0 / len(pix)              # averaging weights

    red_idx = np.arange(RED_CH, M * 3, 3)
    out = np.empty((len(df), len(groups)), dtype=np.float32)
    CH = 2000
    for s in tqdm(range(0, len(df), CH), desc="Group intensities"):
        e = min(s + CH, len(df))
        red = np.stack(df['pixels'].iloc[s:e].values)[:, red_idx].astype(np.float32)
        out[s:e] = red @ member                        # (chunk, n_groups)
    for gi in range(len(groups)):
        df[f'inv_i_g{gi + 1}'] = 1.0 / (out[:, gi] + 1e-6)
    return df


def fit_classical_baseline(train_df, test_df, inv_col: str):
    """
    Verbatim V10.9.0 Two-Site Stern-Volmer fit + analytic quadratic inversion,
    parameterised only by WHICH inverse-intensity column to use (whole-ROI
    'inv_i' in V10.9.0; per-group 'inv_i_g{g}' here).  The physics and the
    numerics are unchanged.
    """
    # DEGENERATE-LOOCV FALLBACK:
    # With a single experiment day, that day is held out and train_df is empty,
    # so there is nothing to calibrate on. This happens only in smoke-tests
    # (N_DAYS_CAP=1). In that case we calibrate the classical baseline on the
    # TEST day's own stable intervals instead — not a valid generalisation
    # benchmark (it "sees" its own calibration), but it lets the pipeline run
    # end to end on one day. A warning is printed so this is never mistaken
    # for a real LOOCV result. With >= 2 days this branch is never taken and
    # the behaviour is identical to V10.9.0.
    if len(train_df) == 0:
        print("  ! [WARN] Single-day run: train set is empty under LOOCV. "
              "Calibrating classical baseline on the TEST day's own stable "
              "intervals (smoke-test only, NOT a generalisation benchmark).")
        calib_df = test_df
    else:
        calib_df = train_df

    stable_train = pd.concat([calib_df[(calib_df['elapsed'] >= s) & (calib_df['elapsed'] <= e)]
                              for s, e in ANALYSIS_INTERVALS_S])
    if stable_train.empty: stable_train = calib_df

    y_train_I  = 1.0 / stable_train[inv_col].values
    x_train_O2 = stable_train['oxygen'].values

    if len(y_train_I) == 0:
        # No usable calibration data at all — return NaNs rather than crashing.
        print("  ! [WARN] No calibration frames available; classical baseline = NaN.")
        return np.full(len(test_df), np.nan, dtype=float)

    def two_site_stern_volmer(o2, I0, f, K1, K2):
        return I0 * ((f / (1.0 + K1 * o2)) + ((1.0 - f) / (1.0 + K2 * o2)))

    p0 = [np.max(y_train_I), 0.8, 0.05, 0.005]
    bounds = ([0, 0, 0, 0], [np.inf, 1.0, np.inf, np.inf])
    try:
        popt, _ = curve_fit(two_site_stern_volmer, x_train_O2, y_train_I,
                            p0=p0, bounds=bounds, maxfev=10000)
        I0_fit, f_fit, K1_fit, K2_fit = popt
    except RuntimeError:
        print("  ! Stern-Volmer Fit Failed. Fallback to Linear.")
        poly = make_pipeline(PolynomialFeatures(1), LinearRegression())
        poly.fit(stable_train[[inv_col]].values, stable_train['oxygen'].values)
        return poly.predict(test_df[[inv_col]].values)

    I_test  = 1.0 / test_df[inv_col].values
    y_ratio = I_test / I0_fit
    A = y_ratio * K1_fit * K2_fit
    B = y_ratio * (K1_fit + K2_fit) - (f_fit * (K2_fit - K1_fit) + K1_fit)
    C = y_ratio - 1.0
    delta = B**2 - 4*A*C
    delta[delta < 0] = 0.0
    o2_pred = (-B - np.sqrt(delta)) / (2 * A + 1e-9)
    mask_neg = o2_pred < 0
    if np.any(mask_neg):
        o2_pred[mask_neg] = (-B[mask_neg] + np.sqrt(delta[mask_neg])) / (2 * A[mask_neg] + 1e-9)
    return o2_pred


# ==============================================================================
# 9. PER-FOLD ANALYTICS (verbatim V10.9.0 plots, routed to group folders)
# ==============================================================================
def plot_chemometrics(df, exp_id, out_dir):
    y_true, y_pred = df['oxygen'], df['o2_pred']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    ax1.scatter(y_true, y_pred, alpha=0.3, s=15, c='blue')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax1.plot(lims, lims, 'k--', lw=2)
    ax1.set_xlabel("Ground Truth Oxygen"); ax1.set_ylabel("Predicted Oxygen")
    ax1.set_title(f"Parity Plot: {exp_id} ($R^2={r2_score(y_true, y_pred):.3f}$)")
    residuals = y_pred - y_true
    sns.histplot(residuals, kde=True, ax=ax2, color='purple')
    ax2.set_xlabel("Error (Pred - True)")
    ax2.set_title(f"Residual Distribution (MAE: {mean_absolute_error(y_true, y_pred):.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "3_Chemometrics", f"Chemometrics_{exp_id}.png"))
    plt.close()


def plot_training_dashboard(history, exp_id, out_dir):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(history['epoch'], history['train_loss'], label='Train Loss (MSE)', color='blue')
    ax2 = ax1.twinx()
    ax2.plot(history['epoch'], history['val_mae'], label='Validation MAE', color='red', marker='o')
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Training MSE"); ax2.set_ylabel("Validation MAE")
    ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    plt.savefig(os.path.join(out_dir, "1_Training_Convergence", f"Convergence_{exp_id}.png"))
    plt.close()
    pd.DataFrame(history).to_csv(
        os.path.join(out_dir, "1_Training_Convergence", f"History_{exp_id}.csv"), index=False)


def analyze_dynamics_v10(df, exp_id, out_dir, phase="Inference"):
    plot_df = df.copy()
    fig, ax = plt.subplots(figsize=(16, 9))
    t = plot_df['elapsed']
    ax.plot(t, plot_df['oxygen'], label='Ground Truth', color='black', lw=3)
    if 'o2_classical' in plot_df.columns:
        ax.plot(t, plot_df['o2_classical'], label='Classical (Stern-Volmer, group)',
                color='gray', ls='--', alpha=0.8)
    if 'o2_pred' in plot_df.columns:
        ax.plot(t, plot_df['o2_pred'], label='AI Prediction (TViT, masked)',
                color='blue', lw=2)
    for (start, end) in ANALYSIS_INTERVALS_S:
        ax.axvspan(start, end, color='green', alpha=0.1)
    ax.set_title(f"Sensor Response: {exp_id} ({phase})")
    ax.set_ylabel("Oxygen Concentration"); ax.set_xlabel("Time (s)")
    ax.legend()
    plt.savefig(os.path.join(out_dir, "2_Inference_Results", f"{phase}_{exp_id}.png"))
    plt.close()
    keep = [c for c in plot_df.columns if c != 'pixels']   # pixels too heavy for CSV
    plot_df[keep].to_csv(os.path.join(out_dir, "csv_exports", f"{phase}_{exp_id}.csv"),
                         index=False)


# ==============================================================================
# 10. TRAINING WITH FULL CHECKPOINT / RESUME
# ==============================================================================
def ckpt_path(g, day):   return os.path.join(CKPT_DIR,   f"ckpt_G{g:02d}_{day}.pth")
def model_path(g, day):  return os.path.join(MODELS_DIR, f"TViT_G{g:02d}_{day}.pth")


def _build_model(n_exps):
    """Construct the TViT (optionally frozen) and, if enabled, torch.compile it.

    torch.compile is LAZY: it returns a wrapped module and only actually
    compiles on the first forward pass — where a missing Triton backend (e.g.
    on Windows, which Triton does not officially support) raises. To keep that
    failure from escaping into the training loop, we trigger compilation here
    with one throwaway forward pass inside the try/except. If anything fails
    (Triton missing, inductor error, OOM during warmup) we discard the compiled
    wrapper and return a plain eager model, so training always proceeds."""
    model = TViT_Decoupler(n_exps, stack_size=N_STACKED_FRAMES,
                           freeze_backbone=FREEZE_BACKBONE).to(DEVICE)
    if USE_COMPILE and hasattr(torch, "compile"):
        try:
            compiled = torch.compile(model)
            # Force actual compilation now with a dummy batch matching the
            # model's expected input: (B, 3*stack, 224, 224).
            dummy = torch.zeros(2, 3 * N_STACKED_FRAMES, RESIZE_DIM[0],
                                RESIZE_DIM[1], device=DEVICE)
            compiled.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE.type == 'cuda')):
                    _ = compiled(dummy)
            model = compiled
            print("    [COMPILE] torch.compile active (warmup passed).")
        except Exception as e:
            # Most common on Windows: TritonMissing. Fall back silently.
            msg = str(e).splitlines()[0] if str(e) else type(e).__name__
            print(f"    [COMPILE] Unavailable — running eager. Reason: {msg}")
            model = TViT_Decoupler(n_exps, stack_size=N_STACKED_FRAMES,
                                   freeze_backbone=FREEZE_BACKBONE).to(DEVICE)
    return model


def _state_dict(model):
    """Unwrap torch.compile's _orig_mod so checkpoints are portable to an
    uncompiled model (e.g. later analysis scripts)."""
    m = getattr(model, "_orig_mod", model)
    return m.state_dict()


def _load_state(model, sd):
    m = getattr(model, "_orig_mod", model)
    m.load_state_dict(sd)


def train_one_fold(g, day, train_df, test_df, exp_map, mask48, n_exps):
    """
    Train the masked-input TViT for one (group, fold), resumable at epoch
    granularity. Every epoch the full state (model, optimiser, AMP scaler,
    early-stop bookkeeping, history, epoch counter) is written to ckpt_path;
    on restart, training continues from the next epoch. The BEST-val-MAE
    weights are saved to model_path and the checkpoint removed.

    Speed features (all toggleable at the top of the file):
      • USE_AMP           mixed-precision train + inference
      • USE_TENSOR_CACHE  masked 48×48 tensors precomputed once per fold
      • FREEZE_BACKBONE   only patch-embed + head receive gradients
      • USE_EARLY_STOPPING stop after EARLY_STOP_PATIENCE stagnant epochs
      • USE_COMPILE       torch.compile the model
      • NUM_WORKERS/PIN_MEMORY  parallel host-side data prep

    Returns (best_val_preds, history).
    """
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    # DEGENERATE-LOOCV FALLBACK (single-day smoke test):
    if len(train_df) == 0:
        train_df = test_df

    dl_kwargs = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                     pin_memory=(PIN_MEMORY and DEVICE.type == 'cuda'))
    if NUM_WORKERS > 0:
        dl_kwargs['persistent_workers'] = True
    train_loader = DataLoader(
        IDC_TemporalDataset(train_df, exp_map, mask48, N_STACKED_FRAMES,
                            FRAME_STACK_STEP, use_cache=USE_TENSOR_CACHE),
        shuffle=True, **dl_kwargs)
    val_loader = DataLoader(
        IDC_TemporalDataset(test_df, exp_map, mask48, N_STACKED_FRAMES,
                            FRAME_STACK_STEP, use_cache=USE_TENSOR_CACHE),
        shuffle=False, **dl_kwargs)

    model     = _build_model(n_exps)
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR)
    criterion = nn.MSELoss()
    scaler    = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE.type == 'cuda'))
    history   = {'epoch': [], 'train_loss': [], 'val_mae': []}
    start_ep  = 0

    # early-stopping bookkeeping (also checkpointed so resume is exact)
    best_mae   = float('inf')
    best_sd    = None
    since_best = 0

    def _infer():
        """AMP-aware validation pass → list of predictions."""
        model.eval(); preds = []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(DEVICE, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE.type == 'cuda')):
                    out = model(x)
                preds.extend(np.atleast_1d(out.float().cpu().numpy()))
        return preds

    cp = ckpt_path(g, day)
    if os.path.exists(cp):                                  # ---- RESUME ----
        try:
            state = torch.load(cp, map_location=DEVICE)
            _load_state(model, state['model'])
            optimizer.load_state_dict(state['optimizer'])
            if 'scaler' in state and state['scaler'] is not None:
                scaler.load_state_dict(state['scaler'])
            history    = state['history']
            start_ep   = state['epoch'] + 1
            best_mae   = state.get('best_mae', float('inf'))
            best_sd    = state.get('best_sd', None)
            since_best = state.get('since_best', 0)
            print(f"    [RESUME] G{g:02d}/{day} from epoch {start_ep} "
                  f"(best MAE so far {best_mae:.4f})")
        except Exception as e:
            print(f"    [RESUME] Checkpoint unreadable ({e}) — restarting fold.")

    stopped_early = False
    for epoch in range(start_ep, N_EPOCHS):
        model.train(); l_sum = 0
        for x, y, _ in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE.type == 'cuda')):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            l_sum += loss.item()

        val_preds = _infer()
        val_mae = mean_absolute_error(test_df['oxygen'], val_preds)
        history['epoch'].append(epoch)
        history['train_loss'].append(l_sum / max(len(train_loader), 1))
        history['val_mae'].append(val_mae)

        # track best weights for early stopping / final save
        if val_mae < best_mae - EARLY_STOP_MIN_DELTA:
            best_mae, best_sd, since_best = val_mae, _state_dict(model), 0
        else:
            since_best += 1

        torch.save({'epoch': epoch, 'model': _state_dict(model),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'history': history, 'best_mae': best_mae,
                    'best_sd': best_sd, 'since_best': since_best}, cp)
        print(f"    G{g:02d}/{day}  ep {epoch+1:02d}/{N_EPOCHS}  "
              f"loss={history['train_loss'][-1]:.4f}  val_MAE={val_mae:.4f}  "
              f"(best {best_mae:.4f}, stale {since_best})")

        if USE_EARLY_STOPPING and since_best >= EARLY_STOP_PATIENCE:
            print(f"    [EARLY STOP] G{g:02d}/{day} — no improvement for "
                  f"{EARLY_STOP_PATIENCE} epochs. Stopping at epoch {epoch+1}.")
            stopped_early = True
            break

    # restore best weights (fallback to current if none recorded)
    if best_sd is not None:
        _load_state(model, best_sd)
    best_preds = _infer()

    torch.save(_state_dict(model), model_path(g, day))
    if os.path.exists(cp): os.remove(cp)
    return best_preds, history


# ==============================================================================
# 11. SUMMARY FIGURES (B–E) + CSVs
# ==============================================================================
def regime_masks(df):
    """stable = inside intervals; transition = outside them after the
    unstructured cutoff; unstructured = free dynamics before the cutoff."""
    stable = np.zeros(len(df), dtype=bool)
    el = df['elapsed'].values
    for s, e in ANALYSIS_INTERVALS_S:
        stable |= (el >= s) & (el <= e)
    unstruct   = el < UNSTRUCTURED_CUTOFF_S
    transition = (~stable) & (~unstruct)
    return stable, transition, unstruct


def _safe_mae(y, p, m):
    return float(mean_absolute_error(y[m], p[m])) if m.sum() > 0 else np.nan


def plot_summary(summary_df, metrics_df):
    """Figures B–E from the per-(group,fold) summary table."""
    agg = (summary_df.groupby('group')
           [['MAE_AI', 'MAE_Classical', 'MAE_AI_stable', 'MAE_AI_transition',
             'MAE_AI_unstructured']].mean().reset_index()
           .merge(metrics_df[['group', 'mpd', 'q_sd', 'q_range']], on='group'))
    agg.to_csv(os.path.join(SUMMARY_DIR, "Fig_B_C_source_data.csv"), index=False)

    def _norm(v):
        v = np.asarray(v, float)
        rng = np.nanmax(v) - np.nanmin(v)
        return (v - np.nanmin(v)) / (rng if rng > 1e-12 else 1.0)

    # ---- Figure B: MAE vs heterogeneity (absolute + min-max normalised) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    for col, colr, lbl in [('MAE_AI', 'royalblue', 'AI (TViT, masked)'),
                           ('MAE_Classical', 'gray', 'Classical (group SV)')]:
        ax1.plot(agg['mpd'], agg[col], 'o-', color=colr, label=lbl)
        ax2.plot(_norm(agg['mpd']), _norm(agg[col]), 'o-', color=colr, label=lbl)
    for _, r in agg.iterrows():
        ax1.annotate(f"G{int(r['group']):02d}", (r['mpd'], r['MAE_AI']),
                     fontsize=7, xytext=(3, 3), textcoords='offset points')
    ax1.set_xlabel("Heterogeneity — Mean Pairwise Distance (MPD)")
    ax1.set_ylabel("MAE (LOOCV mean)")
    ax1.set_title("Absolute"); ax1.legend()
    ax2.set_xlabel("Normalised MPD (0–1)"); ax2.set_ylabel("Normalised MAE (0–1)")
    ax2.set_title("Min-max normalised"); ax2.legend()
    fig.suptitle(f"Reconstruction error vs pixel-group heterogeneity "
                 f"(N={N_PIXELS_PER_GROUP} px/group)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(SUMMARY_DIR, "Fig_B_MAE_vs_Heterogeneity.png"),
                dpi=250, bbox_inches='tight'); plt.close()

    # ---- Figure C: regime-decomposed AI MAE — dynamics-learning evidence ----
    # If heterogeneity helps the model learn DYNAMICS (not just calibration),
    # the transition-regime curve should drop faster with MPD than the
    # stable-regime curve does.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    for col, colr, lbl in [('MAE_AI_stable', 'green', 'Stable plateaux'),
                           ('MAE_AI_transition', 'crimson', 'Transitions'),
                           ('MAE_AI_unstructured', 'darkorange', 'Unstructured (<1000 s)')]:
        ax1.plot(agg['mpd'], agg[col], 'o-', color=colr, label=lbl)
    ax1.set_xlabel("MPD"); ax1.set_ylabel("AI MAE (LOOCV mean)")
    ax1.set_title("AI error by dynamic regime"); ax1.legend()
    ratio = agg['MAE_AI_transition'] / agg['MAE_AI_stable']
    ax2.plot(agg['mpd'], ratio, 's-', color='purple')
    ax2.axhline(1.0, color='k', ls=':', lw=1)
    ax2.set_xlabel("MPD"); ax2.set_ylabel("Transition MAE / Stable MAE")
    ax2.set_title("Dynamics penalty ratio (lower = better at transients)")
    fig.suptitle("Dynamics-learning evidence: where does heterogeneity help?",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(SUMMARY_DIR, "Fig_C_Regime_MAE.png"),
                dpi=250, bbox_inches='tight'); plt.close()

    # ---- Figure D: group × fold AI-MAE heatmap ----
    piv = summary_df.pivot_table(index='group', columns='fold_day',
                                 values='MAE_AI')
    fig, ax = plt.subplots(figsize=(1.1 * max(len(piv.columns), 4) + 3,
                                    0.55 * N_GROUPS + 2))
    sns.heatmap(piv, annot=True, fmt=".3f", cmap='viridis_r',
                cbar_kws={'label': 'AI MAE'}, ax=ax)
    ax.set_title("AI MAE per group × held-out day")
    plt.tight_layout()
    plt.savefig(os.path.join(SUMMARY_DIR, "Fig_D_GroupFold_Heatmap.png"),
                dpi=250, bbox_inches='tight'); plt.close()
    piv.to_csv(os.path.join(SUMMARY_DIR, "Fig_D_source_data.csv"))

    # ---- Figure E: quality-score distribution per group ----
    assign = pd.read_csv(os.path.join(GROUPDEF_DIR, "group_assignments.csv"))
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.violinplot(data=assign, x='group', y='quality', inner='point',
                   density_norm='width', ax=ax, color='steelblue')
    ax.set_xlabel("Group"); ax.set_ylabel("Composite quality score (z)")
    ax.set_title("In-group quality distributions: the homogeneity spectrum")
    plt.tight_layout()
    plt.savefig(os.path.join(SUMMARY_DIR, "Fig_E_Quality_Distributions.png"),
                dpi=250, bbox_inches='tight'); plt.close()
    print(f"[SUMMARY] Figures B–E + source CSVs → {SUMMARY_DIR}")


# ==============================================================================
# 12. EXECUTION ORCHESTRATOR
# ==============================================================================
def main():
    print("=" * 74)
    print("  PIXEL HETEROGENEITY STUDY V11.0.0 — coded chemical aperture test")
    print(f"  N_PIXELS_PER_GROUP={N_PIXELS_PER_GROUP}  N_GROUPS={N_GROUPS}  "
          f"N_DAYS_CAP={N_DAYS_CAP}  DEVICE={DEVICE}")
    print(f"  SPEED: AMP={USE_AMP}  tensor_cache={USE_TENSOR_CACHE}  "
          f"freeze_backbone={FREEZE_BACKBONE}  early_stop={USE_EARLY_STOPPING}"
          f"(patience={EARLY_STOP_PATIENCE})  compile={USE_COMPILE}  "
          f"workers={NUM_WORKERS}")
    print(f"  GROUPS: mode={GROUP_MODE}  T90_in_MPD={INCLUDE_T90_IN_FEATURES}")
    print("=" * 74)

    # ---- data ----
    experiments = find_experiment_files_v10(EXPERIMENTS_ROOT_DIR)
    df = get_master_df(experiments, MASTER_PARQUET)
    day_list = sorted(df['experiment_id'].unique())         # calendar order
    if N_DAYS_CAP: day_list = day_list[:N_DAYS_CAP]
    df = df[df['experiment_id'].isin(day_list)].reset_index(drop=True)
    print(f"[DATA] {len(day_list)} day(s): {day_list}")
    if len(day_list) < 2:
        print("\n" + "!" * 70)
        print("  [WARNING] Only ONE experiment day is available for this run.")
        print("  Leave-one-out CV needs >= 2 days (one to train, one to test).")
        print("  With a single day the TViT trains AND validates on the SAME")
        print("  day and the classical baseline calibrates on its own stable")
        print("  intervals. This is a SMOKE TEST of the pipeline only — the")
        print("  resulting MAEs are NOT valid generalisation numbers.")
        print("  Set N_DAYS_CAP = None (or >= 2) for a real study.")
        print("!" * 70 + "\n")
    n_runs = N_GROUPS * len(day_list)
    print(f"[COST] {N_GROUPS} groups × {len(day_list)} folds = {n_runs} TViT "
          f"trainings of {N_EPOCHS} epochs each. Interrupt any time — "
          f"finished folds are skipped and partial folds resume.")

    # ---- groups ----
    features = compute_pixel_features(df, day_list)
    groups, metrics_df, quality, rank_order = build_groups(features)
    plot_group_panels(df, groups, metrics_df, day_list)

    # ---- per-group classical intensity columns (one pass for all groups) ----
    df = add_group_intensity_columns(df, groups)

    exp_map = {name: i for i, name in enumerate(day_list)}
    masks = [np.isin(np.arange(N_PIXELS_TOTAL), pix).reshape(STORAGE_DIM)
             for pix in groups]

    # ---- resumable summary table ----
    summary_rows = (pd.read_csv(GLOBAL_SUMMARY_PATH).to_dict('records')
                    if os.path.exists(GLOBAL_SUMMARY_PATH) else [])
    done = {(r['group'], r['fold_day']) for r in summary_rows}

    for g, (pix, mask48) in enumerate(zip(groups, masks), start=1):
        out_dir = group_dir(g)
        gm = metrics_df.loc[metrics_df['group'] == g].iloc[0]
        print(f"\n{'='*66}\n[GROUP G{g:02d}] MPD={gm['mpd']:.3f}  "
              f"Q_sd={gm['q_sd']:.3f}  Q_range={gm['q_range']:.1f}%\n{'='*66}")

        for fold_idx, test_day in enumerate(day_list):
            if (g, test_day) in done:
                print(f"  [SKIP] G{g:02d}/{test_day} already complete.")
                continue
            print(f"\n  Fold {fold_idx+1}/{len(day_list)}: hold out {test_day}")
            train_df = df[df['experiment_id'] != test_day]
            test_df  = df[df['experiment_id'] == test_day].copy()

            # -- classical baseline on GROUP intensity (verbatim two-site SV) --
            print("  > Fitting group-level Stern-Volmer Two-Site Model...")
            test_df['o2_classical'] = fit_classical_baseline(
                train_df, test_df, inv_col=f'inv_i_g{g}')

            # -- masked TViT, checkpointed --
            try:
                val_preds, history = train_one_fold(
                    g, test_day, train_df, test_df, exp_map, mask48,
                    n_exps=len(day_list))
            except Exception as e:
                print(f"  [ERROR] Training failed for G{g:02d}/{test_day}: {e}")
                traceback.print_exc(); continue

            test_df['o2_pred'] = val_preds
            plot_training_dashboard(history, test_day, out_dir)
            analyze_dynamics_v10(test_df, test_day, out_dir)
            plot_chemometrics(test_df, test_day, out_dir)

            # -- regime-decomposed metrics --
            y = test_df['oxygen'].values
            pa = test_df['o2_pred'].values
            pc = test_df['o2_classical'].values
            m_st, m_tr, m_un = regime_masks(test_df)
            row = {
                'group': g, 'fold_day': test_day,
                'mpd': float(gm['mpd']), 'q_sd': float(gm['q_sd']),
                'q_range': float(gm['q_range']),
                'MAE_AI':        float(mean_absolute_error(y, pa)),
                'MAE_Classical': float(mean_absolute_error(y, pc)),
                'MAE_AI_stable':        _safe_mae(y, pa, m_st),
                'MAE_AI_transition':    _safe_mae(y, pa, m_tr),
                'MAE_AI_unstructured':  _safe_mae(y, pa, m_un),
                'MAE_Classical_stable':       _safe_mae(y, pc, m_st),
                'MAE_Classical_transition':   _safe_mae(y, pc, m_tr),
                'MAE_Classical_unstructured': _safe_mae(y, pc, m_un),
            }
            summary_rows.append(row)
            pd.DataFrame(summary_rows).to_csv(GLOBAL_SUMMARY_PATH, index=False)
            print(f"  [DONE] G{g:02d}/{test_day}  MAE_AI={row['MAE_AI']:.4f}  "
                  f"MAE_Classical={row['MAE_Classical']:.4f}")

    # ---- summary figures ----
    summary_df = pd.DataFrame(summary_rows)
    if len(summary_df) > 0:
        plot_summary(summary_df, metrics_df)

    print("\n" + "=" * 74)
    print("  HETEROGENEITY STUDY COMPLETE")
    print(f"  Summary:  {GLOBAL_SUMMARY_PATH}")
    print(f"  Figures:  {SUMMARY_DIR}")
    print(f"  Models:   {MODELS_DIR}")
    print("=" * 74)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
