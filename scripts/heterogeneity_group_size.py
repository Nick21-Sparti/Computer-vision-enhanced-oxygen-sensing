# Repository edition: see docs/CHANGES.md and docs/REPRODUCIBILITY.md.
"""
==============================================================================
PROJECT: PINN-IDC (Inverse Dynamic Compensation) Framework
VERSION: 11.4.3-Local — GROUP-SIZE STUDY (WIDEST-MPD GROUPS, LOOCV OVER CHOSEN DAYS)
==============================================================================

--- SCIENTIFIC CONTEXT & GOAL ---
V11.0.0 showed that, at a FIXED group size (16 pixels = 4x4), groups with a
wider spread of pixel performance (higher MPD) give the TViT a lower O2
reconstruction error ("coded chemical aperture"). This iteration asks the
complementary question:

    HOW MANY heterogeneous pixels does the TViT need?

For each group size k x k (k = 2..8, i.e. N = 4, 9, 16, 25, 36, 49, 64
pixels) we build the MOST DIVERSE group the V11.0.0 'expanding' construction
can produce (the window with the WIDEST MPD), train the masked TViT on it and
test on ONE held-out experiment day chosen at the top of the script.

  * MAE falls with N and then plateaus -> a small diverse aperture suffices;
    the plateau size is the practical pixel budget.
  * MAE keeps falling up to 8x8          -> more diverse pixels keep adding
    information.
The classical group-mean two-site SV baseline is reported for every size as
the linear-averaging reference.

EXPERIMENTAL DESIGN
  • Per-pixel features averaged across experiment days (verbatim V11.0.0):
        mean_intensity (median RED over stable plateaus), SV linear-fit R²,
        SV slope Ksv, and T90 (mean over transitions, classical pixel caches).
  • Composite quality (used for RANKING, as V11.0.0):
        Q = W_INTENSITY * z(mean_intensity) + W_R2 * z(sv_r2)
  • Eligible pool: SV R² >= 0.95 (as V11.0.0).
  • For every N = k²: pixels are ranked by Q (best first); windows anchored at
    rank 0 are widened downward on a dense grid; inside each window N pixels
    are stratified-sampled evenly (verbatim V11.0.0 rule); the window giving
    the LARGEST MPD is selected. The best-Q pixel is therefore in every group,
    exactly as in V11.0.0, and each group is the most diverse one of its size.
  • MPD is computed in the V11.0.0 feature space
    [intensity, R², Ksv, (T90)], z-scored over all 2304 pixels.
  • LOOCV over the experiment days listed in DAYS_TO_RUN, identified by the
    number in front of each experiment ID ("2-09_06_2025" -> 2,
    "10-17_06_2025" -> 10, ...): each chosen day is held out in turn and the
    model is trained on the other chosen days. The TViT (masked input),
    loss, optimiser, epochs, early stopping and best-epoch restoration are
    IDENTICAL to V11.0.0-Cluster.

--- EVOLUTION OF THE FRAMEWORK (V1.0 -> V11.4.3-Local) ---
- V1-V7: Static PINNs.
- V8.2: Robust file discovery.
- V10.0-10.6: Dynamic sensing, T90 Benchmarking, LOOCV.
- V10.7: Classical Baseline & Segmented Analytics.
- V10.8: Strict "Black Box" Training.
- V10.8.1: Chemometrics Suite (Parity Plots & Residual Histograms).
- V10.9.0: Two-Site Stern-Volmer classical baseline.
- V11.0.0: Pixel-group coded chemical aperture study (Q ranking, MPD spacing).
- V11.0.0-Cluster: HPC paths, $NSLOTS-aware DataLoader workers.
- V11.4.0-Cluster:
    ADDED VALUE:
    1. GROUP-SIZE SWEEP: one group per size k x k (GROUP_SIDES = 2..8),
       each the WIDEST-MPD group the V11.0.0 expanding construction can make
       at that size (window search exported for audit).
    2. SINGLE HELD-OUT DAY: TEST_DAY is set at the top; training uses all
       other days. Training and testing code are otherwise unchanged
       (masked TViT, early stopping with patience 3, best-epoch restoration).
    3. SIZE-AWARE OUTPUTS: per-size folders (S02_2x2 ... S08_8x8), masks,
       tile grids, and summary figures of MAE vs number of pixels and vs MPD.
    4. ROBUST TWO-SITE SV (bug fix): V10.9 divided the quadratic root by
       (2A + 1e-9); when the fit drove K1 or K2 to its bound of 0 (A = 0)
       every prediction became exactly 0. Replaced by multi-start fitting and
       the exact single-site limit o2 = -C/B when A -> 0.
    5. PORTABILITY FIXES: checkpoints load on torch >= 2.6 (weights_only),
       pandas bfill/ffill instead of the removed fillna(method=), matplotlib
       and seaborn version fallbacks, plotting failures never lose results.
    6. SEPARATE OUTPUT FOLDER (Heterogeneity_Study_GroupSize) - earlier
       studies are never overwritten.
- V11.4.0-Local (CURRENT, this file): identical science and protocol to
  V11.4.0-Cluster, adapted for the local Windows workstation:
    1. LOCAL PATHS: project folder on the F: drive; T90 caches in
       Dynamic_IDC_Analysis_V10_heatmpas_t90_claude/pixel_cache, with a
       recursive search under BASE_PROJECT_DIR if a day's cache is elsewhere
       (as in the local V11.0.0).
    2. WINDOWS-SAFE DATA LOADING: NUM_WORKERS = 0. With 'spawn', every worker
       would get its own copy of the RAM-resident tensor cache and exhaust
       Windows shared memory (error 1455); the cache already removes the
       per-epoch CPU cost workers were meant to hide.
    3. torch.compile stays off (Triton has no official Windows support).
- V11.4.1-Local: LOW-MEMORY DATA PATH for 13-day runs on Windows.
    ADDED VALUE:
    1. SHARED FRAME STORE: all frames are stacked ONCE into a single uint8
       array (N, 48, 48, 3) (~1 GB for 13 days) that every group size reads;
       the per-frame pixel arrays in the DataFrame are then released. Each
       dataset applies its 0/1 mask when a frame is read, so the inputs to
       the TViT are bit-identical to before:
           (uint8 * mask) / 255  ==  (uint8 / 255) * mask.
       Previously every group built its own float32 copy of every frame
       (~3.5 GB), which exhausted the Windows commit limit (RAM + page
       file): first as a CPU "not enough memory" error, then as a CUDA
       "out of memory" error although the GPU itself had ~9.7 GB free.
    2. PIN_MEMORY = False: pinned host memory also counts against the
       Windows commit limit; with NUM_WORKERS = 0 it gives no speed-up.
    3. MEMORY RELEASED between group sizes (gc + torch.cuda.empty_cache).
- V11.4.2-Local: BACK TO LOOCV OVER THE FIRST N DAYS.
    ADDED VALUE:
    1. N_DAYS (manual entry at the top): the first N experiment days are used
       and each is held out in turn (leave-one-day-out), as in V11.0.0.
    2. NUMERIC DAY ORDER: days are ordered by the number in front of the
       experiment ID (2, 3, ..., 15 -> positions 1..13), not alphabetically
       (alphabetical order put "10-17" before "2-09").
    3. ONE OUTPUT FOLDER PER N (Heterogeneity_Study_GroupSize_first{N}days),
       so runs with different N never mix results, checkpoints or groups.
    4. SUMMARY over folds: per size the mean and SD of the MAE across the
       held-out days, plus a size x held-out-day heatmap.
- V11.4.3-Local (CURRENT): DAYS CHOSEN BY ID.
    ADDED VALUE:
    1. DAYS_TO_RUN (manual entry at the top) replaces N_DAYS: list the days
       to use by the number in front of their ID (e.g. [10, 11, 12, 13, 14])
       or by full ID. Unknown numbers stop the run and print the available
       days; the chosen set is printed at the start.
    2. ONE OUTPUT FOLDER PER DAY SET (Heterogeneity_Study_GroupSize_days_<numbers>),
       so different selections never mix results, checkpoints or groups.

--- FOLDER STRUCTURE & OUTPUTS ---
- Heterogeneity_Study_GroupSize/             (inside BASE_PROJECT_DIR)
    - Group_Definitions/
        - pixel_features.npz                 (cached per-pixel features)
        - group_assignments.csv              (pixel -> group size, Q, features)
        - group_metrics.csv                  (MPD, Q stats, T90 stats per size)
        - window_search.csv                  (every candidate window: MPD, chosen)
        - group_masks.npz                    (48x48 boolean mask per size)
        - Fig_A_Group_Panels.png, Fig_A2_Group_Pixel_Grids.png (+ CSV)
    - checkpoints/  ckpt_S{kk}_{day}.pth     (resumable, every epoch)
    - models/       TViT_S{kk}_{day}.pth     (final weights per size)
    - S02_2x2/ ... S08_8x8/                  (per-size V10.9.0 outputs)
        - 1_Training_Convergence/, 2_Inference_Results/, 3_Chemometrics/,
          csv_exports/  Inference_{day}.csv
    - Summary/
        - Global_Group_Performance.csv       (one row per group size)
        - Fig_B_MAE_vs_GroupSize.png         (vs N pixels and vs MPD)
        - Fig_C_Regime_MAE.png               (stable / transition / unstructured)
        - Fig_D_Training_Curves.png          (validation MAE per epoch, all sizes)
        - Fig_E_Quality_Distributions.png    (Q violins per size)
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
import gc
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')          # Headless-safe (no GUI windows during long runs)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

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

try:
    plt.style.use('seaborn-v0_8-whitegrid')       # matplotlib >= 3.6
except OSError:
    plt.style.use('seaborn-whitegrid')            # older matplotlib
plt.rcParams['axes.grid'] = False                 # no grid lines on any plot

# ==============================================================================
# 2. CONFIGURATION & PATHS (LOCAL WINDOWS CONFIGURATION)
# ==============================================================================
BASE_PROJECT_DIR = data_path('aqueous')
EXPERIMENTS_ROOT_DIR = BASE_PROJECT_DIR

# ------------------------------------------------------------------
# ★★★ STUDY-SPECIFIC KNOBS — set these before running ★★★
# ------------------------------------------------------------------
# DAYS TO RUN (manual entry). List the experiment days by the NUMBER IN FRONT
# of their ID (a full ID string such as "10-17_06_2025" also works):
#     number in front:  2     3     4     5     6     7     9     10    11    12    13    14    15
#     ID:               2-09  3-10  4-11  5-12  6-13  7-14  9-16  10-17 11-18 12-19 13-20 14-21 15-22
# Every listed day is held out once (leave-one-day-out CV); the model is
# trained on the other listed days. Cost = 7 group sizes x len(DAYS_TO_RUN)
# trainings. At least 2 days are needed. None = all days.
# [10, 11, 12, 13, 14] are the five days the earlier 5-day studies used.
DAYS_TO_RUN = [11, 12, 13, 14, 15]

# Group sizes: k x k pixels, i.e. N = k^2 pixels per group.
GROUP_SIDES = [2, 3, 4, 5, 6, 7, 8]          # 4, 9, 16, 25, 36, 49, 64 pixels

# Size of the dense window grid searched for the widest-MPD window per size
# (V11.0.0 used max(60, N_GROUPS*8) = 64 candidate ends).
N_WINDOW_CANDIDATES = 64

# ------------------------------------------------------------------
# ★★★ SPEED OPTIMISATIONS / TRAINING PROTOCOL (unchanged from V11.0.0) ★★★
# ------------------------------------------------------------------
# (1) Automatic Mixed Precision: fp16 compute on CUDA. Silently ignored on CPU.
USE_AMP = True
# (2) Frames are held ONCE in a shared uint8 frame store (see section 7 and
#     main); each group size applies its mask on access. (Replaces the
#     per-group float32 tensor cache, which needed ~3.5 GB per group size.)
# (4) Freeze the pretrained ViT backbone; train only the 9-channel patch-embed
#     projection + regression head.
FREEZE_BACKBONE = True
# (5) Early stopping: end a fold once val MAE hasn't improved by MIN_DELTA for
#     PATIENCE consecutive epochs. The BEST-MAE weights are restored & saved.
USE_EARLY_STOPPING = True
EARLY_STOP_PATIENCE = 3
EARLY_STOP_MIN_DELTA = 1e-4
# (6) torch.compile (needs Triton, which has no official Windows support;
#     it falls back to eager automatically, but leave it off locally).
USE_COMPILE = False

# --- LOCAL RESOURCE MANAGEMENT (DataLoader Throughput) ---
# IMPORTANT on Windows: num_workers > 0 uses 'spawn', which starts a new
# Python process per worker and copies the dataset into it - including the
# shared frame store (~1 GB for 13 days) - and can exhaust the Windows commit
# limit. Reading a frame from the store is cheap (index + mask + convert), so
# 0 workers is safe and nearly as fast here. Only raise this on Linux.
NUM_WORKERS = 0
# Pinned host memory counts against the Windows commit limit (RAM + page file)
# and brings no speed-up with NUM_WORKERS = 0, so it is switched off locally.
PIN_MEMORY = False


# Time (elapsed s) of the frame used to visualise the groups in Figure A
# (inside the first, most deoxygenated plateau -> brightest emission).
GROUP_VISUAL_TIME_S = 1500.0

# Feature weights for the composite QUALITY score used for ranking.
# quality = W_INTENSITY * z(mean_intensity) + W_R2 * z(sv_r2)
W_INTENSITY = 1.0
W_R2        = 1.0

# ------------------------------------------------------------------
# CANDIDATE-POOL FILTER (same as V11.0.0)
# Only pixels with a valid Stern-Volmer calibration (per-pixel SV fit
# R² >= SV_R2_POOL_THRESHOLD) can enter a group, so every group consists of
# genuine sensing pixels (~36 % of the ROI at R² >= 0.95).
# ------------------------------------------------------------------
FILTER_BY_SV_R2      = True
SV_R2_POOL_THRESHOLD = 0.95

# ------------------------------------------------------------------
# T90 IN THE MPD FEATURE SPACE (same as V11.0.0)
# Per-pixel classical T90 (mean over transitions, averaged across days) from
# the caches written by response_paper_1_3.py (t90_cache_{day}.npz). Pixels
# without a valid T90 are z-scored as the pool mean (0). T90 does NOT enter Q.
# ------------------------------------------------------------------
INCLUDE_T90_IN_FEATURES = True
# Local location of the caches; if a day's cache is not found here the whole
# project folder is searched recursively (see compute_pixel_features).
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
N_EPOCHS   = 20
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MIN_FRAMES_IN_WINDOW = 5     # frames needed per stable window for SV features

# ------------------------------------------------------------------
# Paths (NEW study folder; earlier studies are never touched)
# ------------------------------------------------------------------
FEATURE_CACHE_DIR = cache_path('aqueous_heterogeneity', "features")
MASTER_PARQUET    = os.path.join(FEATURE_CACHE_DIR, "idc_master_v10.parquet")

# One folder per day selection, so different selections never mix.
def _days_tag(days):
    if days is None:
        return "alldays"
    parts = []
    for d in days:
        m = re.match(r"\s*(\d+)", str(d))
        parts.append(m.group(1) if m else re.sub(r"[^A-Za-z0-9]+", "", str(d)))
    return "days_" + "-".join(sorted(set(parts), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0, x)))


STUDY_DIR = output_path("heterogeneity_group_size", _days_tag(DAYS_TO_RUN))
GROUPDEF_DIR   = os.path.join(STUDY_DIR, "Group_Definitions")
CKPT_DIR       = os.path.join(STUDY_DIR, "checkpoints")
MODELS_DIR     = os.path.join(STUDY_DIR, "models")
SUMMARY_DIR    = os.path.join(STUDY_DIR, "Summary")
PIXEL_FEATURES_PATH  = os.path.join(GROUPDEF_DIR, "pixel_features.npz")
GLOBAL_SUMMARY_PATH  = os.path.join(SUMMARY_DIR, "Global_Group_Performance.csv")



def make_output_dirs():
    """Create the study folders - called only after DAYS_TO_RUN has been
    validated, so a typo in the day list never leaves an empty folder."""
    for d in [STUDY_DIR, GROUPDEF_DIR, CKPT_DIR, MODELS_DIR, SUMMARY_DIR, FEATURE_CACHE_DIR]:
        os.makedirs(d, exist_ok=True)


def day_order_key(exp_id: str):
    """
    Sort key: the number in front of the experiment ID ("2-09_06_2025" -> 2,
    "10-17_06_2025" -> 10). IDs without a leading number go last, alphabetically.
    """
    m = re.match(r"\s*(\d+)", str(exp_id))
    return (0, int(m.group(1)), str(exp_id)) if m else (1, 0, str(exp_id))


def select_days(all_days: list) -> list:
    """
    Resolve DAYS_TO_RUN against the experiment IDs found in the data.
    Entries may be the number in front of an ID (int or digit string) or a
    full ID. Returns the chosen IDs in the order of their leading number.
    Stops with the table of available days if an entry matches nothing.
    """
    if DAYS_TO_RUN is None:
        return list(all_days)
    by_num = {}
    for d in all_days:
        key = day_order_key(d)
        if key[0] == 0:
            by_num.setdefault(key[1], []).append(d)
    chosen, missing = [], []
    for x in DAYS_TO_RUN:
        if isinstance(x, str) and x in all_days:
            chosen.append(x)
        elif str(x).strip().isdigit() and len(by_num.get(int(str(x).strip()), [])) == 1:
            chosen.append(by_num[int(str(x).strip())][0])
        else:
            missing.append(x)
    if missing:
        print(f"CRITICAL ERROR: these DAYS_TO_RUN entries match no (unique) experiment day: {missing}")
        print("Available days (number in front: ID):")
        for d in all_days:
            k = day_order_key(d)
            print(f"   {k[1] if k[0] == 0 else '-':>3}: {d}")
        sys.exit(1)
    return sorted(dict.fromkeys(chosen), key=day_order_key)


def size_label(k: int) -> str:
    """Folder / file label of a k x k group, e.g. 'S04_4x4'."""
    return f"S{k:02d}_{k}x{k}"


def group_dir(k: int) -> str:
    """Per-size output directory with the V10.9.0 substructure."""
    d = os.path.join(STUDY_DIR, size_label(k))
    for sub in ["1_Training_Convergence", "2_Inference_Results",
                "3_Chemometrics", "csv_exports"]:
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    return d


# ==============================================================================
# 3. ROBUST DATA DISCOVERY & PARSING (verbatim V11.0.0)
# ==============================================================================
def find_experiment_files_v10(root_dir):
    print("--- Discovery: Searching for Experiment Directories ---")
    if not os.path.exists(root_dir):
        print(f"CRITICAL ERROR: Root directory not found: {root_dir}")
        print("Please verify the local path or that the external drive is connected.")
        sys.exit(1)

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
    """Reference-sensor O2 (MEA lines of the Arduino log) + temperature log."""
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
    # bfill/ffill instead of fillna(method=...), which newer pandas removes
    df['oxygen'] = df['oxygen'].interpolate().bfill().ffill()
    df['temp'] = df['temp'].interpolate().bfill().ffill()
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
    return s.apply(lambda x: np.asarray(x, dtype=np.uint8))


def _read_parquet_safe(cache_path: str) -> pd.DataFrame:
    """
    Memory-safe parquet load: stream one row group at a time and collapse the
    `pixels` column to compact uint8 arrays immediately, so the inflated Arrow
    list representation never exists for the whole file at once.
    """
    try:
        import pyarrow.parquet as pq
    except Exception:
        df = pd.read_parquet(cache_path)
        df['pixels'] = _decode_pixels_series(df['pixels'])
        return df

    pf = pq.ParquetFile(cache_path)
    print(f"[CACHE] Streaming parquet in {pf.num_row_groups} row group(s) to avoid OOM...")
    parts = []
    for batch in tqdm(pf.iter_batches(batch_size=4096), desc="Reading parquet"):
        pdf = batch.to_pandas()
        pdf['pixels'] = _decode_pixels_series(pdf['pixels'])
        parts.append(pdf)
    return pd.concat(parts, ignore_index=True)


def get_master_df(experiments, cache_path):
    """Verbatim V10.9.0 pixel extraction with the memory-safe cache reader and
    a one-time migration of list-encoded caches to compact raw bytes."""
    if os.path.exists(cache_path):
        df = _read_parquet_safe(cache_path)
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
    save_df = master_df.copy()
    save_df['pixels'] = save_df['pixels'].apply(lambda a: a.tobytes())
    save_df.to_parquet(cache_path, index=False)
    return master_df


# ==============================================================================
# 4. PER-PIXEL FEATURES (verbatim V11.0.0)
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
    Closed-form OLS of the linearised SV model I0/I = a + Ksv·[O2] for ALL
    pixels simultaneously (plateau medians as data points).
    Returns (Ksv, R²) per pixel.
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
    For every pixel, averaged across the experiment days in day_list
    (identical to V11.0.0):
      mean_intensity : median RED intensity over stable intervals
      sv_r2          : R² of the linearised SV fit
      sv_ksv         : SV slope (sensitivity)
      t90            : classical response time (mean over transitions)
    Cached to PIXEL_FEATURES_PATH keyed by the day list.
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

    t90 = np.full(N_PIXELS_TOTAL, np.nan, dtype=np.float32)
    if INCLUDE_T90_IN_FEATURES:
        per_day_t90 = []
        for day in day_list:
            p = os.path.join(T90_PIXEL_CACHE_DIR, f"t90_cache_{day}.npz")
            if not os.path.exists(p):
                # The folder can differ between machines/runs: search the whole
                # project folder before giving up (local V11.0.0 behaviour).
                hits = glob.glob(os.path.join(BASE_PROJECT_DIR, "**",
                                              f"t90_cache_{day}.npz"), recursive=True)
                if hits:
                    p = hits[0]
                    print(f"  [T90] found via search: {p}")
                else:
                    print(f"  [T90] cache missing for {day} (looked in {T90_PIXEL_CACHE_DIR} "
                          f"and recursively under BASE_PROJECT_DIR)")
                    continue
            try:
                d = np.load(p, allow_pickle=True)
                per_day_t90.append(np.asarray(d['T90_mean'], dtype=np.float32).flatten())
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
# 5. GROUP CONSTRUCTION: WIDEST-MPD GROUP FOR EVERY SIZE
# ==============================================================================
def _stratified_sample(start: int, end: int, n: int) -> np.ndarray:
    """Return exactly n UNIQUE integer ranks sampled evenly from [start, end)
    (verbatim V11.0.0; collisions from linspace rounding are repaired)."""
    idx = np.round(np.linspace(start, end - 1, n)).astype(int)
    used, out = set(), []
    for v in idx:
        v = int(np.clip(v, start, end - 1))
        step = 0
        while v in used:
            step += 1
            for cand in (v + step, v - step):
                if start <= cand < end and cand not in used:
                    v = cand; break
            else:
                continue
            break
        used.add(v); out.append(v)
    return np.array(sorted(out), dtype=int)


def build_widest_groups(features: dict) -> tuple:
    """
    One group per size k x k (N = k² pixels), each the MOST DIVERSE group the
    V11.0.0 'expanding' construction can produce at that size.

    POOL     : SV R² >= SV_R2_POOL_THRESHOLD (as V11.0.0).
    RANKING  : descending composite quality Q (rank 0 = best pixel).
    WINDOWS  : anchored at rank 0, end e on a dense grid N..M; inside each
               window N pixels are stratified-sampled evenly (V11.0.0 rule),
               so the best-Q pixel is in every group.
    SELECTION: the window whose sampled group has the LARGEST MPD.
    MPD      : mean pairwise Euclidean distance in the z-scored feature
               space [intensity, R², Ksv, (T90)] - identical to V11.0.0.
    Every candidate window (size, end, MPD, chosen) is written to
    window_search.csv so the choice can be audited.
    Returns (groups, metrics_df, quality, rank_order) where groups is a list
    of dicts {side, n, label, pix}.
    """
    def _z(v):
        s = v.std()
        return (v - v.mean()) / (s if s > 1e-12 else 1.0)

    def _z_nan(v):
        v = np.asarray(v, dtype=np.float64)
        mval = np.nanmean(v); sval = np.nanstd(v)
        z = (v - mval) / (sval if sval > 1e-12 else 1.0)
        z[~np.isfinite(z)] = 0.0
        return z

    quality = (W_INTENSITY * _z(features['mean_intensity'])
               + W_R2      * _z(features['sv_r2']))

    if FILTER_BY_SV_R2:
        eligible = np.where(features['sv_r2'] >= SV_R2_POOL_THRESHOLD)[0]
        print(f"[POOL] SV R² ≥ {SV_R2_POOL_THRESHOLD}: {len(eligible)}/{N_PIXELS_TOTAL} "
              f"pixels eligible ({100.0 * len(eligible) / N_PIXELS_TOTAL:.1f}%).")
    else:
        eligible = np.arange(N_PIXELS_TOTAL)
        print(f"[POOL] No SV filter — all {N_PIXELS_TOTAL} pixels eligible.")
    M = len(eligible)
    largest = max(GROUP_SIDES) ** 2
    if M < largest:
        raise ValueError(f"Only {M} eligible pixels, fewer than the largest group "
                         f"({largest} pixels). Reduce GROUP_SIDES or relax the SV filter.")

    rank_order = eligible[np.argsort(-quality[eligible])]          # best Q first

    feat_cols = [_z(features['mean_intensity']), _z(features['sv_r2']), _z(features['sv_ksv'])]
    t90_used = (INCLUDE_T90_IN_FEATURES and 't90' in features
                and np.any(np.isfinite(features['t90'])))
    if t90_used:
        feat_cols.append(_z_nan(features['t90']))
    feat_mat = np.stack(feat_cols, axis=1)
    print(f"[MPD SPACE] {feat_mat.shape[1]} features ({'incl.' if t90_used else 'NO'} T90)")

    q_pct_pool = np.full(N_PIXELS_TOTAL, np.nan, dtype=np.float32)
    q_pct_pool[eligible] = 100.0 * np.argsort(np.argsort(quality[eligible])) / max(M - 1, 1)

    def _mpd_of(pix):
        return float(pdist(feat_mat[pix]).mean()) if len(pix) > 1 else 0.0

    groups, rows, search_rows = [], [], []
    for k in GROUP_SIDES:
        N = k * k
        cand_ends = np.unique(np.round(np.linspace(N, M, N_WINDOW_CANDIDATES)).astype(int))
        cand_mpds = np.array([_mpd_of(rank_order[_stratified_sample(0, int(e), N)])
                              for e in cand_ends])
        best = int(np.argmax(cand_mpds))                             # widest MPD
        end = int(cand_ends[best])
        for e, m in zip(cand_ends, cand_mpds):
            search_rows.append({'side': k, 'n_pixels': N, 'window_end_rank': int(e),
                                'mpd': float(m), 'chosen': bool(e == end)})
        pix = rank_order[_stratified_sample(0, end, N)]
        groups.append({'side': k, 'n': N, 'label': size_label(k), 'pix': pix})

        row = {
            'group': size_label(k), 'side': k, 'n_pixels': N,
            'window_start_rank': 0, 'window_end_rank': end,
            'window_fraction_of_pool': end / M,
            'mpd': _mpd_of(pix),
            'q_mean': float(quality[pix].mean()), 'q_sd': float(quality[pix].std()),
            'q_range': float(np.nanmax(q_pct_pool[pix]) - np.nanmin(q_pct_pool[pix])),
            'mean_intensity_mean': float(features['mean_intensity'][pix].mean()),
            'sv_r2_mean': float(features['sv_r2'][pix].mean()),
        }
        if t90_used:
            tv = features['t90'][pix]
            fin = tv[np.isfinite(tv)]
            row.update({'t90_mean': float(np.mean(fin)) if len(fin) else np.nan,
                        't90_sd': float(np.std(fin)) if len(fin) else np.nan,
                        't90_n_finite': int(len(fin))})
        rows.append(row)
        print(f"  [{size_label(k)}] N={N:2d}  window 0..{end} of {M} ranks  "
              f"MPD={row['mpd']:.3f}  Q_mean={row['q_mean']:.2f}")
    metrics_df = pd.DataFrame(rows)

    assign_rows = []
    for gr in groups:
        for p in gr['pix']:
            r, c = divmod(int(p), STORAGE_DIM[1])
            assign_rows.append({'group': gr['label'], 'side': gr['side'], 'n_pixels': gr['n'],
                                'pixel_idx': int(p), 'pixel_row': r, 'pixel_col': c,
                                'q_rank': int(np.where(rank_order == p)[0][0]),
                                'quality': float(quality[p]),
                                'mean_intensity': float(features['mean_intensity'][p]),
                                'sv_r2': float(features['sv_r2'][p]),
                                'sv_ksv': float(features['sv_ksv'][p]),
                                't90': float(features['t90'][p])})
    pd.DataFrame(assign_rows).to_csv(os.path.join(GROUPDEF_DIR, "group_assignments.csv"), index=False)
    metrics_df.to_csv(os.path.join(GROUPDEF_DIR, "group_metrics.csv"), index=False)
    pd.DataFrame(search_rows).to_csv(os.path.join(GROUPDEF_DIR, "window_search.csv"), index=False)
    np.savez_compressed(os.path.join(GROUPDEF_DIR, "group_masks.npz"),
                        **{gr['label']: np.isin(np.arange(N_PIXELS_TOTAL), gr['pix']).reshape(STORAGE_DIM)
                           for gr in groups})
    print("[GROUPS] Definitions saved.")
    return groups, metrics_df, quality, rank_order


# ==============================================================================
# 6. FIGURE A — GROUP VISUAL PANELS AT THE LOW-O2 PLATEAU
# ==============================================================================
def plot_group_panels(df, groups, metrics_df, day):
    """Fig A: each group's pixels highlighted on the 48x48 frame.
    Fig A2: each group's pixels as a tidy true-colour k x k tile (brightest
    column first). Both on the frame nearest GROUP_VISUAL_TIME_S of `day`."""
    exp_df = df[df['experiment_id'] == day].sort_values('elapsed')
    idx = (exp_df['elapsed'] - GROUP_VISUAL_TIME_S).abs().idxmin()
    frame = exp_df.loc[idx, 'pixels'].reshape(STORAGE_DIM[0], STORAGE_DIM[1], 3)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t_actual = float(exp_df.loc[idx, 'elapsed'])
    n_g = len(groups)

    fig, axes = plt.subplots(1, n_g, figsize=(2.3 * n_g, 3.4))
    fig.suptitle(f"Widest-MPD pixel group for each size on low-O₂ frame "
                 f"(t={t_actual:.0f}s, day {day})", fontsize=11, y=1.04)
    for ax, gr in zip(np.atleast_1d(axes), groups):
        mask = np.isin(np.arange(N_PIXELS_TOTAL), gr['pix']).reshape(STORAGE_DIM)
        img = frame_rgb * 0.20                              # dim background
        img[mask] = frame_rgb[mask]                         # full-colour group
        ax.imshow(img, origin='upper', aspect='equal')
        yy, xx = np.where(mask)
        ax.scatter(xx, yy, s=6, facecolors='none', edgecolors='cyan', linewidths=0.5)
        m = metrics_df.loc[metrics_df['group'] == gr['label']].iloc[0]
        ax.set_title(f"{gr['side']}x{gr['side']} (N={gr['n']})", fontsize=9)
        ax.set_xlabel(f"MPD={m['mpd']:.2f}\nQ_sd={m['q_sd']:.2f}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    path = os.path.join(GROUPDEF_DIR, "Fig_A_Group_Panels.png")
    plt.savefig(path, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[FIG A] {path}")

    fig2, axes2 = plt.subplots(1, n_g, figsize=(2.3 * n_g, 3.4))
    fig2.suptitle(f"Widest-MPD groups as k x k true-colour tiles at t={t_actual:.0f}s "
                  f"(day {day}); brightest column left", fontsize=11, y=1.04)
    tile_rows = []
    for ax, gr in zip(np.atleast_1d(axes2), groups):
        pix, k = gr['pix'], gr['side']
        colours = frame_rgb[pix // STORAGE_DIM[1], pix % STORAGE_DIM[1], :]
        luma = colours @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        order = np.argsort(-luma)                           # brightest first
        canvas = np.ones((k, k, 3), dtype=np.float32)
        for j, o in enumerate(order):                       # column by column
            canvas[j % k, j // k, :] = colours[o]
            tile_rows.append({'group': gr['label'], 'pixel_idx': int(pix[o]),
                              'tile_row': j % k, 'tile_col': j // k,
                              'R': float(colours[o, 0]), 'G': float(colours[o, 1]),
                              'B': float(colours[o, 2])})
        ax.imshow(canvas, origin='upper', aspect='equal', interpolation='nearest')
        ax.set_xticks(np.arange(-0.5, k, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, k, 1), minor=True)
        ax.grid(which='minor', color='lightgrey', linewidth=0.6)
        ax.tick_params(which='both', length=0)
        ax.set_xticks([]); ax.set_yticks([])
        m = metrics_df.loc[metrics_df['group'] == gr['label']].iloc[0]
        ax.set_title(f"{k}x{k}", fontsize=9)
        ax.set_xlabel(f"MPD={m['mpd']:.2f}", fontsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('grey'); spine.set_linewidth(0.6)
    plt.tight_layout()
    path2 = os.path.join(GROUPDEF_DIR, "Fig_A2_Group_Pixel_Grids.png")
    plt.savefig(path2, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    pd.DataFrame(tile_rows).to_csv(os.path.join(GROUPDEF_DIR, "Fig_A2_source_data.csv"), index=False)
    print(f"[FIG A2] {path2}")


# ==============================================================================
# 7. DATASET & MODEL (V10.9.0 verbatim + spatial input mask)
# ==============================================================================
class IDC_TemporalDataset(Dataset):
    """
    Stacks N_STACKED_FRAMES frames (current + look-backs FRAME_STACK_STEP
    apart, never crossing experiment boundaries) with every pixel outside the
    group (mask48) set to zero - the group IS the input aperture.

    LOW-MEMORY DATA PATH (V11.4.1-Local): frames are NOT copied per group.
    `frame_store` is ONE shared uint8 tensor (N_all, 48, 48, 3) built once in
    main(); each row of `df` points into it through its 'frame_idx' column.
    The 0/1 mask is applied in uint8 when a frame is read and the result is
    converted to float in [0, 1] - bit-identical to the V10.9.0 tensors
    (uint8 / 255) * mask, at a fraction of the memory.
    """
    def __init__(self, df, exp_id_map, mask48, frame_store, stack_size=3, step=5):
        self.df = df.reset_index(drop=True)
        self.exp_id_map = exp_id_map
        self.stack_size = stack_size
        self.step = step
        self.resize = transforms.Resize(RESIZE_DIM)
        self.frames = frame_store                                    # shared, not copied
        self.frame_idx = self.df['frame_idx'].to_numpy()
        self.mask_u8 = torch.from_numpy(mask48.astype(np.uint8)[:, :, None])   # (48, 48, 1)
        self.exp_ids = self.df['experiment_id'].values
        self.oxy = self.df['oxygen'].values.astype(np.float32)

    def __len__(self): return len(self.df)

    def _frame_tensor(self, k):
        """Masked 3x48x48 float tensor in [0, 1] for row k."""
        f = self.frames[self.frame_idx[k]] * self.mask_u8           # uint8, exact zeroing
        return f.permute(2, 0, 1).float() / 255.0

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
    """ViT-Small (ImageNet-pretrained) with a 9-channel patch embedding for 3
    stacked RGB frames and a 2-layer regression head (verbatim V10.9.0)."""
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
            for p in self.vit.parameters():
                p.requires_grad = False
            for p in self.vit.patch_embed.proj.parameters():
                p.requires_grad = True

    def forward(self, x): return self.head(self.vit.forward_features(x)[:, 0]).squeeze(-1)


# ==============================================================================
# 8. GROUP-LEVEL CLASSICAL BASELINE (two-site SV on group intensity, FIXED)
# ==============================================================================
def add_group_intensity_columns(df: pd.DataFrame, groups: list) -> pd.DataFrame:
    """inv_i_<label> = 1 / (mean RED of the group's pixels) for every frame,
    computed in chunks with one matrix product for all groups."""
    print("[CLASSICAL] Computing per-group mean intensities...")
    M = N_PIXELS_TOTAL
    member = np.zeros((M, len(groups)), dtype=np.float32)
    for gi, gr in enumerate(groups):
        member[gr['pix'], gi] = 1.0 / len(gr['pix'])
    red_idx = np.arange(RED_CH, M * 3, 3)
    out = np.empty((len(df), len(groups)), dtype=np.float32)
    CH = 2000
    for s in tqdm(range(0, len(df), CH), desc="Group intensities"):
        e = min(s + CH, len(df))
        red = np.stack(df['pixels'].iloc[s:e].values)[:, red_idx].astype(np.float32)
        out[s:e] = red @ member
    for gi, gr in enumerate(groups):
        df[f"inv_i_{gr['label']}"] = 1.0 / (out[:, gi] + 1e-6)
    return df


def fit_classical_baseline(train_df, test_df, inv_col: str):
    """
    Two-site Stern-Volmer on the group-mean intensity, calibrated on the stable
    plateaus of the TRAINING days only.
      * five starting points, best SSE kept;
      * analytic inversion of y = f/(1+K1 o) + (1-f)/(1+K2 o), y = I/I0:
            A o² + B o + C = 0,  A = y K1 K2,
            B = y (K1+K2) - (f (K2-K1) + K1),  C = y - 1;
      * when A -> 0 (K1 or K2 at its bound: single-site behaviour) the root
        is the linear limit o = -C/B. The V10.9 code divided by (2A + 1e-9)
        instead, which returned o = 0 for every frame in exactly that case.
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
        print("  ! [WARN] No calibration frames available; classical baseline = NaN.")
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
    if best is None:
        print("  ! Stern-Volmer fit failed for every start. Fallback to linear in 1/I.")
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
    lin = -C / np.where(np.abs(B) < 1e-12, np.nan, B)          # A -> 0: single-site limit
    return np.where(np.isfinite(pred), pred, lin).astype(float)


# ==============================================================================
# 9. PER-FOLD ANALYTICS (verbatim V11.0.0)
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
        ax.plot(t, plot_df['o2_pred'], label='AI Prediction (TViT, masked)', color='blue', lw=2)
    for (start, end) in ANALYSIS_INTERVALS_S:
        ax.axvspan(start, end, color='green', alpha=0.1)
    ax.set_title(f"Sensor Response: {exp_id} ({phase})")
    ax.set_ylabel("Oxygen Concentration"); ax.set_xlabel("Time (s)")
    ax.legend()
    plt.savefig(os.path.join(out_dir, "2_Inference_Results", f"{phase}_{exp_id}.png"))
    plt.close()
    keep = [c for c in plot_df.columns if not c.startswith('pixels') and not c.startswith('inv_i_')]
    plot_df[keep].to_csv(os.path.join(out_dir, "csv_exports", f"{phase}_{exp_id}.csv"), index=False)


# ==============================================================================
# 10. TRAINING WITH FULL CHECKPOINT / RESUME (protocol unchanged from V11.0.0)
# ==============================================================================
def ckpt_path(k, day):   return os.path.join(CKPT_DIR,   f"ckpt_S{k:02d}_{day}.pth")
def model_path(k, day):  return os.path.join(MODELS_DIR, f"TViT_S{k:02d}_{day}.pth")


def _build_model(n_exps):
    model = TViT_Decoupler(n_exps, stack_size=N_STACKED_FRAMES,
                           freeze_backbone=FREEZE_BACKBONE).to(DEVICE)
    if USE_COMPILE and hasattr(torch, "compile"):
        try:
            compiled = torch.compile(model)
            dummy = torch.zeros(2, 3 * N_STACKED_FRAMES, RESIZE_DIM[0], RESIZE_DIM[1], device=DEVICE)
            compiled.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE.type == 'cuda')):
                    _ = compiled(dummy)
            model = compiled
            print("    [COMPILE] torch.compile active (warmup passed).")
        except Exception as e:
            msg = str(e).splitlines()[0] if str(e) else type(e).__name__
            print(f"    [COMPILE] Unavailable — running eager. Reason: {msg}")
            model = TViT_Decoupler(n_exps, stack_size=N_STACKED_FRAMES,
                                   freeze_backbone=FREEZE_BACKBONE).to(DEVICE)
    return model


def _torch_load(path):
    """torch.load that works on every torch version: >= 2.6 defaults to
    weights_only=True, which rejects the checkpoint dictionary."""
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:                              # torch < 1.13 has no weights_only
        return torch.load(path, map_location=DEVICE)


def _state_dict(model):
    return getattr(model, "_orig_mod", model).state_dict()


def _load_state(model, sd):
    getattr(model, "_orig_mod", model).load_state_dict(sd)


def train_one_fold(k, day, train_df, test_df, exp_map, mask48, n_exps, frame_store):
    """
    V11.0.0 protocol, unchanged: up to N_EPOCHS, early stopping after
    EARLY_STOP_PATIENCE epochs without improvement of the validation MAE
    (computed on the held-out day), best-MAE weights restored and evaluated.
    Every epoch is checkpointed; an interrupted fold resumes where it stopped.
    Returns (predictions, history, best_epoch).
    """
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    if len(train_df) == 0:
        train_df = test_df

    dl_kwargs = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                     pin_memory=(PIN_MEMORY and DEVICE.type == 'cuda'))
    if NUM_WORKERS > 0:
        dl_kwargs['persistent_workers'] = True
    train_loader = DataLoader(
        IDC_TemporalDataset(train_df, exp_map, mask48, frame_store,
                            N_STACKED_FRAMES, FRAME_STACK_STEP),
        shuffle=True, **dl_kwargs)
    val_loader = DataLoader(
        IDC_TemporalDataset(test_df, exp_map, mask48, frame_store,
                            N_STACKED_FRAMES, FRAME_STACK_STEP),
        shuffle=False, **dl_kwargs)

    model     = _build_model(n_exps)
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    criterion = nn.MSELoss()
    scaler    = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE.type == 'cuda'))
    history   = {'epoch': [], 'train_loss': [], 'val_mae': []}
    start_ep  = 0
    best_mae, best_sd, since_best, best_epoch = float('inf'), None, 0, -1

    def _infer():
        model.eval(); preds = []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(DEVICE, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE.type == 'cuda')):
                    out = model(x)
                preds.extend(np.atleast_1d(out.float().cpu().numpy()))
        return preds

    cp = ckpt_path(k, day)
    if os.path.exists(cp):
        try:
            state = _torch_load(cp)
            _load_state(model, state['model'])
            optimizer.load_state_dict(state['optimizer'])
            if 'scaler' in state and state['scaler'] is not None:
                scaler.load_state_dict(state['scaler'])
            history    = state['history']
            start_ep   = state['epoch'] + 1
            best_mae   = state.get('best_mae', float('inf'))
            best_sd    = state.get('best_sd', None)
            since_best = state.get('since_best', 0)
            best_epoch = state.get('best_epoch', -1)
            print(f"    [RESUME] {size_label(k)}/{day} from epoch {start_ep} "
                  f"(best MAE so far {best_mae:.4f})")
        except Exception as e:
            print(f"    [RESUME] Checkpoint unreadable ({e}) — restarting fold.")

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
        val_mae = float(mean_absolute_error(test_df['oxygen'], val_preds))
        history['epoch'].append(int(epoch))
        history['train_loss'].append(float(l_sum / max(len(train_loader), 1)))
        history['val_mae'].append(val_mae)

        if val_mae < best_mae - EARLY_STOP_MIN_DELTA:
            best_mae, best_sd, since_best, best_epoch = val_mae, _state_dict(model), 0, int(epoch)
        else:
            since_best += 1

        torch.save({'epoch': epoch, 'model': _state_dict(model),
                    'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                    'history': history, 'best_mae': best_mae, 'best_sd': best_sd,
                    'since_best': since_best, 'best_epoch': best_epoch}, cp)
        print(f"    {size_label(k)}/{day}  ep {epoch+1:02d}/{N_EPOCHS}  "
              f"loss={history['train_loss'][-1]:.4f}  val_MAE={val_mae:.4f}  "
              f"(best {best_mae:.4f}, stale {since_best})")

        if USE_EARLY_STOPPING and since_best >= EARLY_STOP_PATIENCE:
            print(f"    [EARLY STOP] {size_label(k)}/{day} — no improvement for "
                  f"{EARLY_STOP_PATIENCE} epochs. Stopping at epoch {epoch+1}.")
            break

    if best_sd is not None:
        _load_state(model, best_sd)
    best_preds = _infer()

    torch.save(_state_dict(model), model_path(k, day))
    if os.path.exists(cp): os.remove(cp)
    return best_preds, history, best_epoch


# ==============================================================================
# 11. SUMMARY FIGURES (B–E) + CSVs
# ==============================================================================
def regime_masks(df):
    """stable = inside the plateau windows; unstructured = before the cutoff;
    transition = everything else."""
    stable = np.zeros(len(df), dtype=bool)
    el = df['elapsed'].values
    for s, e in ANALYSIS_INTERVALS_S:
        stable |= (el >= s) & (el <= e)
    unstruct   = el < UNSTRUCTURED_CUTOFF_S
    transition = (~stable) & (~unstruct)
    return stable, transition, unstruct


def _safe_mae(y, p, m):
    return float(mean_absolute_error(y[m], p[m])) if m.sum() > 0 else np.nan


def _violin(data, x, y, ax, colour):
    """seaborn >= 0.13 uses density_norm; older versions use scale."""
    try:
        sns.violinplot(data=data, x=x, y=y, inner='point', density_norm='width', ax=ax, color=colour)
    except TypeError:
        sns.violinplot(data=data, x=x, y=y, inner='point', scale='width', ax=ax, color=colour)


def plot_summary(summary_df, metrics_df, day_list):
    """
    LOOCV summaries (mean and SD over the held-out days for every size):
      Fig B: MAE (TViT and classical SV) vs number of pixels and vs MPD,
             absolute (mean ± SD) and min-max normalised (means).
      Fig C: TViT MAE per dynamic regime vs number of pixels.
      Fig D: TViT MAE per group size x held-out day (heatmap).
      Fig E: distribution of Q within every group.
    """
    cols = ['MAE_AI', 'MAE_Classical', 'MAE_AI_stable', 'MAE_AI_transition', 'MAE_AI_unstructured']
    g = summary_df.groupby(['group', 'side', 'n_pixels', 'mpd'])
    agg = g[cols].mean().add_suffix('_mean').join(g[cols].std().add_suffix('_sd'))
    agg['n_folds'] = g.size()
    s = agg.reset_index().sort_values('n_pixels').reset_index(drop=True)
    s.to_csv(os.path.join(SUMMARY_DIR, "Fig_B_C_source_data.csv"), index=False)

    def _norm(v):
        v = np.asarray(v, float)
        rng = np.nanmax(v) - np.nanmin(v)
        return (v - np.nanmin(v)) / (rng if rng > 1e-12 else 1.0)

    # ---- Figure B: MAE vs group size (top) and vs MPD (bottom) ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for row, (xcol, xlab) in enumerate([('n_pixels', "Pixels per group (k x k)"),
                                        ('mpd', "Mean pairwise distance (MPD)")]):
        ax1, ax2 = axes[row]
        for col, colr, lbl in [('MAE_AI', 'royalblue', 'AI (TViT, masked)'),
                               ('MAE_Classical', 'gray', 'Classical (group SV)')]:
            ax1.errorbar(s[xcol], s[f'{col}_mean'], yerr=s[f'{col}_sd'].fillna(0), fmt='o-',
                         color=colr, capsize=4, label=lbl)
            ax2.plot(_norm(s[xcol]), _norm(s[f'{col}_mean']), 'o-', color=colr, label=lbl)
        for _, r in s.iterrows():
            ax1.annotate(f"{int(r['side'])}x{int(r['side'])}", (r[xcol], r['MAE_AI_mean']),
                         fontsize=8, xytext=(3, 3), textcoords='offset points')
        if xcol == 'n_pixels':
            ax1.set_xticks(s['n_pixels'])
        ax1.set_xlabel(xlab); ax1.set_ylabel(f"MAE (mean ± SD over {len(day_list)} held-out days)")
        ax1.set_title("Absolute"); ax1.legend()
        ax2.set_xlabel(f"Normalised {xcol} (0–1)"); ax2.set_ylabel("Normalised mean MAE (0–1)")
        ax2.set_title("Min-max normalised"); ax2.legend()
    fig.suptitle("Reconstruction error vs size of the widest-MPD pixel group (LOOCV)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(SUMMARY_DIR, "Fig_B_MAE_vs_GroupSize.png"), dpi=250, bbox_inches='tight')
    plt.close()

    # ---- Figure C: regime-decomposed TViT MAE vs group size ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    for col, colr, lbl in [('MAE_AI_stable', 'green', 'Stable plateaux'),
                           ('MAE_AI_transition', 'crimson', 'Transitions'),
                           ('MAE_AI_unstructured', 'darkorange', 'Unstructured (<1000 s)')]:
        ax1.plot(s['n_pixels'], s[f'{col}_mean'], 'o-', color=colr, label=lbl)
    ax1.set_xticks(s['n_pixels'])
    ax1.set_xlabel("Pixels per group"); ax1.set_ylabel("AI MAE (LOOCV mean)")
    ax1.set_title("AI error by dynamic regime"); ax1.legend()
    ratio = s['MAE_AI_transition_mean'] / s['MAE_AI_stable_mean']
    ax2.plot(s['n_pixels'], ratio, 's-', color='purple')
    ax2.axhline(1.0, color='k', ls=':', lw=1)
    ax2.set_xticks(s['n_pixels'])
    ax2.set_xlabel("Pixels per group"); ax2.set_ylabel("Transition MAE / Stable MAE")
    ax2.set_title("Dynamics penalty ratio (lower = better at transients)")
    fig.suptitle("Where does a larger diverse aperture help?", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(SUMMARY_DIR, "Fig_C_Regime_MAE.png"), dpi=250, bbox_inches='tight')
    plt.close()

    # ---- Figure D: size x held-out day heatmap ----
    piv = summary_df.pivot_table(index='n_pixels', columns='fold_day', values='MAE_AI')
    piv = piv[[d for d in day_list if d in piv.columns]]               # numeric day order
    piv.index = [f"{int(np.sqrt(n))}x{int(np.sqrt(n))}" for n in piv.index]
    piv.to_csv(os.path.join(SUMMARY_DIR, "Fig_D_source_data.csv"))
    fig, ax = plt.subplots(figsize=(1.2 * max(len(piv.columns), 4) + 3, 0.55 * len(piv) + 2))
    sns.heatmap(piv, annot=True, fmt=".2f", cmap='viridis_r', cbar_kws={'label': 'AI MAE'}, ax=ax)
    ax.set_xlabel("Held-out day"); ax.set_ylabel("Group size")
    ax.set_title("AI MAE per group size × held-out day")
    plt.tight_layout()
    plt.savefig(os.path.join(SUMMARY_DIR, "Fig_D_Size_x_Day_Heatmap.png"), dpi=250, bbox_inches='tight')
    plt.close()

    # ---- Figure E: Q distribution within every group ----
    assign = pd.read_csv(os.path.join(GROUPDEF_DIR, "group_assignments.csv"))
    assign['size'] = assign['side'].astype(int).astype(str) + "x" + assign['side'].astype(int).astype(str)
    assign = assign.sort_values('n_pixels')
    assign[['group', 'size', 'n_pixels', 'pixel_idx', 'quality']].to_csv(
        os.path.join(SUMMARY_DIR, "Fig_E_source_data.csv"), index=False)
    fig, ax = plt.subplots(figsize=(14, 6))
    _violin(assign, 'size', 'quality', ax, 'steelblue')
    ax.set_xlabel("Group size"); ax.set_ylabel("Composite quality score (z)")
    ax.set_title("Quality spread within the widest-MPD group of each size")
    plt.tight_layout()
    plt.savefig(os.path.join(SUMMARY_DIR, "Fig_E_Quality_Distributions.png"), dpi=250, bbox_inches='tight')
    plt.close()
    print(f"[SUMMARY] Figures B–E + source CSVs → {SUMMARY_DIR}")


# ==============================================================================
# 12. EXECUTION ORCHESTRATOR
# ==============================================================================
def main():
    print("=" * 74)
    print("  GROUP-SIZE STUDY V11.4.3-Local — widest-MPD groups, LOOCV over chosen days")
    print(f"  GROUP SIZES={[f'{k}x{k}' for k in GROUP_SIDES]}  DAYS_TO_RUN={DAYS_TO_RUN}  DEVICE={DEVICE}")
    print(f"  SPEED: AMP={USE_AMP}  frames=shared uint8 store  pin_memory={PIN_MEMORY}  "
          f"freeze_backbone={FREEZE_BACKBONE}  early_stop={USE_EARLY_STOPPING}"
          f"(patience={EARLY_STOP_PATIENCE})  compile={USE_COMPILE}  workers={NUM_WORKERS}")
    print(f"  OUTPUT: {STUDY_DIR}")
    print("=" * 74)

    # ---- data: the days listed in DAYS_TO_RUN (by the number in front of the ID) ----
    experiments = find_experiment_files_v10(EXPERIMENTS_ROOT_DIR)
    df = get_master_df(experiments, MASTER_PARQUET)
    all_days = sorted(df['experiment_id'].astype(str).unique(), key=day_order_key)
    day_list = select_days(all_days)
    if len(day_list) < 2:
        print("CRITICAL ERROR: leave-one-day-out CV needs at least 2 days; add days to DAYS_TO_RUN.")
        sys.exit(1)
    make_output_dirs()
    print("[DATA] Available days (number in front: ID); * = held out in turn in this run")
    for d in all_days:
        k = day_order_key(d)
        print(f"   {'*' if d in day_list else ' '} {k[1] if k[0] == 0 else '-':>3}: {d}")
    df = df[df['experiment_id'].isin(day_list)].reset_index(drop=True)
    print(f"[COST] {len(GROUP_SIDES)} group sizes × {len(day_list)} held-out days = "
          f"{len(GROUP_SIDES) * len(day_list)} TViT trainings of up to {N_EPOCHS} epochs. "
          f"Finished runs are skipped; partial ones resume.")

    # ---- groups (features over the N days, as V11.0.0) ----
    features = compute_pixel_features(df, day_list)
    groups, metrics_df, quality, rank_order = build_widest_groups(features)
    plot_group_panels(df, groups, metrics_df, day_list[0])
    df = add_group_intensity_columns(df, groups)

    # ---- shared uint8 frame store (built once, read by every size and fold) ----
    df['frame_idx'] = np.arange(len(df))
    frame_store = torch.from_numpy(
        np.stack(df['pixels'].values).reshape(len(df), STORAGE_DIM[0], STORAGE_DIM[1], 3))
    df = df.drop(columns=['pixels'])
    gc.collect()
    print(f"[FRAMES] Shared uint8 frame store: {len(df)} frames, "
          f"{len(df) * STORAGE_DIM[0] * STORAGE_DIM[1] * 3 / 1e9:.2f} GB "
          f"(used by every group size and fold; per-frame pixel arrays released).")

    exp_map = {name: i for i, name in enumerate(day_list)}

    # ---- resumable summary (one row per group size x held-out day) ----
    summary_rows = (pd.read_csv(GLOBAL_SUMMARY_PATH).to_dict('records')
                    if os.path.exists(GLOBAL_SUMMARY_PATH) else [])
    done = {(str(r['group']), str(r['fold_day'])) for r in summary_rows}
    failed = []                     # (size, day) runs whose training raised an error

    for gr in groups:
        k, label = gr['side'], gr['label']
        out_dir = group_dir(k)
        gm = metrics_df.loc[metrics_df['group'] == label].iloc[0]
        mask48 = np.isin(np.arange(N_PIXELS_TOTAL), gr['pix']).reshape(STORAGE_DIM)
        print(f"\n{'='*66}\n[{label}] N={gr['n']}  MPD={gm['mpd']:.3f}  "
              f"Q_sd={gm['q_sd']:.3f}  Q_range={gm['q_range']:.1f}%\n{'='*66}")

        for fold_idx, test_day in enumerate(day_list):
            if (label, test_day) in done:
                print(f"  [SKIP] {label}/{test_day} already complete.")
                continue
            print(f"\n  Fold {fold_idx + 1}/{len(day_list)}: hold out {test_day}")
            train_df = df[df['experiment_id'] != test_day]
            test_df = df[df['experiment_id'] == test_day].copy()

            print("  > Fitting group-level Stern-Volmer Two-Site Model (training days only)...")
            test_df['o2_classical'] = fit_classical_baseline(train_df, test_df, inv_col=f"inv_i_{label}")

            try:
                val_preds, history, best_epoch = train_one_fold(k, test_day, train_df, test_df,
                                                                exp_map, mask48, n_exps=len(day_list),
                                                                frame_store=frame_store)
            except Exception as e:
                print(f"  [ERROR] Training failed for {label}/{test_day}: {e}")
                traceback.print_exc(); failed.append(f"{label}/{test_day}")
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                continue

            test_df['o2_pred'] = val_preds
            try:
                plot_training_dashboard(history, test_day, out_dir)
                analyze_dynamics_v10(test_df, test_day, out_dir)
                plot_chemometrics(test_df, test_day, out_dir)
            except Exception as e:
                print(f"  [WARN] per-fold plots failed for {label}/{test_day} ({e}); metrics still saved.")

            y = test_df['oxygen'].values
            pa = test_df['o2_pred'].values
            pc = test_df['o2_classical'].values
            m_st, m_tr, m_un = regime_masks(test_df)
            row = {
                'group': label, 'side': k, 'n_pixels': gr['n'], 'fold_day': test_day,
                'day_number': day_order_key(test_day)[1],
                'mpd': float(gm['mpd']), 'q_mean': float(gm['q_mean']),
                'q_sd': float(gm['q_sd']), 'q_range': float(gm['q_range']),
                'window_end_rank': int(gm['window_end_rank']),
                'n_epochs_trained': len(history['epoch']), 'best_epoch': int(best_epoch) + 1,
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
            print(f"  [DONE] {label}/{test_day}  MAE_AI={row['MAE_AI']:.4f}  "
                  f"MAE_Classical={row['MAE_Classical']:.4f}")
            del train_df, test_df
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    summary_df = pd.DataFrame(summary_rows)
    if len(summary_df) > 0:
        try:
            plot_summary(summary_df, metrics_df, day_list)
        except Exception:
            print("  [WARN] Summary figures failed - all results are already saved in "
                  f"{GLOBAL_SUMMARY_PATH}. Traceback:")
            traceback.print_exc()

    if failed:
        print("\n" + "!" * 74)
        print(f"  [WARN] {len(failed)} training run(s) FAILED and have no results: {failed}")
        print("  Fix the error shown above and rerun - completed runs are skipped.")
        print("!" * 74)
    print("\n" + "=" * 74)
    print("  GROUP-SIZE STUDY COMPLETE")
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
