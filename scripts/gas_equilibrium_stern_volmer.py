# Repository edition: see docs/CHANGES.md and docs/REPRODUCIBILITY.md.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
PROJECT: PINN-IDC (Inverse Dynamic Compensation) Framework
SCRIPT : Classical Stern-Volmer Baseline — Equilibrium Calibration
VERSION: 15.0-EQ
==============================================================================

--- WHY THIS SCRIPT EXISTS ---
Up to V11.1.3 / V14.1-R the classical two-site Stern-Volmer (SV) baseline was
fitted on EVERY training frame (stable + transient + unstructured). During a
transient the reference O2 has already moved while the optode intensity still
lags behind, so those (O2, I) pairs sit off the equilibrium curve and trace a
hysteresis loop. A static calibration fitted through that loop is biased.

Standard practice is to calibrate on equilibrium points and THEN apply the
static calibration to dynamic data. That is what this script does:

  1. Stable plateaus come from the per-experiment REGION_DEFINITIONS (the same
     definitions used by the T90 response-time script).
  2. Each plateau is trimmed at its leading edge (the optode is still settling
     after the reference has stabilised) and checked for residual drift in
     optode intensity and in the reference O2. Plateaus that fail are logged
     and excluded.
  3. Plateaus that pass become calibration points — by default ONE median
     (O2, I) point per plateau, so long plateaus do not dominate the fit.
  4. Leave-one-experiment-out cross-validation (as V11): the held-out
     experiment never contributes calibration points to its own fold.
  5. The fitted model is applied to EVERY frame of the held-out experiment
     (full inference across stable, transient and unstructured regimes).

For a controlled comparison the legacy all-region fit is recomputed on exactly
the same experiments and folds, so the ONLY difference between the two
classical columns is the calibration set. AI predictions from the V14.1-R
cache are merged in when available so that regime metrics and T90 response
times compare all models side by side.

--- INPUTS ---
  * Per-frame optode intensity + reference O2. The V11 inference CSVs (and therefore the V14
    caches) do not contain intensity, so it comes from the first READABLE parquet among:
    V11_CACHE_FILE (if set) -> this script's own 5_Data/_frame_cache_step{N}.parquet ->
    Dynamic_IDC_Analysis_V11_1/4_Caches/idc_master_*.parquet -> other caches. Empty or corrupt
    files are skipped. If none works, intensities are re-extracted from the raw videos exactly
    as V11 did (mean red channel of the 48x48 frame) and cached for the next run.
  * V14.1-R cache    : PostHoc_Analysis_MultiModel_V14_1_R/_cache/kalman_df_v141R_*.parquet
                       (optional — AI predictions, V14's RTS Kalman columns, regime labels).

--- MODEL COLUMNS PRODUCED ---
  o2_sv_eq      Two-site SV calibrated on equilibrium plateaus (NEW baseline)
  o2_sv_all     Two-site SV calibrated on all frames (legacy V11 method, recomputed)
  *_rts         RTS Kalman-smoothed version of every model column, using the V14.1-R smoother
                and parameters (AI columns are V14's own '<model>_kalman' values)
  I_eq_expected Intensity the equilibrium calibration expects at the reference O2
                (measured - expected intensity shows the lag in the intensity domain)

--- OUTPUTS (every figure is saved as PNG + PDF with a backing CSV of the same name) ---
  PostHoc_ClassicalSV_EquilibriumCal_V15_0/
    0_Equilibrium_Selection/   plateau table + per-experiment timelines of the windows used/rejected
    1_Calibration/             per-fold calibration curves (eq vs legacy), parameters across folds
    2_Inference/               full-inference timelines with error traces, per experiment
    3_Chemometrics/            per-experiment parity/residuals, global + regime parity grid,
                               regime MAE bars, per-experiment MAE distribution
    4_Response_T90/            T90 tables (Savgol + raw-aligned), parity, binned lag bars,
                               step-response zooms for every experiment
    5_Data/                    full inference (Parquet + CSV), per-experiment CSVs, metrics,
                               calibration parameters (CSV + JSON), Excel summary workbook
    csv_source/                drop-in files for the publication plotter
                               (Inference_<exp>.csv, Parity_<regime>.csv). Point the plotter's
                               CSV_SOURCE here: 'Classical SV' in the Parity files is the NEW
                               equilibrium-calibrated baseline; 'o2_classical' in the Inference
                               files is the same column.
    run_manifest.json          configuration, inputs and library versions for this run
==============================================================================
"""

# Portable repository paths; edit scripts/repo_config.py for your computer.
from repo_config import data_path, output_path, cache_path, external_path

import os
import re
import glob
import json
import platform
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
SEED = 42
np.random.seed(SEED)

# --- Paths ---------------------------------------------------------------------
BASE_PATH = data_path('gas')
OUTPUT_FOLDER_NAME = "PostHoc_ClassicalSV_EquilibriumCal_V15_0"
V11_FOLDER_NAME = "Dynamic_IDC_Analysis_V11_1"            # per-frame intensity + reference O2
V11_STEP_RATE = 1                                         # reads idc_master_v11_3_step{N}.parquet
V11_CACHE_FILE = external_path("gas_frame_cache")  # None means automatic discovery    # optional: full path to a specific frame-cache .parquet to use (overrides the search)
V14_FOLDER_NAME = "PostHoc_Analysis_MultiModel_V14_1_R"   # optional AI predictions for comparison

# --- Acquisition (must match V11 so intensities are identical) -------------------
ASSUMED_FPS = 10.0
STORAGE_DIM = (48, 48)

# --- Equilibrium point selection ---------------------------------------------------
EQ_TRIM_LEADING_FRAC = 0.5      # discard this fraction of each plateau's start (optode still settling)
EQ_MIN_FRAMES = 10              # minimum frames left in a plateau after trimming
EQ_MIN_DURATION_S = 2.0         # minimum duration of the kept window [s]
EQ_MAX_REL_I_SLOPE = 0.001      # max |dI/dt| / median(I) inside the kept window [1/s] (0.001 = 0.1 %/s).
                                # A rate (not a total drift) so short, still-settling windows are caught.
                                # Check 0_Equilibrium_Selection/Equilibrium_Plateau_Table.csv and tune.
EQ_MAX_GT_DRIFT = 0.5           # max reference O2 drift across the kept window [%O2]
EQ_FIT_MODE = "plateau_median"  # "plateau_median": one (median O2, median I) point per plateau
                                # "pointwise"     : every frame inside the kept windows
REGIME_PRECEDENCE = "unstructured_over_stable"   # or "stable_over_unstructured" where definitions overlap

# --- Two-site Stern-Volmer fit ------------------------------------------------------
SV_BOUNDS = ([0, 0, 0, 0], [np.inf, 1.0, np.inf, np.inf])   # same as V11
SV_START_SHAPES = [(0.8, 0.05, 0.005),                      # V11 default start (f, K1, K2)
                   (0.5, 0.10, 0.010),
                   (0.9, 0.02, 0.001)]
SV_I0_START_FACTORS = [1.0, 1.1]                            # I0 start = factor * max(I)
SV_MAXFEV = 20000

# --- RTS Kalman smoothing: exact copy of the V14.1-R smoother and parameters ------------
# Classical columns are smoothed here with V14's smoother; AI columns take V14's own
# '<model>_kalman' columns from the V14 cache (smoothed here only if those are missing).
APPLY_RTS = True
KALMAN_MEASUREMENT_NOISE = 8.0      # V14.1-R values, unchanged
KALMAN_PROCESS_NOISE_CONC = 0.2
KALMAN_PROCESS_NOISE_RATE = 0.003
KALMAN_STABLE_TIGHTENING = 0.1

# --- T90 response time ---------------------------------------------------------------
T90_MIN_STEP = 0.5  # [%O2] ignore steps smaller than this (same as the T90 script)

# --- Plotting ------------------------------------------------------------------------
TARGET_EXP = "20260426_164709_50s"
ZOOM_PRE_S = 10.0          # seconds shown before each step in the step-response zooms
N_ZOOM_STEPS = 6           # steps shown per experiment in the zoom figures
MAX_CAL_SCATTER = 30000    # training frames drawn in the calibration plots (subsampled for speed)
FIG_FORMATS = ("png", "pdf")
FIG_DPI = 300
UNIT = "O$_2$ [V/V %]"

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.25)

# --- Model registry ------------------------------------------------------------------
# 'parity_label' is the value written to the 'model' column of csv_source/Parity_*.csv,
# chosen so the publication plotter's lookups ('Classical SV', 'AI (V11.1)', ...) keep working.
MODEL_REGISTRY = {
    'o2_sv_eq':       {'name': 'Classical SV (Eq-Cal)',  'parity_label': 'Classical SV',               'color': '#ff7f0e', 'ls': '-'},
    'o2_sv_all':      {'name': 'Classical SV (All-Cal)', 'parity_label': 'Legacy SV (All-Region Cal)', 'color': '#7f7f7f', 'ls': '--'},
    'o2_ai_v11_1':    {'name': 'TViT (V11.1)',           'parity_label': 'AI (V11.1)',                 'color': '#1f77b4', 'ls': '-'},
    'o2_ai_v11_5_sf': {'name': 'AI (V11.5 SF)',          'parity_label': 'AI (V11.5 SF)',              'color': '#2ca02c', 'ls': '-'},
    'o2_ai_v12_3_ac': {'name': 'Autograd (V12.3 AC)',    'parity_label': 'AI (V12.3 AC)',              'color': '#d62728', 'ls': '-'},
}
CLASSICAL_KEYS = ['o2_sv_eq', 'o2_sv_all']
V14_AI_KEYS = ['o2_ai_v11_1', 'o2_ai_v11_5_sf', 'o2_ai_v12_3_ac']

REGIMES = ['Stable', 'Transient', 'Unstructured']
REGIME_COLORS = {'Stable': '#2ca02c', 'Transient': '#d62728', 'Unstructured': '#7f7f7f'}

# --- Per-experiment region definitions (identical to the T90 script) --------------------
REGION_DEFINITIONS = {
    "20260426_140945_15s": {"filename": "Inference_continuous_experiment_20260426_140945_15s.csv",
                            "regions": [{"type": "stable", "index": 0, "start": 109.7, "end": 125.7},
                                        {"type": "stable", "index": 1, "start": 134.7, "end": 141},
                                        {"type": "stable", "index": 2, "start": 146.6, "end": 154.9},
                                        {"type": "stable", "index": 3, "start": 163.9, "end": 170.9},
                                        {"type": "stable", "index": 4, "start": 179.2, "end": 185.5},
                                        {"type": "stable", "index": 5, "start": 196.6, "end": 199.3},
                                        {"type": "stable", "index": 6, "start": 210.5, "end": 213.2},
                                        {"type": "stable", "index": 7, "start": 225, "end": 230.6},
                                        {"type": "stable", "index": 8, "start": 240.3, "end": 246.6},
                                        {"type": "stable", "index": 9, "start": 255.6, "end": 261.2},
                                        {"type": "stable", "index": 10, "start": 269.5, "end": 275.1},
                                        {"type": "stable", "index": 11, "start": 284.8, "end": 291},
                                        {"type": "stable", "index": 12, "start": 300.1, "end": 306.3},
                                        {"type": "unstructured", "index": 0, "start": 309.1, "end": 382.7},
                                        {"type": "unstructured", "index": 1, "start": 1.4, "end": 93.8}]},
    "20260426_141634_10s": {"filename": "Inference_continuous_experiment_20260426_141634_10s.csv",
                            "regions": [{"type": "stable", "index": 0, "start": 67.2, "end": 73.6},
                                        {"type": "stable", "index": 1, "start": 80.1, "end": 84.6},
                                        {"type": "stable", "index": 2, "start": 91.5, "end": 96},
                                        {"type": "stable", "index": 3, "start": 99.8, "end": 105.5},
                                        {"type": "stable", "index": 4, "start": 112.4, "end": 115},
                                        {"type": "stable", "index": 5, "start": 121.1, "end": 124.9},
                                        {"type": "stable", "index": 6, "start": 132.8, "end": 135.1},
                                        {"type": "stable", "index": 7, "start": 142.3, "end": 146.9},
                                        {"type": "stable", "index": 8, "start": 151.4, "end": 154.5},
                                        {"type": "stable", "index": 9, "start": 161.3, "end": 165.5},
                                        {"type": "stable", "index": 10, "start": 173.1, "end": 178},
                                        {"type": "unstructured", "index": 0, "start": 0, "end": 53.9},
                                        {"type": "unstructured", "index": 1, "start": 178, "end": 209.9},
                                        {"type": "stable", "index": 11, "start": 48.6, "end": 53.9},
                                        {"type": "stable", "index": 12, "start": 63.4, "end": 69.1}]},
    "20260426_142034_20s": {"filename": "Inference_continuous_experiment_20260426_142034_20s.csv",
                            "regions": [{"type": "unstructured", "index": 0, "start": 359.3, "end": 450.5},
                                        {"type": "unstructured", "index": 1, "start": 0, "end": 78.2},
                                        {"type": "stable", "index": 0, "start": 95.3, "end": 117.3},
                                        {"type": "stable", "index": 1, "start": 126.3, "end": 138.5},
                                        {"type": "stable", "index": 2, "start": 147.5, "end": 158},
                                        {"type": "stable", "index": 3, "start": 169.4, "end": 176.8},
                                        {"type": "stable", "index": 4, "start": 185.7, "end": 197.1},
                                        {"type": "stable", "index": 5, "start": 206.9, "end": 217.5},
                                        {"type": "stable", "index": 6, "start": 227.3, "end": 240.3},
                                        {"type": "stable", "index": 7, "start": 251.7, "end": 259.9},
                                        {"type": "stable", "index": 8, "start": 268.8, "end": 277.8},
                                        {"type": "stable", "index": 9, "start": 285.1, "end": 297.3},
                                        {"type": "stable", "index": 10, "start": 303.9, "end": 317.7},
                                        {"type": "stable", "index": 11, "start": 328.3, "end": 336.4},
                                        {"type": "stable", "index": 12, "start": 347, "end": 357.6}]},
    "20260426_142819_25s": {"filename": "Inference_continuous_experiment_20260426_142819_25s.csv",
                            "regions": [{"type": "stable", "index": 0, "start": 113.7, "end": 141.2},
                                        {"type": "stable", "index": 1, "start": 153, "end": 166.7},
                                        {"type": "stable", "index": 2, "start": 176.5, "end": 189.2},
                                        {"type": "stable", "index": 3, "start": 202, "end": 213.7},
                                        {"type": "stable", "index": 4, "start": 226.5, "end": 239.2},
                                        {"type": "stable", "index": 5, "start": 253.9, "end": 264.7},
                                        {"type": "stable", "index": 6, "start": 280.4, "end": 292.2},
                                        {"type": "stable", "index": 7, "start": 303.9, "end": 313.8},
                                        {"type": "stable", "index": 8, "start": 327.5, "end": 337.3},
                                        {"type": "stable", "index": 9, "start": 353.9, "end": 366.7},
                                        {"type": "stable", "index": 10, "start": 379.4, "end": 391.2},
                                        {"type": "stable", "index": 11, "start": 404, "end": 413.8},
                                        {"type": "stable", "index": 12, "start": 426.5, "end": 443.2},
                                        {"type": "unstructured", "index": 0, "start": 0, "end": 90.2},
                                        {"type": "unstructured", "index": 1, "start": 453, "end": 542.2}]},
    "20260426_143744_30s": {"filename": "Inference_continuous_experiment_20260426_143744_30s.csv",
                            "regions": [{"type": "stable", "index": 0, "start": 372.6, "end": 385},
                                        {"type": "stable", "index": 1, "start": 407.8, "end": 420.2},
                                        {"type": "stable", "index": 2, "start": 434.7, "end": 442.9},
                                        {"type": "stable", "index": 3, "start": 465.7, "end": 474},
                                        {"type": "stable", "index": 4, "start": 492.6, "end": 503},
                                        {"type": "stable", "index": 5, "start": 113.8, "end": 147},
                                        {"type": "stable", "index": 6, "start": 161.4, "end": 173.9},
                                        {"type": "stable", "index": 7, "start": 196.6, "end": 207},
                                        {"type": "stable", "index": 8, "start": 223.5, "end": 236},
                                        {"type": "stable", "index": 9, "start": 248.4, "end": 269.1},
                                        {"type": "stable", "index": 10, "start": 281.5, "end": 300.1},
                                        {"type": "stable", "index": 11, "start": 308.4, "end": 320.8},
                                        {"type": "stable", "index": 12, "start": 519.5, "end": 529.9},
                                        {"type": "unstructured", "index": 0, "start": 534, "end": 1144.6},
                                        {"type": "unstructured", "index": 1, "start": 0, "end": 82.8}]},
    "20260426_145711_35s": {"filename": "Inference_continuous_experiment_20260426_145711_35s.csv",
                            "regions": [{"type": "unstructured", "index": 0, "start": 0, "end": 83.2},
                                        {"type": "unstructured", "index": 1, "start": 619.8, "end": 979.3},
                                        {"type": "stable", "index": 0, "start": 113.3, "end": 152.3},
                                        {"type": "stable", "index": 1, "start": 593.2, "end": 609.2},
                                        {"type": "stable", "index": 2, "start": 559.6, "end": 573.8},
                                        {"type": "stable", "index": 3, "start": 533, "end": 547.2},
                                        {"type": "stable", "index": 4, "start": 490.5, "end": 506.5},
                                        {"type": "stable", "index": 5, "start": 455.1, "end": 474.6},
                                        {"type": "stable", "index": 6, "start": 428.6, "end": 441},
                                        {"type": "stable", "index": 7, "start": 357.7, "end": 370.1},
                                        {"type": "stable", "index": 8, "start": 177.1, "end": 196.6},
                                        {"type": "stable", "index": 9, "start": 214.3, "end": 232},
                                        {"type": "stable", "index": 10, "start": 244.4, "end": 260.3},
                                        {"type": "stable", "index": 11, "start": 283.3, "end": 299.3},
                                        {"type": "stable", "index": 12, "start": 313.4, "end": 334.7}]},
    "20260426_151354_60s": {"filename": "Inference_continuous_experiment_20260426_151354_60s.csv",
                            "regions": [{"type": "stable", "index": 0, "start": 161.1, "end": 236},
                                        {"type": "stable", "index": 1, "start": 258.7, "end": 288.2},
                                        {"type": "stable", "index": 2, "start": 315.5, "end": 351.8},
                                        {"type": "stable", "index": 3, "start": 369.9, "end": 408.5},
                                        {"type": "stable", "index": 4, "start": 442.5, "end": 474.3},
                                        {"type": "stable", "index": 5, "start": 492.5, "end": 535.6},
                                        {"type": "stable", "index": 6, "start": 560.6, "end": 590.1},
                                        {"type": "stable", "index": 7, "start": 680.8, "end": 714.9},
                                        {"type": "stable", "index": 8, "start": 739.8, "end": 771.6},
                                        {"type": "stable", "index": 9, "start": 796.6, "end": 828.3},
                                        {"type": "stable", "index": 10, "start": 862.4, "end": 889.6},
                                        {"type": "stable", "index": 11, "start": 923.7, "end": 953.2},
                                        {"type": "stable", "index": 12, "start": 984.9, "end": 1009.9},
                                        {"type": "unstructured", "index": 0, "start": 1034.9, "end": 1255},
                                        {"type": "unstructured", "index": 1, "start": 0, "end": 111.2}]},
    "20260426_153459_45s": {"filename": "Inference_continuous_experiment_20260426_153459_45s.csv",
                            "regions": [{"type": "stable", "index": 0, "start": 139.1, "end": 208.7},
                                        {"type": "stable", "index": 1, "start": 270.5, "end": 293.7},
                                        {"type": "stable", "index": 2, "start": 324.6, "end": 355.5},
                                        {"type": "stable", "index": 3, "start": 757.3, "end": 803.7},
                                        {"type": "stable", "index": 4, "start": 726.4, "end": 749.6},
                                        {"type": "stable", "index": 5, "start": 672.3, "end": 695.5},
                                        {"type": "stable", "index": 6, "start": 626, "end": 649.1},
                                        {"type": "stable", "index": 7, "start": 571.9, "end": 602.8},
                                        {"type": "stable", "index": 8, "start": 510, "end": 564.1},
                                        {"type": "stable", "index": 9, "start": 417.3, "end": 463.7},
                                        {"type": "stable", "index": 10, "start": 231.8, "end": 247.3},
                                        {"type": "unstructured", "index": 0, "start": 834.6, "end": 4273.5},
                                        {"type": "unstructured", "index": 1, "start": 0, "end": 108.2},
                                        {"type": "stable", "index": 11, "start": 394.1, "end": 425},
                                        {"type": "stable", "index": 12, "start": 672.3, "end": 695.5}]},
    "20260426_164709_50s": {"filename": "Inference_continuous_experiment_20260426_164709_50s.csv",
                            "regions": [{"type": "stable", "index": 0, "start": 139.1, "end": 206.2},
                                        {"type": "stable", "index": 1, "start": 228.5, "end": 248.4},
                                        {"type": "stable", "index": 2, "start": 273.2, "end": 305.5},
                                        {"type": "stable", "index": 3, "start": 322.9, "end": 350.3},
                                        {"type": "stable", "index": 4, "start": 380.1, "end": 397.5},
                                        {"type": "stable", "index": 5, "start": 827.2, "end": 849.6},
                                        {"type": "stable", "index": 6, "start": 782.5, "end": 802.4},
                                        {"type": "stable", "index": 7, "start": 732.8, "end": 755.2},
                                        {"type": "stable", "index": 8, "start": 668.2, "end": 703},
                                        {"type": "stable", "index": 9, "start": 623.5, "end": 658.3},
                                        {"type": "stable", "index": 10, "start": 578.8, "end": 601.1},
                                        {"type": "stable", "index": 11, "start": 539, "end": 556.4},
                                        {"type": "stable", "index": 12, "start": 481.9, "end": 504.3},
                                        {"type": "unstructured", "index": 0, "start": 866.9, "end": 1373.7},
                                        {"type": "unstructured", "index": 1, "start": 0, "end": 101.8}]},
    "20260426_171016_55s": {"filename": "Inference_continuous_experiment_20260426_171016_55s.csv",
                            "regions": [{"type": "stable", "index": 0, "start": 162.8, "end": 235.1},
                                        {"type": "stable", "index": 1, "start": 256.2, "end": 280.3},
                                        {"type": "stable", "index": 2, "start": 301.4, "end": 337.6},
                                        {"type": "stable", "index": 3, "start": 364.7, "end": 388.8},
                                        {"type": "stable", "index": 4, "start": 910.3, "end": 946.5},
                                        {"type": "stable", "index": 5, "start": 868.1, "end": 889.2},
                                        {"type": "stable", "index": 6, "start": 804.8, "end": 828.9},
                                        {"type": "stable", "index": 7, "start": 756.6, "end": 771.6},
                                        {"type": "stable", "index": 8, "start": 690.3, "end": 729.4},
                                        {"type": "stable", "index": 9, "start": 648.1, "end": 663.1},
                                        {"type": "stable", "index": 10, "start": 533.5, "end": 548.6},
                                        {"type": "stable", "index": 11, "start": 485.3, "end": 497.3},
                                        {"type": "stable", "index": 12, "start": 428, "end": 443.1},
                                        {"type": "unstructured", "index": 0, "start": 0, "end": 120.6},
                                        {"type": "unstructured", "index": 1, "start": 964.5, "end": 1591.5}]},
    "20260426_173701_40s": {"filename": "Inference_continuous_experiment_20260426_173701_40s_0.csv",
                            "regions": [{"type": "unstructured", "index": 0, "start": 0, "end": 96.4},
                                        {"type": "unstructured", "index": 1, "start": 713, "end": 1065.7},
                                        {"type": "stable", "index": 0, "start": 136.8, "end": 177.3},
                                        {"type": "stable", "index": 1, "start": 202.3, "end": 229.3},
                                        {"type": "stable", "index": 2, "start": 246.7, "end": 264},
                                        {"type": "stable", "index": 3, "start": 281.4, "end": 304.5},
                                        {"type": "stable", "index": 4, "start": 323.8, "end": 343},
                                        {"type": "stable", "index": 5, "start": 360.4, "end": 389.3},
                                        {"type": "stable", "index": 6, "start": 680.3, "end": 705.3},
                                        {"type": "stable", "index": 7, "start": 639.8, "end": 659.1},
                                        {"type": "stable", "index": 8, "start": 597.4, "end": 622.5},
                                        {"type": "stable", "index": 9, "start": 556.9, "end": 576.2},
                                        {"type": "stable", "index": 10, "start": 520.3, "end": 537.7},
                                        {"type": "stable", "index": 11, "start": 481.8, "end": 499.1},
                                        {"type": "stable", "index": 12, "start": 441.3, "end": 462.5}]},
    "20260426_175507_40s": {"filename": "Inference_continuous_experiment_20260426_175507_40s_1.csv",
                            "regions": [{"type": "unstructured", "index": 0, "start": 0, "end": 95.5},
                                        {"type": "unstructured", "index": 1, "start": 700.9, "end": 1389.3},
                                        {"type": "stable", "index": 0, "start": 130.6, "end": 178.4},
                                        {"type": "stable", "index": 1, "start": 680.8, "end": 695.9},
                                        {"type": "stable", "index": 2, "start": 640.6, "end": 660.7},
                                        {"type": "stable", "index": 3, "start": 595.4, "end": 613},
                                        {"type": "stable", "index": 4, "start": 550.2, "end": 577.8},
                                        {"type": "stable", "index": 5, "start": 515, "end": 535.1},
                                        {"type": "stable", "index": 6, "start": 477.3, "end": 494.9},
                                        {"type": "stable", "index": 7, "start": 389.4, "end": 427.1},
                                        {"type": "stable", "index": 8, "start": 356.7, "end": 384.4},
                                        {"type": "stable", "index": 9, "start": 314, "end": 341.7},
                                        {"type": "stable", "index": 10, "start": 268.8, "end": 291.4},
                                        {"type": "stable", "index": 11, "start": 233.6, "end": 256.3},
                                        {"type": "stable", "index": 12, "start": 193.4, "end": 211}]},
    "20260426_181821_40s": {"filename": "Inference_continuous_experiment_20260426_181821_40s_2.csv",
                            "regions": [{"type": "stable", "index": 0, "start": 125.9, "end": 179.2},
                                        {"type": "stable", "index": 1, "start": 194.2, "end": 213.4},
                                        {"type": "stable", "index": 2, "start": 239, "end": 258.2},
                                        {"type": "stable", "index": 3, "start": 682.8, "end": 706.2},
                                        {"type": "stable", "index": 4, "start": 642.2, "end": 661.4},
                                        {"type": "stable", "index": 5, "start": 603.8, "end": 620.9},
                                        {"type": "stable", "index": 6, "start": 561.1, "end": 573.9},
                                        {"type": "stable", "index": 7, "start": 524.9, "end": 537.7},
                                        {"type": "stable", "index": 8, "start": 275.2, "end": 294.4},
                                        {"type": "stable", "index": 9, "start": 488.6, "end": 497.1},
                                        {"type": "stable", "index": 10, "start": 332.8, "end": 341.4},
                                        {"type": "stable", "index": 11, "start": 360.6, "end": 373.4},
                                        {"type": "stable", "index": 12, "start": 396.9, "end": 416.1},
                                        {"type": "unstructured", "index": 0, "start": 0, "end": 96},
                                        {"type": "unstructured", "index": 1, "start": 708.4, "end": 1179.9}]},
    "20260426_183804_40s": {"filename": "Inference_continuous_experiment_20260426_183804_40s_3.csv",
                            "regions": [{"type": "unstructured", "index": 0, "start": 706.5, "end": 1575.3},
                                        {"type": "unstructured", "index": 1, "start": 0, "end": 91.2},
                                        {"type": "stable", "index": 0, "start": 122.5, "end": 173.8},
                                        {"type": "stable", "index": 1, "start": 196.6, "end": 210.8},
                                        {"type": "stable", "index": 2, "start": 233.6, "end": 253.5},
                                        {"type": "stable", "index": 3, "start": 279.2, "end": 299.1},
                                        {"type": "stable", "index": 4, "start": 558.3, "end": 578.3},
                                        {"type": "stable", "index": 5, "start": 595.4, "end": 618.2},
                                        {"type": "stable", "index": 6, "start": 635.2, "end": 655.2},
                                        {"type": "stable", "index": 7, "start": 680.8, "end": 695.1},
                                        {"type": "stable", "index": 8, "start": 518.5, "end": 538.4},
                                        {"type": "stable", "index": 9, "start": 490, "end": 504.2},
                                        {"type": "stable", "index": 10, "start": 401.7, "end": 424.4},
                                        {"type": "stable", "index": 11, "start": 358.9, "end": 376},
                                        {"type": "stable", "index": 12, "start": 316.2, "end": 344.7}]},
    "20260426_190454_40s": {"filename": "Inference_continuous_experiment_20260426_190454_40s_4.csv",
                            "regions": [{"type": "stable", "index": 0, "start": 121.1, "end": 158.8},
                                        {"type": "stable", "index": 1, "start": 180.3, "end": 215.3},
                                        {"type": "stable", "index": 2, "start": 234.1, "end": 250.3},
                                        {"type": "stable", "index": 3, "start": 269.1, "end": 301.4},
                                        {"type": "stable", "index": 4, "start": 309.5, "end": 336.4},
                                        {"type": "stable", "index": 5, "start": 341.8, "end": 368.7},
                                        {"type": "stable", "index": 6, "start": 678.2, "end": 699.7},
                                        {"type": "stable", "index": 7, "start": 637.8, "end": 659.4},
                                        {"type": "stable", "index": 8, "start": 602.9, "end": 616.3},
                                        {"type": "stable", "index": 9, "start": 557.1, "end": 573.3},
                                        {"type": "stable", "index": 10, "start": 516.7, "end": 541},
                                        {"type": "stable", "index": 11, "start": 481.7, "end": 497.9},
                                        {"type": "stable", "index": 12, "start": 390.2, "end": 417.2},
                                        {"type": "unstructured", "index": 0, "start": 0, "end": 83.4},
                                        {"type": "unstructured", "index": 1, "start": 702.4, "end": 1480.2}]}
}

# Resolved at run time by resolve_paths()
P = {}


# ==============================================================================
# 2. GENERAL HELPERS
# ==============================================================================
def resolve_paths():
    out = output_path('gas_equilibrium_stern_volmer')
    P.clear()
    P.update({
        'v11_cache': output_path("gas_tvit", "4_Caches"),
        'v14_cache': external_path("gas_posthoc_cache"),
        'out': out,
        'eq': os.path.join(out, "0_Equilibrium_Selection"),
        'cal': os.path.join(out, "1_Calibration"),
        'cal_fold': os.path.join(out, "1_Calibration", "per_fold"),
        'inf': os.path.join(out, "2_Inference"),
        'chem': os.path.join(out, "3_Chemometrics"),
        'chem_exp': os.path.join(out, "3_Chemometrics", "per_experiment"),
        't90': os.path.join(out, "4_Response_T90"),
        't90_zoom': os.path.join(out, "4_Response_T90", "step_zooms"),
        'data': os.path.join(out, "5_Data"),
        'data_exp': os.path.join(out, "5_Data", "per_experiment"),
        'src': os.path.join(out, "csv_source"),
    })
    for key, path in P.items():
        if key not in ('v11_cache', 'v14_cache'):
            os.makedirs(path, exist_ok=True)


def banner(text):
    print("\n" + "=" * 100)
    print(f"  {text}")
    print("=" * 100)


def save_figure(fig, path_prefix, backing_df=None):
    """Every figure: PNG + PDF + backing CSV with the same name."""
    try:
        fig.tight_layout()
    except Exception:
        pass
    for ext in FIG_FORMATS:
        fig.savefig(f"{path_prefix}.{ext}", dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    if backing_df is not None:
        backing_df.to_csv(f"{path_prefix}.csv", index=False)


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


def model_display(key):
    """Display name and colour for a model column (handles the _rts suffix)."""
    is_rts = key.endswith('_rts')
    base = key[:-4] if is_rts else key
    info = MODEL_REGISTRY.get(base, {'name': base, 'color': '#333333', 'ls': '-'})
    return info['name'] + (' (RTS)' if is_rts else ''), info['color']


def model_style(key):
    base = key[:-4] if key.endswith('_rts') else key
    return MODEL_REGISTRY.get(base, {}).get('ls', '-')


def short_key_from_id(raw_id):
    """Map any folder / experiment id to its REGION_DEFINITIONS key via the YYYYMMDD_HHMMSS timestamp."""
    m = re.search(r'(\d{8}_\d{6})', str(raw_id))
    if not m:
        return None
    for key in REGION_DEFINITIONS:
        if key.startswith(m.group(1)):
            return key
    return None


def get_stable_intervals(exp_id):
    """All stable intervals, sorted by start (identical to the T90 script — used for T90)."""
    defn = REGION_DEFINITIONS.get(exp_id)
    if not defn or not defn.get('regions'):
        return []
    return sorted([(r['start'], r['end']) for r in defn['regions'] if r['type'] == 'stable'], key=lambda x: x[0])


def get_unstructured_intervals(exp_id):
    defn = REGION_DEFINITIONS.get(exp_id)
    if not defn:
        return []
    return sorted([(r['start'], r['end']) for r in defn['regions'] if r['type'] == 'unstructured'], key=lambda x: x[0])


def merge_intervals(intervals):
    """Merge overlapping / duplicated intervals (a time span can only hold one plateau)."""
    merged = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def regression_metrics(gt, pred):
    gt, pred = np.asarray(gt, float), np.asarray(pred, float)
    m = np.isfinite(gt) & np.isfinite(pred)
    n = int(m.sum())
    if n < 2:
        return {'n': n, 'MAE': np.nan, 'RMSE': np.nan, 'Bias': np.nan, 'R2': np.nan, 'MaxAE': np.nan, 'P95_AE': np.nan}
    e = pred[m] - gt[m]
    ae = np.abs(e)
    return {'n': n, 'MAE': float(ae.mean()), 'RMSE': float(np.sqrt(np.mean(e ** 2))), 'Bias': float(e.mean()),
            'R2': float(r2_score(gt[m], pred[m])), 'MaxAE': float(ae.max()), 'P95_AE': float(np.percentile(ae, 95))}


# ==============================================================================
# 3. DATA LOADING
# ==============================================================================
def _parse_reference_log(log_path):
    """Same parser as V11 (parse_arduino_logs_polars): Pyroscience MEA string -> R12 (%O2)."""
    records = []
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        next(f, None)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            ts_str, raw_str = parts[0].strip(), parts[1].strip()
            try:
                ts = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                continue
            last_mea_idx = raw_str.rfind("MEA")
            if last_mea_idx == -1:
                continue
            tokens = raw_str[last_mea_idx:].split()
            if len(tokens) >= 16 and tokens[0] == "MEA":
                try:
                    records.append({'timestamp': ts, 'oxygen': float(tokens[15]) / 1000.0})
                except ValueError:
                    continue
    if not records:
        return pd.DataFrame({'timestamp': pd.Series(dtype='datetime64[ns]'), 'oxygen': pd.Series(dtype=float)})
    df = pl.DataFrame(records).with_columns(pl.col("oxygen").fill_nan(None).interpolate())
    return df.to_pandas()


def extract_frames_from_raw():
    """Fallback when no V11 cache exists: same discovery and intensity definition as V11
    (mean red channel of the 48x48 resized frame + 1e-6), joined per experiment."""
    import cv2
    out = []
    for subdir in sorted(os.listdir(BASE_PATH)):
        full = os.path.join(BASE_PATH, subdir)
        if not os.path.isdir(full) or subdir.startswith(("Dynamic_IDC_Analysis", "PostHoc_")):
            continue
        m = re.search(r'(\d{8}_\d{6})', subdir)
        if not m or short_key_from_id(subdir) is None:
            continue
        ts_str = m.group(1)
        t0 = datetime.strptime(ts_str, '%Y%m%d_%H%M%S')
        mp4s = [f for f in glob.glob(os.path.join(full, "*.mp4")) if ts_str in os.path.basename(f)]
        csvs = [f for f in glob.glob(os.path.join(full, "*.csv")) if ts_str in os.path.basename(f)]
        if not (mp4s and csvs):
            print(f"  [-] Skipping {subdir}: missing .mp4/.csv matching '{ts_str}'.")
            continue
        print(f"  > Extracting intensities: {subdir} ...", end=' ', flush=True)
        cap = cv2.VideoCapture(mp4s[0])
        rows, f_idx = [], 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if f_idx % V11_STEP_RATE == 0:
                resized = cv2.resize(frame, STORAGE_DIM)
                inten = np.mean(resized[:, :, 2].astype(float)) + 1e-6
                rows.append((t0 + timedelta(seconds=f_idx / ASSUMED_FPS), f_idx / ASSUMED_FPS, 1.0 / inten))
            f_idx += 1
        cap.release()
        print(f"{len(rows):,} frames")
        vid = pd.DataFrame(rows, columns=['timestamp', 'elapsed', 'inv_i'])
        vid['experiment_id'] = subdir
        ref = _parse_reference_log(csvs[0])
        if vid.empty or ref.empty:
            print(f"  [-] Skipping {subdir}: no frames or no reference readings.")
            continue
        vid['timestamp'] = pd.to_datetime(vid['timestamp'])
        ref['timestamp'] = pd.to_datetime(ref['timestamp'])
        out.append(pd.merge_asof(vid.sort_values('timestamp'), ref.sort_values('timestamp'),
                                 on='timestamp', direction='nearest'))
    if not out:
        raise RuntimeError("No experiments could be extracted from the raw videos.")
    return pd.concat(out, ignore_index=True)


FRAME_ID_COLS = ('experiment_id', 'exp_id')
FRAME_O2_COLS = ('oxygen', 'oxygen_gt')
FRAME_I_COLS = ('inv_i', 'intensity')


def _own_frame_cache():
    return os.path.join(P['data'], f"_frame_cache_step{V11_STEP_RATE}.parquet")


def _parquet_columns(path):
    """Column names from the parquet footer only. Raises on empty / truncated / corrupt files."""
    if os.path.getsize(path) < 12:
        raise ValueError(f"file is {os.path.getsize(path)} bytes (empty or truncated)")
    try:
        import pyarrow.parquet as pq
        return list(pq.read_schema(path).names)
    except ImportError:
        return list(pl.scan_parquet(path).collect_schema().names())


def _frame_cache_candidates():
    """Ordered list of parquet files that might hold per-frame intensity + reference O2."""
    def step_of(p):
        m = re.search(r'_step(\d+)\.parquet$', p)
        return int(m.group(1)) if m else 10 ** 6

    cands = []
    if V11_CACHE_FILE:
        cands.append(V11_CACHE_FILE)
    cands.append(_own_frame_cache())                                   # written by a previous run of this script
    cands.append(os.path.join(P['v11_cache'], f"idc_master_v11_3_step{V11_STEP_RATE}.parquet"))
    cands += sorted(glob.glob(os.path.join(P['v11_cache'], "idc_master_*.parquet")),
                    key=lambda p: (step_of(p) != V11_STEP_RATE, step_of(p), -os.path.getsize(p)))
    cands += sorted(glob.glob(os.path.join(P['v11_cache'], "*.parquet")))
    cands += sorted(glob.glob(os.path.join(P['v14_cache'], "*.parquet")))
    seen, ordered = set(), []
    for c in cands:
        key = os.path.normcase(os.path.abspath(c))
        if key not in seen and os.path.exists(c):
            seen.add(key)
            ordered.append(c)
    return ordered


def load_v11_frames():
    """Use the first readable parquet that has per-frame intensity + reference O2. Broken or
    unsuitable files are skipped with a message. If none works, re-extract from the raw videos
    once and cache the result in 5_Data/ so later runs start immediately."""
    for path in _frame_cache_candidates():
        name = os.path.basename(path)
        try:
            names = _parquet_columns(path)
        except Exception as e:
            print(f"  [-] Skipping unreadable parquet {name}: {e}")
            continue
        id_col = next((c for c in FRAME_ID_COLS if c in names), None)
        o2_col = next((c for c in FRAME_O2_COLS if c in names), None)
        i_col = next((c for c in FRAME_I_COLS if c in names), None)
        if not (id_col and o2_col and i_col and 'elapsed' in names):
            print(f"  [-] Skipping {name}: no per-frame intensity + reference O2 columns.")
            continue
        cols = [id_col, 'elapsed', i_col, o2_col] + (['timestamp'] if 'timestamp' in names else [])
        try:
            df = pl.read_parquet(path, columns=cols).to_pandas()
        except Exception as e:
            print(f"  [-] Skipping {name}: could not be read ({type(e).__name__}: {e})")
            continue
        if df.empty:
            print(f"  [-] Skipping {name}: no rows.")
            continue
        df = df.rename(columns={id_col: 'experiment_id', o2_col: 'oxygen'})
        if i_col == 'intensity':
            df['inv_i'] = 1.0 / df['intensity'].astype(float)
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.NaT
        print(f"  [+] Frame source ({len(df):,} frames, pixel data not loaded):\n      {path}")
        return df[['experiment_id', 'timestamp', 'elapsed', 'inv_i', 'oxygen']], path

    print("  [!] No readable frame cache found — re-extracting optode intensities from the raw videos.\n"
          "      This is a one-off: the result is cached in 5_Data/ and reused on the next run.")
    df = extract_frames_from_raw()
    cache = _own_frame_cache()
    pl.from_pandas(df).write_parquet(cache)
    print(f"  [+] Frame cache written: {cache}")
    return df, f"raw videos (re-extracted) -> {cache}"


def build_master(raw):
    df = raw.rename(columns={'experiment_id': 'v11_experiment_id', 'oxygen': 'oxygen_gt'}).copy()
    df['v11_experiment_id'] = df['v11_experiment_id'].astype(str)
    df['exp_id'] = df['v11_experiment_id'].map({u: short_key_from_id(u) for u in df['v11_experiment_id'].unique()})
    for s in sorted(df.loc[df['exp_id'].isna(), 'v11_experiment_id'].unique()):
        print(f"  [-] No region definition for '{s}': excluded from this analysis.")
    df = df[df['exp_id'].notna()].copy()
    df['exp_id'] = df['exp_id'].astype(str)

    n_src = df.groupby('exp_id')['v11_experiment_id'].nunique()
    for key in n_src[n_src > 1].index:
        sizes = df[df['exp_id'] == key].groupby('v11_experiment_id').size()
        keep = sizes.idxmax()
        print(f"  [!] {key} matched {len(sizes)} source folders; keeping '{keep}' ({sizes.max()} frames).")
        df = df[(df['exp_id'] != key) | (df['v11_experiment_id'] == keep)]

    df['elapsed'] = df['elapsed'].astype(float)
    df['oxygen_gt'] = df['oxygen_gt'].astype(float)
    df['inv_i'] = df['inv_i'].astype(float)
    df['intensity'] = 1.0 / df['inv_i']
    df = df.sort_values(['exp_id', 'elapsed']).reset_index(drop=True)

    present = set(df['exp_id'])
    for key in REGION_DEFINITIONS:
        if key not in present:
            print(f"  [!] {key} is in REGION_DEFINITIONS but has no frames in the source data.")
    print(f"  [+] {len(present)} experiments, {len(df):,} frames.")
    return df


# ==============================================================================
# 4. REGIMES & EQUILIBRIUM POINT SELECTION
# ==============================================================================
def label_regimes(elapsed, exp_id):
    regions = REGION_DEFINITIONS[exp_id]['regions']
    stable = np.zeros(len(elapsed), bool)
    unstr = np.zeros(len(elapsed), bool)
    stable_id = np.full(len(elapsed), -1, dtype=int)
    for r in regions:
        m = (elapsed >= r['start']) & (elapsed <= r['end'])
        if r['type'] == 'stable':
            stable |= m
            stable_id[m & (stable_id < 0)] = r['index']
        elif r['type'] == 'unstructured':
            unstr |= m
    regime = np.full(len(elapsed), 'Transient', dtype=object)
    if REGIME_PRECEDENCE == 'unstructured_over_stable':
        regime[stable] = 'Stable'
        regime[unstr] = 'Unstructured'
    else:
        regime[unstr] = 'Unstructured'
        regime[stable] = 'Stable'
    stable_id[regime != 'Stable'] = -1
    return regime, stable_id


def add_regimes(master):
    regime = np.empty(len(master), dtype=object)
    stable_id = np.full(len(master), -1, dtype=int)
    for exp_id, idx in master.groupby('exp_id').indices.items():
        r, s = label_regimes(master['elapsed'].values[idx], exp_id)
        regime[idx] = r
        stable_id[idx] = s
    master['regime'] = regime.astype(str)
    master['stable_region_id'] = stable_id
    counts = master['regime'].value_counts()
    print("  [+] Regime frame counts: " + ", ".join(f"{k}={counts.get(k, 0):,}" for k in REGIMES))
    return master


def build_plateau_table(master):
    rows = []
    for exp_id, d in master.groupby('exp_id'):
        t, I, g = d['elapsed'].values, d['intensity'].values, d['oxygen_gt'].values
        is_stable = d['regime'].values == 'Stable'
        for order, (s, e) in enumerate(merge_intervals(get_stable_intervals(exp_id))):
            eq_start = s + EQ_TRIM_LEADING_FRAC * (e - s)
            in_win = (t >= s) & (t <= e) & is_stable
            keep = in_win & (t >= eq_start) & np.isfinite(I) & np.isfinite(g)
            row = {'exp_id': exp_id, 'plateau_order': order, 'start': s, 'end': e, 'eq_start': eq_start,
                   'n_frames_plateau': int(in_win.sum()), 'n_frames_eq': int(keep.sum()), 'eq_duration_s': np.nan,
                   'I_median': np.nan, 'I_mad': np.nan, 'gt_median': np.nan, 'gt_std': np.nan,
                   'rel_I_slope_per_s': np.nan, 'rel_I_drift': np.nan, 'gt_drift': np.nan,
                   'passed': False, 'reason': ''}
            if keep.sum() < EQ_MIN_FRAMES:
                row['reason'] = f"fewer than {EQ_MIN_FRAMES} frames after trimming"
                rows.append(row)
                continue
            tt, II, gg = t[keep], I[keep], g[keep]
            dur = float(tt.max() - tt.min())
            slope_I = np.polyfit(tt, II, 1)[0] if dur > 0 else 0.0
            slope_g = np.polyfit(tt, gg, 1)[0] if dur > 0 else 0.0
            I_med = float(np.median(II))
            rel_slope = float(abs(slope_I) / I_med) if I_med > 0 else np.inf
            row.update(eq_duration_s=dur, I_median=I_med, I_mad=float(np.median(np.abs(II - I_med))),
                       gt_median=float(np.median(gg)), gt_std=float(np.std(gg)),
                       rel_I_slope_per_s=rel_slope, rel_I_drift=rel_slope * dur,
                       gt_drift=float(abs(slope_g) * dur))
            reasons = []
            if dur < EQ_MIN_DURATION_S:
                reasons.append(f"kept window {dur:.1f} s < {EQ_MIN_DURATION_S} s")
            if rel_slope > EQ_MAX_REL_I_SLOPE:
                reasons.append(f"intensity still changing {rel_slope:.2%}/s > {EQ_MAX_REL_I_SLOPE:.2%}/s")
            if row['gt_drift'] > EQ_MAX_GT_DRIFT:
                reasons.append(f"reference drift {row['gt_drift']:.2f} > {EQ_MAX_GT_DRIFT} %O2")
            row['passed'] = not reasons
            row['reason'] = '; '.join(reasons) if reasons else 'ok'
            rows.append(row)
    tbl = pd.DataFrame(rows)
    n_pass = int(tbl['passed'].sum()) if len(tbl) else 0
    print(f"  [+] Equilibrium plateaus: {n_pass}/{len(tbl)} passed the trim + drift checks.")
    return tbl


def mark_eq_frames(master, ptab):
    flag = np.zeros(len(master), bool)
    t, e_ids, reg = master['elapsed'].values, master['exp_id'].values, master['regime'].values
    for _, r in ptab[ptab['passed']].iterrows():
        flag |= (e_ids == r['exp_id']) & (t >= r['eq_start']) & (t <= r['end']) & (reg == 'Stable')
    master['is_eq_frame'] = flag
    return master


# ==============================================================================
# 5. TWO-SITE STERN-VOLMER MODEL
# ==============================================================================
def two_site_sv(o2, I0, f, K1, K2):
    return I0 * ((f / (1.0 + K1 * o2)) + ((1.0 - f) / (1.0 + K2 * o2)))


def invert_two_site(I, I0, f, K1, K2):
    """Analytic inversion of the two-site model (same root as V11, written in a numerically
    stable form). r(1+K1x)(1+K2x) = 1 + x(K1 + f(K2-K1))  ->  A x^2 + B x + C = 0."""
    I = np.asarray(I, float)
    r = I / I0
    A = r * K1 * K2
    B = r * (K1 + K2) - (f * (K2 - K1) + K1)
    C = r - 1.0
    sq = np.sqrt(np.maximum(B * B - 4.0 * A * C, 0.0))
    with np.errstate(divide='ignore', invalid='ignore'):
        if K1 * K2 <= 1e-15:            # degenerate: one site unquenchable -> linear equation
            return -C / B
        x_neg_b = (-B + sq) / (2.0 * A)
        x_pos_b = (2.0 * C) / (-B - sq)
    return np.where(B < 0, x_neg_b, x_pos_b)


class SVModel:
    def __init__(self, method, params=None, se=None, poly=None):
        self.method = method
        self.params = params
        self.se = se if se is not None else (np.nan,) * 4
        self.poly = poly
        self.n_points = 0
        self.sse = np.nan
        self.r2_intensity = np.nan
        self.cal_mae_o2 = np.nan

    def intensity_at(self, o2):
        o2 = np.asarray(o2, float)
        if self.params is None:
            return np.full(o2.shape, np.nan)
        return two_site_sv(o2, *self.params)

    def predict_o2(self, intensity):
        I = np.asarray(intensity, float)
        if self.params is not None:
            return invert_two_site(I, *self.params)
        out = np.full(I.shape, np.nan)
        if self.poly is not None:
            m = np.isfinite(I) & (I > 0)
            if m.any():
                out[m] = self.poly.predict((1.0 / I[m]).reshape(-1, 1))
        return out

    def as_dict(self):
        p = self.params if self.params is not None else (np.nan,) * 4
        return {'fit_method': self.method, 'I0': p[0], 'f': p[1], 'K1': p[2], 'K2': p[3],
                'I0_se': self.se[0], 'f_se': self.se[1], 'K1_se': self.se[2], 'K2_se': self.se[3],
                'n_cal_points': self.n_points, 'sse_intensity': self.sse,
                'r2_intensity': self.r2_intensity, 'cal_MAE_o2': self.cal_mae_o2}


def _safe_se(pcov):
    d = np.diag(pcov)
    return np.sqrt(np.clip(d, 0, None)) if np.all(np.isfinite(d)) else np.full(len(d), np.nan)


def fit_sv(o2, intensity):
    """Multi-start two-site SV fit (V11's start included). Fallbacks: one-site SV, then V11's polynomial."""
    o2, I = np.asarray(o2, float), np.asarray(intensity, float)
    m = np.isfinite(o2) & np.isfinite(I) & (I > 0)
    o2, I = o2[m], I[m]
    if len(o2) < 5:
        print(f"    ! Only {len(o2)} calibration points — no model fitted for this fold.")
        model = SVModel('insufficient_points')
        model.n_points = len(o2)
        return model

    best = None
    for fac in SV_I0_START_FACTORS:
        for (f0, k10, k20) in SV_START_SHAPES:
            try:
                popt, pcov = curve_fit(two_site_sv, o2, I, p0=[fac * np.max(I), f0, k10, k20],
                                       bounds=SV_BOUNDS, maxfev=SV_MAXFEV)
            except (RuntimeError, ValueError):
                continue
            sse = float(np.sum((two_site_sv(o2, *popt) - I) ** 2))
            if best is None or sse < best[2]:
                best = (popt, pcov, sse)

    if best is not None:
        popt, pcov, _ = best
        I0, f, K1, K2 = popt
        sI0, sf, sK1, sK2 = _safe_se(pcov)
        if K2 > K1:   # canonical order: site 1 is the more strongly quenched site
            K1, K2, sK1, sK2, f = K2, K1, sK2, sK1, 1.0 - f
        model = SVModel('two_site', (I0, f, K1, K2), (sI0, sf, sK1, sK2))
    else:
        try:
            popt, pcov = curve_fit(lambda x, I0, K: I0 / (1.0 + K * x), o2, I, p0=[np.max(I), 0.03],
                                   bounds=([0, 0], [np.inf, np.inf]), maxfev=SV_MAXFEV)
            se = _safe_se(pcov)
            model = SVModel('one_site_fallback', (popt[0], 1.0, popt[1], popt[1]), (se[0], 0.0, se[1], se[1]))
            print("    ! Two-site fit failed — one-site SV fallback used.")
        except (RuntimeError, ValueError):
            poly = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit((1.0 / I).reshape(-1, 1), o2)
            model = SVModel('poly2_fallback', poly=poly)
            print("    ! SV fits failed — V11 polynomial fallback used.")

    model.n_points = len(o2)
    fit_I = model.intensity_at(o2)
    if np.all(np.isfinite(fit_I)):
        model.sse = float(np.sum((fit_I - I) ** 2))
        model.r2_intensity = float(r2_score(I, fit_I))
    pred = model.predict_o2(I)
    ok = np.isfinite(pred)
    model.cal_mae_o2 = float(np.mean(np.abs(pred[ok] - o2[ok]))) if ok.any() else np.nan
    return model


# ==============================================================================
# 6. LEAVE-ONE-EXPERIMENT-OUT CALIBRATION + FULL INFERENCE
# ==============================================================================
def run_loocv(master, ptab):
    exps = sorted(master['exp_id'].unique())
    pred_eq = np.full(len(master), np.nan)
    pred_all = np.full(len(master), np.nan)
    i_expected = np.full(len(master), np.nan)
    param_rows = []
    exp_vals = master['exp_id'].values

    for k, test in enumerate(exps, 1):
        print(f"\n  LOOCV fold {k}/{len(exps)} — held out: {test}")
        tr_mask = exp_vals != test
        te_idx = np.where(~tr_mask)[0]
        train = master.loc[tr_mask, ['exp_id', 'elapsed', 'oxygen_gt', 'intensity', 'is_eq_frame']]

        eq_train = ptab[(ptab['exp_id'] != test) & ptab['passed']]
        if EQ_FIT_MODE == 'pointwise':
            fr = train[train['is_eq_frame']]
            x_eq, y_eq = fr['oxygen_gt'].values, fr['intensity'].values
        else:
            x_eq, y_eq = eq_train['gt_median'].values, eq_train['I_median'].values

        m_eq = fit_sv(x_eq, y_eq)
        m_all = fit_sv(train['oxygen_gt'].values, train['intensity'].values)

        I_te = master['intensity'].values[te_idx]
        pred_eq[te_idx] = m_eq.predict_o2(I_te)
        pred_all[te_idx] = m_all.predict_o2(I_te)
        i_expected[te_idx] = m_eq.intensity_at(master['oxygen_gt'].values[te_idx])

        held = ptab[(ptab['exp_id'] == test) & ptab['passed']]
        for key, label, m in (('o2_sv_eq', 'Equilibrium plateaus', m_eq),
                              ('o2_sv_all', 'All regions (legacy V11)', m_all)):
            row = {'fold': k, 'test_exp': test, 'model_key': key, 'calibration_set': label,
                   'n_train_experiments': len(exps) - 1, **m.as_dict()}
            if len(held):
                hp = m.predict_o2(held['I_median'].values)
                ok = np.isfinite(hp)
                row['heldout_plateau_MAE'] = float(np.mean(np.abs(hp[ok] - held['gt_median'].values[ok]))) if ok.any() else np.nan
                row['heldout_plateau_n'] = int(ok.sum())
            else:
                row['heldout_plateau_MAE'], row['heldout_plateau_n'] = np.nan, 0
            param_rows.append(row)
            print(f"    {label:<26}: {m.method:<17} n={m.n_points:<7} I0={row['I0']:.2f} f={row['f']:.3f} "
                  f"K1={row['K1']:.4f} K2={row['K2']:.5f} | held-out plateau MAE={row['heldout_plateau_MAE']:.3f} %")

        plot_calibration_fold(k, test, train, ptab, m_eq, m_all)

    master['o2_sv_eq'] = pred_eq
    master['o2_sv_all'] = pred_all
    master['I_eq_expected'] = i_expected
    return master, pd.DataFrame(param_rows)


# ==============================================================================
# 7. V14.1-R MERGE, RTS SMOOTHING, METRICS
# ==============================================================================
def merge_v14_predictions(master):
    files = glob.glob(os.path.join(P['v14_cache'], "kalman_df_v141R_*.parquet"))
    if not files:
        print(f"  [-] No V14.1-R cache in {P['v14_cache']} — classical models only.")
        return master, [], None
    path = max(files, key=os.path.getmtime)
    try:
        import pyarrow.parquet as pq
        names = pq.ParquetFile(path).schema.names
    except Exception:
        names = None
    wanted = (['exp_id', 'elapsed'] + V14_AI_KEYS + [f"{k}_kalman" for k in V14_AI_KEYS]
              + ['o2_classical', 'o2_classical_kalman', 'regime'])
    cols = [c for c in wanted if names is None or c in names]
    v14 = pd.read_parquet(path, columns=cols if names is not None else None)
    if 'exp_id' not in v14.columns or 'elapsed' not in v14.columns:
        print("  [!] V14 cache lacks exp_id/elapsed columns — skipped.")
        return master, [], path
    v14 = v14[[c for c in wanted if c in v14.columns]].copy()
    v14['exp_id'] = v14['exp_id'].astype(str)
    v14['exp_id'] = v14['exp_id'].map({u: short_key_from_id(u) for u in v14['exp_id'].unique()})
    v14 = v14.dropna(subset=['exp_id'])
    v14['exp_id'] = v14['exp_id'].astype(str)
    v14['elapsed'] = v14['elapsed'].astype(float)
    # V14's own smoothed AI columns become '<model>_rts' here, so AI RTS numbers are exactly V14's
    v14 = v14.rename(columns={'o2_classical': 'o2_classical_v14', 'o2_classical_kalman': 'o2_classical_v14_kalman',
                              'regime': 'regime_v14', **{f"{k}_kalman": f"{k}_rts" for k in V14_AI_KEYS}})
    ai_cols = [k for k in V14_AI_KEYS if k in v14.columns]

    left = master.sort_values('elapsed', kind='mergesort')
    right = v14.sort_values('elapsed', kind='mergesort')
    merged = pd.merge_asof(left, right, on='elapsed', by='exp_id', direction='nearest',
                           tolerance=0.6 / ASSUMED_FPS)
    merged = merged.sort_values(['exp_id', 'elapsed'], kind='mergesort').reset_index(drop=True)
    print(f"  [+] Merged V14.1-R predictions from {os.path.basename(path)}")
    for c in ai_cols:
        has_rts = f"{c}_rts" in merged.columns
        print(f"      {model_display(c)[0]:<22} matched on {merged[c].notna().mean():.1%} of frames"
              f"{' (+ V14 RTS Kalman column)' if has_rts else ''}")
    return merged, ai_cols, path


def kalman_rts_v14(predictions, timestamps):
    """V14.1-R `_kalman_smooth_array_rts` without gradient inputs (as V14 ran it for the classical
    model): constant-velocity state [conc, rate], same R, Q, initialisation, dt handling and RTS
    backward pass. Written with explicit 2x2 algebra for speed; numerically identical."""
    z = np.asarray(predictions, float)
    t = np.asarray(timestamps, float)
    N = len(z)
    if N < 3:
        return z.copy()
    valid = np.isfinite(z)
    if not valid.any():
        return z.copy()
    R = float(KALMAN_MEASUREMENT_NOISE)
    qc = float(KALMAN_PROCESS_NOISE_CONC)
    sv = float(KALMAN_PROCESS_NOISE_RATE) * float(KALMAN_STABLE_TIGHTENING)
    c, v = float(z[valid][0]), 0.0
    p00, p01, p11 = R, 0.0, float(KALMAN_PROCESS_NOISE_RATE)
    dts = np.empty(N)
    xpr = np.empty((N, 2))
    ppr = np.empty((N, 3))
    xpo = np.empty((N, 2))
    ppo = np.empty((N, 3))
    prev_t = t[0]
    for i in range(N):
        dt = max(t[i] - prev_t, 0.05) if i > 0 else 0.5
        prev_t = t[i]
        dts[i] = dt
        cp, vp = c + dt * v, v
        a00 = p00 + 2.0 * dt * p01 + dt * dt * p11 + qc * dt
        a01 = p01 + dt * p11
        a11 = p11 + sv * dt
        xpr[i] = (cp, vp)
        ppr[i] = (a00, a01, a11)
        if valid[i]:
            S = a00 + R
            k0, k1 = a00 / S, a01 / S
            inn = z[i] - cp
            c, v = cp + k0 * inn, vp + k1 * inn
            p00, p01, p11 = (1.0 - k0) * a00, (1.0 - k0) * a01, a11 - k1 * a01
        else:
            c, v, p00, p01, p11 = cp, vp, a00, a01, a11
        xpo[i] = (c, v)
        ppo[i] = (p00, p01, p11)
    xs = xpo.copy()
    for k in range(N - 2, -1, -1):
        dt = dts[k + 1]
        q00, q01, q11 = ppo[k]
        m00, m01 = q00 + dt * q01, q01            # P_post[k] @ F[k+1]^T
        m10, m11 = q01 + dt * q11, q11
        b00, b01, b11 = ppr[k + 1]
        b00 += 1e-10
        b11 += 1e-10
        det = b00 * b11 - b01 * b01
        if det == 0.0:
            continue
        i00, i01, i11 = b11 / det, -b01 / det, b00 / det
        g00, g01 = m00 * i00 + m01 * i01, m00 * i01 + m01 * i11
        g10, g11 = m10 * i00 + m11 * i01, m10 * i01 + m11 * i11
        d0, d1 = xs[k + 1, 0] - xpr[k + 1, 0], xs[k + 1, 1] - xpr[k + 1, 1]
        xs[k, 0] = xpo[k, 0] + g00 * d0 + g01 * d1
        xs[k, 1] = xpo[k, 1] + g10 * d0 + g11 * d1
    return xs[:, 0]


def apply_rts(master, model_cols):
    """Adds '<col>_rts' for each column with the V14.1-R smoother, per experiment, in time order."""
    groups = master.groupby('exp_id').indices
    t_all = master['elapsed'].values
    for col in model_cols:
        out = np.full(len(master), np.nan)
        vals = master[col].values
        for _, idx in groups.items():
            idx = idx[np.argsort(t_all[idx], kind='mergesort')]
            out[idx] = kalman_rts_v14(vals[idx], t_all[idx])
        master[f"{col}_rts"] = out
    if model_cols:
        print(f"  [+] V14.1-R RTS Kalman smoother applied to: {', '.join(model_display(c)[0] for c in model_cols)}")
    return master


def compute_performance(master, model_cols):
    regimes = ['All'] + REGIMES
    rows = []
    for exp_id, d in master.groupby('exp_id'):
        for col in model_cols:
            for reg in regimes:
                sub = d if reg == 'All' else d[d['regime'] == reg]
                rows.append({'exp_id': exp_id, 'model_key': col, 'model': model_display(col)[0], 'regime': reg,
                             **regression_metrics(sub['oxygen_gt'].values, sub[col].values)})
    per_exp = pd.DataFrame(rows)
    grows = []
    for col in model_cols:
        for reg in regimes:
            sub = master if reg == 'All' else master[master['regime'] == reg]
            pe = per_exp[(per_exp['model_key'] == col) & (per_exp['regime'] == reg)]['MAE']
            grows.append({'model_key': col, 'model': model_display(col)[0], 'regime': reg,
                          **regression_metrics(sub['oxygen_gt'].values, sub[col].values),
                          'MAE_mean_over_exps': pe.mean(), 'MAE_std_over_exps': pe.std(), 'n_exps': int(pe.notna().sum())})
    return per_exp, pd.DataFrame(grows)


def headline_comparison(global_perf):
    rows = []
    for reg in ['All'] + REGIMES:
        g = global_perf[global_perf['regime'] == reg].set_index('model_key')
        if 'o2_sv_eq' not in g.index or 'o2_sv_all' not in g.index:
            continue
        rows.append({'regime': reg,
                     'MAE_EqCal': g.loc['o2_sv_eq', 'MAE'], 'MAE_AllCal': g.loc['o2_sv_all', 'MAE'],
                     'Delta_MAE (Eq-All)': g.loc['o2_sv_eq', 'MAE'] - g.loc['o2_sv_all', 'MAE'],
                     'Pct_change': 100 * (g.loc['o2_sv_eq', 'MAE'] / g.loc['o2_sv_all', 'MAE'] - 1),
                     'RMSE_EqCal': g.loc['o2_sv_eq', 'RMSE'], 'RMSE_AllCal': g.loc['o2_sv_all', 'RMSE'],
                     'Bias_EqCal': g.loc['o2_sv_eq', 'Bias'], 'Bias_AllCal': g.loc['o2_sv_all', 'Bias'],
                     'R2_EqCal': g.loc['o2_sv_eq', 'R2'], 'R2_AllCal': g.loc['o2_sv_all', 'R2']})
    df = pd.DataFrame(rows)
    print(f"\n  {'REGIME':<13} | {'MAE Eq-Cal':>10} | {'MAE All-Cal':>11} | {'Δ':>8} | {'% change':>8}")
    print("  " + "-" * 62)
    for _, r in df.iterrows():
        print(f"  {r['regime']:<13} | {r['MAE_EqCal']:>10.3f} | {r['MAE_AllCal']:>11.3f} | "
              f"{r['Delta_MAE (Eq-All)']:>+8.3f} | {r['Pct_change']:>+7.1f}%")
    return df


def reproduction_check(master):
    """Sanity checks against V14.1-R: legacy SV reproduction and regime-label agreement."""
    rows, out = [], {}
    if 'o2_classical_v14' in master.columns:
        for exp_id, d in master.groupby('exp_id'):
            a, b = d['o2_sv_all'].values, d['o2_classical_v14'].values
            m = np.isfinite(a) & np.isfinite(b)
            rows.append({'exp_id': exp_id, 'n': int(m.sum()),
                         'MAE(sv_all vs V14 o2_classical)': float(np.mean(np.abs(a[m] - b[m]))) if m.any() else np.nan,
                         'corr': float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else np.nan})
        rep = pd.DataFrame(rows)
        rep.to_csv(os.path.join(P['data'], "Check_Legacy_SV_vs_V14_o2_classical.csv"), index=False)
        out['legacy_vs_v14_MAE'] = float(rep['MAE(sv_all vs V14 o2_classical)'].mean())
        print(f"  [i] Recomputed legacy SV vs V14 'o2_classical': mean |Δ| = {out['legacy_vs_v14_MAE']:.4f} %O2 "
              f"(≈0 means the legacy baseline is reproduced exactly)")
    if 'o2_classical_v14' in master.columns and 'o2_classical_v14_kalman' in master.columns:
        diffs = []
        for _, d in master.groupby('exp_id'):
            d = d.sort_values('elapsed')
            mine = kalman_rts_v14(d['o2_classical_v14'].values, d['elapsed'].values)
            ref = d['o2_classical_v14_kalman'].values
            m = np.isfinite(mine) & np.isfinite(ref)
            if m.any():
                diffs.append(np.abs(mine[m] - ref[m]))
        if diffs:
            out['rts_smoother_vs_v14_max_abs_diff'] = float(np.max(np.concatenate(diffs)))
            print(f"  [i] RTS smoother check on V14 'o2_classical': max |Δ| vs V14's own Kalman column = "
                  f"{out['rts_smoother_vs_v14_max_abs_diff']:.2e} %O2 (≈0 means identical smoothing)")
    if 'regime_v14' in master.columns:
        m = master['regime_v14'].notna()
        if m.any():
            agree = float((master.loc[m, 'regime'] == master.loc[m, 'regime_v14'].astype(str)).mean())
            ct = pd.crosstab(master.loc[m, 'regime'], master.loc[m, 'regime_v14'].astype(str))
            ct.to_csv(os.path.join(P['data'], "Check_Regime_Labels_vs_V14.csv"))
            out['regime_agreement_with_v14'] = agree
            print(f"  [i] Regime labels agree with V14.1-R on {agree:.2%} of frames "
                  f"(crosstab in 5_Data/Check_Regime_Labels_vs_V14.csv)")
    return out


# ==============================================================================
# 8. PLOTTING — EQUILIBRIUM SELECTION & CALIBRATION
# ==============================================================================
def shade_regions(ax, exp_id):
    for s, e in merge_intervals(get_stable_intervals(exp_id)):
        ax.axvspan(s, e, color=REGIME_COLORS['Stable'], alpha=0.10, lw=0)
    for s, e in get_unstructured_intervals(exp_id):
        ax.axvspan(s, e, color=REGIME_COLORS['Unstructured'], alpha=0.12, lw=0)


def regime_patches():
    return [Patch(facecolor=REGIME_COLORS['Stable'], alpha=0.25, label='Stable region'),
            Patch(facecolor=REGIME_COLORS['Unstructured'], alpha=0.30, label='Unstructured region')]


def plot_equilibrium_selection(d, exp_id, ptab_exp):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    t = d['elapsed']
    for ax in (ax1, ax2):
        shade_regions(ax, exp_id)
    ax1.plot(t, d['intensity'], color='#444444', lw=0.8)
    ax2.plot(t, d['oxygen_gt'], color='black', lw=1.5)
    for _, r in ptab_exp.iterrows():
        c = '#2ca02c' if r['passed'] else '#d62728'
        for ax in (ax1, ax2):
            ax.axvspan(r['eq_start'], r['end'], color=c, alpha=0.30, lw=0)
        if r['passed']:
            ax1.plot([r['eq_start'], r['end']], [r['I_median']] * 2, color='darkgreen', lw=3)
            ax2.plot([r['eq_start'], r['end']], [r['gt_median']] * 2, color='darkgreen', lw=3)
    n_pass = int(ptab_exp['passed'].sum())
    ax1.set_ylabel("Optode intensity [a.u.]")
    ax2.set_ylabel(f"Reference {UNIT}")
    ax2.set_xlabel("Elapsed Time [s]")
    ax1.set_title(f"Equilibrium window selection: {exp_id} — {n_pass}/{len(ptab_exp)} plateaus used "
                  f"(leading {EQ_TRIM_LEADING_FRAC:.0%} trimmed)")
    handles = regime_patches() + [Patch(facecolor='#2ca02c', alpha=0.45, label='Equilibrium window used'),
                                  Patch(facecolor='#d62728', alpha=0.45, label='Window rejected'),
                                  Line2D([], [], color='darkgreen', lw=3, label='Plateau median')]
    ax1.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    backing = d[['elapsed', 'timestamp', 'intensity', 'oxygen_gt', 'regime', 'stable_region_id', 'is_eq_frame']]
    save_figure(fig, os.path.join(P['eq'], f"EqSelection_{exp_id}"), backing)


def plot_calibration_fold(fold, test_exp, train, ptab, m_eq, m_all):
    tr = train[['oxygen_gt', 'intensity']].dropna()
    if len(tr) > MAX_CAL_SCATTER:
        tr = tr.sample(MAX_CAL_SCATTER, random_state=SEED)
    eq_tr = ptab[(ptab['exp_id'] != test_exp) & ptab['passed']]
    eq_rej = ptab[(ptab['exp_id'] != test_exp) & ~ptab['passed'] & ptab['I_median'].notna()]
    eq_te = ptab[(ptab['exp_id'] == test_exp) & ptab['passed']]
    o2_max = float(np.nanmax(train['oxygen_gt'].values)) if len(train) else 100.0
    grid = np.linspace(0, max(o2_max, 1.0) * 1.02, 400)
    I0_ref = m_eq.params[0] if m_eq.params is not None else float(np.nanmax(eq_tr['I_median'])) if len(eq_tr) else 1.0
    c_eq, c_all = MODEL_REGISTRY['o2_sv_eq']['color'], MODEL_REGISTRY['o2_sv_all']['color']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    for ax, transform, ylab in ((ax1, lambda I: I, "Optode intensity [a.u.]"),
                                (ax2, lambda I: I0_ref / I, "I$_0$ / I  (I$_0$ from equilibrium fit)")):
        ax.scatter(tr['oxygen_gt'], transform(tr['intensity']), s=3, c='#9e9e9e', alpha=0.15, rasterized=True,
                   label='All training frames (legacy calibration set)')
        ax.scatter(eq_tr['gt_median'], transform(eq_tr['I_median']), s=55, c=c_eq, edgecolor='k', zorder=5,
                   label='Equilibrium points (training plateaus)')
        if len(eq_rej):
            ax.scatter(eq_rej['gt_median'], transform(eq_rej['I_median']), s=45, marker='s', facecolor='none',
                       edgecolor='#d62728', zorder=4, label='Rejected plateaus')
        if len(eq_te):
            ax.scatter(eq_te['gt_median'], transform(eq_te['I_median']), s=110, marker='X', c='k', zorder=6,
                       label=f'Held-out plateaus ({test_exp})')
        ax.plot(grid, transform(m_eq.intensity_at(grid)), color=c_eq, lw=3, label='Two-site SV — equilibrium cal.')
        ax.plot(grid, transform(m_all.intensity_at(grid)), color=c_all, lw=2.5, ls='--',
                label='Two-site SV — all-region cal. (legacy)')
        ax.set_xlabel(f"Reference {UNIT}")
        ax.set_ylabel(ylab)
    ax1.set_title(f"Fold {fold}: calibration curves (held out: {test_exp})")
    ax2.set_title("Stern-Volmer representation")
    ax2.legend(loc='upper left', fontsize=9)

    parts = [pd.DataFrame({'series': f'train_all_frames_subsample_{len(tr)}', 'oxygen': tr['oxygen_gt'].values,
                           'intensity': tr['intensity'].values}),
             pd.DataFrame({'series': 'train_equilibrium_points', 'oxygen': eq_tr['gt_median'].values,
                           'intensity': eq_tr['I_median'].values}),
             pd.DataFrame({'series': 'rejected_plateaus', 'oxygen': eq_rej['gt_median'].values,
                           'intensity': eq_rej['I_median'].values}),
             pd.DataFrame({'series': 'heldout_equilibrium_points', 'oxygen': eq_te['gt_median'].values,
                           'intensity': eq_te['I_median'].values}),
             pd.DataFrame({'series': 'curve_equilibrium_cal', 'oxygen': grid, 'intensity': m_eq.intensity_at(grid)}),
             pd.DataFrame({'series': 'curve_all_region_cal', 'oxygen': grid, 'intensity': m_all.intensity_at(grid)})]
    backing = pd.concat(parts, ignore_index=True)
    backing['I0_over_I'] = I0_ref / backing['intensity']
    save_figure(fig, os.path.join(P['cal_fold'], f"Calibration_Fold{fold:02d}_{test_exp}"), backing)


def plot_parameter_summary(params):
    panels = [('I0', 'I$_0$ [a.u.]'), ('f', 'f (fraction of site 1)'), ('K1', 'K$_{SV,1}$ [1/%O$_2$]'),
              ('K2', 'K$_{SV,2}$ [1/%O$_2$]'), ('cal_MAE_o2', 'Calibration-set MAE [%O$_2$]'),
              ('heldout_plateau_MAE', 'Held-out plateau MAE [%O$_2$]')]
    exps = params.sort_values('fold')['test_exp'].unique()
    xpos = {e: i for i, e in enumerate(exps)}
    fig, axes = plt.subplots(2, 3, figsize=(21, 11))
    for ax, (col, lab) in zip(axes.ravel(), panels):
        for j, (key, sub) in enumerate(params.groupby('model_key', sort=False)):
            name, color = model_display(key)
            x = np.array([xpos[e] for e in sub['test_exp']]) + (j - 0.5) * 0.25
            yerr = sub[f'{col}_se'].values if f'{col}_se' in sub.columns else None
            ax.errorbar(x, sub[col].values, yerr=yerr, fmt='o', color=color, mec='k', capsize=3, label=name)
        ax.set_xticks(range(len(exps)))
        ax.set_xticklabels([e.split('_', 2)[-1] for e in exps], rotation=60)
        ax.set_ylabel(lab)
        ax.set_xlabel("Held-out experiment")
    axes[0, 0].legend()
    fig.suptitle("Two-site Stern-Volmer parameters across LOOCV folds", fontsize=15)
    save_figure(fig, os.path.join(P['cal'], "Calibration_Parameters_Across_Folds"), params)


# ==============================================================================
# 9. PLOTTING — INFERENCE & CHEMOMETRICS
# ==============================================================================
def plot_inference_timeline(d, exp_id, model_cols):
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(18, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1.3]})
    t, gt = d['elapsed'], d['oxygen_gt']
    for a in (ax, axr):
        shade_regions(a, exp_id)
    ax.plot(t, gt, color='black', lw=2.6, label='Ground Truth (reference)', zorder=10)
    for col in model_cols:
        name, color = model_display(col)
        lw = 2.0 if col in CLASSICAL_KEYS else 1.4
        ax.plot(t, d[col], color=color, lw=lw, ls=model_style(col), alpha=0.9, label=name)
        axr.plot(t, d[col] - gt, color=color, lw=1.0, ls=model_style(col), alpha=0.85)
    axr.axhline(0, color='black', lw=1)
    ax.set_ylabel(UNIT)
    axr.set_ylabel("Error (pred − GT)")
    axr.set_xlabel("Elapsed Time [s]")
    ax.set_title(f"Full inference: {exp_id}")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + regime_patches(), loc='upper left', bbox_to_anchor=(1.01, 1.0))
    backing = d[['elapsed', 'timestamp', 'oxygen_gt', 'regime'] + model_cols].copy()
    for col in model_cols:
        backing[f'err_{col}'] = d[col] - gt
    save_figure(fig, os.path.join(P['inf'], f"Inference_{exp_id}"), backing)


def _parity_panel(ax, gt, pred, regimes, title):
    m = np.isfinite(gt) & np.isfinite(pred)
    for reg in REGIMES:
        rm = m & (regimes == reg)
        if rm.any():
            ax.scatter(gt[rm], pred[rm], s=7, alpha=0.35, c=REGIME_COLORS[reg], label=reg, rasterized=True)
    if m.any():
        lo = min(gt[m].min(), pred[m].min())
        hi = max(gt[m].max(), pred[m].max())
        pad = 0.05 * (hi - lo + 1e-9)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k--', lw=1.5)
        met = regression_metrics(gt, pred)
        title += f"\n$R^2$={met['R2']:.3f} | MAE={met['MAE']:.2f} %"
    ax.set_title(title)
    ax.set_xlabel(f"Ground Truth {UNIT}")
    ax.set_ylabel(f"Predicted {UNIT}")


def plot_chemometrics_exp(d, exp_id):
    gt, regimes = d['oxygen_gt'].values, d['regime'].values
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.8))
    for ax, col in zip(axes[:2], CLASSICAL_KEYS):
        _parity_panel(ax, gt, d[col].values, regimes, model_display(col)[0])
    axes[0].legend(markerscale=3, loc='upper left')
    for col in CLASSICAL_KEYS:
        name, color = model_display(col)
        res = (d[col] - d['oxygen_gt']).dropna()
        if len(res) > 1:
            try:
                sns.histplot(res, kde=True, stat='density', element='step', color=color, alpha=0.3,
                             ax=axes[2], label=f"{name} (MAE {res.abs().mean():.2f})")
            except Exception:
                axes[2].hist(res, bins=60, density=True, alpha=0.4, color=color, label=name)
    axes[2].axvline(0, color='k', lw=1)
    axes[2].set_xlabel("Error (pred − GT) [%O$_2$]")
    axes[2].set_title("Residual distributions")
    axes[2].legend()
    backing = pd.DataFrame({'elapsed': d['elapsed'], 'regime': d['regime'], 'True': d['oxygen_gt'],
                            'Pred_EqCal': d['o2_sv_eq'], 'Pred_AllCal': d['o2_sv_all'],
                            'Residual_EqCal': d['o2_sv_eq'] - d['oxygen_gt'],
                            'Residual_AllCal': d['o2_sv_all'] - d['oxygen_gt']})
    save_figure(fig, os.path.join(P['chem_exp'], f"Chemometrics_{exp_id}"), backing)


def plot_global_parity_grid(master, model_cols):
    regs = ['All'] + REGIMES
    fig, axes = plt.subplots(len(model_cols), len(regs), figsize=(5.2 * len(regs), 5.0 * len(model_cols)), squeeze=False)
    gt_all = master['oxygen_gt'].values
    lo, hi = np.nanmin(gt_all), np.nanmax(gt_all)
    pad = 0.08 * (hi - lo)
    lims = [lo - pad, hi + pad]
    for i, col in enumerate(model_cols):
        name, color = model_display(col)
        for j, reg in enumerate(regs):
            ax = axes[i, j]
            sub = master if reg == 'All' else master[master['regime'] == reg]
            gt, pred = sub['oxygen_gt'].values, sub[col].values
            ax.scatter(gt, pred, s=5, alpha=0.25, c=color, rasterized=True)
            ax.plot(lims, lims, 'k--', lw=1.5)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect('equal')
            met = regression_metrics(gt, pred)
            ax.set_title(f"{name} — {reg}\n$R^2$={met['R2']:.3f} | MAE={met['MAE']:.2f} % | n={met['n']:,}", fontsize=11)
            ax.set_xlabel(f"Ground Truth {UNIT}")
            ax.set_ylabel(f"Predicted {UNIT}")
    backing = pd.concat([pd.DataFrame({'model_key': col, 'model': model_display(col)[0], 'exp_id': master['exp_id'],
                                       'elapsed': master['elapsed'], 'regime': master['regime'],
                                       'gt': master['oxygen_gt'], 'pred': master[col]}) for col in model_cols],
                        ignore_index=True)
    save_figure(fig, os.path.join(P['chem'], "Global_Parity_Grid_AllModels_by_Regime"), backing)


def plot_regime_mae(global_perf, per_exp, model_cols):
    names = [model_display(k)[0] for k in model_cols]
    palette = {model_display(k)[0]: model_display(k)[1] for k in model_cols}
    gp = global_perf[global_perf['model_key'].isin(model_cols)]
    fig, ax = plt.subplots(figsize=(14, 6.5))
    sns.barplot(data=gp, x='regime', y='MAE', hue='model', order=['All'] + REGIMES, hue_order=names,
                palette=palette, edgecolor='black', ax=ax)
    for c in ax.containers:
        ax.bar_label(c, fmt='%.2f', fontsize=8, padding=2)
    ax.set_ylabel("Pooled MAE [%O$_2$]")
    ax.set_xlabel("")
    ax.set_title("Pooled MAE by regime (all held-out frames)")
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc='upper left')
    save_figure(fig, os.path.join(P['chem'], "Regime_MAE_Bars"), gp)

    pe = per_exp[per_exp['model_key'].isin(model_cols)]
    fig, ax = plt.subplots(figsize=(15, 6.5))
    sns.boxplot(data=pe, x='regime', y='MAE', hue='model', order=['All'] + REGIMES, hue_order=names,
                palette=palette, showfliers=False, ax=ax)
    sns.stripplot(data=pe, x='regime', y='MAE', hue='model', order=['All'] + REGIMES, hue_order=names,
                  palette=palette, dodge=True, edgecolor='black', linewidth=0.6, size=5, ax=ax, legend=False)
    ax.set_ylabel("Per-experiment MAE [%O$_2$]")
    ax.set_xlabel("")
    ax.set_title("Per-experiment MAE distribution by regime (one point per held-out experiment)")
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc='upper left')
    save_figure(fig, os.path.join(P['chem'], "PerExperiment_MAE_by_Regime"), pe)


# ==============================================================================
# 10. T90 RESPONSE-TIME ANALYSIS (same two methods as the T90 script)
# ==============================================================================
def extract_t90_savgol(master, model_columns):
    """Original Savgol-based T90 extraction (signed lag relative to GT)."""
    recs = []
    for exp_id, df in master.groupby('exp_id'):
        intervals = get_stable_intervals(exp_id)
        if len(intervals) < 2:
            continue
        for i in range(1, len(intervals)):
            prev_s, prev_e = intervals[i - 1]
            target_s, target_e = intervals[i]
            step_start_time = prev_e
            sub = df[(df['elapsed'] >= prev_s) & (df['elapsed'] <= target_e)]
            if len(sub) < 10:
                continue
            plat = df[(df['elapsed'] >= target_s) & (df['elapsed'] <= target_e)]

            def plat_med(col):
                return plat[col].median() if col in plat.columns and not plat[col].isna().all() else np.nan

            gt_target = plat_med('oxygen_gt')
            start_val = sub['oxygen_gt'].iloc[0]
            delta = gt_target - start_val
            if not np.isfinite(delta) or abs(delta) < T90_MIN_STEP:
                continue
            direction = 1 if delta > 0 else -1

            def t90_duration(col):
                if col not in sub.columns or sub[col].isna().all():
                    return np.nan
                target_plat = plat_med(col)
                if np.isnan(target_plat):
                    return np.nan
                thresh = start_val + 0.9 * (target_plat - start_val)
                try:
                    win = min(11, len(sub) - (1 if len(sub) % 2 == 0 else 0))
                    if win < 5:
                        win = 5
                    if win % 2 == 0:
                        win -= 1
                    sig = savgol_filter(sub[col].values, window_length=win, polyorder=min(2, win - 1))
                    mask = (sig >= thresh) if direction > 0 else (sig <= thresh)
                    if mask.any():
                        return float(sub['elapsed'].iloc[np.argmax(mask)]) - step_start_time
                except Exception:
                    pass
                return np.nan

            gt_dur = t90_duration('oxygen_gt')
            if np.isnan(gt_dur):
                continue
            for col in model_columns:
                mod_dur = t90_duration(col)
                is_rts = col.endswith('_rts')
                recs.append({'exp_id': exp_id, 'step_idx': i, 'target_conc': gt_target, 'step_magnitude': delta,
                             'model_key': col, 'base_key': col[:-4] if is_rts else col, 'is_rts': is_rts,
                             'gt_t90_duration': gt_dur, 'model_t90_duration': mod_dur,
                             'lag_s': mod_dur - gt_dur if not np.isnan(mod_dur) else np.nan})
    return pd.DataFrame(recs)


def extract_t90_raw_aligned(master, model_columns):
    """Raw-signal T90 (matches the Bulk Comparative script): no Savgol, GT-derived v90 threshold,
    window prev_e -> curr_e, absolute lag |t90_model - t90_gt|, T90_Abs = t90_true + lag."""
    recs = []
    for exp_id, df_exp in master.groupby('exp_id'):
        intervals = get_stable_intervals(exp_id)
        if len(intervals) < 2:
            continue
        for i in range(1, len(intervals)):
            prev_s, prev_e = intervals[i - 1]
            curr_s, curr_e = intervals[i]
            base_df = df_exp[(df_exp['elapsed'] >= prev_s) & (df_exp['elapsed'] <= prev_e)].dropna(subset=['oxygen_gt'])
            curr_df = df_exp[(df_exp['elapsed'] >= curr_s) & (df_exp['elapsed'] <= curr_e)].dropna(subset=['oxygen_gt'])
            if len(base_df) < 3 or len(curr_df) < 3:
                continue
            base_o2, target_o2 = base_df['oxygen_gt'].mean(), curr_df['oxygen_gt'].mean()
            delta = target_o2 - base_o2
            if abs(delta) < T90_MIN_STEP:
                continue
            sign = np.sign(delta)
            t_win = df_exp[(df_exp['elapsed'] >= prev_e) & (df_exp['elapsed'] <= curr_e)].dropna(subset=['oxygen_gt'])
            if t_win.empty:
                continue
            v5, v90 = base_o2 + 0.05 * delta, base_o2 + 0.90 * delta
            gt_start = (t_win['oxygen_gt'] - v5) * sign >= 0
            t_start = t_win.loc[gt_start, 'elapsed'].min() if gt_start.any() else np.nan
            gt_90 = (t_win['oxygen_gt'] - v90) * sign >= 0
            t90_gt = t_win.loc[gt_90, 'elapsed'].min() if gt_90.any() else np.nan
            t90_true = (t90_gt - t_start) if pd.notna(t90_gt) and pd.notna(t_start) else np.nan
            for col in model_columns:
                if col not in t_win.columns:
                    continue
                tw = t_win.dropna(subset=[col])
                if tw.empty:
                    continue
                m90 = (tw[col] - v90) * sign >= 0
                t90_m = tw.loc[m90, 'elapsed'].min() if m90.any() else np.nan
                lag = abs(t90_m - t90_gt) if pd.notna(t90_m) and pd.notna(t90_gt) else np.nan
                is_rts = col.endswith('_rts')
                recs.append({'exp_id': exp_id, 'step_idx': i, 'target_conc': target_o2, 'base_conc': base_o2,
                             'step_magnitude': delta, 'model_key': col, 'base_key': col[:-4] if is_rts else col,
                             'is_rts': is_rts, 'gt_t90_true': t90_true, 't90_gt_time': t90_gt, 't90_model_time': t90_m,
                             'T90_Lag': lag, 'T90_Abs': (t90_true + lag) if pd.notna(t90_true) and pd.notna(lag) else np.nan})
    return pd.DataFrame(recs)


def t90_summary_savgol(t90_df, global_perf):
    print(f"\n  {'MODEL NAME':<24} | {'RAW LAG':>9} | {'RTS LAG':>9} | {'RAW STD':>8} | {'RTS STD':>8} | "
          f"{'RAW R2':>6} | {'STABLE':>7} | {'TRANS':>7} | {'UNSTR':>7} | {'DIFF':>6}")
    print("  " + "-" * 125)
    rows = []
    for b_key in t90_df['base_key'].unique():
        name = model_display(b_key)[0]
        raw = t90_df[(t90_df['base_key'] == b_key) & (~t90_df['is_rts'])]
        rts = t90_df[(t90_df['base_key'] == b_key) & (t90_df['is_rts'])]
        valid = raw.dropna(subset=['gt_t90_duration', 'model_t90_duration'])
        raw_r2 = r2_score(valid['gt_t90_duration'], valid['model_t90_duration']) if len(valid) > 1 else np.nan
        gp = global_perf[global_perf['model_key'] == b_key].set_index('regime')['MAE']
        st, tr, un = gp.get('Stable', np.nan), gp.get('Transient', np.nan), gp.get('Unstructured', np.nan)
        row = {'Model Name': name, 'Raw Mean Lag (s)': raw['lag_s'].mean(), 'RTS Mean Lag (s)': rts['lag_s'].mean(),
               'Raw Std Lag (s)': raw['lag_s'].std(), 'RTS Std Lag (s)': rts['lag_s'].std(), 'Raw R2': raw_r2,
               'Stable MAE (%)': st, 'Transient MAE (%)': tr, 'Unstructured MAE (%)': un,
               'MAE Diff (Trans-Stable)': tr - st}
        rows.append(row)
        print(f"  {name:<24} | {row['Raw Mean Lag (s)']:>+7.2f} s | {row['RTS Mean Lag (s)']:>+7.2f} s | "
              f"{row['Raw Std Lag (s)']:>6.2f} s | {row['RTS Std Lag (s)']:>6.2f} s | {raw_r2:>6.2f} | "
              f"{st:>7.2f} | {tr:>7.2f} | {un:>7.2f} | {tr - st:>6.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(P['t90'], "T90_Savgol_Raw_vs_RTS_Summary.csv"), index=False)
    return df


def t90_summary_raw_aligned(t90_raw):
    print(f"\n  {'MODEL NAME':<28} | {'T90 Lag Mean':>13} | {'T90 Lag Std':>12} | {'T90 Abs Mean':>13} | {'T90 Abs Std':>12} |    n")
    print("  " + "-" * 105)
    rows, per_exp = [], []
    for b_key in t90_raw['base_key'].unique():
        for is_rts in (False, True):
            sub = t90_raw[(t90_raw['base_key'] == b_key) & (t90_raw['is_rts'] == is_rts)]
            if sub.empty:
                continue
            name = model_display(b_key + ('_rts' if is_rts else ''))[0]
            lag, ab = sub['T90_Lag'].dropna(), sub['T90_Abs'].dropna()
            rows.append({'Model': name, 'T90 Lag Mean (s)': lag.mean(), 'T90 Lag Std (s)': lag.std(),
                         'T90 Lag n': int(len(lag)), 'T90 Abs Mean (s)': ab.mean(), 'T90 Abs Std (s)': ab.std(),
                         'T90 Abs n': int(len(ab))})
            print(f"  {name:<28} | {lag.mean():>10.3f} s  | {lag.std():>9.3f} s  | {ab.mean():>10.3f} s  | "
                  f"{ab.std():>9.3f} s  | {len(lag):>4}")
            grp = sub.groupby('exp_id').agg(lag_mean=('T90_Lag', 'mean'), lag_std=('T90_Lag', 'std'),
                                            abs_mean=('T90_Abs', 'mean'), abs_std=('T90_Abs', 'std'),
                                            n=('T90_Lag', 'count')).reset_index()
            grp['Model'] = name
            per_exp.append(grp)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(P['t90'], "T90_GrandAverage_RawAligned_AllModels.csv"), index=False)
    if per_exp:
        pd.concat(per_exp, ignore_index=True).to_csv(os.path.join(P['t90'], "T90_PerExperiment_RawAligned.csv"), index=False)
    return summary


def plot_t90_parity(t90_df, suffix, rts):
    df = t90_df[t90_df['is_rts'] == rts]
    models = list(df['model_key'].unique())
    if not models:
        return
    n_cols = min(4, len(models))
    n_rows = int(np.ceil(len(models) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 5.2 * n_rows), squeeze=False)
    axes = axes.ravel()
    for ax, key in zip(axes, models):
        name, color = model_display(key)
        sub = df[df['model_key'] == key].dropna(subset=['gt_t90_duration', 'model_t90_duration'])
        if len(sub) < 2:
            ax.set_visible(False)
            continue
        x, y = sub['gt_t90_duration'], sub['model_t90_duration']
        ax.scatter(x, y, color=color, alpha=0.6, edgecolor='k', s=40)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.75, zorder=0)
        ax.set_title(f"{name}\n$R^2$: {r2_score(x, y):.2f} | MAE: {np.mean(np.abs(x - y)):.1f}s")
        ax.set_xlabel("GT $T_{90}$ Duration (s)")
        ax.set_ylabel("Model $T_{90}$ Duration (s)")
    for ax in axes[len(models):]:
        ax.set_visible(False)
    fig.suptitle(f"{'RTS' if rts else 'RAW'} parity of response times (time to 90% of step)", fontsize=15)
    save_figure(fig, os.path.join(P['t90'], f"T90_Parity_{suffix}"), df)


def plot_t90_binned(t90_df, suffix, rts):
    df = t90_df[t90_df['is_rts'] == rts].dropna(subset=['lag_s']).copy()
    if df.empty:
        return
    edges = np.linspace(t90_df['target_conc'].min(), t90_df['target_conc'].max(), 6)
    # include_lowest=True so the lowest-concentration step is not dropped from the first bin
    df['conc_bin'] = pd.cut(df['target_conc'], bins=edges, include_lowest=True)
    df = df.dropna(subset=['conc_bin'])
    df['Bin Label'] = df['conc_bin'].map(lambda x: f"{x.left:.1f} - {x.right:.1f}\n(%)").astype(str)
    df['Model Name'] = df['model_key'].map(lambda k: model_display(k)[0])
    order = [f"{iv.left:.1f} - {iv.right:.1f}\n(%)" for iv in df['conc_bin'].cat.categories]
    order = [o for o in order if o in set(df['Bin Label'])]
    keys = list(df['model_key'].unique())
    palette = {model_display(k)[0]: model_display(k)[1] for k in keys}
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.barplot(data=df, x='Bin Label', y='lag_s', hue='Model Name', order=order,
                hue_order=[model_display(k)[0] for k in keys], palette=palette,
                errorbar='ci', capsize=0.05, seed=SEED, ax=ax)
    ax.axhline(0, color='black', lw=1.5, ls='--')
    ax.set_title(f"{'RTS' if rts else 'RAW'} — $T_{{90}}$ lag across target O$_2$ concentrations")
    ax.set_xlabel("Target Oxygen Concentration Range (%)")
    ax.set_ylabel("Mean $T_{90}$ Lag (s)\nNegative = faster than GT | Positive = slower than GT")
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc='upper left')
    binned = df.groupby(['Bin Label', 'Model Name'], observed=True)['lag_s'].agg(['mean', 'std', 'count']).reset_index()
    save_figure(fig, os.path.join(P['t90'], f"T90_Binned_Lag_{suffix}"), binned)


def plot_step_zooms(master, exp_id, model_cols):
    d = master[master['exp_id'] == exp_id]
    intervals = get_stable_intervals(exp_id)
    steps = []
    for i in range(1, len(intervals)):
        (ps, pe), (cs, ce) = intervals[i - 1], intervals[i]
        base = d[(d['elapsed'] >= ps) & (d['elapsed'] <= pe)]['oxygen_gt'].dropna()
        tgt = d[(d['elapsed'] >= cs) & (d['elapsed'] <= ce)]['oxygen_gt'].dropna()
        if len(base) < 3 or len(tgt) < 3:
            continue
        delta = tgt.mean() - base.mean()
        if abs(delta) >= T90_MIN_STEP:
            steps.append((i, pe, ce, base.mean(), tgt.mean(), delta))
    steps = steps[:N_ZOOM_STEPS]
    if not steps:
        return
    n_cols = min(3, len(steps))
    n_rows = int(np.ceil(len(steps) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)
    axes = axes.ravel()
    backing = []
    for ax, (i, pe, ce, b, tg, delta) in zip(axes, steps):
        w = d[(d['elapsed'] >= pe - ZOOM_PRE_S) & (d['elapsed'] <= ce)]
        tw = d[(d['elapsed'] >= pe) & (d['elapsed'] <= ce)]
        v90 = b + 0.9 * delta
        ax.plot(w['elapsed'], w['oxygen_gt'], color='black', lw=2.5, label='Ground Truth')
        for col in ['oxygen_gt'] + model_cols:
            if col != 'oxygen_gt':
                name, color = model_display(col)
                ax.plot(w['elapsed'], w[col], color=color, lw=1.8, ls=model_style(col), label=name)
            else:
                color = 'black'
            hit = (tw[col] - v90) * np.sign(delta) >= 0
            if hit.any():
                ax.plot(tw.loc[hit, 'elapsed'].min(), v90, 'o', color=color, mec='k', ms=8, zorder=10)
            backing.append(pd.DataFrame({'step_idx': i, 'series': col, 'elapsed': w['elapsed'].values,
                                         'value': w[col].values, 'v90_threshold': v90}))
        ax.axhline(v90, color='k', ls=':', lw=1)
        ax.axvline(pe, color='grey', ls='--', lw=1)
        ax.set_title(f"Step {i}: {b:.1f} → {tg:.1f} %")
        ax.set_xlabel("Elapsed Time [s]")
        ax.set_ylabel(UNIT)
    for ax in axes[len(steps):]:
        ax.set_visible(False)
    axes[0].legend(fontsize=9)
    fig.suptitle(f"Step responses with T90 crossings (dots on the 90% line): {exp_id}", fontsize=14)
    save_figure(fig, os.path.join(P['t90_zoom'], f"StepZoom_{exp_id}"), pd.concat(backing, ignore_index=True))


# ==============================================================================
# 11. EXPORTS
# ==============================================================================
def export_full_inference(master, model_cols_all):
    base_cols = ['exp_id', 'v11_experiment_id', 'timestamp', 'elapsed', 'regime', 'stable_region_id', 'is_eq_frame',
                 'intensity', 'inv_i', 'I_eq_expected', 'oxygen_gt']
    extra = [c for c in ('o2_classical_v14', 'regime_v14') if c in master.columns]
    out = master[base_cols + model_cols_all + extra].copy()
    for col in model_cols_all:
        out[f'err_{col}'] = out[col] - out['oxygen_gt']
    if 'regime_v14' in out.columns:
        out['regime_v14'] = out['regime_v14'].astype(str)
    pq_path = os.path.join(P['data'], "Full_Inference_AllExperiments.parquet")
    pl.from_pandas(out).write_parquet(pq_path)
    out.to_csv(os.path.join(P['data'], "Full_Inference_AllExperiments.csv"), index=False)
    for exp_id, d in out.groupby('exp_id'):
        d.to_csv(os.path.join(P['data_exp'], f"Inference_{exp_id}.csv"), index=False)
    print(f"  [+] Full inference: {len(out):,} frames -> 5_Data/Full_Inference_AllExperiments.(parquet|csv) "
          f"+ per-experiment CSVs")


def export_csv_source(master, model_cols):
    alias = {'o2_sv_eq': 'o2_classical', 'o2_sv_all': 'o2_classical_allcal',
             'o2_sv_eq_rts': 'o2_classical_rts', 'o2_sv_all_rts': 'o2_classical_allcal_rts'}
    rts_cols = [f"{c}_rts" for c in model_cols if f"{c}_rts" in master.columns]
    for exp_id, d in master.groupby('exp_id'):
        d[['elapsed', 'timestamp', 'oxygen_gt', 'regime'] + model_cols + rts_cols].rename(columns=alias).to_csv(
            os.path.join(P['src'], f"Inference_{exp_id}.csv"), index=False)
    for reg in ['All'] + REGIMES:
        sub = master if reg == 'All' else master[master['regime'] == reg]
        frames = [pd.DataFrame({'model': MODEL_REGISTRY[c]['parity_label'], 'exp_id': sub['exp_id'],
                                'elapsed': sub['elapsed'], 'regime': sub['regime'],
                                'gt': sub['oxygen_gt'], 'pred': sub[c]}) for c in model_cols]
        pd.concat(frames, ignore_index=True).to_csv(os.path.join(P['src'], f"Parity_{reg}.csv"), index=False)
    print("  [+] csv_source/ written (Inference_<exp>.csv, Parity_<regime>.csv) for the publication plotter.")


def export_excel(sheets):
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("  [-] openpyxl not installed — Excel summary skipped (all tables are also CSV).")
        return
    path = os.path.join(P['data'], "Summary_Tables.xlsx")
    with pd.ExcelWriter(path, engine='openpyxl') as xw:
        for name, df in sheets.items():
            if df is not None and len(df):
                df.to_excel(xw, sheet_name=name[:31], index=False)
    print("  [+] Excel summary workbook: 5_Data/Summary_Tables.xlsx")


def write_manifest(info):
    import scipy
    import sklearn
    cfg_keys = ['BASE_PATH', 'V11_STEP_RATE', 'ASSUMED_FPS', 'STORAGE_DIM', 'EQ_TRIM_LEADING_FRAC', 'EQ_MIN_FRAMES',
                'EQ_MIN_DURATION_S', 'EQ_MAX_REL_I_SLOPE', 'EQ_MAX_GT_DRIFT', 'EQ_FIT_MODE', 'REGIME_PRECEDENCE', 'SV_BOUNDS',
                'SV_START_SHAPES', 'SV_I0_START_FACTORS', 'SV_MAXFEV', 'APPLY_RTS', 'KALMAN_MEASUREMENT_NOISE', 'KALMAN_PROCESS_NOISE_CONC',
                'KALMAN_PROCESS_NOISE_RATE', 'KALMAN_STABLE_TIGHTENING', 'T90_MIN_STEP', 'SEED']
    manifest = {'script': os.path.basename(__file__) if '__file__' in globals() else 'ClassicalSV_EquilibriumCal_V15_0',
                'version': '15.0-EQ', 'run_at': datetime.now().isoformat(timespec='seconds'),
                'config': {k: globals()[k] for k in cfg_keys}, **info,
                'library_versions': {'python': platform.python_version(), 'numpy': np.__version__,
                                     'pandas': pd.__version__, 'polars': pl.__version__, 'scipy': scipy.__version__,
                                     'scikit-learn': sklearn.__version__, 'matplotlib': matplotlib.__version__,
                                     'seaborn': sns.__version__}}
    with open(os.path.join(P['out'], "run_manifest.json"), 'w') as f:
        json.dump(json_safe(manifest), f, indent=2)


# ==============================================================================
# 12. MAIN
# ==============================================================================
def main():
    resolve_paths()
    banner("CLASSICAL STERN-VOLMER — EQUILIBRIUM CALIBRATION (V15.0-EQ)")
    print(f"  Output: {P['out']}")

    banner("1/7  Loading frames")
    raw, frames_source = load_v11_frames()
    master = build_master(raw)

    banner("2/7  Regimes and equilibrium points")
    master = add_regimes(master)
    ptab = build_plateau_table(master)
    master = mark_eq_frames(master, ptab)
    ptab.to_csv(os.path.join(P['eq'], "Equilibrium_Plateau_Table.csv"), index=False)
    for exp_id, d in master.groupby('exp_id'):
        plot_equilibrium_selection(d, exp_id, ptab[ptab['exp_id'] == exp_id])

    banner(f"3/7  LOOCV calibration ({EQ_FIT_MODE}) + full inference")
    master, params = run_loocv(master, ptab)
    params.to_csv(os.path.join(P['data'], "Calibration_Parameters_PerFold.csv"), index=False)
    with open(os.path.join(P['data'], "Calibration_Parameters_PerFold.json"), 'w') as f:
        json.dump(json_safe(params.to_dict(orient='records')), f, indent=2)
    plot_parameter_summary(params)

    banner("4/7  Merging V14.1-R predictions + RTS smoothing")
    master, ai_cols, v14_path = merge_v14_predictions(master)
    model_cols = CLASSICAL_KEYS + ai_cols
    rts_cols = []
    if APPLY_RTS:
        # classical columns always; AI columns only if V14's own Kalman columns were not in the cache
        to_smooth = [c for c in model_cols if c in CLASSICAL_KEYS or f"{c}_rts" not in master.columns]
        master = apply_rts(master, to_smooth)
        rts_cols = [f"{c}_rts" for c in model_cols]
    else:
        master = master.drop(columns=[c for c in master.columns if c.endswith('_rts')])
    checks = reproduction_check(master)

    banner("5/7  Performance metrics + inference / chemometric plots")
    per_exp, global_perf = compute_performance(master, model_cols + rts_cols)
    per_exp.to_csv(os.path.join(P['data'], "Performance_PerExperiment_by_Regime.csv"), index=False)
    global_perf.to_csv(os.path.join(P['data'], "Performance_Global_by_Regime.csv"), index=False)
    headline = headline_comparison(global_perf)
    headline.to_csv(os.path.join(P['data'], "Headline_EqCal_vs_AllCal.csv"), index=False)
    for exp_id, d in master.groupby('exp_id'):
        plot_inference_timeline(d, exp_id, model_cols)
        plot_chemometrics_exp(d, exp_id)
    plot_global_parity_grid(master, model_cols)
    plot_regime_mae(global_perf, per_exp, model_cols)

    banner("6/7  T90 response-time analysis")
    t90_cols = model_cols + rts_cols
    t90_sg = extract_t90_savgol(master, t90_cols)
    t90_sg.to_csv(os.path.join(P['t90'], "Extracted_T90_Savgol_Metrics.csv"), index=False)
    t90_raw = extract_t90_raw_aligned(master, t90_cols)
    t90_raw.to_csv(os.path.join(P['t90'], "Extracted_T90_RawAligned_Metrics.csv"), index=False)
    sg_summary = t90_summary_savgol(t90_sg, global_perf) if len(t90_sg) else pd.DataFrame()
    ra_summary = t90_summary_raw_aligned(t90_raw) if len(t90_raw) else pd.DataFrame()
    if len(t90_sg):
        plot_t90_parity(t90_sg, "Raw", rts=False)
        plot_t90_binned(t90_sg, "Raw", rts=False)
        if rts_cols:
            plot_t90_parity(t90_sg, "RTS", rts=True)
            plot_t90_binned(t90_sg, "RTS", rts=True)
    for exp_id in master['exp_id'].unique():
        plot_step_zooms(master, exp_id, model_cols)

    banner("7/7  Exporting data")
    export_full_inference(master, model_cols + rts_cols)
    export_csv_source(master, model_cols)
    export_excel({'Headline_Eq_vs_All': headline, 'Global_by_Regime': global_perf,
                  'PerExp_by_Regime': per_exp, 'Calibration_Params': params, 'Plateaus': ptab,
                  'T90_Savgol_Summary': sg_summary, 'T90_RawAligned_Summary': ra_summary})
    write_manifest({'inputs': {'frames': frames_source, 'v14_cache': v14_path},
                    'experiments': sorted(master['exp_id'].unique()), 'n_frames': len(master),
                    'plateaus': {'total': len(ptab), 'passed': int(ptab['passed'].sum())},
                    'models': model_cols + rts_cols, 'checks': checks})

    banner("DONE")
    print(f"  All outputs in: {P['out']}")


if __name__ == "__main__":
    main()
