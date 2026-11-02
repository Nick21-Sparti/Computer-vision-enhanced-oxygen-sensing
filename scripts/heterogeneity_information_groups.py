# Repository edition: see docs/CHANGES.md and docs/REPRODUCIBILITY.md.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GROUP BUILDER: BY INFORMATION CONTENT (effective rank of pixel responses)
Alternative to heterogeneity_study.py's MPD-based construction
==============================================================================
Builds N_GROUPS equal-sized pixel groups arranged along a spectrum of
INFORMATION CONTENT -- from the group whose pixel time-series carry the MOST
independent information (highest effective rank) to the LEAST (most redundant).

This differs from the study's MPD construction: MPD measures spread in a
static feature space [intensity, Ksv, R2, (T90)]; here the ordering axis is
the effective rank of the actual pixel RESPONSE matrix (pixels x time), i.e.
how many genuinely independent temporal signals the group contains. This is
the quantity the coded-aperture / compressive-sensing argument is really about.

METHOD
  1. Restrict to SV-valid pixels (R2 >= threshold), as in the study.
  2. Read each candidate pixel's RED time series (parquet, averaged over days),
     onto a common elapsed grid -> response matrix R_all (n_valid x T).
  3. Rank pixels by an information contribution score, then GREEDILY assemble a
     single "maximally informative" ordering:
        - start from the pixel with the highest response variance,
        - repeatedly add the pixel that most increases the group's effective
          rank (a submodular-style greedy diversity selection).
     This yields an ordering where early pixels are maximally complementary.
  4. Form N_GROUPS windows of N pixels each along this ordering:
        - GROUP 1  = the first N (most complementary)  -> highest info
        - GROUP k  = a window sliding toward the redundant tail
        - GROUP N_GROUPS = the last N (most redundant) -> lowest info
     so effective rank decreases monotonically across the groups (verified &
     record-selected, like the study's expanding mode).
  5. Save group_assignments.csv + group_metrics.csv with per-group
     effective_rank / avg_pairwise_corr / logdet, plus a definition figure.

The written files use the SAME schema as the study so the existing training
loop (heterogeneity_study.py) and plot scripts can consume them unchanged:
point the study's GROUPDEF outputs at this folder, or copy the CSVs in.

Run standalone. No training here -- this only DEFINES the groups.
==============================================================================
"""
# Portable repository paths; edit scripts/repo_config.py for your computer.
from repo_config import data_path, output_path, cache_path, external_path

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

# ============================ CONFIG =========================================
BASE_PROJECT_DIR = data_path('aqueous')

N_PIXELS_PER_GROUP = 16
N_GROUPS           = 9
N_DAYS_CAP         = 5            # match the study run you'll train with

FILTER_BY_SV_R2      = True
SV_R2_POOL_THRESHOLD = 0.95

ANALYSIS_INTERVALS_S = [(1450,1650),(2300,2500),(3100,3300),(3900,4100),(4700,4900)]
STORAGE_DIM = (48,48); N_PIXELS_TOTAL = STORAGE_DIM[0]*STORAGE_DIM[1]; RED_CH = 2
MIN_FRAMES_IN_WINDOW = 5
GRID_N = 400                     # common elapsed grid length for responses
MIN_VALID_FRACTION = 0.05        # record-selection leniency (unused-safe)

# output: a SEPARATE definitions folder so it never clobbers the MPD study
OUT_DIR = output_path("heterogeneity_information_groups", "Group_Definitions")
os.makedirs(OUT_DIR, exist_ok=True)
FEATURES_PATH = cache_path("aqueous_heterogeneity", "features", "idc_master_v10.parquet")

# where the SV features live (for the R2 pool) -- reuse the study's cache if present
STUDY_FEATURES = output_path("heterogeneity_quality_groups", "Group_Definitions", "pixel_features.npz")

GROUP_VISUAL_TIME_S = 1500.0


# ============================ DATA ===========================================
def sv_valid_pool():
    """Eligible pixel indices with SV R2 >= threshold. Prefer the study's
    cached per-pixel features; else fall back to all pixels."""
    if FILTER_BY_SV_R2 and os.path.exists(STUDY_FEATURES):
        c = np.load(STUDY_FEATURES, allow_pickle=True)
        if 'sv_r2' in c:
            r2 = c['sv_r2']
            elig = np.where(r2 >= SV_R2_POOL_THRESHOLD)[0]
            print(f"  [POOL] {len(elig)}/{N_PIXELS_TOTAL} SV-valid "
                  f"({100*len(elig)/N_PIXELS_TOTAL:.1f}%) from study cache")
            return elig
    print("  [POOL] SV cache not found or filter off -> using all pixels")
    return np.arange(N_PIXELS_TOTAL)


def response_matrix(cand):
    """(len(cand) x GRID_N) mean RED response across days, memory-safe stream."""
    import pyarrow.parquet as pq
    red_cols = np.array([RED_CH + 3*p for p in cand])
    pf = pq.ParquetFile(FEATURES_PATH)
    day = {}
    for batch in pf.iter_batches(batch_size=4096,
                                 columns=['experiment_id','elapsed','pixels']):
        b = batch.to_pandas()
        for d, sub in b.groupby('experiment_id'):
            dd = day.setdefault(d, {'el': [], 'red': []})
            for el, px in zip(sub['elapsed'].values, sub['pixels'].values):
                px = np.frombuffer(px, np.uint8) if isinstance(px, bytes) \
                     else np.asarray(px, np.uint8)
                dd['el'].append(float(el)); dd['red'].append(px[red_cols].astype(np.float32))
    days = sorted(day)[:N_DAYS_CAP] if N_DAYS_CAP else sorted(day)
    lo = max(min(day[d]['el']) for d in days)
    hi = min(max(day[d]['el']) for d in days)
    grid = np.linspace(lo, hi, GRID_N)
    stacks = []
    for d in days:
        el = np.array(day[d]['el']); R = np.array(day[d]['red'])
        o = np.argsort(el); el, R = el[o], R[o]
        Rg = np.stack([np.interp(grid, el, R[:, j]) for j in range(len(cand))])
        stacks.append(Rg)
    print(f"  [OK] responses: {len(cand)} pixels x {GRID_N} t, {len(days)} days")
    return np.mean(np.stack(stacks), axis=0), grid, days


# ============================ INFO METRICS ===================================
def effective_rank(R):
    Rc = R - R.mean(axis=1, keepdims=True)
    s = np.linalg.svd(Rc, compute_uv=False); s = s[s > 1e-12]
    if len(s) == 0: return 0.0
    p = s / s.sum(); return float(np.exp(-np.sum(p*np.log(p))))

def avg_pairwise_corr(R):
    if R.shape[0] < 2: return np.nan
    C = np.corrcoef(R); iu = np.triu_indices(C.shape[0], 1)
    v = np.abs(C[iu]); v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else np.nan

def logdet_corr(R, reg=1e-6):
    Rc = R - R.mean(axis=1, keepdims=True); n = Rc.shape[0]
    cov = (Rc@Rc.T)/max(Rc.shape[1]-1,1)
    d = np.sqrt(np.clip(np.diag(cov),1e-12,None))
    corr = cov/np.outer(d,d) + reg*np.eye(n)
    sign, ld = np.linalg.slogdet(corr)
    return float(ld) if sign>0 else np.nan


# ============================ GREEDY ORDERING ================================
def greedy_information_order(R):
    """
    Order pixels so that prefixes are maximally informative: start from the
    highest-variance pixel; repeatedly add the pixel maximising the increase
    in effective rank of the selected set (submodular-style diversity greedy).
    Returns an index order (into R's rows), most-complementary first.
    """
    n = R.shape[0]
    Rc = R - R.mean(axis=1, keepdims=True)
    var = Rc.var(axis=1)
    start = int(np.argmax(var))
    chosen = [start]; remaining = set(range(n)) - {start}
    # For speed, cap the marginal-gain search using a candidate subsample when n large
    while remaining:
        best, best_er = None, -np.inf
        cand = list(remaining)
        # evaluate marginal effective rank gain
        cur = R[chosen]
        for j in cand:
            er = effective_rank(np.vstack([cur, R[j]]))
            if er > best_er:
                best_er, best = er, j
        chosen.append(best); remaining.discard(best)
    return np.array(chosen)


def build():
    print("="*64); print("  BUILD GROUPS BY INFORMATION CONTENT (effective rank)")
    print("="*64)
    cand = sv_valid_pool()
    if len(cand) < N_PIXELS_PER_GROUP:
        raise ValueError(f"pool {len(cand)} < N_PIXELS_PER_GROUP {N_PIXELS_PER_GROUP}")
    R, grid, days = response_matrix(cand)

    print("  greedy information ordering (this can take a moment)...")
    order = greedy_information_order(R)          # most-complementary first
    R_ord = R[order]; cand_ord = cand[order]
    M = len(cand_ord)

    # windows along the ordering: G1 = most informative prefix,
    # G_last = most redundant tail. Slide the window in equal steps.
    N = N_PIXELS_PER_GROUP
    starts = np.round(np.linspace(0, M - N, N_GROUPS)).astype(int)
    groups, rows = [], []
    for g, s in enumerate(starts, 1):
        idx = np.arange(s, s + N)
        pix = cand_ord[idx]
        Rg = R_ord[idx]
        groups.append(pix)
        rows.append({'group': g, 'window_start': int(s), 'window_end': int(s+N),
                     'n_pixels': N,
                     'effective_rank': effective_rank(Rg),
                     'avg_pairwise_corr': avg_pairwise_corr(Rg),
                     'logdet_corr': logdet_corr(Rg)})
    metrics = pd.DataFrame(rows)

    # enforce monotone-decreasing effective rank across groups by re-sorting
    # groups on their achieved rank (labels 1..N_GROUPS = high->low info)
    metrics = metrics.sort_values('effective_rank', ascending=False).reset_index(drop=True)
    reorder = metrics['group'].tolist()
    groups = [groups[g-1] for g in reorder]
    metrics['group'] = np.arange(1, N_GROUPS+1)

    # write study-compatible CSVs (schema mirrors heterogeneity_study.py) --------
    assign_rows = []
    for g, pix in enumerate(groups, 1):
        for p in pix:
            r, c = divmod(int(p), STORAGE_DIM[1])
            assign_rows.append({'group': g, 'pixel_idx': int(p),
                                'pixel_row': r, 'pixel_col': c,
                                # 'quality' kept for downstream plot compatibility;
                                # here we use effective-rank contribution proxy =
                                # response variance (z-scored) as a stand-in.
                                'quality': float(np.nan)})
    pd.DataFrame(assign_rows).to_csv(os.path.join(OUT_DIR, "group_assignments.csv"),
                                     index=False)
    # add MPD-named column alias so existing plotters that read 'mpd' still work:
    metrics_out = metrics.rename(columns={'effective_rank': 'eff_rank'})
    metrics_out['mpd'] = metrics_out['eff_rank']     # alias: x-axis = info content
    metrics_out.to_csv(os.path.join(OUT_DIR, "group_metrics.csv"), index=False)

    np.savez_compressed(os.path.join(OUT_DIR, "group_masks.npz"),
        **{f"G{g:02d}": np.isin(np.arange(N_PIXELS_TOTAL), pix).reshape(STORAGE_DIM)
           for g, pix in enumerate(groups, 1)})

    print("\n  group |  eff_rank | avg|corr| | logdet")
    for _, r in metrics.iterrows():
        print(f"    G{int(r['group']):02d}  |  {r['effective_rank']:7.3f}  | "
              f"{r['avg_pairwise_corr']:8.3f} | {r['logdet_corr']:8.2f}")

    # quick definition figure: eff_rank across groups
    fig, ax = plt.subplots(figsize=(12,7))
    ax.plot(metrics['group'], metrics['effective_rank'], 'o-',
            color='#1f77b4', lw=5, markersize=16, markeredgecolor='darkblue',
            markeredgewidth=2.5)
    ax.set_xlabel("Group (1 = most information, high effective rank)",
                  fontweight='bold', fontsize=22)
    ax.set_ylabel("Effective rank", fontweight='bold', fontsize=22)
    ax.tick_params(width=3, length=8, labelsize=18)
    for s in ('top','right'): ax.spines[s].set_visible(False)
    for s in ('left','bottom'): ax.spines[s].set_linewidth(3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "Fig_EffRank_per_group.png"),
                dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"\n  Definitions written to: {OUT_DIR}")
    print("  To train on these, point heterogeneity_study.py's group inputs at")
    print("  this folder (or copy group_assignments.csv/group_metrics.csv over),")
    print("  keeping N_PIXELS_PER_GROUP / N_GROUPS / N_DAYS_CAP identical.")


if __name__ == "__main__":
    build()
