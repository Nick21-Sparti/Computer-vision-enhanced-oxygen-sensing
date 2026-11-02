# Data layout and local inputs

## Aqueous recordings

`AQUEOUS_DATA_DIR` must contain one directory per experiment, with names containing a hyphen, such as `2-09_06_2025`. Most aqueous loaders require all three of the following:

| Relative location under an experiment | Meaning |
| --- | --- |
| `<name>_ROI_Output/<name>_YYYYMMDD_HHMMSS.mp4` | Pre-cropped region-of-interest video; `_ROI_Output` must be an immediate child |
| Any subdirectory containing `*_arduino_*.txt` | Oxygen-reference text log |
| Any subdirectory containing `*temperature*.csv` | Temperature log |

Example full paths are `data/aqueous/2-09_06_2025/camera_ROI_Output/roi_20250609_120000.mp4`, `data/aqueous/2-09_06_2025/logs/sensor_arduino_001.txt` and `data/aqueous/2-09_06_2025/logs/temperature.csv`. They illustrate the discovery patterns, not supplied recordings.

The classical-only calibration script searches recursively for an MP4/reference-log pair and does not require a temperature file. Most other aqueous scripts require a temperature file during discovery even where their parser can substitute a default temperature. If a folder is silently skipped, check discovery first. Multiple matching files are ambiguous: many loaders choose the first match, not a user-selected recording.

### Reference and temperature formats

Aqueous text parsers search for `MEA`. The timestamp is the first two whitespace-separated fields in `%Y-%m-%d %H:%M:%S` format. They read the field **five tokens after `MEA`**, convert it to a float and divide by 1000. A schematic line is:

```text
2025-06-09 12:00:00 MEA 0 0 0 0 200000
```

This only illustrates the consumed positions; it is not a full manufacturer protocol record. In this illustration the parsed oxygen value is 200. The scripts do not infer units from the log. For the aqueous study the intended unit is µmol/L; confirm the selected instrument field and scale against your actual export.

The first two temperature CSV columns are interpreted as timestamp and temperature, regardless of their original labels. A minimal example is:

```csv
timestamp,temperature
2025-06-09 12:00:00,23.0
```

Aqueous video timing normally comes from `_YYYYMMDD_HHMMSS` in the video filename, with a reference-log fallback in several loaders. Frame elapsed time uses video FPS metadata. Most scripts sample at approximately 2 frames/s and resize to 48 × 48; inspect the exact loader before reusing an old cache. Timestamp matching differs between variants and is preserved.

## Gas recordings

`GAS_DATA_DIR` must directly contain experiment folders with a timestamp, for example `20260426_164709_50s`.

| Relative location | Meaning |
| --- | --- |
| `20260426_164709_50s/video_20260426_164709.mp4` | Video whose name includes the experiment timestamp |
| `20260426_164709_50s/arduino_20260426_164709.csv` | Corresponding reference log |
| `global_roi_settings.json` | Optional ROI/rotation configuration used only by the gas autograd script |

The main gas TViT and equilibrium loaders match the timestamp in both filenames. The autograd loader instead takes the first MP4 and the first `*arduino*.csv` in each timestamped folder. Use exactly one intended pair to avoid differences between loaders.

Gas reference parsers skip the first line as a header. A record has a `%Y-%m-%d %H:%M:%S` timestamp followed by a comma and the raw instrument string. They select the last `MEA` response and read **token 15, zero-based from `MEA`**, divided by 1000, as oxygen in % V/V. Thus the consumed value is the sixteenth token including `MEA`. The main gas parser reads the second comma-separated field as the raw string, whereas the autograd parser tolerates commas inside the response by normalising separators. A generic two-column oxygen-value CSV is not interchangeable with this raw-protocol CSV.

Gas time is based on `ASSUMED_FPS=10.0`, not automatically adapted to each video's metadata. `FRAME_STEP_RATE=1` retains every frame by default. Confirm the actual capture rate and timestamps before interpreting response times.

### ROI configuration

Only `gas_tvit_autograd_pinn.py` loads `GAS_ROI_FILE`. The accepted dictionary form is:

```json
{"x": 100, "y": 50, "w": 400, "h": 300, "angle": 0.0}
```

Alternatively use `{"roi_coords": [100, 50, 400, 300], "angle": 0.0}`. Values here are illustrative, not the study ROI. They represent x, y, width and height in the source-frame coordinate system. The script applies rotation before cropping. If the file is absent, it uses the full frame; that changes the physical input compared with a cropped run. The other gas scripts do not automatically apply this ROI, so supply consistently preprocessed recordings or explicitly reconcile preprocessing before comparing models.

### Region definitions

Both gas equilibrium scripts contain their own `REGION_DEFINITIONS` dictionaries with the original experiment keys and stable/unstructured intervals. Each entry includes an inference filename and a `regions` list whose items contain `type`, `index`, `start` and `end`. For new recordings, add matching timestamped keys and measured elapsed-time windows to **both scripts**. Unknown experiments are excluded; changing only `GAS_DATA_DIR` does not make a new protocol analysable. `TARGET_EXP` controls a display selection, not the entire experiment list.

## Intermediate files

### Frame caches

Frame caches are generated by the training scripts when missing. A typical aqueous neural cache has:

| Column | Meaning |
| --- | --- |
| `experiment_id` | Experiment folder identifier |
| `timestamp` | Absolute frame time |
| `elapsed` | Seconds from the recording start |
| `oxygen` | Parsed reference concentration |
| `temp` | Logged or substituted temperature where supported |
| `inv_i` | Reciprocal mean red intensity with a small numerical offset |
| `pixels` | Flattened 48 × 48 × 3 uint8 OpenCV image: 6,912 values, BGR order |
| `phase_label` | Present in multitask/curriculum caches: interval-based stable label |
| `target_deriv`, `target_future` | Present in kinetic caches: reference-derived auxiliary targets |

Not all caches have the same schema, alignment or sampling. `inv_std` is included by some variants. The colour convention is **BGR**, with red at index 2; do not convert an old cache to RGB without changing all consuming code.

The shared heterogeneity cache is `cache/<RUN_NAME>/aqueous_heterogeneity/features/idc_master_v10.parquet`. Its readers accept array/list pixels, and the heterogeneity utilities also handle some byte-encoded pixel columns. The information builder requires this file; it does not extract raw video.

For gas equilibrium analysis, `GAS_FRAME_CACHE_FILE` can identify an existing Parquet containing `elapsed`, an experiment ID column (`experiment_id` or `exp_id`), an oxygen column (`oxygen` or `oxygen_gt`) and an intensity column (`inv_i` or `intensity`). Timestamp is optional. Such a scalar cache need not contain pixels. Use `None` to permit automatic discovery in the current gas TViT output or extraction from raw recordings.

### Per-pixel T90

Place `t90_cache_<experiment_id>.npz` files in `PIXEL_T90_DIR`. Each must contain `T90_mean`, matching the 48 × 48 pixel layout and representing per-pixel response time in seconds. The heterogeneity scripts flatten and average these arrays across days. They also retain the original recursive fallback search under the aqueous raw-data root.

These inputs and their generating pixel-map script were **not provided**. Despite a historical comment, the attached `response_paper_1_3(1).py` (now `aqueous_tvit.py`) does not create these files. If no T90 cache is found, the scripts can continue with a reduced feature space. This is a different grouping analysis from four-feature MPD and must be labelled accordingly. `INCLUDE_T90_IN_FEATURES=False` explicitly requests a three-feature analysis.

### Heterogeneity group files

`heterogeneity_quality_groups.py` writes `Group_Definitions/pixel_features.npz`, `group_assignments.csv` and `group_metrics.csv` under its output directory. The pixel archive contains mean intensity, SV R², SV sensitivity, T90 and a cache key. The linear comparator reads these group definitions and can merge existing TViT predictions into its summary.

The information builder needs the frame cache and, for its configured SV filter to be effective, the pixel archive. Without that archive it falls back to all pixels, even when `FILTER_BY_SV_R2=True`. Its output `mpd` field is an alias for effective rank, not physical-feature MPD. Its files belong in its own output directory and do not by themselves cause the quality-group trainer to train on them.

### Optional historical comparison inputs

- `AQUEOUS_T90_REPORTS_DIR`: `Extracted_T90_RawAligned_Metrics.csv` and/or `Extracted_T90_Raw_Metrics.csv` for the dynamic benchmark's historical TViT comparison. Its internal comparisons do not require these imports.
- `GAS_POSTHOC_CACHE_DIR`: `kalman_df_v141R_*.parquet`, with `exp_id`, `elapsed` and optional AI columns `o2_ai_v11_1`, `o2_ai_v11_5_sf`, `o2_ai_v12_3_ac`; corresponding `*_kalman` columns can supply historical smoothed results. The newest matching file is selected. The original posthoc exporter was not attached. Without it the equilibrium scripts run classical analyses only.

Do not rename arbitrary CSVs or Parquet files to match these patterns: the columns, experiment IDs, timestamps, units and preprocessing must also match.
