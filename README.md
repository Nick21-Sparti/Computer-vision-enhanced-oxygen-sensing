# Computer-vision-enhanced-oxygen-sensing

Luminescence-based chemical sensors are almost universally read as point detectors, with the signal inverted through a single calibration model. Here we introduce a concept of computational sensing using computer vision to exploit the heterogeneities present in oxygen-sensitive films.

This repository contains Python research scripts accompanying **Computer vision enhanced oxygen sensing**. A low-cost camera records the phosphorescence of a porous platinum octaethylporphyrin/polystyrene (PtOEP/PS) film. Instead of averaging the image before estimating oxygen, a temporal vision transformer (TViT) combines spatially resolved measurements and recent frames to reconstruct the oxygen concentration measured by a reference sensor. The analyses cover biofouled aqueous experiments and gaseous oxygen experiments.

**Release scope:** 16 research scripts, renamed and adapted for local execution, with setup instructions and a code-level description of each analysis. Raw recordings, reference logs, trained weights, historical result caches and the complete manuscript figure-generation suite are not included. This is a source-code release prepared from the supplied scripts; it has not been validated by rerunning the experiments. See [reproducibility and coverage](docs/REPRODUCIBILITY.md) before interpreting results.

## Scientific rationale

The polymer matrix introduces a diffusion-dependent delay between a change in the surrounding oxygen concentration and the measured luminescence. Film thickness, morphology, illumination and local sensitivity also vary spatially. A single intensity average compresses these different responses into one signal, while an equilibrium calibration does not explicitly model their time dependence.

A spatially resolved readout retains those differences. Temporal stacking gives the decoder access to recent sensor history, allowing it to learn a mapping from the evolving film image to the contemporaneous bulk oxygen concentration. The study tests whether differences between pixels provide useful complementary information, and compares this approach with static calibration, dynamic inverse methods, linear decoders and physics-informed variants.

The supplied manuscript reports reductions of up to **96% in MAE**, **91% in T90**, and **37% in MAE through pixel-performance diversity**, under the particular comparisons described there. These are manuscript-reported results, not benchmarks generated or verified by this repository preparation. They should not be interpreted as expected performance on a new film or dataset.

## What is included

| Location | Contents |
| --- | --- |
| [`scripts/`](scripts/) | The 16 renamed research scripts and shared path configuration |
| [`scripts/repo_config.py`](scripts/repo_config.py) | Raw-data roots, output/cache locations and previously generated inputs |
| [`docs/SCRIPT_GUIDE.md`](docs/SCRIPT_GUIDE.md) | Detailed purpose, method, defaults, outputs and dependencies for every script |
| [`docs/DATA_LAYOUT.md`](docs/DATA_LAYOUT.md) | Required file naming, log formats, units and cache schemas |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | Evaluation limitations, manuscript coverage and missing inputs |
| [`docs/FILE_RENAMING.md`](docs/FILE_RENAMING.md) | Exact original-to-new filename mapping |
| [`docs/CHANGES.md`](docs/CHANGES.md) | Changes made for this repository edition |
| [`docs/VALIDATION.md`](docs/VALIDATION.md) | Checks performed and what was not tested |
| [`tools/check_setup.py`](tools/check_setup.py) | Read-only source, package and recording-discovery checks |
| [`requirements.txt`](requirements.txt) | Analysis dependencies; install PyTorch separately first |

The scripts are independent research entry points, not numbered stages of one pipeline. Run the analysis you need. Some share intermediate files; those dependencies are described below.

## Install

Use a dedicated Python environment. Python 3.11 is a practical starting target for this release, but no original environment lockfile was provided and this is not a tested version guarantee.

On Windows, from the repository folder:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

On Linux or macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a matched `torch` and `torchvision` build using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/), choosing your operating system and compute platform. Then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

Optional Optuna plot exports:

```bash
python -m pip install -r requirements-optional.txt
```

`plotly` and `kaleido` are only needed for certain optimisation-history image exports; their availability does not establish that model training works. The scripts call `timm.create_model(..., pretrained=True)`, which loads pretrained weights and may download them on first use; see the [official timm quickstart](https://huggingface.co/docs/timm/quickstart). Pretrained backbone weights are distinct from this study's trained oxygen-regression checkpoints, which are not supplied.

Most neural scripts choose CUDA when available and otherwise use CPU. CPU execution is possible in principle but full cross-validation can be expensive. The gas autograd script uses a noninteractive plotting backend. Other scripts can be run headlessly by setting `MPLBACKEND=Agg` in the environment before starting Python. Do not install both OpenCV GUI and headless distributions into the same environment.

## Configure your computer

Open **[`scripts/repo_config.py`](scripts/repo_config.py)**. Only the block at the top needs editing to change locations:

```python
AQUEOUS_DATA_DIR = r"D:\oxygen_study\aqueous"
GAS_DATA_DIR = r"D:\oxygen_study\gas"
OUTPUT_ROOT = r"D:\oxygen_study\results"
CACHE_ROOT = r"D:\oxygen_study\cache"
RUN_NAME = "initial_run"
```

These are examples: use the folders on your computer. Each data root must directly contain the experiment folders, not just one experiment. Windows raw strings must not end in a single backslash. Forward slashes also work, for example `"D:/oxygen_study/aqueous"`.

Alternatively, leave the defaults and place your recordings under `data/aqueous/` and `data/gas/`. Relative paths are resolved from the repository root, so changing the IDE working directory does not change the data location.

The same configuration file exposes previously saved inputs:

| Setting | What belongs there |
| --- | --- |
| `PIXEL_T90_DIR` | `t90_cache_<experiment_id>.npz` files containing per-pixel `T90_mean` arrays |
| `AQUEOUS_T90_REPORTS_DIR` | Optional historical T90 CSVs imported by the dynamic benchmark |
| `GAS_POSTHOC_CACHE_DIR` | Optional `kalman_df_v141R_*.parquet` files with historical AI predictions |
| `GAS_ROI_FILE` | Optional JSON ROI/rotation file for the gas autograd model; `None` looks in the gas data root |
| `GAS_FRAME_CACHE_FILE` | Optional existing frame-level Parquet for the two gas equilibrium scripts; `None` enables discovery/raw extraction |

You do not need old caches for the main training scripts: they extract their inputs from videos and logs. Exceptions and fallback behaviour are detailed in [the data guide](docs/DATA_LAYOUT.md). Old saved model files are not automatically migrated into the new locations.

For automation, environment variables override the editable defaults: `OXYGEN_AQUEOUS_DATA`, `OXYGEN_GAS_DATA`, `OXYGEN_OUTPUT_ROOT`, `OXYGEN_CACHE_ROOT`, `OXYGEN_RUN_NAME`, `OXYGEN_PIXEL_T90_DIR`, `OXYGEN_AQUEOUS_T90_REPORTS`, `OXYGEN_GAS_POSTHOC_CACHE`, `OXYGEN_GAS_ROI_FILE` and `OXYGEN_GAS_FRAME_CACHE`. Manual editing is sufficient for normal IDE use.

## Check the setup

Run `scripts/repo_config.py` in your IDE to print the resolved paths. For a read-only source check that requires no scientific packages or recordings:

```bash
python tools/check_setup.py --source-only
```

After installing dependencies and pointing to the data:

```bash
python tools/check_setup.py --dataset aqueous
python tools/check_setup.py --dataset gas
```

The latter checks package presence and filename discovery. It does not parse videos/logs, verify oxygen units or execute a model. An analysis needs at least two valid experiments for leave-one-experiment-out evaluation; the Optuna variants need at least three experiment groups in their sampled tuning subset.

## Choose an analysis

| Script | Purpose | Main prerequisite |
| --- | --- | --- |
| [`aqueous_tvit.py`](scripts/aqueous_tvit.py) | Main aqueous TViT and two-site Stern–Volmer comparison | Aqueous recordings and logs |
| [`aqueous_classical_calibration.py`](scripts/aqueous_classical_calibration.py) | Two-site SV and cubic polynomial; plateau-only versus full-timeline calibration | Aqueous recordings and reference logs |
| [`aqueous_dynamic_benchmarks.py`](scripts/aqueous_dynamic_benchmarks.py) | Static SV, first-order inverse, regularised deconvolution and causal Kalman; optional neural comparators | Aqueous recordings and logs |
| [`aqueous_tvit_mass_transfer.py`](scripts/aqueous_tvit_mass_transfer.py) | TViT with an inverse-intensity mass-transfer penalty | Aqueous recordings and logs |
| [`aqueous_tvit_multitask.py`](scripts/aqueous_tvit_multitask.py) | Oxygen regression, uncertainty, phase classification and adversarial experiment classification | Aqueous recordings and logs |
| [`aqueous_tvit_bayesian.py`](scripts/aqueous_tvit_bayesian.py) | MC-dropout uncertainty with Optuna tuning | Aqueous recordings and logs |
| [`aqueous_tvit_kinetic.py`](scripts/aqueous_tvit_kinetic.py) | Current, recent-change and future-oxygen supervision | Aqueous recordings and logs |
| [`aqueous_tvit_curriculum.py`](scripts/aqueous_tvit_curriculum.py) | Kinetic auxiliary losses followed by current-oxygen-only training | Aqueous recordings and logs |
| [`heterogeneity_quality_groups.py`](scripts/heterogeneity_quality_groups.py) | Quality-ranked pixel groups spanning performance diversity; masked TViT versus group SV | Aqueous recordings; T90 caches for four-feature MPD |
| [`heterogeneity_linear_baselines.py`](scripts/heterogeneity_linear_baselines.py) | Uniform SV, weighted inverse-intensity regression and linear temporal decoder | Preferably completed quality-group definitions/results |
| [`heterogeneity_group_size.py`](scripts/heterogeneity_group_size.py) | Compare 4–64 pixels, selecting high-MPD groups at each size | Aqueous recordings; selected days; optional T90 caches |
| [`heterogeneity_information_groups.py`](scripts/heterogeneity_information_groups.py) | Build pixel groups ranked by effective rank; no model training | Heterogeneity frame cache and pixel-feature file |
| [`gas_tvit.py`](scripts/gas_tvit.py) | Gaseous TViT with timestamp-based discovery and fixed FPS | Gas recordings and reference CSVs |
| [`gas_tvit_autograd_pinn.py`](scripts/gas_tvit_autograd_pinn.py) | Gas TViT with a continuous-time autograd physics penalty | Gas recordings; optional ROI JSON |
| [`gas_equilibrium_stern_volmer.py`](scripts/gas_equilibrium_stern_volmer.py) | Equilibrium-only two-site SV, all-region comparator and optional RTS postprocessing | Gas frames/raw data and matching region definitions |
| [`gas_equilibrium_polynomial.py`](scripts/gas_equilibrium_polynomial.py) | Corresponding equilibrium polynomial analysis | Gas frames/raw data and matching region definitions |

The [script guide](docs/SCRIPT_GUIDE.md) explains what each method actually implements, its settings and its outputs.

## Run locally

**PyCharm, Spyder or another IDE:** select the environment created above, open the desired script and press Run. Keep `repo_config.py` beside the analysis scripts in `scripts/`; downloading one script alone is insufficient. In notebook-style consoles, use `%run scripts/aqueous_tvit.py` from the repository root rather than importing the script to launch it.

Equivalent terminal commands:

```bash
python scripts/aqueous_classical_calibration.py
python scripts/aqueous_tvit.py
python scripts/aqueous_dynamic_benchmarks.py
```

These are independent analyses, not a requirement to run all three. The classical calibration script is a useful first check because it does not train a neural model.

For the linked heterogeneity analyses:

1. Run `heterogeneity_quality_groups.py` to create its frame cache, pixel features, group definitions and TViT results.
2. Run `heterogeneity_linear_baselines.py` with matching day/group settings to compare the same groups. It intentionally updates the shared summary and some comparison plots; run it after the TViT job, not concurrently.
3. Run `heterogeneity_group_size.py` as a separate size study.
4. Run `heterogeneity_information_groups.py` to construct information-ranked definitions. **This script does not train on those groups.** The supplied quality-group trainer rebuilds its own groups; copying the information-group CSVs into its output folder is not a supported training workflow.

For gas experiments, `gas_tvit.py` and `gas_tvit_autograd_pinn.py` are independent training analyses. The equilibrium SV and polynomial scripts can reuse the gas TViT frame cache automatically, but can also extract their own scalar frame data directly from raw recordings. Run the SV equilibrium analysis first if you want the polynomial analysis to reuse its frame cache. Historical AI overlay comparisons require the separate V14 posthoc cache; running `gas_tvit.py` alone does not produce that cache.

## Settings that are not paths

Review the configuration block in each research script before running. Existing experimental defaults are retained: changing paths does not adapt a protocol to new data.

- **Timing and units:** aqueous calibration windows and `UNSTRUCTURED_CUTOFF_S`; gas `ASSUMED_FPS`, `FRAME_STEP_RATE` and `REGION_DEFINITIONS`; reference-log field positions and scaling.
- **Training:** `N_EPOCHS`, `BATCH_SIZE`, `LR`, temporal stack size/spacing and Optuna settings. Several legacy loaders use their dataset constructor's default stack spacing rather than the similarly named global constant; see the reproducibility notes.
- **Heterogeneity:** day selection, group count/size, SV R² threshold, T90 inclusion, backbone freezing and checkpoint selection.
- **Resource use:** start with a small diagnostic run, keep `NUM_WORKERS=0` where exposed on Windows, and reduce batch size if GPU memory is insufficient. The gas autograd script sets worker count directly in its DataLoader calls.

A diagnostic subset is not a reproduction of the full study. Restore the intended fold set and settings before collecting final metrics.

## Outputs and repeat runs

Outputs default to `outputs/default/<analysis>/`; extraction/fold caches generally go to `cache/default/<analysis>/`. Gas neural scripts retain `4_Caches/` within their own output directories. The three heterogeneity training/comparison scripts share `cache/default/aqueous_heterogeneity/features/` intentionally; the information-group builder reads it. The linear comparator also shares the quality-group output directory.

Expect per-experiment prediction CSVs, parity/residual plots and summary tables; neural variants may additionally save histories, weights and checkpoints. Gas equilibrium scripts export calibration parameters, plateau selection, response-time tables, an Excel summary and a run manifest. See the individual guide for exact locations.

**Set a new `RUN_NAME` whenever data or analysis settings change.** Many legacy scripts resume from filenames or completed-summary rows rather than a complete data/configuration fingerprint. A new name separates both caches and results without deleting a previous run. Use the same name for analyses that must exchange files. Also update any explicitly configured external cache paths when switching datasets.

## Reproducibility, citation and reuse

This repository preserves the supplied research logic apart from the explicitly listed runtime fixes. It does not silently rewrite model selection, group construction, temporal alignment or calibration rules. Some variants do not implement a strictly nested evaluation: Optuna tuning precedes the outer folds, or held-out-day MAE determines checkpoint selection. The [reproducibility notes](docs/REPRODUCIBILITY.md) explain these distinctions and the missing pieces needed for complete figure reproduction.

The associated manuscript authors are Nikolaos Salaris, Jianhui Zhang, Abbas Heydari, Evangelos Mazomenos, Adrien Desjardins and Manish K. Tiwari. See [`CITATION.cff`](CITATION.cff). A verified publication DOI, final paper URL and data accession were not supplied for this release; add them when confirmed. No software licence has been selected in this package. The maintainers should add the intended licence before presenting it as an open-source release; see [publication notes](docs/PUBLISHING.md).
