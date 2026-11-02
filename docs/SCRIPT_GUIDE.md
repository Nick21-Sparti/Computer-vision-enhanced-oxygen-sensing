# Script guide

Descriptions below refer to the executable code in the attached versions, not every historical claim in their header comments. Default tables are extracted from the repository scripts. Settings are not a certified recipe for reproducing every manuscript figure. Paths use the common configuration described in the main README.

## 1. Main aqueous temporal vision transformer

[`aqueous_tvit.py`](../scripts/aqueous_tvit.py)

Discovers ROI videos, oxygen logs and temperature logs; extracts approximately 2 frames/s at 48 × 48; aligns each frame with a reference reading; and concatenates three BGR frames along the channel axis. A pretrained `vit_small_patch16_224` has its input projection expanded to nine channels. Its CLS embedding feeds a scalar oxygen regressor trained with MSE and AdamW.

Each experiment is held out in turn. The two-site Stern–Volmer comparator is fitted on the training experiments' configured stable intervals, falling back to all training frames if no stable samples are found. Its fitted forward intensity model is analytically inverted for prediction. The TViT uses a fixed epoch budget and final-epoch predictions; held-out MAE is displayed during training but does not select a checkpoint.

**Outputs:** `outputs/<run>/aqueous_tvit/`, including `Global_Performance_Summary.csv`, `csv_exports/Inference_<experiment>.csv`, inference and chemometric plots, and per-epoch histories. This script does **not** save trained TViT weights and does not generate per-pixel T90 caches. The dataset constructor defaults to spacing 5 samples in this entry point; changing `FRAME_STACK_STEP` alone does not change its two DataLoader constructions.

**Run:** `python scripts/aqueous_tvit.py`

**Retained top-level defaults:**

| Setting | Value |
| --- | --- |
| `N_EXPERIMENTS_CAP` | `None` |
| `N_STACKED_FRAMES` | `3` |
| `FRAME_STACK_STEP` | `10` |
| `STORAGE_DIM` | `(48, 48)` |
| `BATCH_SIZE` | `32` |
| `N_EPOCHS` | `20` |
| `LR` | `0.0001` |

## 2. Static calibration controls

[`aqueous_classical_calibration.py`](../scripts/aqueous_classical_calibration.py)

Uses a reference text log and a video from each aqueous experiment, extracts red-channel intensity and compares two calibration branches: fit only within the prescribed stable windows, or fit the full training timeline. Both branches fit a two-site Stern–Volmer intensity model and a cubic polynomial mapping intensity directly to oxygen, then evaluate the held-out experiment.

The intensity extractor selects bright pixels independently in each sampled frame. With `TOP_N_PIXELS=10000` and the retained 48 × 48 resize, all 2,304 pixels are selected. Thus the default is effectively a whole-frame intensity mean; the older 128 × 128 setting in a comment is not active.

**Outputs:** `outputs/<run>/aqueous_classical_calibration/{Plateau_Only,Full_Timeline}/`, each with `Calibration_Fits`, `Inference_Plots` and `CSVs/Result_<experiment>.csv`. The script does not train a neural model or write a global aggregate MAE table. It is independent of the TViT cache and is a useful first input check. Curve-fit fallback parameters can still produce results after a failed fit, so inspect the calibration plots.

**Run:** `python scripts/aqueous_classical_calibration.py`

**Retained top-level defaults:**

| Setting | Value |
| --- | --- |
| `STORAGE_DIM` | `(48, 48)` |
| `TOP_N_PIXELS` | `10000` |

## 3. Dynamic compensation benchmark suite

[`aqueous_dynamic_benchmarks.py`](../scripts/aqueous_dynamic_benchmarks.py)

Builds per-experiment frame caches and performs experiment-wise outer cross-validation. It fits the steady-state calibration using outer-training data, then compares static SV, a first-order inverse with causal derivative filtering, regularised frequency-domain deconvolution and a causal two-state bulk/surface Kalman filter. Dynamic parameters are selected on outer-training data, not the outer test experiment.

CNN–LSTM, CNN–TCN and TViT implementations are present but disabled in `RUN_METHODS`. Set their entries to `True` to include them; they use a fixed epoch budget. Regularised frequency-domain deconvolution is an offline whole-series method and must not be labelled causal. Parameter searches here are training-data fits/searches, not a claim of a fully nested validation design for every parameter.

**Outputs:** `outputs/<run>/aqueous_dynamic_benchmarks/`, including inference CSVs, metric summaries, fitted parameter files, run logs, parity plots, raw-aligned and Savitzky–Golay T90 reports. Fold/prediction caches include configuration and data-signature checks, but not every reusable intermediate has a full data checksum. Optional V14 T90 imports add a historical comparator only. Deep-learning packages are still imported even when neural methods are disabled.

**Run:** `python scripts/aqueous_dynamic_benchmarks.py`

**Retained top-level defaults:**

| Setting | Value |
| --- | --- |
| `N_EXPERIMENTS_CAP` | `None` |
| `N_STACKED_FRAMES` | `3` |
| `FRAME_STACK_STEP` | `10` |
| `STORAGE_DIM` | `(48, 48)` |
| `BATCH_SIZE` | `32` |
| `N_EPOCHS` | `20` |
| `LR` | `0.0001` |

## 4. TViT with a finite-difference mass-transfer penalty

[`aqueous_tvit_mass_transfer.py`](../scripts/aqueous_tvit_mass_transfer.py)

Adds an auxiliary physics penalty to supervised oxygen MSE. A central image patch supplies an inverse-intensity proxy S from the current and previous stacked frame. The loss compares the oxygen prediction with `alpha*S + (alpha/k)*(S_current - S_previous)`; both `alpha` and `k` are optimised with the network. The reference comparator is a quadratic polynomial in inverse intensity fitted over stable training intervals.

This is a discrete proxy-based penalty, not an autograd solution of a diffusion PDE. In the supplied implementation the difference is not divided by elapsed time, and stack channels 0 and 3 are used (blue in the retained OpenCV BGR convention), although several other analyses use red. These choices are preserved and should be considered when interpreting fitted parameters.

**Outputs:** inference and mechanism plots, chemometrics, training/physics-parameter histories, a global metric summary and model weights under its own fold-cache directory. Only network weights are saved there; learned loss-module parameters `alpha` and `k` are not restored on the cached-weight branch. Consequently `Learned_k` on such a rerun need not represent the original trained value. A pre-existing `slice(None).any()` failure in full-timeline MAE reporting is fixed in this edition.

**Run:** `python scripts/aqueous_tvit_mass_transfer.py`

**Retained top-level defaults:**

| Setting | Value |
| --- | --- |
| `N_EXPERIMENTS_CAP` | `None` |
| `PINN_PHYSICS_WEIGHT` | `0.5` |
| `N_STACKED_FRAMES` | `3` |
| `FRAME_STACK_STEP` | `10` |
| `STORAGE_DIM` | `(48, 48)` |
| `BATCH_SIZE` | `32` |
| `N_EPOCHS` | `20` |
| `LR` | `0.0001` |

## 5. Multitask and domain-adversarial TViT

[`aqueous_tvit_multitask.py`](../scripts/aqueous_tvit_multitask.py)

Uses a shared ViT representation with three tasks: an oxygen head predicting mean and log variance, a stable/transient phase classifier, and an experiment/domain classifier behind a gradient-reversal layer. A heteroscedastic Gaussian negative-log-likelihood trains the oxygen output, with weighted cross-entropy auxiliary losses. Stable phase labels come from the configured time windows; the static comparator is a quadratic inverse-intensity calibration.

The predicted variance is a learned conditional noise model. It is not an independent ensemble standard deviation or an automatically calibrated confidence interval. Domain accuracy is reported on the held-out experiment, whose class is absent from the training samples; that number alone does not establish domain invariance.

**Outputs:** per-fold weights in `7_Model_Checkpoints`, incremental global summary rows, inference CSVs, phase/uncertainty diagnostics and training histories. Completed experiments are skipped according to the existing summary, so use a new run name after changing settings. Final-epoch weights are used. Dataset stack spacing defaults to 5 samples in the main loader calls.

**Run:** `python scripts/aqueous_tvit_multitask.py`

**Retained top-level defaults:**

| Setting | Value |
| --- | --- |
| `N_EXPERIMENTS_CAP` | `None` |
| `N_STACKED_FRAMES` | `3` |
| `FRAME_STACK_STEP` | `10` |
| `STORAGE_DIM` | `(48, 48)` |
| `BATCH_SIZE` | `32` |
| `N_EPOCHS` | `20` |
| `LR` | `0.0001` |

## 6. MC-dropout uncertainty analysis

[`aqueous_tvit_bayesian.py`](../scripts/aqueous_tvit_bayesian.py)

Extends the multitask model with dropout and a mean/log-variance oxygen output. Optuna tunes learning rate, dropout and the phase-loss weight on a random 5% subset. After fold training, repeated stochastic forward passes estimate epistemic variance from the variance of predicted means; mean predicted variance represents aleatoric uncertainty. Their sum gives total variance.

The default is 50 Monte Carlo passes. This is an approximate uncertainty method based on dropout, not exact Bayesian inference or a trained ensemble of independent models. Optuna uses the first split from a three-group `GroupKFold`, and is run once on the whole dataset before outer LOOCV. Its chosen settings therefore are not independent of all subsequent test experiments. Random subset order also changes the meaning of index-based temporal stacking during tuning.

**Outputs:** uncertainty/phase diagnostics, predictions, final model weights, summary rows, and an Optuna SQLite database plus available trial/visualisation exports in `8_Optuna_Optimization`. A persistent study may be reused on rerun; change the run label for a new dataset or search.

**Run:** `python scripts/aqueous_tvit_bayesian.py`

**Retained top-level defaults:**

| Setting | Value |
| --- | --- |
| `N_EXPERIMENTS_CAP` | `None` |
| `N_STACKED_FRAMES` | `3` |
| `FRAME_STACK_STEP` | `10` |
| `STORAGE_DIM` | `(48, 48)` |
| `BATCH_SIZE` | `32` |
| `LR` | `0.0001` |
| `N_EPOCHS` | `20` |
| `MC_SAMPLES` | `50` |
| `OPTUNA_TRIALS` | `20` |
| `OPTUNA_SUBSET_FRAC` | `0.05` |

## 7. Kinetic multitask learning

[`aqueous_tvit_kinetic.py`](../scripts/aqueous_tvit_kinetic.py)

Predicts current oxygen, a recent reference-oxygen change, and a future reference-oxygen target through three heads sharing a ViT backbone. The auxiliary targets are constructed within each experiment: the default change window is 1 s and future horizon is 5 s. End-of-recording future targets are filled with current oxygen rather than discarded. Past frames remain the model input; future reference values are auxiliary supervised targets.

The target named derivative is an oxygen difference, not generally a time-normalised derivative when the configured window changes. Optuna tunes learning rate and the two auxiliary weights on a random 5% subset before outer LOOCV, with the same non-nested tuning and temporal-order limitations as the Bayesian variant. The comparison calibration is quadratic in inverse intensity.

**Outputs:** current/future/change predictions and kinetic diagnostics, histories, per-fold model checkpoints, global summaries and the Optuna study. Fresh Polars-derived pixel lists are converted to NumPy before reshaping in this repository edition. The code's default stack spacing in its loader calls is 5 samples.

**Run:** `python scripts/aqueous_tvit_kinetic.py`

**Retained top-level defaults:**

| Setting | Value |
| --- | --- |
| `N_EXPERIMENTS_CAP` | `None` |
| `N_STACKED_FRAMES` | `3` |
| `FRAME_STACK_STEP` | `10` |
| `STORAGE_DIM` | `(48, 48)` |
| `BATCH_SIZE` | `32` |
| `N_EPOCHS` | `20` |
| `LR` | `0.0001` |
| `DERIVATIVE_WINDOW_S` | `1.0` |
| `FUTURE_PRED_S` | `5.0` |
| `OPTUNA_TRIALS` | `20` |
| `OPTUNA_SUBSET_FRAC` | `0.05` |

## 8. Kinetic curriculum learning

[`aqueous_tvit_curriculum.py`](../scripts/aqueous_tvit_curriculum.py)

Uses the same current/change/future target structure as the kinetic model, with a staged loss schedule. During the initial phase all heads contribute; from zero-based epoch `CURRICULUM_CUTOFF_EPOCH=18`, both auxiliary-loss weights are set to zero and the remaining training optimises only current oxygen. With 25 epochs, this means 18 initial epochs and seven final epochs.

Optuna chooses the initial loss balance and learning rate before outer folds using a random subset. This preserves the same non-nested and shuffled temporal-stack limitations. The configured three-frame stack is the attached default; the manuscript's six-frame curriculum experiment is not automatically selected by this script.

**Outputs:** training dashboards marking the curriculum switch, kinetic diagnostics, coloured parity plots, inference CSVs, checkpoints, summary rows and the Optuna database. Use a new run label after changing frame count or curriculum length. As in the kinetic script, list-valued pixels are normalised to NumPy before reshaping; loader stack arguments still need checking if changing global frame settings.

**Run:** `python scripts/aqueous_tvit_curriculum.py`

**Retained top-level defaults:**

| Setting | Value |
| --- | --- |
| `N_EXPERIMENTS_CAP` | `None` |
| `N_STACKED_FRAMES` | `3` |
| `FRAME_STACK_STEP` | `10` |
| `STORAGE_DIM` | `(48, 48)` |
| `BATCH_SIZE` | `32` |
| `N_EPOCHS` | `25` |
| `LR` | `0.0001` |
| `DERIVATIVE_WINDOW_S` | `1.0` |
| `FUTURE_PRED_S` | `5.0` |
| `CURRICULUM_CUTOFF_EPOCH` | `18` |
| `OPTUNA_TRIALS` | `15` |
| `OPTUNA_SUBSET_FRAC` | `0.05` |

## 9. Quality-ranked pixel-group study

[`heterogeneity_quality_groups.py`](../scripts/heterogeneity_quality_groups.py)

Characterises each 48 × 48 pixel using stable-window intensity, linearised Stern–Volmer R² and sensitivity, with optional T90 loaded from external arrays. Eligible pixels satisfy the configured SV R² threshold. The ranking score is `Q = W_INTENSITY*z(mean_intensity) + W_R2*z(SV_R2)`; mean pairwise distance (MPD) is calculated in the standardised performance-feature space. Expanding quality-ranked windows construct equal-sized groups with different diversity.

For each group the script zeros every other pixel, resizes the masked image and trains a TViT against a group-mean two-site SV comparator. This retains the selected pixels' locations in the original image; the compact 4 × 4 visual tiles are only displays of 16 selected pixels. Defaults select five days, eight groups, 15 epochs, a frozen backbone and early stopping; these are not the full 13-day settings reported elsewhere in the manuscript.

**Outputs:** group assignments/metrics/masks, pixel-feature cache, per-group plots and predictions, model files, resumable checkpoints and `Summary/Global_Group_Performance.csv`. The held-out day is also used for early stopping and best-weight selection, and pixel features/groups are built before outer folds. Missing T90 files reduce the MPD feature space. Disabling early stopping alone does not remove best-checkpoint selection, because best weights are restored unconditionally.

**Run:** `python scripts/heterogeneity_quality_groups.py`

**Retained top-level defaults:**

| Setting | Value |
| --- | --- |
| `N_PIXELS_PER_GROUP` | `16` |
| `N_DAYS_CAP` | `5` |
| `N_GROUPS` | `8` |
| `FREEZE_BACKBONE` | `True` |
| `USE_EARLY_STOPPING` | `True` |
| `EARLY_STOP_PATIENCE` | `3` |
| `SV_R2_POOL_THRESHOLD` | `0.95` |
| `GROUP_MODE` | `'expanding'` |
| `INCLUDE_T90_IN_FEATURES` | `True` |
| `N_STACKED_FRAMES` | `3` |
| `FRAME_STACK_STEP` | `10` |
| `STORAGE_DIM` | `(48, 48)` |
| `BATCH_SIZE` | `32` |
| `N_EPOCHS` | `15` |
| `LR` | `0.0001` |

## 10. Linear comparators for the quality groups

[`heterogeneity_linear_baselines.py`](../scripts/heterogeneity_linear_baselines.py)

Compares three decoders over the selected pixels: uniform-mean two-site SV; ridge regression from per-pixel inverse intensities fitted over stable intervals; and ridge regression from a temporally stacked red-intensity feature vector fitted over the whole training timeline. The weighted method is direct linear concentration regression on inverse intensities, not simply a weighted intensity followed by the identical nonlinear SV curve.

The default ridge search chooses alpha by leave-one-training-day-out validation. If no stable training frames exist, the weighted branch falls back to test frames; valid training windows are essential. `USE_SAVED_GROUP_DEFINITIONS=True` loads existing quality-group CSVs when available; absent files trigger rebuilding. The script also rebuilds features for comparison, so day count and group settings must agree with the TViT run. Its default 13-day cap differs from the quality trainer's five-day cap.

**Outputs:** it intentionally writes into `outputs/<run>/heterogeneity_quality_groups/`, backs up/updates the shared global summary, writes `Inference_Linear` CSVs and regenerates comparison plots. Some chemometric and summary plot filenames are shared and can be replaced by the comparison figures. Run it sequentially after the quality trainer. Its internal ridge selection uses training days, but precomputed global group definitions still need the evaluation qualifications in the reproducibility notes.

**Run:** `python scripts/heterogeneity_linear_baselines.py`

**Retained top-level defaults:**

| Setting | Value |
| --- | --- |
| `N_PIXELS_PER_GROUP` | `16` |
| `N_DAYS_CAP` | `13` |
| `N_GROUPS` | `8` |
| `SV_R2_POOL_THRESHOLD` | `0.95` |
| `GROUP_MODE` | `'expanding'` |
| `INCLUDE_T90_IN_FEATURES` | `True` |
| `N_STACKED_FRAMES` | `3` |
| `FRAME_STACK_STEP` | `10` |
| `STORAGE_DIM` | `(48, 48)` |

## 11. Pixel-count study

[`heterogeneity_group_size.py`](../scripts/heterogeneity_group_size.py)

Compares groups containing 4, 9, 16, 25, 36, 49 and 64 pixels, displayed as 2 × 2 through 8 × 8 tiles. For each size it searches candidate windows in the quality-ranked eligible pool and selects a group with maximum achieved MPD. These are not necessarily contiguous spatial patches and the search is not an exhaustive optimisation over all pixel combinations.

The script selects the requested experiment numbers through its `select_days` helper; the attached setting is `[11, 12, 13, 14, 15]`. Verify the printed selected IDs against your folders. It builds a frame store for reuse, trains a masked TViT per size/fold and fits the classical comparator. Freezing, mixed precision and held-out-day best-checkpoint selection are retained.

**Outputs:** a day-tagged directory under `outputs/<run>/heterogeneity_group_size/`, with group definitions, per-size inference, checkpoints/models and a global size-comparison summary. It shares the heterogeneity extraction cache but has its own pixel-feature/group output. Missing T90 values and global feature construction have the same implications as the quality-group study.

**Run:** `python scripts/heterogeneity_group_size.py`

**Retained top-level defaults:**

| Setting | Value |
| --- | --- |
| `DAYS_TO_RUN` | `[11, 12, 13, 14, 15]` |
| `GROUP_SIDES` | `[2, 3, 4, 5, 6, 7, 8]` |
| `FREEZE_BACKBONE` | `True` |
| `USE_EARLY_STOPPING` | `True` |
| `EARLY_STOP_PATIENCE` | `3` |
| `SV_R2_POOL_THRESHOLD` | `0.95` |
| `INCLUDE_T90_IN_FEATURES` | `True` |
| `N_STACKED_FRAMES` | `3` |
| `FRAME_STACK_STEP` | `10` |
| `STORAGE_DIM` | `(48, 48)` |
| `BATCH_SIZE` | `32` |
| `N_EPOCHS` | `20` |
| `LR` | `0.0001` |

## 12. Information-ranked group construction

[`heterogeneity_information_groups.py`](../scripts/heterogeneity_information_groups.py)

Reads red-channel pixel time series from the shared heterogeneity Parquet and averages interpolated responses across the selected days on a common 400-point grid. A greedy ordering starts with a high-variance response and adds pixels to increase effective rank. Equal-sized windows along this ordering are then sorted by achieved rank.

Effective rank is `exp(-sum(p_k * log(p_k)))`, where `p_k` are normalised singular values of the mean-centred pixel-by-time response matrix. It measures temporal response diversity and is distinct from MPD in calibration-feature space. The output CSV's `mpd` field aliases effective rank for legacy plot compatibility; it must be labelled accordingly.

**Inputs:** the shared frame cache and preferably the quality study's `pixel_features.npz`. Without the latter, the candidate filter falls back to all pixels. **Outputs:** `group_assignments.csv`, `group_metrics.csv`, `group_masks.npz` and `Fig_EffRank_per_group.png` under its own `Group_Definitions` directory. There is no model training or MAE calculation. The attached quality trainer does not load these groups, so a separate external-group training adaptation is needed to reproduce the information-ranked MAE experiment.

**Run:** `python scripts/heterogeneity_information_groups.py`

**Retained top-level defaults:**

| Setting | Value |
| --- | --- |
| `N_PIXELS_PER_GROUP` | `16` |
| `N_GROUPS` | `9` |
| `N_DAYS_CAP` | `5` |
| `SV_R2_POOL_THRESHOLD` | `0.95` |
| `STORAGE_DIM` | `(48, 48)` |

## 13. Gaseous temporal vision transformer

[`gas_tvit.py`](../scripts/gas_tvit.py)

Discovers timestamped experiment folders and matching video/reference filenames. It reads the gaseous oxygen field from the instrument response, uses the fixed assumed capture rate, creates byte-encoded resized frame caches and trains the temporal ViT under leave-one-experiment-out evaluation. The optional alignment preview compares reference oxygen and film brightness before extraction.

The companion two-site SV calibration uses all training frames, including transients; this differs from the dedicated equilibrium analyses. Three stacked BGR frames form the default neural input. Its loader uses a five-sample spacing by default. At 10 retained frames/s this corresponds to 0, 0.5 and 1 s lookbacks, subject to experiment-boundary handling.

**Outputs:** `outputs/<run>/gas_tvit/`, including `4_Caches`, inference CSVs, chemometric plots, training histories and a global summary. This entry point does not save trained TViT model weights. It does not apply the autograd script's ROI JSON. Its frame cache can be reused by the equilibrium scripts, but its output is not the missing V14 posthoc multi-model cache.

**Run:** `python scripts/gas_tvit.py`

**Retained top-level defaults:**

| Setting | Value |
| --- | --- |
| `FRAME_STEP_RATE` | `1` |
| `ASSUMED_FPS` | `10.0` |
| `N_EXPERIMENTS_CAP` | `None` |
| `N_STACKED_FRAMES` | `3` |
| `STORAGE_DIM` | `(48, 48)` |
| `BATCH_SIZE` | `32` |
| `N_EPOCHS` | `20` |
| `LR` | `0.0001` |

## 14. Gaseous continuous-time autograd PINN

[`gas_tvit_autograd_pinn.py`](../scripts/gas_tvit_autograd_pinn.py)

Combines a temporal ViT image embedding with an explicit continuous-time input. Two heads predict an inverse-intensity proxy S and oxygen. Autograd differentiates the predicted proxy with respect to the explicit time coordinate; the physics target is `alpha*S + (alpha/k)*dS/dt`. Softplus constrains the learned alpha and k to positive values. Supervised oxygen/proxy losses and the physics residual train the model jointly.

This derivative is with respect to the explicit time input with the supplied image tensor held fixed; it is not automatically the total derivative along an evolving image trajectory. The script applies optional rotation/ROI cropping, stores byte-encoded frames, uses a headless plotting backend and resumes saved model/result pairs. The two-site SV comparator fits all training frames.

**Outputs:** `outputs/<run>/gas_tvit_autograd_pinn/`, including `4_Caches`, `5_Saved_Models`, inference/physics-target CSVs, histories, chemometrics and a global summary. Model state includes the learnable physics parameters. Windows/local users can reduce `num_workers=min(cpu_count(), 4)` in both DataLoader calls to 0 if multiprocessing causes problems; worker settings were not changed in this release.

**Run:** `python scripts/gas_tvit_autograd_pinn.py`

**Retained top-level defaults:**

| Setting | Value |
| --- | --- |
| `FRAME_STEP_RATE` | `1` |
| `ASSUMED_FPS` | `10.0` |
| `N_EXPERIMENTS_CAP` | `None` |
| `N_STACKED_FRAMES` | `3` |
| `STACK_LOOKBACK_STEP` | `5` |
| `STORAGE_DIM` | `(48, 48)` |
| `BATCH_SIZE` | `32` |
| `N_EPOCHS` | `35` |
| `LR` | `0.0001` |
| `LAMBDA_PHY` | `0.5` |
| `LAMBDA_S` | `1.0` |

## 15. Gaseous equilibrium Stern–Volmer analysis

[`gas_equilibrium_stern_volmer.py`](../scripts/gas_equilibrium_stern_volmer.py)

Loads compatible cached scalar frame data or extracts it from raw gas videos. It matches experiments to the embedded region definitions, trims the leading portion of each stable plateau and checks duration, frame count, intensity slope and reference drift. By default, accepted plateau medians calibrate the two-site SV model using all experiments except the test experiment. A separate all-region calibration is fitted for comparison.

It then predicts over the full held-out timeline, optionally joins historical V14 AI predictions, and optionally applies a Kalman filter plus RTS backward smoother. The backward pass uses future observations and is offline postprocessing. T90 is assessed through both smoothed-crossing and raw-aligned approaches; the definitions and successful-event sets should not be pooled indiscriminately.

**Outputs:** numbered equilibrium-selection, calibration, inference, chemometric, response-time and data directories; CSV source tables, an Excel workbook and `run_manifest.json`. Without a V14 cache the classical analysis still runs. For new recordings, update `REGION_DEFINITIONS`; unrecognised experiments are excluded. Equilibrium settings are stricter than merely selecting all samples in a manually labelled stable window.

**Run:** `python scripts/gas_equilibrium_stern_volmer.py`

**Retained top-level defaults:**

| Setting | Value |
| --- | --- |
| `ASSUMED_FPS` | `10.0` |
| `STORAGE_DIM` | `(48, 48)` |
| `EQ_FIT_MODE` | `'plateau_median'` |
| `APPLY_RTS` | `True` |

## 16. Gaseous equilibrium polynomial analysis

[`gas_equilibrium_polynomial.py`](../scripts/gas_equilibrium_polynomial.py)

Uses the same plateau selection, experiment-held-out calibration, reporting and optional historical/RTS comparison structure as the equilibrium SV script, but fits an empirical polynomial mapping `1/I` to oxygen. The retained default is degree 2 with `POLY_FEATURE="inv_i"`; prediction is direct rather than an analytic inversion of a forward intensity equation. An all-region polynomial provides the corresponding legacy comparator.

The model's physical status differs from the two-site SV model even when both fit the same equilibrium points. Degree and feature choice are explicit settings. The script can reuse its own scalar cache, the SV equilibrium script's scalar cache or the gas TViT frame cache; otherwise it extracts raw recordings. It keeps an independent copy of the region definitions, which must remain consistent with the SV version.

**Outputs:** the same types of plateau, calibration, inference, residual, regime-wise performance and T90 exports as the equilibrium SV script, in `outputs/<run>/gas_equilibrium_polynomial/`. Optional V14 AI comparison inputs are external; absent inputs do not trigger training of missing models.

**Run:** `python scripts/gas_equilibrium_polynomial.py`

**Retained top-level defaults:**

| Setting | Value |
| --- | --- |
| `ASSUMED_FPS` | `10.0` |
| `STORAGE_DIM` | `(48, 48)` |
| `EQ_FIT_MODE` | `'plateau_median'` |
| `POLY_DEGREE` | `2` |
| `POLY_FEATURE` | `'inv_i'` |
| `APPLY_RTS` | `True` |
