# Reproducibility and coverage

## Scope of this release

This package contains the 16 supplied Python scripts, adapted for portable paths and a small number of traceable runtime fixes. The descriptions were checked against those scripts and the supplied main manuscript and supporting information. Raw recordings, acquisition settings, reference logs, trained model weights, T90 pixel caches, historical prediction caches and the original software environment were not attached.

The package has been checked at source/configuration level. Full model training, scientific accuracy, CUDA operation, timing alignment and exact reproduction of manuscript numbers remain untested. Historical version labels in source comments are retained for traceability; a label such as “V11” is not evidence that this file is identical to a published analysis version.

## Evaluation rules in the supplied implementations

| Analysis | What the code does | Consequence for interpretation |
| --- | --- | --- |
| Main aqueous TViT, finite-difference mass-transfer, multitask and main gas TViT | Holds out one experiment; trains a fixed number of epochs; records held-out MAE | Monitoring does not itself select a checkpoint in these loops, but the displayed test curve should not be used to tune later choices |
| Bayesian, kinetic and curriculum variants | Runs Optuna once using the full dataset before the outer experiment loop; uses a sampled subset and only the first three-group split | Outer test experiments are not isolated from hyperparameter selection; these results should not be described as strictly nested LOOCV |
| Optuna temporal subsets | Randomly samples/shuffles rows; dataset stacks by row position without restoring chronological neighbourhoods | Tuning inputs may no longer represent the intended consecutive temporal lookbacks |
| Quality-group and group-size TViT | Uses held-out-day MAE for early stopping and best-weight restoration | The held-out experiment doubles as validation; the final score is not a wholly untouched test estimate |
| Quality-group checkpoint switch | Best weights are restored even if `USE_EARLY_STOPPING=False` | Turning off early termination alone does not make the result final-epoch evaluation |
| Heterogeneity feature/group construction | Computes features and group choices over selected experiment days before the outer folds | Held-out data can influence pixel selection; fold-wise model fitting alone does not remove this selection dependence |
| Linear group comparator | Selects ridge regularisation within outer-training days, but can reuse global pixel definitions | Its regression tuning has an inner training-day split; group selection still has the qualification above |
| Dynamic benchmark | Fits calibration and dynamic parameters on outer-training data; optional neural models use fixed epochs | This code keeps the outer test out of those fit/search operations, but a training-error parameter search is not automatically inner cross-validation |
| Gas equilibrium scripts | Fits calibration using other experiments; relies on supplied, experiment-specific region annotations; optional RTS uses future observations | Separate equilibrium calibration, posthoc annotations and offline smoothing from claims of autonomous causal deployment |

These behaviours were preserved, not silently repaired as part of packaging. A stricter reanalysis would need training-only pixel selection, tuning nested inside each outer fold, chronological temporal inputs, and a separate validation split or fixed final epoch. Such changes can alter the reported results and should be treated as a new scientific analysis.

## Default settings versus manuscript settings

The supplied files are not all configured for the same cohort or training budget:

- The quality-group trainer uses five days, eight 16-pixel groups and 15 epochs with early stopping; the linear comparator uses a 13-day cap. Match these settings for direct comparisons.
- The group-size script selects experiment numbers `[11, 12, 13, 14, 15]`, with group side lengths 2 through 8. Check its printed resolved IDs and exclusions.
- The information builder selects five days and nine 16-pixel groups; it only constructs definitions.
- The curriculum file defaults to three frames and 25 epochs, with auxiliary weights removed at epoch index 18. It does not automatically reproduce the six-frame variant discussed in the SI.
- Several aqueous constructors are called without explicit `stack_size` and/or `step`; the retained defaults are three frames and a five-sample spacing even though the global `FRAME_STACK_STEP` says 10. Specifically inspect the main loader and Optuna loader calls in the main TViT, multitask, Bayesian, kinetic and curriculum files before changing temporal settings. The mass-transfer and heterogeneity trainers pass their configured spacing explicitly.
- The main gas and gas autograd scripts use all-region SV calibration. The gas equilibrium scripts trim and screen plateaus. These baselines answer different calibration questions and should not be assigned the same label without qualification.
- Aqueous reference values are intended as µmol/L and gas reference values as % V/V. The scripts parse a selected raw field divided by 1000; they do not identify units automatically. Some inherited aqueous plot labels say “%”. Numeric conversion has not been changed to match those labels.

## Causal predictions and offline processing

Past-frame stacking supports causal model input when rows are chronologically ordered within each experiment. However, this alone does not make every analysis causal:

- Regularised FFT deconvolution uses the full series.
- RTS smoothing includes a backward pass and uses future observations.
- Savitzky–Golay/median smoothing and reference-aligned T90 event analysis are posthoc measurements, not latency-free online estimators.
- Label interpolation and nearest-time reference matching describe offline dataset construction. They do not establish an acquisition-time latency guarantee.
- Auxiliary future-oxygen targets are supervised training labels. They are not future camera frames, but their evaluation still requires available future reference data.

Report raw and postprocessed predictions separately. T90 values depend on event definitions, reference alignment, thresholds, smoothing and which events pass quality checks. Do not treat Savitzky–Golay and raw-aligned outputs as interchangeable measurements.

## Specific implementation qualifications

1. **Finite-difference physics:** the aqueous mass-transfer loss uses channels 0 and 3 of a BGR stack and an unnormalised inter-frame proxy difference. Its learned parameters should not be interpreted directly as measured physical diffusivity. Its cached network weights do not retain the separate loss-module parameters.
2. **Autograd physics:** the gas model differentiates its proxy output with respect to an explicit time input. This is a partial derivative at fixed image input, not automatically the total sensor-trajectory derivative.
3. **Uncertainty:** heteroscedastic variance and MC-dropout variance are model outputs requiring calibration checks. They are not guaranteed coverage intervals. The domain-classification accuracy on an entirely unseen class is not a standalone test of invariance.
4. **T90 fallback:** missing pixel T90 files can drop T90 from the group feature space. The run can finish successfully while performing a different analysis.
5. **Information filtering:** if the pixel-feature file is missing, the information builder can use all pixels despite the requested SV filter. Confirm its printed pool size.
6. **Frozen backbones:** quality/group-size studies freeze much of the model by default. Their training regime differs from the fully trainable main TViT.
7. **One-day fallback:** some group trainers explicitly train/test on the same day when only one is available. This is a smoke test, not generalisation evidence.
8. **External cache identity:** filenames do not prove compatible ROI, sampling, units, cohort or preprocessing. Several caches and summary-based resume mechanisms omit parts of the configuration from their keys.
9. **Reproducibility of randomness:** seed 42 is retained in many scripts, but CUDA benchmarking/nondeterministic kernels and successive model initialisations can still change results. No claim of bitwise reproducibility is made.
10. **Fallback training data:** if the linear comparator finds no training frames in the stable windows, its weighted regression falls back to the test mask. Confirm that training plateaus exist; otherwise that branch is calibrated on the evaluation experiment.
11. **Gas timestamp joining:** the main gas TViT joins the combined frame and reference tables by timestamp without an experiment key. This assumes recordings do not overlap in absolute time; the autograd implementation explicitly joins by experiment as well.
12. **Historical comments:** some comments describe earlier versions, intended rather than implemented workflows, or stronger scientific claims than the code establishes. The script guide distinguishes the active defaults and behaviour.

## Coverage of the manuscript

| Study component | Coverage in this bundle |
| --- | --- |
| Aqueous TViT and selected auxiliary-loss variants | Included |
| Static aqueous calibration comparisons | Included |
| Dynamic inverse, causal Kalman and optional CNN temporal comparisons | Included in the dynamic benchmark; CNN/TViT switches default off |
| Quality-based heterogeneity and linear comparisons | Included, with evaluation/default qualifications above |
| Pixel-group size comparison | Included |
| Effective-rank group construction | Included; downstream TViT training on those groups is not implemented in the attached trainer |
| Gas TViT and continuous-time autograd PINN | Included |
| Gas equilibrium SV/polynomial and their own RTS/T90 reporting | Included |
| Full V14 posthoc multi-model prediction aggregation | Not included; optional caches can be read but their original generator is absent |
| Per-pixel T90/spatial map production | Not included; existing arrays are consumed by heterogeneity analyses |
| T90-only and mixed-Q/T90 group training experiments | Not included in the attached set |
| Microscopy registration and thickness/performance correlation | Not included |
| Combined aqueous/gas model with common-unit conversion | Not included |
| Dedicated aqueous autograd experiment and all static/six-frame ablation configurations | Not supplied as complete standalone reproduction entry points |
| Final manuscript composite figures and complete bootstrap/statistical-report pipeline | Not included as a complete reproduction workflow |

A complete release supporting every figure would need those missing scripts or a precise mapping to equivalent implementations, the data/configuration versions used, and the relevant derived inputs.

## Recording a reproducible run

Use a distinct `RUN_NAME` and record the code commit, script, data identifiers, unit/field definitions, ROI, time synchronisation, included/excluded experiments, training settings, group settings, hardware and software versions. After a successful installation, `python -m pip freeze > requirements-lock-local.txt` records that local environment; it is not evidence of the authors' original environment. Archive this file and the corresponding results together.

For comparisons, match experiment IDs, selected pixel definitions, temporal context, calibration regime and input units, not only filenames or model names. Retain individual-fold metrics and event counts alongside aggregate means. Rerunning with a new configuration should use a new output/cache namespace rather than silently reuse a completed summary.
