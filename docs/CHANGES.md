# Repository edition changes

## Organisation and path handling

- Renamed all 16 scripts by dataset and analysis; the full mapping is in `FILE_RENAMING.md`.
- Replaced every executable author-machine absolute data path with shared `repo_config.py` lookups. Both Windows drive paths and the cluster home path were removed from executable configuration.
- Added repository-relative defaults, editable external-data locations, optional environment overrides and a run label.
- Separated model-specific outputs and caches that previously shared names. The heterogeneity extraction cache and quality/linear-comparison results remain shared intentionally.
- Connected the gas equilibrium readers to the new gas TViT output location and the polynomial reader to the SV equilibrium scalar-cache location.
- Exposed per-pixel T90, historical aqueous T90, historical gas posthoc, ROI and explicit gas frame-cache locations centrally. Existing filenames and schemas are retained.
- Removed boilerplate instructions addressed to future language models from some source headers; scientific/history comments are otherwise retained. Added a repository-edition pointer to each script.

## Limited execution fixes

- Replaced scalar-regression `.squeeze()` calls with `.squeeze(-1)` where present, preserving the batch dimension when the final batch contains one sample. The gas autograd derivative output receives the same treatment. Normal multi-sample outputs are unchanged.
- Fixed `aqueous_tvit_mass_transfer.py` calling `.any()` on `slice(None)` while reporting whole-timeline MAE.
- Converted fresh Polars-derived pixel lists to NumPy arrays before reshape in the kinetic and curriculum datasets. This preserves pixel values and layout while allowing the first run to consume its newly extracted data.
- In entry points that previously caught and only printed an exception, re-raised it after the traceback so an unsuccessful job returns a failure rather than a misleading success status.

Model architecture, losses, epochs, experiment selection, Optuna placement, early-stopping rules, physics definitions and group construction were not scientifically redesigned. The behaviour qualifications are documented in `REPRODUCIBILITY.md`.

## Traceability

`SOURCE_MANIFEST.json` records the original upload filename and SHA-256 hash, new filename/hash and per-file runtime fixes. `PORTABILITY.patch` is a unified diff against the original source text. It is an audit record, not an extra script to run or a migration command. Historical absolute paths can appear there as removed lines; the executable scripts use the portable configuration.

Original uploads are preserved separately from this repository edition. Neither Word manuscript is copied into the code release.
