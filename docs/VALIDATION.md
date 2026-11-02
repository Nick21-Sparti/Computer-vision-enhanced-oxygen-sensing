# Validation of this repository edition

Checked on 2 November 2026.

## Completed checks

| Check | Result |
| --- | --- |
| Uploaded-to-repository inventory | All 16 uploaded Python scripts are represented once, with source hashes and filename mapping |
| Python source parsing | All 19 Python files parse successfully: 16 research scripts, configuration and two utilities |
| Bytecode compilation of research sources | All 16 pass without importing or executing them |
| Executable absolute-path audit | No original Windows drive or cluster-home path literal remains in the research configuration |
| Configuration regression tests | Eight tests pass: working-directory independence, absolute-root overrides, run isolation, invalid labels, optional inputs, output separation, intended heterogeneity sharing, gas-cache routing and manifest checks across the test cases |
| Full-timeline MAE regression | Extracted metric function handles `slice(None)`, a nonempty Boolean mask and an empty mask |
| Kinetic/curriculum pixel conversion | Extracted reshape expressions accept both lists and arrays and preserve all 6,912 uint8 pixel values |
| Documentation links | Local Markdown links resolve within the repository |
| Archive | ZIP integrity and research-file count verified |

Run the lightweight checks from the repository root:

```bash
python tools/check_setup.py --source-only
python -m unittest discover -s tools -p "test_*.py" -v
```

These utilities do not import or run the scientific training scripts. The targeted MAE and pixel-expression checks were run separately with NumPy/Pandas during package preparation. The one-sample `.squeeze(-1)` change was reviewed at source level; it was not tested in a live PyTorch model here.

## Not tested

No experimental video/log files or historical result caches were supplied. PyTorch and OpenCV were not installed in the preparation runtime. Consequently model imports, pretrained-weight downloads, GPU execution, video decoding, instrument-log alignment, full-data memory use, optimisation, neural batch behaviour and numerical agreement with manuscript results were not tested.

`requirements.txt` is derived from imports, not a verified environment lockfile. Installing these requirements is not itself a reproduction test. The preflight utility reports package presence and filename matches only. It cannot validate calibration windows, units, data leakage, prediction accuracy or the completeness of the manuscript's analysis workflow.
