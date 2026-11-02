#!/usr/bin/env python3
"""Read-only preflight: syntax, dependencies, discovery and optional artifacts.
Does not import research scripts, open videos, create outputs or start training.
"""
import argparse
import ast
import importlib.metadata
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import repo_config as cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true", help="Check syntax and portable paths without installed ML packages or recordings")
    parser.add_argument("--dataset", choices=["aqueous", "gas", "all"], default="all")
    args = parser.parse_args()
    failures = 0
    sources = sorted((ROOT / "scripts").glob("*.py")) + sorted((ROOT / "tools").glob("*.py"))
    for path in sources:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for n in ast.walk(tree):
                if path.parent.name == "scripts" and isinstance(n, ast.Constant) and isinstance(n.value, str):
                    if re.match(r"^[A-Za-z]:\\", n.value) or n.value.startswith("/myriadfs/"):
                        raise ValueError(f"machine-specific path literal on line {n.lineno}")
        except (SyntaxError, ValueError) as exc:
            failures += 1
            print("FAIL", path.name, exc)
    print(f"Source checks: {len(sources)} Python files, {failures} failures")
    print("Aqueous root:", cfg.data_path("aqueous"))
    print("Gas root:", cfg.data_path("gas"))
    print("Example output:", cfg.output_path("aqueous_tvit"))
    if args.source_only:
        return int(failures > 0)

    # This is a package-presence check, not a binary compatibility or CUDA test.
    packages = ["numpy", "pandas", "scipy", "matplotlib", "seaborn", "scikit-learn", "opencv-python", "pyarrow", "polars", "tqdm", "torch", "torchvision", "timm", "optuna", "openpyxl"]
    for name in packages:
        try:
            print("INSTALLED", name, importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError:
            print("MISSING", name)
            failures += 1
    for medium in ["aqueous", "gas"]:
        if args.dataset not in [medium, "all"]:
            continue
        base = Path(cfg.data_path(medium))
        matched = 0
        if base.is_dir():
            for folder in sorted(base.iterdir()):
                if not folder.is_dir():
                    continue
                if medium == "aqueous":
                    if "-" not in folder.name:
                        continue
                    rois = [p for p in folder.glob("*_ROI_Output") if p.is_dir()]
                    videos = [p for r in rois for p in r.glob("*.mp4")]
                    logs = list(folder.rglob("*_arduino_*.txt"))
                    temperatures = list(folder.rglob("*temperature*.csv"))
                    valid = bool(videos and logs and temperatures)
                else:
                    m = re.search(r"\d{8}_\d{6}", folder.name)
                    if not m:
                        continue
                    videos = [p for p in folder.glob("*.mp4") if m[0] in p.name]
                    logs = [p for p in folder.glob("*.csv") if m[0] in p.name]
                    valid = bool(videos and logs)
                print("MATCH" if valid else "INCOMPLETE", folder.name)
                matched += int(valid)
                if valid and (len(videos) > 1 or len(logs) > 1):
                    print("  NOTE: multiple candidates; legacy loaders may select the first. Keep only the intended recording/log pair.")
        print(f"{medium}: {matched} folders match the main loader filename conventions")
        if matched < 2:
            print("  FAIL: experiment-wise cross-validation needs at least two valid recordings.")
            failures += 1
    for label in ["pixel_t90", "aqueous_t90_reports", "gas_posthoc_cache", "gas_roi", "gas_frame_cache"]:
        value = cfg.external_path(label)
        print("OPTIONAL", label, "present" if value and Path(value).exists() else "absent / automatic", value)
    print("No videos or logs were parsed. This check cannot validate units, alignment, ROI, schema or model execution.")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
