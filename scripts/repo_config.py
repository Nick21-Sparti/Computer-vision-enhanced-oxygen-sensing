"""Shared local paths. Edit the values below, then run an analysis in your IDE.
Relative paths resolve from the repository root, never the working directory.
No directories are created by importing this module.
"""
from pathlib import Path
import os

# ===================== EDIT THESE VALUES FOR YOUR COMPUTER =====================
# Keep these defaults if you put recordings inside the repository's data folder.
# Or use absolute Windows paths, e.g. r"D:\oxygen_study\aqueous".
AQUEOUS_DATA_DIR = "data/aqueous"
GAS_DATA_DIR = "data/gas"
OUTPUT_ROOT = "outputs"
CACHE_ROOT = "cache"

# Use a NEW label after changing data, ROI, timings, model settings or group rules.
# This separates both outputs and caches. Related scripts must use the same label.
RUN_NAME = "default"

# Previously computed inputs. These are NOT supplied in the repository.
PIXEL_T90_DIR = "data/derived/pixel_t90"
AQUEOUS_T90_REPORTS_DIR = "data/derived/aqueous_t90_reports"
GAS_POSTHOC_CACHE_DIR = "data/derived/gas_posthoc_cache"
# None: look for global_roi_settings.json in GAS_DATA_DIR; absent => full frame.
GAS_ROI_FILE = None
# None: equilibrium scripts discover caches or extract raw recordings themselves.
# Set a full path only if you already have a compatible frame-level Parquet file.
GAS_FRAME_CACHE_FILE = None
# ==============================================================================

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _resolve(value):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _setting(env, default):
    return _resolve(os.environ.get(env) or default)


def _run_name():
    name = os.environ.get("OXYGEN_RUN_NAME", RUN_NAME)
    if not name or name in {".", ".."} or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("RUN_NAME must use letters, digits, underscores or hyphens only.")
    return name


def data_path(medium, *parts):
    """Return the raw-data root (aqueous or gas), optionally with a suffix."""
    settings = {
        "aqueous": ("OXYGEN_AQUEOUS_DATA", AQUEOUS_DATA_DIR),
        "gas": ("OXYGEN_GAS_DATA", GAS_DATA_DIR),
    }
    env, default = settings[medium]
    return str(_setting(env, default).joinpath(*parts))


def output_path(analysis, *parts):
    return str(_setting("OXYGEN_OUTPUT_ROOT", OUTPUT_ROOT).joinpath(_run_name(), analysis, *parts))


def cache_path(analysis, *parts):
    return str(_setting("OXYGEN_CACHE_ROOT", CACHE_ROOT).joinpath(_run_name(), analysis, *parts))


def external_path(kind):
    """Return a configured derived input; gas_frame_cache may return None."""
    settings = {
        "pixel_t90": ("OXYGEN_PIXEL_T90_DIR", PIXEL_T90_DIR),
        "aqueous_t90_reports": ("OXYGEN_AQUEOUS_T90_REPORTS", AQUEOUS_T90_REPORTS_DIR),
        "gas_posthoc_cache": ("OXYGEN_GAS_POSTHOC_CACHE", GAS_POSTHOC_CACHE_DIR),
        "gas_roi": ("OXYGEN_GAS_ROI_FILE", GAS_ROI_FILE or data_path("gas", "global_roi_settings.json")),
        "gas_frame_cache": ("OXYGEN_GAS_FRAME_CACHE", GAS_FRAME_CACHE_FILE),
    }
    env, default = settings[kind]
    value = os.environ.get(env) or default
    return str(_resolve(value)) if value is not None else None


if __name__ == "__main__":
    print("Repository:", REPOSITORY_ROOT)
    print("Aqueous recordings:", data_path("aqueous"))
    print("Gas recordings:", data_path("gas"))
    print("Run name:", _run_name())
    print("Example output:", output_path("aqueous_tvit"))
    print("Example cache:", cache_path("aqueous_tvit", "features"))
    for key in ("pixel_t90", "aqueous_t90_reports", "gas_posthoc_cache", "gas_roi", "gas_frame_cache"):
        print(key + ":", external_path(key))
