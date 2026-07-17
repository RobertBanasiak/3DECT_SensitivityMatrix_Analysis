from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ect_analysis.pipeline import run_analysis


# =============================================================================
# PYCHARM CONFIGURATION -- EDIT THIS BLOCK, THEN PRESS RUN
# =============================================================================
# Relative paths are resolved from this main.py file, independently of the
# working directory configured in PyCharm. Absolute Windows paths also work,
# for example Path(r"D:\ECT_project\matrices").
PROJECT_DIRECTORY = Path(__file__).resolve().parent

PYCHARM_CONFIG = {
    # Input/output
    "input_dir": PROJECT_DIRECTORY / "matrices",
    "output_dir": PROJECT_DIRECTORY / "SIMPAT_results",
    "files": None,                 # None -> meshA.mat ... meshO.mat
    "allow_missing": False,        # publication run should contain all A--O

    # Geometry and native sensitivity convention
    "mesh_unit": "mm",            # "mm" or "m"
    # The supplied A--O matrices behave as cell-centred sensitivity-density
    # samples (their raw magnitude scales with cell count), hence "density".
    "sensitivity_kind": "density",  # use "element-integrated" only for cell integrals
    "dtype": "float64",

    # Shared equal-volume cylindrical basis (r, theta, z)
    "grid_r": 10,
    "grid_theta": 16,
    "grid_z": 20,

    # Global and subset metrics
    "score_noise_std": 1.0,
    "alpha_rel": 1e-3,
    "alpha_sweep": (1e-4, 1e-3, 1e-2),
    "subset_k": 128,
    "analyses": "global,groups,subsets,phantoms",

    # Paired spherical-phantom benchmark
    "phantom_radius_mm": 20.0,
    "phantom_r_samples": 3,
    "phantom_theta_samples": 4,
    "phantom_z_samples": 5,
    "phantom_noise_repeats": 5,
    "phantom_noise_rel": 0.01,
    "phantom_lambda_rel": 0.01,
    "phantom_threshold": 0.5,
    "phantom_min_bins": 2,
    "phantom_subsamples_per_axis": 3,
    "phantom_min_volume_ratio": 0.80,
    "bootstrap_samples": 2000,

    # Reproducibility and output
    "seed": 123,
    "figure_dpi": 300,
    "self_test": False,            # True -> test installation without .mat files
}

# Keep True for normal PyCharm use. Explicit command-line arguments still take
# precedence, which preserves --help and --self-test for automated checks.
USE_PYCHARM_CONFIG = True


def pycharm_arguments() -> argparse.Namespace:
    """Create an isolated argument namespace from the editable block above."""
    return argparse.Namespace(**dict(PYCHARM_CONFIG))


def main() -> int:
    use_pycharm = USE_PYCHARM_CONFIG and len(sys.argv) == 1
    configured_args = pycharm_arguments() if use_pycharm else None
    return run_analysis(
        args_override=configured_args,
        entrypoint_path=Path(__file__),
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
