from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Common-basis, reproducible analysis of 3D ECT sensitivity matrices."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("simpat_results"))
    parser.add_argument(
        "--files",
        nargs="*",
        type=Path,
        help="Explicit mesh files. If omitted, meshA.mat ... meshO.mat are used.",
    )
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--mesh-unit", choices=("mm", "m"), default="mm")
    parser.add_argument(
        "--sensitivity-kind",
        choices=("element-integrated", "density"),
        default="density",
        help=(
            "element-integrated: each native row already contains the element integral; "
            "density: multiply each row by tetrahedral volume before aggregation."
        ),
    )
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--grid-r", type=int, default=10)
    parser.add_argument("--grid-theta", type=int, default=16)
    parser.add_argument("--grid-z", type=int, default=20)
    parser.add_argument("--score-noise-std", type=float, default=1.0)
    parser.add_argument("--alpha-rel", type=float, default=1e-3)
    parser.add_argument(
        "--alpha-sweep",
        nargs="+",
        type=float,
        default=(1e-4, 1e-3, 1e-2),
        help="Relative regularization values for robustness analysis.",
    )
    parser.add_argument("--subset-k", type=int, default=128)
    parser.add_argument(
        "--analyses",
        default="global,groups,subsets,phantoms",
        help="Comma-separated subset of: global,groups,subsets,phantoms.",
    )
    parser.add_argument("--phantom-radius-mm", type=float, default=20.0)
    parser.add_argument("--phantom-r-samples", type=int, default=3)
    parser.add_argument("--phantom-theta-samples", type=int, default=4)
    parser.add_argument("--phantom-z-samples", type=int, default=5)
    parser.add_argument("--phantom-noise-repeats", type=int, default=5)
    parser.add_argument(
        "--phantom-noise-rel",
        type=float,
        default=0.01,
        help="Noise sigma as a fraction of RMS noiseless measurement per phantom.",
    )
    parser.add_argument(
        "--phantom-lambda-rel",
        type=float,
        default=0.01,
        help="Ridge lambda divided by the largest singular value of each common-basis S.",
    )
    parser.add_argument("--phantom-threshold", type=float, default=0.5)
    parser.add_argument("--phantom-min-bins", type=int, default=2)
    parser.add_argument(
        "--phantom-subsamples-per-axis",
        type=int,
        default=3,
        help="Partial-volume quadrature resolution; 3 means 27 samples per common bin.",
    )
    parser.add_argument(
        "--phantom-min-volume-ratio",
        type=float,
        default=0.80,
        help=(
            "Minimum represented/ideal sphere-volume ratio required for a target "
            "to enter the paired phantom benchmark."
        ),
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Target-level bootstrap samples used for 95%% confidence intervals.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--figure-dpi", type=int, default=300)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)

def validate_args(args: argparse.Namespace) -> set[str]:
    requested = {x.strip().lower() for x in args.analyses.split(",") if x.strip()}
    allowed = {"global", "groups", "subsets", "phantoms"}
    unknown = requested - allowed
    if unknown:
        raise ValueError(f"Unknown analyses: {sorted(unknown)}")
    positive = {
        "grid-r": args.grid_r,
        "grid-theta": args.grid_theta,
        "grid-z": args.grid_z,
        "score-noise-std": args.score_noise_std,
        "alpha-rel": args.alpha_rel,
        "subset-k": args.subset_k,
        "phantom-radius-mm": args.phantom_radius_mm,
        "phantom-r-samples": args.phantom_r_samples,
        "phantom-theta-samples": args.phantom_theta_samples,
        "phantom-z-samples": args.phantom_z_samples,
        "phantom-noise-repeats": args.phantom_noise_repeats,
        "phantom-lambda-rel": args.phantom_lambda_rel,
        "phantom-subsamples-per-axis": args.phantom_subsamples_per_axis,
        "bootstrap-samples": args.bootstrap_samples,
        "figure-dpi": args.figure_dpi,
    }
    bad = [name for name, value in positive.items() if value <= 0]
    if bad:
        raise ValueError(f"Parameters must be positive: {bad}")
    if args.phantom_noise_rel < 0:
        raise ValueError("phantom-noise-rel must be non-negative")
    if not 0 < args.phantom_threshold <= 1:
        raise ValueError("phantom-threshold must be in (0, 1]")
    if not 0 < args.phantom_min_volume_ratio <= 1:
        raise ValueError("phantom-min-volume-ratio must be in (0, 1]")
    if any(v <= 0 for v in args.alpha_sweep):
        raise ValueError("All alpha-sweep values must be positive")
    return requested

def label_from_path(path: Path) -> str:
    match = re.search(r"mesh([A-O])$", path.stem, flags=re.IGNORECASE)
    return match.group(1).upper() if match else path.stem

def discover_files(args: argparse.Namespace) -> list[Path]:
    if args.files:
        paths = [p if p.is_absolute() else args.input_dir / p for p in args.files]
    else:
        paths = [args.input_dir / f"mesh{chr(65 + i)}.mat" for i in range(15)]
    existing = [p.resolve() for p in paths if p.exists()]
    missing = [str(p) for p in paths if not p.exists()]
    if missing and not args.allow_missing:
        raise FileNotFoundError(
            "Missing input files. Supply all A--O files or use --allow-missing:\n  "
            + "\n  ".join(missing)
        )
    if not existing:
        raise FileNotFoundError("No input .mat files were found")
    labels = [label_from_path(p) for p in existing]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Duplicate configuration labels: {labels}")
    return sorted(existing, key=lambda p: label_from_path(p))
