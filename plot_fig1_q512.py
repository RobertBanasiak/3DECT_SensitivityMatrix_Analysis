#!/usr/bin/env python3
"""Generate manuscript Figure 1 from existing Q512 results, without FEM I/O.

Place beside main.py in the existing 3DECT_SensitivityMatrix_Analysis project:
    python plot_fig1_q512.py

Automatically uses the latest complete primary Q512 run and the latest
compatible mapping study containing Q512 spectra. The raw Gram eigenvalues
come from mapping_gram_spectra.csv, case grid_10x16x20_Q512, scope
centroid_common_support. Normalisation uses the FULL-PRECISION primary alpha
from run_manifest.json, not the older fixed alpha of the mapping study.

Input hashes, mapping parameters, fixed-domain mask, and all 15 spectra are
checked against the primary run. No .mat files are loaded or recomputed.
The original project plotting function preserves the manuscript figure style.

Optional explicit paths:
    python plot_fig1_q512.py --primary-run PATH --mapping-run PATH
    python plot_fig1_q512.py --output-dir PATH --dpi 600

Relative paths are resolved from this script's directory. New outputs:
    q512_figures/run_*/fig1_Q512_singular_spectra.png
    q512_figures/run_*/fig1_Q512_singular_spectra.pdf
    q512_figures/run_*/fig1_Q512_singular_spectra.svg
    q512_figures/run_*/fig1_Q512_plot_data.csv
    q512_figures/run_*/fig1_Q512_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIRECTORY = Path(__file__).resolve().parent
LABELS = list("ABCDEFGHIJKLMNO")
CASE = "grid_10x16x20_Q512"
SCOPE = "centroid_common_support"
FIGURE_STEM = "fig1_Q512_singular_spectra"


def resolve_path(path):
    path = Path(path)
    return (path if path.is_absolute() else PROJECT_DIRECTORY / path).resolve()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def candidates(root, explicit):
    if explicit is not None:
        directory = resolve_path(explicit)
        if not directory.is_dir():
            raise FileNotFoundError(f"Result directory does not exist: {directory}")
        return [directory]
    if not root.is_dir():
        raise FileNotFoundError(f"Result directory does not exist: {root}")
    return sorted((p for p in root.glob("run_*") if p.is_dir()),
                  key=lambda p: p.name, reverse=True)


def input_signature(manifest):
    return sorted((item["label"], item["sha256"], item["n_cells"], item["n_channels"])
                  for item in manifest["inputs"])


def load_primary(directory):
    revision = read_json(directory / "q512_revision_manifest.json")
    manifest = read_json(directory / "run_manifest.json")
    if not revision.get("complete") or revision.get("error"):
        raise ValueError("Primary Q512 analysis did not complete successfully")
    policy = manifest["mapping"]
    if policy["quadrature_points_per_tetra"] != 512 or policy["fixed_basis_parameters"] != 2851:
        raise ValueError("Expected primary Q512 mapping on the fixed 2851-bin domain")
    if policy["basis_policy"] != "frozen_original_centroid_common_evaluation_domain":
        raise ValueError("Primary run uses a different evaluation-domain policy")
    grid = manifest["common_grid"]
    if (grid["nr"], grid["ntheta"], grid["nz"]) != (10, 16, 20):
        raise ValueError("Expected the primary 10x16x20 grid")
    if grid["common_occupied_bins"] != 2851:
        raise ValueError("Primary parameter basis has changed")
    alpha = float(manifest["regularization"]["alpha_absolute_primary"])
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Primary alpha must be finite and positive")
    score = pd.read_csv(directory / "tables" / "global_common_basis_scorecard.csv",
                        float_precision="round_trip")
    if len(score) != 15 or sorted(score.configuration.tolist()) != LABELS:
        raise ValueError("Expected exactly one primary scorecard row for each A--O configuration")
    if not ((score.channels == 496) & (score.common_bins == 2851)).all():
        raise ValueError("Primary scorecard dimensions do not match the manuscript")
    if not np.allclose(score.alpha_common, alpha, rtol=1e-9, atol=1e-9):
        raise ValueError("Primary scorecard and manifest use different alpha values")
    return manifest, score.set_index("configuration"), alpha


def find_primary(explicit):
    rejected = []
    root = PROJECT_DIRECTORY / "q512_revision_results"
    for directory in candidates(root, explicit):
        try:
            return directory, *load_primary(directory)
        except (OSError, ValueError, KeyError) as exc:
            if explicit is not None:
                raise
            rejected.append(f"{directory.name}: {exc}")
    raise FileNotFoundError("No complete primary Q512 run found. "
                            "Run run_q512_revision.py first or supply --primary-run. "
                            + " | ".join(rejected[:3]))


def load_mapping_spectra(directory, primary, score, alpha):
    manifest = read_json(directory / "mapping_manifest.json")
    if not manifest.get("complete") or manifest.get("error"):
        raise ValueError("Mapping study did not complete successfully")
    if input_signature(manifest) != input_signature(primary):
        raise ValueError("Mapping study and primary run use different FEM input hashes")
    for key in ("mesh_unit", "sensitivity_kind", "dtype", "score_noise_std",
                "grid_r", "grid_theta", "grid_z"):
        if manifest["source_parameters"][key] != primary["parameters"][key]:
            raise ValueError(f"Incompatible mapping parameter: {key}")
    support = pd.read_csv(directory / "mapping_common_support.csv", float_precision="round_trip")
    support = support[support.grid == "10x16x20"].sort_values("bin_index_zero_based")
    if len(support) != 3200 or not np.array_equal(support.bin_index_zero_based, np.arange(3200)):
        raise ValueError("Mapping study does not contain the complete primary grid mask")
    if support.in_centroid_common_support.dtype.kind != "b":
        raise ValueError("Expected Boolean common-domain membership values")
    mask = support.in_centroid_common_support.to_numpy(dtype=np.uint8)
    mask_digest = hashlib.sha256(mask.tobytes()).hexdigest()
    if mask_digest != primary["mapping"]["fixed_basis_mask_sha256"]:
        raise ValueError("Mapping study and primary run use different 2851-bin domains")
    frame = pd.read_csv(directory / "mapping_gram_spectra.csv", float_precision="round_trip")
    frame = frame[(frame["case"] == CASE) & (frame.scope == SCOPE)]
    if len(frame) != 15 * 496 or sorted(frame.configuration.unique()) != LABELS:
        raise ValueError("Mapping study does not contain all 15 Q512 common-domain spectra")
    eigenvalues_by_label, rows = {}, []
    for label in LABELS:
        values = frame[frame.configuration == label].sort_values("mode_rank_descending")
        ranks = values.mode_rank_descending.to_numpy()
        if not np.array_equal(ranks, np.arange(1, 497)):
            raise ValueError(f"{label}: missing or duplicated spectral modes")
        descending = values.gram_eigenvalue_raw.to_numpy(dtype=float)
        if not np.isfinite(descending).all() or np.any(descending < 0):
            raise ValueError(f"{label}: invalid Gram eigenvalues")
        if np.any(np.diff(descending) > 0):
            raise ValueError(f"{label}: eigenvalues are not ordered by descending magnitude")
        independent = {
            "spectral_energy": float(descending.sum()),
            "largest_eigenvalue": float(descending[0]),
            "regularized_logdet": float(np.log1p(descending / alpha).sum()),
        }
        for field, expected in independent.items():
            if not np.isclose(expected, score.loc[label, field], rtol=1e-9, atol=1e-9):
                raise ValueError(f"{label}: exported spectrum disagrees with primary {field}")
        # The original plotting function expects eigenvalues in ascending order.
        eigenvalues_by_label[label] = descending[::-1].copy()
        rows.append(pd.DataFrame({
            "configuration": label, "mode_rank_descending": ranks,
            "gram_eigenvalue_raw": descending,
            "singular_value": np.sqrt(descending),
            "alpha_primary": alpha,
            "normalized_singular_value": np.sqrt(descending / alpha),
        }))
    return eigenvalues_by_label, pd.concat(rows, ignore_index=True)


def find_mapping(explicit, primary, score, alpha):
    rejected = []
    root = PROJECT_DIRECTORY / "mapping_sensitivity_results"
    for directory in candidates(root, explicit):
        try:
            spectra, data = load_mapping_spectra(directory, primary, score, alpha)
            return directory, spectra, data
        except (OSError, ValueError, KeyError) as exc:
            if explicit is not None:
                raise
            rejected.append(f"{directory.name}: {exc}")
    raise FileNotFoundError("No compatible Q512 mapping spectra found. "
                            "Supply --mapping-run pointing to the Q512 mapping study. "
                            + " | ".join(rejected[:3]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--primary-run", type=Path, help="Directory containing run_manifest.json from a primary Q512 run")
    parser.add_argument("--mapping-run", type=Path, help="Directory containing mapping_gram_spectra.csv and its manifest")
    parser.add_argument("--output-dir", type=Path, default=Path("q512_figures"))
    parser.add_argument("--dpi", type=int, default=None)
    args = parser.parse_args(argv)
    if not (PROJECT_DIRECTORY / "ect_analysis" / "plotting.py").is_file():
        raise RuntimeError("Place plot_fig1_q512.py beside main.py in the existing project")
    primary_dir, primary, score, alpha = find_primary(args.primary_run)
    mapping_dir, spectra, data = find_mapping(args.mapping_run, primary, score, alpha)
    dpi = args.dpi if args.dpi is not None else int(primary["parameters"]["figure_dpi"])
    if dpi <= 0:
        raise ValueError("Figure DPI must be positive")
    output = resolve_path(args.output_dir) / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    print(f"Primary Q512 run: {primary_dir}", flush=True)
    print(f"Q512 mapping spectra: {mapping_dir}", flush=True)
    print(f"Shared primary alpha: {alpha:.17g}", flush=True)
    print("Validated 15 configurations, 496 modes each, on the same 2851-bin domain.", flush=True)
    from ect_analysis.plotting import plot_singular_spectra
    import matplotlib
    files = []
    for extension in ("png", "pdf", "svg"):
        path = output / f"{FIGURE_STEM}.{extension}"
        plot_singular_spectra(spectra, alpha, path, dpi)
        files.append(path.name)
    data_path = output / "fig1_Q512_plot_data.csv"
    data.to_csv(data_path, index=False, float_format="%.17g")
    sources = [primary_dir / "run_manifest.json",
               primary_dir / "tables" / "global_common_basis_scorecard.csv",
               mapping_dir / "mapping_manifest.json",
               mapping_dir / "mapping_common_support.csv",
               mapping_dir / "mapping_gram_spectra.csv",
               PROJECT_DIRECTORY / "ect_analysis" / "plotting.py", Path(__file__)]
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True, "primary_run": str(primary_dir), "mapping_run": str(mapping_dir),
        "spectrum_case": CASE, "spectrum_scope": SCOPE,
        "normalization": "sqrt(raw_Gram_eigenvalue / primary_alpha)",
        "alpha_primary": alpha, "mapping_study_alpha_not_used": True,
        "basis_parameters": 2851, "configurations": LABELS, "modes_per_configuration": 496,
        "dpi": dpi, "figure_files": files, "plot_data_file": data_path.name,
        "source_sha256": {str(path): sha256_file(path) for path in sources},
        "software": {"numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "note": "Existing Q512 spectra plotted with the primary run's alpha; no FEM mapping or native input reads.",
    }
    (output / "fig1_Q512_manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Figure files saved in: {output}", flush=True)
    print("FIGURE 1 Q512 GENERATED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
