#!/usr/bin/env python3
"""Cross-mesh reconstruction experiment for Reviewer 1, Comment 7.

Place beside main.py, run_q512_revision.py, run_q512_selection_revision.py,
analyze_mapping_sensitivity.py and the original ect_analysis package.

    python run_q512_model_mismatch.py --self-test
    python run_q512_model_mismatch.py

Uses main.py's PYCHARM_CONFIG and the latest complete, full-precision primary
Q512 run in q512_revision_results. --primary-run PATH selects it explicitly;
--input-dir PATH and --output-dir PATH override directories without editing
the original project. Relative paths are resolved beside this script.

The existing A/B/C matrices describe the same nominal sensor geometry on
different native FEM meshes. All nine ordered forward/inverse combinations
are evaluated on the original frozen 2851-bin basis using Q512 mapping.
The three diagonal combinations are matched controls; the other six use
different forward and inverse FEM sensitivity operators. The same 40 targets,
normalised Tikhonov rule, threshold and paired Gaussian draws are retained.
The diagonal iid cases must reproduce the primary reconstruction results.

Four noise scenarios: noiseless (one realisation), iid (five realisations),
and shared-electrode correlated noise with rho=.25 or .5 (five each).
R=(1-rho)I+rho BB.T/2, where B is the channel/electrode incidence matrix.
R has unit diagonal. All inversions retain the original unweighted Tikhonov
rule: correlated scenarios deliberately perturb its assumed noise model.
Noise is scaled by 1% of each forward target's clean measurement RMS, as in
the primary experiment. At fixed forward model/noise scenario, all inverses
receive exactly the same measurement array. No parameters are retuned.

This is a linear cross-mesh numerical test using existing FEM matrices and
the shared piecewise-constant parameterisation. It is NOT an independent
nonlinear forward solve, a new FEM calculation, an electrode-position test,
experimental validation, or a comparison of the 128-channel subsets.

Each execution creates a new q512_model_mismatch_results/run_* directory.
Default dimensions: 36 cases, 5760 raw reconstructions, 1440 target records.
Summaries first take the median of noise repetitions for each target, then
the median across targets. Paired changes use the matched inverse for the
SAME forward model/noise scenario as reference. Their percentile bootstrap
intervals resample targets, not individual noise realisations. Failed targets
are retained and counted; incomplete targets are excluded from numerical
summaries and their number is reported explicitly. Results are descriptive.
CSV export uses 17 significant digits and is checked by exact read-back.
No source, native matrix or previous result is overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.linalg import cho_factor, cho_solve, cholesky

PROJECT_DIRECTORY = Path(__file__).resolve().parent
LABELS = ("A", "B", "C")
METRICS = (
    "localization_error_mm", "relative_volume_error_pct", "auc",
    "dice_50pct_peak", "nrmse", "weighted_correlation",
)
CASE_KEYS = ["forward_configuration", "inverse_configuration", "noise_scenario"]
SCENARIOS = (
    ("noiseless", None), ("iid", 0.0),
    ("shared_electrode_rho_0.25", 0.25),
    ("shared_electrode_rho_0.50", 0.50),
)


def local_path(path):
    return (path if path.is_absolute() else PROJECT_DIRECTORY / path).resolve()


def modules():
    try:
        import run_q512_revision as q512
        import run_q512_selection_revision as reference
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Place this script beside main.py, run_q512_revision.py, "
            "run_q512_selection_revision.py and analyze_mapping_sensitivity.py."
        ) from exc
    return q512, reference


def source_hashes(q512, reference):
    from ect_analysis.reporting import sha256_file
    values = q512.source_hashes()
    for path in (Path(__file__), Path(reference.__file__)):
        values[path.name] = sha256_file(path)
    return values


def noise_covariance(channel_count, rho):
    from ect_analysis.channels import channel_table
    if rho == 0:
        return np.eye(channel_count)
    if not 0 < rho < 1:
        raise ValueError("Require 0 < rho < 1 for correlated noise")
    channels = channel_table(channel_count)
    columns = ["electrode_1_one_based", "electrode_2_one_based"]
    if channels[columns].isna().any().any():
        raise ValueError("Shared-electrode noise requires classified measurement channels")
    pairs = channels[columns].to_numpy(dtype=int) - 1
    incidence = np.zeros((channel_count, int(pairs.max()) + 1))
    incidence[np.arange(channel_count)[:, None], pairs] = 1.0
    return (1.0 - rho) * np.eye(channel_count) + rho * (incidence @ incidence.T) / 2.0


def inverse_setup(sensitivity, lambda_rel):
    from ect_analysis.metrics import gram_and_eigenvalues
    sensitivity = np.asarray(sensitivity, dtype=float)
    if sensitivity.ndim != 2 or not np.isfinite(sensitivity).all():
        raise ValueError("Sensitivity must be a finite two-dimensional matrix")
    gram, eigenvalues = gram_and_eigenvalues(sensitivity, 1.0)
    ridge_lambda = lambda_rel * math.sqrt(float(eigenvalues[-1]))
    if not np.isfinite(ridge_lambda) or ridge_lambda <= 0:
        raise ValueError("Reconstruction requires positive regularisation")
    factor = cho_factor(gram + ridge_lambda**2 * np.eye(gram.shape[0]),
                        lower=True, check_finite=False)
    return sensitivity, factor, ridge_lambda, gram, eigenvalues


def reconstruction(setup, measurements):
    sensitivity, factor, *_ = setup
    return sensitivity @ cho_solve(factor, measurements, check_finite=False)


def measurement_batches(clean, standard_noise, repeats, noise_rel):
    sigma = noise_rel * np.sqrt(np.mean(clean**2, axis=0))
    if not np.isfinite(clean).all() or np.any(np.linalg.norm(clean, axis=0) == 0):
        raise ValueError("Every target must have finite, non-zero forward measurements")
    for name, rho in SCENARIOS:
        if rho is None:
            yield name, clean.copy(), np.zeros_like(sigma), 1
        else:
            transformed = standard_noise
            if rho:
                covariance = noise_covariance(clean.shape[0], rho)
                transformed = cholesky(covariance, lower=True, check_finite=False) @ standard_noise
            data = np.repeat(clean, repeats, axis=1) + transformed * np.repeat(sigma, repeats)[None, :]
            yield name, data, sigma, repeats


def score_pair(forward, inverse, scenario, setup, data, sigma, repeats,
               xyz, volumes, truths, positions, threshold):
    from ect_analysis.phantoms import phantom_metrics_one
    estimates = reconstruction(setup, data)
    if not np.isfinite(estimates).all():
        raise FloatingPointError(f"Non-finite reconstruction: {forward}/{inverse}/{scenario}")
    rows = []
    for position_index, position in positions.iterrows():
        for repeat in range(repeats):
            result = phantom_metrics_one(
                xyz, volumes, truths[:, position_index],
                estimates[:, position_index * repeats + repeat], threshold,
            )
            rows.append({
                "configuration": inverse,
                "forward_configuration": forward,
                "inverse_configuration": inverse,
                "noise_scenario": scenario,
                "matched_operators": forward == inverse,
                "target_id": int(position.target_id), "repeat": repeat,
                "r_mm": float(position.r_mm), "theta_deg": float(position.theta_deg),
                "z_mm": float(position.z_mm), "n_truth_bins": int(position.n_truth_bins),
                "effective_truth_bins": float(position.effective_truth_bins),
                "realized_truth_volume_mm3": float(position.effective_truth_bins * volumes[0]),
                "noise_sigma": float(sigma[position_index]), "ridge_lambda": setup[2],
                **{name: result.get(name, float("nan")) for name in METRICS},
                "status": result["status"],
            })
    return pd.DataFrame(rows)


def run_experiment(operators, xyz, volumes, truths, positions, repeats,
                   noise_rel, lambda_rel, threshold, seed):
    labels = sorted(operators)
    channel_count = next(iter(operators.values())).shape[1]
    if any(array.shape != (len(xyz), channel_count) for array in operators.values()):
        raise ValueError("All operators must use identical parameter and measurement bases")
    standard_noise = np.random.default_rng(seed).standard_normal(
        (channel_count, len(positions) * repeats)
    )
    setups = {label: inverse_setup(operators[label], lambda_rel) for label in labels}
    clean = {label: operators[label].T @ truths for label in labels}
    frames, input_rows, discrepancy_rows = [], [], []
    for forward in labels:
        forward_norm = np.linalg.norm(operators[forward])
        signal_norm = np.linalg.norm(clean[forward], axis=0)
        for inverse in labels:
            differences = np.linalg.norm(clean[forward] - clean[inverse], axis=0) / signal_norm
            operator_difference = float(
                np.linalg.norm(operators[forward] - operators[inverse]) / forward_norm
            )
            for index, position in positions.iterrows():
                discrepancy_rows.append({
                    "forward_configuration": forward, "inverse_configuration": inverse,
                    "target_id": int(position.target_id),
                    "relative_clean_measurement_difference": float(differences[index]),
                    "relative_operator_frobenius_difference": operator_difference,
                })
        for scenario, data, sigma, count in measurement_batches(
            clean[forward], standard_noise, repeats, noise_rel
        ):
            input_rows.append({
                "forward_configuration": forward, "noise_scenario": scenario,
                "repeats_per_target": count, "measurement_vectors": data.shape[1],
                "measurement_array_sha256": hashlib.sha256(
                    np.ascontiguousarray(data, dtype=np.float64).tobytes()
                ).hexdigest(),
                "noise_sigma_min": float(sigma.min()), "noise_sigma_max": float(sigma.max()),
                "shared_by_inverse_models": ",".join(labels),
            })
            for inverse in labels:
                frames.append(score_pair(
                    forward, inverse, scenario, setups[inverse], data, sigma, count,
                    xyz, volumes, truths, positions, threshold,
                ))
            print(f"  FORWARD {forward}: {scenario}, {len(labels)} inverse models complete", flush=True)
    return (pd.concat(frames, ignore_index=True), pd.DataFrame(input_rows),
            pd.DataFrame(discrepancy_rows), standard_noise, clean)


def target_records(raw):
    rows = []
    for keys, group in raw.groupby(CASE_KEYS + ["target_id"], sort=True):
        valid = group.status.eq("ok") & np.isfinite(group[list(METRICS)]).all(axis=1)
        complete = bool(valid.all())
        rows.append({
            **dict(zip(CASE_KEYS + ["target_id"], keys)),
            "noise_realisations": len(group), "successful_realisations": int(valid.sum()),
            "complete_target": complete,
            **{name: float(group[name].median()) if complete else float("nan") for name in METRICS},
        })
    return pd.DataFrame(rows)


def summarize_targets(targets):
    rows = []
    for keys, group in targets.groupby(CASE_KEYS, sort=True):
        rows.append({
            **dict(zip(CASE_KEYS, keys)), "n_targets": len(group),
            "n_complete_targets": int(group.complete_target.sum()),
            "failed_realisations": int((group.noise_realisations - group.successful_realisations).sum()),
            **{f"median_{name}": float(group[name].median()) for name in METRICS},
        })
    return pd.DataFrame(rows)


def paired_changes(targets, bootstrap_samples, seed):
    rng = np.random.default_rng(seed)
    baseline = targets[targets.forward_configuration.eq(targets.inverse_configuration)]
    rows = []
    for keys, group in targets.groupby(CASE_KEYS, sort=True):
        forward, inverse, scenario = keys
        if forward == inverse:
            continue
        control = baseline[
            baseline.forward_configuration.eq(forward) & baseline.noise_scenario.eq(scenario)
        ]
        paired = group.merge(control, on="target_id", suffixes=("", "_matched"), validate="one_to_one")
        complete = paired.complete_target & paired.complete_target_matched
        for metric in METRICS:
            delta = (paired.loc[complete, metric] - paired.loc[complete, metric + "_matched"]).to_numpy()
            interval = [float("nan"), float("nan")]
            if len(delta):
                draws = rng.integers(0, len(delta), size=(bootstrap_samples, len(delta)))
                interval = np.percentile(np.median(delta[draws], axis=1), [2.5, 97.5])
            rows.append({
                **dict(zip(CASE_KEYS, keys)), "metric": metric,
                "reference_inverse_configuration": forward,
                "delta_definition": "cross_mesh_minus_matched_same_forward_and_noise",
                "n_targets": len(paired), "n_complete_paired_targets": len(delta),
                "median_paired_change": float(np.median(delta)) if len(delta) else float("nan"),
                "bootstrap_ci_low": float(interval[0]), "bootstrap_ci_high": float(interval[1]),
                "bootstrap_samples": bootstrap_samples,
                "improvement_direction": "positive" if metric in ("auc", "dice_50pct_peak", "weighted_correlation") else "negative",
            })
    return pd.DataFrame(rows)


def read_basis_and_truth(primary_directory, primary):
    from ect_analysis.mesh import make_common_grid
    from ect_analysis.models import MeshMeta
    from ect_analysis.phantoms import common_bin_quadrature_points, phantom_truth_matrix
    q512, _ = modules()
    metas = [MeshMeta(**{key: value for key, value in record.items() if key != "sha256"})
             for record in primary["inputs"]]
    grid = make_common_grid(metas, 10, 16, 20)
    basis = pd.read_csv(primary_directory / "tables" / "q512_fixed_basis.csv", float_precision="round_trip")
    np.testing.assert_array_equal(basis.bin_index_zero_based, np.arange(grid.n_bins))
    if basis.in_fixed_evaluation_basis.dtype != bool:
        raise ValueError("The frozen basis CSV must contain Boolean membership values")
    mask = basis.in_fixed_evaluation_basis.to_numpy(dtype=bool)
    if mask.sum() != 2851 or q512.mask_sha256(mask) != q512.PUBLICATION_MASK_SHA256:
        raise ValueError("Frozen basis identities differ from the primary experiment")
    np.testing.assert_allclose(basis[["x_mm", "y_mm", "z_mm"]], grid.xyz_centres_mm, rtol=0, atol=1e-12)
    np.testing.assert_allclose(basis.bin_volume_mm3, grid.bin_volume_mm3, rtol=1e-14)
    positions = pd.read_csv(primary_directory / "tables" / "phantom_positions.csv", float_precision="round_trip")
    settings = primary["parameters"]
    points = common_bin_quadrature_points(grid, mask, settings["phantom_subsamples_per_axis"])
    truths, recomputed = phantom_truth_matrix(points, positions, settings["phantom_radius_mm"], settings["phantom_min_bins"])
    if len(positions) != 40 or len(recomputed) != 40:
        raise ValueError("This experiment requires the original 40 retained targets")
    for name in ("target_id", "n_truth_bins", "effective_truth_bins"):
        np.testing.assert_allclose(recomputed[name], positions[name], rtol=0, atol=1e-12)
    return grid, mask, truths, positions


def verify_primary_matched(raw, primary_directory):
    primary = pd.read_csv(primary_directory / "tables" / "phantom_metrics_all_realisations.csv",
                          float_precision="round_trip")
    expected = primary[primary.configuration.isin(LABELS)].copy()
    actual = raw[raw.forward_configuration.eq(raw.inverse_configuration) & raw.noise_scenario.eq("iid")].copy()
    keys = ["configuration", "target_id", "repeat"]
    expected = expected.sort_values(keys).reset_index(drop=True)
    actual = actual.sort_values(keys).reset_index(drop=True)
    if len(expected) != 600 or len(actual) != 600:
        raise ValueError("The primary matched controls require 600 realisations")
    pd.testing.assert_frame_equal(actual[expected.columns], expected, check_exact=False,
                                  rtol=3e-10, atol=1e-10, check_dtype=False)
    return {"status": "pass", "realisations": len(actual), "rtol": 3e-10, "atol": 1e-10}


def export_results(output, raw, input_rows, discrepancies, bootstrap_samples, seed):
    q512, _ = modules()
    tables = output / "tables"
    tables.mkdir(exist_ok=True)
    targets = target_records(raw)
    summary = summarize_targets(targets)
    changes = paired_changes(targets, bootstrap_samples, seed)
    frames = {
        "mismatch_metrics_all_realisations.csv": raw,
        "mismatch_target_metrics.csv": targets,
        "mismatch_summary.csv": summary,
        "mismatch_paired_changes.csv": changes,
        "mismatch_measurement_batches.csv": input_rows,
        "mismatch_forward_discrepancies.csv": discrepancies,
    }
    for name, frame in frames.items():
        q512.save_dataframe_full_precision(frame, tables / name)
        saved = pd.read_csv(tables / name, float_precision="round_trip")
        q512.assert_exported_statistics_equal(frame, saved, name)
    read_back = pd.read_csv(tables / "mismatch_metrics_all_realisations.csv", float_precision="round_trip")
    regenerated_targets = target_records(read_back)
    for name, frame in (
        ("mismatch_target_metrics.csv", regenerated_targets),
        ("mismatch_summary.csv", summarize_targets(regenerated_targets)),
        ("mismatch_paired_changes.csv", paired_changes(regenerated_targets, bootstrap_samples, seed)),
    ):
        saved = pd.read_csv(tables / name, float_precision="round_trip")
        q512.assert_exported_statistics_equal(frame, saved, name)
    return {"raw_reconstructions": len(raw), "target_records": len(targets),
            "cases": len(summary), "paired_metric_comparisons": len(changes),
            "failed_realisations": int(raw.status.ne("ok").sum()),
            "incomplete_targets": int((~targets.complete_target).sum()),
            "export_reproducibility": "exact_float64_readback_and_recomputed_target_statistics"}


def numerical_self_test():
    from ect_analysis.phantoms import phantom_metrics_one, run_phantoms
    q512, _ = modules()
    q512.numerical_self_test()
    rng = np.random.default_rng(5938)
    s_a = rng.uniform(0.1, 1.5, size=(7, 496))
    operators = {"A": s_a, "B": s_a * np.linspace(.8, 1.2, 7)[:, None],
                 "C": s_a + rng.uniform(-.08, .08, s_a.shape)}
    xyz = rng.uniform(-1, 1, size=(7, 3))
    volumes = np.ones(7)
    truths = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0]], dtype=float)
    positions = pd.DataFrame({"target_id": [0, 1], "r_mm": [0., 1.], "theta_deg": [0., 90.],
                              "z_mm": [0., .5], "n_truth_bins": [2, 2], "effective_truth_bins": [2., 2.]})
    raw, inputs, discrepancies, standard_noise, clean = run_experiment(
        operators, xyz, volumes, truths, positions, 2, .01, .1, .5, 123
    )
    assert len(raw) == 126 and raw.status.eq("ok").all()
    assert len(inputs) == 12 and len(discrepancies) == 18
    assert discrepancies[discrepancies.forward_configuration.eq(discrepancies.inverse_configuration)][
        "relative_clean_measurement_difference"].eq(0).all()
    assert discrepancies[discrepancies.forward_configuration.ne(discrepancies.inverse_configuration)][
        "relative_operator_frobenius_difference"].gt(0).all()
    for label in operators:
        expected = run_phantoms(label, operators[label], xyz, volumes, truths, positions,
                                standard_noise, 2, .01, .1, .5)
        actual = raw[raw.forward_configuration.eq(label) & raw.inverse_configuration.eq(label) & raw.noise_scenario.eq("iid")]
        pd.testing.assert_frame_equal(actual[expected.columns].reset_index(drop=True), expected,
                                      check_exact=False, rtol=1e-11, atol=1e-12, check_dtype=False)
    setup = inverse_setup(operators["B"], .1)
    primal = np.linalg.solve(operators["B"] @ operators["B"].T + setup[2]**2 * np.eye(7),
                             operators["B"] @ clean["A"])
    np.testing.assert_allclose(reconstruction(setup, clean["A"]), primal, rtol=1e-11, atol=1e-12)
    cross_row = raw[raw.forward_configuration.eq("A") & raw.inverse_configuration.eq("B")
                    & raw.noise_scenario.eq("noiseless") & raw.target_id.eq(0)].iloc[0]
    primal_metrics = phantom_metrics_one(xyz, volumes, truths[:, 0], primal[:, 0], .5)
    for metric in METRICS:
        np.testing.assert_allclose(cross_row[metric], primal_metrics[metric], rtol=1e-11, atol=1e-12)
    for rho in (.25, .5):
        covariance = noise_covariance(496, rho)
        np.testing.assert_array_equal(np.diag(covariance), np.ones(496))
        assert covariance[0, 1] == rho / 2 and covariance[0, 61] == 0
        lower = cholesky(covariance, lower=True)
        np.testing.assert_allclose(lower @ lower.T, covariance, rtol=1e-12, atol=1e-12)
    with tempfile.TemporaryDirectory(prefix="q512_mismatch_selftest_") as folder:
        counts = export_results(Path(folder), raw, inputs, discrepancies, 50, 123)
        assert counts["cases"] == 36 and counts["target_records"] == 72
        assert counts["paired_metric_comparisons"] == 144
    # A failed realisation must not disappear into a median of successful repeats.
    failed = raw.copy()
    first_target = failed.index[
        failed.forward_configuration.eq("A") & failed.inverse_configuration.eq("B")
        & failed.noise_scenario.eq("iid") & failed.target_id.eq(0)
    ][0]
    failed.loc[first_target, "status"] = "nonpositive_reconstruction"
    failed.loc[first_target, list(METRICS)] = np.nan
    target = target_records(failed)
    selected = target[target.forward_configuration.eq("A") & target.inverse_configuration.eq("B")
                      & target.noise_scenario.eq("iid") & target.target_id.eq(0)]
    assert len(selected) == 1 and not selected.complete_target.iloc[0]
    assert selected[list(METRICS)].isna().all().all()
    # A pooled median is 0.1 here, while the required median of target medians is 0.05.
    aggregation_probe = []
    for target_id, values in enumerate(([0., 0., 0., 1., 1.], [.1] * 5)):
        for value in values:
            aggregation_probe.append({
                "forward_configuration": "A", "inverse_configuration": "B",
                "noise_scenario": "iid", "target_id": target_id, "status": "ok",
                **{metric: value for metric in METRICS},
            })
    probe_summary = summarize_targets(target_records(pd.DataFrame(aggregation_probe)))
    assert probe_summary.median_nrmse.iloc[0] == .05
    print("Q512 MODEL MISMATCH SELF-TEST PASSED", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--primary-run", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--q512-block-cells", type=int, default=4096)
    parser.add_argument("--q512-point-budget", type=int, default=262144)
    options = parser.parse_args(argv)
    if options.q512_block_cells <= 0 or options.q512_point_budget < 512:
        parser.error("Require positive block size and point budget >=512")
    if options.self_test:
        numerical_self_test()
        return 0
    q512, reference = modules()
    from ect_analysis.channels import channel_table
    from ect_analysis.cli import discover_files, label_from_path
    from ect_analysis.mesh import inspect_mesh
    from ect_analysis.metrics import regularized_spectral_metrics
    from ect_analysis.reporting import json_ready, sha256_file
    args, mode = q512.mapping_helper().resolve_arguments([])
    if options.input_dir is not None:
        args.input_dir = local_path(options.input_dir)
    primary_directory, primary = reference.find_primary(options.primary_run, q512)
    before = source_hashes(q512, reference)
    reference.validate_reference_settings(args, primary, before)
    for key, value in primary["parameters"].items():
        if key.startswith("phantom_") and getattr(args, key) != value:
            raise ValueError(f"{key} differs from the primary Q512 experiment")
    if args.phantom_noise_repeats != 5 or args.phantom_noise_rel != .01:
        raise ValueError("This publication test requires five repeats and 1% relative noise")
    root = local_path(options.output_dir) if options.output_dir is not None else PROJECT_DIRECTORY / "q512_model_mismatch_results"
    output = root / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    (output / "tables").mkdir()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "complete": False, "error": None,
        "primary_run": str(primary_directory), "configuration_source": mode,
        "parameters": vars(args), "options": vars(options), "source_sha256": before,
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "pandas": pd.__version__},
        "mapping": primary["mapping"], "inputs": [],
        "forward_inverse_labels": list(LABELS), "all_ordered_pairs": True,
        "noise_scenarios": dict(SCENARIOS),
        "noise_covariance_rule": "R=(1-rho)*I+rho*B@B.T/2; unit diagonal",
        "reconstruction_noise_assumption": "original unweighted identity-noise Tikhonov, including correlated scenarios",
        "paired_reference": "matched inverse for the same forward operator and identical noisy data",
        "summary_policy": "median across repetitions per complete target, then median across targets",
        "bootstrap_policy": "percentile 95% interval for median paired changes; resample targets; descriptive",
        "notes": [
            "Different existing FEM mesh operators for cross-mesh cases; same frozen parameter basis.",
            "Common-basis piecewise-constant target representation is retained in data generation.",
            "No new FEM solve, independent nonlinear forward solve or experimental data.",
            "Electrode-position uncertainty, hardware effects and reduced-channel imaging are not tested.",
            "Noise covariance is simulated; inversion is not whitened or retuned for correlated noise.",
            "A/B/C matched iid controls must reproduce the primary Q512 reconstruction table.",
            "Failed reconstructions and incomplete targets are counted rather than silently removed.",
            "Source files, native inputs, primary manuscript results and previous runs are not overwritten.",
        ],
    }
    started = time.perf_counter()
    input_stats = {}
    print(f"Q512 model-mismatch output: {output}", flush=True)
    try:
        grid, mask, truths, positions = read_basis_and_truth(primary_directory, primary)
        paths = {label_from_path(path): path for path in discover_files(args)}
        expected = {row["label"]: row for row in primary["inputs"]}
        mapper = q512.FixedBasisQ512Mapper(grid, mask, options.q512_block_cells, options.q512_point_budget)
        operators = {}
        guard_rows = []
        alpha = primary["regularization"]["alpha_absolute_primary"]
        scores = pd.read_csv(primary_directory / "tables" / "global_common_basis_scorecard.csv", float_precision="round_trip").set_index("configuration")
        for label in LABELS:
            if label not in paths:
                raise FileNotFoundError(f"Native mesh {label} is missing")
            path = paths[label]
            state = path.stat()
            input_stats[path] = (state.st_size, state.st_mtime_ns)
            digest = sha256_file(path)
            meta = inspect_mesh(path, args.mesh_unit)
            record = {**asdict(meta), "sha256": digest}
            for key in ("sha256", "n_cells", "n_vertices", "n_channels"):
                if record[key] != expected[label][key]:
                    raise ValueError(f"Native {label}: {key} differs from the primary input")
            print(f"MAPPING {label}: {meta.n_cells:,} tetrahedra", flush=True)
            mapped = mapper(meta, grid, args.mesh_unit, args.sensitivity_kind, np.float64)
            operators[label] = mapped.sensitivity[mask].copy()
            del mapped
            measures = regularized_spectral_metrics(operators[label], alpha, 1.0)
            for metric in ("regularized_logdet", "spectral_energy", "effective_rank_energy", "stable_rank"):
                value, original = measures[metric], float(scores.loc[label, metric])
                if not np.isclose(value, original, rtol=3e-10, atol=1e-9):
                    raise ValueError(f"{label}: {metric} differs from the primary Q512 result")
                guard_rows.append({"configuration": label, "metric": metric,
                                   "recomputed": value, "primary": original,
                                   "absolute_difference": abs(value - original)})
            manifest["inputs"].append(record)
        q512.save_dataframe_full_precision(pd.DataFrame(guard_rows), output / "tables" / "mismatch_primary_operator_checks.csv")
        q512.save_dataframe_full_precision(pd.DataFrame(mapper.rows), output / "tables" / "mismatch_mapping_diagnostics.csv")
        q512.save_dataframe_full_precision(positions, output / "tables" / "phantom_positions.csv")
        q512.save_dataframe_full_precision(channel_table(496), output / "tables" / "channel_definitions.csv")
        raw, batch_rows, discrepancies, standard_noise, clean = run_experiment(
            operators, grid.xyz_centres_mm[mask], np.full(mask.sum(), grid.bin_volume_mm3),
            truths, positions, args.phantom_noise_repeats, args.phantom_noise_rel,
            args.phantom_lambda_rel, args.phantom_threshold, args.seed,
        )
        manifest["primary_matched_reconstruction_check"] = verify_primary_matched(raw, primary_directory)
        counts = export_results(output, raw, batch_rows, discrepancies, args.bootstrap_samples, args.seed)
        if (counts["cases"], counts["raw_reconstructions"], counts["target_records"]) != (36, 5760, 1440):
            raise ValueError(f"Unexpected experiment dimensions: {counts}")
        manifest["results"] = counts
        np.savez_compressed(output / "experiment_inputs.npz", truths=truths, standard_normal_draws=standard_noise,
                            **{f"clean_measurements_{label}": array for label, array in clean.items()})
        for path, initial in input_stats.items():
            current = path.stat()
            if (current.st_size, current.st_mtime_ns) != initial:
                raise RuntimeError(f"Native input changed during execution: {path.name}")
        after = source_hashes(q512, reference)
        if after != before:
            raise RuntimeError("Project source changed during execution")
        manifest["source_files_unchanged"] = True
        manifest["primary_files_sha256"] = {
            name: sha256_file(primary_directory / name) for name in (
                "run_manifest.json", "q512_revision_manifest.json", "tables/q512_fixed_basis.csv",
                "tables/phantom_positions.csv", "tables/phantom_metrics_all_realisations.csv",
            )
        }
        manifest["output_sha256"] = {str(path.relative_to(output)): sha256_file(path)
                                     for path in sorted(output.rglob("*")) if path.is_file()}
        manifest["complete"] = True
        print("PRIMARY A/B/C Q512 RECONSTRUCTION CONTROLS PASSED", flush=True)
        print(f"Q512 MODEL MISMATCH COMPLETED: {counts['raw_reconstructions']} reconstructions", flush=True)
        if counts["failed_realisations"]:
            print(f"NOTE: {counts['failed_realisations']} failed realisations retained in the output", flush=True)
        return 0
    except BaseException as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["elapsed_seconds"] = time.perf_counter() - started
        (output / "model_mismatch_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=json_ready), encoding="utf-8"
        )
        print(f"Model-mismatch files saved in: {output}", flush=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
