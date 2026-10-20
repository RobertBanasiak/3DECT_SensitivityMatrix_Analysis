#!/usr/bin/env python3
"""Rerun existing channel-selection studies on the verified Q512 basis.

Place beside main.py, run_q512_revision.py, analyze_mapping_sensitivity.py,
compare_measurement_baselines.py and analyze_selection_stability.py.

    python run_q512_selection_revision.py --self-test
    python run_q512_selection_revision.py
    python run_q512_selection_revision.py --selection-study stability

The default study is the original baseline comparison: four deterministic
selectors, 100 paired random sets, and five deterministic timing repeats.
The stability study uses the existing covariance, alpha and subset-size
definitions. Only one study runs at a time; their pipeline observers are
never nested. Neither study runs reconstruction or a new FEM solve.

The Q512 mapper, frozen 2851-bin mask, unit-noise convention and shared-alpha
rule come from run_q512_revision.py. Results must reproduce a completed
primary Q512 run, including its native-input hashes and ordered 128-channel
sets. The latest complete primary run is found in q512_revision_results;
--primary-run PATH selects it explicitly. Relative paths use this file's
directory. Original-project CLI flags still work; wrapper-only flags retain
main.py's PYCHARM_CONFIG. Stability-specific flags have their old meanings.

Each run uses a new directory under q512_baseline_comparison_results or
q512_selection_stability_results. --output-dir ROOT changes that root.
Original scripts, native inputs and previous results are not overwritten.
CSV tables retain float64 precision; read them with float_precision='round_trip'.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIRECTORY = Path(__file__).resolve().parent
EXPECTED_LABELS = list("ABCDEFGHIJKLMNO")


def project_module(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        raise RuntimeError(f"Place {name}.py beside this script and main.py.") from exc


def options_and_arguments(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--selection-study", choices=("baselines", "stability"), default="baselines")
    parser.add_argument("--primary-run", type=Path)
    options, remaining = parser.parse_known_args(argv)
    q512 = project_module("run_q512_revision")
    mapping_options, remaining = q512.revision_options(remaining)
    stability_options = None
    if options.selection_study == "stability":
        module = project_module("analyze_selection_stability")
        stability_options, remaining = module.stability_options(remaining)
    args, mode = q512.mapping_helper().resolve_arguments(remaining)
    return options, mapping_options, stability_options, args, mode, remaining


def read_primary(directory, q512):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    revision = json.loads((directory / "q512_revision_manifest.json").read_text(encoding="utf-8"))
    if revision.get("complete") is not True or revision.get("error") is not None:
        raise ValueError("The primary Q512 run is incomplete or has an error.")
    mapping = manifest.get("mapping", {})
    required = {
        "quadrature_points_per_tetra": 512,
        "fixed_basis_parameters": 2851,
        "fixed_basis_mask_sha256": q512.PUBLICATION_MASK_SHA256,
        "basis_policy": "frozen_original_centroid_common_evaluation_domain",
        "shared_alpha_policy": "recalibrated_once_from_Q512_operators_with_original_rule",
    }
    if any(mapping.get(key) != value for key, value in required.items()):
        raise ValueError("The primary run does not use the verified frozen Q512 basis.")
    if manifest.get("table_export", {}).get("csv_significant_digits") != 17:
        raise ValueError("Select the primary Q512 run with full-precision CSV export.")
    if sorted(row["label"] for row in manifest.get("inputs", [])) != EXPECTED_LABELS:
        raise ValueError("The primary run must contain all configurations A--O.")
    if manifest.get("parameters", {}).get("subset_k") != 128:
        raise ValueError("Select the primary run with 128 selected channels.")
    for name in ("global_common_basis_scorecard.csv", "subset_scorecard.csv", "selected_channels.csv"):
        if not (directory / "tables" / name).is_file():
            raise ValueError(f"Primary run is missing tables/{name}.")
    alpha = manifest.get("regularization", {}).get("alpha_absolute_primary")
    if alpha is None or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Primary run has an invalid shared alpha.")
    return directory, manifest


def find_primary(path, q512):
    if path is not None:
        path = path if path.is_absolute() else PROJECT_DIRECTORY / path
        return read_primary(path, q512)
    root = PROJECT_DIRECTORY / "q512_revision_results"
    for candidate in sorted(root.glob("run_*"), reverse=True):
        try:
            return read_primary(candidate, q512)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    raise FileNotFoundError(
        "No complete full-precision primary Q512 run was found. "
        "Use --primary-run PATH to select its run directory."
    )


def source_hashes(q512):
    from ect_analysis.reporting import sha256_file
    values = q512.source_hashes()
    for name in (Path(__file__).name, "run_q512_revision.py",
                 "compare_measurement_baselines.py", "analyze_selection_stability.py"):
        path = PROJECT_DIRECTORY / name
        if path.is_file():
            values[name] = sha256_file(path)
    return values


def validate_reference_settings(args, primary, current_sources):
    expected = primary.get("parameters", {})
    for name in ("alpha_rel", "subset_k", "seed", "mesh_unit", "dtype",
                 "sensitivity_kind", "score_noise_std", "grid_r", "grid_theta", "grid_z"):
        if getattr(args, name) != expected.get(name):
            raise ValueError(f"{name} differs from the primary Q512 run.")
    if list(args.alpha_sweep) != list(expected.get("alpha_sweep", [])):
        raise ValueError("The alpha sweep differs from the primary Q512 run.")
    if args.subset_k != 128:
        raise ValueError("This revision compares the primary 128-channel selections.")
    for name, value in primary.get("source_sha256", {}).items():
        if name.startswith("ect_analysis/") or name in (
            "analyze_mapping_sensitivity.py", "run_q512_revision.py"
        ):
            if current_sources.get(name) != value:
                raise ValueError(f"Numerical source {name} changed since the primary Q512 run.")


def input_signature(manifest):
    return sorted((row["label"], row["sha256"], row["n_cells"],
                   row["n_vertices"], row["n_channels"]) for row in manifest["inputs"])


def compare_numeric_tables(left, right, name):
    left = left.sort_values("configuration").reset_index(drop=True)
    right = right.sort_values("configuration").reset_index(drop=True)
    if left.configuration.tolist() != EXPECTED_LABELS or right.configuration.tolist() != EXPECTED_LABELS:
        raise ValueError(f"{name}: configuration labels differ from A--O.")
    if list(left.columns) != list(right.columns):
        raise ValueError(f"{name}: table columns differ from the primary run.")
    for column in left.select_dtypes(include=[np.number]).columns:
        if not np.allclose(left[column], right[column], rtol=1e-10, atol=1e-10, equal_nan=True):
            raise ValueError(f"{name}: {column} differs from the primary Q512 run.")


def verify_primary_match(output, primary_directory, primary):
    current = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    if input_signature(current) != input_signature(primary):
        raise ValueError("Native-input hashes or dimensions differ from the primary Q512 run.")
    alpha = current["regularization"]["alpha_absolute_primary"]
    if not np.isclose(alpha, primary["regularization"]["alpha_absolute_primary"], rtol=5e-12, atol=0):
        raise ValueError("Shared alpha differs from the primary Q512 run.")
    if current["common_grid"]["common_occupied_bins"] != 2851:
        raise ValueError("The pipeline changed the frozen 2851-bin basis.")
    for name in ("global_common_basis_scorecard.csv", "subset_scorecard.csv"):
        left = pd.read_csv(output / "tables" / name, float_precision="round_trip")
        right = pd.read_csv(primary_directory / "tables" / name, float_precision="round_trip")
        compare_numeric_tables(left, right, name)
    keys = ["configuration", "selection_rank", "channel_index_zero_based"]
    selections = []
    for directory in (output, primary_directory):
        frame = pd.read_csv(directory / "tables" / "selected_channels.csv", float_precision="round_trip")
        selections.append(frame.sort_values(keys[:2])[keys].reset_index(drop=True))
    if len(selections[0]) != 15 * 128 or not selections[0].equals(selections[1]):
        raise ValueError("The ordered primary 128-channel sets were not reproduced.")
    print("PRIMARY Q512 CONSISTENCY CHECK PASSED", flush=True)
    return current, {"status": "pass", "native_input_hashes": "identical",
                     "global_and_subset_metrics": "matched_with_float64_tolerance",
                     "ordered_reference_channel_sets": "identical", "alpha_common": alpha}


def save_study_tables(observer, output, study, q512):
    observer.save(output)
    prefix = "baseline_" if study == "baselines" else "stability_"
    for path in sorted(output.glob(prefix + "*.csv")):
        frame = pd.read_csv(path, float_precision="round_trip")
        q512.save_dataframe_full_precision(frame, path)
    name = "baseline_scorecard.csv" if study == "baselines" else "stability_scorecard.csv"
    saved = pd.read_csv(output / name, float_precision="round_trip")
    q512.assert_exported_statistics_equal(pd.DataFrame(observer.rows), saved, name)
    return {"status": "pass", "raw_scorecard_rows": len(saved), "csv_significant_digits": 17,
            "reader_float_precision": "round_trip"}


def observer_integration_self_test(q512):
    from ect_analysis import mesh, pipeline
    from ect_analysis.models import MeshMeta
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    cells = np.array([[0, 1, 2, 3]])
    density = np.array([[10., 8., 3., 2., 1., .1]])
    metas = [MeshMeta(label, "unused", 1, 4, 6, 1., 0., 1.) for label in ("A", "B")]
    grid = mesh.make_common_grid(metas, 1, 1, 2)
    original_loader = mesh.load_mesh
    functions = {name: getattr(pipeline, name) for name in (
        "map_to_common_grid", "gram_and_eigenvalues", "greedy_regularized_logdet"
    )}
    try:
        mesh.load_mesh = lambda *unused: (density.copy(), vertices.copy(), cells.copy())
        for study in ("baselines", "stability"):
            mapper = q512.FixedBasisQ512Mapper(grid, np.ones(2, dtype=bool), 1, 512)
            if study == "baselines":
                module = project_module("compare_measurement_baselines")
                observer = module.BaselineComparison(pipeline, 123, random_draws=2, timing_repeats=2)
            else:
                module = project_module("analyze_selection_stability")
                args = argparse.Namespace(seed=123, alpha_rel=.2, alpha_sweep=(.1, .2, .3),
                                          score_noise_std=1.)
                options = argparse.Namespace(stability_k=[2, 3, 4, 5], noise_profile_count=1,
                                             noise_std_factor=2., correlation_strengths=[.25])
                observer = module.StabilityAnalysis(pipeline, args, options)
            with q512.pipeline_mapping(mapper), observer:
                datasets = [pipeline.map_to_common_grid(meta, grid, "mm", "density", np.float64)
                            for meta in metas]
                for dataset in datasets:
                    np.testing.assert_allclose(dataset.sensitivity,
                                               np.array([[7/48], [1/48]]) * density, rtol=1e-13)
                    gram, values = pipeline.gram_and_eigenvalues(dataset.sensitivity, 1.)
                    pipeline.greedy_regularized_logdet(gram, .2, 3)
            assert observer.labels == ["A", "B"] and observer.spectral_calls == 2
            assert len(mapper.rows) == 2
            if study == "baselines":
                assert len(observer.rows) == 14
            else:
                assert observer.completed == ["A", "B"]
                assert len(observer.rows) == 2 * observer.expected_cases
            with tempfile.TemporaryDirectory(prefix="q512_observer_selftest_") as directory:
                save_study_tables(observer, Path(directory), study, q512)
            assert all(getattr(pipeline, name) is function for name, function in functions.items())
        mapper = q512.FixedBasisQ512Mapper(grid, np.ones(2, dtype=bool))
        observer = project_module("compare_measurement_baselines").BaselineComparison(
            pipeline, 123, random_draws=1, timing_repeats=1
        )
        try:
            with q512.pipeline_mapping(mapper), observer:
                raise RuntimeError("observer-restoration-test")
        except RuntimeError as exc:
            assert str(exc) == "observer-restoration-test"
        assert all(getattr(pipeline, name) is function for name, function in functions.items())
    finally:
        mesh.load_mesh = original_loader
    print("Q512 STUDY OBSERVER INTEGRATION SELF-TEST PASSED", flush=True)


def numerical_self_test():
    q512 = project_module("run_q512_revision")
    q512.numerical_self_test()
    project_module("compare_measurement_baselines").numerical_self_test()
    project_module("analyze_selection_stability").numerical_self_test()
    observer_integration_self_test(q512)
    # The adapter must preserve near-tied values when exporting observer rows.
    rows = [{"configuration": "A", "value": np.nextafter(1., 0.), "missing": np.nan},
            {"configuration": "B", "value": 1., "missing": .12345678912345678}]
    class FakeObserver:
        def __init__(self):
            self.rows = rows
        def save(self, directory):
            pd.DataFrame(self.rows).to_csv(directory / "baseline_scorecard.csv", index=False)
    with tempfile.TemporaryDirectory(prefix="q512_selection_selftest_") as directory:
        save_study_tables(FakeObserver(), Path(directory), "baselines", q512)
    for study in ("baselines", "stability"):
        options, _, _, args, mode, remaining = options_and_arguments(
            ["--selection-study", study, "--self-test"]
        )
        assert options.selection_study == study and args.self_test
        assert remaining == ["--self-test"]
    print("Q512 SELECTION REVISION SELF-TEST PASSED", flush=True)


def main(argv=None):
    if not (PROJECT_DIRECTORY / "main.py").is_file():
        raise RuntimeError("Place this script beside main.py and the existing revision helpers.")
    options, mapping_options, stability_options, args, mode, remaining = options_and_arguments(
        list(sys.argv[1:] if argv is None else argv)
    )
    if args.self_test:
        numerical_self_test()
        return 0
    q512 = project_module("run_q512_revision")
    from ect_analysis import pipeline
    from ect_analysis.cli import discover_files, validate_args
    from ect_analysis.mesh import inspect_mesh, make_common_grid
    from ect_analysis.reporting import json_ready
    validate_args(args)
    primary_directory, primary = find_primary(options.primary_run, q512)
    before = source_hashes(q512)
    validate_reference_settings(args, primary, before)
    source_parameters = dict(vars(args))
    args = argparse.Namespace(**source_parameters)
    args.analyses = "global,subsets"
    metas = [inspect_mesh(path, args.mesh_unit) for path in discover_files(args)]
    shape = (args.grid_r, args.grid_theta, args.grid_z)
    grid = make_common_grid(metas, *shape)
    masks, counts = q512.mapping_helper().common_supports(metas, {shape: grid}, args.mesh_unit)
    mask = masks[shape]
    q512.validate_publication_basis(args, metas, grid, mask)
    mapper = q512.FixedBasisQ512Mapper(grid, mask, mapping_options.q512_block_cells,
                                     mapping_options.q512_point_budget)
    study = options.selection_study
    if study == "baselines":
        module = project_module("compare_measurement_baselines")
        observer = module.BaselineComparison(pipeline, args.seed, random_draws=100, timing_repeats=5)
        settings = {"random_draws_per_configuration": 100, "deterministic_timing_repeats": 5,
                    "same_random_sets_across_configurations": True, "random_seed": args.seed}
        root_name = "q512_baseline_comparison_results"
    else:
        module = project_module("analyze_selection_stability")
        observer = module.StabilityAnalysis(pipeline, args, stability_options)
        settings = {"stability_options": vars(stability_options), "noise_profiles_seed": args.seed,
                    "same_noise_profiles_across_configurations": True}
        root_name = "q512_selection_stability_results"
    explicit_output = any(flag == "--output-dir" or flag.startswith("--output-dir=") for flag in remaining)
    root = args.output_dir if explicit_output else PROJECT_DIRECTORY / root_name
    output = (root / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    args.output_dir = output
    mapping = dict(primary["mapping"])
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "study": study,
        "configuration_source": mode, "source_parameters": source_parameters,
        "executed_parameters": vars(args), "study_settings": settings,
        "mapping_options": vars(mapping_options), "mapping": mapping,
        "primary_reference_run": str(primary_directory), "source_sha256": before,
        "platform": platform.platform(), "processor": platform.processor(),
        "logical_processor_count": os.cpu_count(),
        "notes": [
            "Original study classes and selection algorithms are used without changing their definitions.",
            "Q512 approximately integrates unchanged native piecewise-constant FEM sensitivities.",
            "The original 2851-bin evaluation mask is held fixed; no exact bin-intersection claim is made.",
            "Shared alpha is recalibrated once on Q512 using the original rule and checked against the primary run.",
            "For stability scenarios, alpha is not recalibrated separately for the imposed noise models.",
            "Only global/subset analyses run; no reconstruction, new FEM solve or hardware test is performed.",
            "Selection timers retain their original scope; mapping, shared Gram construction and scoring are excluded.",
            "QR selection includes its factorisation; deterministic baseline times use five repeats.",
            "Random/profile quantiles describe the imposed ensemble, not confidence intervals for a mean.",
            "Native inputs, project source and previous results are not overwritten.",
        ],
        "complete": False, "error": None,
    }
    print(f"Q512 {study} output: {output}", flush=True)
    print(f"Primary Q512 reference: {primary_directory}", flush=True)
    print("Frozen basis: 2851/3200 bins; Q=512", flush=True)
    started = time.perf_counter()
    try:
        with q512.full_precision_tables():
            q512.save_fixed_basis(output, grid, mask, counts[shape])
        with q512.pipeline_mapping(mapper), q512.full_precision_tables(), observer:
            code = pipeline.run_analysis(args_override=args, entrypoint_path=Path(__file__))
        if code != 0:
            raise RuntimeError(f"The original pipeline returned {code}.")
        if len(mapper.rows) != 15 or sorted(row["configuration"] for row in mapper.rows) != EXPECTED_LABELS:
            raise RuntimeError("Every configuration must be Q512-mapped exactly once.")
        if study == "baselines":
            if len(observer.rows) != 15 * 105 or observer.n_channels != 496 or observer.k != 128:
                raise RuntimeError("Incomplete baseline comparison.")
        elif sorted(observer.completed) != EXPECTED_LABELS or len(observer.rows) != 15 * observer.expected_cases:
            raise RuntimeError("Incomplete stability analysis.")
        metadata["table_export"] = save_study_tables(observer, output, study, q512)
        current, check = verify_primary_match(output, primary_directory, primary)
        metadata["primary_consistency_check"] = check
        metadata["regularization"] = current["regularization"]
        metadata["inputs"] = current["inputs"]
        metadata["software"] = current["software"]
        if source_hashes(q512) != before:
            raise RuntimeError("Project source changed during the selection study.")
        current["mapping"] = mapping
        current["table_export"] = metadata["table_export"]
        current["primary_consistency_check"] = check
        current["source_sha256"] = before
        current["interpretation_notes"].extend(metadata["notes"])
        (output / "run_manifest.json").write_text(
            json.dumps(current, indent=2, ensure_ascii=False, default=json_ready), encoding="utf-8"
        )
        metadata["complete"] = True
        print(f"Q512 {study.upper()} ANALYSIS COMPLETED", flush=True)
        return 0
    except BaseException as exc:
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata["elapsed_seconds"] = time.perf_counter() - started
        metadata["completed_scorecard_rows"] = len(observer.rows)
        metadata["source_files_unchanged"] = source_hashes(q512) == before
        if mapper.rows:
            q512.save_dataframe_full_precision(pd.DataFrame(mapper.rows), output / "tables" / "q512_mapping_diagnostics.csv")
        if not metadata["complete"] and observer.rows:
            try:
                observer.save(output)
            except Exception as exc:
                metadata["partial_export_error"] = f"{type(exc).__name__}: {exc}"
        (output / "q512_selection_revision_manifest.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, default=json_ready), encoding="utf-8"
        )
        print(f"Q512 selection files saved in: {output}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
