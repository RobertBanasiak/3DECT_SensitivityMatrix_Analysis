#!/usr/bin/env python3
"""Benchmark the primary frozen-basis Q512 analysis.

Place beside main.py and the existing files:
    benchmark_revision.py
    run_q512_revision.py
    run_q512_selection_revision.py
    analyze_mapping_sensitivity.py

First run:
    python run_q512_benchmark_revision.py --self-test

Then run:
    python run_q512_benchmark_revision.py

No original-project CLI flags means main.py's PYCHARM_CONFIG is reused.
The existing --q512-block-cells and --q512-point-budget options are supported.
The latest complete full-precision primary run is found automatically under
q512_revision_results; --primary-run PATH selects it explicitly.

Results go to a fresh q512_benchmark_results/run_* directory. --output-dir
changes its parent directory, retaining the original parser's CLI semantics.
Original numerical source and native input files are not modified.

The original StageProfiler measures Q512 mapping including native HDF5 loading,
full-set Gram/eigenvalues, greedy selection with a precomputed Gram matrix,
and the matched reconstruction routine including data generation and metrics.
Basis preparation, pipeline execution, and verification have separate clocks.
Peak RSS is an OS-reported lifetime peak for the whole Python process, not a
per-stage memory peak. This is one run without controlled disk-cache state.

The benchmark requires the manuscript settings and checks agreement with the
primary Q512 operators, ordered selections, and reconstruction tables. These
are computational timings; acquisition and hardware performance are not timed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import os
import platform
import sys
import tempfile
import time
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIRECTORY = Path(__file__).resolve().parent
EXPECTED_LABELS = list("ABCDEFGHIJKLMNO")
STAGES = (
    "common_basis_mapping",
    "full_set_gram_and_eigenvalues",
    "greedy_selection",
    "matched_reconstruction_benchmark",
)
HOOKS = (
    "map_to_common_grid", "gram_and_eigenvalues",
    "greedy_regularized_logdet", "run_phantoms",
)


def project_module(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        raise RuntimeError(f"Place {name}.py beside this script and main.py.") from exc


def dependencies():
    q512 = project_module("run_q512_revision")
    benchmark = project_module("benchmark_revision")
    reference = project_module("run_q512_selection_revision")
    if tuple(benchmark.STAGES) != STAGES or not callable(benchmark.StageProfiler):
        raise RuntimeError("The existing benchmark_revision.py has an incompatible profiler.")
    return q512, benchmark, reference


def options_and_arguments(argv, q512):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--primary-run", type=Path)
    options, remaining = parser.parse_known_args(argv)
    mapping_options, remaining = q512.revision_options(remaining)
    args, mode = q512.mapping_helper().resolve_arguments(remaining)
    return options, mapping_options, args, mode, remaining


def source_hashes(q512, benchmark, reference):
    values = q512.source_hashes()
    for path in (Path(__file__), Path(benchmark.__file__), Path(reference.__file__)):
        values[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return values


def validate_settings(args, primary, sources, reference):
    from ect_analysis.cli import validate_args

    requested = validate_args(args)
    if requested != {"global", "groups", "subsets", "phantoms"}:
        raise ValueError("This benchmark requires global,groups,subsets,phantoms.")
    reference.validate_reference_settings(args, primary, sources)
    expected = primary["parameters"]
    fields = (
        "phantom_radius_mm", "phantom_r_samples", "phantom_theta_samples",
        "phantom_z_samples", "phantom_noise_repeats", "phantom_noise_rel",
        "phantom_lambda_rel", "phantom_threshold", "phantom_min_bins",
        "phantom_subsamples_per_axis", "phantom_min_volume_ratio",
        "bootstrap_samples", "figure_dpi",
    )
    for name in fields:
        if getattr(args, name) != expected.get(name):
            raise ValueError(f"{name} differs from the primary Q512 run.")


def validate_stage_dimensions(profiler):
    rows = profiler.rows
    if len(rows) != 60 or any(row["status"] != "ok" for row in rows):
        raise ValueError("Expected 60 successful stage measurements: four for each of A--O.")
    for stage in STAGES:
        labels = [row["configuration"] for row in rows if row["stage"] == stage]
        if sorted(labels) != EXPECTED_LABELS:
            raise ValueError(f"Incomplete or repeated configuration labels for {stage}.")
    if sorted(profiler.mapped_labels) != EXPECTED_LABELS:
        raise ValueError("Not all configurations were mapped exactly once.")
    for row in rows:
        if row["measurement_channels"] != 496:
            raise ValueError("A measured stage did not use 496 measurement channels.")
        if row.get("common_parameters", 2851) != 2851:
            raise ValueError("A measured stage changed the frozen 2851-bin basis.")
        if row.get("selected_channels", 128) != 128:
            raise ValueError("A measured selection did not contain 128 channels.")
        if row["stage"] == STAGES[0] and row["grid_bins_before_filter"] != 3200:
            raise ValueError("Mapping did not use the original 3200-bin cylindrical grid.")
        if row["stage"] == STAGES[3]:
            for key, expected in (
                ("target_positions", 40), ("reconstruction_vectors", 200),
                ("successful_reconstructions", 200),
            ):
                if row[key] != expected:
                    raise ValueError(f"{row['configuration']}: expected {key}={expected}.")
    return True


def verify_reconstruction_reference(output, primary_directory):
    tables = (
        ("phantom_metrics_all_realisations.csv", ["configuration", "target_id", "repeat"]),
        ("phantom_summary.csv", ["configuration"]),
        ("phantom_pairwise_comparisons.csv",
         ["metric", "configuration_a", "configuration_b"]),
    )
    for name, keys in tables:
        frames = [
            pd.read_csv(directory / "tables" / name, float_precision="round_trip")
            .sort_values(keys).reset_index(drop=True)
            for directory in (output, primary_directory)
        ]
        try:
            pd.testing.assert_frame_equal(
                frames[0], frames[1], check_dtype=False, check_exact=False,
                rtol=1e-10, atol=1e-10,
            )
        except AssertionError as exc:
            raise ValueError(f"{name} differs from the primary reconstruction benchmark.") from exc
    raw = pd.read_csv(output / "tables" / tables[0][0], float_precision="round_trip")
    if len(raw) != 3000 or not raw["status"].eq("ok").all():
        raise ValueError("Expected all 3000 successful matched reconstructions.")
    if raw.groupby("configuration")["target_id"].nunique().to_dict() != {
        label: 40 for label in EXPECTED_LABELS
    }:
        raise ValueError("Expected 40 paired target positions for every configuration.")
    return {
        "status": "pass", "successful_reconstructions": 3000,
        "comparison": "raw, summary and pairwise tables matched at float64 tolerance",
    }


def environment_details(benchmark):
    import scipy

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print("NumPy build configuration:")
        np.show_config()
        print("SciPy build configuration:")
        scipy.show_config()
    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
    except ImportError:
        pools = None
    return {
        "platform": platform.platform(), "python": platform.python_version(),
        "cpu_model": benchmark.cpu_model(), "logical_cpu_count": os.cpu_count(),
        "physical_ram_bytes": benchmark.physical_ram_bytes(),
        "blas_build_configuration": buffer.getvalue(),
        "threadpools_if_available": pools,
        "thread_environment": {name: os.environ.get(name) for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
        )},
    }


def numerical_self_test(q512, benchmark):
    from ect_analysis import mesh, pipeline
    from ect_analysis.models import MeshMeta

    q512.numerical_self_test()
    vertices = np.array([[0., 0., 0.], [1., 0., 0.],
                         [0., 1., 0.], [0., 0., 1.]])
    cells = np.array([[0, 1, 2, 3]])
    native = np.array([[6., -3., 2., 1.]])
    meta = MeshMeta("test", "unused", 1, 4, 4, 1., 0., 1.)
    grid = mesh.make_common_grid([meta], 1, 1, 2)
    mask = np.ones(2, dtype=bool)
    previous_loader = mesh.load_mesh
    previous_hooks = {name: getattr(pipeline, name) for name in HOOKS}
    profiler = benchmark.StageProfiler(pipeline)
    try:
        mesh.load_mesh = lambda *unused: (native.copy(), vertices.copy(), cells.copy())
        mapper = q512.FixedBasisQ512Mapper(grid, mask, 1, 512)
        with q512.pipeline_mapping(mapper), profiler:
            mapped = pipeline.map_to_common_grid(meta, grid, "mm", "density", np.float64)
            np.testing.assert_allclose(mapped.sensitivity[:, 0], [7/8, 1/8], rtol=1e-13)
            gram, eigenvalues = pipeline.gram_and_eigenvalues(mapped.sensitivity, 1.)
            alpha = float(np.trace(gram)) / 4 * 1e-3
            selected, cumulative, marginals = pipeline.greedy_regularized_logdet(gram, alpha, 2)
            assert len(selected) == len(cumulative) == len(marginals) == 2
            np.testing.assert_allclose(cumulative, np.cumsum(marginals), rtol=1e-13)
            positions = pd.DataFrame([{
                "target_id": 0, "r_mm": 0., "theta_deg": 0., "z_mm": .25,
                "x_mm": 0., "y_mm": 0., "n_truth_bins": 1, "effective_truth_bins": 1.,
            }])
            reconstructed = pipeline.run_phantoms(
                "test", mapped.sensitivity, grid.xyz_centres_mm,
                np.full(2, grid.bin_volume_mm3), np.array([[1.], [0.]]),
                positions, np.zeros((4, 2)), 2, 0.01, 0.01, 0.5,
            )
            assert len(reconstructed) == 2 and reconstructed["status"].eq("ok").all()
        assert Counter(row["stage"] for row in profiler.rows) == Counter(STAGES)
        assert all(row["status"] == "ok" and row["elapsed_seconds"] >= 0
                   for row in profiler.rows)
        assert all(row["configuration"] == "test" for row in profiler.rows)
        assert profiler.rows[-1]["successful_reconstructions"] == 2
        with tempfile.TemporaryDirectory(prefix="q512_benchmark_selftest_") as folder:
            profiler.save(Path(folder))
            saved = pd.read_csv(Path(folder) / "runtime_by_stage.csv")
            summary = pd.read_csv(Path(folder) / "runtime_summary.csv")
            assert len(saved) == 4 and set(summary["stage"]) == set(STAGES)
            assert summary["calls"].eq(1).all()
        for name in HOOKS:
            assert getattr(pipeline, name) is previous_hooks[name]

        failed_profiler = benchmark.StageProfiler(pipeline)

        def failing_loader(*unused):
            raise RuntimeError("benchmark-restoration-selftest")

        mesh.load_mesh = failing_loader
        try:
            with q512.pipeline_mapping(mapper), failed_profiler:
                pipeline.map_to_common_grid(meta, grid, "mm", "density", np.float64)
        except RuntimeError as exc:
            assert str(exc) == "benchmark-restoration-selftest"
        else:
            raise AssertionError("Expected the synthetic loader failure.")
        assert len(failed_profiler.rows) == 1
        assert failed_profiler.rows[0]["status"] == "error"
        for name in HOOKS:
            assert getattr(pipeline, name) is previous_hooks[name]
    finally:
        mesh.load_mesh = previous_loader
        for name, function in previous_hooks.items():
            setattr(pipeline, name, function)
    print("Q512 BENCHMARK INTEGRATION SELF-TEST PASSED", flush=True)
    print("Q512 BENCHMARK SELF-TEST PASSED", flush=True)


def main(argv=None):
    q512, benchmark, reference = dependencies()
    argv = list(sys.argv[1:] if argv is None else argv)
    options, mapping_options, args, mode, original = options_and_arguments(argv, q512)
    if args.self_test:
        numerical_self_test(q512, benchmark)
        return 0
    if not (PROJECT_DIRECTORY / "main.py").is_file():
        raise RuntimeError("Place this script beside main.py and the existing revision scripts.")
    from ect_analysis import pipeline
    from ect_analysis.cli import discover_files
    from ect_analysis.mesh import inspect_mesh, make_common_grid

    primary_directory, primary = reference.find_primary(options.primary_run, q512)
    before = source_hashes(q512, benchmark, reference)
    validate_settings(args, primary, before, reference)
    explicit_output = any(value == "--output-dir" or value.startswith("--output-dir=")
                          for value in original)
    root = args.output_dir if explicit_output else PROJECT_DIRECTORY / "q512_benchmark_results"
    output = (root / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "tables").mkdir()
    parameters = dict(vars(args))
    args.output_dir = output
    profiler = benchmark.StageProfiler(pipeline)
    mapper = None
    environment = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration_source": mode, "parameters": parameters,
        "mapping_options": vars(mapping_options),
        "primary_reference_directory": str(primary_directory),
        "source_sha256": before,
        "table_export": {"csv_significant_digits": 17, "reader_float_precision": "round_trip"},
        "timing_scopes": dict(zip(STAGES, (
            "Native HDF5 loading, Q512 child-centroid volume integration, mapping "
            "diagnostics and restriction to the frozen basis.",
            "Full-set noise scaling, Gram construction, symmetrisation and all eigenvalues.",
            "Original greedy selection with a precomputed Gram matrix; subset scoring excluded.",
            "Original phantom routine, including Gram/eigenvalues, data and noise generation, "
            "Cholesky setup, batched inversion and image metrics.",
        ))),
        "notes": [
            "One execution; disk-cache state is not controlled.",
            "Pipeline time is measured around the original pipeline.run_analysis call. "
            "It includes auxiliary scores, figures, statistics, reporting and input hashing.",
            "Frozen-basis preparation and postprocessing checks are timed separately.",
            "Wrapper time includes setup, basis preparation, pipeline execution and checks, "
            "before final benchmark-file export.",
            "Peak RSS is the OS lifetime peak of the entire Python process, "
            "including imports, preparation, native allocations and verification.",
            "No isolated per-stage memory peaks, acquisition timings, hardware savings "
            "or 128-versus-496 reconstruction speed-ups are measured.",
            "Q512 is approximate integration of unchanged native piecewise-constant "
            "sensitivity, not a new FEM solve.",
            "The reconstruction benchmark still uses matched forward/inverse operators.",
        ],
        "baseline_dimensions_expected": {
            "configurations": EXPECTED_LABELS, "common_parameters": 2851,
            "measurement_channels": 496, "selected_channels": 128,
            "target_positions": 40, "reconstruction_vectors_per_configuration": 200,
        },
        "baseline_dimensions_match_manuscript": False,
        "complete": False, "exit_code": 1, "error": None,
    }
    started = time.perf_counter()
    print(f"Q512 benchmark output: {output}", flush=True)
    print(f"Primary reference: {primary_directory}", flush=True)
    try:
        environment.update(environment_details(benchmark))
        preparation_start = time.perf_counter()
        paths = discover_files(args)
        metas = [inspect_mesh(path, args.mesh_unit) for path in paths]
        shape = (args.grid_r, args.grid_theta, args.grid_z)
        grid = make_common_grid(metas, *shape)
        masks, counts = q512.mapping_helper().common_supports(metas, {shape: grid}, args.mesh_unit)
        mask = masks[shape]
        q512.validate_publication_basis(args, metas, grid, mask)
        with q512.full_precision_tables():
            q512.save_fixed_basis(output, grid, mask, counts[shape])
        mapping = {
            "quadrature_points_per_tetra": 512, "subdivision_level": 3,
            "basis_policy": "frozen_original_centroid_common_evaluation_domain",
            "fixed_basis_parameters": int(mask.sum()),
            "fixed_basis_mask_sha256": q512.mask_sha256(mask),
            "shared_alpha_policy": "recalibrated_once_from_Q512_operators_with_original_rule",
            "mapped_native_volume_fraction_scope": "frozen_evaluation_basis",
            "bin_volume_mm3": grid.bin_volume_mm3,
            "fixed_basis_volume_mm3": float(mask.sum()) * grid.bin_volume_mm3,
            "domain_fraction_of_cylinder": float(mask.mean()),
        }
        environment["mapping"] = mapping
        environment["basis_preparation_seconds"] = time.perf_counter() - preparation_start
        mapper = q512.FixedBasisQ512Mapper(
            grid, mask, mapping_options.q512_block_cells, mapping_options.q512_point_budget,
        )
        pipeline_start = time.perf_counter()
        try:
            with q512.pipeline_mapping(mapper), q512.full_precision_tables(), profiler:
                code = pipeline.run_analysis(args_override=args, entrypoint_path=Path(__file__))
        finally:
            environment["pipeline_elapsed_seconds"] = time.perf_counter() - pipeline_start
        if code != 0:
            raise RuntimeError(f"Original pipeline returned {code}.")
        check_start = time.perf_counter()
        environment["baseline_dimensions_match_manuscript"] = validate_stage_dimensions(profiler)
        if len(mapper.rows) != 15 or sorted(row["configuration"] for row in mapper.rows) != EXPECTED_LABELS:
            raise ValueError("Q512 mapping diagnostics do not cover A--O exactly once.")
        manifest_path = output / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["mapping"] = mapping
        manifest["table_export"] = environment["table_export"]
        manifest["source_sha256"] = before
        manifest["interpretation_notes"].extend(environment["notes"])
        manifest_path.write_text(json.dumps(
            manifest, indent=2, ensure_ascii=False, default=benchmark.json_ready,
        ), encoding="utf-8")
        manifest, match = reference.verify_primary_match(output, primary_directory, primary)
        environment["primary_consistency"] = match
        environment["reconstruction_reference_consistency"] = verify_reconstruction_reference(
            output, primary_directory,
        )
        environment["exported_phantom_statistics_reproducibility"] = (
            q512.verify_exported_phantom_statistics(output, args.bootstrap_samples, args.seed)
        )
        if source_hashes(q512, benchmark, reference) != before:
            raise RuntimeError("Project or revision source changed during the benchmark.")
        environment["study_common_grid"] = manifest["common_grid"]
        environment["study_regularization"] = manifest["regularization"]
        environment["study_software"] = manifest["software"]
        environment["study_inputs"] = manifest["inputs"]
        environment["postprocessing_checks_seconds"] = time.perf_counter() - check_start
        environment["complete"] = True
        environment["exit_code"] = 0
        print("Q512 COMPUTATIONAL BENCHMARK COMPLETED", flush=True)
        return 0
    except BaseException as exc:
        environment["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        environment["wrapper_elapsed_seconds"] = time.perf_counter() - started
        environment["has_stage_measurements"] = bool(profiler.rows)
        environment["source_files_unchanged"] = source_hashes(q512, benchmark, reference) == before
        peak, method = benchmark.process_peak_rss()
        environment["peak_process_rss_bytes"] = peak
        environment["peak_process_rss_MB"] = None if peak is None else peak / 1e6
        environment["peak_process_rss_method"] = method
        profiler.save(output)
        if mapper is not None and mapper.rows:
            q512.save_dataframe_full_precision(
                pd.DataFrame(mapper.rows).sort_values("configuration"),
                output / "tables" / "q512_mapping_diagnostics.csv",
            )
        manifest_path = output / "run_manifest.json"
        if manifest_path.is_file():
            environment["run_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        (output / "benchmark_environment.json").write_text(json.dumps(
            environment, indent=2, ensure_ascii=False, default=benchmark.json_ready,
        ), encoding="utf-8")
        print(f"Q512 benchmark files saved in: {output}", flush=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

