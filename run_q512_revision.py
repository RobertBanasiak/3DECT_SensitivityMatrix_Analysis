#!/usr/bin/env python3
"""Rerun the manuscript pipeline with Q512 on its frozen 2851-bin domain.

Place beside main.py and the already supplied analyze_mapping_sensitivity.py:
    python run_q512_revision.py --self-test
    python run_q512_revision.py

Uses main.py's PYCHARM_CONFIG when no original-project CLI flags are supplied.
Original-project flags retain the same parsing rules as the mapping-study
helper. Extra flags alone also preserve PYCHARM_CONFIG:
    --q512-block-cells 4096
    --q512-point-budget 262144

This entry point is intentionally for the manuscript's A--O, 496-channel,
10x16x20, density/float64/unit-noise experiment. It verifies both the size AND
the identity of the original centroid-common evaluation mask. Q512 mapping is
restricted to that fixed domain; bins outside it are zero-padded solely to
interface with the original pipeline. Actual quadrature-projected volumes,
not centroid volumes, are returned. A missing quadrature-supported bin causes
an error instead of silently changing the parameter basis.

The original pipeline runs global, group, subset and phantom analyses as
configured. Its shared-alpha rule is recalibrated ONCE from the Q512 operators:
alpha = alpha_rel * median_configuration(trace(G_Q512)/496).
This differs from the mapping sensitivity study, which deliberately held the
centroid-reference alpha fixed. Reconstruction/noise/phantom settings retain
their original meanings. Separate baseline, noise-stability and timing scripts
are NOT run by this entry point and still require a later Q512 adaptation.

Uses the same equal-volume child-centroid quadrature as the checked mapping
study: approximate integration of unchanged native piecewise-constant fields,
not exact curved-bin intersections, a new FEM solve, or experimental validation.

No project source or native input is written. Each analysis gets a new
q512_revision_results/run_* directory, including with --output-dir ROOT.
Tables use 17 significant digits. Float64 values must be read back using
pandas.read_csv(..., float_precision="round_trip"). Phantom summary and paired
statistics are automatically recomputed from the exported raw metrics and
compared exactly with the exported tables before the run is marked complete.
The per-configuration mapped_native_volume_fraction in the main scorecard
refers here to the FROZEN EVALUATION DOMAIN. The mapping diagnostics separately
report the full-cylinder fraction for clarity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIRECTORY = Path(__file__).resolve().parent
PUBLICATION_GRID = (10, 16, 20)
PUBLICATION_PARAMETERS = 2851
# SHA256 of the 3200-element, r-fastest Boolean mask encoded as uint8 bytes.
# Verified against the supplied mapping study run_20261013T155031_270748Z.
PUBLICATION_MASK_SHA256 = (
    "9e39d980294e2de86bb10618e924d32100fe77a6483328d448646142b3db8338"
)
SUBDIVISION_LEVEL = 3
QUADRATURE_POINTS = 512


def mapping_helper():
    try:
        import analyze_mapping_sensitivity as helper
    except ModuleNotFoundError as exc:
        if exc.name != "analyze_mapping_sensitivity":
            raise
        raise RuntimeError(
            "Place run_q512_revision.py beside main.py and the previously "
            "supplied analyze_mapping_sensitivity.py."
        ) from exc
    for name in ("map_rules", "common_supports", "resolve_arguments", "numerical_self_test"):
        if not callable(getattr(helper, name, None)):
            raise RuntimeError(f"Mapping-study helper is incompatible: missing {name}")
    return helper


def mask_sha256(mask):
    return hashlib.sha256(np.asarray(mask, dtype=np.uint8).tobytes()).hexdigest()


def revision_options(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--q512-block-cells", type=int, default=4096)
    parser.add_argument("--q512-point-budget", type=int, default=262144)
    options, original = parser.parse_known_args(argv)
    if options.q512_block_cells <= 0:
        parser.error("--q512-block-cells must be positive")
    if options.q512_point_budget < QUADRATURE_POINTS:
        parser.error("--q512-point-budget must be at least 512")
    return options, original


def validate_publication_basis(args, metas, grid, mask):
    expected = {
        "grid_r": 10, "grid_theta": 16, "grid_z": 20,
        "sensitivity_kind": "density", "dtype": "float64", "score_noise_std": 1.0,
    }
    for name, value in expected.items():
        if getattr(args, name) != value:
            raise ValueError(f"This manuscript rerun requires {name}={value!r}")
    if [m.label for m in metas] != list("ABCDEFGHIJKLMNO"):
        raise ValueError("The frozen manuscript basis requires all 15 configurations A--O")
    if any(m.n_channels != 496 for m in metas):
        raise ValueError("The manuscript rerun requires 496 channels in every configuration")
    if grid.n_bins != 3200 or int(mask.sum()) != PUBLICATION_PARAMETERS:
        raise ValueError(
            f"Expected the original 2851/3200-bin basis; found {int(mask.sum())}/{grid.n_bins}"
        )
    if mask_sha256(mask) != PUBLICATION_MASK_SHA256:
        raise ValueError(
            "The centroid-common bin identities differ from the verified manuscript "
            "domain. Do not replace the fixed domain with a different 2851-bin mask."
        )
    geometry = [grid.r_edges_mm[-1], grid.z_edges_mm[0], grid.z_edges_mm[-1]]
    if not np.allclose(geometry, [75.0, 0.0, 315.0], rtol=0, atol=1e-7):
        raise ValueError("Expected the original cylindrical domain: R=75 mm, z=0..315 mm")


class FixedBasisQ512Mapper:
    """Actual Q512 volume integration, explicitly restricted to a frozen ROI."""

    def __init__(self, grid, mask, block_cells=4096, point_budget=262144):
        self.grid = grid
        self.mask = np.array(mask, dtype=bool, copy=True)
        if self.mask.shape != (grid.n_bins,) or not self.mask.any():
            raise ValueError("The fixed basis mask must contain at least one valid grid bin")
        if block_cells <= 0 or point_budget < QUADRATURE_POINTS:
            raise ValueError("Require positive block size and point budget >= 512")
        self.block_cells = block_cells
        self.point_budget = point_budget
        self.rows = []

    def __call__(self, meta, grid, mesh_unit, sensitivity_kind, dtype):
        from ect_analysis.mesh import load_mesh
        from ect_analysis.models import MappedDataset

        if (grid.nr, grid.ntheta, grid.nz) != (
            self.grid.nr, self.grid.ntheta, self.grid.nz
        ):
            raise ValueError("The pipeline changed the fixed grid dimensions")
        for name in ("r_edges_mm", "theta_edges_rad", "z_edges_mm"):
            if not np.array_equal(getattr(grid, name), getattr(self.grid, name)):
                raise ValueError(f"The pipeline changed the fixed grid edges: {name}")
        start = time.perf_counter()
        sensitivity, vertices, simplices = load_mesh(meta, mesh_unit, dtype)
        shape = (grid.nr, grid.ntheta, grid.nz)
        maps, occupied, diagnostics = mapping_helper().map_rules(
            sensitivity, vertices, simplices, {shape: grid}, [SUBDIVISION_LEVEL],
            sensitivity_kind, block_cells=self.block_cells,
            point_budget=self.point_budget, progress_label=meta.label,
        )
        array = maps[(shape, SUBDIVISION_LEVEL)]
        volumes = occupied[(shape, SUBDIVISION_LEVEL)]
        if not np.all(np.isfinite(array)) or not np.all(np.isfinite(volumes)):
            raise FloatingPointError(f"{meta.label}: non-finite Q512 mapping")
        missing = np.flatnonzero(self.mask & (volumes <= 0))
        if len(missing):
            raise ValueError(
                f"{meta.label}: Q512 leaves {len(missing)} frozen basis bins unsupported "
                f"(first indices: {missing[:10].tolist()}); the basis will not be shortened"
            )
        row = dict(diagnostics[0])
        native_volume = row["native_volume_mm3"]
        basis_volume = float(volumes[self.mask].sum())
        row.update({
            "configuration": meta.label,
            "fixed_basis_parameters": int(self.mask.sum()),
            "fixed_basis_geometric_volume_mm3": int(self.mask.sum()) * grid.bin_volume_mm3,
            "quadrature_projected_native_volume_in_basis_mm3": basis_volume,
            "quadrature_occupied_basis_bins": int((volumes[self.mask] > 0).sum()),
            "full_cylinder_mapped_native_volume_fraction": row["mapped_native_volume_fraction"],
            "fixed_basis_mapped_native_volume_fraction": basis_volume / native_volume,
            "q512_mapping_elapsed_seconds": time.perf_counter() - start,
        })
        self.rows.append(row)
        # The original pipeline intersects occupied-volume masks. Zero-padding
        # BOTH arrays outside the declared evaluation domain makes that mask
        # exactly the frozen ROI, without substituting fictitious voxel volumes.
        array[~self.mask] = 0.0
        volumes[~self.mask] = 0.0
        return MappedDataset(meta, array, volumes, basis_volume / native_volume)


@contextmanager
def pipeline_mapping(mapper):
    """Select the mapper for this process only; restore it even after failure."""
    from ect_analysis import pipeline
    original = pipeline.map_to_common_grid
    pipeline.map_to_common_grid = mapper
    try:
        yield
    finally:
        pipeline.map_to_common_grid = original


def save_dataframe_full_precision(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.17g")


@contextmanager
def full_precision_tables():
    """Use a float64 round-trip format in this execution only."""
    from ect_analysis import pipeline, reporting
    previous_pipeline, previous_reporting = pipeline.save_dataframe, reporting.save_dataframe
    pipeline.save_dataframe = save_dataframe_full_precision
    reporting.save_dataframe = save_dataframe_full_precision
    try:
        yield
    finally:
        pipeline.save_dataframe = previous_pipeline
        reporting.save_dataframe = previous_reporting


def assert_exported_statistics_equal(recomputed, exported, name):
    if list(recomputed.columns) != list(exported.columns) or len(recomputed) != len(exported):
        raise ValueError(f"{name}: exported table dimensions or columns differ")
    for column in recomputed.columns:
        left, right = recomputed[column], exported[column]
        if pd.api.types.is_numeric_dtype(left):
            equal = np.array_equal(left.to_numpy(), right.to_numpy(), equal_nan=True)
        else:
            equal = left.equals(right)
        if not equal:
            raise ValueError(f"{name}: values in {column} cannot be reproduced from exported raw metrics")


def verify_exported_phantom_statistics(output, bootstrap_samples, seed):
    from ect_analysis.phantoms import summarize_phantoms, paired_phantom_comparisons
    tables = Path(output) / "tables"
    raw_path = tables / "phantom_metrics_all_realisations.csv"
    if not raw_path.is_file():
        return {"status": "not_applicable", "reason": "phantom analysis was not requested"}
    raw = pd.read_csv(raw_path, float_precision="round_trip")
    summary = summarize_phantoms(raw, bootstrap_samples, seed)
    pairs = paired_phantom_comparisons(raw, bootstrap_samples, seed)
    for name, expected in (("phantom_summary.csv", summary),
                           ("phantom_pairwise_comparisons.csv", pairs)):
        saved = pd.read_csv(tables / name, float_precision="round_trip")
        assert_exported_statistics_equal(expected, saved, name)
    print("EXPORTED PHANTOM STATISTICS REPRODUCIBILITY PASSED", flush=True)
    return {"status": "pass", "comparison": "exact_float64_equality_with_nan_equivalence",
            "raw_realisations": len(raw), "summary_rows": len(summary), "pairwise_rows": len(pairs),
            "csv_significant_digits": 17, "csv_reader_float_precision": "round_trip"}


def source_hashes():
    from ect_analysis.reporting import sha256_file
    files = [Path(__file__), PROJECT_DIRECTORY / "main.py",
             PROJECT_DIRECTORY / "analyze_mapping_sensitivity.py",
             *sorted((PROJECT_DIRECTORY / "ect_analysis").glob("*.py"))]
    return {p.name if p.parent == PROJECT_DIRECTORY else f"ect_analysis/{p.name}":
            sha256_file(p) for p in files if p.is_file()}


def save_fixed_basis(output, grid, mask, counts):
    from ect_analysis.reporting import save_dataframe
    table = pd.DataFrame({
        "bin_index_zero_based": np.arange(grid.n_bins),
        "in_fixed_evaluation_basis": mask,
        "centroid_occupied_configurations": counts,
        "x_mm": grid.xyz_centres_mm[:, 0], "y_mm": grid.xyz_centres_mm[:, 1],
        "z_mm": grid.xyz_centres_mm[:, 2], "bin_volume_mm3": grid.bin_volume_mm3,
    })
    save_dataframe(table, output / "tables" / "q512_fixed_basis.csv")


def numerical_self_test():
    from ect_analysis import mesh, pipeline, reporting
    from ect_analysis.models import MeshMeta

    mapping_helper().numerical_self_test()
    vertices = np.array([[0., 0., 0.], [1., 0., 0.],
                         [0., 1., 0.], [0., 0., 1.]])
    cells = np.array([[0, 1, 2, 3]])
    sensitivity = np.array([[6., -3.]])
    meta = MeshMeta("test", "unused", 1, 4, 2, 1., 0., 1.)
    grid = mesh.make_common_grid([meta], 1, 1, 2)
    mask = np.array([True, False])
    original_loader, original_mapper = mesh.load_mesh, pipeline.map_to_common_grid
    try:
        mesh.load_mesh = lambda *unused: (sensitivity.copy(), vertices.copy(), cells.copy())
        mapper = FixedBasisQ512Mapper(grid, mask, 1, 512)
        result = mapper(meta, grid, "mm", "density", np.float64)
        # Analytic tetrahedron volume above z=1/2 is 1/8 of its total.
        # The Q512 rule reproduces this cut; the retained ROI is the lower half.
        np.testing.assert_allclose(result.sensitivity, [[7/8, -7/16], [0., 0.]],
                                   rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(result.occupied_volume_mm3, [7/48, 0.], rtol=1e-13)
        assert np.isclose(result.mapped_native_volume_fraction, 7/8)
        assert mapper.rows[0]["relative_channel_integral_conservation_error"] < 1e-12
        assert np.isclose(mapper.rows[0]["full_cylinder_mapped_native_volume_fraction"], 1.)
        np.testing.assert_array_equal(result.occupied_volume_mm3 > 0, mask)
        mesh.load_mesh = lambda *unused: (sensitivity.copy()/6, vertices.copy(), cells.copy())
        integrated = FixedBasisQ512Mapper(grid, mask)(
            meta, grid, "mm", "element-integrated", np.float64
        )
        np.testing.assert_allclose(result.sensitivity, integrated.sensitivity, rtol=1e-13)
        mesh.load_mesh = lambda *unused: (sensitivity.copy(), vertices.copy(), cells.copy())
        fine = mesh.make_common_grid([meta], 1, 1, 20)
        unsupported = np.zeros(fine.n_bins, dtype=bool)
        unsupported[-1] = True
        try:
            FixedBasisQ512Mapper(fine, unsupported)(meta, fine, "mm", "density", np.float64)
        except ValueError as exc:
            assert "unsupported" in str(exc)
        else:
            raise AssertionError("An unsupported fixed-domain bin must cause an error")
        try:
            with pipeline_mapping(mapper):
                assert pipeline.map_to_common_grid is mapper
                raise RuntimeError("restoration-test")
        except RuntimeError as exc:
            assert str(exc) == "restoration-test"
        assert pipeline.map_to_common_grid is original_mapper
    finally:
        mesh.load_mesh = original_loader
        pipeline.map_to_common_grid = original_mapper
    # Values close to ties must survive CSV export with their original bits.
    values = [1., np.nextafter(1., 0.), .9991234567891234,
              np.nextafter(.9991234567891234, 1.), 19.72622412345678, 1.2e-12]
    previous_pipeline_writer, previous_reporting_writer = pipeline.save_dataframe, reporting.save_dataframe
    with tempfile.TemporaryDirectory(prefix="q512_csv_selftest_") as folder:
        path = Path(folder) / "roundtrip.csv"
        table = pd.DataFrame({"value": values})
        with full_precision_tables():
            pipeline.save_dataframe(table, path)
            assert reporting.save_dataframe is save_dataframe_full_precision
        read_back = pd.read_csv(path, float_precision="round_trip")
        assert_exported_statistics_equal(table, read_back, "roundtrip self-test")
    assert pipeline.save_dataframe is previous_pipeline_writer
    assert reporting.save_dataframe is previous_reporting_writer
    print("Q512 REVISION SELF-TEST PASSED", flush=True)


def main(argv=None):
    if not (PROJECT_DIRECTORY / "main.py").is_file():
        raise RuntimeError("Place run_q512_revision.py in the project folder beside main.py")
    helper = mapping_helper()
    from ect_analysis import pipeline
    from ect_analysis.cli import discover_files, validate_args
    from ect_analysis.mesh import inspect_mesh, make_common_grid
    from ect_analysis.reporting import json_ready, save_dataframe

    options, original = revision_options(list(sys.argv[1:] if argv is None else argv))
    args, mode = helper.resolve_arguments(original)
    if args.self_test:
        numerical_self_test()
        return 0
    validate_args(args)
    supplied_parameters = dict(vars(args))
    explicit_output = any(v == "--output-dir" or v.startswith("--output-dir=") for v in original)
    root = args.output_dir if explicit_output else PROJECT_DIRECTORY / "q512_revision_results"
    output = (root / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "tables").mkdir()
    args.output_dir = output
    before = source_hashes()
    mapper = None
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration_source": mode, "source_parameters": supplied_parameters,
        "revision_options": vars(options),
        "mapping": {
            "quadrature_points_per_tetra": QUADRATURE_POINTS,
            "subdivision_level": SUBDIVISION_LEVEL,
            "basis_policy": "frozen_original_centroid_common_evaluation_domain",
            "expected_basis_parameters": PUBLICATION_PARAMETERS,
            "expected_fixed_basis_mask_sha256": PUBLICATION_MASK_SHA256,
            "shared_alpha_policy": "recalibrated_once_from_Q512_operators_with_original_rule",
            "mapped_native_volume_fraction_scope": "frozen_evaluation_basis",
        },
        "table_export": {"csv_significant_digits": 17, "reader_float_precision": "round_trip"},
        "notes": [
            "Approximate Q512 integration of unchanged native piecewise-constant FEM sensitivity.",
            "Not exact curved-bin intersections, a new FEM solve, or experimental validation.",
            "Finite quadrature stabilization does not establish full operator convergence.",
            "Both sensitivity and occupied volume are zero-padded outside the declared frozen ROI.",
            "The main mapped_native_volume_fraction refers to that ROI; diagnostics also give the full-cylinder fraction.",
            "Primary alpha uses the original shared rule on Q512, unlike the fixed-alpha mapping sensitivity study.",
            "Phantom data generation and inversion still use the same Q512 operator (matched linear model).",
            "Extra baseline, noise-stability and computational-benchmark scripts are not executed here.",
            "CSV tables retain 17 significant digits; use float_precision='round_trip' when reading with pandas.",
            "Phantom summary and paired statistics are checked for exact reproducibility from exported raw metrics.",
            "Source files, native inputs and earlier results are not overwritten.",
        ],
        "complete": False, "error": None,
    }
    print(f"Q512 revision output: {output}", flush=True)
    started = time.perf_counter()
    try:
        paths = discover_files(args)
        metas = [inspect_mesh(path, args.mesh_unit) for path in paths]
        shape = (args.grid_r, args.grid_theta, args.grid_z)
        grid = make_common_grid(metas, *shape)
        masks, counts = helper.common_supports(metas, {shape: grid}, args.mesh_unit)
        mask = masks[shape]
        validate_publication_basis(args, metas, grid, mask)
        with full_precision_tables():
            save_fixed_basis(output, grid, mask, counts[shape])
        metadata["mapping"].update({
            "fixed_basis_parameters": int(mask.sum()),
            "fixed_basis_mask_sha256": mask_sha256(mask),
            "bin_volume_mm3": grid.bin_volume_mm3,
            "fixed_basis_volume_mm3": float(mask.sum()) * grid.bin_volume_mm3,
            "domain_fraction_of_cylinder": float(mask.mean()),
        })
        print(f"Frozen original basis: {int(mask.sum())}/{grid.n_bins} bins; Q=512", flush=True)
        mapper = FixedBasisQ512Mapper(grid, mask, options.q512_block_cells, options.q512_point_budget)
        with pipeline_mapping(mapper), full_precision_tables():
            code = pipeline.run_analysis(args_override=args, entrypoint_path=Path(__file__))
        if code != 0:
            raise RuntimeError(f"Original pipeline returned {code}")
        if len(mapper.rows) != 15 or {r["configuration"] for r in mapper.rows} != set("ABCDEFGHIJKLMNO"):
            raise RuntimeError("The pipeline did not map every configuration exactly once")
        if source_hashes() != before:
            raise RuntimeError("Project source changed during execution")
        metadata["exported_phantom_statistics_reproducibility"] = verify_exported_phantom_statistics(
            output, args.bootstrap_samples, args.seed
        )
        main_path = output / "run_manifest.json"
        manifest = json.loads(main_path.read_text(encoding="utf-8"))
        if manifest["common_grid"]["common_occupied_bins"] != PUBLICATION_PARAMETERS:
            raise RuntimeError("The pipeline changed the frozen basis parameter count")
        manifest["mapping"] = metadata["mapping"]
        manifest["table_export"] = metadata["table_export"]
        manifest["exported_phantom_statistics_reproducibility"] = metadata[
            "exported_phantom_statistics_reproducibility"
        ]
        manifest["source_sha256"] = before
        manifest["interpretation_notes"].extend(metadata["notes"])
        main_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False,
                                       default=json_ready), encoding="utf-8")
        metadata["regularization"] = manifest["regularization"]
        raw_phantoms = output / "tables" / "phantom_metrics_all_realisations.csv"
        if raw_phantoms.is_file():
            phantom = pd.read_csv(raw_phantoms)
            metadata["phantom_realisations"] = len(phantom)
            metadata["phantom_positions_per_configuration"] = (
                phantom.groupby("configuration")["target_id"].nunique().to_dict()
            )
        metadata["complete"] = True
        print("Q512 REVISION ANALYSIS COMPLETED", flush=True)
        return 0
    except BaseException as exc:
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata["elapsed_seconds"] = time.perf_counter() - started
        metadata["source_sha256"] = before
        metadata["source_files_unchanged"] = source_hashes() == before
        if mapper is not None and mapper.rows:
            save_dataframe_full_precision(pd.DataFrame(mapper.rows).sort_values("configuration"),
                                          output / "tables" / "q512_mapping_diagnostics.csv")
        (output / "q512_revision_manifest.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, default=json_ready),
            encoding="utf-8",
        )
        print(f"Q512 revision files saved in: {output}", flush=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
