#!/usr/bin/env python3
"""Common-grid and tetrahedron-mapping sensitivity study for ECT revision.

Place beside main.py in the existing v1.0.0 project:
    python analyze_mapping_sensitivity.py --self-test
    python analyze_mapping_sensitivity.py

No original-project flags -> reuse main.py's PYCHARM_CONFIG, including when
only this helper's extra flags are supplied. Original flags use its parser.
Original source, native .mat files and previous results are not modified.

Default grids are half, reference and twice the configured (r, theta, z)
counts: 5x8x10, 10x16x20, 20x32x40. On EACH grid, compare the original
centroid rule with equal-volume tetrahedral subdivision quadrature: 8 and
64 child-tetrahedron centroids. This is approximate volume integration of
the original piecewise-constant native sensitivity, NOT exact intersections
with curved bins, new FEM solutions, or proof of asymptotic convergence.

Two evaluation scopes keep separate:
  centroid_common_support: original intersection of centroid-occupied bins,
    held fixed across all mapping rules WITHIN each grid. Its physical
    coverage can change BETWEEN grids and is explicitly reported.
  full_cylinder: all bins on the same modelled cylindrical domain, including
    empty bins, held fixed in physical extent across grids.

The reference-grid centroid/common-support case reproduces the manuscript's
operator, metrics, alpha and greedy selector. To compare resolutions, use
G/voxel_volume, spatial densities S/voxel_volume, and a fixed continuous
regularisation density alpha_density=alpha_reference/voxel_volume_reference.
Equivalently alpha_grid=alpha_reference*voxel_volume/voxel_volume_reference.
Thus changes in voxel volume alone do not change scores for a constant
sensitivity-density field. Raw metrics are exported too. Alpha is NOT
recalibrated for each mapping rule or grid. Cross-grid common-support
comparisons include changes in domain coverage; the full-cylinder scope
avoids that confound. Zero bins remain in the fixed evaluation bases.

Extra flags (can be combined with original-project flags):
    --mapping-grids 5x8x10 10x16x20 20x32x40
    --subdivision-levels 0 1 2       # level 0=centroid; Q=8**level
    --mapping-block-cells 4096
    --quadrature-point-budget 262144
    --no-selection                 # omit supplementary subset diagnostics

Results go to mapping_sensitivity_results/run_* by default:
    mapping_scorecard.csv, mapping_changes.csv, mapping_summary.csv,
    mapping_geometry.csv, mapping_common_support.csv,
    mapping_rank_stability.csv, mapping_gram_spectra.csv,
    mapping_selected_channels.csv, mapping_manifest.json.
No reconstruction or acquisition benchmark runs.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr

PROJECT_DIRECTORY = Path(__file__).resolve().parent
SCOPES = ("centroid_common_support", "full_cylinder")
SPATIAL_AMPLITUDES = (
    "local_norm_p50", "local_norm_p90", "local_norm_p99",
    "envelope_p50", "envelope_p90", "envelope_p99",
)
COMPARISON_METRICS = (
    "regularized_logdet", "spectral_energy_volume_normalized",
    "effective_rank_energy", "stable_rank", "coherence", "envelope_gini",
    "weak_volume_5pct_e99", "weak_volume_10pct_e99",
    *(f"{name}_density" for name in SPATIAL_AMPLITUDES),
)


def grid_name(shape):
    return "x".join(map(str, shape))


def case_name(shape, level):
    return f"grid_{grid_name(shape)}_Q{8**level}"


def child_tetrahedra(tetra):
    a, b, c, d = np.asarray(tetra, dtype=float)
    ab, ac, ad = (a+b)/2, (a+c)/2, (a+d)/2
    bc, bd, cd = (b+c)/2, (b+d)/2, (c+d)/2
    # Four corner tetrahedra; divide the central octahedron along ab--cd.
    return np.array([
        [a, ab, ac, ad], [ab, b, bc, bd], [ac, bc, c, cd], [ad, bd, cd, d],
        [ab, cd, ac, ad], [ab, cd, ad, bd],
        [ab, cd, bd, bc], [ab, cd, bc, ac],
    ])


def subdivision_rule(level):
    """Equal-volume child centroids expressed in parent barycentric coordinates."""
    if level < 0 or level > 3:
        raise ValueError("Subdivision levels must lie between 0 and 3")
    cells = np.eye(4)[None, :, :]
    for _ in range(level):
        cells = np.concatenate([child_tetrahedra(t) for t in cells], axis=0)
    return cells.mean(axis=1)


def point_bins(points, grid):
    """Use precisely the production bin-boundary convention; invalid -> -1."""
    from ect_analysis.mesh import _bin_indices
    points = np.asarray(points, dtype=float)
    shape = points.shape[:-1]
    xyz = points.reshape(-1, 3)
    ir = _bin_indices(np.hypot(xyz[:, 0], xyz[:, 1]), grid.r_edges_mm)
    it = _bin_indices(np.arctan2(xyz[:, 1], xyz[:, 0]), grid.theta_edges_rad)
    iz = _bin_indices(xyz[:, 2], grid.z_edges_mm)
    valid = ((ir >= 0) & (ir < grid.nr) & (it >= 0) & (it < grid.ntheta)
             & (iz >= 0) & (iz < grid.nz))
    flat = np.full(len(xyz), -1, dtype=np.int64)
    flat[valid] = np.ravel_multi_index((iz[valid], it[valid], ir[valid]), grid.shape)
    return flat.reshape(shape)


def load_geometry(meta, mesh_unit):
    from ect_analysis import mesh
    if mesh.h5py is None:
        raise ModuleNotFoundError("h5py is required for .mat input; install with python -m pip install h5py")
    with mesh.h5py.File(meta.path, "r") as handle:
        vertices = mesh._orient_columns(mesh._real_array(np.array(handle["vtx"]), "vtx"), 3, "vtx")
        simplices = mesh._orient_columns(np.array(handle["simp"]), 4, "simp")
    vertices = np.asarray(vertices, dtype=float) * (1 if mesh_unit == "mm" else 1000)
    simplices = np.asarray(simplices, dtype=np.int64)
    if simplices.min() == 1:
        simplices -= 1
    if len(simplices) != meta.n_cells or simplices.min() < 0 or simplices.max() >= len(vertices):
        raise ValueError("Invalid or changed tetrahedral connectivity")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("Vertices contain NaN or infinite values")
    return vertices, simplices


def common_supports(metas, grids, mesh_unit):
    from ect_analysis.mesh import tetra_volumes
    counts = {shape: np.zeros(grid.n_bins, dtype=int) for shape, grid in grids.items()}
    for meta in metas:
        vertices, simplices = load_geometry(meta, mesh_unit)
        centres = vertices[simplices].mean(axis=1)
        volumes = tetra_volumes(vertices, simplices)
        for shape, grid in grids.items():
            bins = point_bins(centres, grid)
            valid = bins >= 0
            occupied = np.bincount(bins[valid], weights=volumes[valid], minlength=grid.n_bins) > 0
            counts[shape] += occupied
    masks = {shape: count == len(metas) for shape, count in counts.items()}
    for shape, mask in masks.items():
        if not np.any(mask):
            raise ValueError(f"No common centroid-occupied bins on grid {grid_name(shape)}")
    return masks, counts


def map_rules(sensitivity, vertices, simplices, grids, levels, sensitivity_kind,
              block_cells=4096, point_budget=262144, progress_label=None):
    from ect_analysis.mesh import tetra_volumes
    sensitivity = np.asarray(sensitivity)
    n_cells, n_channels = sensitivity.shape
    volumes = tetra_volumes(vertices, simplices)
    centres = vertices[simplices].mean(axis=1)
    centroid_bins = {shape: point_bins(centres, grid) for shape, grid in grids.items()}
    mapped, occupied, geometry = {}, {}, {}
    total_native_volume = float(volumes.sum())
    for shape, grid in grids.items():
        for level in levels:
            key = (shape, level)
            mapped[key] = np.zeros((grid.n_bins, n_channels), dtype=float)
            occupied[key] = np.zeros(grid.n_bins, dtype=float)
            geometry[key] = {"multibin_volume": 0., "moved_volume": 0., "max_sampled_bins": 0,
                             "expected_total": np.zeros(n_channels), "absolute_total": np.zeros(n_channels)}
        # Match the production centroid implementation, including summation order.
        if 0 in levels:
            bins = centroid_bins[shape]
            valid = bins >= 0
            rows = np.flatnonzero(valid)
            weights = volumes[valid] if sensitivity_kind == "density" else np.ones(len(rows))
            projector = csr_matrix((weights, (bins[valid], rows)), shape=(grid.n_bins, n_cells))
            mapped[(shape, 0)] = np.asarray(projector @ sensitivity, dtype=float)
            occupied[(shape, 0)] = np.bincount(bins[valid], weights=volumes[valid], minlength=grid.n_bins)
    for level in levels:
        rule = subdivision_rule(level)
        q = len(rule)
        step = min(block_cells, max(1, point_budget // q))
        next_progress = .25
        for first in range(0, n_cells, step):
            last = min(first+step, n_cells)
            v = volumes[first:last]
            contribution = np.array(sensitivity[first:last], dtype=float, copy=True)
            if sensitivity_kind == "density":
                contribution *= v[:, None]
            xyz = (centres[first:last, None, :] if level == 0 else
                   np.einsum("qa,nad->nqd", rule, vertices[simplices[first:last]], optimize=True))
            for shape, grid in grids.items():
                key = (shape, level)
                bins = point_bins(xyz, grid)
                valid = bins >= 0
                fractions = valid.sum(axis=1) / q
                geo = geometry[key]
                geo["expected_total"] += fractions @ contribution
                geo["absolute_total"] += fractions @ np.abs(contribution)
                same = valid & (bins == centroid_bins[shape][first:last, None])
                geo["moved_volume"] += float(v @ ((valid & ~same).sum(axis=1)/q))
                ordered = np.sort(bins, axis=1)
                distinct = ((ordered[:, 1:] != ordered[:, :-1]) & (ordered[:, 1:] >= 0)).sum(axis=1)
                distinct += ordered[:, 0] >= 0
                geo["multibin_volume"] += float(v[distinct > 1].sum())
                geo["max_sampled_bins"] = max(geo["max_sampled_bins"], int(distinct.max()))
                if level == 0 or not valid.any():
                    continue
                flat = bins[valid]
                local_bins, inverse = np.unique(flat, return_inverse=True)
                cell_ids = np.repeat(np.arange(last-first), q)[valid.ravel()]
                projector = csr_matrix((np.full(len(flat), 1/q), (inverse, cell_ids)),
                                       shape=(len(local_bins), last-first))
                mapped[key][local_bins] += projector @ contribution
                occupied[key] += np.bincount(flat, weights=np.repeat(v/q, q)[valid.ravel()],
                                             minlength=grid.n_bins)
            if progress_label and n_cells >= 100000 and last/n_cells >= next_progress:
                print(f"  {progress_label}: Q={q}, {100*last/n_cells:.0f}%", flush=True)
                next_progress += .25
        if progress_label:
            print(f"  {progress_label}: Q={q} complete", flush=True)
    rows = []
    for (shape, level), array in mapped.items():
        grid, geo = grids[shape], geometry[(shape, level)]
        numerator = np.abs(array.sum(axis=0)-geo["expected_total"])
        scale = max(float(geo["absolute_total"].max()), np.finfo(float).tiny)
        conservation_error = float(numerator.max()/scale)
        if conservation_error > 1e-9:
            raise FloatingPointError("Mapped channel integrals failed conservation check")
        ratio = volumes/grid.bin_volume_mm3
        rows.append({
            "case": case_name(shape, level), "grid": grid_name(shape), "subdivision_level": level,
            "quadrature_points_per_tetra": 8**level, "total_bins": grid.n_bins,
            "bin_volume_mm3": grid.bin_volume_mm3, "native_volume_mm3": total_native_volume,
            "mapped_native_volume_fraction": float(occupied[(shape, level)].sum()/total_native_volume),
            "occupied_bins": int((occupied[(shape, level)] > 0).sum()),
            "native_cell_volume_to_bin_volume_median": float(np.median(ratio)),
            "native_cell_volume_to_bin_volume_p95": float(np.quantile(ratio, .95)),
            "native_cell_volume_to_bin_volume_max": float(ratio.max()),
            "sampled_multibin_native_volume_fraction": geo["multibin_volume"]/total_native_volume,
            "sampled_volume_moved_from_centroid_bin_fraction": geo["moved_volume"]/total_native_volume,
            "maximum_sampled_bins_per_tetra": geo["max_sampled_bins"],
            "relative_channel_integral_conservation_error": conservation_error,
        })
    return mapped, occupied, rows


def relative_norm(new, reference):
    denominator = float(np.linalg.norm(reference))
    return float(np.linalg.norm(new-reference)/denominator) if denominator > 0 else float("nan")


def overlap(selected, reference):
    a, b = set(selected), set(reference)
    shared = len(a & b)
    return shared, shared/len(a) if a else np.nan, shared/len(a | b) if a | b else np.nan


def subset_information(gram, alpha, selected):
    indices = np.asarray(selected, dtype=int)
    matrix = np.eye(len(indices)) + gram[np.ix_(indices, indices)]/alpha
    factor = np.linalg.cholesky(matrix)
    return float(2*np.log(np.diag(factor)).sum())


def make_states(label, maps, occupied, grids, masks, levels, args):
    from ect_analysis.metrics import gram_and_eigenvalues, spatial_metrics
    states = []
    for shape, grid in grids.items():
        mask_common = masks[shape]
        for scope in SCOPES:
            mask = mask_common if scope == SCOPES[0] else np.ones(grid.n_bins, dtype=bool)
            volumes = np.full(int(mask.sum()), grid.bin_volume_mm3)
            centroid = maps[(shape, 0)][mask]
            previous = None
            for level in levels:
                array = maps[(shape, level)][mask]
                gram, eig = gram_and_eigenvalues(array, args.score_noise_std)
                spatial, _, _ = spatial_metrics(array, args.score_noise_std, volumes)
                density = {name+"_density": spatial[name]/grid.bin_volume_mm3 for name in SPATIAL_AMPLITUDES}
                state = {
                    "configuration": label, "case": case_name(shape, level), "scope": scope,
                    "grid": grid_name(shape), "grid_r": shape[0], "grid_theta": shape[1], "grid_z": shape[2],
                    "subdivision_level": level, "quadrature_points_per_tetra": 8**level,
                    "available_channels": gram.shape[0], "basis_parameters": int(mask.sum()),
                    "total_bins": grid.n_bins, "bin_volume_mm3": grid.bin_volume_mm3,
                    "domain_volume_mm3": float(volumes.sum()),
                    "domain_fraction_of_cylinder": float(mask.mean()),
                    "centroid_common_bins": int(mask_common.sum()),
                    "centroid_common_domain_fraction": float(mask_common.mean()),
                    "occupied_basis_bins": int((occupied[(shape, level)][mask] > 0).sum()),
                    "geometric_volume_in_basis_mm3": float(occupied[(shape, level)][mask].sum()),
                    "relative_operator_frobenius_to_same_grid_centroid": relative_norm(array, centroid),
                    "relative_operator_frobenius_to_previous_quadrature": relative_norm(array, previous) if previous is not None else 0.,
                    "previous_quadrature_points": 8**levels[levels.index(level)-1] if previous is not None else 1,
                    "zero_operator": bool(not np.any(array)),
                    **spatial, **density, "_gram": gram, "_eig": eig,
                }
                states.append(state)
                previous = array
    return states


def score_states(states, primary_shape, args, options):
    from ect_analysis.channels import channel_table, greedy_regularized_logdet
    from ect_analysis.metrics import regularized_spectral_metrics
    ref_case = case_name(primary_shape, 0)
    primary = [s for s in states if s["case"] == ref_case and s["scope"] == SCOPES[0]]
    reference_scale = float(np.median([np.trace(s["_gram"])/s["available_channels"] for s in primary]))
    if reference_scale <= 0:
        raise ValueError("Reference regularisation scale is not positive")
    alpha_ref = args.alpha_rel*reference_scale
    voxel_ref = primary[0]["bin_volume_mm3"]
    alpha_density = alpha_ref/voxel_ref
    spectra, channel_rows = [], []
    definitions = channel_table(primary[0]["available_channels"]).set_index("channel_index_zero_based").to_dict(orient="index")
    for s in states:
        gram, eig = s["_gram"], s["_eig"]
        alpha = alpha_ref if s["grid"] == grid_name(primary_shape) else alpha_density*s["bin_volume_mm3"]
        # Only shape is needed when a precomputed Gram/eigenspectrum is supplied.
        shape_view = np.broadcast_to(np.array(0.), (s["basis_parameters"], s["available_channels"]))
        spectral = regularized_spectral_metrics(shape_view, alpha, args.score_noise_std, gram, eig)
        s.update(spectral)
        s.update({"reference_case": ref_case, "alpha_reference": alpha_ref,
                  "alpha_grid": alpha, "alpha_density": alpha_density,
                  "reference_bin_volume_mm3": voxel_ref,
                  "spectral_energy_volume_normalized": spectral["spectral_energy"]/s["bin_volume_mm3"],
                  "largest_eigenvalue_volume_normalized": spectral["largest_eigenvalue"]/s["bin_volume_mm3"]})
        for rank, value in enumerate(eig[::-1], 1):
            spectra.append({"configuration": s["configuration"], "case": s["case"], "scope": s["scope"],
                            "mode_rank_descending": rank, "gram_eigenvalue_raw": value,
                            "gram_eigenvalue_volume_normalized": value/s["bin_volume_mm3"],
                            "alpha_density": alpha_density})
        if not options.no_selection:
            start = time.perf_counter()
            selected = greedy_regularized_logdet(gram, alpha, args.subset_k)[0]
            s["selection_seconds"] = time.perf_counter()-start
            s["_selected"] = selected
            s["selected_channels"] = len(selected)
            information = subset_information(gram, alpha, selected)
            s["selected_regularized_information"] = information
            s["selected_information_retention"] = information/spectral["regularized_logdet"] if spectral["regularized_logdet"] > 0 else np.nan
            energy = float(np.diag(gram)[selected].sum())
            s["selected_spectral_energy_retention"] = energy/spectral["spectral_energy"] if spectral["spectral_energy"] > 0 else np.nan
            for order, channel in enumerate(selected, 1):
                channel_rows.append({"configuration": s["configuration"], "case": s["case"], "scope": s["scope"],
                                     "selection_order": order, "channel_index_zero_based": channel,
                                     **definitions[channel]})
    refs = {(s["configuration"], s["scope"]): s for s in states if s["case"] == ref_case}
    same_grid_refs = {(s["configuration"], s["scope"], s["grid"]): s for s in states if s["subdivision_level"] == 0}
    for s in states:
        ref = refs[(s["configuration"], s["scope"])]
        same = same_grid_refs[(s["configuration"], s["scope"], s["grid"])]
        normalised = s["_gram"]/s["bin_volume_mm3"]
        s["relative_volume_normalized_gram_to_reference"] = relative_norm(normalised, ref["_gram"]/ref["bin_volume_mm3"])
        if not options.no_selection:
            for suffix, target in [("reference", ref), ("same_grid_centroid", same)]:
                shared, fraction, jaccard = overlap(s["_selected"], target["_selected"])
                s[f"shared_channels_{suffix}"] = shared
                s[f"channel_overlap_fraction_{suffix}"] = fraction
                s[f"channel_jaccard_{suffix}"] = jaccard
    return {"common_reference_scale": reference_scale, "alpha_reference": alpha_ref,
            "alpha_density": alpha_density, "reference_bin_volume_mm3": voxel_ref}, spectra, channel_rows


def result_frames(states, geometry, support_rows, spectra, channels):
    rows = [{k:v for k,v in s.items() if not k.startswith("_")} for s in states]
    frame = pd.DataFrame(rows)
    reference_case = frame.reference_case.iloc[0]
    refs = frame[frame.case == reference_case].set_index(["configuration", "scope"])
    same_refs = frame[frame.subdivision_level == 0].set_index(["configuration", "scope", "grid"])
    changes = []
    for row in rows:
        ref = refs.loc[(row["configuration"], row["scope"])]
        same = same_refs.loc[(row["configuration"], row["scope"], row["grid"])]
        change = {k:row[k] for k in ("configuration", "case", "scope", "grid", "quadrature_points_per_tetra", "basis_parameters", "domain_fraction_of_cylinder")}
        change["reference_case"] = reference_case
        change["reference_domain_fraction_of_cylinder"] = ref.domain_fraction_of_cylinder
        change["physical_domain_held_fixed_in_comparison"] = row["scope"] == SCOPES[1] or row["grid"] == ref.grid
        for name in COMPARISON_METRICS:
            value = float(row[name])
            for suffix, target in [("reference", ref), ("same_grid_centroid", same)]:
                base = float(target[name])
                change[f"{name}_difference_vs_{suffix}"] = value-base
                change[f"{name}_relative_percent_vs_{suffix}"] = 100*(value/base-1) if np.isfinite(base) and base != 0 else np.nan
        changes.append(change)
    changes = pd.DataFrame(changes)
    summary = []
    for (case, scope), group in changes.groupby(["case", "scope"], sort=True):
        row = {"case": case, "scope": scope, "configurations": len(group),
               "domain_fraction_of_cylinder": group.domain_fraction_of_cylinder.iloc[0]}
        for name in [c for c in changes.columns if "_relative_percent_vs_" in c or "_difference_vs_" in c]:
            values = group[name].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            for stat, function in [("min", np.min), ("mean", np.mean), ("median", np.median), ("max", np.max)]:
                row[f"{name}_{stat}"] = float(function(values)) if len(values) else np.nan
        summary.append(row)
    ranks = []
    for (case, scope), group in frame.groupby(["case", "scope"], sort=True):
        reference = frame[(frame.case == reference_case) & (frame.scope == scope)].set_index("configuration")
        for name in COMPARISON_METRICS:
            a = group[name].to_numpy(dtype=float)
            b = reference.loc[group.configuration, name].to_numpy(dtype=float)
            valid = np.isfinite(a) & np.isfinite(b)
            coefficient = float(spearmanr(a[valid], b[valid]).statistic) if valid.sum() >= 2 and np.unique(a[valid]).size > 1 and np.unique(b[valid]).size > 1 else np.nan
            ranks.append({"case": case, "scope": scope, "metric": name, "reference_case": reference_case,
                          "finite_paired_configurations": int(valid.sum()), "spearman_rho": coefficient})
    return {
        "mapping_scorecard.csv": frame, "mapping_changes.csv": changes,
        "mapping_summary.csv": pd.DataFrame(summary), "mapping_geometry.csv": pd.DataFrame(geometry),
        "mapping_common_support.csv": pd.DataFrame(support_rows), "mapping_rank_stability.csv": pd.DataFrame(ranks),
        "mapping_gram_spectra.csv": pd.DataFrame(spectra), "mapping_selected_channels.csv": pd.DataFrame(channels),
    }


def run_study(args, options, output):
    from ect_analysis import mesh
    from ect_analysis.cli import discover_files
    from ect_analysis.reporting import sha256_file
    paths = discover_files(args)
    metas = [mesh.inspect_mesh(p, args.mesh_unit) for p in paths]
    if len({m.n_channels for m in metas}) != 1:
        raise ValueError("All configurations must share the same channel count")
    primary_shape = (args.grid_r, args.grid_theta, args.grid_z)
    shapes = options.mapping_grids or [tuple(max(1, n//2) for n in primary_shape), primary_shape, tuple(2*n for n in primary_shape)]
    shapes = sorted(set(shapes) | {primary_shape}, key=lambda s:(math.prod(s), s))
    grids = {shape:mesh.make_common_grid(metas, *shape) for shape in shapes}
    levels = sorted(set(options.subdivision_levels) | {0})
    print("Inspecting original centroid common supports...", flush=True)
    masks, counts = common_supports(metas, grids, args.mesh_unit)
    support_rows = []
    for shape, grid in grids.items():
        for i, xyz in enumerate(grid.xyz_centres_mm):
            support_rows.append({"grid":grid_name(shape), "bin_index_zero_based":i,
                                 "bin_volume_mm3":grid.bin_volume_mm3, "x_mm":xyz[0], "y_mm":xyz[1], "z_mm":xyz[2],
                                 "centroid_occupied_configurations":int(counts[shape][i]),
                                 "in_centroid_common_support":bool(masks[shape][i])})
    states, geometry, inputs = [], [], []
    start = time.perf_counter()
    for meta in metas:
        print(f"MAPPING {meta.label}: {meta.n_cells:,} native tetrahedra", flush=True)
        path = Path(meta.path)
        initial = path.stat()
        record = {**asdict(meta), "sha256":sha256_file(path)}
        sensitivity, vertices, simplices = mesh.load_mesh(meta, args.mesh_unit, np.dtype(args.dtype))
        maps, occupied, geo_rows = map_rules(sensitivity, vertices, simplices, grids, levels, args.sensitivity_kind,
                                            options.mapping_block_cells, options.quadrature_point_budget, meta.label)
        states.extend(make_states(meta.label, maps, occupied, grids, masks, levels, args))
        for row in geo_rows:
            shape = tuple(map(int, row["grid"].split("x")))
            row.update({"configuration":meta.label, "native_cells":meta.n_cells,
                        "centroid_common_bins":int(masks[shape].sum()),
                        "centroid_common_domain_fraction":float(masks[shape].mean())})
        geometry.extend(geo_rows)
        current = path.stat()
        if (current.st_size, current.st_mtime_ns) != (initial.st_size, initial.st_mtime_ns):
            raise RuntimeError(f"Input file changed while being analysed: {path.name}")
        inputs.append(record)
        del maps, occupied, sensitivity, vertices, simplices
    mapping_and_global_seconds = time.perf_counter()-start
    print("Scoring with the reference regularisation density and comparing cases...", flush=True)
    regularisation, spectra, channels = score_states(states, primary_shape, args, options)
    frames = result_frames(states, geometry, support_rows, spectra, channels)
    for name, frame in frames.items():
        frame.to_csv(output/name, index=False, encoding="utf-8-sig")
    return {
        "completed_configurations":[m.label for m in metas], "inputs":inputs,
        "grid_shapes":shapes, "subdivision_levels":levels,
        "primary_grid":primary_shape, "primary_centroid_common_parameters":int(masks[primary_shape].sum()),
        "regularization":regularisation, "mapping_and_global_metrics_seconds":mapping_and_global_seconds,
        "scorecard_rows":len(frames["mapping_scorecard.csv"]),
        "available_channels":metas[0].n_channels,
        "study_dimensions_match_manuscript":(len(metas)==15 and [m.label for m in metas]==list("ABCDEFGHIJKLMNO")
                                             and metas[0].n_channels==496 and primary_shape==(10,16,20)
                                             and int(masks[primary_shape].sum())==2851
                                             and args.sensitivity_kind=="density" and args.score_noise_std==1
                                             and args.alpha_rel==.001),
    }


def numerical_self_test():
    from ect_analysis.mesh import make_common_grid, tetra_volumes
    from ect_analysis.models import MeshMeta
    from ect_analysis.selftest import run_self_test
    run_self_test()
    unit = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    for level in range(4):
        bary = subdivision_rule(level)
        assert bary.shape == (8**level,4)
        assert np.all(bary >= 0) and np.allclose(bary.sum(axis=1),1)
        assert np.allclose(bary.mean(axis=0),.25)
    children = child_tetrahedra(unit)
    child_v = tetra_volumes(children.reshape(-1,3), np.arange(32).reshape(-1,4))
    assert np.allclose(child_v,1/48) and np.isclose(child_v.sum(),1/6)
    meta = MeshMeta("test","unused",1,4,2,2.,-.18,1.)
    grid = make_common_grid([meta],1,1,2)
    sensitivity = np.array([[6.,-3.]])
    maps, occupied, rows = map_rules(sensitivity, unit, np.array([[0,1,2,3]]), {(1,1,2):grid}, [0,1,2], "density",1,128)
    exact_upper = (1-.41)**3
    errors = [abs(maps[((1,1,2),level)][1,0]-exact_upper) for level in [0,1,2]]
    assert errors[2] < errors[0] and errors[2] < .02
    for level in [0,1,2]:
        assert np.allclose(maps[((1,1,2),level)].sum(axis=0),[1.,-.5])
        assert np.isclose(occupied[((1,1,2),level)].sum(),1/6)
    integrated, _, _ = map_rules(sensitivity/6, unit, np.array([[0,1,2,3]]), {(1,1,2):grid}, [0,1,2], "element-integrated",1,128)
    assert all(np.allclose(maps[key],integrated[key]) for key in maps)
    # Independent analytic constant-field check of cross-grid volume scaling.
    from ect_analysis.metrics import gram_and_eigenvalues, regularized_spectral_metrics, spatial_metrics
    density = np.array([2.,-1.])
    alpha_density = .7
    invariants = []
    for shape in [(1,1,2),(2,4,4),(4,8,8)]:
        g = make_common_grid([meta],*shape)
        array = np.tile(density*g.bin_volume_mm3,(g.n_bins,1))
        gram,eig = gram_and_eigenvalues(array,1.)
        spectral = regularized_spectral_metrics(array,alpha_density*g.bin_volume_mm3,1.,gram,eig)
        spatial,_,_ = spatial_metrics(array,1.,np.full(g.n_bins,g.bin_volume_mm3))
        invariants.append([spectral["regularized_logdet"],spectral["spectral_energy"]/g.bin_volume_mm3,
                           spatial["envelope_p50"]/g.bin_volume_mm3])
    assert np.allclose(invariants,invariants[0],rtol=1e-11,atol=1e-11)
    print("MAPPING SENSITIVITY SELF-TEST PASSED",flush=True)


def helper_options(argv):
    parser = argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument("--mapping-grids",nargs="+",default=None)
    parser.add_argument("--subdivision-levels",nargs="+",type=int,default=[0,1,2])
    parser.add_argument("--mapping-block-cells",type=int,default=4096)
    parser.add_argument("--quadrature-point-budget",type=int,default=262144)
    parser.add_argument("--no-selection",action="store_true")
    options, original = parser.parse_known_args(argv)
    if options.mapping_grids:
        try:
            options.mapping_grids = [tuple(map(int,s.lower().replace(",","x").split("x"))) for s in options.mapping_grids]
            if any(len(s)!=3 or min(s)<=0 for s in options.mapping_grids):raise ValueError()
        except ValueError:
            parser.error("--mapping-grids requires positive rxthetaxz triples")
    if not options.subdivision_levels or any(v<0 or v>3 for v in options.subdivision_levels):
        parser.error("--subdivision-levels must contain values between 0 and 3")
    if options.mapping_block_cells<=0 or options.quadrature_point_budget<=0:
        parser.error("Mapping block size and quadrature point budget must be positive")
    return options, original


def resolve_arguments(original):
    from ect_analysis.cli import parse_args
    if original:
        args,mode = parse_args(original),"original_command_line_parser"
    else:
        import main as project_main
        args,mode = (project_main.pycharm_arguments(),"main.py PYCHARM_CONFIG") if project_main.USE_PYCHARM_CONFIG else (parse_args([]),"original_command_line_defaults")
    return argparse.Namespace(**dict(vars(args))),mode


def main(argv=None):
    if not (PROJECT_DIRECTORY/"ect_analysis"/"mesh.py").is_file():
        raise RuntimeError("Place analyze_mapping_sensitivity.py beside main.py")
    from ect_analysis import mesh
    from ect_analysis.cli import validate_args
    from ect_analysis.reporting import json_ready,sha256_file
    options,original = helper_options(list(sys.argv[1:] if argv is None else argv))
    args,mode = resolve_arguments(original)
    if args.self_test:
        numerical_self_test()
        return 0
    validate_args(args)
    explicit_output = any(v=="--output-dir" or v.startswith("--output-dir=") for v in original)
    root = args.output_dir if explicit_output else PROJECT_DIRECTORY/"mapping_sensitivity_results"
    output = (root/datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")).resolve()
    output.mkdir(parents=True,exist_ok=False)
    metadata = {
        "created_utc":datetime.now(timezone.utc).isoformat(),"configuration_source":mode,
        "source_parameters":dict(vars(args)),"mapping_options":vars(options),
        "platform":platform.platform(),
        "software":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,
                    "pandas":pd.__version__,"h5py":getattr(mesh.h5py,"__version__",None)},
        "notes":[
            "Quadrature samples equal-volume child-tetrahedron centroids of the unchanged native piecewise-constant sensitivity.",
            "This is approximate volume integration, not exact curved tetrahedron--bin intersection or an independent FEM solution.",
            "Finite subdivision levels do not prove asymptotic convergence or guaranteed monotonic error reduction.",
            "Within each grid the original centroid common support is fixed for all quadrature levels.",
            "Between grids the centroid-common support and its physical coverage can change; both are reported.",
            "The full-cylinder scope includes zero bins on a fixed physical domain and isolates this domain-coverage confound.",
            "Reference-grid centroid/common-support metrics use the original shared-alpha rule and original greedy routine.",
            "Cross-grid comparison uses G/bin_volume and spatial S/bin_volume with fixed alpha_density=alpha_ref/bin_volume_ref.",
            "Equivalent raw alpha_grid scales with bin volume; alpha is not separately recalibrated for variants.",
            "This volume scaling corresponds to an L2 volume penalty on the piecewise-constant parameter field.",
            "Cross-grid raw integrated sensitivities and raw spectral energy must not be compared without their volume scaling.",
            "Frobenius operator comparisons are within a grid; channel-Gram comparisons use volume-normalised matrices.",
            "Reported geometric multibin fractions are quadrature diagnostics, not exact intersection statistics.",
            "Times describe this analysis execution, not reconstruction or acquisition performance.",
            "No original source, native input, or previous analysis output is overwritten.",
        ],
    }
    print(f"Mapping sensitivity output: {output}",flush=True)
    start,error = time.perf_counter(),None
    try:
        metadata.update(run_study(args,options,output))
        metadata["complete"] = True
        return 0
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        metadata["complete"] = False
        raise
    finally:
        metadata["error"] = error
        metadata["complete_analysis_elapsed_seconds"] = time.perf_counter()-start
        paths = [Path(__file__),PROJECT_DIRECTORY/"main.py",*sorted((PROJECT_DIRECTORY/"ect_analysis").glob("*.py"))]
        metadata["source_sha256"] = {p.relative_to(PROJECT_DIRECTORY).as_posix():sha256_file(p) for p in paths if p.is_file()}
        (output/"mapping_manifest.json").write_text(json.dumps(metadata,indent=2,default=json_ready),encoding="utf-8")
        print(f"Mapping sensitivity files saved in: {output}",flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
