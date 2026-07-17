from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
import pandas as pd
import scipy

from .channels import channel_table, greedy_regularized_logdet, group_masks
from .cli import discover_files, parse_args, validate_args
from .constants import EXPECTED_CHANNELS
from .mesh import h5py, inspect_mesh, make_common_grid, map_to_common_grid
from .metrics import (
    gram_and_eigenvalues,
    regularized_spectral_metrics,
    spatial_metrics,
    weighted_percentiles,
)
from .models import MappedDataset
from .phantoms import (
    common_bin_quadrature_points,
    generate_phantom_positions,
    paired_phantom_comparisons,
    phantom_truth_matrix,
    run_phantoms,
    summarize_phantoms,
)
from .plotting import plot_cdfs, plot_phantom_maps, plot_singular_spectra, plot_subset_curves
from .reporting import json_ready, save_dataframe, sha256_file
from .selftest import run_self_test


def run_analysis(
    argv: Sequence[str] | None = None,
    args_override: argparse.Namespace | None = None,
    entrypoint_path: Path | None = None,
) -> int:
    args = args_override if args_override is not None else parse_args(argv)
    run_mode = "pycharm_configuration" if args_override is not None else "command_line"
    if args.self_test:
        run_self_test()
        return 0
    analyses = validate_args(args)
    paths = discover_files(args)
    dtype = np.float64 if args.dtype == "float64" else np.float32
    output = args.output_dir.resolve()
    tables_dir = output / "tables"
    figures_dir = output / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metas = [inspect_mesh(path, args.mesh_unit) for path in paths]
    channel_counts = {meta.n_channels for meta in metas}
    if len(channel_counts) != 1:
        raise ValueError(
            f"All configurations must contain the same measurement channels; found {sorted(channel_counts)}"
        )
    quality_rows: list[dict[str, str]] = []
    for meta in metas:
        status = "pass" if meta.n_channels == EXPECTED_CHANNELS else "warning"
        quality_rows.append(
            {
                "check": f"{meta.label}: channel count",
                "status": status,
                "detail": f"{meta.n_channels} channels; expected {EXPECTED_CHANNELS}",
            }
        )
    if len(metas) != 15:
        quality_rows.append(
            {
                "check": "configuration count",
                "status": "warning",
                "detail": f"{len(metas)} configurations supplied; publication table expects 15",
            }
        )
    else:
        quality_rows.append(
            {"check": "configuration count", "status": "pass", "detail": "15 configurations"}
        )

    grid = make_common_grid(metas, args.grid_r, args.grid_theta, args.grid_z)
    mapped: dict[str, MappedDataset] = {}
    print(f"Mapping {len(metas)} configurations to {grid.n_bins} equal-volume 3D bins...")
    for meta in metas:
        print(f"  {meta.label}: {Path(meta.path).name} ({meta.n_cells:,} cells)")
        mapped[meta.label] = map_to_common_grid(
            meta, grid, args.mesh_unit, args.sensitivity_kind, dtype
        )

    occupancy_stack = np.vstack([item.occupied_volume_mm3 > 0 for item in mapped.values()])
    common_mask = np.all(occupancy_stack, axis=0)
    n_common = int(common_mask.sum())
    if n_common == 0:
        raise ValueError("No common grid bin is occupied in every configuration")
    common_xyz = grid.xyz_centres_mm[common_mask]
    common_volumes = np.full(n_common, grid.bin_volume_mm3)
    common_s = {label: item.sensitivity[common_mask] for label, item in mapped.items()}
    quality_rows.append(
        {
            "check": "common physical basis",
            "status": "pass" if n_common >= EXPECTED_CHANNELS else "warning",
            "detail": f"{n_common}/{grid.n_bins} bins occupied by every mesh",
        }
    )

    grams: dict[str, np.ndarray] = {}
    eigenvalues_by_label: dict[str, np.ndarray] = {}
    reference_scales = []
    for label, sensitivity in common_s.items():
        gram, eigenvalues = gram_and_eigenvalues(sensitivity, args.score_noise_std)
        grams[label] = gram
        eigenvalues_by_label[label] = eigenvalues
        reference_scales.append(float(np.trace(gram) / gram.shape[0]))
    common_reference = float(np.median(reference_scales))
    if common_reference <= 0:
        raise ValueError("Common spectral reference scale is not positive")
    alpha = args.alpha_rel * common_reference

    global_rows = []
    alpha_rows = []
    local_by_label: dict[str, np.ndarray] = {}
    envelope_by_label: dict[str, np.ndarray] = {}
    for label, sensitivity in common_s.items():
        meta = mapped[label].meta
        spectral = regularized_spectral_metrics(
            sensitivity,
            alpha,
            args.score_noise_std,
            grams[label],
            eigenvalues_by_label[label],
        )
        spatial, local_norm, envelope = spatial_metrics(
            sensitivity, args.score_noise_std, common_volumes
        )
        local_by_label[label] = local_norm
        envelope_by_label[label] = envelope
        global_rows.append(
            {
                "configuration": label,
                "native_cells": meta.n_cells,
                "channels": meta.n_channels,
                "common_bins": n_common,
                "mapped_native_volume_fraction": mapped[label].mapped_native_volume_fraction,
                "alpha_common": alpha,
                **spectral,
                **spatial,
            }
        )
        for alpha_rel in sorted(set(args.alpha_sweep) | {args.alpha_rel}):
            alpha_test = alpha_rel * common_reference
            robust = regularized_spectral_metrics(
                sensitivity,
                alpha_test,
                args.score_noise_std,
                grams[label],
                eigenvalues_by_label[label],
            )
            alpha_rows.append(
                {
                    "configuration": label,
                    "alpha_relative": alpha_rel,
                    "alpha_absolute_common": alpha_test,
                    **robust,
                }
            )
    global_frame = pd.DataFrame(global_rows).sort_values("configuration")
    alpha_frame = pd.DataFrame(alpha_rows).sort_values(["alpha_relative", "configuration"])
    gini_ok = global_frame["envelope_gini"].between(0.0, 1.0, inclusive="both").all()
    quality_rows.append(
        {
            "check": "Gini bounds",
            "status": "pass" if gini_ok else "fail",
            "detail": (
                f"global envelope Gini range {global_frame['envelope_gini'].min():.4f}--"
                f"{global_frame['envelope_gini'].max():.4f}"
            ),
        }
    )
    resolution_family = global_frame[
        global_frame["configuration"].isin(["A", "B", "C"])
    ]
    if len(resolution_family) == 3:
        scale_ratio = float(
            resolution_family["local_norm_p50"].max()
            / resolution_family["local_norm_p50"].min()
        )
        quality_rows.append(
            {
                "check": "A/B/C mesh-resolution scale consistency",
                "status": "pass" if scale_ratio <= 2.0 else "warning",
                "detail": (
                    f"max/min common-basis local_norm_p50 = {scale_ratio:.3f}; "
                    f"sensitivity_kind={args.sensitivity_kind}"
                ),
            }
        )
    save_dataframe(global_frame, tables_dir / "global_common_basis_scorecard.csv")
    save_dataframe(alpha_frame, tables_dir / "regularization_sensitivity.csv")
    plot_singular_spectra(
        eigenvalues_by_label, alpha, figures_dir / "common_basis_singular_spectra.png", args.figure_dpi
    )
    plot_cdfs(
        local_by_label,
        common_volumes,
        "Unit-noise local sensitivity norm",
        "Common-basis local sensitivity distributions",
        figures_dir / "local_sensitivity_cdfs.png",
        args.figure_dpi,
    )
    normalized_envelope = {
        label: values / max(weighted_percentiles(values, common_volumes, (99,))[0], 1e-30)
        for label, values in envelope_by_label.items()
    }
    plot_cdfs(
        normalized_envelope,
        common_volumes,
        r"Sensitivity envelope $E/E_{99}$",
        "Common-basis coverage distributions",
        figures_dir / "coverage_cdfs.png",
        args.figure_dpi,
    )

    channels = channel_table(metas[0].n_channels)
    save_dataframe(channels, tables_dir / "channel_definitions.csv")

    if "groups" in analyses:
        group_rows = []
        masks = group_masks(channels)
        for label, sensitivity in common_s.items():
            total_energy = float(np.sum(sensitivity**2))
            for group_name, mask in masks.items():
                if not mask.any():
                    continue
                group_s = sensitivity[:, mask]
                group_spectral = regularized_spectral_metrics(
                    group_s, alpha, args.score_noise_std
                )
                group_spatial, _, _ = spatial_metrics(group_s, args.score_noise_std, common_volumes)
                group_energy = float(np.sum(group_s**2))
                group_rows.append(
                    {
                        "configuration": label,
                        "group": group_name,
                        "n_channels": int(mask.sum()),
                        "energy_fraction_of_all_channels": group_energy / total_energy,
                        "mean_energy_per_channel": group_energy / int(mask.sum()),
                        **group_spectral,
                        **group_spatial,
                    }
                )
        group_frame = pd.DataFrame(group_rows).sort_values(["configuration", "group"])
        group_gini_ok = group_frame["envelope_gini"].between(
            0.0, 1.0, inclusive="both"
        ).all()
        quality_rows.append(
            {
                "check": "group Gini bounds",
                "status": "pass" if group_gini_ok else "fail",
                "detail": (
                    f"group envelope Gini range {group_frame['envelope_gini'].min():.4f}--"
                    f"{group_frame['envelope_gini'].max():.4f}"
                ),
            }
        )
        save_dataframe(group_frame, tables_dir / "group_scorecards.csv")

    if "subsets" in analyses:
        subset_rows = []
        selection_rows = []
        composition_rows = []
        curves: dict[str, Sequence[float]] = {}
        for label, sensitivity in common_s.items():
            selected, cumulative, marginal = greedy_regularized_logdet(
                grams[label], alpha, args.subset_k
            )
            curves[label] = cumulative
            subset_s = sensitivity[:, selected]
            subset_spectral = regularized_spectral_metrics(
                subset_s, alpha, args.score_noise_std
            )
            subset_spatial, _, _ = spatial_metrics(
                subset_s, args.score_noise_std, common_volumes
            )
            subset_rows.append(
                {
                    "configuration": label,
                    "selected_channels": len(selected),
                    "alpha_common": alpha,
                    **subset_spectral,
                    **subset_spatial,
                }
            )
            channel_lookup = channels.set_index("channel_index_zero_based")
            for rank, channel in enumerate(selected, start=1):
                channel_info = channel_lookup.loc[channel].to_dict()
                selection_rows.append(
                    {
                        "configuration": label,
                        "selection_rank": rank,
                        "channel_index_zero_based": channel,
                        **channel_info,
                        "marginal_regularized_logdet": marginal[rank - 1],
                        "cumulative_regularized_logdet": cumulative[rank - 1],
                    }
                )
            selected_classes = channel_lookup.loc[selected, "channel_class"].astype(str).tolist()
            cutoffs = sorted(set([min(x, len(selected)) for x in (16, 32, 64, args.subset_k)]))
            for cutoff in cutoffs:
                counts = pd.Series(selected_classes[:cutoff]).value_counts()
                for class_name, count in counts.items():
                    composition_rows.append(
                        {
                            "configuration": label,
                            "K": cutoff,
                            "channel_class": class_name,
                            "count": int(count),
                            "fraction": float(count / cutoff),
                        }
                    )
        save_dataframe(
            pd.DataFrame(subset_rows).sort_values("configuration"),
            tables_dir / "subset_scorecard.csv",
        )
        save_dataframe(
            pd.DataFrame(selection_rows).sort_values(["configuration", "selection_rank"]),
            tables_dir / "selected_channels.csv",
        )
        save_dataframe(
            pd.DataFrame(composition_rows).sort_values(["configuration", "K", "channel_class"]),
            tables_dir / "subset_class_composition.csv",
        )
        plot_subset_curves(curves, figures_dir / "subset_information_curves.png", args.figure_dpi)
        monotone = all(np.all(np.diff(values) >= -1e-10) for values in curves.values())
        quality_rows.append(
            {
                "check": "subset objective monotonicity",
                "status": "pass" if monotone else "fail",
                "detail": "All regularized log-determinant curves are non-decreasing"
                if monotone
                else "At least one curve decreased",
            }
        )

    if "phantoms" in analyses:
        positions_requested = generate_phantom_positions(
            grid,
            args.phantom_radius_mm,
            args.phantom_r_samples,
            args.phantom_theta_samples,
            args.phantom_z_samples,
        )
        phantom_quadrature = common_bin_quadrature_points(
            grid, common_mask, args.phantom_subsamples_per_axis
        )
        truths_supported, positions_supported = phantom_truth_matrix(
            phantom_quadrature,
            positions_requested,
            args.phantom_radius_mm,
            args.phantom_min_bins,
        )
        ideal_sphere_volume = (4.0 / 3.0) * np.pi * args.phantom_radius_mm**3
        represented_volume = (
            positions_supported["effective_truth_bins"] * grid.bin_volume_mm3
        )
        positions_supported["ideal_sphere_volume_mm3"] = ideal_sphere_volume
        positions_supported["realized_truth_volume_mm3"] = represented_volume
        positions_supported["sphere_volume_ratio"] = (
            represented_volume / ideal_sphere_volume
        )
        volume_fidelity_mask = (
            positions_supported["sphere_volume_ratio"].to_numpy()
            >= args.phantom_min_volume_ratio
        )
        positions_supported["included_in_benchmark"] = volume_fidelity_mask
        save_dataframe(
            positions_supported, tables_dir / "phantom_position_screening.csv"
        )
        positions = positions_supported.loc[volume_fidelity_mask].reset_index(drop=True)
        truths = truths_supported[:, volume_fidelity_mask]
        if positions.empty:
            raise ValueError(
                "No phantom satisfies phantom-min-volume-ratio="
                f"{args.phantom_min_volume_ratio:g}"
            )
        save_dataframe(positions, tables_dir / "phantom_positions.csv")
        standard_rng = np.random.default_rng(args.seed)
        n_columns = len(positions) * args.phantom_noise_repeats
        standard_noise = standard_rng.standard_normal((metas[0].n_channels, n_columns))
        raw_frames = []
        for label, sensitivity in common_s.items():
            print(f"Paired spherical phantoms: {label}")
            raw_frames.append(
                run_phantoms(
                    label,
                    sensitivity,
                    common_xyz,
                    common_volumes,
                    truths,
                    positions,
                    standard_noise,
                    args.phantom_noise_repeats,
                    args.phantom_noise_rel,
                    args.phantom_lambda_rel,
                    args.phantom_threshold,
                )
            )
        phantom_raw = pd.concat(raw_frames, ignore_index=True)
        phantom_summary = summarize_phantoms(
            phantom_raw, args.bootstrap_samples, args.seed
        )
        phantom_pairwise = paired_phantom_comparisons(
            phantom_raw, args.bootstrap_samples, args.seed
        )
        save_dataframe(phantom_raw, tables_dir / "phantom_metrics_all_realisations.csv")
        save_dataframe(phantom_summary, tables_dir / "phantom_summary.csv")
        save_dataframe(
            phantom_pairwise, tables_dir / "phantom_pairwise_comparisons.csv"
        )
        plot_phantom_maps(
            phantom_raw,
            "localization_error_mm",
            figures_dir / "phantom_localization_maps.png",
            args.figure_dpi,
        )
        plot_phantom_maps(
            phantom_raw, "auc", figures_dir / "phantom_auc_maps.png", args.figure_dpi
        )
        plot_phantom_maps(
            phantom_raw,
            "dice_50pct_peak",
            figures_dir / "phantom_dice_maps.png",
            args.figure_dpi,
        )
        plot_phantom_maps(
            phantom_raw, "nrmse", figures_dir / "phantom_nrmse_maps.png", args.figure_dpi
        )
        successful_positions = phantom_raw[phantom_raw["status"] == "ok"].groupby(
            "configuration"
        )["target_id"].nunique()
        paired = successful_positions.nunique() == 1 and int(successful_positions.min()) == len(positions)
        expected_pairwise_rows = (
            len(common_s) * (len(common_s) - 1) // 2 * 6
        )
        pairwise_complete = (
            len(phantom_pairwise) == expected_pairwise_rows
            and int(phantom_pairwise["n_paired_positions"].min()) == len(positions)
        )
        quality_rows.append(
            {
                "check": "phantom grid support",
                "status": "pass"
                if len(positions_supported) == len(positions_requested)
                else "warning",
                "detail": (
                    f"{len(positions_supported)}/{len(positions_requested)} requested physical positions "
                    f"contain at least {args.phantom_min_bins} common bins"
                ),
            }
        )
        quality_rows.append(
            {
                "check": "phantom volume-fidelity filter",
                "status": "pass",
                "detail": (
                    f"{len(positions)}/{len(positions_supported)} supported positions retained at "
                    f"realized/ideal volume ratio >= {args.phantom_min_volume_ratio:.2f}"
                ),
            }
        )
        volume_ratio = positions["sphere_volume_ratio"]
        quality_rows.append(
            {
                "check": "phantom partial-volume accuracy",
                "status": "pass"
                if float(volume_ratio.min()) >= args.phantom_min_volume_ratio
                and float(volume_ratio.max()) <= 1.05
                else "warning",
                "detail": (
                    "realized/ideal sphere-volume ratio "
                    f"{float(volume_ratio.min()):.3f}--{float(volume_ratio.max()):.3f}; "
                    f"{args.phantom_subsamples_per_axis ** 3} quadrature points per bin"
                ),
            }
        )
        quality_rows.append(
            {
                "check": "paired phantom positions",
                "status": "pass" if paired else "fail",
                "detail": f"{len(positions)} common physical positions; {args.phantom_noise_repeats} repeats",
            }
        )
        quality_rows.append(
            {
                "check": "paired statistical comparisons",
                "status": "pass" if pairwise_complete else "fail",
                "detail": (
                    f"{len(phantom_pairwise)}/{expected_pairwise_rows} metric-wise pairs; "
                    "target-paired bootstrap, Wilcoxon tests, and per-metric BH-FDR"
                ),
            }
        )
        quality_rows.append(
            {
                "check": "phantom interpretation",
                "status": "warning",
                "detail": "Same linearized S is used for forward and inverse modelling (inverse crime)",
            }
        )
        if args.phantom_noise_repeats < 3:
            quality_rows.append(
                {
                    "check": "noise replication",
                    "status": "warning",
                    "detail": f"Only {args.phantom_noise_repeats} noise realization(s) per position",
                }
            )

    save_dataframe(pd.DataFrame(quality_rows), tables_dir / "quality_checks.csv")
    mesh_frame = pd.DataFrame([asdict(m) for m in metas])
    save_dataframe(mesh_frame, tables_dir / "mesh_inventory.csv")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str((entrypoint_path or Path(__file__)).resolve()),
        "run_mode": run_mode,
        "command": [sys.executable, *sys.argv],
        "parameters": vars(args),
        "analyses": analyses,
        "common_grid": {
            "nr": grid.nr,
            "ntheta": grid.ntheta,
            "nz": grid.nz,
            "total_bins": grid.n_bins,
            "common_occupied_bins": n_common,
            "bin_volume_mm3": grid.bin_volume_mm3,
            "r_max_mm": float(grid.r_edges_mm[-1]),
            "z_min_mm": float(grid.z_edges_mm[0]),
            "z_max_mm": float(grid.z_edges_mm[-1]),
        },
        "regularization": {
            "common_reference_scale": common_reference,
            "alpha_relative_primary": args.alpha_rel,
            "alpha_absolute_primary": alpha,
        },
        "inputs": [
            {**asdict(meta), "sha256": sha256_file(Path(meta.path))} for meta in metas
        ],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "h5py": h5py.__version__ if h5py is not None else "not installed",
            "matplotlib": matplotlib.__version__,
        },
        "interpretation_notes": [
            "Primary rankings use a shared equal-volume 3D cylindrical basis.",
            "Unit-noise local sensitivity norms are proxies unless score-noise-std has physical calibration.",
            "Phantom results are internal linear-model consistency results, not experimental validation.",
        ],
    }
    with (output / "run_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, default=json_ready)

    print(f"Results written to: {output}")
    return 0
