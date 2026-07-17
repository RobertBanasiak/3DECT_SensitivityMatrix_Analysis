from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import scipy
from scipy.linalg import cho_factor, cho_solve

from .metrics import gram_and_eigenvalues
from .models import CommonGrid


def binary_auc_tie_aware(
    scores: np.ndarray, labels: np.ndarray, weights: np.ndarray | None = None
) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    weights = np.ones_like(scores) if weights is None else np.asarray(weights, dtype=float)
    valid = np.isfinite(scores) & np.isfinite(weights) & (weights > 0)
    scores, labels, weights = scores[valid], labels[valid], weights[valid]
    positive_weight = float(weights[labels].sum())
    negative_weight = float(weights[~labels].sum())
    if positive_weight == 0 or negative_weight == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    scores, labels, weights = scores[order], labels[order], weights[order]
    favourable = 0.0
    negative_below = 0.0
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[end] == scores[start]:
            end += 1
        block_labels = labels[start:end]
        block_weights = weights[start:end]
        pos_equal = float(block_weights[block_labels].sum())
        neg_equal = float(block_weights[~block_labels].sum())
        favourable += pos_equal * (negative_below + 0.5 * neg_equal)
        negative_below += neg_equal
        start = end
    return favourable / (positive_weight * negative_weight)

def generate_phantom_positions(
    grid: CommonGrid,
    radius_mm: float,
    r_samples: int,
    theta_samples: int,
    z_samples: int,
) -> pd.DataFrame:
    r_limit = grid.r_edges_mm[-1] - radius_mm
    z_low = grid.z_edges_mm[0] + radius_mm
    z_high = grid.z_edges_mm[-1] - radius_mm
    if r_limit < 0 or z_high < z_low:
        raise ValueError("Phantom radius does not fit inside the common domain")
    radial = np.linspace(0.0, r_limit, r_samples)
    axial = np.linspace(z_low, z_high, z_samples)
    angular = np.linspace(0.0, 2.0 * np.pi, theta_samples, endpoint=False)
    rows = []
    target_id = 0
    for z_mm in axial:
        for r_mm in radial:
            angles = np.array([0.0]) if np.isclose(r_mm, 0.0) else angular
            for theta_rad in angles:
                rows.append(
                    {
                        "target_id": target_id,
                        "r_mm": float(r_mm),
                        "theta_deg": float(np.degrees(theta_rad)),
                        "z_mm": float(z_mm),
                        "x_mm": float(r_mm * np.cos(theta_rad)),
                        "y_mm": float(r_mm * np.sin(theta_rad)),
                    }
                )
                target_id += 1
    return pd.DataFrame(rows)

def common_bin_quadrature_points(
    grid: CommonGrid, common_mask: np.ndarray, samples_per_axis: int
) -> np.ndarray:
    """Return equal-volume quadrature points for each retained cylindrical bin.

    Radial samples are uniform in r^2, angular samples are uniform in theta,
    and axial samples are uniform in z.  Consequently every returned point has
    the same volume weight within its bin.
    """
    if samples_per_axis <= 0:
        raise ValueError("samples_per_axis must be positive")
    full_indices = np.flatnonzero(common_mask)
    iz, it, ir = np.unravel_index(full_indices, grid.shape)
    fractions = (np.arange(samples_per_axis, dtype=float) + 0.5) / samples_per_axis
    points = []
    for fz in fractions:
        z = grid.z_edges_mm[iz] + fz * (
            grid.z_edges_mm[iz + 1] - grid.z_edges_mm[iz]
        )
        for ft in fractions:
            theta = grid.theta_edges_rad[it] + ft * (
                grid.theta_edges_rad[it + 1] - grid.theta_edges_rad[it]
            )
            for fr in fractions:
                radius_squared = grid.r_edges_mm[ir] ** 2 + fr * (
                    grid.r_edges_mm[ir + 1] ** 2 - grid.r_edges_mm[ir] ** 2
                )
                radius = np.sqrt(radius_squared)
                points.append(
                    np.column_stack(
                        [radius * np.cos(theta), radius * np.sin(theta), z]
                    )
                )
    # (n_bins, n_quadrature_points, xyz)
    return np.stack(points, axis=1)

def phantom_truth_matrix(
    bin_quadrature_xyz_mm: np.ndarray,
    positions: pd.DataFrame,
    radius_mm: float,
    min_bins: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    truths = []
    accepted = []
    for row in positions.itertuples(index=False):
        centre = np.array([row.x_mm, row.y_mm, row.z_mm])
        inside = np.linalg.norm(bin_quadrature_xyz_mm - centre, axis=2) <= radius_mm
        partial_volume_fraction = inside.mean(axis=1)
        effective_bins = float(partial_volume_fraction.sum())
        if effective_bins < min_bins:
            continue
        truths.append(partial_volume_fraction)
        record = row._asdict()
        record["n_truth_bins"] = int(np.count_nonzero(partial_volume_fraction > 0))
        record["effective_truth_bins"] = effective_bins
        accepted.append(record)
    if not truths:
        raise ValueError("No phantom contains the required number of common-grid bins")
    return np.column_stack(truths), pd.DataFrame(accepted)

def _weighted_centroid(xyz: np.ndarray, field: np.ndarray, weights: np.ndarray) -> np.ndarray:
    positive = np.maximum(np.asarray(field, dtype=float), 0.0)
    total = float(np.sum(positive * weights))
    if total <= 0:
        return np.full(3, np.nan)
    return np.sum(xyz * (positive * weights)[:, None], axis=0) / total

def phantom_metrics_one(
    xyz: np.ndarray,
    volumes: np.ndarray,
    truth: np.ndarray,
    reconstruction: np.ndarray,
    threshold_fraction: float,
) -> dict[str, float | str]:
    positive = np.maximum(reconstruction, 0.0)
    if not np.any(np.isfinite(positive)) or float(np.nanmax(positive)) <= 0:
        return {"status": "nonpositive_reconstruction"}
    true_centroid = _weighted_centroid(xyz, truth, volumes)
    reco_centroid = _weighted_centroid(xyz, positive, volumes)
    localization = float(np.linalg.norm(reco_centroid - true_centroid))
    threshold = threshold_fraction * float(np.max(positive))
    detected = positive >= threshold
    true_mask = truth >= 0.5
    if not np.any(true_mask):
        true_mask = truth > 0
    true_volume = float(np.sum(volumes * truth))
    reco_volume = float(volumes[detected].sum())
    overlap = float(np.sum(volumes * truth * detected))
    volume_error_pct = 100.0 * abs(reco_volume - true_volume) / true_volume
    dice = 2.0 * overlap / (true_volume + reco_volume + 1e-30)
    auc = binary_auc_tie_aware(reconstruction, true_mask, volumes)
    nrmse = float(
        np.sqrt(np.sum(volumes * (reconstruction - truth) ** 2) / np.sum(volumes * truth**2))
    )
    truth_mean = float(np.average(truth, weights=volumes))
    reco_mean = float(np.average(reconstruction, weights=volumes))
    covariance = float(np.sum(volumes * (truth - truth_mean) * (reconstruction - reco_mean)))
    variance_t = float(np.sum(volumes * (truth - truth_mean) ** 2))
    variance_r = float(np.sum(volumes * (reconstruction - reco_mean) ** 2))
    correlation = covariance / math.sqrt(max(variance_t * variance_r, 1e-30))
    return {
        "status": "ok",
        "localization_error_mm": localization,
        "relative_volume_error_pct": volume_error_pct,
        "auc": auc,
        "dice_50pct_peak": dice,
        "nrmse": nrmse,
        "weighted_correlation": correlation,
    }

def run_phantoms(
    label: str,
    sensitivity: np.ndarray,
    xyz: np.ndarray,
    volumes: np.ndarray,
    truths: np.ndarray,
    positions: pd.DataFrame,
    standard_noise: np.ndarray,
    repeats: int,
    noise_rel: float,
    lambda_rel: float,
    threshold_fraction: float,
) -> pd.DataFrame:
    gram, eigenvalues = gram_and_eigenvalues(sensitivity, noise_std=1.0)
    largest_singular = math.sqrt(float(eigenvalues[-1]))
    ridge_lambda = lambda_rel * largest_singular
    system = gram + ridge_lambda**2 * np.eye(gram.shape[0])
    factor = cho_factor(system, lower=True, check_finite=False)
    clean = sensitivity.T @ truths
    clean_repeated = np.repeat(clean, repeats, axis=1)
    sigma = noise_rel * np.sqrt(np.mean(clean**2, axis=0))
    sigma_repeated = np.repeat(sigma, repeats)
    noisy = clean_repeated + standard_noise * sigma_repeated[None, :]
    coefficients = cho_solve(factor, noisy, check_finite=False)
    reconstructions = sensitivity @ coefficients

    rows = []
    for position_index, position in positions.iterrows():
        for repeat in range(repeats):
            column = position_index * repeats + repeat
            metric = phantom_metrics_one(
                xyz,
                volumes,
                truths[:, position_index],
                reconstructions[:, column],
                threshold_fraction,
            )
            rows.append(
                {
                    "configuration": label,
                    "target_id": int(position.target_id),
                    "repeat": repeat,
                    "r_mm": float(position.r_mm),
                    "theta_deg": float(position.theta_deg),
                    "z_mm": float(position.z_mm),
                    "n_truth_bins": int(position.n_truth_bins),
                    "effective_truth_bins": float(position.effective_truth_bins),
                    "realized_truth_volume_mm3": float(
                        position.effective_truth_bins * volumes[0]
                    ),
                    "noise_sigma": float(sigma[position_index]),
                    "ridge_lambda": ridge_lambda,
                    **metric,
                }
            )
    return pd.DataFrame(rows)

def summarize_phantoms(
    raw: pd.DataFrame, bootstrap_samples: int, seed: int
) -> pd.DataFrame:
    metric_columns = [
        "localization_error_mm",
        "relative_volume_error_pct",
        "auc",
        "dice_50pct_peak",
        "nrmse",
        "weighted_correlation",
    ]
    rows = []
    for label_index, (label, group) in enumerate(raw.groupby("configuration", sort=True)):
        valid = group[group["status"] == "ok"]
        row: dict[str, float | int | str] = {
            "configuration": label,
            "n_requested_reconstructions": len(group),
            "n_successful_reconstructions": len(valid),
            "n_unique_positions": valid["target_id"].nunique(),
        }
        for column in metric_columns:
            # Repeated noise realizations are nested within a physical target.
            # Aggregate them first so that the target, not the realization, is
            # the independent unit in quartiles and bootstrap intervals.
            values = (
                valid.groupby("target_id")[column]
                .median()
                .dropna()
                .to_numpy(float)
            )
            row[f"{column}_median"] = float(np.median(values)) if values.size else np.nan
            row[f"{column}_q25"] = float(np.percentile(values, 25)) if values.size else np.nan
            row[f"{column}_q75"] = float(np.percentile(values, 75)) if values.size else np.nan
            if values.size and bootstrap_samples > 0:
                rng = np.random.default_rng(seed + 1009 * label_index + len(column))
                sampled = rng.choice(
                    values, size=(bootstrap_samples, len(values)), replace=True
                )
                boot_medians = np.median(sampled, axis=1)
                row[f"{column}_bootstrap95_low"] = float(
                    np.percentile(boot_medians, 2.5)
                )
                row[f"{column}_bootstrap95_high"] = float(
                    np.percentile(boot_medians, 97.5)
                )
            else:
                row[f"{column}_bootstrap95_low"] = np.nan
                row[f"{column}_bootstrap95_high"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini--Hochberg adjusted q-values, preserving NaN entries."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_values, np.nan)
    valid = np.flatnonzero(np.isfinite(p_values))
    if not len(valid):
        return adjusted
    order = valid[np.argsort(p_values[valid])]
    ranked = p_values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted

def paired_phantom_comparisons(
    raw: pd.DataFrame, bootstrap_samples: int, seed: int
) -> pd.DataFrame:
    """Target-paired configuration comparisons with multiplicity control.

    Repeated noise realizations are first collapsed to a per-target median.
    Positive ``median_improvement_a_over_b`` always favours configuration A,
    regardless of whether the original metric is minimized or maximized.
    """
    metric_direction = {
        "localization_error_mm": "lower",
        "relative_volume_error_pct": "lower",
        "auc": "higher",
        "dice_50pct_peak": "higher",
        "nrmse": "lower",
        "weighted_correlation": "higher",
    }
    valid = raw[raw["status"] == "ok"]
    target_medians = (
        valid.groupby(["configuration", "target_id"])[list(metric_direction)]
        .median()
        .reset_index()
    )
    labels = sorted(target_medians["configuration"].unique())
    rows = []
    for pair_index, (label_a, label_b) in enumerate(itertools.combinations(labels, 2)):
        group_a = target_medians[target_medians["configuration"] == label_a]
        group_b = target_medians[target_medians["configuration"] == label_b]
        paired = group_a.merge(group_b, on="target_id", suffixes=("_a", "_b"))
        for metric_index, (metric, direction) in enumerate(metric_direction.items()):
            a = paired[f"{metric}_a"].to_numpy(float)
            b = paired[f"{metric}_b"].to_numpy(float)
            finite = np.isfinite(a) & np.isfinite(b)
            a, b = a[finite], b[finite]
            improvement = b - a if direction == "lower" else a - b
            if len(improvement) and bootstrap_samples > 0:
                rng = np.random.default_rng(
                    seed + 100_003 * pair_index + 1009 * metric_index
                )
                sampled = rng.choice(
                    improvement,
                    size=(bootstrap_samples, len(improvement)),
                    replace=True,
                )
                boot = np.median(sampled, axis=1)
                ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
            else:
                ci_low = ci_high = np.nan
            if not len(improvement) or np.allclose(improvement, 0.0):
                p_value = 1.0
            else:
                p_value = float(
                    scipy.stats.wilcoxon(
                        improvement, zero_method="pratt", alternative="two-sided"
                    ).pvalue
                )
            rows.append(
                {
                    "metric": metric,
                    "direction": direction,
                    "configuration_a": label_a,
                    "configuration_b": label_b,
                    "n_paired_positions": len(improvement),
                    "median_a": float(np.median(a)) if len(a) else np.nan,
                    "median_b": float(np.median(b)) if len(b) else np.nan,
                    "median_improvement_a_over_b": float(np.median(improvement))
                    if len(improvement)
                    else np.nan,
                    "bootstrap95_low": float(ci_low),
                    "bootstrap95_high": float(ci_high),
                    "paired_win_fraction_a": float(
                        np.mean(improvement > 0) + 0.5 * np.mean(improvement == 0)
                    )
                    if len(improvement)
                    else np.nan,
                    "wilcoxon_p": p_value,
                }
            )
    result = pd.DataFrame(rows)
    result["fdr_bh_q"] = np.nan
    for metric, indices in result.groupby("metric").groups.items():
        result.loc[indices, "fdr_bh_q"] = _benjamini_hochberg(
            result.loc[indices, "wilcoxon_p"].to_numpy(float)
        )
    return result
