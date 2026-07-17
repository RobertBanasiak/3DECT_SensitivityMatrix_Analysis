from __future__ import annotations

import numpy as np
import pandas as pd

from .channels import channel_table, greedy_regularized_logdet, group_masks
from .constants import EXPECTED_CHANNELS
from .metrics import gram_and_eigenvalues, regularized_spectral_metrics, weighted_gini
from .phantoms import binary_auc_tie_aware, paired_phantom_comparisons, phantom_truth_matrix


def run_self_test() -> None:
    rng = np.random.default_rng(7)
    sensitivity = rng.normal(size=(80, 20))
    gram, eigenvalues = gram_and_eigenvalues(sensitivity, 1.0)
    alpha = 0.1 * np.trace(gram) / gram.shape[0]
    metrics = regularized_spectral_metrics(sensitivity, alpha, 1.0, gram, eigenvalues)
    assert np.isfinite(list(metrics.values())).all()
    selected, cumulative, marginals = greedy_regularized_logdet(gram, alpha, 15)
    assert len(selected) == len(set(selected)) == 15
    assert np.all(np.diff(cumulative) >= -1e-12)
    assert np.all(np.asarray(marginals) >= -1e-12)
    assert np.isclose(binary_auc_tie_aware(np.array([0.0, 0.0]), np.array([0, 1])), 0.5)
    perfect = binary_auc_tie_aware(np.array([0.0, 1.0]), np.array([0, 1]))
    assert np.isclose(perfect, 1.0)
    assert np.isclose(weighted_gini(np.array([1.0, 1.0]), np.ones(2)), 0.0)
    assert np.isclose(weighted_gini(np.array([1.0, 2.0]), np.ones(2)), 1.0 / 6.0)
    xyz = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    positions = pd.DataFrame(
        [{"target_id": 0, "r_mm": 1.0, "theta_deg": 0.0, "z_mm": 0.0, "x_mm": 1.0, "y_mm": 0.0}]
    )
    truths, _ = phantom_truth_matrix(
        xyz[:, None, :], positions, radius_mm=0.25, min_bins=1
    )
    assert truths[:, 0].tolist() == [1.0, 0.0]
    channels = channel_table(EXPECTED_CHANNELS)
    masks = group_masks(channels)
    assert np.all(np.sum(np.column_stack(list(masks.values())), axis=1) == 1)
    comparison_rows = []
    for label, localization, auc in (("A", 1.0, 0.9), ("B", 2.0, 0.8)):
        for target_id in range(3):
            comparison_rows.append(
                {
                    "configuration": label,
                    "target_id": target_id,
                    "status": "ok",
                    "localization_error_mm": localization,
                    "relative_volume_error_pct": localization,
                    "auc": auc,
                    "dice_50pct_peak": auc,
                    "nrmse": localization,
                    "weighted_correlation": auc,
                }
            )
    comparisons = paired_phantom_comparisons(
        pd.DataFrame(comparison_rows), bootstrap_samples=100, seed=7
    )
    assert len(comparisons) == 6
    assert np.all(comparisons["median_improvement_a_over_b"] > 0)
    assert np.all((comparisons["fdr_bh_q"] >= 0) & (comparisons["fdr_bh_q"] <= 1))
    print("SELF-TEST PASSED")
