#!/usr/bin/env python3
"""Evaluate the stability of greedy regularised ECT measurement selection.

Place beside main.py in the existing v1.0.0 project:
    python analyze_selection_stability.py --self-test
    python analyze_selection_stability.py

With no original-project arguments, reuse main.py's PYCHARM_CONFIG. Otherwise
use the original parser. Run only global/subset analyses, on the original
common occupied basis. All original numerical routines, files and settings
are preserved; additional results go to a new timestamped output directory.

One factor changes at a time:
  * alpha: the original alpha_sweep, normally 1e-4, 1e-3 and 1e-2 times the
    ORIGINAL shared spectral reference. Do not recalibrate alpha for noise.
  * k: 64, 96, 128, 160, 192, 256 channels (plus the configured reference k).
    These are prefixes of one greedy trajectory at the largest requested k.
  * unequal noise variances: ten reproducible profiles, identical across
    configurations. Relative standard deviations are log-uniform in
    [1/2, 2], then normalised to mean variance one. The range ratio is at
    most four; the normalisation changes the absolute endpoints.
  * correlated noise: R=(1-rho)I + rho B B.T/2, rho=0.25 and 0.50. B is the
    channel/electrode incidence matrix with two ones per row. R has unit
    diagonal and correlation rho/2 between distinct channels sharing an
    electrode. These are illustrative sensitivity-analysis noise scenarios,
    not measured device covariances.

G is the original scalar-noise-weighted Gram matrix, and R is a covariance
relative to that baseline noise variance. For each chosen set K, evaluate
    J(K) = logdet(R_K + G_K/alpha) - logdet(R_K).
Select the principal covariance submatrix BEFORE inversion/factorisation.
Globally whitening all channels and then selecting transformed columns
would not represent selection of the original acquisition channels.
For diagonal R, reuse the project's original selector on the appropriately
weighted Gram matrix. For correlated R, use two Schur complements.

Report overlap with the configured reference set, same-cardinality overlap,
and pairwise Jaccard/overlap for equal-size sets. Smaller/larger k sets are
nested by construction; their Jaccard scores against the reference describe
set size, not instability. Rescore every set under the common reference
model as well as its own design model. Own-model retention denominators
change with that model and must not be mistaken for common-model scores.

Additional options (original-project options can also be supplied):
    --stability-k 64 96 128 160 192 256
    --noise-profile-count 10
    --noise-std-factor 2
    --correlation-strengths 0.25 0.50

Outputs under selection_stability_results/run_*:
    stability_scorecard.csv
    stability_selected_channels.csv
    stability_overlap.csv
    stability_summary.csv
    stability_noise_profiles.csv
    stability_noise_summary.csv
    stability_manifest.json
The original pipeline outputs, including run_manifest.json, are retained.
No reconstruction, target-distribution or physical-acquisition tests run.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import itertools
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_K = (64, 96, 128, 160, 192, 256)
SUMMARY_METRICS = (
    "reference_information_retention",
    "own_information_retention",
    "reference_information_relative_to_same_k",
    "own_information_improvement_percent_vs_same_k_reference",
    "spectral_energy_retention",
    "overlap_fraction_reference",
    "reference_set_coverage",
    "jaccard_reference",
    "overlap_fraction_same_k",
    "jaccard_same_k",
)


def logdet_spd(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return 0.0
    factor = np.linalg.cholesky(0.5 * (matrix + matrix.T))
    return float(2.0 * np.log(np.diag(factor)).sum())


def information(gram, alpha, selected=None, covariance=None):
    """Restrict G and R together; never restrict the inverse of full R."""
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be positive and finite")
    gram = np.asarray(gram, dtype=float)
    n = gram.shape[0]
    if gram.shape != (n, n):
        raise ValueError("gram must be square")
    if selected is None:
        indices = np.arange(n, dtype=int)
    else:
        indices = np.asarray(selected, dtype=int)
        if len(np.unique(indices)) != len(indices) or np.any((indices < 0) | (indices >= n)):
            raise ValueError("Selected channel indices must be distinct and in range")
    sub_gram = gram[np.ix_(indices, indices)]
    if covariance is None:
        return logdet_spd(np.eye(len(indices)) + sub_gram / alpha)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != gram.shape:
        raise ValueError("covariance must have the same shape as gram")
    sub_noise = covariance[np.ix_(indices, indices)]
    return logdet_spd(sub_noise + sub_gram / alpha) - logdet_spd(sub_noise)


def update_inverse(inverse, cross, schur):
    if inverse is None:
        return np.array([[1.0 / schur]])
    z = inverse @ cross
    n = inverse.shape[0]
    result = np.empty((n + 1, n + 1), dtype=float)
    result[:n, :n] = inverse + np.outer(z, z) / schur
    result[:n, n] = -z / schur
    result[n, :n] = -z / schur
    result[n, n] = 1.0 / schur
    return result


def greedy_correlated(gram, covariance, alpha, k):
    """Greedily maximise the ratio of signal-plus-noise and noise determinants."""
    gram = np.asarray(gram, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    n = gram.shape[0]
    if gram.shape != (n, n) or covariance.shape != gram.shape:
        raise ValueError("G and R must be square matrices of identical shape")
    if not np.isfinite(alpha) or alpha <= 0 or not 1 <= k <= n:
        raise ValueError("Invalid alpha or subset size")
    if not np.all(np.isfinite(gram)) or not np.all(np.isfinite(covariance)):
        raise ValueError("G and R must be finite")
    if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-14):
        raise ValueError("R must be symmetric")
    np.linalg.cholesky(covariance)
    signal_noise = covariance + gram / alpha
    remaining = np.arange(n, dtype=int)
    selected, cumulative, marginals = [], [], []
    inverse_q = inverse_r = None
    total = 0.0
    for _ in range(k):
        if not selected:
            schur_q = np.diag(signal_noise)[remaining]
            schur_r = np.diag(covariance)[remaining]
        else:
            q_cross = signal_noise[np.ix_(selected, remaining)]
            r_cross = covariance[np.ix_(selected, remaining)]
            schur_q = np.diag(signal_noise)[remaining] - np.sum(
                q_cross * (inverse_q @ q_cross), axis=0)
            schur_r = np.diag(covariance)[remaining] - np.sum(
                r_cross * (inverse_r @ r_cross), axis=0)
        if np.any(schur_q <= 0) or np.any(schur_r <= 0):
            raise FloatingPointError("A covariance Schur complement became non-positive")
        ratios = schur_q / schur_r
        if np.any(ratios < 1.0 - 1e-8):
            raise FloatingPointError("Noise-aware marginal information became negative")
        best_position = int(np.argmax(ratios))
        best = int(remaining[best_position])
        marginal = math.log(max(float(ratios[best_position]), 1.0))
        q_cross_best = signal_noise[np.ix_(selected, [best])].ravel()
        r_cross_best = covariance[np.ix_(selected, [best])].ravel()
        inverse_q = update_inverse(inverse_q, q_cross_best, float(schur_q[best_position]))
        inverse_r = update_inverse(inverse_r, r_cross_best, float(schur_r[best_position]))
        selected.append(best)
        total += marginal
        marginals.append(marginal)
        cumulative.append(total)
        remaining = np.delete(remaining, best_position)
    return selected, cumulative, marginals


def electrode_pairs(n_channels):
    n_electrodes = (1 + math.isqrt(1 + 8 * n_channels)) // 2
    if n_electrodes * (n_electrodes - 1) // 2 != n_channels:
        raise ValueError("Correlated-noise scenarios require complete electrode-pair channels")
    return list(itertools.combinations(range(n_electrodes), 2))


def shared_electrode_covariance(n_channels, rho):
    if not np.isfinite(rho) or not 0 <= rho < 1:
        raise ValueError("Shared-noise rho must lie in [0, 1)")
    pairs = electrode_pairs(n_channels)
    n_electrodes = max(max(pair) for pair in pairs) + 1
    incidence = np.zeros((n_channels, n_electrodes))
    for i, pair in enumerate(pairs):
        incidence[i, list(pair)] = 1.0
    return (1.0 - rho) * np.eye(n_channels) + (rho / 2.0) * (incidence @ incidence.T)


def unequal_noise_profiles(n_channels, count, factor, seed):
    if count < 0 or not np.isfinite(factor) or factor < 1:
        raise ValueError("Invalid unequal-variance profile count or standard-deviation factor")
    rng = np.random.default_rng(seed)
    sigma = np.exp(rng.uniform(-math.log(factor), math.log(factor), (count, n_channels)))
    if count:
        sigma /= np.sqrt(np.mean(sigma**2, axis=1, keepdims=True))
    return sigma


def set_overlap(selected, reference):
    a, b = set(selected), set(reference)
    shared = len(a & b)
    return {
        "shared_channels": shared,
        "overlap_fraction": shared / len(a),
        "reference_coverage": shared / len(b),
        "jaccard": shared / len(a | b),
    }


class StabilityAnalysis:
    def __init__(self, pipeline, args, options):
        self.pipeline, self.args, self.options = pipeline, args, options
        self.originals, self.captured, self.reference_cache = {}, {}, {}
        self.labels, self.rows, self.channel_rows, self.overlap_rows = [], [], [], []
        self.profile_rows, self.completed = [], []
        self.spectral_calls = 0
        self.alpha_reference = self.reference_k = self.n_channels = None
        self.expected_cases = None
        self.profiles = self.channel_lookup = None
        self.correlated_models = {}
        self.excluded_k = []

    def prepare_noise(self, n_channels):
        from ect_analysis.channels import channel_table, classify_pair
        definitions = channel_table(n_channels)
        # The production table handles the 496-channel study. Decode smaller
        # complete-pair tables too, to support independent integration checks.
        if definitions["electrode_1_one_based"].isna().any():
            pairs = electrode_pairs(n_channels)
            definitions = pd.DataFrame([
                {"channel_index_zero_based": i,
                 "electrode_1_one_based": a + 1, "electrode_2_one_based": b + 1,
                 "plane_1_zero_based": a // 8, "plane_2_zero_based": b // 8,
                 "channel_class": classify_pair(a, b)}
                for i, (a, b) in enumerate(pairs)])
        self.channel_lookup = definitions.set_index("channel_index_zero_based").to_dict(orient="index")
        self.profiles = unequal_noise_profiles(
            n_channels, self.options.noise_profile_count,
            self.options.noise_std_factor, self.args.seed)
        for draw, sigma in enumerate(self.profiles, 1):
            for channel, std in enumerate(sigma):
                self.profile_rows.append({
                    "profile_index": draw, "channel_index_zero_based": channel,
                    "relative_noise_std": float(std),
                    "relative_noise_variance": float(std**2),
                    **self.channel_lookup[channel],
                })
        self.correlated_models = {
            rho: shared_electrode_covariance(n_channels, rho)
            for rho in sorted(set(self.options.correlation_strengths))}
        self.excluded_k = sorted(k for k in set(self.options.stability_k) if k > n_channels)

    def record(self, label, sensitivity, gram, case, family, selected, alpha, alpha_rel,
               covariance, reference, same_k, full_reference, full_own,
               trajectory_seconds, trajectory_k, noise_model="iid",
               profile=0, rho=0.0, sigma=None):
        selected = list(map(int, selected))
        n = gram.shape[0]
        if len(set(selected)) != len(selected) or not all(0 <= j < n for j in selected):
            raise ValueError("A selector returned invalid channel indices")
        ref_information = information(gram, self.alpha_reference, selected)
        own_information = information(gram, alpha, selected, covariance)
        same_ref_information = information(gram, self.alpha_reference, same_k)
        own_reference_information = information(gram, alpha, same_k, covariance)
        if min(full_reference, full_own, same_ref_information, own_reference_information) <= 0:
            raise ValueError("A reference information denominator is non-positive")
        reference_overlap = set_overlap(selected, reference)
        same_overlap = set_overlap(selected, same_k)
        row = {
            "configuration": label, "case": case, "family": family,
            "noise_model": noise_model, "noise_profile_index": profile,
            "shared_noise_fraction_rho": rho,
            "noise_std_ratio": float(sigma.max() / sigma.min()) if sigma is not None else 1.0,
            "relative_mean_noise_variance": float(np.mean(sigma**2)) if sigma is not None else 1.0,
            "common_parameters": sensitivity.shape[0], "available_channels": n,
            "selected_channels": len(selected), "reference_channels": len(reference),
            "scalar_reference_noise_std": self.args.score_noise_std,
            "alpha_relative": alpha_rel, "alpha_absolute": alpha,
            "reference_alpha_absolute": self.alpha_reference,
            "reference_regularized_information": ref_information,
            "reference_full_information": full_reference,
            "reference_information_retention": ref_information / full_reference,
            "own_regularized_information": own_information,
            "own_full_information": full_own,
            "own_information_retention": own_information / full_own,
            "reference_information_relative_to_same_k": ref_information / same_ref_information,
            "own_information_retention_of_same_k_reference": own_reference_information / full_own,
            "own_information_improvement_percent_vs_same_k_reference":
                100.0 * (own_information / own_reference_information - 1.0),
            "spectral_energy_retention":
                float(np.diag(gram)[selected].sum() / np.trace(gram)),
            "shared_channels_reference": reference_overlap["shared_channels"],
            "overlap_fraction_reference": reference_overlap["overlap_fraction"],
            "reference_set_coverage": reference_overlap["reference_coverage"],
            "jaccard_reference": reference_overlap["jaccard"],
            "shared_channels_same_k": same_overlap["shared_channels"],
            "overlap_fraction_same_k": same_overlap["overlap_fraction"],
            "jaccard_same_k": same_overlap["jaccard"],
            "trajectory_selection_seconds": trajectory_seconds,
            "trajectory_channels_computed": trajectory_k,
        }
        for name in ("reference_information_retention", "own_information_retention",
                     "spectral_energy_retention"):
            if not -1e-8 <= row[name] <= 1 + 1e-8:
                raise FloatingPointError(f"Invalid retention value: {name}={row[name]}")
        self.rows.append(row)
        for rank, channel in enumerate(selected, 1):
            self.channel_rows.append({
                "configuration": label, "case": case, "family": family,
                "noise_model": noise_model, "noise_profile_index": profile,
                "alpha_relative": alpha_rel, "selected_channels": len(selected),
                "selection_order": rank, "channel_index_zero_based": channel,
                **self.channel_lookup[channel],
            })
        return row

    def compare(self, label, sensitivity, gram, eigenvalues, alpha, k, original, elapsed):
        n = gram.shape[0]
        reference = list(original[0])
        k = len(reference)
        if self.alpha_reference is None:
            self.alpha_reference, self.reference_k, self.n_channels = float(alpha), k, n
            self.prepare_noise(n)
        if not np.isclose(alpha, self.alpha_reference, rtol=1e-14, atol=0):
            raise ValueError("The original shared alpha changed between configurations")
        if (n, k) != (self.n_channels, self.reference_k):
            raise ValueError("Reference dimensions changed between configurations")
        requested_k = sorted({k} | {value for value in self.options.stability_k if value <= n})
        max_k = max(requested_k)
        original_selector = self.originals["greedy_regularized_logdet"]
        if max_k == k:
            trajectory, trajectory_time = reference, elapsed
        else:
            start = time.perf_counter()
            trajectory = original_selector(gram, alpha, max_k)[0]
            trajectory_time = time.perf_counter() - start
            if list(trajectory[:k]) != reference:
                raise FloatingPointError("Reference selection differs from the larger-k greedy prefix")
        full_reference = float(np.log1p(eigenvalues / alpha).sum())
        if not np.isclose(information(gram, alpha, reference), original[1][-1], rtol=1e-7, atol=1e-6):
            raise FloatingPointError("Independent reference score differs from the original greedy curve")
        cases = []
        row = self.record(label, sensitivity, gram, "reference", "reference",
                          reference, alpha, self.args.alpha_rel, None, reference,
                          reference, full_reference, full_reference, elapsed, k)
        cases.append((row, reference))
        for size in requested_k:
            if size != k:
                subset = trajectory[:size]
                row = self.record(label, sensitivity, gram, f"k_{size}", "subset_size",
                                  subset, alpha, self.args.alpha_rel, None, reference,
                                  subset, full_reference, full_reference,
                                  trajectory_time, max_k)
                cases.append((row, subset))
        for relative in sorted(set(self.args.alpha_sweep) | {self.args.alpha_rel}):
            if relative == self.args.alpha_rel:
                continue
            alpha_test = alpha * relative / self.args.alpha_rel
            start = time.perf_counter()
            selected = original_selector(gram, alpha_test, k)[0]
            seconds = time.perf_counter() - start
            row = self.record(label, sensitivity, gram, f"alpha_{relative:.12g}",
                              "regularization", selected, alpha_test, relative, None,
                              reference, reference, full_reference,
                              information(gram, alpha_test), seconds, k)
            cases.append((row, selected))
        for draw, sigma in enumerate(self.profiles, 1):
            weighted_gram = gram / np.outer(sigma, sigma)
            start = time.perf_counter()
            selected = original_selector(weighted_gram, alpha, k)[0]
            seconds = time.perf_counter() - start
            covariance = np.diag(sigma**2)
            row = self.record(label, sensitivity, gram, f"noise_unequal_{draw:02d}",
                              "unequal_variance", selected, alpha, self.args.alpha_rel,
                              covariance, reference, reference, full_reference,
                              information(gram, alpha, covariance=covariance),
                              seconds, k, "unequal_variance", profile=draw, sigma=sigma)
            cases.append((row, selected))
        for rho, covariance in self.correlated_models.items():
            start = time.perf_counter()
            selected, cumulative, _ = greedy_correlated(gram, covariance, alpha, k)
            seconds = time.perf_counter() - start
            independent = information(gram, alpha, selected, covariance)
            if not np.isclose(independent, cumulative[-1], rtol=1e-7, atol=1e-6):
                raise FloatingPointError("Correlated-noise selection and direct determinant disagree")
            row = self.record(label, sensitivity, gram, f"noise_correlated_{rho:.12g}",
                              "correlated_noise", selected, alpha, self.args.alpha_rel,
                              covariance, reference, reference, full_reference,
                              information(gram, alpha, covariance=covariance),
                              seconds, k, "shared_electrode", rho=rho)
            cases.append((row, selected))
        for (a, sa), (b, sb) in itertools.combinations(cases, 2):
            if len(sa) != len(sb):
                continue
            overlap = set_overlap(sa, sb)
            self.overlap_rows.append({
                "configuration": label, "case_a": a["case"], "case_b": b["case"],
                "family_a": a["family"], "family_b": b["family"],
                "selected_channels": len(sa), "shared_channels": overlap["shared_channels"],
                "overlap_fraction": overlap["overlap_fraction"], "jaccard": overlap["jaccard"],
            })
        if self.expected_cases is None:
            self.expected_cases = len(cases)
        elif self.expected_cases != len(cases):
            raise ValueError("The scenario count changed between configurations")
        self.completed.append(label)
        print(f"  STABILITY {label}: {len(cases)} cases completed.", flush=True)

    def __enter__(self):
        for name in ("map_to_common_grid", "gram_and_eigenvalues", "greedy_regularized_logdet"):
            self.originals[name] = getattr(self.pipeline, name)

        @functools.wraps(self.originals["map_to_common_grid"])
        def mapped(meta, *args, **kwargs):
            result = self.originals["map_to_common_grid"](meta, *args, **kwargs)
            self.labels.append(meta.label)
            return result

        @functools.wraps(self.originals["gram_and_eigenvalues"])
        def spectral(sensitivity, noise_std):
            if self.spectral_calls >= len(self.labels):
                raise RuntimeError("Unexpected Gram-call order; this helper supports v1.0.0")
            label = self.labels[self.spectral_calls]
            self.spectral_calls += 1
            gram, values = self.originals["gram_and_eigenvalues"](sensitivity, noise_std)
            self.captured[id(gram)] = (label, sensitivity, values)
            return gram, values

        @functools.wraps(self.originals["greedy_regularized_logdet"])
        def selected(gram, alpha, k, candidate_mask=None):
            if candidate_mask is not None:
                raise ValueError("This stability experiment assumes unrestricted channel selection")
            start = time.perf_counter()
            result = self.originals["greedy_regularized_logdet"](gram, alpha, k)
            elapsed = time.perf_counter() - start
            if id(gram) not in self.captured:
                raise RuntimeError("Selected Gram matrix does not match a captured full operator")
            label, sensitivity, values = self.captured[id(gram)]
            self.compare(label, sensitivity, gram, values, alpha, k, result, elapsed)
            return result

        self.pipeline.map_to_common_grid = mapped
        self.pipeline.gram_and_eigenvalues = spectral
        self.pipeline.greedy_regularized_logdet = selected
        return self

    def __exit__(self, *exc):
        for name, function in self.originals.items():
            setattr(self.pipeline, name, function)

    def save(self, output):
        frames = {
            "stability_scorecard.csv": pd.DataFrame(self.rows),
            "stability_selected_channels.csv": pd.DataFrame(self.channel_rows),
            "stability_overlap.csv": pd.DataFrame(self.overlap_rows),
            "stability_noise_profiles.csv": pd.DataFrame(self.profile_rows),
        }
        summary, noise_summary = [], []
        scorecard = frames["stability_scorecard.csv"]
        if len(scorecard):
            for case, group in scorecard.groupby("case", sort=True):
                row = {"case": case, "family": group["family"].iloc[0],
                       "configurations": len(group),
                       "selected_channels": int(group["selected_channels"].iloc[0])}
                for name in SUMMARY_METRICS:
                    values = group[name].to_numpy(dtype=float)
                    for stat, value in (("min", values.min()), ("mean", values.mean()),
                                        ("median", np.median(values)), ("max", values.max())):
                        row[f"{name}_{stat}"] = float(value)
                summary.append(row)
            random_profiles = scorecard.loc[scorecard["family"] == "unequal_variance"]
            for label, group in random_profiles.groupby("configuration", sort=True):
                row = {"configuration": label, "noise_profiles": len(group)}
                for name in SUMMARY_METRICS:
                    values = group[name].to_numpy(dtype=float)
                    row[f"{name}_mean"] = float(values.mean())
                    row[f"{name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
                    for stat, q in (("q025", .025), ("median", .5), ("q975", .975)):
                        row[f"{name}_{stat}"] = float(np.quantile(values, q))
                noise_summary.append(row)
        frames["stability_summary.csv"] = pd.DataFrame(summary)
        frames["stability_noise_summary.csv"] = pd.DataFrame(noise_summary)
        for name, frame in frames.items():
            frame.to_csv(output / name, index=False, encoding="utf-8-sig")


def numerical_self_test():
    from ect_analysis.channels import greedy_regularized_logdet
    from ect_analysis.selftest import run_self_test
    run_self_test()
    rng = np.random.default_rng(9187)
    sensitivity = rng.normal(size=(7, 6))
    gram = sensitivity.T @ sensitivity
    alpha = .3
    identity = np.eye(6)
    original = greedy_regularized_logdet(gram, alpha, 4)
    identity_result = greedy_correlated(gram, identity, alpha, 4)
    assert original[0] == identity_result[0]
    assert np.allclose(original[1], identity_result[1], rtol=1e-12)
    sigma = unequal_noise_profiles(6, 3, 2., 123)
    assert np.allclose(np.mean(sigma**2, axis=1), 1.0)
    assert np.array_equal(sigma, unequal_noise_profiles(6, 3, 2., 123))
    noise_models = [np.diag(sigma[0]**2), shared_electrode_covariance(6, .5)]
    for covariance in noise_models:
        selected, cumulative, _ = greedy_correlated(gram, covariance, alpha, 4)
        prefix = []
        for step, best in enumerate(selected):
            candidates = [j for j in range(6) if j not in prefix]
            direct = [information(gram, alpha, prefix + [j], covariance) for j in candidates]
            assert best == candidates[int(np.argmax(direct))]
            prefix.append(best)
            sub_noise = covariance[np.ix_(prefix, prefix)]
            sub_s = sensitivity[:, prefix]
            state_score = logdet_spd(
                np.eye(7) + sub_s @ np.linalg.solve(sub_noise, sub_s.T) / alpha)
            assert np.isclose(state_score, cumulative[step], rtol=1e-11, atol=1e-11)
        assert np.all(np.diff(cumulative) >= -1e-12)
        assert np.isclose(cumulative[-1], information(gram, alpha, selected, covariance))
    diagonal_result = greedy_correlated(gram, noise_models[0], alpha, 4)
    weighted = greedy_regularized_logdet(gram / np.outer(sigma[0], sigma[0]), alpha, 4)
    assert diagonal_result[0] == weighted[0]
    assert np.allclose(diagonal_result[1], weighted[1], rtol=1e-12)
    covariance = noise_models[1]
    chosen = [0, 3]
    sub_s = sensitivity[:, chosen]
    correct = information(gram, alpha, chosen, covariance)
    restricted_full_precision = np.linalg.inv(covariance)[np.ix_(chosen, chosen)]
    wrong = logdet_spd(np.eye(7) + sub_s @ restricted_full_precision @ sub_s.T / alpha)
    assert not np.isclose(correct, wrong, rtol=1e-4)
    assert np.isclose(correct, information(16 * gram, 16 * alpha, chosen, covariance))
    overlap = set_overlap([0, 1], [0, 1, 2, 3])
    assert overlap == {"shared_channels": 2, "overlap_fraction": 1.0,
                       "reference_coverage": .5, "jaccard": .5}
    try:
        greedy_correlated(gram, np.zeros((6, 6)), alpha, 3)
    except np.linalg.LinAlgError:
        pass
    else:
        raise AssertionError("A singular covariance was accepted")
    print("STABILITY SELF-TEST PASSED", flush=True)


def stability_options(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--stability-k", nargs="+", type=int, default=list(DEFAULT_K))
    parser.add_argument("--noise-profile-count", type=int, default=10)
    parser.add_argument("--noise-std-factor", type=float, default=2.0)
    parser.add_argument("--correlation-strengths", nargs="+", type=float, default=[.25, .50])
    options, original = parser.parse_known_args(argv)
    if not options.stability_k or any(k <= 0 for k in options.stability_k):
        parser.error("--stability-k must contain positive integers")
    if options.noise_profile_count < 0:
        parser.error("--noise-profile-count must be non-negative")
    if not np.isfinite(options.noise_std_factor) or options.noise_std_factor < 1:
        parser.error("--noise-std-factor must be finite and at least one")
    if any(not np.isfinite(rho) or not 0 < rho < 1 for rho in options.correlation_strengths):
        parser.error("--correlation-strengths must lie strictly between zero and one")
    return options, original


def main(argv=None):
    if not (PROJECT_DIRECTORY / "ect_analysis" / "pipeline.py").is_file():
        raise RuntimeError("Place analyze_selection_stability.py beside main.py")
    from ect_analysis import pipeline
    from ect_analysis.cli import parse_args
    from ect_analysis.reporting import json_ready

    argv = list(sys.argv[1:] if argv is None else argv)
    options, original_argv = stability_options(argv)
    if original_argv:
        args, mode = parse_args(original_argv), "original_command_line_parser"
    else:
        import main as project_main
        if project_main.USE_PYCHARM_CONFIG:
            args, mode = project_main.pycharm_arguments(), "main.py PYCHARM_CONFIG"
        else:
            args, mode = parse_args([]), "original_command_line_defaults"
    if args.self_test:
        numerical_self_test()
        return 0
    source_parameters = dict(vars(args))
    args = argparse.Namespace(**source_parameters)
    args.analyses = "global,subsets"
    explicit_output = any(a == "--output-dir" or a.startswith("--output-dir=") for a in original_argv)
    base = args.output_dir if explicit_output else PROJECT_DIRECTORY / "selection_stability_results"
    stamp = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    output = (base / stamp).resolve()
    output.mkdir(parents=True, exist_ok=False)
    args.output_dir = output
    analysis = StabilityAnalysis(pipeline, args, options)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration_source": mode,
        "source_parameters": source_parameters, "executed_parameters": vars(args),
        "stability_options": vars(options),
        "noise_profiles_seed": args.seed,
        "same_noise_profiles_across_configurations": True,
        "platform": platform.platform(), "processor": platform.processor(),
        "logical_processor_count": os.cpu_count(),
        "objective": "logdet(R_K + G_K/alpha) - logdet(R_K); restrict covariance before inversion",
        "noise_models": {
            "iid": "R=I relative to the original scalar noise variance",
            "unequal_variance": "sigma=exp(U[-log(f),log(f)]), normalised to mean sigma^2=1; R=diag(sigma^2)",
            "shared_electrode": "R=(1-rho)I+rho B B.T/2; B has two electrode-incidence ones per row",
        },
        "notes": [
            "These covariance scenarios are illustrative, not measured hardware noise.",
            "All selected indices refer to original acquisition channels.",
            "Alpha is derived from the original shared reference and is not recalibrated for each noise model.",
            "Each variant changes one factor; this is not a full factorial interaction analysis.",
            "All sets are rescored under the common reference model; own-model denominators also provided.",
            "Same-k reference information is a greedy baseline, not a proven global optimum.",
            "The k sweep uses one max-k trajectory; smaller/larger sets are nested by construction.",
            "Trajectory timings include the whole trajectory_channels_computed, not just the reported prefix.",
            "Timings exclude shared mapping, Gram/eigenvalue construction, noise-profile generation and scoring; correlated selection includes covariance validation.",
            "Noise-profile quantiles describe the imposed profile ensemble, not confidence intervals for a mean.",
            "Disk-cache state and thread limits are not controlled.",
            "This experiment does not evaluate reconstruction, target-distribution weights, geometry perturbations or hardware constraints.",
        ],
    }
    start, exit_code, error = time.perf_counter(), 1, None
    print(f"Selection stability output: {output}", flush=True)
    try:
        with analysis:
            exit_code = pipeline.run_analysis(args_override=args, entrypoint_path=Path(__file__))
        return int(exit_code)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata["complete_analysis_elapsed_seconds"] = time.perf_counter() - start
        metadata["exit_code"], metadata["error"] = exit_code, error
        metadata["completed_configurations"] = sorted(analysis.completed)
        metadata["completed_cases"] = len(analysis.rows)
        metadata["cases_per_configuration"] = analysis.expected_cases
        metadata["reference_channels"] = analysis.reference_k
        metadata["reference_alpha_absolute"] = analysis.alpha_reference
        metadata["requested_k_excluded_above_channel_count"] = analysis.excluded_k
        analysis.save(output)
        original_path = output / "run_manifest.json"
        if original_path.is_file():
            original_bytes = original_path.read_bytes()
            original = json.loads(original_bytes.decode("utf-8"))
            metadata["original_run_manifest_sha256"] = hashlib.sha256(original_bytes).hexdigest()
            for key in ("common_grid", "regularization", "inputs", "software"):
                metadata[key] = original.get(key)
        paths = [Path(__file__), PROJECT_DIRECTORY / "main.py"]
        paths.extend(sorted((PROJECT_DIRECTORY / "ect_analysis").glob("*.py")))
        metadata["source_sha256"] = {
            path.relative_to(PROJECT_DIRECTORY).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.is_file()}
        metadata["study_dimensions_match_manuscript"] = (
            exit_code == 0 and error is None
            and sorted(analysis.completed) == list("ABCDEFGHIJKLMNO")
            and analysis.n_channels == 496 and analysis.reference_k == 128
            and metadata.get("common_grid", {}).get("common_occupied_bins") == 2851
            and len(analysis.rows) == 15 * (analysis.expected_cases or 0))
        with (output / "stability_manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, ensure_ascii=False, default=json_ready)
        print(f"Selection stability files saved in: {output}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
