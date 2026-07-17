from __future__ import annotations

from typing import Iterable

import numpy as np

from .constants import EPS


def weighted_percentiles(x: np.ndarray, w: np.ndarray, ps: Iterable[float]) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    weights = np.asarray(w, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[valid], weights[valid]
    if values.size == 0:
        return np.full(len(tuple(ps)), np.nan)
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    return np.array([np.interp(p / 100.0, cumulative, values) for p in ps])

def weighted_gini(x: np.ndarray, w: np.ndarray) -> float:
    values = np.asarray(x, dtype=float)
    weights = np.asarray(w, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0) & (values >= 0)
    values, weights = values[valid], weights[valid]
    if values.size == 0 or np.all(values == 0):
        return float("nan")
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    cumulative_population = np.concatenate([[0.0], np.cumsum(weights)])
    cumulative_value = np.concatenate([[0.0], np.cumsum(values * weights)])
    cumulative_population /= cumulative_population[-1]
    cumulative_value /= cumulative_value[-1]
    # G = 1 - 2 * area under the Lorenz curve.  The trapezoidal form works
    # for arbitrary positive weights and is bounded by [0, 1].
    lorenz_area = 0.5 * np.sum(
        (cumulative_value[1:] + cumulative_value[:-1])
        * np.diff(cumulative_population)
    )
    return float(np.clip(1.0 - 2.0 * lorenz_area, 0.0, 1.0))

def weighted_cdf(x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(x, dtype=float)
    weights = np.asarray(w, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[valid], weights[valid]
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    return values, cumulative

def gram_and_eigenvalues(sensitivity: np.ndarray, noise_std: float) -> tuple[np.ndarray, np.ndarray]:
    whitened = sensitivity / noise_std
    gram = np.asarray(whitened.T @ whitened, dtype=float)
    gram = 0.5 * (gram + gram.T)
    eigenvalues = np.linalg.eigvalsh(gram)
    numerical_floor = 100.0 * EPS * max(float(np.max(np.abs(eigenvalues))), 1.0)
    eigenvalues[(eigenvalues < 0) & (eigenvalues > -numerical_floor)] = 0.0
    if np.any(eigenvalues < 0):
        raise ValueError("Gram matrix has materially negative eigenvalues")
    return gram, eigenvalues

def regularized_spectral_metrics(
    sensitivity: np.ndarray,
    alpha: float,
    noise_std: float,
    gram: np.ndarray | None = None,
    eigenvalues: np.ndarray | None = None,
) -> dict[str, float | int]:
    if gram is None or eigenvalues is None:
        gram, eigenvalues = gram_and_eigenvalues(sensitivity, noise_std)
    maximum = float(eigenvalues[-1]) if eigenvalues.size else 0.0
    tolerance = max(sensitivity.shape) * EPS * maximum
    numerical_rank = int(np.count_nonzero(eigenvalues > tolerance))
    energy_sum = float(eigenvalues.sum())
    if energy_sum > 0:
        probabilities = eigenvalues / energy_sum
        nonzero = probabilities > 0
        effective_rank = float(np.exp(-np.sum(probabilities[nonzero] * np.log(probabilities[nonzero]))))
        stable_rank = float(energy_sum / maximum)
    else:
        effective_rank = 0.0
        stable_rank = 0.0

    information_fraction = eigenvalues / (eigenvalues + alpha)
    posterior_fraction = alpha / (eigenvalues + alpha)
    logdet_information = float(np.sum(np.log1p(eigenvalues / alpha)))

    norms = np.sqrt(np.clip(np.diag(gram), 0.0, None))
    denom = np.outer(norms, norms)
    correlation = np.zeros_like(gram)
    np.divide(gram, denom, out=correlation, where=denom > 0)
    np.fill_diagonal(correlation, 0.0)
    coherence = float(np.max(np.abs(correlation))) if correlation.size else 0.0
    return {
        "regularized_logdet": logdet_information,
        "mean_posterior_fraction": float(np.mean(posterior_fraction)),
        "worst_information_fraction": float(np.min(information_fraction)),
        "effective_rank_energy": effective_rank,
        "stable_rank": stable_rank,
        "numerical_rank": numerical_rank,
        "coherence": coherence,
        "spectral_energy": energy_sum,
        "largest_eigenvalue": maximum,
    }

def spatial_metrics(
    sensitivity: np.ndarray, noise_std: float, volumes: np.ndarray
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    local_norm = np.linalg.norm(sensitivity / noise_std, axis=1)
    envelope = np.max(np.abs(sensitivity), axis=1)
    local_p = weighted_percentiles(local_norm, volumes, (50, 90, 99))
    env_p = weighted_percentiles(envelope, volumes, (50, 90, 99))
    e99 = env_p[2]
    weak5 = float(np.sum(volumes[envelope <= 0.05 * e99]) / volumes.sum()) if e99 > 0 else np.nan
    weak10 = float(np.sum(volumes[envelope <= 0.10 * e99]) / volumes.sum()) if e99 > 0 else np.nan
    metrics = {
        "local_norm_p50": float(local_p[0]),
        "local_norm_p90": float(local_p[1]),
        "local_norm_p99": float(local_p[2]),
        "envelope_p50": float(env_p[0]),
        "envelope_p90": float(env_p[1]),
        "envelope_p99": float(env_p[2]),
        "envelope_gini": weighted_gini(envelope, volumes),
        "weak_volume_5pct_e99": weak5,
        "weak_volume_10pct_e99": weak10,
    }
    return metrics, local_norm, envelope
