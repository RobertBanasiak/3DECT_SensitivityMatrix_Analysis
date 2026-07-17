from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .constants import ELECTRODES_PER_PLANE, EXPECTED_CHANNELS, EXPECTED_ELECTRODES


def index_to_pair(index: int, n_electrodes: int = EXPECTED_ELECTRODES) -> tuple[int, int]:
    if not 0 <= index < n_electrodes * (n_electrodes - 1) // 2:
        raise ValueError(f"Measurement index {index} is outside the electrode-pair range")
    remaining = index
    for first in range(n_electrodes - 1):
        block = n_electrodes - first - 1
        if remaining < block:
            return first, first + 1 + remaining
        remaining -= block
    raise RuntimeError("Electrode-pair decoder failed")

def classify_pair(first: int, second: int) -> str:
    plane_a, position_a = divmod(first, ELECTRODES_PER_PLANE)
    plane_b, position_b = divmod(second, ELECTRODES_PER_PLANE)
    d_position = min(
        (position_b - position_a) % ELECTRODES_PER_PLANE,
        (position_a - position_b) % ELECTRODES_PER_PLANE,
    )
    if plane_a == plane_b:
        return {
            1: "same-plane adjacent",
            2: "same-plane next-nearest",
            3: "same-plane third-nearest",
            4: "same-plane opposite",
        }.get(d_position, "same-plane other")
    if position_a == position_b:
        return "cross-plane same-angle"
    return "cross-plane other"

def channel_table(n_channels: int) -> pd.DataFrame:
    if n_channels != EXPECTED_CHANNELS:
        return pd.DataFrame(
            {
                "channel_index_zero_based": np.arange(n_channels),
                "electrode_1_one_based": np.nan,
                "electrode_2_one_based": np.nan,
                "channel_class": "unclassified",
            }
        )
    rows = []
    for index in range(n_channels):
        first, second = index_to_pair(index)
        rows.append(
            {
                "channel_index_zero_based": index,
                "electrode_1_one_based": first + 1,
                "electrode_2_one_based": second + 1,
                "plane_1_zero_based": first // ELECTRODES_PER_PLANE,
                "plane_2_zero_based": second // ELECTRODES_PER_PLANE,
                "channel_class": classify_pair(first, second),
            }
        )
    return pd.DataFrame(rows)

def group_masks(channels: pd.DataFrame) -> dict[str, np.ndarray]:
    classes = channels["channel_class"].astype(str)
    return {
        "same-plane adjacent": (classes == "same-plane adjacent").to_numpy(),
        "same-plane next-nearest": (
            classes == "same-plane next-nearest"
        ).to_numpy(),
        "same-plane third-nearest": (
            classes == "same-plane third-nearest"
        ).to_numpy(),
        "same-plane opposite": (classes == "same-plane opposite").to_numpy(),
        "cross-plane same-angle": (
            classes == "cross-plane same-angle"
        ).to_numpy(),
        "cross-plane other": (classes == "cross-plane other").to_numpy(),
    }

def greedy_regularized_logdet(
    gram: np.ndarray,
    alpha: float,
    k: int,
    candidate_mask: np.ndarray | None = None,
) -> tuple[list[int], list[float], list[float]]:
    """Greedy maximization of log det(I + S_K S_K^T / alpha).

    The matrix determinant lemma lets us operate entirely on the small channel
    Gram matrix.  Every accepted marginal gain is non-negative apart from
    floating-point roundoff.
    """
    n_channels = gram.shape[0]
    if gram.shape != (n_channels, n_channels):
        raise ValueError("gram must be square")
    if candidate_mask is None:
        remaining = np.arange(n_channels, dtype=int)
    else:
        candidate_mask = np.asarray(candidate_mask, dtype=bool)
        if candidate_mask.shape != (n_channels,):
            raise ValueError("candidate_mask has the wrong length")
        remaining = np.flatnonzero(candidate_mask)
    target = min(int(k), len(remaining))
    scaled = gram / alpha
    diagonal = 1.0 + np.diag(scaled)
    selected: list[int] = []
    cumulative: list[float] = []
    marginals: list[float] = []
    inverse: np.ndarray | None = None
    total = 0.0

    for _ in range(target):
        if not selected:
            schur = diagonal[remaining]
        else:
            cross = scaled[np.ix_(selected, remaining)]
            schur = diagonal[remaining] - np.sum(cross * (inverse @ cross), axis=0)
        best_position = int(np.argmax(schur))
        best = int(remaining[best_position])
        best_schur = float(schur[best_position])
        if best_schur < 1.0 - 1e-8:
            raise FloatingPointError(f"Regularized Schur complement fell below one: {best_schur}")
        best_schur = max(best_schur, 1.0)
        marginal = math.log(best_schur)

        if inverse is None:
            inverse = np.array([[1.0 / best_schur]])
        else:
            cross_best = scaled[np.ix_(selected, [best])]
            inverse_cross = inverse @ cross_best
            inverse = np.block(
                [
                    [inverse + inverse_cross @ inverse_cross.T / best_schur, -inverse_cross / best_schur],
                    [(-inverse_cross / best_schur).T, np.array([[1.0 / best_schur]])],
                ]
            )
        selected.append(best)
        total += marginal
        marginals.append(marginal)
        cumulative.append(total)
        remaining = np.delete(remaining, best_position)
    return selected, cumulative, marginals
