from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import weighted_cdf


def publication_line_styles(labels: Sequence[str]) -> dict[str, dict[str, object]]:
    """Consistent, distinguishable styles for A--O across every line figure."""
    ordered = sorted(labels)
    colours = plt.get_cmap("tab20")(np.linspace(0.0, 0.95, len(ordered)))
    line_styles = ("-", "--", "-.")
    return {
        label: {
            "color": colours[index],
            "linestyle": line_styles[index // 5],
            "linewidth": 1.7,
        }
        for index, label in enumerate(ordered)
    }

def plot_singular_spectra(
    eigenvalues_by_label: dict[str, np.ndarray], alpha: float, path: Path, dpi: int
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    styles = publication_line_styles(eigenvalues_by_label.keys())
    for label, eigenvalues in sorted(eigenvalues_by_label.items()):
        normalized_singular = np.sqrt(np.maximum(eigenvalues[::-1], 0.0) / alpha)
        ax.semilogy(
            np.arange(1, len(normalized_singular) + 1),
            normalized_singular,
            label=label,
            **styles[label],
        )
    ax.set_xlabel("Mode index")
    ax.set_ylabel(r"Normalized singular value $s_k/\sqrt{\alpha}$")
    ax.set_title("Common-basis sensitivity spectra")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def plot_cdfs(
    values_by_label: dict[str, np.ndarray],
    volumes: np.ndarray,
    xlabel: str,
    title: str,
    path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    styles = publication_line_styles(values_by_label.keys())
    for label, values in sorted(values_by_label.items()):
        x, cumulative = weighted_cdf(values, volumes)
        ax.plot(x, cumulative, label=label, **styles[label])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Common-domain volume fraction")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def plot_subset_curves(curves: dict[str, Sequence[float]], path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    styles = publication_line_styles(curves.keys())
    for label, values in sorted(curves.items()):
        ax.plot(np.arange(1, len(values) + 1), values, label=label, **styles[label])
    ax.set_xlabel("Selected channel count K")
    ax.set_ylabel(r"$\log\det(I + S_K S_K^T / \alpha)$")
    ax.set_title("Monotone regularized information objective")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def plot_phantom_maps(raw: pd.DataFrame, metric: str, path: Path, dpi: int) -> None:
    valid = raw[(raw["status"] == "ok") & np.isfinite(raw[metric])]
    labels = sorted(valid["configuration"].unique())
    if not labels:
        return
    radial = np.sort(valid["r_mm"].unique())
    axial = np.sort(valid["z_mm"].unique())
    aggregate = valid.groupby(["configuration", "r_mm", "z_mm"], as_index=False)[metric].median()
    global_min = float(aggregate[metric].min())
    global_max = float(aggregate[metric].max())
    if np.isclose(global_min, global_max):
        global_max = global_min + 1e-12
    ncols = 5
    nrows = math.ceil(len(labels) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.0 * ncols, 2.7 * nrows), sharex=True, sharey=True, squeeze=False
    )
    image = None
    for axis, label in zip(axes.ravel(), labels):
        subset = aggregate[aggregate["configuration"] == label]
        matrix = np.full((len(axial), len(radial)), np.nan)
        for row in subset.itertuples(index=False):
            iz = int(np.where(np.isclose(axial, row.z_mm))[0][0])
            ir = int(np.where(np.isclose(radial, row.r_mm))[0][0])
            matrix[iz, ir] = getattr(row, metric)
        image = axis.imshow(
            matrix,
            origin="lower",
            extent=[radial.min(), radial.max(), axial.min(), axial.max()],
            aspect="auto",
            vmin=global_min,
            vmax=global_max,
            cmap="viridis",
        )
        axis.set_title(label)
    for axis in axes.ravel()[len(labels) :]:
        axis.axis("off")
    for row in range(nrows):
        axes[row, 0].set_ylabel("z [mm]")
    for axis in axes[-1, :]:
        if axis.axison:
            axis.set_xlabel("r [mm]")
    metric_labels = {
        "localization_error_mm": "localization error [mm]",
        "auc": "AUC",
        "dice_50pct_peak": "Dice coefficient at 50% of peak",
        "nrmse": "NRMSE",
    }
    display_label = metric_labels.get(metric, metric.replace("_", " "))
    if image is not None:
        color_axis = fig.add_axes([0.925, 0.17, 0.014, 0.66])
        colorbar = fig.colorbar(image, cax=color_axis)
        colorbar.set_label(display_label)
    fig.suptitle(f"Paired partial-volume spherical phantoms: median {display_label}")
    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.09, top=0.88, wspace=0.12, hspace=0.25)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
