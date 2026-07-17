from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MeshMeta:
    label: str
    path: str
    n_cells: int
    n_vertices: int
    n_channels: int
    r_max_mm: float
    z_min_mm: float
    z_max_mm: float

@dataclass(frozen=True)
class CommonGrid:
    nr: int
    ntheta: int
    nz: int
    r_edges_mm: np.ndarray
    theta_edges_rad: np.ndarray
    z_edges_mm: np.ndarray
    xyz_centres_mm: np.ndarray
    r_centres_mm: np.ndarray
    theta_centres_rad: np.ndarray
    z_centres_mm: np.ndarray
    bin_volume_mm3: float

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nz, self.ntheta, self.nr)

    @property
    def n_bins(self) -> int:
        return self.nr * self.ntheta * self.nz

@dataclass
class MappedDataset:
    meta: MeshMeta
    sensitivity: np.ndarray
    occupied_volume_mm3: np.ndarray
    mapped_native_volume_fraction: float
