from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix

from .cli import label_from_path
from .models import CommonGrid, MappedDataset, MeshMeta

try:
    import h5py
except ModuleNotFoundError:  # Help and self-test remain available without HDF5 I/O.
    h5py = None


def _orient_columns(array: np.ndarray, n_columns: int, name: str) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got {array.shape}")
    if array.shape[1] == n_columns:
        return array
    if array.shape[0] == n_columns:
        return array.T
    raise ValueError(f"Cannot orient {name}={array.shape} as (*,{n_columns})")

def _real_array(array: np.ndarray, name: str) -> np.ndarray:
    if np.iscomplexobj(array):
        imag = float(np.max(np.abs(array.imag)))
        real = float(np.max(np.abs(array.real))) + 1e-30
        if imag / real > 1e-10:
            raise ValueError(f"{name} contains material imaginary components")
        array = array.real
    return np.asarray(array)

def inspect_mesh(path: Path, mesh_unit: str) -> MeshMeta:
    if h5py is None:
        raise ModuleNotFoundError(
            "h5py is required to read MATLAB v7.3 files; install it with 'python -m pip install h5py'"
        )
    factor = 1.0 if mesh_unit == "mm" else 1000.0
    with h5py.File(path, "r") as handle:
        for key in ("S", "vtx", "simp"):
            if key not in handle:
                raise KeyError(f"{path}: missing HDF5 dataset {key!r}")
        vertices = _orient_columns(_real_array(np.array(handle["vtx"]), "vtx"), 3, "vtx")
        simplices = _orient_columns(np.array(handle["simp"]), 4, "simp")
        s_shape = tuple(handle["S"].shape)
    vertices = np.asarray(vertices, dtype=float) * factor
    simplices = np.asarray(simplices, dtype=np.int64)
    if simplices.min() == 1:
        simplices -= 1
    if simplices.min() < 0 or simplices.max() >= len(vertices):
        raise ValueError(f"{path}: simplex indices are outside the vertex array")
    n_cells = len(simplices)
    if s_shape[0] == n_cells:
        n_channels = s_shape[1]
    elif s_shape[1] == n_cells:
        n_channels = s_shape[0]
    else:
        raise ValueError(f"{path}: S shape {s_shape} is incompatible with {n_cells} cells")
    r_vertex = np.hypot(vertices[:, 0], vertices[:, 1])
    return MeshMeta(
        label=label_from_path(path),
        path=str(path),
        n_cells=n_cells,
        n_vertices=len(vertices),
        n_channels=int(n_channels),
        r_max_mm=float(r_vertex.max()),
        z_min_mm=float(vertices[:, 2].min()),
        z_max_mm=float(vertices[:, 2].max()),
    )

def load_mesh(
    meta: MeshMeta, mesh_unit: str, dtype: np.dtype
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if h5py is None:
        raise ModuleNotFoundError(
            "h5py is required to read MATLAB v7.3 files; install it with 'python -m pip install h5py'"
        )
    factor = 1.0 if mesh_unit == "mm" else 1000.0
    with h5py.File(meta.path, "r") as handle:
        vertices = _orient_columns(_real_array(np.array(handle["vtx"]), "vtx"), 3, "vtx")
        simplices = _orient_columns(np.array(handle["simp"]), 4, "simp")
        sensitivity_raw = _real_array(np.array(handle["S"]), "S")
    vertices = np.asarray(vertices, dtype=float) * factor
    simplices = np.asarray(simplices, dtype=np.int64)
    if simplices.min() == 1:
        simplices -= 1
    if sensitivity_raw.shape[0] == meta.n_cells:
        sensitivity = sensitivity_raw
    elif sensitivity_raw.shape[1] == meta.n_cells:
        sensitivity = sensitivity_raw.T
    else:
        raise ValueError(f"{meta.path}: sensitivity shape changed after inspection")
    sensitivity = np.asarray(sensitivity, dtype=dtype)
    if not np.all(np.isfinite(sensitivity)):
        raise ValueError(f"{meta.path}: S contains NaN or infinite values")
    return sensitivity, vertices, simplices

def tetra_volumes(vertices: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    points = vertices[simplices]
    matrices = np.stack(
        [
            points[:, 1] - points[:, 0],
            points[:, 2] - points[:, 0],
            points[:, 3] - points[:, 0],
        ],
        axis=-1,
    )
    volumes = np.abs(np.linalg.det(matrices)) / 6.0
    if np.any(volumes <= 0):
        raise ValueError("Degenerate tetrahedra detected")
    return volumes

def make_common_grid(metas: Sequence[MeshMeta], nr: int, ntheta: int, nz: int) -> CommonGrid:
    r_max = min(m.r_max_mm for m in metas)
    z_min = max(m.z_min_mm for m in metas)
    z_max = min(m.z_max_mm for m in metas)
    if r_max <= 0 or z_max <= z_min:
        raise ValueError("Meshes do not have a non-empty common cylindrical domain")

    # sqrt spacing gives equal cross-sectional area, hence equal 3D bin volume.
    r_edges = np.sqrt(np.linspace(0.0, r_max**2, nr + 1))
    theta_edges = np.linspace(-np.pi, np.pi, ntheta + 1)
    z_edges = np.linspace(z_min, z_max, nz + 1)
    theta_centres = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    z_centres = 0.5 * (z_edges[:-1] + z_edges[1:])
    # Radial centroid of an annular sector, not merely the midpoint.
    numerator = (2.0 / 3.0) * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)
    denominator = r_edges[1:] ** 2 - r_edges[:-1] ** 2
    r_centres = numerator / denominator
    zz, tt, rr = np.meshgrid(z_centres, theta_centres, r_centres, indexing="ij")
    xyz = np.column_stack(
        [
            (rr * np.cos(tt)).ravel(),
            (rr * np.sin(tt)).ravel(),
            zz.ravel(),
        ]
    )
    dtheta = theta_edges[1] - theta_edges[0]
    dz = z_edges[1] - z_edges[0]
    bin_volume = 0.5 * (r_edges[1] ** 2 - r_edges[0] ** 2) * dtheta * dz
    return CommonGrid(
        nr=nr,
        ntheta=ntheta,
        nz=nz,
        r_edges_mm=r_edges,
        theta_edges_rad=theta_edges,
        z_edges_mm=z_edges,
        xyz_centres_mm=xyz,
        r_centres_mm=r_centres,
        theta_centres_rad=theta_centres,
        z_centres_mm=z_centres,
        bin_volume_mm3=float(bin_volume),
    )

def _bin_indices(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(edges, values, side="right") - 1
    idx[np.isclose(values, edges[-1], rtol=0.0, atol=1e-10)] = len(edges) - 2
    return idx

def map_to_common_grid(
    meta: MeshMeta,
    grid: CommonGrid,
    mesh_unit: str,
    sensitivity_kind: str,
    dtype: np.dtype,
) -> MappedDataset:
    sensitivity, vertices, simplices = load_mesh(meta, mesh_unit, dtype)
    centres = vertices[simplices].mean(axis=1)
    volumes = tetra_volumes(vertices, simplices)
    radius = np.hypot(centres[:, 0], centres[:, 1])
    theta = np.arctan2(centres[:, 1], centres[:, 0])
    z_coord = centres[:, 2]
    ir = _bin_indices(radius, grid.r_edges_mm)
    it = _bin_indices(theta, grid.theta_edges_rad)
    iz = _bin_indices(z_coord, grid.z_edges_mm)
    valid = (
        (ir >= 0)
        & (ir < grid.nr)
        & (it >= 0)
        & (it < grid.ntheta)
        & (iz >= 0)
        & (iz < grid.nz)
    )
    flat = np.ravel_multi_index((iz[valid], it[valid], ir[valid]), grid.shape)
    native_rows = np.flatnonzero(valid)
    aggregation_weight = (
        np.ones(len(native_rows), dtype=float)
        if sensitivity_kind == "element-integrated"
        else volumes[valid]
    )
    projector = csr_matrix(
        (aggregation_weight, (flat, native_rows)),
        shape=(grid.n_bins, meta.n_cells),
    )
    common_sensitivity = np.asarray(projector @ sensitivity, dtype=float)
    occupied_volume = np.bincount(
        flat, weights=volumes[valid], minlength=grid.n_bins
    ).astype(float)
    fraction = float(volumes[valid].sum() / volumes.sum())
    return MappedDataset(meta, common_sensitivity, occupied_volume, fraction)
