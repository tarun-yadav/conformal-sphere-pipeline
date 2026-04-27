"""Resampling of spherical face signals to equiangular grids."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .quality import normalize_rows, spherical_face_centers


def equiangular_grid(nlat: int, nlon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return colatitude, longitude, and unit directions for an equiangular grid."""

    theta = np.pi * (np.arange(nlat, dtype=float) + 0.5) / float(nlat)
    phi = 2.0 * np.pi * np.arange(nlon, dtype=float) / float(nlon)
    st = np.sin(theta)[:, None]
    directions = np.empty((nlat, nlon, 3), dtype=np.float64)
    directions[:, :, 0] = st * np.cos(phi)[None, :]
    directions[:, :, 1] = st * np.sin(phi)[None, :]
    directions[:, :, 2] = np.cos(theta)[:, None]
    return theta, phi, directions


def resample_face_signal_to_equiangular(
    u: np.ndarray,
    faces: np.ndarray,
    values_face: np.ndarray,
    *,
    nlat: int,
    nlon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-face-center resampling of a face signal to an equiangular grid."""

    centers = spherical_face_centers(u, faces)
    values = np.asarray(values_face)
    if values.shape[0] != len(centers):
        raise ValueError("values_face must have one entry per face")
    theta, phi, directions = equiangular_grid(int(nlat), int(nlon))
    tree = cKDTree(centers)
    _, idx = tree.query(normalize_rows(directions.reshape(-1, 3)), k=1)
    grid = values[idx].reshape(int(nlat), int(nlon))
    return grid, theta, phi


def resample_channels_to_equiangular(
    u: np.ndarray,
    faces: np.ndarray,
    channels_face: list[np.ndarray],
    *,
    nlat: int,
    nlon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample multiple face channels to ``(C, nlat, nlon)``."""

    grids = []
    theta = phi = None
    for channel in channels_face:
        grid, theta, phi = resample_face_signal_to_equiangular(
            u, faces, channel, nlat=nlat, nlon=nlon
        )
        grids.append(grid)
    return np.stack(grids, axis=0), theta, phi
