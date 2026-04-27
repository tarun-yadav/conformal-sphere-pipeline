"""Optional lapy-backed spherical conformal parameterization."""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from .quality import normalize_rows

logger = logging.getLogger(__name__)
_lapy_patch_lock = threading.Lock()


def _as_mesh_arrays(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3)")
    if faces.min(initial=0) < 0 or faces.max(initial=-1) >= len(vertices):
        raise ValueError("faces contain vertex indices outside the vertex array")
    return vertices, faces


def _compute_lapy_map(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, int]:
    try:
        from lapy import TriaMesh
        from lapy.conformal import spherical_conformal_map
        import lapy.conformal as lapy_conformal
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "lapy is required for method='conformal'. Reinstall "
            "conformal-sphere-pipeline with its runtime dependencies."
        ) from exc

    tria = TriaMesh(vertices.astype(np.float64), faces.astype(np.int32))
    euler_number = int(tria.euler())
    if euler_number != 2:
        raise ValueError(f"mesh must be genus-0 for conformal mapping; Euler number is {euler_number}")

    def _patched_ensure_inplace(arr: np.ndarray, name: str = "array") -> None:
        near_zero = np.abs(arr) < 1e-15
        if np.any(near_zero):
            logger.warning(
                "lapy pole safeguard perturbed %d near-zero value(s) in %s",
                int(np.sum(near_zero)),
                name,
            )
            arr[near_zero] = 1e-10

    original_ensure: Any = getattr(lapy_conformal, "_ensure_nonzero_array", None)
    if original_ensure is None:
        return spherical_conformal_map(tria), euler_number

    with _lapy_patch_lock:
        try:
            lapy_conformal._ensure_nonzero_array = _patched_ensure_inplace
            sphere = spherical_conformal_map(tria)
        finally:
            lapy_conformal._ensure_nonzero_array = original_ensure
    return sphere, euler_number


def _fix_global_orientation(sphere: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, float, bool]:
    v0 = sphere[faces[:, 0]]
    v1 = sphere[faces[:, 1]]
    v2 = sphere[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    centroid_dir = (v0 + v1 + v2) / 3.0
    signed_area = np.sum(cross * centroid_dir, axis=1)
    positive_fraction = float(np.mean(signed_area > 0.0))
    flipped = positive_fraction < 0.5
    if flipped:
        sphere = sphere.copy()
        sphere[:, 0] *= -1.0
    return sphere, positive_fraction, flipped


def compute_spherical_conformal_map(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Compute a genus-0 spherical conformal map with lapy.

    The backend is optional so ``method='auto'`` can fall back cleanly when
    lapy is unavailable. The output is normalized onto the unit sphere and
    globally flipped if lapy returns an orientation-reversing parameterization.
    """

    vertices, faces = _as_mesh_arrays(vertices, faces)
    sphere, euler_number = _compute_lapy_map(vertices, faces)
    sphere = normalize_rows(np.asarray(sphere, dtype=np.float64))
    sphere, positive_fraction, flipped = _fix_global_orientation(sphere, faces)
    sphere = normalize_rows(sphere)

    info = {
        "method": "conformal",
        "backend": "lapy",
        "euler_number": euler_number,
        "max_sphere_norm_error": float(np.max(np.abs(np.linalg.norm(sphere, axis=1) - 1.0))),
        "orientation_positive_fraction": positive_fraction,
        "orientation_flipped": bool(flipped),
    }
    if return_info:
        return sphere, info
    return sphere
