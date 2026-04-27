"""Spherical parameterization wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .quality import normalize_rows

PARAMETERIZATION_METHODS = {"auto", "conformal", "radial"}


class SphericalParameterizer(Protocol):
    """Interface for spherical parameterizers."""

    def parameterize(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Return one unit-sphere coordinate per vertex."""


@dataclass
class ParameterizationResult:
    sphere: np.ndarray
    info: dict


def radial_parameterization(vertices: np.ndarray) -> np.ndarray:
    """Fallback sphere map by centroid radial projection."""

    vertices = np.asarray(vertices, dtype=np.float64)
    centered = vertices - vertices.mean(axis=0)
    norms = np.linalg.norm(centered, axis=1)
    if np.any(norms < 1e-12):
        centered = centered + np.array([1e-6, 0.0, 0.0])
    return normalize_rows(centered)


def parameterize_sphere(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    method: str = "auto",
) -> ParameterizationResult:
    """Compute or reuse a spherical parameterization.

    ``method='auto'`` first attempts the optional lapy conformal backend and
    falls back to deterministic radial projection when dependencies or mesh
    conditions prevent conformal mapping. ``method='conformal'`` requires that
    backend and raises on failure. ``method='radial'`` selects the fallback
    explicitly, which is useful for tests and diagnostics.
    """

    if method not in PARAMETERIZATION_METHODS:
        raise ValueError("parameterization method must be auto, conformal, or radial")

    if method in {"auto", "conformal"}:
        try:
            from .conformal import compute_spherical_conformal_map

            result = compute_spherical_conformal_map(vertices, faces, return_info=True)
            if isinstance(result, tuple):
                sphere, info = result
            else:
                sphere, info = result, {}
            return ParameterizationResult(
                sphere=normalize_rows(sphere),
                info={"method": "conformal", **info},
            )
        except Exception as exc:
            if method == "conformal":
                raise
            info = {
                "method": "radial",
                "fallback_reason": f"{type(exc).__name__}: {exc}",
            }
            return ParameterizationResult(sphere=radial_parameterization(vertices), info=info)

    return ParameterizationResult(
        sphere=radial_parameterization(vertices),
        info={"method": "radial", "fallback_reason": "requested"},
    )
