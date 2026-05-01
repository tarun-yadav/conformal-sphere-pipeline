"""Geometry utilities and quality metrics on Euclidean and spherical meshes."""

from __future__ import annotations

import numpy as np


def normalize_rows(x: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Normalize each row of ``x`` to unit length."""

    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms > eps, norms, 1.0)


def face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return Euclidean triangle areas."""

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def face_barycenters(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return Euclidean face barycenters."""

    return np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)].mean(axis=1)


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return unit face normals, leaving degenerate normals as zero."""

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    normals = np.cross(b - a, c - a)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.where(norms > 1e-15, norms, 1.0)


def spherical_face_centers(u: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return normalized spherical face centers."""

    return normalize_rows(face_barycenters(u, faces))


def spherical_triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the area of one spherical triangle on the unit sphere."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    ab = float(np.clip(np.dot(a, b), -1.0, 1.0))
    bc = float(np.clip(np.dot(b, c), -1.0, 1.0))
    ca = float(np.clip(np.dot(c, a), -1.0, 1.0))
    side_a = np.arccos(bc)
    side_b = np.arccos(ca)
    side_c = np.arccos(ab)
    s = 0.5 * (side_a + side_b + side_c)
    tan_term = (
        np.tan(0.5 * s)
        * np.tan(0.5 * (s - side_a))
        * np.tan(0.5 * (s - side_b))
        * np.tan(0.5 * (s - side_c))
    )
    if np.isfinite(tan_term) and tan_term > 1e-30:
        return float(4.0 * np.arctan(np.sqrt(tan_term)))

    planar = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
    return float(max(planar, 0.0))


def spherical_face_areas(u: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return spherical triangle areas for all faces."""

    u = normalize_rows(u)
    faces = np.asarray(faces, dtype=np.int64)
    areas = np.empty(len(faces), dtype=np.float64)
    for idx, tri in enumerate(faces):
        areas[idx] = spherical_triangle_area(u[tri[0]], u[tri[1]], u[tri[2]])
    return areas


def sphere_norm_metrics(u: np.ndarray) -> dict:
    """Return basic unit-sphere norm diagnostics."""

    norms = np.linalg.norm(np.asarray(u, dtype=np.float64), axis=1)
    return {
        "sphere_norm_max_error": float(np.max(np.abs(norms - 1.0))),
        "sphere_norm_mean_error": float(np.mean(np.abs(norms - 1.0))),
    }
