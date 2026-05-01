"""Transition-map metrics for known-correspondence atlas validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransitionMetrics:
    raw_median_deg: float
    raw_p95_deg: float
    raw_max_deg: float
    aligned_median_deg: float
    aligned_p95_deg: float
    aligned_max_deg: float
    smoothness_p95: float
    rotation: np.ndarray
    per_vertex_aligned_deg: np.ndarray


@dataclass(frozen=True)
class CellPullbackMetrics:
    median_jaccard: float
    p10_jaccard: float
    occupied_cell_count: int
    per_cell_jaccard: np.ndarray
    source_labels: np.ndarray
    target_labels: np.ndarray


def unit_rows(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("points must not contain zero rows")
    return array / norms[:, None]


def angular_degrees(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_unit = unit_rows(source)
    target_unit = unit_rows(target)
    if source_unit.shape != target_unit.shape:
        raise ValueError("source and target must have matching shape")
    dots = np.sum(source_unit * target_unit, axis=1)
    return np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))


def procrustes_rotation(moving: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """Return a column-vector SO(3) rotation mapping moving points to fixed."""

    moving_unit = unit_rows(moving)
    fixed_unit = unit_rows(fixed)
    if moving_unit.shape != fixed_unit.shape:
        raise ValueError("moving and fixed must have matching shape")
    covariance = moving_unit.T @ fixed_unit
    u, _, vt = np.linalg.svd(covariance)
    row_rotation = u @ vt
    if np.linalg.det(row_rotation) < 0.0:
        u[:, -1] *= -1.0
        row_rotation = u @ vt
    return row_rotation.T


def matched_vertex_transition_metrics(
    source: np.ndarray,
    target: np.ndarray,
    *,
    faces: np.ndarray | None = None,
) -> TransitionMetrics:
    source_unit = unit_rows(source)
    target_unit = unit_rows(target)
    if source_unit.shape != target_unit.shape:
        raise ValueError("source and target must have matching shape")

    raw_angles = angular_degrees(source_unit, target_unit)
    rotation = procrustes_rotation(target_unit, source_unit)
    aligned_target = target_unit @ rotation.T
    aligned_angles = angular_degrees(source_unit, aligned_target)
    displacement = aligned_target - source_unit
    smoothness = _transition_smoothness(displacement, len(source_unit), faces)

    return TransitionMetrics(
        raw_median_deg=float(np.median(raw_angles)),
        raw_p95_deg=float(np.percentile(raw_angles, 95.0)),
        raw_max_deg=float(np.max(raw_angles)),
        aligned_median_deg=float(np.median(aligned_angles)),
        aligned_p95_deg=float(np.percentile(aligned_angles, 95.0)),
        aligned_max_deg=float(np.max(aligned_angles)),
        smoothness_p95=smoothness,
        rotation=rotation,
        per_vertex_aligned_deg=aligned_angles,
    )


def atlas_cell_pullback_metrics(
    source: np.ndarray,
    target: np.ndarray,
    *,
    n_z: int = 6,
    n_lon: int = 12,
    align: bool = True,
) -> CellPullbackMetrics:
    if n_z < 1 or n_lon < 1:
        raise ValueError("n_z and n_lon must be positive")
    source_unit = unit_rows(source)
    target_unit = unit_rows(target)
    if source_unit.shape != target_unit.shape:
        raise ValueError("source and target must have matching shape")
    if align:
        rotation = procrustes_rotation(target_unit, source_unit)
        target_unit = target_unit @ rotation.T

    source_labels = _sphere_cell_labels(source_unit, n_z=n_z, n_lon=n_lon)
    target_labels = _sphere_cell_labels(target_unit, n_z=n_z, n_lon=n_lon)
    jaccards: list[float] = []
    for label in sorted(set(source_labels.tolist())):
        source_indices = set(np.flatnonzero(source_labels == label).tolist())
        target_indices = set(np.flatnonzero(target_labels == label).tolist())
        union = source_indices | target_indices
        if not union:
            continue
        jaccards.append(len(source_indices & target_indices) / len(union))
    values = np.asarray(jaccards, dtype=np.float64)
    if len(values) == 0:
        values = np.asarray([0.0], dtype=np.float64)
    return CellPullbackMetrics(
        median_jaccard=float(np.median(values)),
        p10_jaccard=float(np.percentile(values, 10.0)),
        occupied_cell_count=int(len(jaccards)),
        per_cell_jaccard=values,
        source_labels=source_labels,
        target_labels=target_labels,
    )


def _sphere_cell_labels(points: np.ndarray, *, n_z: int, n_lon: int) -> np.ndarray:
    unit = unit_rows(points)
    z_bin = np.floor((np.clip(unit[:, 2], -1.0, 1.0) + 1.0) * 0.5 * n_z).astype(np.int64)
    z_bin = np.clip(z_bin, 0, n_z - 1)
    longitude = np.mod(np.arctan2(unit[:, 1], unit[:, 0]), 2.0 * np.pi)
    lon_bin = np.floor(longitude / (2.0 * np.pi) * n_lon).astype(np.int64)
    lon_bin = np.clip(lon_bin, 0, n_lon - 1)
    return z_bin * n_lon + lon_bin


def _transition_smoothness(displacement: np.ndarray, vertex_count: int, faces: np.ndarray | None) -> float:
    edges = _unique_edges(vertex_count, faces)
    if not edges:
        return 0.0
    edge_values = [np.linalg.norm(displacement[a] - displacement[b]) for a, b in edges]
    return float(np.percentile(np.asarray(edge_values, dtype=np.float64), 95.0))


def _unique_edges(vertex_count: int, faces: np.ndarray | None) -> list[tuple[int, int]]:
    if faces is None:
        return [(index, index + 1) for index in range(max(0, vertex_count - 1))]
    faces_array = np.asarray(faces, dtype=np.int64)
    if faces_array.ndim != 2 or faces_array.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3)")
    edges: set[tuple[int, int]] = set()
    for tri in faces_array:
        a, b, c = map(int, tri)
        for left, right in ((a, b), (b, c), (c, a)):
            if left < 0 or right < 0 or left >= vertex_count or right >= vertex_count:
                raise ValueError("faces reference an invalid vertex index")
            if left == right:
                continue
            edges.add((left, right) if left < right else (right, left))
    return sorted(edges)
