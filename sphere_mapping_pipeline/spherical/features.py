"""Physical-surface features projected to canonical spherical coordinates."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .quality import face_areas, spherical_face_areas


def weighted_zscore(values: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Weighted z-score over masked entries, zeros elsewhere for invalid values."""

    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    active = mask & np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    out = np.zeros_like(values, dtype=np.float64)
    if not np.any(active):
        return out
    w = weights[active] / float(weights[active].sum())
    mean = float(np.sum(w * values[active]))
    std = float(np.sqrt(np.sum(w * (values[active] - mean) ** 2)))
    std = max(std, 1e-12)
    out[active] = (values[active] - mean) / std
    return out


def _mean_curvature_vertices(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Approximate unsigned mean curvature with cotangent weights."""

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    n_vertices = len(vertices)
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
    vi, vj, vk = vertices[i], vertices[j], vertices[k]
    cross = np.cross(vj - vi, vk - vi)
    twice_area = np.linalg.norm(cross, axis=1)
    safe = twice_area > 1e-15
    cot_i = np.where(safe, np.sum((vj - vi) * (vk - vi), axis=1) / twice_area, 0.0)
    cot_j = np.where(safe, np.sum((vk - vj) * (vi - vj), axis=1) / twice_area, 0.0)
    cot_k = np.where(safe, np.sum((vi - vk) * (vj - vk), axis=1) / twice_area, 0.0)

    h = np.zeros((n_vertices, 3), dtype=np.float64)
    np.add.at(h, i, cot_j[:, None] * (vk - vi) + cot_k[:, None] * (vj - vi))
    np.add.at(h, j, cot_k[:, None] * (vi - vj) + cot_i[:, None] * (vk - vj))
    np.add.at(h, k, cot_i[:, None] * (vj - vk) + cot_j[:, None] * (vi - vk))

    area = np.zeros(n_vertices, dtype=np.float64)
    face_area = 0.5 * twice_area
    for corner in range(3):
        np.add.at(area, faces[:, corner], face_area / 3.0)
    out = np.zeros(n_vertices, dtype=np.float64)
    valid = area > 1e-15
    out[valid] = 0.25 * np.linalg.norm(h[valid], axis=1) / area[valid]
    return out


def _boundary_distance_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    physical_face_mask: np.ndarray,
) -> np.ndarray:
    """Return normalized physical-surface distance from virtual interfaces."""

    physical_face_mask = np.asarray(physical_face_mask, dtype=bool)
    if np.all(physical_face_mask) or not np.any(physical_face_mask):
        return np.zeros(len(faces), dtype=np.float64)

    physical_faces = faces[physical_face_mask]
    virtual_faces = faces[~physical_face_mask]
    physical_vertices = np.unique(physical_faces.ravel())
    virtual_vertices = np.unique(virtual_faces.ravel())
    boundary_vertices = np.intersect1d(physical_vertices, virtual_vertices, assume_unique=False)
    if len(boundary_vertices) == 0:
        return np.zeros(len(faces), dtype=np.float64)

    edge_i: list[int] = []
    edge_j: list[int] = []
    edge_w: list[float] = []
    for tri in physical_faces:
        for corner in range(3):
            a = int(tri[corner])
            b = int(tri[(corner + 1) % 3])
            weight = float(np.linalg.norm(vertices[a] - vertices[b]))
            edge_i.extend([a, b])
            edge_j.extend([b, a])
            edge_w.extend([weight, weight])

    graph = csr_matrix((edge_w, (edge_i, edge_j)), shape=(len(vertices), len(vertices)))
    distances = dijkstra(graph, directed=False, indices=boundary_vertices, min_only=True)
    distances = np.asarray(distances, dtype=np.float64)
    distances[~np.isfinite(distances)] = 0.0
    active = physical_vertices[np.isfinite(distances[physical_vertices])]
    if len(active) == 0:
        return np.zeros(len(faces), dtype=np.float64)
    scale = float(np.max(distances[active]))
    if scale <= 1e-15:
        return np.zeros(len(faces), dtype=np.float64)
    distances = distances / scale
    out = np.zeros(len(faces), dtype=np.float64)
    out[physical_face_mask] = distances[physical_faces].mean(axis=1)
    return out


def compute_face_features(
    vertices: np.ndarray,
    faces: np.ndarray,
    sphere: np.ndarray,
    physical_face_mask: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compute default per-face sphere channels and supporting area arrays."""

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    physical_face_mask = np.asarray(physical_face_mask, dtype=bool)

    area_3d = face_areas(vertices, faces)
    area_sphere = spherical_face_areas(sphere, faces)
    lambda_face = area_3d / np.maximum(area_sphere, 1e-15)
    sphere_per_physical = area_sphere / np.maximum(area_3d, 1e-15)
    log_lambda = np.log(np.maximum(lambda_face, 1e-30))
    log_lambda_norm = weighted_zscore(log_lambda, area_3d, physical_face_mask)

    physical_faces = faces[physical_face_mask]
    curvature_v = _mean_curvature_vertices(vertices, physical_faces)
    curvature_face = np.zeros(len(faces), dtype=np.float64)
    curvature_face[physical_face_mask] = curvature_v[physical_faces].mean(axis=1)
    curvature_norm = weighted_zscore(curvature_face, area_3d, physical_face_mask)

    physical_vertices = np.unique(physical_faces.ravel())
    centroid = vertices[physical_vertices].mean(axis=0) if len(physical_vertices) else vertices.mean(axis=0)
    radial_v = np.linalg.norm(vertices - centroid, axis=1)
    scale = max(float(np.sqrt(area_3d[physical_face_mask].sum() / (4.0 * np.pi))), 1e-12)
    radial_face = radial_v[faces].mean(axis=1) / scale
    radial_norm = weighted_zscore(radial_face, area_3d, physical_face_mask)

    channels = {
        "radius": radial_face,
        "forward_jacobian": sphere_per_physical,
        "inverse_density": lambda_face,
        "conformal_factor": 0.5 * np.log(np.maximum(sphere_per_physical, 1e-30)),
        "mean_curvature": curvature_face,
        "log_conformal_factor": log_lambda_norm,
        "mean_curvature_zscore": curvature_norm,
        "radial_distance": radial_norm,
        "physical_mask": physical_face_mask.astype(np.float64),
        "boundary_distance": _boundary_distance_faces(vertices, faces, physical_face_mask),
    }
    support = {
        "area_3d": area_3d,
        "area_sphere": area_sphere,
        "lambda_face": lambda_face,
        "log_lambda": log_lambda,
    }
    return channels, support
