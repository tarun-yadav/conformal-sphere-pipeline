"""Gauge helpers for known-correspondence atlas validation."""

from __future__ import annotations

import numpy as np

from conformal_sphere_pipeline.spherical.orientation import canonical_orient_sphere
from conformal_sphere_pipeline.spherical.quality import face_areas
from conformal_sphere_pipeline.validation.synthetic_cases import MaterialMesh


def select_material_pole_face(
    mesh: MaterialMesh,
    *,
    target_s: float = 0.5,
    target_theta: float = 0.0,
    theta_weight: float = 0.05,
) -> int:
    """Select a stable pole face from known material/anatomical coordinates."""

    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("mesh.faces must have shape (M, 3)")
    material_s = np.asarray(mesh.material_s, dtype=np.float64)
    material_theta = np.asarray(mesh.material_theta, dtype=np.float64)
    if material_s.shape != (len(mesh.vertices),) or material_theta.shape != (len(mesh.vertices),):
        raise ValueError("material arrays must match mesh vertex count")
    face_s = np.mean(material_s[faces], axis=1)
    face_theta = _circular_mean(material_theta[faces], axis=1)
    theta_delta = np.arctan2(np.sin(face_theta - target_theta), np.cos(face_theta - target_theta))
    score = (face_s - float(target_s)) ** 2 + float(theta_weight) * theta_delta * theta_delta
    return int(np.argmin(score))


def material_face_signal(mesh: MaterialMesh) -> np.ndarray:
    """A non-degenerate scalar material signal for SH gauge experiments."""

    faces = np.asarray(mesh.faces, dtype=np.int64)
    s = np.mean(np.asarray(mesh.material_s, dtype=np.float64)[faces], axis=1)
    theta = _circular_mean(np.asarray(mesh.material_theta, dtype=np.float64)[faces], axis=1)
    values = s + 0.35 * np.cos(theta) + 0.17 * np.sin(theta) + 0.11 * np.cos(2.0 * theta)
    return np.asarray(values, dtype=np.float64)


def harmonic_orient_material_sphere(
    sphere: np.ndarray,
    mesh: MaterialMesh,
    values_face: np.ndarray | None = None,
    *,
    lmax_orientation: int = 6,
) -> tuple[np.ndarray, dict]:
    """Orient a spherical map using Li-Hartley-style SH constraints on a material signal."""

    faces = np.asarray(mesh.faces, dtype=np.int64)
    values = material_face_signal(mesh) if values_face is None else np.asarray(values_face, dtype=np.float64)
    if values.shape != (len(faces),):
        raise ValueError("values_face must have one value per face")
    weights = face_areas(np.asarray(sphere, dtype=np.float64), faces)
    mask = np.ones(len(faces), dtype=bool)
    oriented, rotation, info = canonical_orient_sphere(
        sphere,
        faces,
        values,
        weights,
        mask,
        lmax_orientation=lmax_orientation,
    )
    return oriented, {**info, "rotation_matrix": rotation.tolist()}


def _circular_mean(theta: np.ndarray, axis: int) -> np.ndarray:
    return np.arctan2(np.mean(np.sin(theta), axis=axis), np.mean(np.cos(theta), axis=axis))
