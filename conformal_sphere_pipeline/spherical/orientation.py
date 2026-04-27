"""Canonical SO(3) orientation from spherical-harmonic constraints."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .harmonics import compute_coeffs_on_directions, sh_band_power
from .quality import normalize_rows, spherical_face_centers


def rotation_matrix_z(angle: float) -> np.ndarray:
    """Return a right-handed rotation about the z-axis."""

    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _weighted_zscore(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    total = float(weights.sum())
    if total <= 1e-30:
        return values - float(np.nanmean(values))
    weights = weights / total
    mean = float(np.sum(weights * values))
    var = float(np.sum(weights * (values - mean) ** 2))
    std = max(np.sqrt(var), 1e-12)
    return (values - mean) / std


def _rotation_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source = source / max(float(np.linalg.norm(source)), 1e-15)
    target = target / max(float(np.linalg.norm(target)), 1e-15)
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if np.linalg.norm(cross) < 1e-12:
        if dot > 0.0:
            return np.eye(3)
        axis = np.array([1.0, 0.0, 0.0])
        if abs(source[0]) > 0.8:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - np.dot(axis, source) * source
        axis = axis / np.linalg.norm(axis)
        return Rotation.from_rotvec(np.pi * axis).as_matrix()
    skew = np.array(
        [[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]]
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - dot) / (np.linalg.norm(cross) ** 2))


def align_anchor_to_axis(
    u: np.ndarray,
    anchor_index: int,
    target_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate one anchor vertex onto ``target_axis`` without resolving yaw."""

    u = normalize_rows(u)
    rotation = _rotation_from_vectors(u[int(anchor_index)], target_axis)
    return normalize_rows((rotation @ u.T).T), rotation


def _prepare_orientation_samples(
    u: np.ndarray,
    faces: np.ndarray,
    values_face: np.ndarray,
    weights_face: np.ndarray,
    physical_face_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = spherical_face_centers(u, faces)
    mask = np.asarray(physical_face_mask, dtype=bool)
    values = np.asarray(values_face, dtype=np.float64)
    weights = np.asarray(weights_face, dtype=np.float64)
    weights = np.where(mask & np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if float(weights.sum()) <= 1e-30:
        weights = mask.astype(float)
    values_z = _weighted_zscore(values, weights)
    weights = weights / float(weights.sum())
    return centers, values_z, weights


def _band2_loss(coeffs: dict[tuple[int, int], complex]) -> tuple[float, dict]:
    eps = 1e-15
    band2 = np.sqrt(sh_band_power(coeffs, 2)) + eps
    c21 = coeffs.get((2, 1), 0.0)
    c22 = coeffs.get((2, 2), 0.0)
    c11 = coeffs.get((1, 1), 0.0)
    loss = (
        100.0 * abs(c21 / band2) ** 2
        + 25.0 * (float(np.imag(c22)) / band2) ** 2
        - float(np.real(c22)) / band2
    )
    if float(np.real(c22)) < 0.0:
        loss += 10.0
    if abs(c11) > 1e-8:
        if float(np.real(c11)) < 0.0:
            loss += 0.1
        if float(np.imag(c11)) < 0.0:
            loss += 0.1

    band3 = np.sqrt(sh_band_power(coeffs, 3)) + eps
    if band3 > 1e-12:
        loss += -0.01 * float(np.real(coeffs.get((3, 3), 0.0))) / band3
        loss += -0.005 * float(np.real(coeffs.get((3, 2), 0.0))) / band3
        loss += -0.002 * float(np.imag(coeffs.get((3, 1), 0.0))) / band3

    metrics = {
        "orientation_c21_rel": float(abs(c21) / band2),
        "orientation_c22_imag_rel": float(abs(np.imag(c22)) / band2),
        "orientation_c22_real_rel": float(np.real(c22) / band2),
        "band2_norm": float(band2),
    }
    return float(loss), metrics


def _candidate_rotations_from_q(q: np.ndarray) -> list[np.ndarray]:
    _, evecs = np.linalg.eigh(q)
    rotations: list[np.ndarray] = []
    for perm in itertools.permutations(range(3)):
        base = evecs[:, perm]
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            frame = base * np.asarray(signs)[None, :]
            if np.linalg.det(frame) < 0.0:
                continue
            rotations.append(frame.T)
    return rotations


def resolve_yaw_with_sh(
    u: np.ndarray,
    faces: np.ndarray,
    values_face: np.ndarray,
    weights_face: np.ndarray,
    physical_face_mask: np.ndarray,
    *,
    lmax: int = 4,
) -> tuple[np.ndarray, float, dict]:
    """Resolve yaw around z using the phase of the ``C_2^2`` coefficient."""

    centers, values_z, weights = _prepare_orientation_samples(
        u, faces, values_face, weights_face, physical_face_mask
    )
    coeffs = compute_coeffs_on_directions(centers, values_z, weights, lmax)
    c22 = coeffs.get((2, 2), 0.0)
    if abs(c22) < 1e-12:
        c11 = coeffs.get((1, 1), 0.0)
        psi0 = float(np.angle(c11)) if abs(c11) > 1e-12 else 0.0
    else:
        psi0 = float(np.angle(c22) / 2.0)

    best = None
    for psi in (psi0, psi0 + np.pi):
        r = rotation_matrix_z(psi)
        rotated_centers = (r @ centers.T).T
        coeffs_rot = compute_coeffs_on_directions(rotated_centers, values_z, weights, lmax)
        loss, metrics = _band2_loss(coeffs_rot)
        c11 = coeffs_rot.get((1, 1), 0.0)
        if abs(c11) > 1e-8:
            if np.real(c11) < 0.0:
                loss += 0.1
            if np.imag(c11) < 0.0:
                loss += 0.1
        if best is None or loss < best[0]:
            best = (loss, psi, r, metrics)

    assert best is not None
    _, psi, rotation, metrics = best
    result = normalize_rows((rotation @ np.asarray(u, dtype=float).T).T)
    info = {
        "yaw_radians": float(psi),
        "c22_imag_rel": metrics["orientation_c22_imag_rel"],
        "c22_real_rel": metrics["orientation_c22_real_rel"],
    }
    return result, float(psi), info


def canonical_orient_sphere(
    u: np.ndarray,
    faces: np.ndarray,
    orientation_values_face: np.ndarray,
    sphere_area_face: np.ndarray,
    physical_face_mask: np.ndarray,
    *,
    lmax_orientation: int = 8,
    refine_bands: tuple[int, ...] = (2, 3, 4, 5),
    optional_anchor: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return a canonical SO(3) orientation determined by SH constraints."""

    del refine_bands, optional_anchor
    u = normalize_rows(u)
    faces = np.asarray(faces, dtype=np.int64)
    centers, values_z, weights = _prepare_orientation_samples(
        u, faces, orientation_values_face, sphere_area_face, physical_face_mask
    )

    q = np.zeros((3, 3), dtype=np.float64)
    eye = np.eye(3)
    for weight, value, center in zip(weights, values_z, centers):
        q += weight * value * (np.outer(center, center) - eye / 3.0)

    candidates = _candidate_rotations_from_q(q)
    if not candidates:
        candidates = [np.eye(3)]

    best = None
    for rotation in candidates:
        rotated_centers = (rotation @ centers.T).T
        coeffs = compute_coeffs_on_directions(
            rotated_centers, values_z, weights, int(lmax_orientation)
        )
        loss, metrics = _band2_loss(coeffs)
        if best is None or loss < best[0]:
            best = (loss, rotation, metrics)

    assert best is not None
    init_rotation = best[1]

    def objective(rotvec: np.ndarray) -> float:
        delta = Rotation.from_rotvec(rotvec).as_matrix()
        total = delta @ init_rotation
        coeffs = compute_coeffs_on_directions(
            (total @ centers.T).T, values_z, weights, int(lmax_orientation)
        )
        loss, _ = _band2_loss(coeffs)
        return loss

    opt = minimize(
        objective,
        np.zeros(3, dtype=float),
        method="BFGS",
        options={"maxiter": 200, "gtol": 1e-8},
    )
    if np.isfinite(opt.fun):
        final_rotation = Rotation.from_rotvec(opt.x).as_matrix() @ init_rotation
    else:
        final_rotation = init_rotation

    # Recheck equivalent pi rotations around canonical axes; this catches
    # remaining sign choices while keeping the quadrupole constraints intact.
    sign_rotations = [
        np.eye(3),
        np.diag([1.0, -1.0, -1.0]),
        np.diag([-1.0, 1.0, -1.0]),
        np.diag([-1.0, -1.0, 1.0]),
    ]
    best_sign = None
    for sign_rotation in sign_rotations:
        rotation = sign_rotation @ final_rotation
        coeffs = compute_coeffs_on_directions(
            (rotation @ centers.T).T, values_z, weights, int(lmax_orientation)
        )
        loss, metrics = _band2_loss(coeffs)
        key = (
            loss,
            -metrics["orientation_c22_real_rel"],
            -float(np.real(coeffs.get((3, 3), 0.0))),
            -float(np.real(coeffs.get((3, 2), 0.0))),
        )
        if best_sign is None or key < best_sign[0]:
            best_sign = (key, rotation, metrics, coeffs)

    assert best_sign is not None
    final_rotation = best_sign[1]
    metrics = best_sign[2]
    u_canonical = normalize_rows((final_rotation @ u.T).T)
    det = float(np.linalg.det(final_rotation))
    info = {
        **metrics,
        "rotation_matrix": final_rotation.tolist(),
        "rotation_determinant": det,
        "rotation_det_error": float(abs(det - 1.0)),
        "rotation_orthogonality_error": float(
            np.linalg.norm(final_rotation.T @ final_rotation - np.eye(3))
        ),
        "optimizer_success": bool(getattr(opt, "success", False)),
        "optimizer_message": str(getattr(opt, "message", "")),
    }
    return u_canonical, final_rotation, info
