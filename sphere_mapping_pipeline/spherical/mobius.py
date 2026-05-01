"""Möbius centering for spherical conformal maps."""

from __future__ import annotations

import numpy as np

from .quality import normalize_rows, spherical_face_centers


def apply_mobius_center_shift(u: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Apply the Baden-Crane-Kazhdan center-shift Möbius transform."""

    u = normalize_rows(u)
    c = np.asarray(c, dtype=np.float64).reshape(3)
    c2 = float(np.dot(c, c))
    denom = np.sum((u + c) * (u + c), axis=1, keepdims=True)
    denom = np.where(denom > 1e-15, denom, 1e-15)
    out = (1.0 - c2) * (u + c) / denom + c
    return normalize_rows(out)


def _weighted_centroid(u: np.ndarray, faces: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centers = spherical_face_centers(u, faces)
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    total = float(weights.sum())
    if total <= 1e-30:
        raise ValueError("face_weights must contain positive finite mass")
    weights = weights / total
    return np.sum(weights[:, None] * centers, axis=0), centers


def mobius_center(
    u: np.ndarray,
    faces: np.ndarray,
    face_weights: np.ndarray,
    *,
    tol: float = 1e-7,
    max_iters: int = 100,
    max_center_norm: float = 0.95,
    line_search: bool = True,
) -> tuple[np.ndarray, dict]:
    """Center conformal area distortion by iterated Möbius transformations."""

    current = normalize_rows(u)
    faces = np.asarray(faces, dtype=np.int64)
    weights = np.asarray(face_weights, dtype=np.float64)
    if weights.shape != (len(faces),):
        raise ValueError(f"face_weights must have shape ({len(faces)},)")
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if float(weights.sum()) <= 1e-30:
        raise ValueError("face_weights must contain positive finite mass")
    weights = weights / float(weights.sum())

    initial_mu, centers = _weighted_centroid(current, faces, weights)
    initial_norm = float(np.linalg.norm(initial_mu))
    cumulative: list[list[float]] = []
    converged = initial_norm < tol
    last_norm = initial_norm

    for iteration in range(int(max_iters)):
        mu, centers = _weighted_centroid(current, faces, weights)
        mu_norm = float(np.linalg.norm(mu))
        last_norm = mu_norm
        if mu_norm < tol:
            converged = True
            break

        jac = np.zeros((3, 3), dtype=np.float64)
        eye = np.eye(3)
        for weight, center in zip(weights, centers):
            jac += 2.0 * weight * (eye - np.outer(center, center))
        try:
            step = -np.linalg.solve(jac + 1e-12 * eye, mu)
        except np.linalg.LinAlgError:
            step = -np.linalg.lstsq(jac + 1e-10 * eye, mu, rcond=None)[0]

        step_norm = float(np.linalg.norm(step))
        if step_norm >= max_center_norm:
            step = step / step_norm * float(max_center_norm)

        accepted = False
        best = current
        best_norm = mu_norm
        alpha_values = [1.0] if not line_search else [0.5 ** k for k in range(24)]
        for alpha in alpha_values:
            trial_step = alpha * step
            if float(np.linalg.norm(trial_step)) >= 1.0:
                continue
            trial = apply_mobius_center_shift(current, trial_step)
            trial_mu, _ = _weighted_centroid(trial, faces, weights)
            trial_norm = float(np.linalg.norm(trial_mu))
            if np.isfinite(trial_norm) and trial_norm < best_norm:
                best = trial
                best_norm = trial_norm
                cumulative.append(trial_step.tolist())
                accepted = True
                break

        if not accepted:
            break
        current = best
        last_norm = best_norm

    final_mu, _ = _weighted_centroid(current, faces, weights)
    final_norm = float(np.linalg.norm(final_mu))
    info = {
        "initial_centroid": initial_mu.tolist(),
        "initial_centroid_norm": initial_norm,
        "final_centroid": final_mu.tolist(),
        "final_centroid_norm": final_norm,
        "iterations": int(len(cumulative)),
        "converged": bool(final_norm < tol),
        "accepted_centers": cumulative,
        "last_iteration_centroid_norm": float(last_norm),
    }
    return current, info
