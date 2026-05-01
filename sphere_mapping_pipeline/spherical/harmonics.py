"""Direct spherical-harmonic quadrature for irregular spherical meshes."""

from __future__ import annotations

import numpy as np
try:
    from scipy.special import sph_harm_y as _sph_harm_y
except ImportError:  # pragma: no cover - older SciPy fallback
    _sph_harm_y = None
    from scipy.special import sph_harm as _sph_harm_legacy

from .quality import normalize_rows, spherical_face_centers


def _complex_sph_harm(m: int, ell: int, phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    if _sph_harm_y is not None:
        return _sph_harm_y(ell, m, theta, phi)
    return _sph_harm_legacy(m, ell, phi, theta)


def directions_to_angles(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert unit vectors to SciPy spherical angles ``theta, phi``."""

    d = normalize_rows(directions)
    theta = np.arccos(np.clip(d[:, 2], -1.0, 1.0))
    phi = np.mod(np.arctan2(d[:, 1], d[:, 0]), 2.0 * np.pi)
    return theta, phi


def compute_coeffs_on_directions(
    directions: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    lmax: int,
) -> dict[tuple[int, int], complex]:
    """Compute complex SH coefficients from weighted directional samples."""

    directions = normalize_rows(directions)
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape != (len(directions),):
        raise ValueError("values must have one entry per direction")
    if weights.shape != (len(directions),):
        raise ValueError("weights must have one entry per direction")
    theta, phi = directions_to_angles(directions)
    coeffs: dict[tuple[int, int], complex] = {}
    weighted_values = weights * values
    for ell in range(int(lmax) + 1):
        for m in range(-ell, ell + 1):
            y_lm = _complex_sph_harm(m, ell, phi, theta)
            coeffs[(ell, m)] = complex(np.sum(weighted_values * np.conjugate(y_lm)))
    return coeffs


def compute_complex_sh_coeffs_from_faces(
    u: np.ndarray,
    faces: np.ndarray,
    values_face: np.ndarray,
    integrate_weights: np.ndarray,
    lmax: int,
) -> dict[tuple[int, int], complex]:
    """Compute complex SH coefficients using one quadrature sample per face."""

    centers = spherical_face_centers(u, faces)
    return compute_coeffs_on_directions(centers, values_face, integrate_weights, lmax)


def reconstruct_signal_from_coeffs(
    coeffs: dict[tuple[int, int], complex],
    directions: np.ndarray,
    lmax: int,
) -> np.ndarray:
    """Evaluate an SH expansion at directions."""

    theta, phi = directions_to_angles(directions)
    out = np.zeros(len(theta), dtype=np.complex128)
    for ell in range(int(lmax) + 1):
        for m in range(-ell, ell + 1):
            out += coeffs.get((ell, m), 0.0) * _complex_sph_harm(m, ell, phi, theta)
    return out.real


def sh_band_power(coeffs: dict[tuple[int, int], complex], ell: int) -> float:
    """Return total coefficient power in one SH degree."""

    return float(sum(abs(coeffs.get((ell, m), 0.0)) ** 2 for m in range(-ell, ell + 1)))


def lowpass_face_signal(
    u: np.ndarray,
    faces: np.ndarray,
    values_face: np.ndarray,
    weights: np.ndarray,
    lmax: int,
) -> np.ndarray:
    """Low-pass a face signal by direct SH projection and reconstruction."""

    coeffs = compute_complex_sh_coeffs_from_faces(u, faces, values_face, weights, lmax)
    return reconstruct_signal_from_coeffs(coeffs, spherical_face_centers(u, faces), lmax)
