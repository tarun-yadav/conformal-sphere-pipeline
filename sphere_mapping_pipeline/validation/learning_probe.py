"""Small dependency-free learning probes for atlas substrate validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sphere_mapping_pipeline.validation.transition import unit_rows


@dataclass(frozen=True)
class MaterialDecoder:
    coefficients: np.ndarray
    ridge: float


@dataclass(frozen=True)
class MaterialDecoderMetrics:
    s_rmse: float
    s_mae: float
    theta_mean_abs_deg: float
    theta_p95_abs_deg: float


def fit_material_decoder(
    sphere: np.ndarray,
    material_s: np.ndarray,
    material_theta: np.ndarray,
    *,
    ridge: float = 1e-8,
) -> MaterialDecoder:
    if ridge < 0.0:
        raise ValueError("ridge must be nonnegative")
    features = spherical_decoder_features(sphere)
    s = np.asarray(material_s, dtype=np.float64)
    theta = np.asarray(material_theta, dtype=np.float64)
    if s.shape != (len(features),) or theta.shape != (len(features),):
        raise ValueError("material_s and material_theta must match vertex count")
    targets = np.column_stack([s, np.sin(theta), np.cos(theta)])
    lhs = features.T @ features
    if ridge > 0.0:
        lhs = lhs + ridge * np.eye(lhs.shape[0], dtype=np.float64)
    coefficients = np.linalg.solve(lhs, features.T @ targets)
    return MaterialDecoder(coefficients=coefficients, ridge=float(ridge))


def evaluate_material_decoder(
    decoder: MaterialDecoder,
    sphere: np.ndarray,
    material_s: np.ndarray,
    material_theta: np.ndarray,
) -> MaterialDecoderMetrics:
    features = spherical_decoder_features(sphere)
    predictions = features @ decoder.coefficients
    true_s = np.asarray(material_s, dtype=np.float64)
    true_theta = np.asarray(material_theta, dtype=np.float64)
    if true_s.shape != (len(features),) or true_theta.shape != (len(features),):
        raise ValueError("material_s and material_theta must match vertex count")
    pred_s = predictions[:, 0]
    pred_theta = np.arctan2(predictions[:, 1], predictions[:, 2])
    s_error = pred_s - true_s
    theta_error = _circular_error(pred_theta, true_theta)
    theta_abs_deg = np.degrees(np.abs(theta_error))
    return MaterialDecoderMetrics(
        s_rmse=float(np.sqrt(np.mean(s_error * s_error))),
        s_mae=float(np.mean(np.abs(s_error))),
        theta_mean_abs_deg=float(np.mean(theta_abs_deg)),
        theta_p95_abs_deg=float(np.percentile(theta_abs_deg, 95.0)),
    )


def spherical_decoder_features(sphere: np.ndarray) -> np.ndarray:
    points = unit_rows(sphere)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    rho = np.sqrt(x * x + y * y)
    safe_rho = np.where(rho > 1e-12, rho, 1.0)
    cos_theta = x / safe_rho
    sin_theta = y / safe_rho
    return np.column_stack(
        [
            np.ones(len(points), dtype=np.float64),
            x,
            y,
            z,
            x * x,
            y * y,
            z * z,
            x * y,
            x * z,
            y * z,
            cos_theta,
            sin_theta,
        ]
    )


def _circular_error(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta = np.asarray(predicted, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return np.arctan2(np.sin(delta), np.cos(delta))
