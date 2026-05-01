"""Null models for transition-map validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sphere_mapping_pipeline.validation.transition import matched_vertex_transition_metrics, unit_rows


@dataclass(frozen=True)
class TransitionNullDistribution:
    raw_median_deg: np.ndarray
    aligned_median_deg: np.ndarray
    aligned_p95_deg: np.ndarray

    def separation_z(self, observed_aligned_median_deg: float) -> float:
        null_values = np.asarray(self.aligned_median_deg, dtype=np.float64)
        if len(null_values) < 2:
            return 0.0
        spread = float(np.std(null_values, ddof=1))
        if spread <= 0.0:
            return float("inf") if observed_aligned_median_deg < float(np.mean(null_values)) else 0.0
        return float((np.mean(null_values) - observed_aligned_median_deg) / spread)


def shuffled_transition_null(
    source: np.ndarray,
    target: np.ndarray,
    *,
    n_trials: int = 128,
    seed: int = 0,
    faces: np.ndarray | None = None,
) -> TransitionNullDistribution:
    if n_trials < 1:
        raise ValueError("n_trials must be positive")
    source_unit = unit_rows(source)
    target_unit = unit_rows(target)
    if source_unit.shape != target_unit.shape:
        raise ValueError("source and target must have matching shape")

    rng = np.random.default_rng(seed)
    raw_medians: list[float] = []
    aligned_medians: list[float] = []
    aligned_p95s: list[float] = []
    for _ in range(n_trials):
        permuted_target = target_unit[rng.permutation(len(target_unit))]
        metrics = matched_vertex_transition_metrics(source_unit, permuted_target, faces=faces)
        raw_medians.append(metrics.raw_median_deg)
        aligned_medians.append(metrics.aligned_median_deg)
        aligned_p95s.append(metrics.aligned_p95_deg)
    return TransitionNullDistribution(
        raw_median_deg=np.asarray(raw_medians, dtype=np.float64),
        aligned_median_deg=np.asarray(aligned_medians, dtype=np.float64),
        aligned_p95_deg=np.asarray(aligned_p95s, dtype=np.float64),
    )
