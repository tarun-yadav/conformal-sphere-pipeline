"""Claim ladder for spherical atlas-substrate validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationThresholds:
    max_unit_norm_error: float = 1e-10
    min_certified_det_jacobian: float = 1e-12
    min_sphere_z_range: float = 1.25
    exact_repeat_max_deg: float = 1e-5
    rigid_rotation_median_deg: float = 1e-5
    global_scale_median_deg: float = 1e-5
    transition_aligned_median_deg: float = 5.0
    transition_aligned_p95_deg: float = 15.0
    transition_smoothness_p95: float = 0.08
    cell_pullback_jaccard_median: float = 0.80
    null_separation_z: float = 3.0


@dataclass(frozen=True)
class AtlasValidationMetrics:
    max_unit_norm_error: float
    uncertified_patch_count: int
    min_certified_det_jacobian: float
    sphere_z_range: float
    exact_repeat_median_deg: float
    exact_repeat_max_deg: float
    rigid_rotation_median_deg: float
    global_scale_median_deg: float
    transition_aligned_median_deg: float
    transition_aligned_p95_deg: float
    transition_smoothness_p95: float
    cell_pullback_jaccard_median: float
    null_separation_z: float


@dataclass(frozen=True)
class ClaimClassification:
    functional_pass: bool
    common_frame_pass: bool
    atlas_pass: bool
    claim_level: str
    failed_gates: tuple[str, ...]


def classify_claim_level(
    metrics: AtlasValidationMetrics,
    thresholds: ValidationThresholds,
) -> ClaimClassification:
    functional_checks = {
        "unit_norm": metrics.max_unit_norm_error <= thresholds.max_unit_norm_error,
        "injectivity_certified": metrics.uncertified_patch_count == 0,
        "positive_det": metrics.min_certified_det_jacobian >= thresholds.min_certified_det_jacobian,
        "noncollapsed_coverage": metrics.sphere_z_range >= thresholds.min_sphere_z_range,
        "exact_repeat": metrics.exact_repeat_max_deg <= thresholds.exact_repeat_max_deg,
    }
    common_checks = {
        "rigid_rotation": metrics.rigid_rotation_median_deg <= thresholds.rigid_rotation_median_deg,
        "global_scale": metrics.global_scale_median_deg <= thresholds.global_scale_median_deg,
    }
    atlas_checks = {
        "transition_median": metrics.transition_aligned_median_deg <= thresholds.transition_aligned_median_deg,
        "transition_p95": metrics.transition_aligned_p95_deg <= thresholds.transition_aligned_p95_deg,
        "transition_smoothness": metrics.transition_smoothness_p95 <= thresholds.transition_smoothness_p95,
        "cell_pullback_overlap": metrics.cell_pullback_jaccard_median >= thresholds.cell_pullback_jaccard_median,
        "null_separation": metrics.null_separation_z >= thresholds.null_separation_z,
    }
    all_checks = {**functional_checks, **common_checks, **atlas_checks}
    failed = tuple(name for name, ok in all_checks.items() if not ok)
    functional_pass = all(functional_checks.values())
    common_frame_pass = functional_pass and all(common_checks.values())
    atlas_pass = common_frame_pass and all(atlas_checks.values())
    if atlas_pass:
        level = "atlas_substrate"
    elif common_frame_pass:
        level = "common_measurement_frame"
    elif functional_pass:
        level = "functional_parameterization"
    else:
        level = "failed_functional_gate"
    return ClaimClassification(
        functional_pass=functional_pass,
        common_frame_pass=common_frame_pass,
        atlas_pass=atlas_pass,
        claim_level=level,
        failed_gates=failed,
    )
