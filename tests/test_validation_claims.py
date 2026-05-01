import unittest

from sphere_mapping_pipeline.validation.claims import (
    AtlasValidationMetrics,
    ValidationThresholds,
    classify_claim_level,
)


class ValidationClaimTests(unittest.TestCase):
    def test_bijective_parameterization_does_not_imply_common_gauge_or_atlas(self):
        metrics = AtlasValidationMetrics(
            max_unit_norm_error=1e-15,
            uncertified_patch_count=0,
            min_certified_det_jacobian=1e-8,
            sphere_z_range=1.75,
            exact_repeat_median_deg=0.0,
            exact_repeat_max_deg=1e-7,
            rigid_rotation_median_deg=4.0,
            global_scale_median_deg=4.0,
            transition_aligned_median_deg=18.0,
            transition_aligned_p95_deg=42.0,
            transition_smoothness_p95=0.25,
            cell_pullback_jaccard_median=0.62,
            null_separation_z=1.0,
        )
        result = classify_claim_level(metrics, ValidationThresholds())
        self.assertTrue(result.functional_pass)
        self.assertFalse(result.common_frame_pass)
        self.assertFalse(result.atlas_pass)
        self.assertEqual(result.claim_level, "functional_parameterization")
        self.assertIn("rigid_rotation", result.failed_gates)
        self.assertIn("global_scale", result.failed_gates)

    def test_anchored_rigid_and_scale_gauge_stability_promotes_common_frame_only(self):
        metrics = AtlasValidationMetrics(
            max_unit_norm_error=1e-15,
            uncertified_patch_count=0,
            min_certified_det_jacobian=1e-8,
            sphere_z_range=1.75,
            exact_repeat_median_deg=0.0,
            exact_repeat_max_deg=1e-7,
            rigid_rotation_median_deg=0.0,
            global_scale_median_deg=0.0,
            transition_aligned_median_deg=18.0,
            transition_aligned_p95_deg=42.0,
            transition_smoothness_p95=0.25,
            cell_pullback_jaccard_median=0.62,
            null_separation_z=1.0,
        )
        result = classify_claim_level(metrics, ValidationThresholds())
        self.assertTrue(result.functional_pass)
        self.assertTrue(result.common_frame_pass)
        self.assertFalse(result.atlas_pass)
        self.assertEqual(result.claim_level, "common_measurement_frame")

    def test_deformation_atlas_stability_requires_transition_pullback_and_null_gates(self):
        metrics = AtlasValidationMetrics(
            max_unit_norm_error=1e-15,
            uncertified_patch_count=0,
            min_certified_det_jacobian=1e-6,
            sphere_z_range=1.9,
            exact_repeat_median_deg=0.0,
            exact_repeat_max_deg=1e-7,
            rigid_rotation_median_deg=0.0,
            global_scale_median_deg=0.0,
            transition_aligned_median_deg=3.0,
            transition_aligned_p95_deg=8.0,
            transition_smoothness_p95=0.04,
            cell_pullback_jaccard_median=0.88,
            null_separation_z=4.0,
        )
        result = classify_claim_level(metrics, ValidationThresholds())
        self.assertTrue(result.functional_pass)
        self.assertTrue(result.common_frame_pass)
        self.assertTrue(result.atlas_pass)
        self.assertEqual(result.claim_level, "atlas_substrate")

    def test_small_transition_residual_alone_is_not_an_atlas_claim(self):
        metrics = AtlasValidationMetrics(
            max_unit_norm_error=1e-15,
            uncertified_patch_count=0,
            min_certified_det_jacobian=1e-6,
            sphere_z_range=1.9,
            exact_repeat_median_deg=0.0,
            exact_repeat_max_deg=1e-7,
            rigid_rotation_median_deg=0.0,
            global_scale_median_deg=0.0,
            transition_aligned_median_deg=3.0,
            transition_aligned_p95_deg=8.0,
            transition_smoothness_p95=0.04,
            cell_pullback_jaccard_median=0.62,
            null_separation_z=1.0,
        )
        result = classify_claim_level(metrics, ValidationThresholds())
        self.assertTrue(result.functional_pass)
        self.assertTrue(result.common_frame_pass)
        self.assertFalse(result.atlas_pass)
        self.assertEqual(result.claim_level, "common_measurement_frame")
        self.assertEqual(result.failed_gates, ("cell_pullback_overlap", "null_separation"))


if __name__ == "__main__":
    unittest.main()
