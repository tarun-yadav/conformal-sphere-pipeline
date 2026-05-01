import unittest

import numpy as np
import trimesh


class StereographicMathTests(unittest.TestCase):
    def test_inverse_stereographic_has_unit_norm_and_metric_scale(self):
        from sphere_mapping_pipeline.spherical.stereographic import (
            inverse_stereographic,
            inverse_stereographic_jacobian,
        )

        z = np.array([[0.0, 0.0], [0.5, -0.25], [3.0, 4.0]], dtype=float)
        sphere = inverse_stereographic(z, pole_sign=1)
        self.assertLess(np.max(np.abs(np.linalg.norm(sphere, axis=1) - 1.0)), 1e-12)

        jac = inverse_stereographic_jacobian(np.array([0.5, -0.25]), pole_sign=1)
        metric = jac.T @ jac
        scale = 4.0 / (1.0 + 0.5 * 0.5 + 0.25 * 0.25) ** 2
        self.assertTrue(np.allclose(metric, scale * np.eye(2), atol=1e-12))

    def test_lyness_jespersen_rule_10_weights_are_normalized(self):
        from sphere_mapping_pipeline.spherical.stereographic import lyness_jespersen_rule_10

        points, weights = lyness_jespersen_rule_10()
        self.assertEqual(points.shape, (12, 2))
        self.assertEqual(weights.shape, (12,))
        self.assertTrue(np.isclose(weights.sum(), 1.0))
        self.assertTrue(np.all(points >= -1e-15))
        self.assertTrue(np.all(points.sum(axis=1) <= 1.0 + 1e-15))


class QuadraticBezierPatchTests(unittest.TestCase):
    def test_quadratic_bezier_affine_controls_reproduce_linear_triangle(self):
        from sphere_mapping_pipeline.spherical.stereographic import QuadraticBezierPatch

        z0 = np.array([0.0, 0.0])
        z1 = np.array([2.0, 0.0])
        z2 = np.array([0.0, 3.0])
        patch = QuadraticBezierPatch.from_affine_triangle(z0, z1, z2)
        value, jac = patch.evaluate_with_jacobian(np.array([0.25, 0.5]))
        self.assertTrue(np.allclose(value, np.array([0.5, 1.5])))
        self.assertTrue(np.allclose(jac, np.array([[2.0, 0.0], [0.0, 3.0]])))
        self.assertGreater(patch.min_sampled_det_jacobian(), 0.0)


class StereographicInitializerTests(unittest.TestCase):
    def test_stereographic_initializer_builds_positive_disk_for_icosphere(self):
        from sphere_mapping_pipeline.spherical.stereographic import initialize_stereographic_disk

        mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
        init = initialize_stereographic_disk(
            np.asarray(mesh.vertices, dtype=float),
            np.asarray(mesh.faces, dtype=np.int64),
        )
        self.assertAlmostEqual(init.scaled_area, 4.0 * np.pi, places=12)
        self.assertGreaterEqual(init.pole_face_index, 0)
        self.assertEqual(len(init.disk_faces), len(mesh.faces) - 1)
        self.assertGreater(init.min_planar_double_area, 1e-12)
        self.assertEqual(init.boundary_vertices.shape, (3,))

    def test_stereographic_initializer_accepts_explicit_pole_face(self):
        from sphere_mapping_pipeline.spherical.stereographic import initialize_stereographic_disk

        mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
        init = initialize_stereographic_disk(
            np.asarray(mesh.vertices, dtype=float),
            np.asarray(mesh.faces, dtype=np.int64),
            pole_face_index=7,
        )

        self.assertEqual(init.pole_face_index, 7)
        self.assertEqual(init.input_pole_face_index, 7)
        self.assertEqual(init.pole_selection, "explicit_face_index")
        self.assertTrue(np.array_equal(init.pole_face, np.asarray(mesh.faces, dtype=np.int64)[7]))

    def test_stereographic_initializer_rejects_invalid_explicit_pole_face(self):
        from sphere_mapping_pipeline.spherical.stereographic import initialize_stereographic_disk

        mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
        with self.assertRaisesRegex(ValueError, "pole_face_index"):
            initialize_stereographic_disk(
                np.asarray(mesh.vertices, dtype=float),
                np.asarray(mesh.faces, dtype=np.int64),
                pole_face_index=len(mesh.faces),
            )


class StereographicBackendTests(unittest.TestCase):
    def test_public_api_exposes_stereographic_backend(self):
        from sphere_mapping_pipeline import StereographicConfig, parameterize_sphere

        mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
        result = parameterize_sphere(
            np.asarray(mesh.vertices, dtype=float),
            np.asarray(mesh.faces, dtype=np.int64),
            method="stereographic",
            stereographic_config=StereographicConfig(
                pole_area_threshold=5e-2,
                stage_a_max_iter=2,
                stage_b_max_iter=0,
                diagnostics_samples_per_face=4,
            ),
        )

        self.assertEqual(result.info["method"], "stereographic")
        self.assertLess(np.max(np.abs(np.linalg.norm(result.sphere, axis=1) - 1.0)), 1e-12)

    def test_stereographic_backend_returns_optimized_non_collapsed_sphere(self):
        from sphere_mapping_pipeline.spherical.stereographic import StereographicConfig
        from sphere_mapping_pipeline.spherical.parameterize import parameterize_sphere

        mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
        result = parameterize_sphere(
            np.asarray(mesh.vertices, dtype=float),
            np.asarray(mesh.faces, dtype=np.int64),
            method="stereographic",
            stereographic_config=StereographicConfig(
                pole_area_threshold=5e-2,
                stage_a_max_iter=8,
                stage_b_max_iter=4,
                diagnostics_samples_per_face=8,
            ),
        )
        self.assertEqual(result.info["method"], "stereographic")
        self.assertEqual(result.info["backend"], "stereographic_bezier")
        self.assertIn(result.info["optimization_status"], {"converged", "max_iter", "line_search_failed"})
        self.assertNotEqual(result.info["optimization_status"], "not_implemented")
        self.assertLess(np.max(np.abs(np.linalg.norm(result.sphere, axis=1) - 1.0)), 1e-12)
        self.assertGreater(result.info["min_certified_det_jacobian"], 1e-12)
        self.assertLess(result.info["south_polar_cap_fraction_z_lt_neg_0_95"], 0.95)
        self.assertGreater(result.info["sphere_z_range"], 1.25)
        self.assertGreater(result.info["max_sigma_iso"], 0.0)

    def test_stereographic_config_accepts_explicit_pole_face(self):
        from sphere_mapping_pipeline.spherical.stereographic import (
            StereographicConfig,
            stereographic_spherical_parameterization,
        )

        mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
        result = stereographic_spherical_parameterization(
            np.asarray(mesh.vertices, dtype=float),
            np.asarray(mesh.faces, dtype=np.int64),
            config=StereographicConfig(
                pole_face_index=7,
                pole_area_threshold=5e-2,
                stage_a_max_iter=4,
                stage_b_max_iter=0,
                diagnostics_samples_per_face=4,
            ),
        )

        self.assertEqual(result.info["input_pole_face_index"], 7)
        self.assertEqual(result.info["pole_selection"], "explicit_face_index")
        self.assertTrue(result.success)

    def test_stereographic_full_result_evaluator_handles_pole_centroid_and_boundary(self):
        from sphere_mapping_pipeline.spherical.stereographic import (
            StereographicConfig,
            stereographic_spherical_parameterization,
            inverse_stereographic,
        )

        mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
        result = stereographic_spherical_parameterization(
            np.asarray(mesh.vertices, dtype=float),
            np.asarray(mesh.faces, dtype=np.int64),
            config=StereographicConfig(
                pole_area_threshold=5e-2,
                stage_a_max_iter=4,
                stage_b_max_iter=0,
                diagnostics_samples_per_face=4,
            ),
        )

        self.assertTrue(result.success)
        pole_value = result.evaluate(result.pole_face_id, np.array([1.0 / 3.0] * 3))
        self.assertTrue(np.allclose(pole_value, np.array([0.0, 0.0, 1.0]), atol=1e-12))

        boundary_vertex = int(result.pole_face[0])
        pole_bary = np.zeros(3)
        pole_bary[0] = 1.0
        boundary_from_pole = result.evaluate(result.pole_face_id, pole_bary)
        boundary_from_vertex = inverse_stereographic(
            result.vertex_controls[boundary_vertex][None, :],
            pole_sign=1,
        )[0]
        self.assertTrue(np.allclose(boundary_from_pole, boundary_from_vertex, atol=1e-10))


if __name__ == "__main__":
    unittest.main()
