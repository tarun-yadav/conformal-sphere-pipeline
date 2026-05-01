import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from sphere_mapping_pipeline.validation.runner import (
    SphericalValidationRunConfig,
    mesh_sha256,
    run_spherical_validation_map,
)


class ValidationRunnerTests(unittest.TestCase):
    def test_mesh_hash_is_content_sensitive(self):
        mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        shifted = vertices.copy()
        shifted[0, 0] += 1e-3

        self.assertEqual(mesh_sha256(vertices, faces), mesh_sha256(vertices.copy(), faces.copy()))
        self.assertNotEqual(mesh_sha256(vertices, faces), mesh_sha256(shifted, faces))

    def test_runner_writes_and_reuses_cache(self):
        mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
        config = SphericalValidationRunConfig(
            pole_area_threshold=5e-2,
            stage_a_max_iter=4,
            stage_b_max_iter=2,
            diagnostics_samples_per_face=4,
            injectivity_max_depth=4,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            first = run_spherical_validation_map(
                "ico",
                np.asarray(mesh.vertices, dtype=np.float64),
                np.asarray(mesh.faces, dtype=np.int64),
                cache_dir,
                config=config,
            )
            second = run_spherical_validation_map(
                "ico",
                np.asarray(mesh.vertices, dtype=np.float64),
                np.asarray(mesh.faces, dtype=np.int64),
                cache_dir,
                config=config,
            )

            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            self.assertTrue(first.sphere_path.exists())
            self.assertTrue(first.manifest_path.exists())
            self.assertTrue(np.allclose(first.sphere, second.sphere))
            self.assertEqual(first.input_hash, second.input_hash)
            self.assertEqual(first.config_hash, second.config_hash)
            self.assertIn(first.info["optimization_status"], {"converged", "max_iter", "line_search_failed"})

            manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["name"], "ico")
            self.assertEqual(manifest["input_hash"], first.input_hash)
            self.assertEqual(manifest["config_hash"], first.config_hash)

    def test_runner_passes_explicit_pole_anchor_to_stereographic(self):
        mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
        config = SphericalValidationRunConfig(
            pole_face_index=7,
            pole_area_threshold=5e-2,
            stage_a_max_iter=4,
            stage_b_max_iter=0,
            diagnostics_samples_per_face=4,
            injectivity_max_depth=4,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_spherical_validation_map(
                "anchored",
                np.asarray(mesh.vertices, dtype=np.float64),
                np.asarray(mesh.faces, dtype=np.int64),
                Path(tmpdir),
                config=config,
            )

            self.assertEqual(result.info["input_pole_face_index"], 7)
            self.assertEqual(result.info["pole_selection"], "explicit_face_index")

    @unittest.skipIf(importlib.util.find_spec("lapy") is None, "lapy is not installed")
    def test_runner_can_use_conformal_backend_without_injectivity_certificate(self):
        mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
        config = SphericalValidationRunConfig(parameterization_method="conformal")

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_spherical_validation_map(
                "conformal",
                np.asarray(mesh.vertices, dtype=np.float64),
                np.asarray(mesh.faces, dtype=np.int64),
                Path(tmpdir),
                config=config,
            )

            self.assertEqual(result.info["method"], "conformal")
            self.assertEqual(result.info["backend"], "lapy")
            self.assertEqual(result.info["injectivity_certification"], "unavailable")
            self.assertGreater(result.info["uncertified_patch_count"], 0)
            self.assertIn("sphere_z_range", result.info)
            self.assertGreaterEqual(result.info["sphere_z_range"], 0.0)


if __name__ == "__main__":
    unittest.main()
