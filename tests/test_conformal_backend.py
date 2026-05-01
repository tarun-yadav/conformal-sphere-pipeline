import importlib.util
import unittest

import numpy as np
import trimesh

from sphere_mapping_pipeline.spherical.parameterize import parameterize_sphere


class ConformalBackendTests(unittest.TestCase):
    @unittest.skipIf(importlib.util.find_spec("lapy") is None, "lapy is not installed")
    def test_lapy_conformal_backend_maps_closed_genus_zero_mesh_to_sphere(self):
        mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
        result = parameterize_sphere(
            np.asarray(mesh.vertices, dtype=float),
            np.asarray(mesh.faces, dtype=np.int64),
            method="conformal",
        )

        self.assertEqual(result.info["method"], "conformal")
        self.assertEqual(result.info["backend"], "lapy")
        self.assertEqual(result.info["euler_number"], 2)
        self.assertLess(np.max(np.abs(np.linalg.norm(result.sphere, axis=1) - 1.0)), 1e-10)
        self.assertGreater(result.info["orientation_positive_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
