import unittest

import numpy as np
import trimesh

from conformal_sphere_pipeline.spherical.mobius import apply_mobius_center_shift, mobius_center
from conformal_sphere_pipeline.spherical.quality import face_areas


class MobiusCenterTests(unittest.TestCase):
    def test_baden_centering_recenters_shifted_weighted_sphere(self):
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        weights = face_areas(vertices, faces)
        centers = vertices[faces].mean(axis=1)
        weights *= 1.0 + 0.35 * np.maximum(centers[:, 2], 0.0)

        shifted = apply_mobius_center_shift(vertices, np.array([0.32, -0.18, 0.21]))
        centered, info = mobius_center(
            shifted,
            faces,
            weights,
            tol=1e-8,
            max_iters=100,
            max_center_norm=0.85,
        )

        self.assertLess(info["final_centroid_norm"], 1e-6)
        self.assertTrue(info["converged"])
        self.assertLess(np.max(np.abs(np.linalg.norm(centered, axis=1) - 1.0)), 1e-10)
        self.assertTrue(np.isfinite(centered).all())


if __name__ == "__main__":
    unittest.main()
