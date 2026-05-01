import unittest

import numpy as np

from sphere_mapping_pipeline.validation.synthetic_cases import (
    build_candy_cane_case,
    cap_open_tube_with_material_ids,
)


class SyntheticCaseTests(unittest.TestCase):
    def test_candy_cane_variants_share_material_vertices(self):
        case = build_candy_cane_case(n_s=18, n_theta=10)
        self.assertEqual(case.reference.vertices.shape, case.uniform_2x.vertices.shape)
        self.assertEqual(case.reference.vertices.shape, case.local_bulge.vertices.shape)
        self.assertTrue(np.array_equal(case.reference.faces, case.uniform_2x.faces))
        self.assertTrue(np.array_equal(case.reference.faces, case.local_bulge.faces))
        self.assertEqual(case.reference.faces.shape[1], 3)
        self.assertEqual(case.material_s.shape, (180,))
        self.assertEqual(case.material_theta.shape, (180,))

    def test_candy_cane_centerline_is_continuous(self):
        n_s = 48
        n_theta = 12
        case = build_candy_cane_case(n_s=n_s, n_theta=n_theta)
        ring_centers = case.reference.vertices.reshape(n_s, n_theta, 3).mean(axis=1)
        segment_lengths = np.linalg.norm(np.diff(ring_centers, axis=0), axis=1)

        self.assertLess(float(segment_lengths.max()), 0.16)
        self.assertGreater(float(ring_centers[:, 0].max()), 1.2)
        self.assertGreater(float(ring_centers[:, 2].max()), 2.7)

    def test_capping_preserves_physical_vertex_prefix(self):
        case = build_candy_cane_case(n_s=12, n_theta=8)
        capped = cap_open_tube_with_material_ids(case.reference)
        self.assertEqual(capped.original_vertex_count, len(case.reference.vertices))
        self.assertTrue(np.allclose(capped.vertices[: capped.original_vertex_count], case.reference.vertices))
        self.assertGreater(len(capped.vertices), capped.original_vertex_count)
        self.assertGreater(len(capped.faces), len(case.reference.faces))
        self.assertEqual(capped.faces.shape[1], 3)


if __name__ == "__main__":
    unittest.main()
