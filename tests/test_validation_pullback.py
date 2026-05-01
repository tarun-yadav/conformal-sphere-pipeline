import unittest

import numpy as np

from sphere_mapping_pipeline.validation.transition import atlas_cell_pullback_metrics, unit_rows


class AtlasCellPullbackTests(unittest.TestCase):
    def test_identity_pullback_has_unit_overlap(self):
        rng = np.random.default_rng(11)
        sphere = unit_rows(rng.normal(size=(240, 3)))
        metrics = atlas_cell_pullback_metrics(sphere, sphere, n_z=4, n_lon=8)

        self.assertAlmostEqual(metrics.median_jaccard, 1.0, places=12)
        self.assertAlmostEqual(metrics.p10_jaccard, 1.0, places=12)
        self.assertGreater(metrics.occupied_cell_count, 8)

    def test_random_correspondence_has_low_pullback_overlap(self):
        rng = np.random.default_rng(12)
        source = unit_rows(rng.normal(size=(300, 3)))
        target = source[rng.permutation(len(source))]
        metrics = atlas_cell_pullback_metrics(source, target, n_z=5, n_lon=10, align=False)

        self.assertLess(metrics.median_jaccard, 0.5)


if __name__ == "__main__":
    unittest.main()
