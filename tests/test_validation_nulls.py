import unittest

import numpy as np

from conformal_sphere_pipeline.validation.nulls import shuffled_transition_null
from conformal_sphere_pipeline.validation.transition import unit_rows


class NullModelTests(unittest.TestCase):
    def test_shuffled_null_is_reproducible(self):
        rng = np.random.default_rng(123)
        source = unit_rows(rng.normal(size=(100, 3)))
        target = source.copy()

        first = shuffled_transition_null(source, target, n_trials=8, seed=10)
        second = shuffled_transition_null(source, target, n_trials=8, seed=10)

        self.assertTrue(np.allclose(first.aligned_median_deg, second.aligned_median_deg))
        self.assertTrue(np.allclose(first.aligned_p95_deg, second.aligned_p95_deg))

    def test_identity_correspondence_beats_shuffled_null(self):
        rng = np.random.default_rng(456)
        source = unit_rows(rng.normal(size=(120, 3)))
        target = source.copy()
        null = shuffled_transition_null(source, target, n_trials=16, seed=2)

        self.assertGreater(float(np.median(null.aligned_median_deg)), 25.0)
        self.assertGreater(null.separation_z(observed_aligned_median_deg=0.0), 3.0)


if __name__ == "__main__":
    unittest.main()
