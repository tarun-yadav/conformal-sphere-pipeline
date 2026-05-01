import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from sphere_mapping_pipeline.validation.transition import (
    angular_degrees,
    matched_vertex_transition_metrics,
    procrustes_rotation,
    unit_rows,
)


class TransitionMetricTests(unittest.TestCase):
    def test_angular_degrees_reports_known_separation(self):
        source = np.array([[1.0, 0.0, 0.0]])
        target = np.array([[0.0, 1.0, 0.0]])
        self.assertAlmostEqual(float(angular_degrees(source, target)[0]), 90.0, places=12)

    def test_procrustes_rotation_removes_global_so3(self):
        rng = np.random.default_rng(42)
        source = unit_rows(rng.normal(size=(80, 3)))
        rotation = Rotation.from_euler("z", 47.0, degrees=True).as_matrix()
        target = source @ rotation.T

        recovered = procrustes_rotation(target, source)
        self.assertTrue(np.allclose(recovered @ target.T, source.T, atol=1e-12))

    def test_matched_transition_metrics_are_zero_for_identity(self):
        rng = np.random.default_rng(7)
        sphere = unit_rows(rng.normal(size=(120, 3)))
        metrics = matched_vertex_transition_metrics(sphere, sphere)

        self.assertLess(metrics.raw_median_deg, 1e-12)
        self.assertLess(metrics.aligned_median_deg, 1e-6)
        self.assertLess(metrics.aligned_p95_deg, 1e-6)
        self.assertLess(metrics.smoothness_p95, 1e-12)

    def test_smoothness_detects_nonuniform_transition(self):
        source = unit_rows(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.9, 0.1, 0.0],
                    [0.8, 0.2, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
        )
        target = source.copy()
        target[3] = np.array([0.0, 1.0, 0.0])
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        metrics = matched_vertex_transition_metrics(source, target, faces=faces)

        self.assertGreater(metrics.smoothness_p95, 0.1)


if __name__ == "__main__":
    unittest.main()
