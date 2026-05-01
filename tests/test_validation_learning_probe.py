import unittest

import numpy as np

from sphere_mapping_pipeline.validation.learning_probe import (
    evaluate_material_decoder,
    fit_material_decoder,
)
from sphere_mapping_pipeline.validation.transition import unit_rows


class LearningProbeTests(unittest.TestCase):
    def test_decoder_recovers_linear_material_signal(self):
        rng = np.random.default_rng(21)
        sphere = unit_rows(rng.normal(size=(300, 3)))
        material_s = 0.5 * (sphere[:, 2] + 1.0)
        material_theta = np.mod(np.arctan2(sphere[:, 1], sphere[:, 0]), 2.0 * np.pi)

        decoder = fit_material_decoder(sphere, material_s, material_theta, ridge=1e-10)
        metrics = evaluate_material_decoder(decoder, sphere, material_s, material_theta)

        self.assertLess(metrics.s_rmse, 1e-8)
        self.assertLess(metrics.theta_mean_abs_deg, 1e-8)

    def test_decoder_reports_transfer_error(self):
        rng = np.random.default_rng(22)
        sphere = unit_rows(rng.normal(size=(200, 3)))
        material_s = 0.5 * (sphere[:, 2] + 1.0)
        material_theta = np.mod(np.arctan2(sphere[:, 1], sphere[:, 0]), 2.0 * np.pi)
        shifted = sphere[:, [1, 0, 2]]

        decoder = fit_material_decoder(sphere, material_s, material_theta)
        metrics = evaluate_material_decoder(decoder, shifted, material_s, material_theta)

        self.assertGreater(metrics.theta_mean_abs_deg, 5.0)


if __name__ == "__main__":
    unittest.main()
