import tempfile
import unittest
from pathlib import Path

import numpy as np

from conformal_sphere_pipeline.validation.figures import write_transition_panel
from conformal_sphere_pipeline.validation.transition import (
    atlas_cell_pullback_metrics,
    matched_vertex_transition_metrics,
)


class ValidationFigureTests(unittest.TestCase):
    def test_transition_panel_writes_png(self):
        source_phys = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        target_phys = source_phys + np.array([0.1, 0.0, 0.0])
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int64)
        source_sphere = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0]], dtype=float)
        target_sphere = source_sphere.copy()
        metrics = matched_vertex_transition_metrics(source_sphere, target_sphere)
        pullback = atlas_cell_pullback_metrics(source_sphere, target_sphere, n_z=2, n_lon=4)

        with tempfile.TemporaryDirectory() as tmp:
            path = write_transition_panel(
                Path(tmp),
                "toy",
                source_phys,
                target_phys,
                source_sphere,
                target_sphere,
                metrics,
                source_faces=faces,
                target_faces=faces,
                pullback_metrics=pullback,
            )
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
