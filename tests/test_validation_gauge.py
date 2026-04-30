import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from conformal_sphere_pipeline.validation.gauge import (
    harmonic_orient_material_sphere,
    material_face_signal,
    select_material_pole_face,
)
from conformal_sphere_pipeline.validation.runner import SphericalValidationRunConfig, run_spherical_validation_map
from conformal_sphere_pipeline.validation.synthetic_cases import (
    MaterialMesh,
    build_candy_cane_case,
    cap_open_tube_with_material_ids,
)
from conformal_sphere_pipeline.validation.transition import matched_vertex_transition_metrics


class ValidationGaugeTests(unittest.TestCase):
    def test_material_pole_face_is_topology_stable_for_known_correspondence_case(self):
        case = build_candy_cane_case(n_s=10, n_theta=8)

        reference_anchor = select_material_pole_face(case.reference, target_s=0.5, target_theta=0.0)
        uniform_anchor = select_material_pole_face(case.uniform_2x, target_s=0.5, target_theta=0.0)
        bulge_anchor = select_material_pole_face(case.local_bulge, target_s=0.5, target_theta=0.0)

        self.assertEqual(reference_anchor, uniform_anchor)
        self.assertEqual(reference_anchor, bulge_anchor)

    def test_harmonic_material_orientation_removes_global_rotation_for_nondegenerate_signal(self):
        case = build_candy_cane_case(n_s=12, n_theta=10)
        rotation = Rotation.from_euler("zyx", [33.0, -18.0, 11.0], degrees=True).as_matrix()
        rotated = MaterialMesh(
            vertices=case.reference.vertices @ rotation.T,
            faces=case.reference.faces,
            material_s=case.reference.material_s,
            material_theta=case.reference.material_theta,
        )

        signal = material_face_signal(case.reference)
        source_oriented, source_info = harmonic_orient_material_sphere(
            case.reference.vertices,
            case.reference,
            signal,
            lmax_orientation=6,
        )
        target_oriented, target_info = harmonic_orient_material_sphere(
            rotated.vertices,
            rotated,
            signal,
            lmax_orientation=6,
        )
        metrics = matched_vertex_transition_metrics(source_oriented, target_oriented, faces=case.reference.faces)

        self.assertLess(metrics.raw_median_deg, 1e-5)
        self.assertGreater(source_info["band2_norm"], 1e-8)
        self.assertGreater(target_info["band2_norm"], 1e-8)

    def test_material_anchor_makes_stereographic_rigid_and_scale_gauge_stable(self):
        case = build_candy_cane_case(n_s=8, n_theta=6)
        rotation = Rotation.from_euler("zyx", [33.0, -18.0, 11.0], degrees=True).as_matrix()
        rotated = MaterialMesh(
            vertices=case.reference.vertices @ rotation.T,
            faces=case.reference.faces,
            material_s=case.reference.material_s,
            material_theta=case.reference.material_theta,
        )
        scaled = MaterialMesh(
            vertices=case.reference.vertices * 1.35,
            faces=case.reference.faces,
            material_s=case.reference.material_s,
            material_theta=case.reference.material_theta,
        )
        anchor = select_material_pole_face(case.reference, target_s=0.5, target_theta=0.0)
        config = SphericalValidationRunConfig(
            pole_face_index=anchor,
            pole_area_threshold=5e-2,
            stage_a_max_iter=4,
            stage_b_max_iter=0,
            diagnostics_samples_per_face=4,
            injectivity_max_depth=4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            reference_capped = cap_open_tube_with_material_ids(case.reference)
            rotated_capped = cap_open_tube_with_material_ids(rotated)
            scaled_capped = cap_open_tube_with_material_ids(scaled)
            reference_map = run_spherical_validation_map(
                "reference",
                reference_capped.vertices,
                reference_capped.faces,
                cache_dir,
                config=config,
            )
            rotated_map = run_spherical_validation_map(
                "rotated",
                rotated_capped.vertices,
                rotated_capped.faces,
                cache_dir,
                config=config,
            )
            scaled_map = run_spherical_validation_map(
                "scaled",
                scaled_capped.vertices,
                scaled_capped.faces,
                cache_dir,
                config=config,
            )

            n = reference_capped.original_vertex_count
            rigid_metrics = matched_vertex_transition_metrics(
                reference_map.sphere[:n],
                rotated_map.sphere[:n],
                faces=case.reference.faces,
            )
            scale_metrics = matched_vertex_transition_metrics(
                reference_map.sphere[:n],
                scaled_map.sphere[:n],
                faces=case.reference.faces,
            )

        self.assertEqual(reference_map.info["input_pole_face_index"], anchor)
        self.assertEqual(rotated_map.info["input_pole_face_index"], anchor)
        self.assertEqual(scaled_map.info["input_pole_face_index"], anchor)
        self.assertLess(rigid_metrics.aligned_max_deg, 1e-5)
        self.assertLess(scale_metrics.aligned_max_deg, 1e-5)


if __name__ == "__main__":
    unittest.main()
