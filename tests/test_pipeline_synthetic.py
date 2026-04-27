import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import trimesh

from conformal_sphere_pipeline.cli import main
from conformal_sphere_pipeline.pipeline import ConformalSphereConfig


def make_curved_open_tube(n_sections=16, n_ring=20):
    t = np.linspace(0.0, 1.0, n_sections)
    centers = np.column_stack(
        [
            0.8 * np.sin(0.7 * np.pi * t),
            np.zeros_like(t),
            3.0 * (t - 0.5),
        ]
    )
    vertices = []
    for idx, c in enumerate(centers):
        theta = np.linspace(0.0, 2.0 * np.pi, n_ring, endpoint=False)
        radius = 0.35 + 0.08 * np.sin(np.pi * t[idx])
        vertices.append(
            np.column_stack(
                [
                    c[0] + radius * np.cos(theta),
                    c[1] + radius * np.sin(theta),
                    np.full(n_ring, c[2]),
                ]
            )
        )
    vertices = np.vstack(vertices)
    faces = []
    for section in range(n_sections - 1):
        base = section * n_ring
        nxt = (section + 1) * n_ring
        for i in range(n_ring):
            j = (i + 1) % n_ring
            faces.append([base + i, base + j, nxt + j])
            faces.append([base + i, nxt + j, nxt + i])
    return vertices.astype(float), np.asarray(faces, dtype=np.int64)


class SyntheticPipelineTests(unittest.TestCase):
    def test_config_rejects_invalid_grid_and_orientation_bandwidth(self):
        cfg = ConformalSphereConfig(nlat=1)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_default_config_resource_loads(self):
        cfg = ConformalSphereConfig.from_yaml("default")
        self.assertEqual(cfg.lmax_orientation, 16)
        cfg = ConformalSphereConfig.from_yaml("configs/canonical_sphere.yaml")
        self.assertEqual(cfg.nlat, 128)
        cfg = ConformalSphereConfig(lmax_orientation=1)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_cli_canonicalize_synthetic_open_tube_outputs_artifacts(self):
        vertices, faces = make_curved_open_tube()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mesh_path = tmp_path / "tube.stl"
            out_dir = tmp_path / "case_001"
            trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(mesh_path)

            with redirect_stdout(StringIO()):
                exit_code = main(
                    [
                        "canonicalize",
                        str(mesh_path),
                        "--out",
                        str(out_dir),
                        "--nlat",
                        "24",
                        "--nlon",
                        "48",
                        "--parameterizer",
                        "radial",
                    ]
                )

            self.assertEqual(exit_code, 0)
            expected = [
                "physical_mesh.ply",
                "closed_augmented_mesh.ply",
                "sphere_raw.ply",
                "sphere_mobius.ply",
                "sphere_canonical.ply",
                "sphere_physical_support.ply",
                "physical_mesh_display.json",
                "sphere_display.json",
                "sphere_augmented_qc.json",
                "features_grid.npz",
                "sh_coeffs.npz",
                "canonical_result.json",
                "quality_report.json",
            ]
            for name in expected:
                self.assertTrue((out_dir / name).exists(), name)

            quality = json.loads((out_dir / "quality_report.json").read_text())
            self.assertEqual(quality["schema_version"], "canonical-sphere.quality.v1")
            self.assertEqual(quality["topology"]["genus"], 0)
            self.assertEqual(quality["topology"]["boundary_edge_count"], 0)
            self.assertLess(quality["mobius"]["final_centroid_norm"], 1e-4)
            self.assertLess(quality["orientation"]["rotation_det_error"], 1e-8)
            self.assertEqual(quality["warnings"], [])

            display = json.loads((out_dir / "sphere_display.json").read_text())
            mesh_display = json.loads((out_dir / "physical_mesh_display.json").read_text())
            augmented = json.loads((out_dir / "sphere_augmented_qc.json").read_text())
            self.assertEqual(display["schema_version"], "canonical-sphere.display.v2")
            self.assertEqual(display["display_support"], "physical_surface_only")
            self.assertEqual(display["virtual_closure_role"], "numerical_parameterization_only")
            self.assertEqual(display["index_basis"], "package_physical_mesh")
            self.assertEqual(display["n_vertices_display"], mesh_display["n_vertices_display"])
            self.assertEqual(display["n_faces_display"], mesh_display["n_faces_display"])
            self.assertLess(display["n_vertices_display"], augmented["n_vertices_augmented"])
            self.assertLess(display["n_faces_display"], augmented["n_faces_augmented"])
            self.assertLess(max(display["faces"]), display["n_vertices_display"])
            self.assertEqual(len(display["physical_vertex_mask"]), display["n_vertices_display"])
            self.assertTrue(all(display["physical_vertex_mask"]))
            self.assertEqual(len(display["physical_face_mask"]), display["n_faces_display"])
            self.assertTrue(all(display["physical_face_mask"]))
            self.assertEqual(len(display["interface_face_mask"]), display["n_faces_display"])
            self.assertIn("forward_jacobian", display["fields"])
            self.assertIn("inverse_density", display["fields"])
            self.assertIn("conformal_factor", display["fields"])
            self.assertEqual(display["n_vertices_display"], len(display["fields"]["forward_jacobian"]))
            self.assertEqual(display["n_vertices_display"], len(display["fields"]["boundary_distance"]))
            self.assertEqual(display["field_stats"]["mean_curvature"]["support"], "physical_vertices")
            self.assertEqual(display["source_mesh_display"]["index_basis"], "package_physical_mesh")
            self.assertEqual(display["source_mesh_display"]["n_vertices_display"], display["n_vertices_display"])
            self.assertEqual(augmented["schema_version"], "canonical-sphere.augmented-qc.v1")
            self.assertFalse(augmented["display_default"])
            self.assertEqual(augmented["virtual_buffer"]["rings"], 12)
            self.assertEqual(quality["display"]["display_support"], "physical_surface_only")
            self.assertEqual(quality["display"]["curvature_domain"], "physical_mesh_pre_virtual_closure")

            features = np.load(out_dir / "features_grid.npz")
            self.assertEqual(str(features["schema_version"]), "canonical-sphere.features-grid.v1")
            self.assertEqual(list(features["channel_names"]), [
                "radius",
                "forward_jacobian",
                "inverse_density",
                "conformal_factor",
                "mean_curvature",
                "log_conformal_factor",
                "mean_curvature_zscore",
                "radial_distance",
                "physical_mask",
                "boundary_distance",
            ])
            self.assertEqual(int(features["channel_axis"]), 0)

            coeffs = np.load(out_dir / "sh_coeffs.npz")
            self.assertEqual(str(coeffs["schema_version"]), "canonical-sphere.sh-coeffs.v1")
            self.assertEqual(str(coeffs["orientation_signal"]), "log_conformal_factor")


if __name__ == "__main__":
    unittest.main()
