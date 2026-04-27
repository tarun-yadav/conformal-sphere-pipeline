import unittest

import numpy as np

from conformal_sphere_pipeline.boundary import extract_boundary_loops
from conformal_sphere_pipeline.mesh_qc import validate_triangle_mesh
from conformal_sphere_pipeline.spherical.features import compute_face_features
from conformal_sphere_pipeline.spherical.parameterize import radial_parameterization
from conformal_sphere_pipeline.virtual_buffer import add_virtual_boundary_buffers


def make_open_cylinder(n=32, height=2.0, radius=1.0, levels=5):
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    z_values = np.linspace(-height / 2.0, height / 2.0, levels)
    vertices = np.vstack([
        np.column_stack([radius * np.cos(theta), radius * np.sin(theta), np.full(n, z)])
        for z in z_values
    ])
    faces = []
    for level in range(levels - 1):
        base = level * n
        nxt = (level + 1) * n
        for i in range(n):
            j = (i + 1) % n
            faces.append([base + i, base + j, nxt + j])
            faces.append([base + i, nxt + j, nxt + i])
    return vertices.astype(float), np.asarray(faces, dtype=np.int64)


class BoundaryBufferTests(unittest.TestCase):
    def test_virtual_buffer_closes_two_open_cylinder_boundaries(self):
        vertices, faces = make_open_cylinder()

        loops = extract_boundary_loops(vertices, faces)
        self.assertEqual(len(loops), 2)
        self.assertEqual(sorted(len(loop.vertex_indices) for loop in loops), [32, 32])

        result = add_virtual_boundary_buffers(
            vertices,
            faces,
            loops=loops,
            rings=6,
            length_factor=1.5,
            min_length_factor_bbox=0.01,
            apex_radius_fraction=0.04,
        )

        report = validate_triangle_mesh(
            result.vertices, result.faces, require_manifold=True
        )
        self.assertTrue(report.is_watertight)
        self.assertEqual(report.boundary_edge_count, 0)
        self.assertEqual(report.nonmanifold_edge_count, 0)
        self.assertEqual(report.genus, 0)
        self.assertTrue(np.all(result.physical_vertex_mask[: len(vertices)]))
        self.assertTrue(np.all(result.virtual_vertex_mask[len(vertices) :]))
        self.assertEqual(int(result.physical_face_mask.sum()), len(faces))
        self.assertGreater(int(result.interface_face_mask.sum()), 0)

        sphere = radial_parameterization(result.vertices)
        channels, _ = compute_face_features(
            result.vertices,
            result.faces,
            sphere,
            result.physical_face_mask,
        )
        boundary_distance = channels["boundary_distance"]
        self.assertGreater(float(boundary_distance[result.physical_face_mask].max()), 0.0)
        self.assertEqual(float(boundary_distance[result.virtual_face_mask].max()), 0.0)


if __name__ == "__main__":
    unittest.main()
