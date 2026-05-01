import unittest

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from sphere_mapping_pipeline.spherical.orientation import canonical_orient_sphere
from sphere_mapping_pipeline.spherical.quality import face_areas, spherical_face_centers
from sphere_mapping_pipeline.spherical.resample import resample_face_signal_to_equiangular


def orientation_signal(directions):
    x = directions[:, 0]
    y = directions[:, 1]
    z = directions[:, 2]
    return 0.9 * (3.0 * z * z - 1.0) + 0.55 * (x * x - y * y) + 0.25 * x * y * z


def corrcoef(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a = a - a.mean()
    b = b - b.mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class SHOrientationTests(unittest.TestCase):
    def test_randomly_rotated_signals_canonicalize_to_same_grid(self):
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        areas = face_areas(vertices, faces)
        values = orientation_signal(spherical_face_centers(vertices, faces))
        mask = np.ones(len(faces), dtype=bool)

        reference_u, _, reference_info = canonical_orient_sphere(
            vertices,
            faces,
            values,
            areas,
            mask,
            lmax_orientation=6,
            refine_bands=(2, 3, 4),
        )
        reference_grid, _, _ = resample_face_signal_to_equiangular(
            reference_u, faces, values, nlat=36, nlon=72
        )
        self.assertLess(reference_info["orientation_c21_rel"], 1e-2)
        self.assertLess(reference_info["orientation_c22_imag_rel"], 1e-2)
        self.assertGreater(reference_info["orientation_c22_real_rel"], 0.0)

        rotations = Rotation.random(8, random_state=42).as_matrix()
        for rotation in rotations:
            rotated = (rotation @ vertices.T).T
            canonical_u, recovered, info = canonical_orient_sphere(
                rotated,
                faces,
                values,
                areas,
                mask,
                lmax_orientation=6,
                refine_bands=(2, 3, 4),
            )
            grid, _, _ = resample_face_signal_to_equiangular(
                canonical_u, faces, values, nlat=36, nlon=72
            )
            self.assertGreater(corrcoef(grid, reference_grid), 0.99)
            self.assertLess(info["orientation_c21_rel"], 1e-2)
            self.assertLess(info["orientation_c22_imag_rel"], 1e-2)
            self.assertLess(abs(np.linalg.det(recovered) - 1.0), 1e-8)


if __name__ == "__main__":
    unittest.main()
