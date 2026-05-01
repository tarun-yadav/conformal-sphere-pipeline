import unittest

import numpy as np
import trimesh

from sphere_mapping_pipeline.spherical.orientation import (
    align_anchor_to_axis,
    resolve_yaw_with_sh,
    rotation_matrix_z,
)
from sphere_mapping_pipeline.spherical.quality import face_areas, spherical_face_centers
from sphere_mapping_pipeline.spherical.resample import resample_face_signal_to_equiangular


def yaw_sensitive_signal(directions):
    x = directions[:, 0]
    y = directions[:, 1]
    z = directions[:, 2]
    return 0.8 * (x * x - y * y) + 0.3 * x * z + 0.2 * y


def corrcoef(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a -= a.mean()
    b -= b.mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class OneAnchorTests(unittest.TestCase):
    def test_one_anchor_leaves_yaw_unresolved_and_sh_resolves_it(self):
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        areas = face_areas(vertices, faces)
        values = yaw_sensitive_signal(spherical_face_centers(vertices, faces))
        anchor_idx = int(np.argmax(vertices[:, 2]))

        anchored, _ = align_anchor_to_axis(vertices, anchor_idx, np.array([0.0, 0.0, 1.0]))
        yawed = (rotation_matrix_z(0.83) @ anchored.T).T

        self.assertLess(np.linalg.norm(anchored[anchor_idx] - yawed[anchor_idx]), 1e-12)
        grid_anchor, _, _ = resample_face_signal_to_equiangular(
            anchored, faces, values, nlat=32, nlon=64
        )
        grid_yawed, _, _ = resample_face_signal_to_equiangular(
            yawed, faces, values, nlat=32, nlon=64
        )
        self.assertLess(corrcoef(grid_anchor, grid_yawed), 0.98)

        resolved_a, psi_a, info_a = resolve_yaw_with_sh(
            anchored, faces, values, areas, np.ones(len(faces), dtype=bool), lmax=4
        )
        resolved_b, psi_b, info_b = resolve_yaw_with_sh(
            yawed, faces, values, areas, np.ones(len(faces), dtype=bool), lmax=4
        )
        grid_a, _, _ = resample_face_signal_to_equiangular(
            resolved_a, faces, values, nlat=32, nlon=64
        )
        grid_b, _, _ = resample_face_signal_to_equiangular(
            resolved_b, faces, values, nlat=32, nlon=64
        )

        self.assertGreater(corrcoef(grid_a, grid_b), 0.99)
        self.assertTrue(np.isfinite([psi_a, psi_b]).all())
        self.assertLess(info_a["c22_imag_rel"], 1e-2)
        self.assertLess(info_b["c22_imag_rel"], 1e-2)


if __name__ == "__main__":
    unittest.main()
