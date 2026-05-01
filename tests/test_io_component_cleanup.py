from pathlib import Path

import numpy as np
import trimesh

from sphere_mapping_pipeline.io import load_triangle_mesh


def test_load_triangle_mesh_keeps_largest_component(tmp_path):
    large = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    small = trimesh.creation.icosphere(subdivisions=1, radius=0.1)
    small.apply_translation([4.0, 0.0, 0.0])
    combined = trimesh.util.concatenate([large, small])
    path = tmp_path / "two_components.ply"
    combined.export(path)

    vertices, faces = load_triangle_mesh(path)
    loaded = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    assert len(loaded.split(only_watertight=False)) == 1
    assert np.isclose(loaded.area, large.area, rtol=1e-2)
