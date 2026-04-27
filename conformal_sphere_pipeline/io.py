"""Mesh IO helpers for preprocessing triangle surfaces."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh


def _coerce_trimesh(obj) -> trimesh.Trimesh:
    if isinstance(obj, trimesh.Scene):
        meshes = [g for g in obj.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("scene contains no triangle meshes")
        return trimesh.util.concatenate(meshes)
    if not isinstance(obj, trimesh.Trimesh):
        raise ValueError(f"unsupported mesh object {type(obj)!r}")
    return obj


def load_triangle_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a triangle mesh and return ``float64`` vertices and ``int64`` faces."""

    path = Path(path)
    mesh = _coerce_trimesh(trimesh.load(path, process=True))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    finite_faces = np.isfinite(vertices[faces]).all(axis=(1, 2))
    distinct = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
    area = 0.5 * np.linalg.norm(
        np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]]),
        axis=1,
    )
    keep = finite_faces & distinct & (area > 1e-15)
    faces = faces[keep]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        mesh = max(components, key=lambda item: float(item.area))
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def write_mesh(path: str | Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a triangular mesh to disk."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    mesh.export(path)


def json_default(obj):
    """JSON serializer for numpy values."""

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: str | Path, payload: dict) -> None:
    """Write indented JSON with numpy conversion."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default))
