"""Mesh validation and topology metrics for triangle surface meshes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import defaultdict, deque

import numpy as np


@dataclass
class MeshQCReport:
    """Topology and validity summary for a triangular surface mesh."""

    vertex_count: int
    face_count: int
    edge_count: int
    connected_components: int
    boundary_edge_count: int
    nonmanifold_edge_count: int
    euler_characteristic: int
    genus: int | None
    is_watertight: bool
    total_area: float
    bbox_min: list[float]
    bbox_max: list[float]
    warnings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _as_arrays(vertices, faces):
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3)")
    return v, f


def face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return Euclidean triangle areas."""

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def edge_incidence(faces: np.ndarray) -> dict[tuple[int, int], int]:
    """Count incident faces for each undirected edge."""

    counts: dict[tuple[int, int], int] = defaultdict(int)
    for tri in np.asarray(faces, dtype=np.int64):
        for i in range(3):
            a = int(tri[i])
            b = int(tri[(i + 1) % 3])
            if a == b:
                continue
            counts[(a, b) if a < b else (b, a)] += 1
    return counts


def connected_face_components(faces: np.ndarray) -> int:
    """Count face-connected components via shared undirected edges."""

    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) == 0:
        return 0

    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, tri in enumerate(faces):
        for i in range(3):
            a = int(tri[i])
            b = int(tri[(i + 1) % 3])
            edge = (a, b) if a < b else (b, a)
            edge_to_faces[edge].append(fi)

    adjacency = [[] for _ in range(len(faces))]
    for incident in edge_to_faces.values():
        for i in incident:
            adjacency[i].extend(j for j in incident if j != i)

    seen = np.zeros(len(faces), dtype=bool)
    components = 0
    for start in range(len(faces)):
        if seen[start]:
            continue
        components += 1
        queue = deque([start])
        seen[start] = True
        while queue:
            cur = queue.popleft()
            for nxt in adjacency[cur]:
                if not seen[nxt]:
                    seen[nxt] = True
                    queue.append(nxt)
    return components


def validate_triangle_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    require_manifold: bool,
) -> MeshQCReport:
    """Validate a triangle mesh and return topology metrics.

    The function fails hard for invalid arrays, degenerate topology, and
    non-manifold edges when ``require_manifold`` is true. Boundary edges are
    reported rather than rejected because open input meshes are expected before
    virtual closure.
    """

    vertices, faces = _as_arrays(vertices, faces)
    warnings: list[str] = []

    if len(vertices) < 4:
        raise ValueError("mesh must contain at least 4 vertices")
    if len(faces) < 4:
        raise ValueError("mesh must contain at least 4 faces")
    if not np.isfinite(vertices).all():
        raise ValueError("vertices contain NaN or infinite values")
    if faces.min(initial=0) < 0 or faces.max(initial=-1) >= len(vertices):
        raise ValueError("faces contain vertex indices outside the vertex array")
    if np.any((faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 2] == faces[:, 0])):
        raise ValueError("faces contain duplicate vertex indices")

    areas = face_areas(vertices, faces)
    if np.any(~np.isfinite(areas)) or np.any(areas <= 1e-15):
        raise ValueError("mesh contains degenerate or non-finite faces")

    counts = edge_incidence(faces)
    edge_count = len(counts)
    boundary_edges = [edge for edge, count in counts.items() if count == 1]
    nonmanifold_edges = [edge for edge, count in counts.items() if count > 2]
    if require_manifold and nonmanifold_edges:
        raise ValueError(f"mesh has {len(nonmanifold_edges)} non-manifold edges")

    components = connected_face_components(faces)
    if components > 1:
        warnings.append(f"mesh has {components} connected face components")

    chi = int(len(vertices) - edge_count + len(faces))
    is_watertight = len(boundary_edges) == 0 and len(nonmanifold_edges) == 0
    genus = None
    if is_watertight:
        genus_float = (2 - chi) / 2
        if abs(genus_float - round(genus_float)) < 1e-8:
            genus = int(round(genus_float))
        else:
            warnings.append(f"non-integral genus estimate {genus_float:.6g}")

    bbox_min = vertices.min(axis=0).tolist()
    bbox_max = vertices.max(axis=0).tolist()
    return MeshQCReport(
        vertex_count=int(len(vertices)),
        face_count=int(len(faces)),
        edge_count=int(edge_count),
        connected_components=int(components),
        boundary_edge_count=int(len(boundary_edges)),
        nonmanifold_edge_count=int(len(nonmanifold_edges)),
        euler_characteristic=chi,
        genus=genus,
        is_watertight=bool(is_watertight),
        total_area=float(areas.sum()),
        bbox_min=[float(x) for x in bbox_min],
        bbox_max=[float(x) for x in bbox_max],
        warnings=warnings,
    )
