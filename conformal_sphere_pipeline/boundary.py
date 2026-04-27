"""Ordered boundary-loop extraction and loop geometry summaries."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from .mesh_qc import edge_incidence


@dataclass
class BoundaryLoop:
    """Ordered mesh boundary with fitted plane and descriptive metadata."""

    id: int
    vertex_indices: np.ndarray
    center: np.ndarray
    normal: np.ndarray
    radius_mean: float
    radius_max: float
    circumference: float
    area_planar: float
    plane_basis_u: np.ndarray
    plane_basis_v: np.ndarray
    classification: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "id": int(self.id),
            "size": int(len(self.vertex_indices)),
            "center": self.center.tolist(),
            "normal": self.normal.tolist(),
            "radius_mean": float(self.radius_mean),
            "radius_max": float(self.radius_max),
            "circumference": float(self.circumference),
            "area_planar": float(self.area_planar),
            "classification": self.classification,
        }


def _normalize(v: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros(3, dtype=float)
    return np.asarray(v, dtype=float) / n


def _ordered_boundary_components(boundary_edges: list[tuple[int, int]]) -> list[np.ndarray]:
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary_edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))

    loops: list[np.ndarray] = []
    seen_vertices: set[int] = set()
    for start in sorted(adj):
        if start in seen_vertices:
            continue

        component = []
        queue = deque([start])
        seen_vertices.add(start)
        while queue:
            cur = queue.popleft()
            component.append(cur)
            for nxt in adj[cur]:
                if nxt not in seen_vertices:
                    seen_vertices.add(nxt)
                    queue.append(nxt)

        for vertex in component:
            if len(adj[vertex]) != 2:
                raise ValueError("boundary component is not a simple cycle")

        ordered = [min(component)]
        prev = None
        cur = ordered[0]
        while True:
            candidates = [n for n in adj[cur] if n != prev]
            if not candidates:
                raise ValueError("boundary cycle terminated unexpectedly")
            nxt = candidates[0]
            if nxt == ordered[0]:
                break
            ordered.append(nxt)
            prev, cur = cur, nxt
            if len(ordered) > len(component):
                raise ValueError("boundary cycle traversal did not close")

        if len(ordered) != len(component):
            raise ValueError("boundary component contains branches or repeated vertices")
        loops.append(np.asarray(ordered, dtype=np.int64))
    return loops


def _plane_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    centered = points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = _normalize(vh[-1])
    basis_u = _normalize(vh[0])
    basis_v = _normalize(np.cross(normal, basis_u))
    normal = _normalize(np.cross(basis_u, basis_v))
    return normal, basis_u, basis_v


def _polygon_area_2d(xy: np.ndarray) -> float:
    x = xy[:, 0]
    y = xy[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _orient_normal_outward(
    normal: np.ndarray,
    center: np.ndarray,
    mesh_centroid: np.ndarray,
) -> np.ndarray:
    away = center - mesh_centroid
    if float(np.dot(normal, away)) < 0.0:
        return -normal
    return normal


def extract_boundary_loops(vertices: np.ndarray, faces: np.ndarray) -> list[BoundaryLoop]:
    """Return ordered simple boundary loops for an open triangular mesh."""

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    counts = edge_incidence(faces)
    nonmanifold = [edge for edge, count in counts.items() if count > 2]
    if nonmanifold:
        raise ValueError(f"mesh has {len(nonmanifold)} non-manifold edges")

    boundary_edges = [edge for edge, count in counts.items() if count == 1]
    if not boundary_edges:
        return []

    raw_loops = _ordered_boundary_components(boundary_edges)
    mesh_centroid = vertices.mean(axis=0)
    loops: list[BoundaryLoop] = []
    radii_for_classification = []
    for loop_id, indices in enumerate(raw_loops):
        pts = vertices[indices]
        center = pts.mean(axis=0)
        normal, basis_u, basis_v = _plane_basis(pts)
        normal = _orient_normal_outward(normal, center, mesh_centroid)

        offsets = pts - center
        xy = np.column_stack([offsets @ basis_u, offsets @ basis_v])
        signed_area = _polygon_area_2d(xy)
        if signed_area < 0.0:
            indices = indices[::-1].copy()
            pts = vertices[indices]
            offsets = pts - center
            xy = np.column_stack([offsets @ basis_u, offsets @ basis_v])
            signed_area = _polygon_area_2d(xy)

        radii = np.linalg.norm(offsets, axis=1)
        circumference = float(np.sum(np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1)))
        radii_for_classification.append(float(radii.mean()))
        loops.append(
            BoundaryLoop(
                id=loop_id,
                vertex_indices=indices,
                center=center,
                normal=normal,
                radius_mean=float(radii.mean()),
                radius_max=float(radii.max()),
                circumference=circumference,
                area_planar=float(abs(signed_area)),
                plane_basis_u=basis_u,
                plane_basis_v=basis_v,
                classification="unknown",
            )
        )

    if loops:
        largest = int(np.argmax(radii_for_classification))
        for loop in loops:
            loop.classification = "root" if loop.id == largest else "branch"
    return loops
