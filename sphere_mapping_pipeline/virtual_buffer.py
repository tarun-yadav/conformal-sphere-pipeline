"""Smooth tapered virtual buffers for closing open mesh boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boundary import BoundaryLoop, extract_boundary_loops


@dataclass
class VirtualBufferResult:
    """Closed mesh and masks identifying physical versus virtual geometry."""

    vertices: np.ndarray
    faces: np.ndarray
    physical_vertex_mask: np.ndarray
    virtual_vertex_mask: np.ndarray
    physical_face_mask: np.ndarray
    virtual_face_mask: np.ndarray
    interface_face_mask: np.ndarray
    loops: list[BoundaryLoop]


def _smoothstep(t: float) -> float:
    return 3.0 * t * t - 2.0 * t * t * t


def _face_normal(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = np.cross(b - a, c - a)
    norm = float(np.linalg.norm(n))
    if norm <= 1e-15:
        return np.zeros(3)
    return n / norm


def _append_oriented(
    faces: list[list[int]],
    vertices: list[np.ndarray],
    tri: list[int],
    expected: np.ndarray,
) -> None:
    a, b, c = (np.asarray(vertices[i], dtype=float) for i in tri)
    normal = _face_normal(a, b, c)
    if float(np.dot(normal, expected)) < 0.0:
        tri = [tri[0], tri[2], tri[1]]
    faces.append(tri)


def add_virtual_boundary_buffers(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    loops: list[BoundaryLoop] | None = None,
    rings: int = 12,
    length_factor: float = 2.5,
    min_length_factor_bbox: float = 0.03,
    apex_radius_fraction: float = 0.02,
) -> VirtualBufferResult:
    """Close every boundary loop with a tapered virtual tube and apex cap."""

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if loops is None:
        loops = extract_boundary_loops(vertices, faces)

    out_vertices: list[np.ndarray] = [v.copy() for v in vertices]
    out_faces: list[list[int]] = [list(map(int, f)) for f in faces]
    physical_face_count = len(out_faces)
    interface_face_indices: list[int] = []

    bbox_diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    for loop in loops:
        loop_indices = [int(i) for i in loop.vertex_indices]
        loop_pts = vertices[loop_indices]
        center = loop.center
        normal = loop.normal
        offsets = loop_pts - center
        radius = max(loop.radius_mean, 1e-12)
        length = max(float(length_factor) * radius, float(min_length_factor_bbox) * bbox_diag)
        ring_count = int(rings)
        if ring_count < 1:
            raise ValueError("rings must be at least 1")

        all_rings = [loop_indices]
        for ring_idx in range(1, ring_count + 1):
            t = ring_idx / ring_count
            smooth = _smoothstep(t)
            scale = (1.0 - smooth) + float(apex_radius_fraction) * smooth
            offset = length * smooth
            positions = center + scale * offsets + offset * normal
            start = len(out_vertices)
            out_vertices.extend([p.copy() for p in positions])
            all_rings.append(list(range(start, start + len(loop_indices))))

        for ring_idx in range(len(all_rings) - 1):
            inner = all_rings[ring_idx]
            outer = all_rings[ring_idx + 1]
            for i in range(len(loop_indices)):
                j = (i + 1) % len(loop_indices)
                v0, v1 = inner[i], inner[j]
                v2, v3 = outer[i], outer[j]
                expected = vertices[loop_indices[i]] - center
                if np.linalg.norm(expected) < 1e-15:
                    expected = normal
                before = len(out_faces)
                _append_oriented(out_faces, out_vertices, [v0, v1, v3], expected)
                _append_oriented(out_faces, out_vertices, [v0, v3, v2], expected)
                if ring_idx == 0:
                    interface_face_indices.extend([before, before + 1])

        apex_idx = len(out_vertices)
        out_vertices.append(center + length * normal)
        final_ring = all_rings[-1]
        for i in range(len(loop_indices)):
            j = (i + 1) % len(loop_indices)
            expected = normal
            _append_oriented(out_faces, out_vertices, [final_ring[i], final_ring[j], apex_idx], expected)

    closed_vertices = np.asarray(out_vertices, dtype=np.float64)
    closed_faces = np.asarray(out_faces, dtype=np.int64)
    physical_vertex_mask = np.zeros(len(closed_vertices), dtype=bool)
    physical_vertex_mask[: len(vertices)] = True
    virtual_vertex_mask = ~physical_vertex_mask
    physical_face_mask = np.zeros(len(closed_faces), dtype=bool)
    physical_face_mask[:physical_face_count] = True
    virtual_face_mask = ~physical_face_mask
    interface_face_mask = np.zeros(len(closed_faces), dtype=bool)
    valid_interface = [i for i in interface_face_indices if i < len(interface_face_mask)]
    interface_face_mask[valid_interface] = True

    return VirtualBufferResult(
        vertices=closed_vertices,
        faces=closed_faces,
        physical_vertex_mask=physical_vertex_mask,
        virtual_vertex_mask=virtual_vertex_mask,
        physical_face_mask=physical_face_mask,
        virtual_face_mask=virtual_face_mask,
        interface_face_mask=interface_face_mask,
        loops=loops,
    )
