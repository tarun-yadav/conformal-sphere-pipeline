"""Synthetic known-correspondence cases for atlas validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from conformal_sphere_pipeline.boundary import extract_boundary_loops


@dataclass(frozen=True)
class MaterialMesh:
    vertices: np.ndarray
    faces: np.ndarray
    material_s: np.ndarray
    material_theta: np.ndarray


@dataclass(frozen=True)
class CappedMaterialMesh:
    vertices: np.ndarray
    faces: np.ndarray
    original_vertex_count: int
    material_s: np.ndarray
    material_theta: np.ndarray


@dataclass(frozen=True)
class CandyCaneCase:
    reference: MaterialMesh
    uniform_2x: MaterialMesh
    local_bulge: MaterialMesh
    material_s: np.ndarray
    material_theta: np.ndarray


def _centerline(s_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stem_length = 2.2
    hook_radius = 0.65
    tip_length = 0.9
    total_length = stem_length + np.pi * hook_radius + tip_length

    center = np.zeros((len(s_values), 3), dtype=np.float64)
    tangent = np.zeros_like(center)
    normal = np.zeros_like(center)
    for idx, value in enumerate(s_values):
        arclength = float(value) * total_length
        if arclength <= stem_length:
            center[idx] = [0.0, 0.0, arclength]
            tangent[idx] = [0.0, 0.0, 1.0]
            normal[idx] = [1.0, 0.0, 0.0]
        else:
            hook_arclength = arclength - stem_length
            if hook_arclength <= np.pi * hook_radius:
                angle = hook_arclength / hook_radius
                center[idx] = [
                    hook_radius * (1.0 - np.cos(angle)),
                    0.0,
                    stem_length + hook_radius * np.sin(angle),
                ]
                tangent[idx] = [np.sin(angle), 0.0, np.cos(angle)]
                normal[idx] = [np.cos(angle), 0.0, -np.sin(angle)]
            else:
                tip_arclength = hook_arclength - np.pi * hook_radius
                center[idx] = [2.0 * hook_radius, 0.0, stem_length - tip_arclength]
                tangent[idx] = [0.0, 0.0, -1.0]
                normal[idx] = [-1.0, 0.0, 0.0]
    del tangent
    binormal = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float64), (len(s_values), 1))
    return center, normal, binormal


def _build_faces(n_s: int, n_theta: int) -> np.ndarray:
    faces: list[list[int]] = []
    for i in range(n_s - 1):
        for j in range(n_theta):
            jn = (j + 1) % n_theta
            v00 = i * n_theta + j
            v01 = i * n_theta + jn
            v10 = (i + 1) * n_theta + j
            v11 = (i + 1) * n_theta + jn
            faces.append([v00, v10, v01])
            faces.append([v01, v10, v11])
    return np.asarray(faces, dtype=np.int64)


def _build_variant(n_s: int, n_theta: int, radius_mode: str) -> MaterialMesh:
    if n_s < 4 or n_theta < 4:
        raise ValueError("n_s and n_theta must both be at least 4")
    s_values = np.linspace(0.0, 1.0, n_s)
    theta_values = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    center, normal, binormal = _centerline(s_values)
    vertices: list[np.ndarray] = []
    material_s: list[float] = []
    material_theta: list[float] = []
    for i, s_value in enumerate(s_values):
        for theta in theta_values:
            radius = 0.22
            if radius_mode == "uniform_2x":
                radius *= 2.0
            elif radius_mode == "local_bulge":
                bump = np.exp(-0.5 * ((s_value - 0.52) / 0.075) ** 2) * np.exp(1.2 * np.cos(theta))
                radius *= 1.0 + 0.35 * bump
            offset = radius * (np.cos(theta) * normal[i] + np.sin(theta) * binormal[i])
            vertices.append(center[i] + offset)
            material_s.append(float(s_value))
            material_theta.append(float(theta))
    return MaterialMesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=_build_faces(n_s, n_theta),
        material_s=np.asarray(material_s, dtype=np.float64),
        material_theta=np.asarray(material_theta, dtype=np.float64),
    )


def build_candy_cane_case(n_s: int = 48, n_theta: int = 24) -> CandyCaneCase:
    reference = _build_variant(n_s, n_theta, "reference")
    uniform_2x = _build_variant(n_s, n_theta, "uniform_2x")
    local_bulge = _build_variant(n_s, n_theta, "local_bulge")
    return CandyCaneCase(
        reference=reference,
        uniform_2x=uniform_2x,
        local_bulge=local_bulge,
        material_s=reference.material_s,
        material_theta=reference.material_theta,
    )


def cap_open_tube_with_material_ids(mesh: MaterialMesh) -> CappedMaterialMesh:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = [list(map(int, face)) for face in np.asarray(mesh.faces, dtype=np.int64)]
    loops = extract_boundary_loops(vertices, np.asarray(faces, dtype=np.int64))
    for loop in loops:
        indices = [int(index) for index in loop.vertex_indices]
        center_index = len(vertices)
        vertices = np.vstack([vertices, vertices[indices].mean(axis=0)])
        for a, b in zip(indices, indices[1:] + indices[:1]):
            faces.append([center_index, a, b])
    closed = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64), process=False)
    trimesh.repair.fix_normals(closed)
    return CappedMaterialMesh(
        vertices=np.asarray(closed.vertices, dtype=np.float64),
        faces=np.asarray(closed.faces, dtype=np.int64),
        original_vertex_count=len(mesh.vertices),
        material_s=np.asarray(mesh.material_s, dtype=np.float64),
        material_theta=np.asarray(mesh.material_theta, dtype=np.float64),
    )
