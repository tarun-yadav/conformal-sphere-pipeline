"""Stereographic spherical parameterization primitives."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from itertools import permutations
import time

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


def _validate_pole_sign(pole_sign: int) -> int:
    if int(pole_sign) not in {-1, 1}:
        raise ValueError("pole_sign must be +1 or -1")
    return int(pole_sign)


@dataclass(frozen=True)
class StereographicConfig:
    """Deterministic implementation choices for the stereographic backend."""

    pole_face_index: int | None = None
    pole_sign: int = 1
    boundary_radius: float = 1.0
    pole_area_threshold: float = 1e-4
    stage_a_max_iter: int = 200
    stage_b_max_iter: int = 300
    gradient_tol: float = 1e-6
    relative_energy_tol: float = 1e-6
    armijo_c1: float = 1e-4
    max_backtracks: int = 30
    injectivity_max_depth: int = 12
    diagnostics_samples_per_face: int = 1000


@dataclass(frozen=True)
class StereographicDiskInitialization:
    """Initial disk map and metadata before Newton optimization."""

    scaled_vertices: np.ndarray
    faces: np.ndarray
    disk_faces: np.ndarray
    disk_face_indices: np.ndarray
    pole_face_index: int
    pole_face: np.ndarray
    input_pole_face_index: int
    pole_selection: str
    boundary_vertices: np.ndarray
    planar_vertices: np.ndarray
    scaled_area: float
    pole_regularness: float
    min_planar_double_area: float
    pole_refinement_status: str
    original_vertex_count: int | None = None
    pole_area: float | None = None


@dataclass
class _BezierControlMesh:
    vertex_count: int
    controls: np.ndarray
    face_control_indices: np.ndarray
    edge_lookup: dict[tuple[int, int], int]


@dataclass
class StereographicParameterizationResult:
    """Full stereographic result with optimized controls and continuous evaluator."""

    success: bool
    sphere: np.ndarray
    vertex_controls: np.ndarray
    controls: np.ndarray
    refined_vertices: np.ndarray
    refined_faces: np.ndarray
    disk_faces: np.ndarray
    disk_face_indices: np.ndarray
    disk_face_control_indices: np.ndarray
    pole_face_id: int
    pole_face: np.ndarray
    original_vertex_count: int
    edge_lookup: dict[tuple[int, int], int]
    info: dict
    pole_sign: int = 1

    def evaluate(self, face_id: int, barycentric: np.ndarray) -> np.ndarray:
        """Evaluate the continuous stereographic map on an internal refined face."""

        bary = np.asarray(barycentric, dtype=np.float64)
        if bary.shape != (3,):
            raise ValueError("barycentric must have shape (3,)")
        if np.any(bary < -1e-12) or not np.isclose(float(np.sum(bary)), 1.0, atol=1e-10):
            raise ValueError("barycentric coordinates must be nonnegative and sum to 1")
        face_id = int(face_id)
        if face_id == self.pole_face_id:
            planar = _evaluate_pole_exterior_planar(
                self.pole_face,
                bary,
                self.vertex_controls,
                self.controls,
                self.edge_lookup,
            )
            if planar is None:
                return np.array([0.0, 0.0, float(self.pole_sign)], dtype=np.float64)
            return inverse_stereographic(planar[None, :], pole_sign=self.pole_sign)[0]

        matches = np.flatnonzero(self.disk_face_indices == face_id)
        if len(matches) != 1:
            raise ValueError(f"face_id {face_id} is not present in the refined stereographic map")
        local = int(matches[0])
        uv = np.array([bary[1], bary[2]], dtype=np.float64)
        patch = _patch_from_controls(self.controls, self.disk_face_control_indices[local])
        return inverse_stereographic(patch.evaluate(uv)[None, :], pole_sign=self.pole_sign)[0]


def inverse_stereographic(points: np.ndarray, *, pole_sign: int = 1) -> np.ndarray:
    """Map planar coordinates to the unit sphere with a consistent pole sign."""

    sigma = _validate_pole_sign(pole_sign)
    z = np.asarray(points, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(1, 2)
    if z.ndim != 2 or z.shape[1] != 2:
        raise ValueError("points must have shape (N, 2) or (2,)")

    r2 = np.sum(z * z, axis=1)
    out = np.empty((len(z), 3), dtype=np.float64)
    huge = r2 > 1e30
    safe = ~huge
    if np.any(safe):
        denom = 1.0 + r2[safe]
        out[safe, 0] = 2.0 * z[safe, 0] / denom
        out[safe, 1] = 2.0 * z[safe, 1] / denom
        out[safe, 2] = sigma * (r2[safe] - 1.0) / denom
    if np.any(huge):
        out[huge] = np.array([0.0, 0.0, float(sigma)], dtype=np.float64)
    return out


def inverse_stereographic_jacobian(point: np.ndarray, *, pole_sign: int = 1) -> np.ndarray:
    """Return the 3 x 2 derivative of inverse stereographic projection."""

    sigma = _validate_pole_sign(pole_sign)
    z = np.asarray(point, dtype=np.float64)
    if z.shape != (2,):
        raise ValueError("point must have shape (2,)")
    u = float(z[0])
    v = float(z[1])
    r2 = u * u + v * v
    denom2 = (1.0 + r2) ** 2
    return np.array(
        [
            [2.0 * (1.0 - u * u + v * v) / denom2, -4.0 * u * v / denom2],
            [-4.0 * u * v / denom2, 2.0 * (1.0 + u * u - v * v) / denom2],
            [sigma * 4.0 * u / denom2, sigma * 4.0 * v / denom2],
        ],
        dtype=np.float64,
    )


def _distinct_permutations(values: tuple[float, float, float]) -> list[tuple[float, float, float]]:
    seen: set[tuple[float, float, float]] = set()
    out: list[tuple[float, float, float]] = []
    for perm in permutations(values):
        key = tuple(round(float(x), 15) for x in perm)
        if key in seen:
            continue
        seen.add(key)
        out.append(perm)
    return out


def lyness_jespersen_rule_10() -> tuple[np.ndarray, np.ndarray]:
    """Return rule-10 quadrature points as ``(u, v)`` and normalized weights."""

    orbits = [
        (
            (0.5014265096581342, 0.2492867451709329, 0.2492867451709329),
            0.3503588271790222,
        ),
        (
            (0.8738219710169965, 0.06308901449150177, 0.06308901449150169),
            0.1525347191106164,
        ),
        (
            (0.6365024991213939, 0.05314504984483216, 0.3103524510337740),
            0.4971064537103575,
        ),
    ]
    points: list[tuple[float, float]] = []
    weights: list[float] = []
    for barycentric, orbit_weight in orbits:
        expanded = _distinct_permutations(barycentric)
        point_weight = orbit_weight / len(expanded)
        for lambda0, lambda1, lambda2 in expanded:
            points.append((lambda1, lambda2))
            weights.append(point_weight)
    return np.asarray(points, dtype=np.float64), np.asarray(weights, dtype=np.float64)


def initialize_stereographic_disk(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    boundary_radius: float = 1.0,
    pole_face_index: int | None = None,
) -> StereographicDiskInitialization:
    """Build the paper's scaled one-pole Tutte disk initialization."""

    vertices, faces = _as_mesh_arrays(vertices, faces)
    _validate_closed_genus_zero(vertices, faces)
    scaled_vertices, scaled_area = _scale_to_unit_sphere_area(vertices, faces)
    areas = _face_areas(scaled_vertices, faces)
    input_pole_face_index, pole_regularness, pole_selection = _select_pole_face(
        scaled_vertices,
        faces,
        areas,
        pole_face_index=pole_face_index,
    )
    pole_face = faces[input_pole_face_index].copy()
    disk_face_indices = np.asarray(
        [idx for idx in range(len(faces)) if idx != input_pole_face_index],
        dtype=np.int64,
    )
    disk_faces = faces[disk_face_indices].copy()
    boundary_vertices = pole_face.copy()

    candidates = [
        boundary_vertices,
        boundary_vertices[[0, 2, 1]],
    ]
    best_planar = None
    best_min_area = -np.inf
    best_boundary = None
    for boundary_order in candidates:
        planar = _solve_tutte_disk(
            len(vertices),
            disk_faces,
            boundary_order,
            boundary_radius=boundary_radius,
        )
        signed = _planar_double_areas(planar, disk_faces)
        min_signed = float(np.min(signed))
        if min_signed > best_min_area:
            best_planar = planar
            best_min_area = min_signed
            best_boundary = boundary_order.copy()
        if min_signed > 1e-14:
            return StereographicDiskInitialization(
                scaled_vertices=scaled_vertices,
                faces=faces,
                disk_faces=disk_faces,
                disk_face_indices=disk_face_indices,
                pole_face_index=int(input_pole_face_index),
                pole_face=pole_face,
                input_pole_face_index=int(input_pole_face_index),
                pole_selection=pole_selection,
                boundary_vertices=boundary_order.copy(),
                planar_vertices=planar,
                scaled_area=scaled_area,
                pole_regularness=float(pole_regularness),
                min_planar_double_area=min_signed,
                pole_refinement_status="not_implemented",
            )

    assert best_planar is not None and best_boundary is not None
    raise ValueError(
        "Tutte initialization produced non-positive disk face orientation; "
        f"best minimum signed double area was {best_min_area:.3e}"
    )


def compute_stereographic_parameterization(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    return_info: bool = False,
    config: StereographicConfig | None = None,
    pole_sign: int = 1,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Compute stereographic spherical coordinates for input vertices."""

    if config is None:
        config = StereographicConfig(pole_sign=pole_sign)
    elif pole_sign != 1 and config.pole_sign != pole_sign:
        config = StereographicConfig(**{**config.__dict__, "pole_sign": pole_sign})
    result = stereographic_spherical_parameterization(vertices, faces, config=config)
    sphere = result.sphere
    info = {"method": "stereographic", **result.info}
    if return_info:
        return sphere, info
    return sphere


def stereographic_spherical_parameterization(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    config: StereographicConfig | None = None,
) -> StereographicParameterizationResult:
    """Run the stereographic Bezier spherical parameterization pipeline."""

    cfg = _validate_stereographic_config(config or StereographicConfig())
    start = time.perf_counter()
    init = _initialize_stereographic_disk(vertices, faces, cfg)
    control_mesh = _build_quadratic_control_mesh(init.disk_faces, init.planar_vertices)
    controls, optimization_info = _optimize_stereographic_controls(init, control_mesh, cfg)
    vertex_controls = controls[: control_mesh.vertex_count]
    sphere = inverse_stereographic(vertex_controls[: init.original_vertex_count], pole_sign=cfg.pole_sign)
    diagnostics = _compute_stereographic_diagnostics(init, control_mesh, controls, cfg, sphere)
    status = optimization_info["optimization_status"]
    info = {
        "backend": "stereographic_bezier",
        "optimization_status": status,
        "pole_refinement_status": init.pole_refinement_status,
        "pole_face_index": int(init.pole_face_index),
        "pole_face_id": int(init.pole_face_index),
        "input_pole_face_index": int(init.input_pole_face_index),
        "pole_selection": init.pole_selection,
        "pole_regularness": float(init.pole_regularness),
        "pole_area": float(init.pole_area if init.pole_area is not None else np.nan),
        "scaled_area": float(init.scaled_area),
        "refined_vertex_count": int(len(init.scaled_vertices)),
        "refined_face_count": int(len(init.faces)),
        "disk_face_count": int(len(init.disk_faces)),
        "min_planar_double_area": float(init.min_planar_double_area),
        "pole_sign": int(cfg.pole_sign),
        "max_sphere_norm_error": float(np.max(np.abs(np.linalg.norm(sphere, axis=1) - 1.0))),
        "runtime_seconds": float(time.perf_counter() - start),
        **optimization_info,
        **diagnostics,
    }
    return StereographicParameterizationResult(
        success=bool(status in {"converged", "max_iter", "line_search_failed"}),
        sphere=sphere,
        vertex_controls=vertex_controls,
        controls=controls,
        refined_vertices=init.scaled_vertices,
        refined_faces=init.faces,
        disk_faces=init.disk_faces,
        disk_face_indices=init.disk_face_indices,
        disk_face_control_indices=control_mesh.face_control_indices,
        pole_face_id=int(init.pole_face_index),
        pole_face=init.pole_face.copy(),
        original_vertex_count=int(init.original_vertex_count or len(vertices)),
        edge_lookup=control_mesh.edge_lookup,
        info=info,
        pole_sign=cfg.pole_sign,
    )


@dataclass(frozen=True)
class QuadraticBezierPatch:
    """Quadratic Bezier triangle in the local vertex convention."""

    b002: np.ndarray
    b200: np.ndarray
    b020: np.ndarray
    b101: np.ndarray
    b011: np.ndarray
    b110: np.ndarray

    @classmethod
    def from_affine_triangle(
        cls,
        z0: np.ndarray,
        z1: np.ndarray,
        z2: np.ndarray,
    ) -> "QuadraticBezierPatch":
        z0 = _as_planar_point(z0, "z0")
        z1 = _as_planar_point(z1, "z1")
        z2 = _as_planar_point(z2, "z2")
        return cls(
            b002=z0,
            b200=z1,
            b020=z2,
            b101=0.5 * (z0 + z1),
            b011=0.5 * (z0 + z2),
            b110=0.5 * (z1 + z2),
        )

    def evaluate(self, uv: np.ndarray) -> np.ndarray:
        value, _ = self.evaluate_with_jacobian(uv)
        return value

    def evaluate_with_jacobian(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u, v, w = _reference_uv(uv)
        value = (
            w * w * self.b002
            + u * u * self.b200
            + v * v * self.b020
            + 2.0 * u * w * self.b101
            + 2.0 * v * w * self.b011
            + 2.0 * u * v * self.b110
        )
        d_du = (
            -2.0 * w * self.b002
            + 2.0 * u * self.b200
            + 2.0 * (w - u) * self.b101
            - 2.0 * v * self.b011
            + 2.0 * v * self.b110
        )
        d_dv = (
            -2.0 * w * self.b002
            + 2.0 * v * self.b020
            - 2.0 * u * self.b101
            + 2.0 * (w - v) * self.b011
            + 2.0 * u * self.b110
        )
        jac = np.column_stack([d_du, d_dv])
        return value, jac

    def min_sampled_det_jacobian(self) -> float:
        points, _ = lyness_jespersen_rule_10()
        dets = []
        for point in points:
            _, jac = self.evaluate_with_jacobian(point)
            dets.append(float(np.linalg.det(jac)))
        return float(np.min(dets))


def _as_planar_point(value: np.ndarray, name: str) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (2,):
        raise ValueError(f"{name} must have shape (2,)")
    return point


def _reference_uv(value: np.ndarray) -> tuple[float, float, float]:
    uv = np.asarray(value, dtype=np.float64)
    if uv.shape != (2,):
        raise ValueError("uv must have shape (2,)")
    u = float(uv[0])
    v = float(uv[1])
    w = 1.0 - u - v
    if u < -1e-14 or v < -1e-14 or w < -1e-14:
        raise ValueError("uv is outside the reference triangle")
    return u, v, w


def _as_mesh_arrays(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3)")
    if len(v) < 4 or len(f) < 4:
        raise ValueError("mesh must contain at least 4 vertices and 4 faces")
    if not np.isfinite(v).all():
        raise ValueError("vertices contain NaN or infinite values")
    if f.min(initial=0) < 0 or f.max(initial=-1) >= len(v):
        raise ValueError("faces contain vertex indices outside the vertex array")
    if np.any((f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 2] == f[:, 0])):
        raise ValueError("faces contain duplicate vertex indices")
    return v, f


def _face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def _edge_counts(faces: np.ndarray) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for tri in faces:
        for corner in range(3):
            a = int(tri[corner])
            b = int(tri[(corner + 1) % 3])
            counts[(a, b) if a < b else (b, a)] += 1
    return counts


def _validate_closed_genus_zero(vertices: np.ndarray, faces: np.ndarray) -> None:
    areas = _face_areas(vertices, faces)
    bbox = vertices.max(axis=0) - vertices.min(axis=0)
    eps_area = 1e-14 * max(float(np.dot(bbox, bbox)), 1.0)
    if np.any(~np.isfinite(areas)) or np.any(areas <= eps_area):
        raise ValueError("mesh contains degenerate or non-finite faces")
    counts = _edge_counts(faces)
    bad_edges = [edge for edge, count in counts.items() if count != 2]
    if bad_edges:
        raise ValueError(f"mesh must be closed 2-manifold; {len(bad_edges)} edge(s) have incident count != 2")
    chi = int(len(vertices) - len(counts) + len(faces))
    if chi != 2:
        raise ValueError(f"mesh must be closed genus-zero with Euler characteristic 2; got {chi}")


def _scale_to_unit_sphere_area(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, float]:
    area = float(np.sum(_face_areas(vertices, faces)))
    if area <= 0.0:
        raise ValueError("mesh area must be positive")
    scale = float(np.sqrt((4.0 * np.pi) / area))
    scaled = vertices * scale
    scaled_area = float(np.sum(_face_areas(scaled, faces)))
    return scaled, scaled_area


def _select_most_regular_face(
    vertices: np.ndarray,
    faces: np.ndarray,
    areas: np.ndarray,
) -> tuple[int, float]:
    tri = vertices[faces]
    l0 = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1)
    l1 = np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1)
    l2 = np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)
    rho = (l0 * l0 + l1 * l1 + l2 * l2) / (4.0 * np.sqrt(3.0) * np.maximum(areas, 1e-300))
    index = int(np.argmin(rho))
    return index, float(rho[index])


def _select_pole_face(
    vertices: np.ndarray,
    faces: np.ndarray,
    areas: np.ndarray,
    *,
    pole_face_index: int | None,
) -> tuple[int, float, str]:
    if pole_face_index is None:
        index, regularness = _select_most_regular_face(vertices, faces, areas)
        return index, regularness, "most_regular_face"
    index = int(pole_face_index)
    if index < 0 or index >= len(faces):
        raise ValueError(f"pole_face_index must be in [0, {len(faces)})")
    return index, _triangle_regularness(vertices, faces[index]), "explicit_face_index"


def _solve_tutte_disk(
    n_vertices: int,
    faces: np.ndarray,
    boundary_vertices: np.ndarray,
    *,
    boundary_radius: float,
) -> np.ndarray:
    boundary_vertices = np.asarray(boundary_vertices, dtype=np.int64)
    boundary_set = {int(v) for v in boundary_vertices}
    boundary_positions = _equilateral_boundary(boundary_radius)
    planar = np.zeros((n_vertices, 2), dtype=np.float64)
    for idx, vertex in enumerate(boundary_vertices):
        planar[int(vertex)] = boundary_positions[idx]

    neighbors: list[set[int]] = [set() for _ in range(n_vertices)]
    used_vertices: set[int] = set(int(v) for v in boundary_vertices)
    for tri in faces:
        for corner in range(3):
            a = int(tri[corner])
            b = int(tri[(corner + 1) % 3])
            neighbors[a].add(b)
            neighbors[b].add(a)
            used_vertices.add(a)
            used_vertices.add(b)

    interior = np.asarray(sorted(used_vertices - boundary_set), dtype=np.int64)
    if len(interior) == 0:
        return planar
    interior_lookup = {int(vertex): row for row, vertex in enumerate(interior)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros((len(interior), 2), dtype=np.float64)

    for row, vertex in enumerate(interior):
        nbrs = sorted(neighbors[int(vertex)])
        if not nbrs:
            raise ValueError(f"interior vertex {vertex} has no disk neighbors")
        weight = 1.0 / len(nbrs)
        rows.append(row)
        cols.append(row)
        data.append(1.0)
        for neighbor in nbrs:
            if neighbor in boundary_set:
                rhs[row] += weight * planar[neighbor]
            else:
                rows.append(row)
                cols.append(interior_lookup[neighbor])
                data.append(-weight)

    matrix = csr_matrix((data, (rows, cols)), shape=(len(interior), len(interior)))
    planar[interior, 0] = spsolve(matrix, rhs[:, 0])
    planar[interior, 1] = spsolve(matrix, rhs[:, 1])
    if not np.isfinite(planar).all():
        raise ValueError("Tutte solve produced non-finite planar coordinates")
    return planar


def _equilateral_boundary(radius: float) -> np.ndarray:
    r = float(radius)
    if r <= 0.0:
        raise ValueError("boundary_radius must be positive")
    return np.array(
        [
            [r, 0.0],
            [-0.5 * r, np.sqrt(3.0) * 0.5 * r],
            [-0.5 * r, -np.sqrt(3.0) * 0.5 * r],
        ],
        dtype=np.float64,
    )


def _planar_double_areas(planar_vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a = planar_vertices[faces[:, 0]]
    b = planar_vertices[faces[:, 1]]
    c = planar_vertices[faces[:, 2]]
    return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])


def _validate_stereographic_config(config: StereographicConfig) -> StereographicConfig:
    sigma = _validate_pole_sign(config.pole_sign)
    if config.pole_face_index is not None and int(config.pole_face_index) < 0:
        raise ValueError("pole_face_index must be nonnegative")
    if config.boundary_radius <= 0.0:
        raise ValueError("boundary_radius must be positive")
    if config.pole_area_threshold <= 0.0:
        raise ValueError("pole_area_threshold must be positive")
    if config.stage_a_max_iter < 0 or config.stage_b_max_iter < 0:
        raise ValueError("stage iteration counts must be nonnegative")
    if config.gradient_tol <= 0.0 or config.relative_energy_tol <= 0.0:
        raise ValueError("optimization tolerances must be positive")
    if config.max_backtracks < 1:
        raise ValueError("max_backtracks must be at least 1")
    if config.injectivity_max_depth < 0:
        raise ValueError("injectivity_max_depth must be nonnegative")
    if config.diagnostics_samples_per_face < 1:
        raise ValueError("diagnostics_samples_per_face must be positive")
    if sigma == config.pole_sign:
        return config
    return StereographicConfig(**{**config.__dict__, "pole_sign": sigma})


def _initialize_stereographic_disk(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: StereographicConfig,
) -> StereographicDiskInitialization:
    vertices, faces = _as_mesh_arrays(vertices, faces)
    _validate_closed_genus_zero(vertices, faces)
    scaled_vertices, scaled_area = _scale_to_unit_sphere_area(vertices, faces)
    areas = _face_areas(scaled_vertices, faces)
    pole_face_index, pole_regularness, pole_selection = _select_pole_face(
        scaled_vertices,
        faces,
        areas,
        pole_face_index=config.pole_face_index,
    )
    refined_vertices, refined_faces, refined_pole_index, refinement_steps = _refine_pole_triangle(
        scaled_vertices,
        faces,
        pole_face_index,
        area_threshold=config.pole_area_threshold,
    )
    refined_areas = _face_areas(refined_vertices, refined_faces)
    pole_face = refined_faces[refined_pole_index].copy()
    disk_face_indices = np.asarray(
        [idx for idx in range(len(refined_faces)) if idx != refined_pole_index],
        dtype=np.int64,
    )
    disk_faces = refined_faces[disk_face_indices].copy()
    boundary_vertices = pole_face.copy()

    candidates = [boundary_vertices, boundary_vertices[[0, 2, 1]]]
    best_planar = None
    best_min_area = -np.inf
    best_boundary = None
    for boundary_order in candidates:
        planar = _solve_tutte_disk(
            len(refined_vertices),
            disk_faces,
            boundary_order,
            boundary_radius=config.boundary_radius,
        )
        signed = _planar_double_areas(planar, disk_faces)
        min_signed = float(np.min(signed))
        if min_signed > best_min_area:
            best_planar = planar
            best_min_area = min_signed
            best_boundary = boundary_order.copy()
        if min_signed > 1e-14:
            return StereographicDiskInitialization(
                scaled_vertices=refined_vertices,
                faces=refined_faces,
                disk_faces=disk_faces,
                disk_face_indices=disk_face_indices,
                pole_face_index=int(refined_pole_index),
                pole_face=pole_face,
                input_pole_face_index=int(pole_face_index),
                pole_selection=pole_selection,
                boundary_vertices=boundary_order.copy(),
                planar_vertices=planar,
                scaled_area=scaled_area,
                pole_regularness=float(pole_regularness),
                min_planar_double_area=min_signed,
                pole_refinement_status=f"refined_{refinement_steps}_step(s)",
                original_vertex_count=len(vertices),
                pole_area=float(refined_areas[refined_pole_index]),
            )

    assert best_planar is not None and best_boundary is not None
    raise ValueError(
        "stereographic Tutte initialization produced non-positive disk face orientation; "
        f"best minimum signed double area was {best_min_area:.3e}"
    )


def _refine_pole_triangle(
    vertices: np.ndarray,
    faces: np.ndarray,
    pole_face_index: int,
    *,
    area_threshold: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    refined_vertices = np.asarray(vertices, dtype=np.float64).copy()
    refined_faces = np.asarray(faces, dtype=np.int64).copy()
    pole_idx = int(pole_face_index)
    steps = 0

    while float(_face_areas(refined_vertices, refined_faces[[pole_idx]])[0]) > area_threshold:
        tri = refined_faces[pole_idx]
        edge_local = _longest_local_edge(refined_vertices, tri)
        edge_start = int(tri[edge_local])
        edge_end = int(tri[(edge_local + 1) % 3])
        midpoint = 0.5 * (refined_vertices[edge_start] + refined_vertices[edge_end])
        midpoint_index = len(refined_vertices)
        refined_vertices = np.vstack([refined_vertices, midpoint])

        adjacent_idx = _find_adjacent_face(refined_faces, pole_idx, edge_start, edge_end)
        new_faces: list[np.ndarray] = []
        new_pole_candidates: list[int] = []
        for face_idx, face in enumerate(refined_faces):
            if face_idx == pole_idx:
                children = _split_oriented_face(face, edge_start, edge_end, midpoint_index)
                for child in children:
                    if len(new_pole_candidates) < 2:
                        new_pole_candidates.append(len(new_faces))
                    new_faces.append(child)
            elif face_idx == adjacent_idx:
                children = _split_oriented_face(face, edge_start, edge_end, midpoint_index)
                new_faces.extend(children)
            else:
                new_faces.append(face.copy())
        refined_faces = np.asarray(new_faces, dtype=np.int64)
        candidate_faces = refined_faces[np.asarray(new_pole_candidates, dtype=np.int64)]
        rhos = [
            _triangle_regularness(refined_vertices, candidate_faces[0]),
            _triangle_regularness(refined_vertices, candidate_faces[1]),
        ]
        pole_idx = int(new_pole_candidates[int(np.argmin(rhos))])
        steps += 1
        if steps > 10_000:
            raise RuntimeError("pole refinement exceeded 10000 steps")
    return refined_vertices, refined_faces, pole_idx, steps


def _longest_local_edge(vertices: np.ndarray, face: np.ndarray) -> int:
    lengths = []
    for local in range(3):
        a = vertices[int(face[local])]
        b = vertices[int(face[(local + 1) % 3])]
        lengths.append(float(np.linalg.norm(b - a)))
    return int(np.argmax(lengths))


def _find_adjacent_face(faces: np.ndarray, excluded_face: int, a: int, b: int) -> int:
    target = {int(a), int(b)}
    matches = []
    for idx, face in enumerate(faces):
        if idx == excluded_face:
            continue
        if target.issubset({int(x) for x in face}):
            matches.append(idx)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one adjacent face across edge {(a, b)}, found {len(matches)}")
    return int(matches[0])


def _split_oriented_face(face: np.ndarray, a: int, b: int, midpoint_index: int) -> list[np.ndarray]:
    values = [int(x) for x in face]
    for local in range(3):
        start = values[local]
        end = values[(local + 1) % 3]
        if {start, end} == {int(a), int(b)}:
            opposite = values[(local + 2) % 3]
            return [
                np.asarray([start, midpoint_index, opposite], dtype=np.int64),
                np.asarray([midpoint_index, end, opposite], dtype=np.int64),
            ]
    raise ValueError("face does not contain the requested split edge")


def _triangle_regularness(vertices: np.ndarray, face: np.ndarray) -> float:
    tri = vertices[np.asarray(face, dtype=np.int64)]
    area = 0.5 * np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
    l0 = np.linalg.norm(tri[1] - tri[0])
    l1 = np.linalg.norm(tri[2] - tri[1])
    l2 = np.linalg.norm(tri[0] - tri[2])
    return float((l0 * l0 + l1 * l1 + l2 * l2) / (4.0 * np.sqrt(3.0) * max(area, 1e-300)))


def _build_quadratic_control_mesh(faces: np.ndarray, planar_vertices: np.ndarray) -> _BezierControlMesh:
    edge_lookup: dict[tuple[int, int], int] = {}
    controls = [np.asarray(p, dtype=np.float64).copy() for p in planar_vertices]
    face_control_indices: list[list[int]] = []
    for face in faces:
        v0, v1, v2 = [int(x) for x in face]
        e01 = _edge_control_index(edge_lookup, controls, planar_vertices, v0, v1)
        e02 = _edge_control_index(edge_lookup, controls, planar_vertices, v0, v2)
        e12 = _edge_control_index(edge_lookup, controls, planar_vertices, v1, v2)
        face_control_indices.append([v0, v1, v2, e01, e02, e12])
    return _BezierControlMesh(
        vertex_count=len(planar_vertices),
        controls=np.asarray(controls, dtype=np.float64),
        face_control_indices=np.asarray(face_control_indices, dtype=np.int64),
        edge_lookup=edge_lookup,
    )


def _edge_control_index(
    edge_lookup: dict[tuple[int, int], int],
    controls: list[np.ndarray],
    planar_vertices: np.ndarray,
    a: int,
    b: int,
) -> int:
    key = (int(a), int(b)) if int(a) < int(b) else (int(b), int(a))
    if key not in edge_lookup:
        edge_lookup[key] = len(controls)
        controls.append(0.5 * (planar_vertices[key[0]] + planar_vertices[key[1]]))
    return edge_lookup[key]


def _optimize_stereographic_controls(
    init: StereographicDiskInitialization,
    control_mesh: _BezierControlMesh,
    config: StereographicConfig,
) -> tuple[np.ndarray, dict]:
    controls = _precondition_global_scale(init, control_mesh, control_mesh.controls.copy())
    total_iterations = 0
    termination_reasons: list[str] = []
    stage_energies: dict[str, float] = {}

    controls, info_a = _run_newton_stage(
        init,
        control_mesh,
        controls,
        config,
        stage="symmetric_dirichlet",
        max_iter=config.stage_a_max_iter,
        alpha=0.0,
    )
    total_iterations += info_a["iterations"]
    termination_reasons.append(f"stage_a:{info_a['termination_reason']}")
    stage_energies["stage_a_final_energy"] = info_a["final_energy"]

    if config.stage_b_max_iter > 0:
        face_energies = _face_energy_values(init, control_mesh, controls)
        finite_face_energies = face_energies[np.isfinite(face_energies)]
        if len(finite_face_energies) == 0:
            alpha = 0.0
        else:
            alpha = min(5.0, max(0.0, 10.0 / max(float(np.max(finite_face_energies)), 1e-12)))
        controls, info_b = _run_newton_stage(
            init,
            control_mesh,
            controls,
            config,
            stage="exponential_sd",
            max_iter=config.stage_b_max_iter,
            alpha=alpha,
        )
        total_iterations += info_b["iterations"]
        termination_reasons.append(f"stage_b:{info_b['termination_reason']}")
        stage_energies["stage_b_final_energy"] = info_b["final_energy"]
        stage_energies["stage_b_alpha"] = alpha

    final_energy = _assemble_energy_gradient_hessian(
        init,
        control_mesh,
        controls,
        energy_kind="symmetric_dirichlet",
        alpha=0.0,
        need_derivatives=False,
    )[0]
    if any("converged" in item for item in termination_reasons):
        status = "converged"
    elif any("line_search_failed" in item for item in termination_reasons):
        status = "line_search_failed"
    else:
        status = "max_iter"
    return controls, {
        "optimization_status": status,
        "newton_iterations": int(total_iterations),
        "termination_reason": ";".join(termination_reasons),
        "final_symmetric_dirichlet_energy": float(final_energy),
        **stage_energies,
    }


def _precondition_global_scale(
    init: StereographicDiskInitialization,
    control_mesh: _BezierControlMesh,
    controls: np.ndarray,
) -> np.ndarray:
    scales = np.geomspace(0.5, 32.0, 24)
    best_energy = np.inf
    best_controls = controls
    for scale in scales:
        candidate = controls * float(scale)
        if not _all_patches_injective(candidate, control_mesh, max_depth=2):
            continue
        energy = _assemble_energy_gradient_hessian(
            init,
            control_mesh,
            candidate,
            energy_kind="symmetric_dirichlet",
            alpha=0.0,
            need_derivatives=False,
        )[0]
        if np.isfinite(energy) and energy < best_energy:
            best_energy = float(energy)
            best_controls = candidate
    return best_controls.copy()


def _run_newton_stage(
    init: StereographicDiskInitialization,
    control_mesh: _BezierControlMesh,
    controls: np.ndarray,
    config: StereographicConfig,
    *,
    stage: str,
    max_iter: int,
    alpha: float,
) -> tuple[np.ndarray, dict]:
    if max_iter == 0:
        energy = _assemble_energy_gradient_hessian(
            init,
            control_mesh,
            controls,
            energy_kind="symmetric_dirichlet" if stage == "symmetric_dirichlet" else "exponential_sd",
            alpha=alpha,
            need_derivatives=False,
        )[0]
        return controls, {"iterations": 0, "termination_reason": "skipped", "final_energy": float(energy)}

    current = controls.copy()
    previous_energy = np.inf
    termination = "max_iter"
    final_energy = np.inf
    iterations = 0
    for iteration in range(max_iter):
        energy_kind = "symmetric_dirichlet" if stage == "symmetric_dirichlet" else "exponential_sd"
        energy, gradient, hessian = _assemble_energy_gradient_hessian(
            init,
            control_mesh,
            current,
            energy_kind=energy_kind,
            alpha=alpha,
            need_derivatives=True,
        )
        final_energy = float(energy)
        if not np.isfinite(energy) or gradient is None or hessian is None:
            termination = "invalid_energy"
            break
        grad_norm = float(np.linalg.norm(gradient))
        if grad_norm < config.gradient_tol:
            termination = "converged_grad"
            break
        if np.isfinite(previous_energy):
            rel = abs(previous_energy - energy) / max(1.0, abs(previous_energy))
            if rel < config.relative_energy_tol:
                termination = "converged_energy"
                break
        direction = _solve_newton_direction(hessian, gradient)
        directional_derivative = float(np.dot(gradient, direction))
        if not np.isfinite(directional_derivative) or directional_derivative >= 0.0:
            direction = -gradient / max(1.0, grad_norm)
            directional_derivative = float(np.dot(gradient, direction))
        flat = current.reshape(-1)
        t_inj = _max_injective_step(flat, direction, control_mesh, config.injectivity_max_depth)
        step = min(1.0, 0.95 * t_inj)
        accepted = False
        for _ in range(config.max_backtracks):
            if step <= 1e-14:
                break
            candidate = (flat + step * direction).reshape(current.shape)
            if not _all_patches_injective(candidate, control_mesh, config.injectivity_max_depth):
                step *= 0.5
                continue
            candidate_energy = _assemble_energy_gradient_hessian(
                init,
                control_mesh,
                candidate,
                energy_kind=energy_kind,
                alpha=alpha,
                need_derivatives=False,
            )[0]
            if np.isfinite(candidate_energy) and candidate_energy <= energy + config.armijo_c1 * step * directional_derivative:
                current = candidate
                previous_energy = float(energy)
                final_energy = float(candidate_energy)
                accepted = True
                iterations = iteration + 1
                break
            step *= 0.5
        if not accepted:
            termination = "line_search_failed"
            break
    return current, {
        "iterations": int(iterations),
        "termination_reason": termination,
        "final_energy": float(final_energy),
    }


def _solve_newton_direction(hessian: csr_matrix, gradient: np.ndarray) -> np.ndarray:
    n = len(gradient)
    regularized = hessian + coo_matrix(
        (np.full(n, 1e-10, dtype=np.float64), (np.arange(n), np.arange(n))),
        shape=(n, n),
    ).tocsr()
    try:
        direction = spsolve(regularized, -gradient)
    except Exception:
        direction = np.linalg.lstsq(regularized.toarray(), -gradient, rcond=None)[0]
    if not np.isfinite(direction).all():
        direction = -gradient / max(1.0, float(np.linalg.norm(gradient)))
    return np.asarray(direction, dtype=np.float64)


def _max_injective_step(
    flat_controls: np.ndarray,
    direction: np.ndarray,
    control_mesh: _BezierControlMesh,
    max_depth: int,
) -> float:
    shape = control_mesh.controls.shape
    candidate = (flat_controls + direction).reshape(shape)
    if _all_patches_injective(candidate, control_mesh, max_depth):
        return 1.0
    lo = 0.0
    hi = 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        candidate = (flat_controls + mid * direction).reshape(shape)
        if _all_patches_injective(candidate, control_mesh, max_depth):
            lo = mid
        else:
            hi = mid
    return float(lo)


def _assemble_energy_gradient_hessian(
    init: StereographicDiskInitialization,
    control_mesh: _BezierControlMesh,
    controls: np.ndarray,
    *,
    energy_kind: str,
    alpha: float,
    need_derivatives: bool,
) -> tuple[float, np.ndarray | None, csr_matrix | None]:
    points, weights = lyness_jespersen_rule_10()
    n_vars = controls.size
    gradient = np.zeros(n_vars, dtype=np.float64) if need_derivatives else None
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    total = 0.0

    for face, control_ids in zip(init.disk_faces, control_mesh.face_control_indices):
        j_f = _source_triangle_jacobian(init.scaled_vertices, face)
        local_controls = controls[control_ids]
        local = _local_face_energy_derivatives(local_controls, j_f, points, weights, need_derivatives)
        if local is None:
            return np.inf, gradient, None
        face_energy, local_grad, local_hess = local
        if energy_kind == "symmetric_dirichlet":
            contribution = 0.25 * face_energy
            scale_grad = 0.25
            local_contribution_hess = None if local_hess is None else 0.25 * local_hess
        elif energy_kind == "exponential_sd":
            exp_arg = float(alpha) * face_energy / 4.0
            if exp_arg > 700.0:
                return np.inf, gradient, None
            contribution = float(np.exp(exp_arg))
            scale_grad = contribution * float(alpha) / 4.0
            if local_hess is None or local_grad is None:
                local_contribution_hess = None
            else:
                a = float(alpha) / 4.0
                local_contribution_hess = contribution * (a * local_hess + a * a * np.outer(local_grad, local_grad))
        else:
            raise ValueError(f"unknown energy kind {energy_kind}")
        total += contribution
        if need_derivatives and local_grad is not None and local_contribution_hess is not None:
            var_ids = _control_variable_ids(control_ids)
            gradient[var_ids] += scale_grad * local_grad
            h_pd = _make_positive_definite(local_contribution_hess)
            rr, cc = np.meshgrid(var_ids, var_ids, indexing="ij")
            rows.extend(rr.reshape(-1).tolist())
            cols.extend(cc.reshape(-1).tolist())
            data.extend(h_pd.reshape(-1).tolist())

    if not need_derivatives:
        return float(total), None, None
    hessian = coo_matrix((data, (rows, cols)), shape=(n_vars, n_vars)).tocsr()
    return float(total), gradient, hessian


def _face_energy_values(
    init: StereographicDiskInitialization,
    control_mesh: _BezierControlMesh,
    controls: np.ndarray,
) -> np.ndarray:
    points, weights = lyness_jespersen_rule_10()
    values = []
    for face, control_ids in zip(init.disk_faces, control_mesh.face_control_indices):
        j_f = _source_triangle_jacobian(init.scaled_vertices, face)
        local = _local_face_energy_derivatives(controls[control_ids], j_f, points, weights, False)
        values.append(np.inf if local is None else float(local[0]))
    return np.asarray(values, dtype=np.float64)


def _control_variable_ids(control_ids: np.ndarray) -> np.ndarray:
    ids = []
    for control_id in control_ids:
        ids.extend([2 * int(control_id), 2 * int(control_id) + 1])
    return np.asarray(ids, dtype=np.int64)


def _source_triangle_jacobian(vertices: np.ndarray, face: np.ndarray) -> np.ndarray:
    x0, x1, x2 = vertices[np.asarray(face, dtype=np.int64)]
    edge = x1 - x0
    length = float(np.linalg.norm(edge))
    if length <= 1e-15:
        raise ValueError("degenerate source face edge")
    e1 = edge / length
    tmp = x2 - x0
    x2_local = float(np.dot(tmp, e1))
    y2_sq = max(float(np.dot(tmp, tmp) - x2_local * x2_local), 0.0)
    y2 = float(np.sqrt(y2_sq))
    if y2 <= 1e-15:
        raise ValueError("degenerate source face area")
    return np.array(
        [
            [1.0 / length, -x2_local / (length * y2)],
            [0.0, 1.0 / y2],
        ],
        dtype=np.float64,
    )


def _quadratic_basis(uv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, v, w = _reference_uv(uv)
    basis = np.array(
        [w * w, u * u, v * v, 2.0 * u * w, 2.0 * v * w, 2.0 * u * v],
        dtype=np.float64,
    )
    d_du = np.array(
        [-2.0 * w, 2.0 * u, 0.0, 2.0 * (w - u), -2.0 * v, 2.0 * v],
        dtype=np.float64,
    )
    d_dv = np.array(
        [-2.0 * w, 0.0, 2.0 * v, -2.0 * u, 2.0 * (w - v), 2.0 * u],
        dtype=np.float64,
    )
    return basis, d_du, d_dv


def _local_face_energy_derivatives(
    local_controls: np.ndarray,
    j_f: np.ndarray,
    points: np.ndarray,
    weights: np.ndarray,
    need_derivatives: bool,
) -> tuple[float, np.ndarray | None, np.ndarray | None] | None:
    x = np.asarray(local_controls, dtype=np.float64).reshape(12)
    total = 0.0
    grad_total = np.zeros(12, dtype=np.float64) if need_derivatives else None
    hess_total = np.zeros((12, 12), dtype=np.float64) if need_derivatives else None
    for uv, weight in zip(points, weights):
        basis, d_du, d_dv = _quadratic_basis(uv)
        value = _integrand_derivatives(x, basis, d_du, d_dv, j_f, need_derivatives)
        if value is None:
            return None
        energy, grad, hess = value
        total += float(weight) * energy
        if need_derivatives and grad is not None and hess is not None:
            grad_total += float(weight) * grad
            hess_total += float(weight) * hess
    return total, grad_total, hess_total


def _integrand_derivatives(
    x: np.ndarray,
    basis: np.ndarray,
    d_du: np.ndarray,
    d_dv: np.ndarray,
    j_f: np.ndarray,
    need_derivatives: bool,
) -> tuple[float, np.ndarray | None, np.ndarray | None] | None:
    controls = x.reshape(6, 2)
    point = basis @ controls
    jb = np.array(
        [
            [float(np.dot(d_du, controls[:, 0])), float(np.dot(d_dv, controls[:, 0]))],
            [float(np.dot(d_du, controls[:, 1])), float(np.dot(d_dv, controls[:, 1]))],
        ],
        dtype=np.float64,
    )
    c_mat = jb @ j_f
    denom = 1.0 + float(np.dot(point, point))
    scale = 2.0 / denom
    a_mat = scale * c_mat
    a = a_mat.reshape(4)
    det = float(a[0] * a[3] - a[1] * a[2])
    if det <= 1e-14:
        return None
    norm2 = float(np.dot(a, a))
    energy = norm2 + norm2 / (det * det)
    if not need_derivatives:
        return energy, None, None

    d_point = np.zeros((2, 12), dtype=np.float64)
    d_point[0, 0::2] = basis
    d_point[1, 1::2] = basis

    d_jb = np.zeros((2, 2, 12), dtype=np.float64)
    d_jb[0, 0, 0::2] = d_du
    d_jb[0, 1, 0::2] = d_dv
    d_jb[1, 0, 1::2] = d_du
    d_jb[1, 1, 1::2] = d_dv
    d_c = np.einsum("abk,bc->ack", d_jb, j_f).reshape(4, 12)

    x_planar, y_planar = float(point[0]), float(point[1])
    ds_d_point = np.array([-4.0 * x_planar / (denom * denom), -4.0 * y_planar / (denom * denom)])
    h_scale_point = (
        -4.0 * np.eye(2, dtype=np.float64) / (denom * denom)
        + 16.0 * np.outer(point, point) / (denom * denom * denom)
    )
    d_scale = ds_d_point @ d_point
    h_scale = d_point.T @ h_scale_point @ d_point

    d_a = scale * d_c + np.outer(c_mat.reshape(4), d_scale)
    h_a = np.empty((4, 12, 12), dtype=np.float64)
    for idx in range(4):
        c_grad = d_c[idx]
        h_a[idx] = (
            c_mat.reshape(4)[idx] * h_scale
            + np.outer(d_scale, c_grad)
            + np.outer(c_grad, d_scale)
        )

    grad_a, hess_a = _sd_integrand_grad_hess_wrt_a(a, det, norm2)
    grad_x = d_a.T @ grad_a
    hess_x = d_a.T @ hess_a @ d_a
    for idx in range(4):
        hess_x += grad_a[idx] * h_a[idx]
    hess_x = 0.5 * (hess_x + hess_x.T)
    return energy, grad_x, hess_x


def _sd_integrand_grad_hess_wrt_a(
    a: np.ndarray,
    det: float,
    norm2: float,
) -> tuple[np.ndarray, np.ndarray]:
    dn = 2.0 * a
    hn = 2.0 * np.eye(4, dtype=np.float64)
    dd = np.array([a[3], -a[2], -a[1], a[0]], dtype=np.float64)
    hd = np.zeros((4, 4), dtype=np.float64)
    hd[0, 3] = hd[3, 0] = 1.0
    hd[1, 2] = hd[2, 1] = -1.0
    inv_det2 = 1.0 / (det * det)
    h = 1.0 + inv_det2
    dh = -2.0 * dd / (det**3)
    hh = 6.0 * np.outer(dd, dd) / (det**4) - 2.0 * hd / (det**3)
    grad = h * dn + norm2 * dh
    hess = h * hn + np.outer(dn, dh) + np.outer(dh, dn) + norm2 * hh
    return grad, hess


def _make_positive_definite(hessian: np.ndarray) -> np.ndarray:
    sym = 0.5 * (hessian + hessian.T)
    values, vectors = np.linalg.eigh(sym)
    positive = values[values > 0.0]
    if len(positive) == 0:
        floor = 1e-9
    else:
        floor = 1e-9 * max(1.0, float(np.mean(np.abs(positive))))
    clamped = np.maximum(values, floor)
    return (vectors * clamped) @ vectors.T


def _patch_from_controls(controls: np.ndarray, control_ids: np.ndarray) -> QuadraticBezierPatch:
    local = controls[np.asarray(control_ids, dtype=np.int64)]
    return QuadraticBezierPatch(
        b002=local[0].copy(),
        b200=local[1].copy(),
        b020=local[2].copy(),
        b101=local[3].copy(),
        b011=local[4].copy(),
        b110=local[5].copy(),
    )


def _all_patches_injective(
    controls: np.ndarray,
    control_mesh: _BezierControlMesh,
    max_depth: int,
) -> bool:
    for control_ids in control_mesh.face_control_indices:
        patch = _patch_from_controls(controls, control_ids)
        ok, _ = _patch_injective_bernstein(patch, max_depth=max_depth)
        if not ok:
            return False
    return True


def _patch_injective_bernstein(
    patch: QuadraticBezierPatch,
    *,
    max_depth: int,
    eps_det: float = 1e-12,
) -> tuple[bool, float]:
    root = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    return _patch_injective_on_subtriangle(patch, root, 0, max_depth, eps_det)


def _patch_injective_on_subtriangle(
    patch: QuadraticBezierPatch,
    triangle: np.ndarray,
    depth: int,
    max_depth: int,
    eps_det: float,
) -> tuple[bool, float]:
    coeffs = _det_bernstein_coefficients_on_subtriangle(patch, triangle)
    min_coeff = float(np.min(coeffs))
    if min_coeff > eps_det:
        return True, min_coeff
    if depth >= max_depth:
        return False, min_coeff
    a, b, c = triangle
    ab = 0.5 * (a + b)
    bc = 0.5 * (b + c)
    ca = 0.5 * (c + a)
    children = [
        np.asarray([a, ab, ca], dtype=np.float64),
        np.asarray([ab, b, bc], dtype=np.float64),
        np.asarray([ca, bc, c], dtype=np.float64),
        np.asarray([ab, bc, ca], dtype=np.float64),
    ]
    mins = []
    for child in children:
        ok, child_min = _patch_injective_on_subtriangle(patch, child, depth + 1, max_depth, eps_det)
        mins.append(child_min)
        if not ok:
            return False, float(min(mins))
    return True, float(min(mins))


def _det_bernstein_coefficients_on_subtriangle(
    patch: QuadraticBezierPatch,
    triangle: np.ndarray,
) -> np.ndarray:
    a, b, c = triangle
    ab = 0.5 * (a + b)
    ac = 0.5 * (a + c)
    bc = 0.5 * (b + c)
    d002 = _patch_det_jacobian_at(patch, a)
    d200 = _patch_det_jacobian_at(patch, b)
    d020 = _patch_det_jacobian_at(patch, c)
    vab = _patch_det_jacobian_at(patch, ab)
    vac = _patch_det_jacobian_at(patch, ac)
    vbc = _patch_det_jacobian_at(patch, bc)
    d101 = 2.0 * vab - 0.5 * (d002 + d200)
    d011 = 2.0 * vac - 0.5 * (d002 + d020)
    d110 = 2.0 * vbc - 0.5 * (d200 + d020)
    return np.array([d002, d200, d020, d101, d011, d110], dtype=np.float64)


def _patch_det_jacobian_at(patch: QuadraticBezierPatch, uv: np.ndarray) -> float:
    _, jac = patch.evaluate_with_jacobian(np.asarray(uv, dtype=np.float64))
    return float(np.linalg.det(jac))


def _compute_stereographic_diagnostics(
    init: StereographicDiskInitialization,
    control_mesh: _BezierControlMesh,
    controls: np.ndarray,
    config: StereographicConfig,
    sphere: np.ndarray,
) -> dict:
    min_det = np.inf
    uncertified = 0
    for control_ids in control_mesh.face_control_indices:
        patch = _patch_from_controls(controls, control_ids)
        ok, det_value = _patch_injective_bernstein(patch, max_depth=config.injectivity_max_depth)
        min_det = min(min_det, det_value)
        if not ok:
            uncertified += 1

    samples = _reference_triangle_samples(config.diagnostics_samples_per_face)
    sigma_values: list[float] = []
    sd_values: list[float] = []
    for face, control_ids in zip(init.disk_faces, control_mesh.face_control_indices):
        patch = _patch_from_controls(controls, control_ids)
        j_f = _source_triangle_jacobian(init.scaled_vertices, face)
        for uv in samples:
            value, jb = patch.evaluate_with_jacobian(uv)
            denom = 1.0 + float(np.dot(value, value))
            a_mat = (2.0 / denom) * (jb @ j_f)
            singular = np.linalg.svd(a_mat, compute_uv=False)
            gamma = float(np.min(singular))
            gamma_big = float(np.max(singular))
            if gamma <= 0.0:
                continue
            sigma_values.append(max(gamma_big, 1.0 / gamma))
            sd_values.append(0.25 * (gamma_big * gamma_big + gamma * gamma + gamma_big**-2 + gamma**-2))

    z = sphere[:, 2]
    return {
        "min_certified_det_jacobian": float(min_det),
        "uncertified_patch_count": int(uncertified),
        "sphere_z_range": float(np.max(z) - np.min(z)),
        "south_polar_cap_fraction_z_lt_neg_0_95": float(np.mean(z < -0.95)),
        "max_sigma_iso": float(np.max(sigma_values) if sigma_values else np.inf),
        "average_sigma_iso": float(np.mean(sigma_values) if sigma_values else np.inf),
        "max_symmetric_dirichlet": float(np.max(sd_values) if sd_values else np.inf),
        "average_symmetric_dirichlet": float(np.mean(sd_values) if sd_values else np.inf),
        "diagnostic_samples_per_face": int(config.diagnostics_samples_per_face),
    }


def _reference_triangle_samples(count: int) -> np.ndarray:
    samples = []
    for idx in range(int(count)):
        u = (idx + 0.5) / float(count)
        v = _radical_inverse_base2(idx + 1)
        if u + v > 1.0:
            u = 1.0 - u
            v = 1.0 - v
        samples.append([u, v])
    return np.asarray(samples, dtype=np.float64)


def _radical_inverse_base2(index: int) -> float:
    inv = 0.5
    value = 0.0
    while index > 0:
        if index & 1:
            value += inv
        index >>= 1
        inv *= 0.5
    return value


def _evaluate_pole_exterior_planar(
    pole_face: np.ndarray,
    barycentric: np.ndarray,
    vertex_controls: np.ndarray,
    controls: np.ndarray,
    edge_lookup: dict[tuple[int, int], int],
) -> np.ndarray | None:
    bary = np.asarray(barycentric, dtype=np.float64)
    if np.allclose(bary, np.array([1.0 / 3.0] * 3), atol=1e-14):
        return None
    missing = int(np.argmin(bary))
    outer0 = (missing + 1) % 3
    outer1 = (missing + 2) % 3
    alpha = 3.0 * bary[missing]
    u = bary[outer0] - bary[missing]
    v = bary[outer1] - bary[missing]
    if alpha < -1e-10 or u < -1e-10 or v < -1e-10:
        missing = int(np.argmin(bary + 1e-14 * np.arange(3)))
        outer0 = (missing + 1) % 3
        outer1 = (missing + 2) % 3
        alpha = 3.0 * bary[missing]
        u = bary[outer0] - bary[missing]
        v = bary[outer1] - bary[missing]
    del alpha
    r = float(u + v)
    if r <= 1e-14:
        return None
    a = int(pole_face[outer0])
    b = int(pole_face[outer1])
    key = (a, b) if a < b else (b, a)
    edge_control = controls[edge_lookup[key]]
    center = np.mean(vertex_controls[np.asarray(pole_face, dtype=np.int64)], axis=0)
    subpatch = QuadraticBezierPatch(
        b002=center,
        b200=vertex_controls[a],
        b020=vertex_controls[b],
        b101=0.5 * (center + vertex_controls[a]),
        b011=0.5 * (center + vertex_controls[b]),
        b110=edge_control,
    )
    p_param = np.array([max(0.0, u), max(0.0, v)], dtype=np.float64)
    p_star_param = p_param / r
    p_star, jac = subpatch.evaluate_with_jacobian(p_star_param)
    norm = float(np.linalg.norm(p_param))
    if norm <= 1e-14:
        return None
    direction = p_param / norm
    tangent = jac @ direction
    ratio = 1.0 / r - 1.0
    return p_star + ratio * tangent
