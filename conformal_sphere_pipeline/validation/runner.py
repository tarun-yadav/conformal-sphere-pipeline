"""Cached spherical validation runs for reproducible atlas evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from conformal_sphere_pipeline.spherical.stereographic import (
    StereographicConfig,
    compute_stereographic_parameterization,
)

VALIDATION_PARAMETERIZERS = {"stereographic", "conformal"}


@dataclass(frozen=True)
class SphericalValidationRunConfig:
    parameterization_method: str = "stereographic"
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

    def to_stereographic_config(self) -> StereographicConfig:
        values = asdict(self)
        values.pop("parameterization_method")
        return StereographicConfig(**values)


@dataclass(frozen=True)
class SphericalValidationMap:
    name: str
    sphere: np.ndarray
    info: dict
    input_hash: str
    config_hash: str
    sphere_path: Path
    manifest_path: Path
    cache_hit: bool


def mesh_sha256(vertices: np.ndarray, faces: np.ndarray) -> str:
    vertices_array = np.ascontiguousarray(np.asarray(vertices, dtype="<f8"))
    faces_array = np.ascontiguousarray(np.asarray(faces, dtype="<i8"))
    if vertices_array.ndim != 2 or vertices_array.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if faces_array.ndim != 2 or faces_array.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3)")
    digest = hashlib.sha256()
    digest.update(str(vertices_array.shape).encode("ascii"))
    digest.update(vertices_array.tobytes(order="C"))
    digest.update(str(faces_array.shape).encode("ascii"))
    digest.update(faces_array.tobytes(order="C"))
    return digest.hexdigest()


def config_sha256(config: SphericalValidationRunConfig) -> str:
    encoded = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_spherical_validation_map(
    name: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    cache_dir: Path | str,
    *,
    config: SphericalValidationRunConfig | None = None,
) -> SphericalValidationMap:
    cfg = config or SphericalValidationRunConfig()
    if cfg.parameterization_method not in VALIDATION_PARAMETERIZERS:
        raise ValueError("parameterization_method must be stereographic or conformal")
    vertices_array = np.ascontiguousarray(np.asarray(vertices, dtype=np.float64))
    faces_array = np.ascontiguousarray(np.asarray(faces, dtype=np.int64))
    input_hash = mesh_sha256(vertices_array, faces_array)
    cfg_hash = config_sha256(cfg)
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    stem = f"{_safe_name(name)}-{input_hash[:12]}-{cfg_hash[:12]}"
    sphere_path = cache_root / f"{stem}.npz"
    manifest_path = cache_root / f"{stem}.json"

    if sphere_path.exists() and manifest_path.exists():
        data = np.load(sphere_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return SphericalValidationMap(
            name=str(manifest["name"]),
            sphere=np.asarray(data["sphere"], dtype=np.float64),
            info=dict(manifest["info"]),
            input_hash=str(manifest["input_hash"]),
            config_hash=str(manifest["config_hash"]),
            sphere_path=sphere_path,
            manifest_path=manifest_path,
            cache_hit=True,
        )

    sphere, info = _compute_parameterization(vertices_array, faces_array, cfg)
    manifest = {
        "name": name,
        "input_hash": input_hash,
        "config_hash": cfg_hash,
        "config": asdict(cfg),
        "vertex_count": int(len(vertices_array)),
        "face_count": int(len(faces_array)),
        "info": _json_safe(info),
    }
    np.savez_compressed(sphere_path, sphere=np.asarray(sphere, dtype=np.float64))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return SphericalValidationMap(
        name=name,
        sphere=np.asarray(sphere, dtype=np.float64),
        info=dict(manifest["info"]),
        input_hash=input_hash,
        config_hash=cfg_hash,
        sphere_path=sphere_path,
        manifest_path=manifest_path,
        cache_hit=False,
    )


def _compute_parameterization(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: SphericalValidationRunConfig,
) -> tuple[np.ndarray, dict]:
    if config.parameterization_method == "stereographic":
        sphere, info = compute_stereographic_parameterization(
            vertices,
            faces,
            return_info=True,
            config=config.to_stereographic_config(),
        )
        info = dict(info)
        info.setdefault("injectivity_certification", "adaptive_patch_jacobian")
    else:
        from conformal_sphere_pipeline.spherical.parameterize import parameterize_sphere

        result = parameterize_sphere(vertices, faces, method="conformal")
        sphere = np.asarray(result.sphere, dtype=np.float64)
        info = {
            **dict(result.info),
            "injectivity_certification": "unavailable",
            "uncertified_patch_count": int(len(faces)),
            "min_certified_det_jacobian": 0.0,
        }
    norms = np.linalg.norm(np.asarray(sphere, dtype=np.float64), axis=1)
    info.setdefault("max_sphere_norm_error", float(np.max(np.abs(norms - 1.0))))
    info.setdefault("sphere_z_range", float(np.ptp(np.asarray(sphere, dtype=np.float64)[:, 2])))
    info.setdefault("parameterization_method", config.parameterization_method)
    return np.asarray(sphere, dtype=np.float64), info


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip())
    return safe.strip("-") or "mesh"


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value
