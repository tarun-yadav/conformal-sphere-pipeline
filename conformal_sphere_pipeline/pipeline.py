"""End-to-end conformal sphere pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path

import numpy as np
import yaml

from .boundary import extract_boundary_loops
from .io import load_triangle_mesh, write_json, write_mesh
from .mesh_qc import validate_triangle_mesh
from .spherical.features import compute_face_features
from .spherical.harmonics import compute_complex_sh_coeffs_from_faces
from .spherical.mobius import mobius_center
from .spherical.orientation import canonical_orient_sphere
from .spherical.parameterize import PARAMETERIZATION_METHODS, parameterize_sphere
from .spherical.quality import sphere_norm_metrics
from .spherical.resample import resample_channels_to_equiangular
from .virtual_buffer import add_virtual_boundary_buffers

DISPLAY_SCHEMA_VERSION = "canonical-sphere.display.v2"
AUGMENTED_QC_SCHEMA_VERSION = "canonical-sphere.augmented-qc.v1"
MESH_DISPLAY_SCHEMA_VERSION = "canonical-sphere.physical-mesh-display.v1"


@dataclass
class ConformalSphereConfig:
    """Runtime configuration for one mesh canonicalization call."""

    virtual_buffer_enabled: bool = True
    virtual_buffer_rings: int = 12
    virtual_buffer_length_factor: float = 2.5
    virtual_buffer_min_length_factor_bbox: float = 0.03
    virtual_buffer_apex_radius_fraction: float = 0.02
    parameterizer: str = "auto"
    mobius_tol: float = 1e-7
    mobius_max_iters: int = 100
    lmax_orientation: int = 16
    nlat: int = 128
    nlon: int = 256
    orientation_signal: str = "log_conformal_factor"
    fail_on_warning: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path | None) -> "ConformalSphereConfig":
        if path is None:
            return cls()
        data = yaml.safe_load(cls._read_yaml_text(path)) or {}
        cfg = cls()
        virtual = data.get("virtual_buffer", {})
        orientation = data.get("orientation", {})
        features = data.get("features", {})
        parameterization = data.get("parameterization", {})
        mobius = data.get("mobius", {})
        cfg.virtual_buffer_enabled = bool(virtual.get("enabled", cfg.virtual_buffer_enabled))
        cfg.virtual_buffer_rings = int(virtual.get("rings", cfg.virtual_buffer_rings))
        cfg.virtual_buffer_length_factor = float(virtual.get("length_factor", cfg.virtual_buffer_length_factor))
        cfg.virtual_buffer_min_length_factor_bbox = float(
            virtual.get("min_length_factor_bbox", cfg.virtual_buffer_min_length_factor_bbox)
        )
        cfg.virtual_buffer_apex_radius_fraction = float(
            virtual.get("apex_radius_fraction", cfg.virtual_buffer_apex_radius_fraction)
        )
        cfg.lmax_orientation = int(orientation.get("lmax_orientation", cfg.lmax_orientation))
        cfg.nlat = int(features.get("nlat", cfg.nlat))
        cfg.nlon = int(features.get("nlon", cfg.nlon))
        cfg.mobius_tol = float(mobius.get("centroid_tol", cfg.mobius_tol))
        cfg.mobius_max_iters = int(mobius.get("max_iters", cfg.mobius_max_iters))
        if "method" in parameterization:
            method = str(parameterization["method"])
            if method not in PARAMETERIZATION_METHODS:
                raise ValueError("parameterization.method must be auto, conformal, or radial")
            cfg.parameterizer = method
        if "signal" in orientation:
            signal = str(orientation["signal"])
            if signal not in {"log_conformal_factor", "radial", "combined"}:
                raise ValueError("orientation.signal must be log_conformal_factor, radial, or combined")
            cfg.orientation_signal = signal
        cfg.validate()
        return cfg

    @staticmethod
    def _read_yaml_text(path: str | Path) -> str:
        path_text = str(path)
        if path_text in {"default", "canonical_sphere", "canonical_sphere.yaml"}:
            return _default_config_text()
        candidate = Path(path)
        if candidate.exists():
            return candidate.read_text()
        if candidate.as_posix() == "configs/canonical_sphere.yaml":
            return _default_config_text()
        raise FileNotFoundError(f"config file was not found: {path}")

    def validate(self) -> None:
        """Validate scalar configuration values before a run starts."""

        if self.parameterizer not in PARAMETERIZATION_METHODS:
            raise ValueError("parameterizer must be auto, conformal, or radial")
        if self.orientation_signal not in {"log_conformal_factor", "radial", "combined"}:
            raise ValueError("orientation_signal must be log_conformal_factor, radial, or combined")
        if self.virtual_buffer_rings < 1:
            raise ValueError("virtual_buffer_rings must be at least 1")
        if self.virtual_buffer_length_factor <= 0.0:
            raise ValueError("virtual_buffer_length_factor must be positive")
        if self.virtual_buffer_min_length_factor_bbox < 0.0:
            raise ValueError("virtual_buffer_min_length_factor_bbox must be non-negative")
        if not 0.0 < self.virtual_buffer_apex_radius_fraction < 1.0:
            raise ValueError("virtual_buffer_apex_radius_fraction must be between 0 and 1")
        if self.mobius_tol <= 0.0:
            raise ValueError("mobius_tol must be positive")
        if self.mobius_max_iters < 1:
            raise ValueError("mobius_max_iters must be at least 1")
        if self.lmax_orientation < 2:
            raise ValueError("lmax_orientation must be at least 2")
        if self.nlat < 2:
            raise ValueError("nlat must be at least 2")
        if self.nlon < 4:
            raise ValueError("nlon must be at least 4")

    def to_dict(self) -> dict:
        return asdict(self)


def _default_config_text() -> str:
    return (
        resources.files("conformal_sphere_pipeline")
        .joinpath("configs/canonical_sphere.yaml")
        .read_text()
    )


@dataclass
class ConformalSphereResult:
    case_id: str
    physical_vertices: np.ndarray
    physical_faces: np.ndarray
    closed_vertices: np.ndarray
    closed_faces: np.ndarray
    physical_vertex_mask: np.ndarray
    physical_face_mask: np.ndarray
    virtual_vertex_mask: np.ndarray
    virtual_face_mask: np.ndarray
    interface_face_mask: np.ndarray
    sphere_raw: np.ndarray
    sphere_mobius: np.ndarray
    sphere_canonical: np.ndarray
    mobius_info: dict
    orientation_info: dict
    boundary_info: dict
    quality: dict
    feature_grid: np.ndarray | None
    grid_theta: np.ndarray | None
    grid_phi: np.ndarray | None
    sh_coeffs: dict | None


def _coeffs_to_arrays(coeffs: dict[tuple[int, int], complex]) -> dict[str, np.ndarray]:
    keys = sorted(coeffs)
    return {
        "ell": np.asarray([k[0] for k in keys], dtype=np.int64),
        "m": np.asarray([k[1] for k in keys], dtype=np.int64),
        "real": np.asarray([coeffs[k].real for k in keys], dtype=np.float64),
        "imag": np.asarray([coeffs[k].imag for k in keys], dtype=np.float64),
    }


FEATURE_CHANNEL_UNITS = {
    "radius": "area_normalized_radius",
    "forward_jacobian": "unit_sphere_area_per_physical_area",
    "inverse_density": "physical_area_per_unit_sphere_area",
    "conformal_factor": "half_log_sphere_per_physical_area_ratio",
    "mean_curvature": "inverse_source_length_unit",
    "log_conformal_factor": "weighted_zscore_log_area_ratio",
    "mean_curvature_zscore": "weighted_zscore_unsigned_mean_curvature",
    "radial_distance": "weighted_zscore_area_normalized_radius",
    "physical_mask": "binary",
    "boundary_distance": "normalized_geodesic_distance_to_virtual_interface",
}


def _face_channels_to_vertex_fields(
    faces: np.ndarray,
    channels: dict[str, np.ndarray],
    feature_names: list[str],
    n_vertices: int,
) -> dict[str, list[float]]:
    counts = np.zeros(n_vertices, dtype=np.float64)
    for tri in faces:
        counts[tri] += 1.0
    counts[counts == 0.0] = 1.0

    fields: dict[str, list[float]] = {}
    for name in feature_names:
        values = np.asarray(channels[name], dtype=np.float64)
        accum = np.zeros(n_vertices, dtype=np.float64)
        for face_index, tri in enumerate(faces):
            accum[tri] += values[face_index]
        fields[name] = (accum / counts).astype(float).tolist()
    return fields


def _bounds_for_vertices(vertices: np.ndarray) -> dict:
    vertices = np.asarray(vertices, dtype=np.float64)
    return {
        "min": vertices.min(axis=0).astype(float).tolist(),
        "max": vertices.max(axis=0).astype(float).tolist(),
    }


def _field_stats(values: list[float]) -> dict:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {}
    sorted_values = np.sort(finite)
    return {
        "min": float(sorted_values[0]),
        "p02": float(np.quantile(sorted_values, 0.02)),
        "p50": float(np.quantile(sorted_values, 0.5)),
        "p98": float(np.quantile(sorted_values, 0.98)),
        "max": float(sorted_values[-1]),
        "mean": float(np.mean(sorted_values)),
        "count": int(sorted_values.size),
        "finite_count": int(sorted_values.size),
        "support": "physical_vertices",
    }


def _interface_vertex_mask(result: ConformalSphereResult) -> np.ndarray:
    mask = np.zeros(len(result.physical_vertices), dtype=bool)
    if not np.any(result.interface_face_mask):
        return mask
    interface_faces = result.closed_faces[result.interface_face_mask]
    interface_vertices = np.unique(interface_faces.ravel())
    interface_vertices = interface_vertices[interface_vertices < len(result.physical_vertices)]
    mask[interface_vertices] = True
    return mask


def _display_interface_face_mask(result: ConformalSphereResult, interface_vertices: np.ndarray) -> np.ndarray:
    if not np.any(interface_vertices):
        return np.zeros(len(result.physical_faces), dtype=bool)
    return np.any(interface_vertices[result.physical_faces], axis=1)


def _quality_warnings(
    *,
    topology_warnings: list[str],
    parameterization_info: dict,
    mobius_info: dict,
    orientation_info: dict,
) -> list[str]:
    warnings = list(topology_warnings)
    fallback_reason = parameterization_info.get("fallback_reason")
    if parameterization_info.get("method") == "radial" and fallback_reason and fallback_reason != "requested":
        warnings.append(f"parameterization fell back to radial: {fallback_reason}")
    if not mobius_info.get("converged", False):
        warnings.append(
            f"Mobius centering did not reach tolerance; final centroid norm "
            f"{mobius_info.get('final_centroid_norm'):.3e}"
        )
    elif float(mobius_info.get("final_centroid_norm", 0.0)) >= 1e-4:
        warnings.append(
            f"Mobius centering centroid norm is above acceptable threshold: "
            f"{mobius_info.get('final_centroid_norm'):.3e}"
        )
    if float(orientation_info.get("orientation_c21_rel", 0.0)) >= 1e-2:
        warnings.append(
            f"orientation C21 residual is above acceptable threshold: "
            f"{orientation_info.get('orientation_c21_rel'):.3e}"
        )
    if float(orientation_info.get("orientation_c22_imag_rel", 0.0)) >= 1e-2:
        warnings.append(
            f"orientation Im(C22) residual is above acceptable threshold: "
            f"{orientation_info.get('orientation_c22_imag_rel'):.3e}"
        )
    return warnings


def _write_outputs(
    out_dir: Path,
    result: ConformalSphereResult,
    feature_names: list[str],
    display_fields: dict[str, list[float]],
    sh_arrays: dict[str, np.ndarray],
    *,
    config: ConformalSphereConfig,
    orientation_signal: str,
    lmax_orientation: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_mesh(out_dir / "physical_mesh.ply", result.physical_vertices, result.physical_faces)
    write_mesh(out_dir / "closed_augmented_mesh.ply", result.closed_vertices, result.closed_faces)
    write_mesh(out_dir / "sphere_raw.ply", result.sphere_raw, result.closed_faces)
    write_mesh(out_dir / "sphere_mobius.ply", result.sphere_mobius, result.closed_faces)
    write_mesh(out_dir / "sphere_canonical.ply", result.sphere_canonical, result.closed_faces)
    physical_sphere = result.sphere_canonical[: len(result.physical_vertices)]
    write_mesh(out_dir / "sphere_physical_support.ply", physical_sphere, result.physical_faces)

    interface_vertices = _interface_vertex_mask(result)
    display_interface_faces = _display_interface_face_mask(result, interface_vertices)
    physical_vertex_mask = np.ones(len(result.physical_vertices), dtype=bool)
    physical_face_mask = np.ones(len(result.physical_faces), dtype=bool)
    field_stats = {name: _field_stats(display_fields[name]) for name in feature_names}
    source_mesh_display = {
        "schema_version": MESH_DISPLAY_SCHEMA_VERSION,
        "mesh_display_format": "indexed_triangle_json",
        "topology": "package_physical_mesh_pre_virtual_closure",
        "index_basis": "package_physical_mesh",
        "vertices": result.physical_vertices.astype(float).reshape(-1).tolist(),
        "faces": result.physical_faces.astype(int).reshape(-1).tolist(),
        "n_vertices_display": int(len(result.physical_vertices)),
        "n_faces_display": int(len(result.physical_faces)),
        "bounds": _bounds_for_vertices(result.physical_vertices),
    }
    write_json(out_dir / "physical_mesh_display.json", source_mesh_display)

    sphere_display = {
        "schema_version": DISPLAY_SCHEMA_VERSION,
        "case_id": result.case_id,
        "mesh_display_format": "indexed_triangle_json",
        "display_support": "physical_surface_only",
        "virtual_closure_role": "numerical_parameterization_only",
        "coordinate_frame": "mobius_centered_sh_canonical",
        "index_basis": "package_physical_mesh",
        "vertices": physical_sphere.astype(float).reshape(-1).tolist(),
        "faces": result.physical_faces.astype(int).reshape(-1).tolist(),
        "n_vertices_display": int(len(result.physical_vertices)),
        "n_faces_display": int(len(result.physical_faces)),
        "bounds": _bounds_for_vertices(physical_sphere),
        "field_names": feature_names,
        "field_units": {name: FEATURE_CHANNEL_UNITS[name] for name in feature_names},
        "fields": display_fields,
        "field_stats": field_stats,
        "physical_vertex_mask": physical_vertex_mask.tolist(),
        "physical_face_mask": physical_face_mask.tolist(),
        "interface_vertex_mask": interface_vertices.tolist(),
        "interface_face_mask": display_interface_faces.tolist(),
        "source_mesh_display": source_mesh_display,
        "physical_counts": {
            "vertices": int(len(result.physical_vertices)),
            "faces": int(len(result.physical_faces)),
        },
        "augmented_counts": {
            "vertices": int(len(result.closed_vertices)),
            "faces": int(len(result.closed_faces)),
            "virtual_vertices": int(np.count_nonzero(result.virtual_vertex_mask)),
            "virtual_faces": int(np.count_nonzero(result.virtual_face_mask)),
        },
        "virtual_buffer": result.boundary_info.get("virtual_buffer", {}),
        "boundary": {
            "physical_boundary_vertex_count": int(np.count_nonzero(interface_vertices)),
            "boundary_sensitive_fields": ["mean_curvature"],
            "reliability_field": "boundary_distance",
            "curvature_domain": "physical_mesh_pre_virtual_closure",
        },
        "metadata": {
            "display_support": "physical_surface_only",
            "virtual_closure_role": "numerical_parameterization_only",
            "index_basis": "package_physical_mesh",
            "histogram_support": "physical_vertices",
            "field_stat_support": "physical_vertices",
        },
        "quality": {
            "mobius_centroid_norm": float(result.mobius_info["final_centroid_norm"]),
            "orientation_c21_rel": float(result.orientation_info["orientation_c21_rel"]),
            "orientation_c22_imag_rel": float(result.orientation_info["orientation_c22_imag_rel"]),
        },
    }
    write_json(out_dir / "sphere_display.json", sphere_display)
    sphere_augmented_qc = {
        "schema_version": AUGMENTED_QC_SCHEMA_VERSION,
        "case_id": result.case_id,
        "virtual_closure_role": "numerical_parameterization_only",
        "display_default": False,
        "vertices": result.sphere_canonical.astype(float).reshape(-1).tolist(),
        "faces": result.closed_faces.astype(int).reshape(-1).tolist(),
        "n_vertices_augmented": int(len(result.closed_vertices)),
        "n_faces_augmented": int(len(result.closed_faces)),
        "physical_vertex_mask": result.physical_vertex_mask.tolist(),
        "physical_face_mask": result.physical_face_mask.tolist(),
        "virtual_vertex_mask": result.virtual_vertex_mask.tolist(),
        "virtual_face_mask": result.virtual_face_mask.tolist(),
        "interface_face_mask": result.interface_face_mask.tolist(),
        "virtual_buffer": result.boundary_info.get("virtual_buffer", {}),
        "quality": result.quality,
    }
    write_json(out_dir / "sphere_augmented_qc.json", sphere_augmented_qc)

    np.savez_compressed(
        out_dir / "features_grid.npz",
        schema_version=np.asarray("canonical-sphere.features-grid.v1"),
        case_id=np.asarray(result.case_id),
        features=result.feature_grid,
        channel_names=np.asarray(feature_names),
        channel_units=np.asarray([FEATURE_CHANNEL_UNITS[name] for name in feature_names]),
        channel_axis=np.asarray(0, dtype=np.int64),
        theta=result.grid_theta,
        phi=result.grid_phi,
        rotation_matrix=np.asarray(result.orientation_info["rotation_matrix"]),
        mobius_centroid_norm=result.mobius_info["final_centroid_norm"],
        physical_mask=result.feature_grid[feature_names.index("physical_mask")],
        config_json=np.asarray(_json_dumps(config.to_dict())),
    )
    np.savez_compressed(
        out_dir / "sh_coeffs.npz",
        schema_version=np.asarray("canonical-sphere.sh-coeffs.v1"),
        case_id=np.asarray(result.case_id),
        orientation_signal=np.asarray(orientation_signal),
        lmax_orientation=np.asarray(lmax_orientation, dtype=np.int64),
        coefficient_basis=np.asarray("complex_scipy_spherical_harmonics"),
        **sh_arrays,
    )

    summary = {
        "schema_version": "canonical-sphere.case.v1",
        "case_id": result.case_id,
        "config": config.to_dict(),
        "boundary_info": result.boundary_info,
        "mobius_info": result.mobius_info,
        "orientation_info": result.orientation_info,
        "quality": result.quality,
    }
    write_json(out_dir / "canonical_result.json", summary)
    write_json(out_dir / "quality_report.json", {"schema_version": "canonical-sphere.quality.v1", **result.quality})


def _json_dumps(payload: dict) -> str:
    import json

    return json.dumps(payload, sort_keys=True)


def canonicalize_mesh_file(
    mesh_path: str | Path,
    out_dir: str | Path,
    *,
    config: ConformalSphereConfig | None = None,
) -> ConformalSphereResult:
    """Run the conformal sphere pipeline on one mesh and write artifacts."""

    cfg = config or ConformalSphereConfig()
    cfg.validate()
    mesh_path = Path(mesh_path)
    out_dir = Path(out_dir)
    vertices, faces = load_triangle_mesh(mesh_path)
    validate_triangle_mesh(vertices, faces, require_manifold=False)

    loops = extract_boundary_loops(vertices, faces)

    buffer_settings = {
        "enabled": cfg.virtual_buffer_enabled,
        "rings": cfg.virtual_buffer_rings,
        "length_factor": cfg.virtual_buffer_length_factor,
        "min_length_factor_bbox": cfg.virtual_buffer_min_length_factor_bbox,
        "apex_radius_fraction": cfg.virtual_buffer_apex_radius_fraction,
        "adaptive_retry_used": False,
    }
    if cfg.virtual_buffer_enabled:
        closed = add_virtual_boundary_buffers(
            vertices,
            faces,
            loops=loops,
            rings=cfg.virtual_buffer_rings,
            length_factor=cfg.virtual_buffer_length_factor,
            min_length_factor_bbox=cfg.virtual_buffer_min_length_factor_bbox,
            apex_radius_fraction=cfg.virtual_buffer_apex_radius_fraction,
        )
    else:
        closed = add_virtual_boundary_buffers(vertices, faces, loops=[], rings=0)

    topology = validate_triangle_mesh(closed.vertices, closed.faces, require_manifold=True)
    if topology.boundary_edge_count != 0:
        raise ValueError("closed mesh still has boundary edges")
    if topology.genus != 0:
        raise ValueError(f"closed mesh genus is {topology.genus}, expected 0")

    param = parameterize_sphere(closed.vertices, closed.faces, method=cfg.parameterizer)
    if (
        cfg.parameterizer == "auto"
        and cfg.virtual_buffer_enabled
        and loops
        and param.info.get("method") == "radial"
        and param.info.get("fallback_reason")
    ):
        retry_presets = [
            (4, 0.75, 0.02),
            (6, 1.0, 0.02),
            (8, 1.5, 0.02),
        ]
        for retry_rings, retry_length, retry_apex in retry_presets:
            if (
                retry_rings == cfg.virtual_buffer_rings
                and retry_length == cfg.virtual_buffer_length_factor
                and retry_apex == cfg.virtual_buffer_apex_radius_fraction
            ):
                continue
            retry_closed = add_virtual_boundary_buffers(
                vertices,
                faces,
                loops=loops,
                rings=retry_rings,
                length_factor=retry_length,
                min_length_factor_bbox=cfg.virtual_buffer_min_length_factor_bbox,
                apex_radius_fraction=retry_apex,
            )
            retry_topology = validate_triangle_mesh(retry_closed.vertices, retry_closed.faces, require_manifold=True)
            if retry_topology.boundary_edge_count != 0 or retry_topology.genus != 0:
                continue
            retry_param = parameterize_sphere(retry_closed.vertices, retry_closed.faces, method=cfg.parameterizer)
            if retry_param.info.get("method") != "radial":
                retry_param.info["virtual_buffer_retry_used"] = True
                retry_param.info["initial_fallback_reason"] = param.info.get("fallback_reason", "")
                retry_param.info["retry_rings"] = retry_rings
                retry_param.info["retry_length_factor"] = retry_length
                retry_param.info["retry_apex_radius_fraction"] = retry_apex
                closed = retry_closed
                topology = retry_topology
                param = retry_param
                buffer_settings.update(
                    {
                        "rings": retry_rings,
                        "length_factor": retry_length,
                        "apex_radius_fraction": retry_apex,
                        "adaptive_retry_used": True,
                    }
                )
                break
    sphere_raw = param.sphere
    phys_weights = np.zeros(len(closed.faces), dtype=np.float64)
    phys_areas = np.zeros(len(closed.faces), dtype=np.float64)
    from .spherical.quality import face_areas as _face_areas

    phys_areas[:] = _face_areas(closed.vertices, closed.faces)
    phys_weights[closed.physical_face_mask] = phys_areas[closed.physical_face_mask]
    sphere_mobius, mobius_info = mobius_center(
        sphere_raw,
        closed.faces,
        phys_weights,
        tol=cfg.mobius_tol,
        max_iters=cfg.mobius_max_iters,
    )

    channels, support = compute_face_features(
        closed.vertices,
        closed.faces,
        sphere_mobius,
        closed.physical_face_mask,
    )
    if cfg.orientation_signal == "radial":
        orientation_values = channels["radial_distance"]
    elif cfg.orientation_signal == "combined":
        orientation_values = channels["log_conformal_factor"] + 0.25 * channels["radial_distance"]
    else:
        orientation_values = channels["log_conformal_factor"]

    sphere_canonical, rotation, orientation_info = canonical_orient_sphere(
        sphere_mobius,
        closed.faces,
        orientation_values,
        support["area_sphere"],
        closed.physical_face_mask,
        lmax_orientation=cfg.lmax_orientation,
    )

    feature_names = [
        "radius",
        "forward_jacobian",
        "inverse_density",
        "conformal_factor",
        "mean_curvature",
        "log_conformal_factor",
        "mean_curvature_zscore",
        "radial_distance",
        "physical_mask",
        "boundary_distance",
    ]
    feature_grid, theta, phi = resample_channels_to_equiangular(
        sphere_canonical,
        closed.faces,
        [channels[name] for name in feature_names],
        nlat=cfg.nlat,
        nlon=cfg.nlon,
    )

    coeffs = compute_complex_sh_coeffs_from_faces(
        sphere_canonical,
        closed.faces,
        orientation_values,
        support["area_sphere"] * closed.physical_face_mask.astype(float),
        cfg.lmax_orientation,
    )
    sh_arrays = _coeffs_to_arrays(coeffs)
    display_fields = _face_channels_to_vertex_fields(
        closed.faces[closed.physical_face_mask],
        {name: values[closed.physical_face_mask] for name, values in channels.items()},
        feature_names,
        len(vertices),
    )

    quality = {
        "topology": topology.to_dict(),
        "sphere_raw": sphere_norm_metrics(sphere_raw),
        "sphere_mobius": sphere_norm_metrics(sphere_mobius),
        "sphere_canonical": sphere_norm_metrics(sphere_canonical),
        "mobius": mobius_info,
        "orientation": {
            "orientation_c21_rel": orientation_info["orientation_c21_rel"],
            "orientation_c22_imag_rel": orientation_info["orientation_c22_imag_rel"],
            "orientation_c22_real_rel": orientation_info["orientation_c22_real_rel"],
            "rotation_det_error": orientation_info["rotation_det_error"],
            "rotation_orthogonality_error": orientation_info["rotation_orthogonality_error"],
        },
        "parameterization": param.info,
        "area_distortion": {
            "physical_log_lambda_mean": float(np.mean(support["log_lambda"][closed.physical_face_mask])),
            "physical_log_lambda_std": float(np.std(support["log_lambda"][closed.physical_face_mask])),
            "physical_log_lambda_p95": float(np.percentile(np.abs(support["log_lambda"][closed.physical_face_mask]), 95)),
        },
        "display": {
            "schema_version": DISPLAY_SCHEMA_VERSION,
            "display_support": "physical_surface_only",
            "virtual_closure_role": "numerical_parameterization_only",
            "field_stat_support": "physical_vertices",
            "curvature_domain": "physical_mesh_pre_virtual_closure",
        },
    }
    quality["warnings"] = _quality_warnings(
        topology_warnings=topology.warnings,
        parameterization_info=param.info,
        mobius_info=mobius_info,
        orientation_info=orientation_info,
    )
    if cfg.fail_on_warning and quality["warnings"]:
        joined = "; ".join(quality["warnings"])
        raise RuntimeError(f"quality warnings encountered with fail_on_warning=true: {joined}")

    result = ConformalSphereResult(
        case_id=mesh_path.stem,
        physical_vertices=vertices,
        physical_faces=faces,
        closed_vertices=closed.vertices,
        closed_faces=closed.faces,
        physical_vertex_mask=closed.physical_vertex_mask,
        physical_face_mask=closed.physical_face_mask,
        virtual_vertex_mask=closed.virtual_vertex_mask,
        virtual_face_mask=closed.virtual_face_mask,
        interface_face_mask=closed.interface_face_mask,
        sphere_raw=sphere_raw,
        sphere_mobius=sphere_mobius,
        sphere_canonical=sphere_canonical,
        mobius_info=mobius_info,
        orientation_info=orientation_info,
        boundary_info={"loops": [loop.to_dict() for loop in loops], "virtual_buffer": buffer_settings},
        quality=quality,
        feature_grid=feature_grid,
        grid_theta=theta,
        grid_phi=phi,
        sh_coeffs=coeffs,
    )
    _write_outputs(
        out_dir,
        result,
        feature_names,
        display_fields,
        sh_arrays,
        config=cfg,
        orientation_signal=cfg.orientation_signal,
        lmax_orientation=cfg.lmax_orientation,
    )
    return result
