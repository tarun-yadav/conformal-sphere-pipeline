#!/usr/bin/env python3
"""Run a self-contained spherical atlas-validation battery."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conformal_sphere_pipeline.validation.claims import (  # noqa: E402
    AtlasValidationMetrics,
    ValidationThresholds,
    classify_claim_level,
)
from conformal_sphere_pipeline.validation.figures import write_transition_panel  # noqa: E402
from conformal_sphere_pipeline.validation.gauge import (  # noqa: E402
    harmonic_orient_material_sphere,
    material_face_signal,
    select_material_pole_face,
)
from conformal_sphere_pipeline.validation.learning_probe import (  # noqa: E402
    evaluate_material_decoder,
    fit_material_decoder,
)
from conformal_sphere_pipeline.validation.nulls import shuffled_transition_null  # noqa: E402
from conformal_sphere_pipeline.validation.runner import (  # noqa: E402
    SphericalValidationMap,
    SphericalValidationRunConfig,
    run_spherical_validation_map,
)
from conformal_sphere_pipeline.validation.synthetic_cases import (  # noqa: E402
    MaterialMesh,
    build_candy_cane_case,
    cap_open_tube_with_material_ids,
)
from conformal_sphere_pipeline.validation.transition import (  # noqa: E402
    atlas_cell_pullback_metrics,
    matched_vertex_transition_metrics,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Output directory for figures and ledgers.")
    parser.add_argument("--quick", action="store_true", help="Use low iteration counts for smoke validation.")
    parser.add_argument("--n-s", type=int, default=None, help="Override synthetic cane centerline samples.")
    parser.add_argument("--n-theta", type=int, default=None, help="Override synthetic cane hoop samples.")
    parser.add_argument("--null-trials", type=int, default=None, help="Number of shuffled null trials per case.")
    parser.add_argument("--cell-z", type=int, default=6, help="Equal-area z bands for atlas-cell pullback.")
    parser.add_argument("--cell-lon", type=int, default=12, help="Longitude bands for atlas-cell pullback.")
    parser.add_argument(
        "--parameterizer",
        choices=["stereographic", "conformal"],
        default="stereographic",
        help="Spherical parameterizer to validate.",
    )
    parser.add_argument(
        "--material-anchor-pole",
        action="store_true",
        help="Use material/anatomical coordinates to select a shared stereographic pole face.",
    )
    parser.add_argument(
        "--sh-orientation",
        action="store_true",
        help="Orient each spherical map with spherical-harmonic constraints on a material signal before metrics.",
    )
    parser.add_argument("--anchor-s", type=float, default=0.5, help="Material s-coordinate for the pole anchor.")
    parser.add_argument(
        "--anchor-theta",
        type=float,
        default=0.0,
        help="Material theta-coordinate, in radians, for the pole anchor.",
    )
    args = parser.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / "cache"
    if args.material_anchor_pole and args.parameterizer != "stereographic":
        parser.error("--material-anchor-pole only applies to --parameterizer stereographic")

    config = replace(_validation_config(args.quick), parameterization_method=args.parameterizer)
    n_s = args.n_s or (14 if args.quick else 36)
    n_theta = args.n_theta or (8 if args.quick else 18)
    null_trials = args.null_trials or (8 if args.quick else 64)

    case = build_candy_cane_case(n_s=n_s, n_theta=n_theta)
    gauge_info = {
        "parameterizer": args.parameterizer,
        "material_anchor_pole": bool(args.material_anchor_pole),
        "sh_orientation": bool(args.sh_orientation),
        "anchor_s": float(args.anchor_s),
        "anchor_theta": float(args.anchor_theta),
    }
    if args.material_anchor_pole:
        anchor_face = select_material_pole_face(
            case.reference,
            target_s=float(args.anchor_s),
            target_theta=float(args.anchor_theta),
        )
        config = replace(config, pole_face_index=int(anchor_face))
        gauge_info["material_anchor_face_index"] = int(anchor_face)

    rotated_reference = _transform_material_mesh(case.reference, _rotation_z_y())
    scaled_reference = _transform_material_mesh(case.reference, 1.35 * np.eye(3))

    pair_results = {
        "exact_repeat": _run_pair(
            out,
            cache_dir,
            config,
            case.reference,
            case.reference,
            "exact_repeat",
            null_trials=null_trials,
            null_seed=101,
            cell_z=args.cell_z,
            cell_lon=args.cell_lon,
            sh_orientation=args.sh_orientation,
        ),
        "rigid_rotation": _run_pair(
            out,
            cache_dir,
            config,
            case.reference,
            rotated_reference,
            "rigid_rotation",
            null_trials=null_trials,
            null_seed=102,
            cell_z=args.cell_z,
            cell_lon=args.cell_lon,
            sh_orientation=args.sh_orientation,
        ),
        "global_scale": _run_pair(
            out,
            cache_dir,
            config,
            case.reference,
            scaled_reference,
            "global_scale",
            null_trials=null_trials,
            null_seed=103,
            cell_z=args.cell_z,
            cell_lon=args.cell_lon,
            sh_orientation=args.sh_orientation,
        ),
        "uniform_2x": _run_pair(
            out,
            cache_dir,
            config,
            case.reference,
            case.uniform_2x,
            "uniform_2x",
            null_trials=null_trials,
            null_seed=104,
            cell_z=args.cell_z,
            cell_lon=args.cell_lon,
            sh_orientation=args.sh_orientation,
        ),
        "local_bulge": _run_pair(
            out,
            cache_dir,
            config,
            case.reference,
            case.local_bulge,
            "local_bulge",
            null_trials=null_trials,
            null_seed=105,
            cell_z=args.cell_z,
            cell_lon=args.cell_lon,
            sh_orientation=args.sh_orientation,
        ),
    }
    aggregate = _aggregate_metrics(pair_results)
    classification = classify_claim_level(aggregate, ValidationThresholds())
    report = {
        "schema_version": 1,
        "validation_scope": "synthetic_known_correspondence_candy_cane",
        "config": asdict(config),
        "gauge": gauge_info,
        "mesh_resolution": {"n_s": n_s, "n_theta": n_theta},
        "null_trials": null_trials,
        "cell_grid": {"n_z": args.cell_z, "n_lon": args.cell_lon},
        **pair_results,
        "aggregate_metrics": asdict(aggregate),
        "claim_classification": asdict(classification),
        "interpretation_rule": (
            "The reported claim level is the strongest permitted conclusion. "
            "Do not upgrade to atlas_substrate unless transition-map, pullback, and null-separation gates pass."
        ),
    }
    ledger_json = out / "validation_claim_ledger.json"
    ledger_json.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(out / "validation_claim_ledger.md", report)
    print(json.dumps(_json_safe({"claim_level": classification.claim_level, "failed_gates": classification.failed_gates}), indent=2))
    return 0


def _validation_config(quick: bool) -> SphericalValidationRunConfig:
    if quick:
        return SphericalValidationRunConfig(
            pole_area_threshold=5e-2,
            stage_a_max_iter=4,
            stage_b_max_iter=2,
            diagnostics_samples_per_face=4,
            injectivity_max_depth=4,
        )
    return SphericalValidationRunConfig(
        pole_area_threshold=2e-2,
        stage_a_max_iter=24,
        stage_b_max_iter=16,
        diagnostics_samples_per_face=16,
        injectivity_max_depth=8,
    )


def _run_pair(
    out: Path,
    cache_dir: Path,
    config: SphericalValidationRunConfig,
    source: MaterialMesh,
    target: MaterialMesh,
    case_id: str,
    *,
    null_trials: int,
    null_seed: int,
    cell_z: int,
    cell_lon: int,
    sh_orientation: bool,
) -> dict[str, Any]:
    capped_source = cap_open_tube_with_material_ids(source)
    capped_target = cap_open_tube_with_material_ids(target)
    source_map = run_spherical_validation_map(
        f"{case_id}_source",
        capped_source.vertices,
        capped_source.faces,
        cache_dir,
        config=config,
    )
    target_map = run_spherical_validation_map(
        f"{case_id}_target",
        capped_target.vertices,
        capped_target.faces,
        cache_dir,
        config=config,
    )
    n = capped_source.original_vertex_count
    source_sphere = source_map.sphere[:n]
    target_sphere = target_map.sphere[:n]
    orientation_info = None
    if sh_orientation:
        source_signal = material_face_signal(source)
        target_signal = material_face_signal(target)
        source_sphere, source_sh_info = harmonic_orient_material_sphere(
            source_sphere,
            source,
            source_signal,
        )
        target_sphere, target_sh_info = harmonic_orient_material_sphere(
            target_sphere,
            target,
            target_signal,
        )
        orientation_info = {"source": source_sh_info, "target": target_sh_info}

    metrics = matched_vertex_transition_metrics(source_sphere, target_sphere, faces=source.faces)
    pullback = atlas_cell_pullback_metrics(source_sphere, target_sphere, n_z=cell_z, n_lon=cell_lon)
    null = shuffled_transition_null(source_sphere, target_sphere, n_trials=null_trials, seed=null_seed, faces=source.faces)
    null_z = null.separation_z(metrics.aligned_median_deg)

    decoder = fit_material_decoder(source_sphere, source.material_s, source.material_theta)
    aligned_target = target_sphere @ metrics.rotation.T
    decoder_metrics = evaluate_material_decoder(decoder, aligned_target, target.material_s, target.material_theta)

    figure = write_transition_panel(
        out,
        case_id,
        capped_source.vertices[:n],
        capped_target.vertices[:n],
        source_sphere,
        target_sphere,
        metrics,
        source_faces=source.faces,
        target_faces=target.faces,
        pullback_metrics=pullback,
        null_separation_z=null_z,
    )
    return {
        "transition": asdict(metrics),
        "pullback": asdict(pullback),
        "null": {
            "aligned_median_deg": null.aligned_median_deg,
            "aligned_p95_deg": null.aligned_p95_deg,
            "separation_z": null_z,
        },
        "decoder_transfer": asdict(decoder_metrics),
        "sh_orientation": orientation_info,
        "source_map": _map_manifest(source_map),
        "target_map": _map_manifest(target_map),
        "figure": str(figure),
    }


def _aggregate_metrics(pair_results: dict[str, dict[str, Any]]) -> AtlasValidationMetrics:
    maps = []
    for result in pair_results.values():
        maps.extend([result["source_map"], result["target_map"]])
    atlas_cases = [pair_results["uniform_2x"], pair_results["local_bulge"]]
    return AtlasValidationMetrics(
        max_unit_norm_error=max(float(item["info"]["max_sphere_norm_error"]) for item in maps),
        uncertified_patch_count=max(int(item["info"]["uncertified_patch_count"]) for item in maps),
        min_certified_det_jacobian=min(float(item["info"]["min_certified_det_jacobian"]) for item in maps),
        sphere_z_range=min(float(item["info"]["sphere_z_range"]) for item in maps),
        exact_repeat_median_deg=float(pair_results["exact_repeat"]["transition"]["raw_median_deg"]),
        exact_repeat_max_deg=float(pair_results["exact_repeat"]["transition"]["raw_max_deg"]),
        rigid_rotation_median_deg=float(pair_results["rigid_rotation"]["transition"]["aligned_median_deg"]),
        global_scale_median_deg=float(pair_results["global_scale"]["transition"]["aligned_median_deg"]),
        transition_aligned_median_deg=max(float(item["transition"]["aligned_median_deg"]) for item in atlas_cases),
        transition_aligned_p95_deg=max(float(item["transition"]["aligned_p95_deg"]) for item in atlas_cases),
        transition_smoothness_p95=max(float(item["transition"]["smoothness_p95"]) for item in atlas_cases),
        cell_pullback_jaccard_median=min(float(item["pullback"]["median_jaccard"]) for item in atlas_cases),
        null_separation_z=min(float(item["null"]["separation_z"]) for item in atlas_cases),
    )


def _map_manifest(result: SphericalValidationMap) -> dict[str, Any]:
    return {
        "name": result.name,
        "input_hash": result.input_hash,
        "config_hash": result.config_hash,
        "cache_hit": result.cache_hit,
        "sphere_path": str(result.sphere_path),
        "manifest_path": str(result.manifest_path),
        "info": result.info,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    classification = report["claim_classification"]
    aggregate = report["aggregate_metrics"]
    gauge = report["gauge"]
    lines = [
        "# Spherical Atlas Validation Claim Ledger",
        "",
        "## Claim",
        "",
        f"- Claim level: `{classification['claim_level']}`",
        f"- Functional pass: `{classification['functional_pass']}`",
        f"- Common-frame pass: `{classification['common_frame_pass']}`",
        f"- Atlas pass: `{classification['atlas_pass']}`",
        f"- Failed gates: `{', '.join(classification['failed_gates']) if classification['failed_gates'] else 'none'}`",
        "",
        "## Gauge",
        "",
        f"- Material pole anchor: `{gauge['material_anchor_pole']}`",
        f"- SH material orientation: `{gauge['sh_orientation']}`",
        f"- Anchor material coordinate: `s={gauge['anchor_s']:.6g}`, `theta={gauge['anchor_theta']:.6g}`",
    ]
    if "material_anchor_face_index" in gauge:
        lines.append(f"- Anchor face index: `{gauge['material_anchor_face_index']}`")
    lines.extend(
        [
            "",
            "## Aggregate Metrics",
            "",
            f"- Exact repeat max drift: `{aggregate['exact_repeat_max_deg']:.6g}` deg",
            f"- Rigid-rotation aligned median: `{aggregate['rigid_rotation_median_deg']:.6g}` deg",
            f"- Global-scale aligned median: `{aggregate['global_scale_median_deg']:.6g}` deg",
            f"- Deformation aligned median: `{aggregate['transition_aligned_median_deg']:.6g}` deg",
            f"- Deformation aligned p95: `{aggregate['transition_aligned_p95_deg']:.6g}` deg",
            f"- Transition smoothness p95: `{aggregate['transition_smoothness_p95']:.6g}`",
            f"- Cell pullback median Jaccard: `{aggregate['cell_pullback_jaccard_median']:.6g}`",
            f"- Null separation z: `{aggregate['null_separation_z']:.6g}`",
            "",
            "## Figures",
            "",
        ]
    )
    for case_id in ["exact_repeat", "rigid_rotation", "global_scale", "uniform_2x", "local_bulge"]:
        lines.append(f"- `{Path(report[case_id]['figure']).name}`")
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            report["interpretation_rule"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _transform_material_mesh(mesh: MaterialMesh, matrix: np.ndarray) -> MaterialMesh:
    return MaterialMesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64) @ np.asarray(matrix, dtype=np.float64).T,
        faces=np.asarray(mesh.faces, dtype=np.int64).copy(),
        material_s=np.asarray(mesh.material_s, dtype=np.float64).copy(),
        material_theta=np.asarray(mesh.material_theta, dtype=np.float64).copy(),
    )


def _rotation_z_y() -> np.ndarray:
    z_angle = np.deg2rad(33.0)
    y_angle = np.deg2rad(-18.0)
    rz = np.array(
        [
            [np.cos(z_angle), -np.sin(z_angle), 0.0],
            [np.sin(z_angle), np.cos(z_angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    ry = np.array(
        [
            [np.cos(y_angle), 0.0, np.sin(y_angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(y_angle), 0.0, np.cos(y_angle)],
        ],
        dtype=np.float64,
    )
    return rz @ ry


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _safe_float(float(value))
    if isinstance(value, float):
        return _safe_float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _safe_float(value: float) -> float | str:
    if np.isfinite(value):
        return float(value)
    if np.isnan(value):
        return "nan"
    return "inf" if value > 0.0 else "-inf"


if __name__ == "__main__":
    raise SystemExit(main())
