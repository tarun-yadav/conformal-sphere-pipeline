"""Figure writers for spherical atlas-validation evidence."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

_mpl_cache = Path(tempfile.gettempdir()) / "conformal_sphere_pipeline_matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from conformal_sphere_pipeline.validation.transition import (
    CellPullbackMetrics,
    TransitionMetrics,
    angular_degrees,
    unit_rows,
)


def write_transition_panel(
    out_dir: Path | str,
    case_id: str,
    source_physical: np.ndarray,
    target_physical: np.ndarray,
    source_sphere: np.ndarray,
    target_sphere: np.ndarray,
    metrics: TransitionMetrics,
    *,
    source_faces: np.ndarray | None = None,
    target_faces: np.ndarray | None = None,
    pullback_metrics: CellPullbackMetrics | None = None,
    null_separation_z: float | None = None,
) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    source_physical = np.asarray(source_physical, dtype=np.float64)
    target_physical = np.asarray(target_physical, dtype=np.float64)
    source_unit = unit_rows(source_sphere)
    target_unit = unit_rows(target_sphere)
    aligned_target = target_unit @ metrics.rotation.T
    raw_residual = angular_degrees(source_unit, target_unit)
    aligned_residual = angular_degrees(source_unit, aligned_target)
    colors = _material_colors(len(source_unit))

    fig = plt.figure(figsize=(14.0, 8.4))
    axes = [fig.add_subplot(2, 3, index + 1, projection="3d") for index in range(5)]
    hist_ax = fig.add_subplot(2, 3, 6)

    _draw_mesh_or_points(axes[0], source_physical, source_faces, colors)
    axes[0].set_title("physical source", fontsize=10)
    _set_equal_3d(axes[0], source_physical)

    _draw_mesh_or_points(axes[1], target_physical, target_faces, colors)
    axes[1].set_title("physical target", fontsize=10)
    _set_equal_3d(axes[1], target_physical)

    _draw_reference_sphere(axes[2])
    _draw_mesh_or_points(axes[2], source_unit, source_faces, colors, vertex_size=3.0, mesh_alpha=0.55)
    axes[2].set_title("source sphere", fontsize=10)

    _draw_reference_sphere(axes[3])
    _draw_mesh_or_points(axes[3], target_unit, target_faces, colors, vertex_size=3.0, mesh_alpha=0.55)
    axes[3].set_title("target sphere", fontsize=10)

    _draw_reference_sphere(axes[4])
    scatter = axes[4].scatter(
        source_unit[:, 0],
        source_unit[:, 1],
        source_unit[:, 2],
        c=aligned_residual,
        cmap="magma",
        s=14,
    )
    axes[4].quiver(
        source_unit[:, 0],
        source_unit[:, 1],
        source_unit[:, 2],
        aligned_target[:, 0] - source_unit[:, 0],
        aligned_target[:, 1] - source_unit[:, 1],
        aligned_target[:, 2] - source_unit[:, 2],
        length=0.45,
        normalize=False,
        color=(0.05, 0.05, 0.05, 0.55),
        linewidth=0.35,
    )
    axes[4].set_title("aligned transition residual", fontsize=10)
    fig.colorbar(scatter, ax=axes[4], fraction=0.035, pad=0.02, label="deg")

    hist_ax.hist(raw_residual, bins=24, color="#7a5195", alpha=0.45, label="raw")
    hist_ax.hist(aligned_residual, bins=24, color="#2f4b7c", alpha=0.78, label="SO(3) aligned")
    hist_ax.set_xlabel("angular residual (deg)")
    hist_ax.set_ylabel("matched vertices")
    hist_ax.legend(frameon=False, fontsize=8)
    hist_ax.set_title("matched material residuals", fontsize=10)
    hist_ax.text(
        0.02,
        0.98,
        _metric_text(metrics, pullback_metrics, null_separation_z),
        transform=hist_ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.9},
    )

    fig.suptitle(f"Spherical atlas-validation transition panel: {case_id}", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    path = out_path / f"{case_id}_transition_panel.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _metric_text(
    metrics: TransitionMetrics,
    pullback_metrics: CellPullbackMetrics | None,
    null_separation_z: float | None,
) -> str:
    lines = [
        f"raw median: {metrics.raw_median_deg:.2f} deg",
        f"aligned median: {metrics.aligned_median_deg:.2f} deg",
        f"aligned p95: {metrics.aligned_p95_deg:.2f} deg",
        f"smoothness p95: {metrics.smoothness_p95:.4f}",
    ]
    if pullback_metrics is not None:
        lines.append(f"cell Jaccard median: {pullback_metrics.median_jaccard:.2f}")
    if null_separation_z is not None:
        lines.append(f"null separation z: {null_separation_z:.2f}")
    return "\n".join(lines)


def _set_equal_3d(ax, points: np.ndarray, radius: float | None = None) -> None:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        center = np.zeros(3, dtype=np.float64)
        radius = 1.0
    else:
        center = 0.5 * (pts.max(axis=0) + pts.min(axis=0))
        if radius is None:
            radius = 0.55 * float(np.max(pts.max(axis=0) - pts.min(axis=0)))
    radius = max(float(radius if radius is not None else 1.0), 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


def _draw_mesh_or_points(
    ax,
    points: np.ndarray,
    faces: np.ndarray | None,
    colors: np.ndarray,
    *,
    vertex_size: float = 2.0,
    mesh_alpha: float = 0.78,
) -> None:
    pts = np.asarray(points, dtype=np.float64)
    face_array = None if faces is None else np.asarray(faces, dtype=np.int64)
    if face_array is not None and face_array.ndim == 2 and face_array.shape[1] == 3 and len(face_array) > 0:
        triangles = pts[face_array]
        face_colors = np.mean(colors[face_array], axis=1)
        face_colors[:, 3] = mesh_alpha
        mesh = Poly3DCollection(
            triangles,
            facecolors=face_colors,
            edgecolors=(0.04, 0.04, 0.04, 0.22),
            linewidths=0.15,
        )
        ax.add_collection3d(mesh)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=vertex_size, depthshade=False)
        return
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=max(vertex_size, 9.0), depthshade=False)


def _draw_reference_sphere(ax) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 36)
    v = np.linspace(0.0, np.pi, 18)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color=(0.55, 0.55, 0.55, 0.23), linewidth=0.35)
    _set_equal_3d(ax, np.column_stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)]), radius=1.05)


def _material_colors(n: int) -> np.ndarray:
    values = np.linspace(0.0, 1.0, n, endpoint=False)
    colors = plt.cm.hsv(values)
    colors[:, 3] = 0.88
    return colors
