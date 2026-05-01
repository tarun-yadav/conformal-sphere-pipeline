# sphere-mapping-pipeline

`sphere-mapping-pipeline` is a Python package and CLI for mapping triangle
surface meshes to a canonical sphere, `S^2`. It closes open boundary loops with
virtual buffer geometry, computes a spherical parameterization, recenters the
map with a Mobius transform, fixes a reproducible spherical-harmonic
orientation, and writes meshes, grids, spherical harmonic coefficients, display
artifacts, and JSON quality reports.

The package can be used directly for batch preprocessing or as the local
computational layer for a downstream browser dashboard. A dashboard can either
read precomputed outputs from this CLI or call the installed CLI on demand when
a local case is first rendered.

## Installation

For development and verification:

```bash
python3 -m pip install -e ".[dev]"
```

For normal local use:

```bash
python3 -m pip install -e .
```

The conformal backend uses `lapy`. `method="auto"` tries that conformal backend
first and falls back to radial projection only when the backend rejects the
mesh. `method="conformal"` fails explicitly if conformal mapping cannot be
computed. The package also includes a stereographic Bezier backend that can be
selected with the same `parameterizer` argument.

Supported verification targets are Python 3.11 and 3.12 in CI. Local release
checks may additionally exercise newer workstation Python versions.

## Python API

```python
from sphere_mapping_pipeline import (
    SphereMappingConfig,
    StereographicConfig,
    canonicalize_mesh_file,
    load_triangle_mesh,
    parameterize_sphere,
    validate_triangle_mesh,
)

config = SphereMappingConfig(parameterizer="auto", nlat=128, nlon=256)
result = canonicalize_mesh_file("input_mesh.ply", "out/case_001", config=config)

stereo_config = SphereMappingConfig(parameterizer="stereographic")
stereo_result = canonicalize_mesh_file("input_mesh.ply", "out/case_001_stereo", config=stereo_config)
```

Public API:

- `SphereMappingConfig`
- `SphereMappingResult`
- `StereographicConfig`
- `canonicalize_mesh_file`
- `load_triangle_mesh`
- `parameterize_sphere`
- `validate_triangle_mesh`

Parameterization methods:

- `auto`: try the `lapy` conformal backend, then fall back to radial.
- `conformal`: require the `lapy` conformal backend.
- `stereographic`: use the stereographic Bezier backend.
- `radial`: use centroid radial projection.

For direct array-level use, call the shared parameterization entry point:

```python
from sphere_mapping_pipeline import StereographicConfig, parameterize_sphere

param = parameterize_sphere(
    vertices,
    faces,
    method="stereographic",
    stereographic_config=StereographicConfig(
        pole_area_threshold=5e-2,
        stage_a_max_iter=24,
        stage_b_max_iter=16,
    ),
)

sphere_vertices = param.sphere
diagnostics = param.info
```

`parameterize_sphere` expects a closed genus-zero triangular mesh for the
conformal and stereographic backends. `canonicalize_mesh_file` is the safer
entry point for open vascular surfaces because it performs the package's
virtual boundary closure before parameterization.

The previous package import name, `conformal_sphere_pipeline`, is kept as a
compatibility shim. New code should import `sphere_mapping_pipeline` and use
`SphereMappingConfig` / `SphereMappingResult`.

## CLI

```bash
sphere-map canonicalize input_mesh.ply --out out/case_001 --parameterizer auto
sphere-map canonicalize input_mesh.ply --out out/case_001_stereo --parameterizer stereographic
```

Useful options:

- `--config default`
- `--config configs/canonical_sphere.yaml` when running from a source checkout
- `--parameterizer auto|conformal|stereographic|radial`
- `--no-virtual-buffer`
- `--orientation-signal log_conformal_factor|radial|combined`
- `--lmax-orientation 16`
- `--nlat 128 --nlon 256`

The old `conformal-sphere` executable is kept as a compatibility alias. New
commands should use `sphere-map`.

The default configuration is in `configs/canonical_sphere.yaml`. Its default
orientation bandwidth is `lmax_orientation: 16`; raise it for slower high-band
runs when the cohort analysis needs more angular detail. Installed wheel users
can pass `--config default` to load the packaged copy of the same configuration.

To make the stereographic backend the default for a local run configuration:

```yaml
parameterization:
  method: stereographic
```

## Stereographic Backend

The stereographic backend is an alternative to the conformal backend, not a
separate pipeline surface. It uses the same high-level CLI, the same
`SphereMappingConfig.parameterizer` field, and the same `parameterize_sphere`
entry point. The backend removes one pole triangle, maps the remaining surface
to a planar disk, optimizes shared quadratic Bezier controls, maps the result
back to `S^2` by inverse stereographic projection, and records diagnostics for
unit-sphere error, pole choice, certified local Jacobian sign, sphere coverage,
and sampled distortion.

This backend is useful when the downstream question is not angle preservation
itself, but whether a stable spherical coordinate substrate can support
comparison across meshes. It does not by itself prove anatomical atlas
validity. For that, use known-correspondence validation: identical meshes,
rigid rotations, global scale changes, radial expansions, and localized bulges
should produce small transition-map residuals after the declared gauge.

The maintained validation runner can exercise both backends with the same
synthetic known-correspondence cases:

```bash
python scripts/run_spherical_atlas_validation.py --out /tmp/atlas-stereo --quick --parameterizer stereographic
python scripts/run_spherical_atlas_validation.py --out /tmp/atlas-conformal --quick --parameterizer conformal
```

The runner writes its report and figures to the requested output directory. Put
that output outside the repository unless you intentionally want to archive a
specific result.

## Outputs

Each run writes:

- `physical_mesh.ply`
- `closed_augmented_mesh.ply`
- `sphere_raw.ply`
- `sphere_mobius.ply`
- `sphere_canonical.ply`
- `sphere_physical_support.ply`
- `physical_mesh_display.json`
- `sphere_display.json`
- `sphere_augmented_qc.json`
- `features_grid.npz`
- `sh_coeffs.npz`
- `canonical_result.json`
- `quality_report.json`

`sphere_display.json` is a lightweight indexed sphere with vertex fields. It is
the artifact intended for browser viewers and other interactive inspection
tools. The statistical and learning-ready data remain in `features_grid.npz`,
`sh_coeffs.npz`, and the quality JSON files. These artifacts include schema
versions and metadata for channel order, units, orientation signal, harmonic
bandwidth, and runtime configuration.

## Dashboard Use

Batch mode:

```bash
sphere-map canonicalize input.stl --out /outside/repo/dashboard-dataset/cases/CASE001
```

Lazy local mode:

1. Install this package in the Python environment that the dashboard preview
   server will use.
2. Put private source mesh paths in an external dataset manifest, outside Git.
3. Start the dashboard with its cache directory outside the source checkout.

A dashboard server may call:

```bash
sphere-map canonicalize /path/to/source.stl --out /path/to/cache/run
```

and then serves the generated `sphere_display.json`.

## Tests

```bash
python3 -m pytest tests -q
python3 -m sphere_mapping_pipeline.cli --help
sphere-map --help
```

The end-to-end synthetic test uses the radial parameterizer for speed. The
package dependency set still installs `lapy` so the real conformal backend is
available for actual runs. The stereographic backend and the shared validation
runner are covered by focused tests:

```bash
python3 -m pytest tests/test_stereographic_parameterizer.py tests/test_validation_runner.py tests/test_validation_cli.py -q
```
