# conformal-sphere-pipeline

`conformal-sphere-pipeline` is a Python package and CLI for mapping triangle
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
computed.

Supported verification targets are Python 3.11 and 3.12 in CI. Local release
checks may additionally exercise newer workstation Python versions.

## Python API

```python
from conformal_sphere_pipeline import (
    ConformalSphereConfig,
    canonicalize_mesh_file,
    load_triangle_mesh,
    validate_triangle_mesh,
)

config = ConformalSphereConfig(parameterizer="auto", nlat=128, nlon=256)
result = canonicalize_mesh_file("input_mesh.ply", "out/case_001", config=config)
```

Public API:

- `ConformalSphereConfig`
- `ConformalSphereResult`
- `canonicalize_mesh_file`
- `load_triangle_mesh`
- `validate_triangle_mesh`

Parameterization methods:

- `auto`: try the `lapy` conformal backend, then fall back to radial.
- `conformal`: require the `lapy` conformal backend.
- `radial`: use centroid radial projection.

## CLI

```bash
conformal-sphere canonicalize input_mesh.ply --out out/case_001 --parameterizer auto
```

Useful options:

- `--config default`
- `--config configs/canonical_sphere.yaml` when running from a source checkout
- `--parameterizer auto|conformal|radial`
- `--no-virtual-buffer`
- `--orientation-signal log_conformal_factor|radial|combined`
- `--lmax-orientation 16`
- `--nlat 128 --nlon 256`

The default configuration is in `configs/canonical_sphere.yaml`. Its default
orientation bandwidth is `lmax_orientation: 16`; raise it for slower high-band
runs when the cohort analysis needs more angular detail. Installed wheel users
can pass `--config default` to load the packaged copy of the same configuration.

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
conformal-sphere canonicalize input.stl --out /outside/repo/dashboard-dataset/cases/CASE001
```

Lazy local mode:

1. Install this package in the Python environment that the dashboard preview
   server will use.
2. Put private source mesh paths in an external dataset manifest, outside Git.
3. Start the dashboard with its cache directory outside the source checkout.

A dashboard server may call:

```bash
conformal-sphere canonicalize /path/to/source.stl --out /path/to/cache/run
```

and then serves the generated `sphere_display.json`.

## Tests

```bash
python3 -m pytest tests -q
python3 -m conformal_sphere_pipeline.cli --help
conformal-sphere --help
```

The end-to-end synthetic test uses the radial parameterizer for speed. The
package dependency set still installs `lapy` so the real conformal backend is
available for actual runs.
