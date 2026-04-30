"""Spherical parameterization and canonicalization tools."""

from .io import load_triangle_mesh
from .mesh_qc import validate_triangle_mesh
from .pipeline import ConformalSphereConfig, ConformalSphereResult, canonicalize_mesh_file
from .spherical.parameterize import ParameterizationResult, parameterize_sphere
from .spherical.stereographic import StereographicConfig

__all__ = [
    "ConformalSphereConfig",
    "ConformalSphereResult",
    "ParameterizationResult",
    "StereographicConfig",
    "__version__",
    "canonicalize_mesh_file",
    "load_triangle_mesh",
    "parameterize_sphere",
    "validate_triangle_mesh",
]

__version__ = "0.1.0"
