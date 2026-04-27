"""Conformal spherical parameterization and canonicalization tools."""

from .io import load_triangle_mesh
from .mesh_qc import validate_triangle_mesh
from .pipeline import ConformalSphereConfig, ConformalSphereResult, canonicalize_mesh_file

__all__ = [
    "ConformalSphereConfig",
    "ConformalSphereResult",
    "__version__",
    "canonicalize_mesh_file",
    "load_triangle_mesh",
    "validate_triangle_mesh",
]

__version__ = "0.1.0"
