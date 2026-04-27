"""Small topology wrappers used by the conformal sphere pipeline."""

from .mesh_qc import MeshQCReport, edge_incidence, validate_triangle_mesh

__all__ = ["MeshQCReport", "edge_incidence", "validate_triangle_mesh"]
