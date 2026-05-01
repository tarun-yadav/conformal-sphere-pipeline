"""Compatibility shim for the renamed :mod:`sphere_mapping_pipeline` package."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys

import sphere_mapping_pipeline as _real_package
from sphere_mapping_pipeline import *  # noqa: F401,F403


class _RenamedPackageLoader(importlib.abc.Loader):
    def __init__(self, real_name: str):
        self.real_name = real_name
        self.real_spec = importlib.util.find_spec(real_name)

    def create_module(self, spec):
        module = importlib.import_module(self.real_name)
        sys.modules[spec.name] = module
        return module

    def exec_module(self, module) -> None:
        if self.real_spec is not None:
            module.__spec__ = self.real_spec
            module.__loader__ = self.real_spec.loader
        module.__package__ = self.real_name.rpartition(".")[0]
        return None


class _RenamedPackageFinder(importlib.abc.MetaPathFinder):
    prefix = "conformal_sphere_pipeline."
    real_prefix = "sphere_mapping_pipeline."

    def find_spec(self, fullname: str, path, target=None):
        if not fullname.startswith(self.prefix):
            return None
        real_name = self.real_prefix + fullname[len(self.prefix) :]
        real_spec = importlib.util.find_spec(real_name)
        if real_spec is None:
            return None
        loader = _RenamedPackageLoader(real_name)
        is_package = real_spec.submodule_search_locations is not None
        spec = importlib.util.spec_from_loader(fullname, loader, is_package=is_package)
        if spec is not None and is_package:
            spec.submodule_search_locations = real_spec.submodule_search_locations
        return spec


if not any(isinstance(finder, _RenamedPackageFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _RenamedPackageFinder())

sys.modules[__name__] = _real_package
