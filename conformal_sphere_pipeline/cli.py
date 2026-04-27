"""Command-line interface for the conformal sphere pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .pipeline import ConformalSphereConfig, canonicalize_mesh_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="conformal-sphere")
    sub = parser.add_subparsers(dest="command", required=True)

    canonicalize = sub.add_parser("canonicalize", help="canonicalize one triangle mesh")
    canonicalize.add_argument("input", type=Path)
    canonicalize.add_argument("--out", required=True, type=Path)
    canonicalize.add_argument("--config", default=None)
    canonicalize.add_argument("--no-virtual-buffer", action="store_true")
    canonicalize.add_argument("--orientation-signal", choices=["log_conformal_factor", "radial", "combined"], default=None)
    canonicalize.add_argument("--lmax-orientation", type=int)
    canonicalize.add_argument("--nlat", type=int)
    canonicalize.add_argument("--nlon", type=int)
    canonicalize.add_argument("--parameterizer", choices=["auto", "conformal", "radial"], default=None)
    canonicalize.add_argument("--fail-on-warning", action="store_true")
    canonicalize.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if getattr(args, "verbose", False) else logging.WARNING)

    if args.command == "canonicalize":
        try:
            cfg = ConformalSphereConfig.from_yaml(args.config)
            if args.no_virtual_buffer:
                cfg.virtual_buffer_enabled = False
            if args.orientation_signal is not None:
                cfg.orientation_signal = args.orientation_signal
            if args.lmax_orientation is not None:
                cfg.lmax_orientation = args.lmax_orientation
            if args.nlat is not None:
                cfg.nlat = args.nlat
            if args.nlon is not None:
                cfg.nlon = args.nlon
            if args.parameterizer is not None:
                cfg.parameterizer = args.parameterizer
            if args.fail_on_warning:
                cfg.fail_on_warning = True

            result = canonicalize_mesh_file(args.input, args.out, config=cfg)
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            parser.exit(1, f"error: {exc}\n")
        topo = result.quality["topology"]
        print(f"Loaded mesh: {len(result.physical_vertices):,} vertices, {len(result.physical_faces):,} faces")
        print(f"Boundary loops: {len(result.boundary_info['loops'])}")
        print(f"Virtual closure: watertight {'yes' if topo['is_watertight'] else 'no'}, genus {topo['genus']}")
        print(f"Parameterization: method {result.quality['parameterization']['method']}")
        print(f"Mobius centroid: {result.mobius_info['final_centroid_norm']:.3e}")
        print(
            "Orientation: "
            f"C21 rel {result.orientation_info['orientation_c21_rel']:.3e}, "
            f"Im(C22) rel {result.orientation_info['orientation_c22_imag_rel']:.3e}"
        )
        print(f"Wrote {args.out / 'features_grid.npz'}")
        return 0

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
