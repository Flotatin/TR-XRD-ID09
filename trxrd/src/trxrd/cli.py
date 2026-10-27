"""Command line: `trxrd build | convert | info`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional


def _cmd_build(args: argparse.Namespace) -> int:
    from trxrd.io.calibration import calibration_from_files
    from trxrd.io.h5store import save_run
    from trxrd.io.jcpds import read_library
    from trxrd.services.run_builder import build_run, output_path

    calibration = calibration_from_files(
        args.poni,
        args.mask,
        two_theta_range=(args.tth_min, args.tth_max),
        points_per_degree=args.points_per_degree,
        energy_ev=args.energy,
        flip_axis=args.flip,
    )
    run = build_run(
        args.scan,
        calibration,
        name=args.name,
        phases=read_library(args.phases) if args.phases else None,
        time_mode=args.time,
        frame_period=args.period,
    )
    dst = Path(args.out) if args.out and Path(args.out).suffix == ".h5" else output_path(run, args.out or ".")
    dst.parent.mkdir(parents=True, exist_ok=True)
    save_run(run, dst)
    print(f"{dst}  ({len(run)} spectra)")
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    from trxrd.io.legacy import convert_file, is_legacy_file

    sources: List[Path] = []
    for item in map(Path, args.paths):
        if item.is_dir():
            sources.extend(p for p in sorted(item.iterdir()) if is_legacy_file(p))
        else:
            sources.append(item)
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
    failures = 0
    for src in sources:
        dst = Path(args.out) / f"{src.stem}.trxrd.h5" if args.out else None
        try:
            print(f"{src} -> {convert_file(src, dst)}")
        except Exception as exc:  # keep going on a batch of old files
            failures += 1
            print(f"FAILED {src}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 1 if failures else 0


def _cmd_info(args: argparse.Namespace) -> int:
    from trxrd.notebook import XRDRun

    for path in args.paths:
        run = XRDRun(path)
        print(run)
        if run.note:
            print("  note:", run.note)
        print(run.summary.describe().T.to_string() if len(run) else "  (empty)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trxrd", description="Time-resolved XRD runs (HDF5)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="integrate a raw LIMA scan into a new run file")
    build.add_argument("scan", help="raw image stack, e.g. .../scan0027/scan_jf1m_0000.h5")
    build.add_argument("--poni", required=True)
    build.add_argument("--mask")
    build.add_argument("--tth-min", type=float, default=0.0)
    build.add_argument("--tth-max", type=float, default=40.0)
    build.add_argument("--points-per-degree", type=int, help="2θ sampling (default: detector pixel size)")
    build.add_argument("--energy", type=float, help="eV (default: wavelength of the .poni)")
    build.add_argument("--flip", type=int, choices=(0, 1, 2))
    build.add_argument("--phases", nargs="*", help=".jcpds files or folders to embed")
    build.add_argument("--time", choices=("trigger", "period", "none"), default="trigger")
    build.add_argument("--period", type=float, default=1e-3, help="frame period (s) for --time period")
    build.add_argument("--name")
    build.add_argument("-o", "--out", help="output folder or .h5 file")
    build.set_defaults(func=_cmd_build)

    convert = sub.add_parser("convert", help="convert legacy dill files (.CEDX...) to HDF5")
    convert.add_argument("paths", nargs="+", help="files or folders")
    convert.add_argument("-o", "--out", help="output folder (default: next to each file)")
    convert.set_defaults(func=_cmd_convert)

    info = sub.add_parser("info", help="print the content of run files")
    info.add_argument("paths", nargs="+")
    info.set_defaults(func=_cmd_info)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
