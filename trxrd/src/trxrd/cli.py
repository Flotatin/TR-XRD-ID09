"""Command line: `trxrd build | auto | compare | convert | info`."""

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


def _frames(text: Optional[str], n: int) -> range:
    if not text:
        return range(n)
    start, _, stop = text.partition(":")
    return range(n)[slice(int(start) if start else None, int(stop) if stop else None)]


def _add_library(run, paths) -> None:
    from trxrd.io.jcpds import read_library

    for name, card in read_library(paths or []).items():
        run.phases.setdefault(name, card)  # the cards already in the run (those really used) win


def _auto_options(args: argparse.Namespace):
    from trxrd.processing.peaks import DetectionOptions
    from trxrd.processing.phase_search import SearchOptions
    from trxrd.services.identification import AutoOptions

    return AutoOptions(
        detection=DetectionOptions(min_snr=args.min_snr),
        search=SearchOptions(
            pressure_range=(args.pmin, args.pmax), tolerance=args.tolerance, max_phases=args.max_phases,
            max_pressure_step=args.max_step, max_pressure_spread=args.max_spread,
        ),
        window=(args.tth_min, args.tth_max) if args.tth_min is not None and args.tth_max is not None else None,
        exclude=[tuple(float(v) for v in zone.split(":")) for zone in args.exclude or []] or None,
        track=not args.no_track,
    )


def _report_static_peaks(run, frames, options) -> None:
    from trxrd.services.identification import find_static_peaks

    if len(frames) < 5:
        return
    for peak in find_static_peaks(run, frames, options.window, options.detection):
        if not any(lo <= peak.position <= hi for lo, hi in options.exclude or ()):
            print(f"note: peak at {peak.position:.3f}° does not move (seen in {peak.fraction:.0%} of the spectra, "
                  f"±{peak.spread:.3f}°): gasket/diamond? if so use --exclude {peak.zone()[0]}:{peak.zone()[1]}")


def _phases_text(pressures) -> str:
    return ", ".join(f"{name} {p:.3f}" for name, p in sorted(pressures.items())) or "-"


def _cmd_auto(args: argparse.Namespace) -> int:
    from trxrd.io.h5store import load_run, save_run
    from trxrd.services.identification import auto_analyse

    run = load_run(args.run)
    _add_library(run, args.library)
    options = _auto_options(args)
    frames = _frames(args.frames, len(run))
    _report_static_peaks(run, frames, options)
    for result in auto_analyse(run, frames, args.phases or None, options):
        spectrum = run.spectra[result.index]
        found = ", ".join(f"{f.phase} {f.pressure:.3f}±{f.pressure_err:.3f}" for f in spectrum.phases) or "-"
        mode = "tracked" if result.tracked else "search "
        print(f"{result.index:5d}  t={spectrum.time:.6g}  {mode} {len(result.peaks):3d} peaks  {found}")
    print(save_run(run, args.out or args.run))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    import copy

    from trxrd.io.h5store import load_run
    from trxrd.io.legacy import is_legacy_file, legacy_to_run, load_legacy
    from trxrd.services.identification import auto_analyse
    from trxrd.services.validation import compare_runs

    path = Path(args.reference)
    reference = legacy_to_run(load_legacy(path), path.stem) if is_legacy_file(path) else load_run(path)
    run = copy.deepcopy(reference)
    for spectrum in run.spectra:
        spectrum.phases, spectrum.model = [], None
    _add_library(run, args.library)
    options = _auto_options(args)
    frames = _frames(args.frames, len(run))
    _report_static_peaks(run, frames, options)
    results = {r.index: r for r in auto_analyse(run, frames, args.phases or None, options)}
    summary = compare_runs(reference, run, frames)
    for item in summary.spectra:
        mode = "T" if results[item.index].tracked else "S"
        mark = "==" if item.same_phases else "!="
        print(f"{item.index:5d} {mode} {mark}  reference: {_phases_text(item.reference):45s} found: {_phases_text(item.found)}")
    print(f"same phases in {summary.agreement}/{len(summary.spectra)} spectra")
    for phase, delta in sorted(summary.max_pressure_difference().items()):
        print(f"  max |ΔP| {phase}: {delta:.3f} GPa")
    return 0


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
    from trxrd import __version__

    parser = argparse.ArgumentParser(prog="trxrd", description=f"TR-XRD v2 ({__version__}): time-resolved XRD runs (HDF5)")
    parser.add_argument("--version", action="version", version=f"trxrd {__version__}")
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

    def automatic_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--phases", nargs="*", help="candidate phases (default: every phase of the run)")
        command.add_argument("--library", nargs="*",
                             help=".jcpds files or folders: extra candidate phases (cards already in the run are kept)")
        command.add_argument("--pmin", type=float, default=0.0, help="GPa")
        command.add_argument("--pmax", type=float, default=30.0, help="GPa")
        command.add_argument("--frames", help="spectra to analyse, python slice, e.g. 10:200")
        command.add_argument("--max-phases", type=int, default=3)
        command.add_argument("--tolerance", type=float, default=0.1, help="2θ matching tolerance (deg)")
        command.add_argument("--min-snr", type=float, default=5.0, help="detection threshold (local noise sigmas)")
        command.add_argument("--max-step", type=float, help="max pressure change between spectra (GPa)")
        command.add_argument("--max-spread", type=float, default=1.5,
                             help="soft limit on the pressure difference between phases (GPa)")
        command.add_argument("--exclude", nargs="*", help="2θ zones to ignore, e.g. 18.35:18.85 (gasket, diamond...)")
        command.add_argument("--no-track", action="store_true", help="run the phase search on every spectrum")
        command.add_argument("--tth-min", type=float)
        command.add_argument("--tth-max", type=float)

    auto = sub.add_parser("auto", help="detect peaks, identify phases and pressures, fit (Pearson IV)")
    auto.add_argument("run", help=".trxrd.h5 run file (updated in place unless -o)")
    automatic_options(auto)
    auto.add_argument("-o", "--out", help="output file")
    auto.set_defaults(func=_cmd_auto)

    compare = sub.add_parser("compare", help="re-analyse a run automatically and compare with its existing analysis")
    compare.add_argument("reference", help="legacy .CEDX or .trxrd.h5 file holding the reference analysis (not modified)")
    automatic_options(compare)
    compare.set_defaults(func=_cmd_compare)

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
