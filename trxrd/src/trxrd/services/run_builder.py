"""Use case: build a new Run from raw ESRF data (images + oscilloscope)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Union

from trxrd.domain import Calibration, Oscilloscope, PhaseCard, Processing, Run, Source, Spectrum
from trxrd.io import esrf
from trxrd.io.calibration import calibration_from_files
from trxrd.io.h5store import save_run
from trxrd.io.jcpds import read_library
from trxrd.processing import background, timing
from trxrd.processing.integration import Integrator

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
Progress = Callable[[int, int], bool]  # (done, total) -> keep going?


@dataclass
class OscilloscopeSpec:
    """Which BLISS counters hold the oscilloscope traces (ID09 defaults)."""

    # names of the v1 (Channel2 = trigger, Channel3 = piezo drive); ch2 exists only in some scans
    channels: Dict[str, str] = field(default_factory=lambda: {"Channel2": "ch5", "Channel3": "ch6", "ch2": "ch2"})
    time_counter: str = "ch5_time"
    trigger_channel: str = "Channel2"
    optional: Sequence[str] = ("ch2",)


ID09_OSCILLOSCOPE = OscilloscopeSpec()


class TimeMode:
    TRIGGER = "trigger"  # detector triggers detected on the oscilloscope
    PERIOD = "period"  # constant frame period
    NONE = "none"


def build_run(
    scan_file: PathLike,
    calibration: Calibration,
    *,
    name: Optional[str] = None,
    processing: Optional[Processing] = None,
    phases: Optional[Dict[str, PhaseCard]] = None,
    frames: Optional[Sequence[int]] = None,
    oscilloscope: Optional[OscilloscopeSpec] = ID09_OSCILLOSCOPE,
    time_mode: str = TimeMode.TRIGGER,
    frame_period: float = 1e-3,
    progress: Optional[Progress] = None,
) -> Run:
    """Integrate every frame of `scan_file` and attach oscilloscope + time base."""
    scan_file = Path(scan_file)
    processing = processing or Processing()
    stack = esrf.ImageStack.open(scan_file)
    integrator = Integrator(calibration)
    indices = list(range(len(stack))) if frames is None else list(frames)

    spectra = []
    for done, (index, image) in enumerate(stack.frames(indices), start=1):
        two_theta, intensity = integrator.integrate(image)
        spectrum = Spectrum(
            two_theta=two_theta,
            intensity=intensity,
            frame=index,
            label=f"frame_{index}",
            processing=Processing(processing.baseline_degree, processing.filter_type, processing.filter_params),
        )
        spectra.append(background.apply(spectrum))
        if progress is not None and not progress(done, len(indices)):
            break

    source = Source(raw_file=str(scan_file.resolve()), raw_dataset=stack.dataset)
    osc = None
    if oscilloscope is not None:
        try:
            master, scan = esrf.locate_bliss_master(scan_file)
            time, channels = esrf.read_bliss_channels(master, scan, oscilloscope.channels, oscilloscope.time_counter,
                                                      oscilloscope.optional)
            osc = Oscilloscope(time=time, channels=channels, trigger_channel=oscilloscope.trigger_channel)
            source.master_file, source.scan = str(master.resolve()), scan
        except (OSError, KeyError, ValueError) as exc:
            logger.warning("No oscilloscope data for %s: %s", scan_file, exc)

    if time_mode == TimeMode.TRIGGER and osc is not None and osc.trigger_channel in osc.channels:
        times = timing.detect_trigger_times(osc.time, osc.channels[osc.trigger_channel], max_triggers=len(stack))
        for spectrum in spectra:
            if 0 <= spectrum.frame < times.size:
                spectrum.time = float(times[spectrum.frame])
        if times.size < len(stack):
            logger.warning("%d triggers found for %d frames", times.size, len(stack))
    elif time_mode == TimeMode.PERIOD:
        for spectrum in spectra:
            spectrum.time = spectrum.frame * frame_period

    return Run(
        name=name or default_run_name(scan_file),
        spectra=spectra,
        calibration=calibration,
        phases=dict(phases or {}),
        source=source,
        oscilloscope=osc,
    )


class Cancelled(Exception):
    """The user stopped the task (its progress callback returned False)."""


def build_and_save(
    scan_file: PathLike,
    folder: PathLike,
    poni_file: PathLike,
    mask_file: Optional[PathLike] = None,
    *,
    two_theta_range=(0.0, 40.0),
    phase_paths: Sequence[PathLike] = (),
    progress: Optional[Progress] = None,
    **options,
) -> Path:
    """build_run from file names (calibration, phase cards), then save `<folder>/<name>.trxrd.h5`.

    Raises Cancelled, and saves nothing, when `progress` asks to stop.
    """
    stopped = []

    def watched(done: int, total: int) -> bool:
        keep_going = progress is None or progress(done, total)
        if not keep_going:
            stopped.append(done)
        return keep_going

    calibration = calibration_from_files(poni_file, mask_file, two_theta_range=two_theta_range)
    phases = read_library(list(phase_paths)) if phase_paths else None
    run = build_run(scan_file, calibration, phases=phases, progress=watched, **options)
    if stopped:
        raise Cancelled(f"stopped after {stopped[0]} frames")
    destination = output_path(run, folder)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return save_run(run, destination)


def default_run_name(scan_file: PathLike) -> str:
    """.../water02_0001/scan0027/scan_jf1m_0000.h5 -> water02_0001_scan0027"""
    scan_file = Path(scan_file)
    if scan_file.parent.name.startswith("scan"):
        return f"{scan_file.parent.parent.name}_{scan_file.parent.name}"
    return scan_file.stem


def output_path(run: Run, folder: PathLike) -> Path:
    return Path(folder) / f"{run.name}.trxrd.h5"

