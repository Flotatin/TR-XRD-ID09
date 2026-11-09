"""Use case: the times of the acquisition of a run, integration of a frame and time between two frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np

from trxrd.domain import Run
from trxrd.io import esrf
from trxrd.processing import timing
from trxrd.services import raw_data
from trxrd.services.run_builder import ID09_OSCILLOSCOPE

PathLike = Union[str, Path]


@dataclass(frozen=True)
class Acquisition:
    """What a run says about the timing of its frames (s; None: unknown)."""

    exposure: Optional[float] = None  # integration of a frame, as set on the detector
    exposure_from_raw: bool = False  # read from the raw file found now (the run does not store it)
    pulse: Optional[float] = None  # width of a trigger pulse on the oscilloscope
    inter_frame: Optional[float] = None  # end of a pulse to the start of the next, as in the v1
    period: Optional[float] = None  # start of a pulse to the start of the next
    pulses: int = 0  # trigger pulses on the oscilloscope (cut ones included)
    step: Optional[float] = None  # sampling step of the oscilloscope: the precision of the measured times
    regular: bool = True
    channel: str = ""
    pulses_from_raw: bool = False  # oscilloscope of the BLISS master file (the run has none: v1 continuous mode)
    spectra: int = 0

    @property
    def known(self) -> bool:
        return self.exposure is not None or self.inter_frame is not None

    @property
    def triggers(self) -> bool:
        """The pulses are frame triggers: one clock, not faster than the integration. In 41 of the 744 scans of
        January 2026 (D2O_01 1-20, Water01 7-41), the trigger channel holds only noise: thousands of pulses
        3 to 10 µs apart, for 16 to 5000 frames of 10 µs."""
        return (self.inter_frame is not None and self.regular
                and (self.exposure is None or self.period >= self.exposure))


def acquisition_of(run: Run, near: Iterable[PathLike] = ()) -> Acquisition:
    """Exposure stored in the run and pulses of its trigger channel; what the run lacks is read from its raw
    files, if found (runs of format 1.2, .CEDX: `near` as for `raw_data.locate_raw_scan`).

    Measuring the pulses takes 0.1 to 0.2 s on 100 000 points: call it from a worker thread.
    """
    raw = _RawFiles(run, near)
    exposure = float(run.source.exposure_time) if np.isfinite(run.source.exposure_time) else raw.exposure()
    osc = run.oscilloscope
    found, from_raw = None, False
    if osc is not None and osc.trigger_channel in osc.channels and osc.time.size:
        found = osc.trigger_channel, timing.find_pulses(osc.time, osc.channels[osc.trigger_channel])
    if found is None or found[1].start.size < 2:
        from_master = raw.pulses()
        if from_master is not None:
            found, from_raw = from_master, True
    channel, pulses = found if found is not None else ("", None)
    rhythm = timing.pulse_timing(pulses) if pulses is not None else None
    return Acquisition(
        exposure=exposure,
        exposure_from_raw=exposure is not None and not np.isfinite(run.source.exposure_time),
        pulse=rhythm.width if rhythm else None,
        inter_frame=rhythm.gap if rhythm else None,
        period=rhythm.period if rhythm else None,
        pulses=int(pulses.start.size) if rhythm else 0,
        step=rhythm.step if rhythm else None,
        regular=rhythm.regular if rhythm else True,
        channel=channel if rhythm else "",
        pulses_from_raw=from_raw and rhythm is not None,
        spectra=len(run.spectra),
    )


class _RawFiles:
    """The raw image file of a run and its BLISS master file, looked for once, when needed."""

    def __init__(self, run: Run, near: Iterable[PathLike]) -> None:
        self.run, self.near = run, list(near)
        self._image: Optional[Path] = None
        self._looked = False

    def image(self) -> Optional[Path]:
        if not self._looked:
            self._image, self._looked = raw_data.locate_raw_scan(self.run, self.near), True
        return self._image

    def exposure(self) -> Optional[float]:
        if self.image() is None:
            return None
        try:
            return esrf.read_exposure(self.image(), self.run.source.raw_dataset or None)
        except (OSError, KeyError, ValueError):
            return None

    def pulses(self) -> Optional[Tuple[str, timing.Pulses]]:
        if self.image() is None:
            return None
        spec = ID09_OSCILLOSCOPE
        try:
            master, scan = esrf.locate_bliss_master(self.image())
            time, channels = esrf.read_bliss_channels(
                master, scan, {spec.trigger_channel: spec.channels[spec.trigger_channel]}, spec.time_counter)
        except (OSError, KeyError, ValueError):
            return None
        return spec.trigger_channel, timing.find_pulses(time, channels[spec.trigger_channel])


def duration_text(seconds: Optional[float]) -> str:
    """'10 µs', '6.56 µs', '1.013 ms' (4 significant digits: the 1.35 % of a continuous scan stay visible)."""
    if seconds is None or not np.isfinite(seconds):
        return "?"
    if abs(seconds) >= 1:
        return f"{seconds:.4g} s"
    if abs(seconds) >= 1e-3:
        return f"{seconds * 1e3:.4g} ms"
    return f"{seconds * 1e6:.4g} µs"
