"""Use case: the automatic analysis of a run as the user asks for it (GUI dialog, `trxrd auto` / `compare`).

The user's choices (`AnalysisSettings`: candidate phases, pressure range,
excluded zones...) become the options of `identification.auto_analyse`. In
the GUI the analysis runs in a worker thread on a copy of the analysed
spectra (`analyse`); the result is merged into the run afterwards, in the
GUI thread (`Analysis.apply_to`), so that the views never read a spectrum
being fitted. The choices are kept in the run (`Run.ui_state`) for the next
analysis, excluded zones included.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from trxrd.domain import PhaseCard, Run, Spectrum
from trxrd.io.jcpds import read_library
from trxrd.physics.eos import T_REF
from trxrd.processing.peaks import DetectionOptions
from trxrd.processing.phase_search import SearchOptions
from trxrd.services.fitting import FitOptions
from trxrd.services.identification import (
    AutoOptions, AutoResult, Progress, StaticPeak, auto_analyse, find_static_peaks, screen_static_peaks,
)

PathLike = Union[str, Path]
UI_KEY = "auto_analysis"  # Run.ui_state entry holding the last settings
STATIC_SPECTRA = 200  # spectra looked at for the static peaks of a long series (evenly spaced)
Zone = Tuple[float, float]


@dataclass
class AnalysisSettings:
    """What the user chooses; JSON-able (kept in Run.ui_state)."""

    phases: Optional[List[str]] = None  # candidate phases (None: every phase of the run)
    library: List[str] = field(default_factory=list)  # .jcpds files or folders: extra candidates
    pressure_range: Zone = (0.0, 30.0)  # GPa
    temperature: Optional[float] = None  # K, for the search and the fit (None: 298 K)
    tolerance: float = 0.1  # 2θ matching tolerance (deg)
    max_phases: int = 3
    min_snr: float = 5.0  # detection threshold (local noise sigmas)
    max_fwhm: float = 1.5  # deg: wider bumps are not detected (background, amorphous)
    noise_window: float = 1.5  # deg, window of the local noise and level of the detection
    shoulders: bool = True  # detect shoulders (peaks without their own maximum)
    # peaks broader than this (deg) neither explain nor cost anything in the phase search (a liquid or amorphous
    # hump next to crystalline lines); None: every peak counts (to identify a liquid from its broad peaks)
    broad_fwhm: Optional[float] = 0.5
    # maxima searched on the filtered signal of the spectrum (what is shown) rather than on a Savitzky-Golay
    # filter of 7 points: 805 instead of 797 spectra with the phases of the manual analysis on 29 runs of
    # January 2026 (docs/ARCHITECTURE.md §6.5)
    spectrum_filter: bool = True
    max_step: Optional[float] = None  # largest pressure change between two spectra (GPa)
    max_spread: float = 1.5  # soft limit on the pressure difference between phases (GPa)
    exclude: List[Zone] = field(default_factory=list)  # 2θ zones of no phase (gasket, diamond...)
    window: Optional[Zone] = None  # 2θ range analysed (None: whole pattern)
    track: bool = True  # peak tracking
    peak_model: str = "PearsonIV"  # profile of the new peaks
    fit_window: float = 5.0  # fit of one peak: points within this many widths of its centre
    fit_variation: Optional[float] = None  # freedom of the widths in a fit, 1.2 = 120 % (None: 3)
    fit_center_tolerance: float = 0.3  # deg: how far a peak may move in a manual fit

    def options(self) -> AutoOptions:
        temperature = self.temperature or T_REF  # search and fit at the same T
        return AutoOptions(
            detection=DetectionOptions(min_snr=self.min_snr, max_fwhm=self.max_fwhm, noise_window=self.noise_window,
                                       shoulders=self.shoulders, spectrum_filter=self.spectrum_filter),
            fit=FitOptions(model=self.peak_model, temperature=temperature, variation=self.fit_variation,
                           center_tolerance=self.fit_center_tolerance),
            search=SearchOptions(
                pressure_range=tuple(self.pressure_range), tolerance=self.tolerance, max_phases=self.max_phases,
                max_pressure_step=self.max_step, max_pressure_spread=self.max_spread, temperature=temperature,
                max_fwhm=self.broad_fwhm or None,
            ),
            window=tuple(self.window) if self.window is not None else None,
            exclude=[tuple(zone) for zone in self.exclude] or None,
            track=self.track,
        )

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AnalysisSettings":
        """Settings saved by another version: unknown keys are ignored, JSON lists become tuples."""
        known = {f.name for f in dataclasses.fields(cls)}
        values = {k: v for k, v in (data or {}).items() if k in known}
        for key in ("pressure_range", "window"):
            if values.get(key) is not None:
                values[key] = tuple(float(v) for v in values[key])
        values["exclude"] = [tuple(float(v) for v in zone) for zone in values.get("exclude") or []]
        return cls(**values)


def merged_zones(zones: Iterable[Zone]) -> List[Zone]:
    """Zones in order, those that overlap made into one."""
    merged: List[Zone] = []
    for low, high in sorted(tuple(sorted(float(v) for v in zone)) for zone in zones):
        if merged and low <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], high))
        else:
            merged.append((low, high))
    return merged


def saved_settings(run: Run) -> AnalysisSettings:
    """The last settings of this run; the temperature defaults to that of its existing fits (old analyses)."""
    settings = AnalysisSettings.from_dict(run.ui_state.get(UI_KEY))
    if settings.temperature is None:
        temperatures = [f.temperature for s in run.spectra for f in s.phases if np.isfinite(f.temperature)]
        if temperatures:
            settings.temperature = float(np.median(temperatures))
    return settings


def add_library(run: Run, paths: Iterable[PathLike]) -> List[str]:
    """Add the cards of `paths` to the candidate phases of `run`; returns their names.

    The cards already in the run (those really used) win, but the P range,
    a prior of the search only, is the library's.
    """
    cards = read_library(list(paths)) if paths else {}
    for name, card in cards.items():
        own = run.phases.setdefault(name, card)
        if own is not card and card.p_range is not None:
            own.p_range = card.p_range
    return list(cards)


def library_phases(paths: Iterable[PathLike]) -> List[Tuple[str, Optional[Zone]]]:
    """(name, P range) of the cards found in `paths`, for a list of candidates."""
    return [(name, card.p_range) for name, card in read_library(list(paths)).items()] if paths else []


def static_peaks(run: Run, indices: Sequence[int], options: AutoOptions,
                 max_spectra: int = STATIC_SPECTRA) -> List[StaticPeak]:
    """After an analysis: peaks that do not move and are no line of the sample (gasket, diamond...).

    Candidates for excluded zones; those already excluded are not repeated.
    A long series is sampled (`max_spectra` evenly spaced spectra).
    """
    indices = list(indices)
    if len(indices) < 5:
        return []
    sample = indices[::max(1, math.ceil(len(indices) / max_spectra))]
    found = find_static_peaks(run, sample, options.window, options.detection)
    return [peak for peak in screen_static_peaks(run, found, indices)
            if not any(lo <= peak.position <= hi for lo, hi in options.exclude or ())]


@dataclass
class Analysis:
    """Result of `analyse`: analysed copies of the spectra, to merge into the run with `apply_to`."""

    run: Run  # the run analysed (not modified)
    settings: AnalysisSettings
    indices: List[int]  # spectra analysed, in order (fewer than asked when stopped)
    results: List[AutoResult]
    spectra: Dict[int, Spectrum]
    phases: Dict[str, PhaseCard]  # cards of the run after the analysis (library added)
    static: List[StaticPeak]
    elapsed: float  # s
    stopped: bool = False

    @property
    def tracked(self) -> int:
        return sum(r.tracked for r in self.results)

    def phase_counts(self) -> Dict[str, int]:
        """Spectra where each phase was found, in order of first appearance."""
        counts: Dict[str, int] = {}
        for index in self.indices:
            for fit in self.spectra[index].phases:
                counts[fit.phase] = counts.get(fit.phase, 0) + 1
        return counts

    def apply_to(self, run: Run) -> None:
        """Put the analysed spectra and the cards into `run`, and remember the settings (GUI thread)."""
        for index, spectrum in self.spectra.items():
            run.spectra[index] = spectrum
        run.phases = self.phases
        run.ui_state[UI_KEY] = self.settings.to_dict()


def analyse(run: Run, indices: Iterable[int], settings: AnalysisSettings,
            progress: Optional[Progress] = None) -> Analysis:
    """auto_analyse on a copy of the spectra `indices` (and of the cards): `run` is not modified.

    When `progress` stops it, the spectra already analysed are returned
    (`stopped`), and no static peak is looked for.
    """
    start = time.perf_counter()
    indices = list(indices)
    work = dataclasses.replace(run, spectra=list(run.spectra),
                               phases={name: copy.copy(card) for name, card in run.phases.items()})
    for index in indices:
        work.spectra[index] = copy.deepcopy(run.spectra[index])
    add_library(work, settings.library)
    phases = None if settings.phases is None else [name for name in settings.phases if name in work.phases]
    if phases is not None and not phases:
        raise ValueError("Aucune phase candidate : en cocher au moins une")
    options = settings.options()
    stopped = []

    def watched(done: int, total: int) -> bool:
        keep_going = progress is None or progress(done, total)
        if not keep_going:
            stopped.append(done)
        return keep_going

    results = auto_analyse(work, indices, phases, options, watched)
    done = [r.index for r in results]
    found = [] if stopped else static_peaks(work, done, options)
    return Analysis(run, settings, done, results, {i: work.spectra[i] for i in done}, work.phases, found,
                    time.perf_counter() - start, bool(stopped))
