"""Use cases: manual work on the phases of the spectra, after or instead of the automatic analysis.

Add a phase at a starting pressure, move it, remove it; fit the spectrum;
fit a series from the current spectrum outwards, each spectrum starting from
its neighbour; copy the model of a spectrum onto others. A phase is placed on
the peaks the detection finds (position, height, width): the fit starts from
what is really there, with the settings of the automatic analysis.
"""

from __future__ import annotations

import copy
import dataclasses
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from trxrd.domain import Peak, PhaseCard, PhaseFit, Run, Spectrum
from trxrd.physics import profiles
from trxrd.processing.peaks import DetectedPeak
from trxrd.services.analysis import AnalysisSettings
from trxrd.services.fitting import fit_spectrum, refresh_model
from trxrd.services.identification import Progress, analysis_window, detect

STRONG_LINE = 10.0  # % of the card: a line this strong is kept even when no peak is detected there


def _lines(card: PhaseCard, pressure: float, energy_ev: float, temperature: float,
           window: Tuple[float, float]) -> List[tuple]:
    """(2θ, reflection) inside `window`, strongest first."""
    lines = card.two_theta_lines(pressure, energy_ev, temperature, max_lines=len(card.reflections))
    inside = [(t, r) for t, r in lines if np.isfinite(t) and window[0] <= t <= window[1]]
    return sorted(inside, key=lambda line: -line[1].intensity)


def place_peaks(card: PhaseCard, pressure: float, energy_ev: float, temperature: float, window: Tuple[float, float],
                detected: Sequence[DetectedPeak], taken: Sequence[float], tolerance: float,
                model: str, width: float) -> List[Peak]:
    """Peaks of `card` at `pressure`, on the detected peaks within `tolerance` of its lines.

    A detected peak goes to the strongest line near it: sharing it would make
    the fit degenerate. So do the peaks already explained by another phase
    (`taken`, 2θ): those lines are left out. Strong lines without any
    detected peak are kept at their expected position.
    """
    shape = profiles.model(model).shape_default
    free = [p for p in detected if not any(abs(p.position - t) <= tolerance for t in taken)]
    peaks = []
    for two_theta, reflection in _lines(card, pressure, energy_ev, temperature, window):
        name = f"{card.name}_{reflection.label}"
        match = min(free, key=lambda p: abs(p.position - two_theta), default=None)
        if match is not None and abs(match.position - two_theta) <= tolerance:
            free.remove(match)
            fwhm = match.fwhm if np.isfinite(match.fwhm) else None
            peaks.append(Peak(name, model, center=match.position, height=match.height, shape=shape, hkl=reflection.hkl,
                              width=profiles.width_from_fwhm(model, fwhm, shape) if fwhm else width))
        elif reflection.intensity >= STRONG_LINE and not any(abs(two_theta - t) <= tolerance for t in taken):
            peaks.append(Peak(name, model, center=two_theta, width=width, shape=shape, hkl=reflection.hkl))
    return sorted(peaks, key=lambda p: p.center)


def add_phase(run: Run, index: int, phase: str, pressure: float, settings: Optional[AnalysisSettings] = None) -> PhaseFit:
    """Put `phase` on spectrum `index` at `pressure` (it replaces the phase if already there). Not fitted yet."""
    if run.calibration is None:
        raise ValueError("Le run n'a pas de calibration (énergie inconnue)")
    settings = settings or AnalysisSettings()
    options = settings.options()
    spectrum = run.spectra[index]
    window = analysis_window(run, options)
    detected = detect(run, index, window, options.detection)
    taken = [p.center for f in spectrum.phases if f.phase != phase for p in f.peaks]
    taken += [p.center for p in spectrum.free_peaks]
    temperature = options.fit.temperature
    peaks = place_peaks(run.phases[phase], float(pressure), run.calibration.energy_ev, temperature, window, detected,
                        taken, options.search.tolerance, options.fit.model, options.fit.width)
    fit = PhaseFit(phase=phase, pressure=float(pressure), temperature=temperature, peaks=peaks)
    names = [f.phase for f in spectrum.phases]
    if phase in names:
        spectrum.phases[names.index(phase)] = fit
    else:
        spectrum.phases.append(fit)
    spectrum.fit_range = spectrum.fit_range or window
    refresh_model(spectrum)
    return fit


def remove_phase(run: Run, index: int, phase: str) -> None:
    spectrum = run.spectra[index]
    spectrum.phases = [f for f in spectrum.phases if f.phase != phase]
    refresh_model(spectrum)


def fit_current(run: Run, index: int, settings: Optional[AnalysisSettings] = None) -> Spectrum:
    """Fit spectrum `index` from the peaks it holds."""
    spectrum = run.spectra[index]
    if not spectrum.phases:
        raise ValueError("Aucune phase sur ce spectre : en ajouter une avant d'ajuster")
    return fit_spectrum(run, index, options=(settings or AnalysisSettings()).options().fit)


@dataclass
class FitSeries:
    """Result of `fit_series`: fitted copies of the spectra, to put into the run with `apply_to`."""

    run: Run  # the run fitted (not modified)
    spectra: Dict[int, Spectrum]
    elapsed: float  # s
    stopped: bool = False

    @property
    def indices(self) -> List[int]:
        return sorted(self.spectra)

    def apply_to(self, run: Run) -> None:
        for index, spectrum in self.spectra.items():
            run.spectra[index] = spectrum


def fit_series(run: Run, start: int, indices: Iterable[int], settings: Optional[AnalysisSettings] = None,
               follow: bool = True, progress: Optional[Progress] = None) -> FitSeries:
    """Fit spectrum `start` and the spectra `indices`, on copies: `run` is not modified.

    With `follow`, the model goes outwards from `start` (later spectra in
    increasing order, earlier ones in decreasing order), each spectrum
    starting from its neighbour: this tracks a compression or a decompression.
    Otherwise each spectrum starts from its own model, and those without any
    phase are left out. When `progress` stops it, what is done is returned.
    """
    began = time.perf_counter()
    options = (settings or AnalysisSettings()).options().fit
    chosen = sorted({int(i) for i in indices} | {int(start)})
    if not run.spectra[start].phases and follow:
        raise ValueError("Aucune phase sur le spectre courant : rien à propager")
    work = dataclasses.replace(run, spectra=list(run.spectra))
    for index in chosen:
        work.spectra[index] = copy.deepcopy(run.spectra[index])
    later = [i for i in chosen if i > start]
    earlier = [i for i in reversed(chosen) if i < start]
    steps = [(start, None)]
    if follow:
        steps += list(zip(later, [start] + later)) + list(zip(earlier, [start] + earlier))
    else:
        steps += [(i, None) for i in later + earlier]
    done: Dict[int, Spectrum] = {}
    stopped = False
    for count, (index, previous) in enumerate(steps, start=1):
        spectrum = work.spectra[index]
        if previous is not None:
            source = work.spectra[previous]
            spectrum.phases = copy.deepcopy(source.phases)
            spectrum.free_peaks = copy.deepcopy(source.free_peaks)
            spectrum.fit_range = source.fit_range
        if spectrum.phases:
            fit_spectrum(work, index, options=options)
            done[index] = spectrum
        if progress is not None and not progress(count, len(steps)):
            stopped = True
            break
    return FitSeries(run, done, time.perf_counter() - began, stopped)


@dataclass
class Model:
    """Phases and free peaks of a spectrum, to put on others."""

    source: int
    phases: List[PhaseFit]
    free_peaks: List[Peak]
    fit_range: Optional[Tuple[float, float]]


def copy_model(run: Run, index: int) -> Model:
    spectrum = run.spectra[index]
    return Model(index, copy.deepcopy(spectrum.phases), copy.deepcopy(spectrum.free_peaks), spectrum.fit_range)


def paste_model(run: Run, model: Model, indices: Iterable[int]) -> List[int]:
    """Give `model` to the spectra `indices` (not fitted: their curve is the model as it is)."""
    indices = list(indices)
    for index in indices:
        spectrum = run.spectra[index]
        spectrum.phases = copy.deepcopy(model.phases)
        spectrum.free_peaks = copy.deepcopy(model.free_peaks)
        spectrum.fit_range = model.fit_range
        refresh_model(spectrum)
    return indices


def clear_model(run: Run, indices: Iterable[int]) -> List[int]:
    """Remove the phases and the free peaks of the spectra `indices`."""
    indices = list(indices)
    for index in indices:
        spectrum = run.spectra[index]
        spectrum.phases, spectrum.free_peaks, spectrum.model = [], [], None
    return indices
