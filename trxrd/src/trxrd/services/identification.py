"""Use case: automatic analysis of spectra.

For each spectrum: detect the peaks, find which phases at which pressures
explain them, place Pearson IV peaks on the matched reflections, fit them,
then refine the cell and the pressure of every phase. Along a time series the
previous result is the prior of the next spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from trxrd.domain import Peak, PhaseFit, Run, Spectrum
from trxrd.physics import profiles
from trxrd.processing import background
from trxrd.processing.peaks import DetectedPeak, DetectionOptions, detect_peaks
from trxrd.processing.phase_search import LineTables, SearchOptions, Solution, search_composition
from trxrd.services.fitting import FitOptions, fit_spectrum

Progress = Callable[[int, int], bool]  # (done, total) -> keep going?


@dataclass
class AutoOptions:
    detection: DetectionOptions = field(default_factory=DetectionOptions)
    search: SearchOptions = field(default_factory=SearchOptions)
    fit: FitOptions = field(default_factory=FitOptions)
    window: Optional[Tuple[float, float]] = None  # 2θ range analysed (default: whole pattern)
    follow: bool = True  # the previous spectrum is the prior of the next one


@dataclass
class AutoResult:
    index: int
    peaks: List[DetectedPeak]
    solutions: List[Solution]  # ranked, best first
    applied: Optional[Solution] = None


def analysis_window(run: Run, options: AutoOptions) -> Tuple[float, float]:
    if options.window is not None:
        return tuple(options.window)
    x = np.concatenate([np.asarray(s.two_theta, dtype=float) for s in run.spectra])
    return float(np.nanmin(x)), float(np.nanmax(x))


def detect(run: Run, index: int, window: Tuple[float, float], options: Optional[DetectionOptions] = None) -> List[DetectedPeak]:
    """Peaks of spectrum `index` inside `window` (raw intensity minus background)."""
    spectrum = run.spectra[index]
    if spectrum.baseline is None:
        background.apply(spectrum)
    x = np.asarray(spectrum.two_theta, dtype=float)
    inside = (x >= window[0]) & (x <= window[1])
    return detect_peaks(x[inside], (spectrum.intensity - spectrum.baseline)[inside], options)


def apply_solution(
    run: Run,
    index: int,
    solution: Solution,
    peaks: Sequence[DetectedPeak],
    window: Tuple[float, float],
    options: Optional[FitOptions] = None,
) -> Spectrum:
    """Replace the phases of spectrum `index` by `solution`, then fit them.

    Peaks start at the detected position, height and width; a detected peak
    shared by several lines (overlap) starts each of them at its expected
    position instead.
    """
    options = options or FitOptions()
    shared: Dict[int, int] = {}
    for match in solution.matches:
        shared[match.peak] = shared.get(match.peak, 0) + 1
    best_match = {}  # (phase, reflection) -> closest match
    for match in solution.matches:
        key = (match.phase, match.reflection)
        gap = abs(peaks[match.peak].position - match.two_theta)
        if key not in best_match or gap < best_match[key][0]:
            best_match[key] = (gap, match)

    shape = profiles.model(options.model).shape_default
    fits = []
    for phase, pressure in solution.pressures.items():
        card = run.phases[phase]
        new_peaks = []
        for (name, reflection_index), (_, match) in sorted(best_match.items(), key=lambda kv: kv[1][1].two_theta):
            if name != phase:
                continue
            detected = peaks[match.peak]
            reflection = card.reflections[reflection_index]
            fwhm = detected.fwhm if np.isfinite(detected.fwhm) else np.nan
            new_peaks.append(
                Peak(
                    name=f"{card.name}_{reflection.label}",
                    model=options.model,
                    center=detected.position if shared[match.peak] == 1 else match.two_theta,
                    height=detected.height / shared[match.peak],
                    width=profiles.width_from_fwhm(options.model, fwhm, shape) if np.isfinite(fwhm) else options.width,
                    shape=shape,
                    hkl=reflection.hkl,
                )
            )
        fits.append(PhaseFit(phase=phase, pressure=pressure, temperature=options.temperature, peaks=new_peaks))
    spectrum = run.spectra[index]
    spectrum.phases = fits
    return fit_spectrum(run, index, fit_range=window, options=options)


def auto_analyse(
    run: Run,
    indices: Iterable[int],
    phases: Optional[Sequence[str]] = None,
    options: Optional[AutoOptions] = None,
    progress: Optional[Progress] = None,
) -> List[AutoResult]:
    """Analyse spectra `indices` in order with the candidate `phases` (default: all of the run)."""
    options = options or AutoOptions()
    if run.calibration is None:
        raise ValueError("The run has no calibration (energy needed)")
    names = list(phases) if phases is not None else list(run.phases)
    window = analysis_window(run, options)
    tables = LineTables({n: run.phases[n] for n in names}, run.calibration.energy_ev, window, options.search)
    indices = list(indices)
    results: List[AutoResult] = []
    previous: Optional[Dict[str, float]] = None
    for k, index in enumerate(indices):
        peaks = detect(run, index, window, options.detection)
        solutions = search_composition(peaks, tables, previous if options.follow else None)
        result = AutoResult(index, peaks, solutions)
        if solutions:
            result.applied = solutions[0]
            spectrum = apply_solution(run, index, solutions[0], peaks, window, options.fit)
            found = {f.phase: f.pressure for f in spectrum.phases if np.isfinite(f.pressure)}
            previous = found or previous
        else:
            run.spectra[index].phases = []
            run.spectra[index].model = None
        results.append(result)
        if progress is not None and not progress(k + 1, len(indices)):
            break
    return results
