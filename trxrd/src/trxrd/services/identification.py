"""Use case: automatic analysis of spectra.

For each spectrum: detect the peaks, find which phases at which pressures
explain them, place Pearson IV peaks on the matched reflections, fit them,
then refine the cell and the pressure of every phase. Along a time series the
previous result is the prior of the next spectrum.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from trxrd.domain import Peak, PhaseFit, Run, Spectrum
from trxrd.physics import profiles
from trxrd.processing import background
from trxrd.processing.peaks import DetectedPeak, DetectionOptions, detect_peaks
from trxrd.processing.phase_search import LineTables, Match, SearchOptions, Solution, search_composition
from trxrd.services.fitting import FitOptions, fit_spectrum

Progress = Callable[[int, int], bool]  # (done, total) -> keep going?


@dataclass
class AutoOptions:
    detection: DetectionOptions = field(default_factory=DetectionOptions)
    search: SearchOptions = field(default_factory=SearchOptions)
    fit: FitOptions = field(default_factory=FitOptions)
    window: Optional[Tuple[float, float]] = None  # 2θ range analysed (default: whole pattern)
    # 2θ zones whose peaks belong to no phase (gasket, diamond...): no phase is matched there,
    # but their peaks are still fitted as free peaks so that their tails do not bias the neighbours
    exclude: Optional[Sequence[Tuple[float, float]]] = None
    follow: bool = True  # the previous spectrum is the prior of the next one
    # peak tracking: while no significant peak appears or vanishes, the previous phases and
    # reflection assignments are kept and only refitted (no phase search)
    track: bool = True
    track_tolerance: float = 0.15  # deg, largest shift of a peak between two consecutive spectra
    track_min_snr: float = 10.0  # weaker peaks may come and go without breaking the tracking


@dataclass
class AutoResult:
    index: int
    peaks: List[DetectedPeak]
    solutions: List[Solution]  # ranked, best first
    applied: Optional[Solution] = None
    tracked: bool = False  # True: assignment carried over from the previous spectrum, no search


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
    others: Sequence[DetectedPeak] = (),
) -> Spectrum:
    """Replace the phases of spectrum `index` by `solution`, then fit them.

    Peaks start at the detected position, height and width; a detected peak
    shared by several lines (overlap) starts each of them at its expected
    position instead. Detected peaks explained by no phase, and `others`
    (peaks of the excluded zones), are fitted as free peaks so that they do
    not bias their neighbours.
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
    free = [
        Peak(f"unassigned_{k}", options.model, center=p.position, height=p.height,
             width=profiles.width_from_fwhm(options.model, p.fwhm, shape) if np.isfinite(p.fwhm) else options.width,
             shape=shape)
        for k, p in enumerate([peaks[j] for j in solution.unexplained] + list(others))
    ]
    spectrum = run.spectra[index]
    spectrum.phases = fits
    return fit_spectrum(run, index, fit_range=window, options=options, extra_peaks=free)


def track_peaks(
    previous_peaks: Sequence[DetectedPeak],
    previous: Solution,
    pressures: Dict[str, float],
    peaks: Sequence[DetectedPeak],
    tolerance: float = 0.15,
    min_snr: float = 10.0,
) -> Optional[Solution]:
    """Carry the previous assignment over to `peaks`, or None if the pattern changed.

    The peaks of two consecutive spectra are paired one to one (optimal
    assignment, shifts up to `tolerance`). If a significant peak (SNR >=
    `min_snr`) appeared or vanished, or if a phase lost all its peaks, the
    composition may have changed: None, a full search is needed. Otherwise the
    same phases keep the same reflections, starting from `pressures`.
    """
    if not previous_peaks or not peaks or not previous.matches:
        return None
    before = np.array([p.position for p in previous_peaks])
    now = np.array([p.position for p in peaks])
    shift = np.abs(before[:, None] - now[None, :])
    far = shift > tolerance
    rows, cols = linear_sum_assignment(np.where(far, 1e6, shift))
    pairs = {int(r): int(c) for r, c in zip(rows, cols) if not far[r, c]}
    if any(p.snr >= min_snr and i not in pairs for i, p in enumerate(previous_peaks)):
        return None  # a significant peak vanished
    paired = set(pairs.values())
    if any(p.snr >= min_snr and j not in paired for j, p in enumerate(peaks)):
        return None  # a significant peak appeared
    matches = [
        Match(pairs[m.peak], m.phase, m.reflection, float(now[pairs[m.peak]]))
        for m in previous.matches
        if m.peak in pairs
    ]
    if {m.phase for m in matches} != set(previous.pressures):
        return None  # a phase lost all its peaks
    explained = {m.peak for m in matches}
    return Solution(
        pressures={name: pressures.get(name, p) for name, p in previous.pressures.items()},
        cost=np.nan,
        matches=matches,
        unexplained=[j for j in range(len(peaks)) if j not in explained],
    )


@dataclass
class StaticPeak:
    position: float  # mean 2θ (deg)
    spread: float  # standard deviation of the position along the series (deg)
    fraction: float  # fraction of the spectra where it is detected

    def zone(self, half_width: float = 0.1) -> Tuple[float, float]:
        return (round(self.position - half_width, 3), round(self.position + half_width, 3))


def find_static_peaks(
    run: Run,
    indices: Iterable[int],
    window: Optional[Tuple[float, float]] = None,
    detection: Optional[DetectionOptions] = None,
    tolerance: float = 0.1,
    min_fraction: float = 0.8,
) -> List[StaticPeak]:
    """Peaks found at the same 2θ (within `tolerance`) in most spectra of a series.

    In a run where the pressure changes, a peak that does not move belongs to
    no sample phase (gasket, diamond, window...) and is a candidate for an
    exclusion zone. Only a suggestion: in a static run every peak is static.
    """
    indices = list(indices)
    if window is None:
        window = analysis_window(run, AutoOptions())
    positions = [[p.position for p in detect(run, i, window, detection)] for i in indices]
    reference = max(positions, key=len) if positions else []
    static = []
    for x0 in reference:
        hits = [min(ps, key=lambda x: abs(x - x0)) for ps in positions if ps and min(abs(x - x0) for x in ps) <= tolerance]
        if len(hits) >= min_fraction * len(indices):
            static.append(StaticPeak(float(np.mean(hits)), float(np.std(hits)), len(hits) / len(indices)))
    return static


def auto_analyse(
    run: Run,
    indices: Iterable[int],
    phases: Optional[Sequence[str]] = None,
    options: Optional[AutoOptions] = None,
    progress: Optional[Progress] = None,
) -> List[AutoResult]:
    """Analyse spectra `indices` in order with the candidate `phases` (default: all of the run).

    The first spectrum, and every spectrum where the peak tracking breaks
    (see `track_peaks`), goes through the full phase search; the others reuse
    the previous phases and are only refitted.
    """
    options = options or AutoOptions()
    if run.calibration is None:
        raise ValueError("The run has no calibration (energy needed)")
    names = list(phases) if phases is not None else list(run.phases)
    window = analysis_window(run, options)
    exclude = list(options.exclude or [])
    search = dataclasses.replace(options.search, exclude=exclude + list(options.search.exclude or []))
    # positions are already known within the matching tolerance: the fit must not wander further
    fit = dataclasses.replace(options.fit, center_tolerance=min(options.fit.center_tolerance, options.search.tolerance))

    def excluded(peak: DetectedPeak) -> bool:
        return any(lo <= peak.position <= hi for lo, hi in exclude)

    tables = LineTables({n: run.phases[n] for n in names}, run.calibration.energy_ev, window, search)
    indices = list(indices)
    results: List[AutoResult] = []
    previous: Optional[Dict[str, float]] = None
    last: Optional[AutoResult] = None
    for k, index in enumerate(indices):
        detected = detect(run, index, window, options.detection)
        peaks = [p for p in detected if not excluded(p)]
        tracked = None
        if options.track and last is not None and last.applied is not None:
            tracked = track_peaks(last.peaks, last.applied, previous or {}, peaks,
                                  options.track_tolerance, options.track_min_snr)
        if tracked is not None:
            result = AutoResult(index, peaks, [tracked], tracked=True)
        else:
            result = AutoResult(index, peaks, search_composition(peaks, tables, previous if options.follow else None))
        if result.solutions:
            result.applied = result.solutions[0]
            others = [p for p in detected if excluded(p)]
            spectrum = apply_solution(run, index, result.applied, peaks, window, fit, others)
            found = {f.phase: f.pressure for f in spectrum.phases if np.isfinite(f.pressure)}
            previous = found or previous
            last = result
        else:
            run.spectra[index].phases = []
            run.spectra[index].model = None
        results.append(result)
        if progress is not None and not progress(k + 1, len(indices)):
            break
    return results
