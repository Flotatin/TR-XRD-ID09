"""Use cases: manual work on the phases of the spectra, after or instead of the automatic analysis.

Add a phase at a starting pressure, move it, remove it; add, move, disable
or remove single peaks; fit the spectrum; fit a series from the current
spectrum outwards, each spectrum starting from its neighbour; copy the model
of a spectrum onto others. A phase or a peak is placed on the peaks the
detection finds (position, height, width): the fit starts from what is really
there, with the settings of the automatic analysis.
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
from trxrd.processing import background
from trxrd.processing.peaks import DetectedPeak
from trxrd.services.analysis import AnalysisSettings
from trxrd.processing.peakfit import fit_peaks, total_curve
from trxrd.services.fitting import FitOptions, corrected_signal, fit_spectrum, refresh_model, update_phase
from trxrd.services.identification import Progress, analysis_window, detect

LINE_REACH = 0.3  # deg: a peak added this close to a line of the phase, not yet used, takes that line
RANGE_MARGIN = 0.5  # deg kept around a peak at the end of the fit range
NO_PEAK = "Aucun pic à ajuster : amener les raies sur des pics détectés (pression), ou cocher une raie"
PeakRef = Tuple[Optional[str], int]  # (phase, or None for a free peak; position in its list of peaks)


def _fit_options(settings: Optional[AnalysisSettings]) -> FitOptions:
    """Those of the automatic analysis; peaks added by hand are fitted wherever they are."""
    return dataclasses.replace((settings or AnalysisSettings()).options().fit, fit_far_free_peaks=True)


def _cover(spectrum: Spectrum, centers: Sequence[float], window: Optional[Tuple[float, float]] = None) -> None:
    """The fit range holds `centers`: an old analysis may have a narrow one, and what is outside would be ignored."""
    x = np.asarray(spectrum.two_theta, dtype=float)
    low, high = spectrum.fit_range or window or (float(x.min()), float(x.max()))
    if len(centers):
        low = max(min(low, min(centers) - RANGE_MARGIN), float(x.min()))
        high = min(max(high, max(centers) + RANGE_MARGIN), float(x.max()))
    spectrum.fit_range = (low, high)


def _lines(card: PhaseCard, pressure: float, energy_ev: float, temperature: float,
           window: Tuple[float, float]) -> List[tuple]:
    """(2θ, reflection) inside `window`, strongest first."""
    try:
        lines = card.two_theta_lines(pressure, energy_ev, temperature, max_lines=len(card.reflections))
    except ValueError:
        raise ValueError(f"{card.name} : {pressure:g} GPa est hors du domaine de son équation d'état") from None
    inside = [(t, r) for t, r in lines if np.isfinite(t) and window[0] <= t <= window[1]]
    return sorted(inside, key=lambda line: -line[1].intensity)


def place_peaks(card: PhaseCard, pressure: float, energy_ev: float, temperature: float, window: Tuple[float, float],
                detected: Sequence[DetectedPeak], taken: Sequence[float], tolerance: float,
                model: str, width: float) -> List[Peak]:
    """Peaks of `card` at `pressure`: only the lines a detected peak explains, within `tolerance`.

    A detected peak goes to the strongest line near it: sharing it would make
    the fit degenerate. Peaks already explained by another phase (`taken`,
    2θ) are left out for the same reason. The other lines get no peak: while
    the pressure is moved they simply follow, and a box can still add one.
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
    _cover(spectrum, [p.center for p in peaks], window)
    refresh_model(spectrum)
    return fit


def remove_phase(run: Run, index: int, phase: str) -> None:
    spectrum = run.spectra[index]
    spectrum.phases = [f for f in spectrum.phases if f.phase != phase]
    refresh_model(spectrum)


def peak_refs(spectrum: Spectrum) -> List[PeakRef]:
    """Every peak of the spectrum: the phases in their order, then the free peaks (the order of the table)."""
    refs: List[PeakRef] = [(fit.phase, k) for fit in spectrum.phases for k in range(len(fit.peaks))]
    return refs + [(None, k) for k in range(len(spectrum.free_peaks))]


def _peaks_of(spectrum: Spectrum, phase: Optional[str]) -> List[Peak]:
    if phase is None:
        return spectrum.free_peaks
    fit = spectrum.phase_fit(phase)
    if fit is None:
        raise ValueError(f"{phase} n'est pas sur ce spectre")
    return fit.peaks


def get_peak(spectrum: Spectrum, ref: PeakRef) -> Peak:
    peaks = _peaks_of(spectrum, ref[0])
    if not 0 <= ref[1] < len(peaks):
        raise ValueError("Ce pic n'existe plus")
    return peaks[ref[1]]


def _peak_width(spectrum: Spectrum, default: float) -> float:
    widths = [p.width for f in spectrum.phases for p in f.peaks] + [p.width for p in spectrum.free_peaks]
    widths = [w for w in widths if np.isfinite(w) and w > 0]
    return float(np.median(widths)) if widths else default


def _unused_line(run: Run, fit: Optional[PhaseFit], center: float, temperature: float):
    """The reflection of the phase of `fit` closest to `center`, within LINE_REACH, that has no peak yet."""
    if fit is None or run.calibration is None:
        return None
    card = run.phases[fit.phase]
    used = {p.hkl for p in fit.peaks}
    lines = card.two_theta_lines(fit.pressure, run.calibration.energy_ev, temperature, max_lines=len(card.reflections))
    close = [(abs(t - center), r.label, r) for t, r in lines if r.hkl not in used and abs(t - center) <= LINE_REACH]
    return min(close, key=lambda item: item[:2])[2] if close else None


def _new_peak(run: Run, index: int, position: float, options) -> Peak:
    """A peak at 2θ = `position`, on the peak detected there if there is one."""
    spectrum = run.spectra[index]
    window = analysis_window(run, options)
    near = min(detect(run, index, window, options.detection), key=lambda p: abs(p.position - position), default=None)
    if near is not None and abs(near.position - position) > options.search.tolerance:
        near = None
    model = options.fit.model
    shape = profiles.model(model).shape_default
    center = float(position) if near is None else near.position
    height = near.height if near is not None else float(
        np.interp(center, np.asarray(spectrum.two_theta, dtype=float), background.corrected(spectrum)[0]))
    width = _peak_width(spectrum, options.fit.width)
    if near is not None and np.isfinite(near.fwhm):
        width = profiles.width_from_fwhm(model, near.fwhm, shape)
    return Peak(f"free_{len(spectrum.free_peaks)}", model, center=center, height=max(height, 1.0), width=width, shape=shape)


def add_peak(run: Run, index: int, position: float, phase: Optional[str] = None,
             settings: Optional[AnalysisSettings] = None) -> PeakRef:
    """A peak at 2θ = `position`, on the peak detected there if there is one. Not fitted yet.

    Near a line of `phase` that has no peak yet it takes that line (hkl);
    otherwise it is a free peak. Returns its reference.
    """
    options = (settings or AnalysisSettings()).options()
    spectrum = run.spectra[index]
    peak = _new_peak(run, index, position, options)
    center = peak.center
    fit = spectrum.phase_fit(phase) if phase is not None else None
    line = _unused_line(run, fit, center, options.fit.temperature)
    if line is not None:
        peak.name, peak.hkl = f"{run.phases[phase].name}_{line.label}", line.hkl
        fit.peaks.append(peak)
        fit.peaks.sort(key=lambda p: p.center)
        ref: PeakRef = (phase, next(k for k, p in enumerate(fit.peaks) if p is peak))
    else:
        spectrum.free_peaks.append(peak)
        ref = (None, len(spectrum.free_peaks) - 1)
    _cover(spectrum, [center], analysis_window(run, options))
    refresh_model(spectrum)
    return ref


@dataclass
class Line:
    """A reflection of a phase of the spectrum at its pressure, and whether it has a peak."""

    hkl: Tuple[int, int, int]
    label: str
    two_theta: float
    intensity: float  # % of the card
    used: bool


def phase_lines(run: Run, index: int, phase: str, settings: Optional[AnalysisSettings] = None) -> List[Line]:
    """The lines of `phase` inside the pattern, in increasing 2θ."""
    spectrum = run.spectra[index]
    fit = spectrum.phase_fit(phase)
    if fit is None or run.calibration is None or not np.isfinite(fit.pressure):
        return []
    temperature = (settings or AnalysisSettings()).options().fit.temperature
    x = np.asarray(spectrum.two_theta, dtype=float)
    used = {p.hkl for p in fit.peaks}
    card = run.phases[phase]
    positions = card.two_theta_table([fit.pressure], run.calibration.energy_ev, temperature)[0]  # NaN outside the EoS
    inside = [(t, r) for t, r in zip(positions, card.reflections) if np.isfinite(t) and x.min() <= t <= x.max()]
    return [Line(r.hkl, r.label, float(t), float(r.intensity), r.hkl in used) for t, r in sorted(inside, key=lambda l: l[0])]


def set_line(run: Run, index: int, phase: str, hkl: Tuple[int, int, int], used: bool,
             settings: Optional[AnalysisSettings] = None) -> None:
    """Give a line of the phase a peak, on what is detected there, or take its peak away."""
    spectrum = run.spectra[index]
    fit = spectrum.phase_fit(phase)
    if fit is None:
        raise ValueError(f"{phase} n'est pas sur ce spectre")
    hkl = tuple(hkl)
    if not used:
        fit.peaks = [p for p in fit.peaks if p.hkl != hkl]
    elif all(p.hkl != hkl for p in fit.peaks):
        line = next((l for l in phase_lines(run, index, phase, settings) if l.hkl == hkl), None)
        if line is None:
            raise ValueError(f"Pas de raie {hkl} dans ce spectre")
        options = (settings or AnalysisSettings()).options()
        peak = _new_peak(run, index, line.two_theta, options)
        peak.name, peak.hkl = f"{run.phases[phase].name}_{line.label}", hkl
        fit.peaks.append(peak)
        fit.peaks.sort(key=lambda p: p.center)
        _cover(spectrum, [peak.center], analysis_window(run, options))
    refresh_model(spectrum)


def misfit(spectrum: Spectrum) -> Optional[float]:
    """Sum of the squared residuals in the fit range, to compare two fits; None without a model."""
    if spectrum.model is None:
        return None
    x = np.asarray(spectrum.two_theta, dtype=float)
    low, high = spectrum.fit_range or (float(x.min()), float(x.max()))
    inside = (x >= low) & (x <= high)
    return float(np.sum((corrected_signal(spectrum) - spectrum.model)[inside] ** 2))


def move_peak(run: Run, index: int, ref: PeakRef, center: float, height: Optional[float] = None) -> None:
    """New position, and height if given, of a peak: the start of the next fit."""
    spectrum = run.spectra[index]
    peak = get_peak(spectrum, ref)
    peak.center, peak.center_err = float(center), np.nan
    if height is not None:
        peak.height, peak.height_err = max(float(height), 1.0), np.nan
    _cover(spectrum, [peak.center])
    refresh_model(spectrum)


def set_peak(run: Run, index: int, ref: PeakRef, *, model: Optional[str] = None, center: Optional[float] = None,
             height: Optional[float] = None, width: Optional[float] = None,
             shape: Optional[Sequence[float]] = None) -> None:
    """Parameters of a peak as the user sets them: the start of the next fit.

    A new model keeps the position, the height and the full width at half
    maximum, with the default shape of that model.
    """
    spectrum = run.spectra[index]
    peak = get_peak(spectrum, ref)
    if model is not None and model != peak.model:
        profile = profiles.model(model)
        full = profiles.fwhm(peak.model, peak.width, peak.shape) if np.isfinite(peak.width) and peak.width > 0 else None
        peak.model, peak.shape = model, profile.shape_default
        if full:
            peak.width = profiles.width_from_fwhm(model, full, peak.shape)
            if model == "SplitLorentzian":
                peak.shape = (peak.width,)  # symmetric to start with
    if center is not None:
        peak.center, peak.center_err = float(center), np.nan
    if height is not None:
        peak.height, peak.height_err = max(float(height), 1.0), np.nan
    if width is not None:
        if not width > 0:
            raise ValueError("La largeur d'un pic doit être positive")
        peak.width, peak.width_err = float(width), np.nan
    if shape is not None:
        bounds = profiles.model(peak.model).shape_bounds
        if len(shape) != len(bounds):
            raise ValueError(f"{peak.model} : {len(bounds)} paramètre(s) de forme attendu(s)")
        peak.shape = tuple(float(np.clip(v, low, high)) for v, (low, high) in zip(shape, bounds))
    _cover(spectrum, [peak.center])
    refresh_model(spectrum)


def fit_peak(run: Run, index: int, ref: PeakRef, settings: Optional[AnalysisSettings] = None) -> Peak:
    """Fit one peak alone, on the signal minus every other peak, within `fit_window` widths of its centre.

    The other peaks do not move. Then the cell and the pressure of its phase
    follow. ValueError when the fit fails: the peak stays as it was.
    """
    settings = settings or AnalysisSettings()
    options = _fit_options(settings)
    spectrum = run.spectra[index]
    peak = get_peak(spectrum, ref)
    x = np.asarray(spectrum.two_theta, dtype=float)
    others = [p for r in peak_refs(spectrum) for p in [get_peak(spectrum, r)] if p is not peak]
    signal = corrected_signal(spectrum) - total_curve(x, others)
    step = float(np.median(np.abs(np.diff(x))))
    half = max(settings.fit_window * peak.width, 4 * step)
    near = np.abs(x - peak.center) <= half
    start = dataclasses.replace(peak, inter=peak.inter if options.variation is None else options.variation)
    result = fit_peaks(x[near], signal[near], [start], center_tolerance=options.center_tolerance)
    if not result.success:
        raise ValueError(f"Ajustement du pic impossible : {result.message}")
    fitted = result.peaks[0]
    _peaks_of(spectrum, ref[0])[ref[1]] = fitted
    fit = spectrum.phase_fit(ref[0]) if ref[0] is not None else None
    if fit is not None:
        update_phase(run, fit, options)
    _cover(spectrum, [fitted.center])
    refresh_model(spectrum)
    return fitted


def remove_peak(run: Run, index: int, ref: PeakRef) -> None:
    spectrum = run.spectra[index]
    get_peak(spectrum, ref)
    del _peaks_of(spectrum, ref[0])[ref[1]]
    refresh_model(spectrum)


def toggle_peak(run: Run, index: int, ref: PeakRef) -> bool:
    """A disabled peak stays in the list but leaves the curve, the fit and the cell. Returns its new state."""
    spectrum = run.spectra[index]
    peak = get_peak(spectrum, ref)
    peak.enabled = not peak.enabled
    refresh_model(spectrum)
    return peak.enabled


def has_peaks(spectrum: Spectrum) -> bool:
    return bool(spectrum.free_peaks) or any(fit.peaks for fit in spectrum.phases)


def fit_current(run: Run, index: int, settings: Optional[AnalysisSettings] = None) -> Spectrum:
    """Fit spectrum `index` from the peaks it holds."""
    spectrum = run.spectra[index]
    if not spectrum.phases and not spectrum.free_peaks:
        raise ValueError("Aucune phase sur ce spectre : en ajouter une avant d'ajuster")
    if not has_peaks(spectrum):
        raise ValueError(NO_PEAK)
    return fit_spectrum(run, index, options=_fit_options(settings))


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
    options = _fit_options(settings)
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
