"""Use cases: fit the peaks of one or several spectra and derive P for each phase."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from trxrd.domain import FitState, Peak, PhaseCard, PhaseFit, Run, Spectrum
from trxrd.physics import eos
from trxrd.processing import background
from trxrd.processing.cell import Reflex, pressure_with_error, refine_cell, sigma_two_theta_from_distance
from trxrd.processing.peakfit import fit_peaks, total_curve

logger = logging.getLogger(__name__)

Progress = Callable[[int, int], bool]  # (done, total) -> keep going?


@dataclass
class FitOptions:
    model: str = "PearsonIV"
    width: float = 0.03  # initial width of new peaks (deg)
    center_tolerance: float = 0.3  # how far a peak centre may move during the fit (deg)
    min_intensity: float = 0.0  # reflections of the card weaker than this (%) are not used
    temperature: float = eos.T_REF


def initial_phase_fit(
    card: PhaseCard,
    energy_ev: float,
    pressure: float,
    window: Tuple[float, float],
    options: Optional[FitOptions] = None,
) -> PhaseFit:
    """Peaks placed at the 2θ expected for `card` at `pressure`, inside `window`."""
    options = options or FitOptions()
    peaks = []
    for two_theta, reflection in card.two_theta_lines(pressure, energy_ev, options.temperature, max_lines=len(card.reflections)):
        if window[0] <= two_theta <= window[1] and reflection.intensity >= options.min_intensity:
            peaks.append(
                Peak(
                    name=f"{card.name}_{reflection.label}",
                    model=options.model,
                    center=two_theta,
                    width=options.width,
                    hkl=reflection.hkl,
                )
            )
    return PhaseFit(phase=card.name, pressure=pressure, temperature=options.temperature, peaks=peaks)


def corrected_signal(spectrum: Spectrum) -> np.ndarray:
    """Smoothed intensity minus background (the background is computed if missing)."""
    if spectrum.baseline is None:
        background.apply(spectrum)
    smoothed = background.smooth(spectrum.two_theta, spectrum.intensity, spectrum.processing)
    return smoothed - spectrum.baseline


def _window(spectrum: Spectrum, fit_range: Optional[Tuple[float, float]]) -> Tuple[float, float]:
    if fit_range is not None:
        return tuple(fit_range)
    if spectrum.fit_range is not None:
        return spectrum.fit_range
    return float(np.nanmin(spectrum.two_theta)), float(np.nanmax(spectrum.two_theta))


def _update_phase(run: Run, phase_fit: PhaseFit, temperature: float) -> None:
    card = run.phases.get(phase_fit.phase)
    calib = run.calibration
    reflexes = []
    for peak in phase_fit.peaks:
        if peak.enabled and peak.hkl is not None and np.isfinite(peak.center):
            sigma = peak.center_err if np.isfinite(peak.center_err) else 0.0
            if calib is not None and calib.sigma_distance_m:
                geometric = sigma_two_theta_from_distance(peak.center, calib.distance_m, calib.sigma_distance_m)
                sigma = float(np.hypot(sigma, geometric))
            reflexes.append(Reflex(peak.hkl, peak.center, sigma))
    if card is None or not reflexes or calib is None:
        return
    cell = refine_cell(card, reflexes, calib.wavelength_m)
    pressure, pressure_err = pressure_with_error(card, cell.volume, cell.volume_err, temperature)
    phase_fit.a, phase_fit.a_err = cell.a, cell.a_err
    phase_fit.b, phase_fit.b_err = cell.b, cell.b_err
    phase_fit.c, phase_fit.c_err = cell.c, cell.c_err
    phase_fit.volume, phase_fit.volume_err = cell.volume, cell.volume_err
    phase_fit.pressure, phase_fit.pressure_err = pressure, pressure_err
    phase_fit.temperature = temperature


def fit_spectrum(
    run: Run,
    index: int,
    phases: Optional[Dict[str, float]] = None,
    *,
    fit_range: Optional[Tuple[float, float]] = None,
    options: Optional[FitOptions] = None,
) -> Spectrum:
    """Fit spectrum `index` and refine every fitted phase (in place).

    `phases` maps phase names to a starting pressure and (re)creates their
    peaks from the cards; otherwise the peaks already on the spectrum are the
    starting point. Phases whose state is not "fitted" are left untouched.
    """
    options = options or FitOptions()
    spectrum = run.spectra[index]
    if run.calibration is None:
        raise ValueError("The run has no calibration (energy needed)")
    window = _window(spectrum, fit_range)
    if phases:
        kept = [f for f in spectrum.phases if f.phase not in phases]
        new = [initial_phase_fit(run.phases[name], run.calibration.energy_ev, p, window, options) for name, p in phases.items()]
        spectrum.phases = kept + new
    spectrum.fit_range = window

    x = np.asarray(spectrum.two_theta, dtype=float)
    y = corrected_signal(spectrum)
    inside = (x >= window[0]) & (x <= window[1])
    active = [f for f in spectrum.phases if f.state == FitState.FITTED]
    peaks = [p for f in active for p in f.peaks if p.enabled and window[0] <= p.center <= window[1]]
    result = fit_peaks(x[inside], y[inside], peaks, center_tolerance=options.center_tolerance)
    if not result.success:
        logger.warning("Spectrum %d: %s", index, result.message)
    fitted = {id(old): new for old, new in zip(peaks, result.peaks)}
    for phase_fit in active:
        phase_fit.peaks = [fitted.get(id(p), p) for p in phase_fit.peaks]
        _update_phase(run, phase_fit, options.temperature)
    spectrum.model = total_curve(x, [p for f in active for p in f.peaks if window[0] <= p.center <= window[1]])
    return spectrum


def fit_spectra(
    run: Run,
    indices: Iterable[int],
    phases: Optional[Dict[str, float]] = None,
    *,
    follow: bool = True,
    fit_range: Optional[Tuple[float, float]] = None,
    options: Optional[FitOptions] = None,
    progress: Optional[Progress] = None,
) -> List[int]:
    """Fit a sequence of spectra. With `follow`, each spectrum starts from the
    result of the previous one (peaks and pressures), which tracks a
    compression or a decompression smoothly. Returns the fitted indices."""
    options = options or FitOptions()
    indices = list(indices)
    done: List[int] = []
    previous: Optional[Spectrum] = None
    for k, index in enumerate(indices):
        spectrum = run.spectra[index]
        if phases and (k == 0 or not follow):
            fit_spectrum(run, index, phases, fit_range=fit_range, options=options)
        else:
            if follow and previous is not None and previous.phases:
                spectrum.phases = copy.deepcopy(previous.phases)
                spectrum.fit_range = previous.fit_range
            fit_spectrum(run, index, fit_range=fit_range, options=options)
        previous = spectrum
        done.append(index)
        if progress is not None and not progress(k + 1, len(indices)):
            break
    return done
