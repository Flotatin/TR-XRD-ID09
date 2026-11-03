"""Use cases: fit the peaks of one or several spectra and derive P for each phase."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from trxrd.domain import FitState, Peak, PhaseCard, PhaseFit, Run, Spectrum
from trxrd.physics import eos, profiles
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
    local_window: Optional[float] = 0.3  # deg: only points this close to a peak are fitted (None: whole range)
    # ...or this many FWHM of the peak when it is broader (liquid, amorphous, strained): its tails are fitted too
    local_fwhm: float = 1.0
    exclude: Optional[Sequence[Tuple[float, float]]] = None  # 2θ zones left out of the fit (gasket, diamond...)
    max_center_err: float = 0.1  # deg: peaks known worse than this do not enter the cell refinement


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
    """Raw intensity minus background (the background is computed if missing).

    The fit is done on the raw signal: smoothing would shift the maximum of
    asymmetric peaks and correlate the noise, i.e. bias the positions and
    underestimate their errors. Smoothing is only for detection and display.
    """
    if spectrum.baseline is None:
        background.apply(spectrum)
    return np.asarray(spectrum.intensity, dtype=float) - spectrum.baseline


def _window(spectrum: Spectrum, fit_range: Optional[Tuple[float, float]]) -> Tuple[float, float]:
    if fit_range is not None:
        return tuple(fit_range)
    if spectrum.fit_range is not None:
        return spectrum.fit_range
    return float(np.nanmin(spectrum.two_theta)), float(np.nanmax(spectrum.two_theta))


def _update_phase(run: Run, phase_fit: PhaseFit, options: FitOptions) -> None:
    card = run.phases.get(phase_fit.phase)
    calib = run.calibration
    temperature = options.temperature
    reflexes = []
    for peak in phase_fit.peaks:
        badly_known = np.isfinite(peak.center_err) and peak.center_err > options.max_center_err
        vanished = np.isfinite(peak.height) and peak.height <= 0
        if peak.enabled and peak.hkl is not None and np.isfinite(peak.center) and not badly_known and not vanished:
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


def fit_mask(x: np.ndarray, window: Tuple[float, float], centers: Sequence[float], options: FitOptions,
             fwhms: Optional[Sequence[float]] = None) -> np.ndarray:
    """Points used by the fit: inside the window, near a peak, outside the excluded zones.

    "Near" is within `local_window` of the centre, or `local_fwhm` FWHM for a broader peak.
    """
    mask = (x >= window[0]) & (x <= window[1])
    if options.local_window and len(centers):
        reach = np.full(len(centers), float(options.local_window))
        if fwhms is not None:
            reach = np.maximum(reach, options.local_fwhm * np.nan_to_num(np.asarray(fwhms, dtype=float)))
        near = np.any(np.abs(x[:, None] - np.asarray(centers)[None, :]) <= reach[None, :], axis=1)
        mask &= near
    for lo, hi in options.exclude or ():
        mask &= ~((x >= lo) & (x <= hi))
    return mask


def fit_spectrum(
    run: Run,
    index: int,
    phases: Optional[Dict[str, float]] = None,
    *,
    fit_range: Optional[Tuple[float, float]] = None,
    options: Optional[FitOptions] = None,
    extra_peaks: Sequence[Peak] = (),
) -> Spectrum:
    """Fit spectrum `index` and refine every fitted phase (in place).

    `phases` maps phase names to a starting pressure and (re)creates their
    peaks from the cards; otherwise the peaks already on the spectrum are the
    starting point. Phases whose state is not "fitted" are left untouched.

    Only the points within `local_window` of a peak are fitted, outside the
    `exclude` zones: a strong peak that belongs to no phase (gasket,
    diamond) cannot drag the fit. `extra_peaks` (peaks seen but assigned to no
    phase) are fitted alongside so that they do not bias their neighbours,
    then dropped.
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
    active = [f for f in spectrum.phases if f.state == FitState.FITTED]
    peaks = [p for f in active for p in f.peaks if p.enabled and window[0] <= p.center <= window[1]]
    centers = [p.center for p in peaks]
    reach = 2 * (options.local_window or np.inf)  # free peaks far from every phase peak are irrelevant
    extra = [
        p for p in extra_peaks
        if window[0] <= p.center <= window[1] and centers and min(abs(p.center - c) for c in centers) <= reach
    ]
    fwhms = [profiles.fwhm(p.model, p.width, p.shape) for p in peaks + extra]
    mask = fit_mask(x, window, centers + [p.center for p in extra], options, fwhms)
    result = fit_peaks(x[mask], y[mask], peaks + extra, center_tolerance=options.center_tolerance)
    if not result.success:
        logger.warning("Spectrum %d: %s", index, result.message)
    fitted = {id(old): new for old, new in zip(peaks, result.peaks[: len(peaks)])}
    for phase_fit in active:
        phase_fit.peaks = [fitted.get(id(p), p) for p in phase_fit.peaks]
        _update_phase(run, phase_fit, options)
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
