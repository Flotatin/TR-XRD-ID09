"""Simultaneous fit of several peaks with scipy.optimize.curve_fit."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from scipy.optimize import curve_fit

from trxrd.domain import Peak
from trxrd.physics import profiles


@dataclass
class PeakFitResult:
    peaks: List[Peak]  # fitted copies, same order as the input (disabled peaks unchanged)
    success: bool
    message: str = ""
    reduced_chi2: float = np.nan


def total_curve(x: np.ndarray, peaks: Sequence[Peak]) -> np.ndarray:
    y = np.zeros_like(np.asarray(x, dtype=float))
    for peak in peaks:
        if peak.enabled and np.isfinite([peak.center, peak.height, peak.width]).all():
            y += profiles.evaluate(peak.model, x, peak.center, peak.height, peak.width, peak.shape)
    return y


def _start_and_bounds(peak: Peak, x: np.ndarray, y: np.ndarray, center_tolerance: float):
    profile = profiles.model(peak.model)
    step = float(np.median(np.abs(np.diff(x)))) if x.size > 1 else 1e-3
    y_max = float(np.nanmax(np.abs(y))) if y.size else 1.0
    center = peak.center
    height = peak.height if np.isfinite(peak.height) and peak.height > 0 else max(float(np.interp(center, x, y)), 1e-6 * y_max)
    width = peak.width if np.isfinite(peak.width) and peak.width > 0 else 5 * step
    inter = peak.inter if np.isfinite(peak.inter) and peak.inter > 0 else 3.0
    shape = tuple(peak.shape) or profile.shape_default

    lower = [center - center_tolerance, 0.0, max(step / 4, width / (1 + inter))]
    upper = [center + center_tolerance, 3 * y_max + 1e-12, width * (1 + inter)]
    for lo, hi in profile.shape_bounds:
        lower.append(lo)
        upper.append(hi)
    start = np.clip([center, height, width, *shape], lower, upper)
    return start, lower, upper


def fit_peaks(
    x: np.ndarray,
    y: np.ndarray,
    peaks: Sequence[Peak],
    *,
    center_tolerance: float = 0.3,
    noise: Optional[np.ndarray] = None,
    max_nfev: int = 20000,
) -> PeakFitResult:
    """Fit every enabled peak at once on (x, y) (y already background-corrected).

    Errors are the curve_fit standard errors, scaled by the residual
    variance (absolute_sigma=False) unless `noise` is given.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    active = [i for i, p in enumerate(peaks) if p.enabled]
    result = [dataclasses.replace(p) for p in peaks]
    if not active or x.size == 0:
        return PeakFitResult(result, success=False, message="nothing to fit")

    sizes, starts, lowers, uppers = [], [], [], []
    for i in active:
        start, lower, upper = _start_and_bounds(peaks[i], x, y, center_tolerance)
        sizes.append(len(start))
        starts.extend(start)
        lowers.extend(lower)
        uppers.extend(upper)
    if x.size <= len(starts):
        return PeakFitResult(result, success=False, message=f"{x.size} points for {len(starts)} parameters")
    functions = [profiles.model(peaks[i].model).function for i in active]
    splits = np.cumsum(sizes)[:-1]

    def model(xx, *params):
        out = np.zeros_like(xx)
        for function, p in zip(functions, np.split(np.asarray(params), splits)):
            out += function(xx, *p)
        return out

    try:
        popt, pcov = curve_fit(
            model, x, y, p0=starts, bounds=(lowers, uppers), sigma=noise,
            absolute_sigma=noise is not None, max_nfev=max_nfev,
        )
        success, message = True, ""
    except (RuntimeError, ValueError) as exc:
        popt, pcov = np.asarray(starts), np.full((len(starts), len(starts)), np.nan)
        success, message = False, str(exc)
    errors = np.sqrt(np.clip(np.diag(pcov), 0, None)) if np.isfinite(pcov).all() else np.full(len(popt), np.nan)

    for i, values, errs in zip(active, np.split(popt, splits), np.split(errors, splits)):
        result[i] = dataclasses.replace(
            peaks[i],
            center=float(values[0]), center_err=float(errs[0]),
            height=float(values[1]), height_err=float(errs[1]),
            width=float(values[2]), width_err=float(errs[2]),
            shape=tuple(float(v) for v in values[3:]),
        )
    residual = y - model(x, *popt)
    dof = max(x.size - len(popt), 1)
    weights = 1.0 if noise is None else 1.0 / np.asarray(noise) ** 2
    return PeakFitResult(result, success, message, float(np.sum(weights * residual**2) / dof))
