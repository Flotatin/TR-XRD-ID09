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


def _least_squares(model, x, y, starts, lowers, uppers, noise, max_nfev):
    """curve_fit -> (parameters, standard errors, success, message)."""
    try:
        popt, pcov = curve_fit(
            model, x, y, p0=starts, bounds=(lowers, uppers), sigma=noise,
            absolute_sigma=noise is not None, x_scale="jac",
            max_nfev=max_nfev,
        )
    except (RuntimeError, ValueError) as exc:
        return np.asarray(starts, dtype=float), np.full(len(starts), np.nan), False, str(exc)
    errors = np.sqrt(np.clip(np.diag(pcov), 0, None)) if np.isfinite(pcov).all() else np.full(len(popt), np.nan)
    return popt, errors, True, ""


def fit_peaks(
    x: np.ndarray,
    y: np.ndarray,
    peaks: Sequence[Peak],
    *,
    center_tolerance: float = 0.3,
    noise: Optional[np.ndarray] = None,
    max_nfev: Optional[int] = None,
    free_shape: bool = True,
) -> PeakFitResult:
    """Fit every enabled peak at once on (x, y) (y already background-corrected).

    Two stages: centre, height and width with the shape parameters fixed
    (fast, well conditioned even with 3 points per peak), then everything
    free from there, kept only if it fits better. Errors are the curve_fit
    standard errors, scaled by the residual variance (absolute_sigma=False)
    unless `noise` is given.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    active = [i for i, p in enumerate(peaks) if p.enabled]
    result = [dataclasses.replace(p) for p in peaks]
    if not active or x.size == 0:
        return PeakFitResult(result, success=False, message="nothing to fit")

    bounds = [_start_and_bounds(peaks[i], x, y, center_tolerance) for i in active]
    functions = [profiles.model(peaks[i].model).function for i in active]
    shapes = [list(start[3:]) for start, _, _ in bounds]
    weights = 1.0 if noise is None else 1.0 / np.asarray(noise) ** 2

    def stage(values, sizes, fixed_shape, budget):
        splits = np.cumsum(sizes)[:-1]

        def model(xx, *params):
            out = np.zeros_like(xx)
            for function, p, shape in zip(functions, np.split(np.asarray(params), splits), shapes):
                out += function(xx, *p, *shape) if fixed_shape else function(xx, *p)
            return out

        starts = [v for vs in values for v in vs]
        lowers = [v for (_, lo, _), n in zip(bounds, sizes) for v in lo[:n]]
        uppers = [v for (_, _, hi), n in zip(bounds, sizes) for v in hi[:n]]
        if x.size <= len(starts):
            return None
        popt, errors, ok, message = _least_squares(
            model, x, y, np.clip(starts, lowers, uppers), lowers, uppers, noise, max_nfev or budget * len(starts)
        )
        chi2 = float(np.sum(weights * (y - model(x, *popt)) ** 2) / max(x.size - len(popt), 1))
        return np.split(popt, splits), np.split(errors, splits), ok, message, chi2

    first = stage([start[:3] for start, _, _ in bounds], [3] * len(active), fixed_shape=True, budget=100)
    if first is None:
        return PeakFitResult(result, success=False, message=f"{x.size} points for {3 * len(active)} parameters")
    values, errors, ok, message, chi2 = first
    values = [np.concatenate([v, shape]) for v, shape in zip(values, shapes)]
    errors = [np.concatenate([e, np.full(len(shape), np.nan)]) for e, shape in zip(errors, shapes)]
    if free_shape and any(shapes):
        # starts close to the optimum: if it does not converge quickly, the shapes are not determined
        second = stage(values, [len(v) for v in values], fixed_shape=False, budget=20)
        if second is not None and second[2] and second[4] <= chi2:
            values, errors, ok, message, chi2 = second

    for i, v, e in zip(active, values, errors):
        result[i] = dataclasses.replace(
            peaks[i],
            center=float(v[0]), center_err=float(e[0]),
            height=float(v[1]), height_err=float(e[1]),
            width=float(v[2]), width_err=float(e[2]),
            shape=tuple(float(c) for c in v[3:]),
        )
    return PeakFitResult(result, ok, message, chi2)
