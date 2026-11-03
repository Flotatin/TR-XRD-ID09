"""Rate of change of a time series: local slope (sliding straight line) and mean rate over an interval.

Used for dP/dt. A straight line fitted by least squares on the points of a
window gives the slope and its error from the scatter of the points around
the line: unlike the difference of two neighbours, the noise of one spectrum
does not go straight into the rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinearFit:
    """y = intercept + slope * t, least squares."""

    slope: float
    slope_err: float  # 1σ from the scatter around the line (NaN with only 2 points)
    intercept: float  # value at t = 0
    n: int  # points used
    rms: float  # scatter of the points around the line

    def at(self, t):
        return self.intercept + self.slope * np.asarray(t, dtype=float)


def linear_fit(t, y) -> LinearFit:
    """Straight line through the finite (t, y) pairs; NaN slope with fewer than 2 distinct times."""
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(t) & np.isfinite(y)
    t, y = t[keep], y[keep]
    n = int(t.size)
    if n < 2 or np.ptp(t) == 0:
        return LinearFit(np.nan, np.nan, np.nan, n, np.nan)
    dt = t - t.mean()
    sxx = float(np.sum(dt**2))
    slope = float(np.sum(dt * (y - y.mean())) / sxx)
    intercept = float(y.mean() - slope * t.mean())
    residuals = y - (intercept + slope * t)
    error = float(np.sqrt(np.sum(residuals**2) / (n - 2) / sxx)) if n > 2 else np.nan
    return LinearFit(slope, error, intercept, n, float(np.sqrt(np.mean(residuals**2))))


@dataclass
class LocalFit:
    """Straight line over each point and its neighbours: slope, its 1σ error, and the smoothed value."""

    slope: np.ndarray
    slope_err: np.ndarray
    value: np.ndarray  # the line at the point itself (the series smoothed)


def local_fit(t, y, half_window: int) -> LocalFit:
    """Straight line over each point and `half_window` neighbours on each side (as in a sliding window).

    The window is cut at the ends of the series (fewer points there); with
    fewer than 3 points the slope is NaN and the value is the point itself.
    `t` must be sorted. Computed with running sums: fast for thousands of
    spectra.
    """
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    n = t.size
    slope, error, value = np.full(n, np.nan), np.full(n, np.nan), y.copy()
    if n < 3:
        return LocalFit(slope, error, value)
    k = max(int(half_window), 1)
    t_mean, y_mean = t.mean(), y.mean()
    tc, yc = t - t_mean, y - y_mean  # centred: the running sums stay well conditioned

    def running(values):
        return np.concatenate([[0.0], np.cumsum(values)])

    s_t, s_y, s_tt, s_ty, s_yy = (running(v) for v in (tc, yc, tc * tc, tc * yc, yc * yc))
    index = np.arange(n)
    lo, hi = np.clip(index - k, 0, n), np.clip(index + k + 1, 0, n)
    count = (hi - lo).astype(float)
    sum_t, sum_y = s_t[hi] - s_t[lo], s_y[hi] - s_y[lo]
    sxx = s_tt[hi] - s_tt[lo] - sum_t**2 / count
    sxy = s_ty[hi] - s_ty[lo] - sum_t * sum_y / count
    syy = s_yy[hi] - s_yy[lo] - sum_y**2 / count
    good = (count >= 3) & (sxx > 1e-12 * max(float(np.ptp(tc)) ** 2, 1e-300))
    slope[good] = sxy[good] / sxx[good]
    residual = np.clip(syy[good] - slope[good] * sxy[good], 0.0, None)
    error[good] = np.sqrt(residual / (count[good] - 2) / sxx[good])
    value[good] = y_mean + sum_y[good] / count[good] + slope[good] * (tc[good] - sum_t[good] / count[good])
    return LocalFit(slope, error, value)


def local_slopes(t, y, half_window: int):
    """dy/dt at every point and its 1σ error (see `local_fit`)."""
    fit = local_fit(t, y, half_window)
    return fit.slope, fit.slope_err
