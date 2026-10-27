"""Time base of a run: detection of the detector triggers on an oscilloscope channel."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy import linalg


def iterative_poly_baseline(y: np.ndarray, deg: int = 3, max_it: int = 100, tol: float = 1e-3) -> np.ndarray:
    """Polynomial baseline clipped iteratively (same algorithm as peakutils.baseline,
    including scipy's pinv: high degrees are ill-conditioned and depend on it)."""
    y = np.asarray(y, dtype=float).copy()
    order = deg + 1
    coeffs = np.ones(order)
    cond = math.pow(np.abs(y).max(), 1.0 / order)
    x = np.linspace(0.0, cond, y.size)
    base = y.copy()
    vander = np.vander(x, order)
    vander_pinv = linalg.pinv(vander)
    for _ in range(max_it):
        coeffs_new = vander_pinv @ y
        if linalg.norm(coeffs_new - coeffs) / linalg.norm(coeffs) < tol:
            break
        coeffs = coeffs_new
        base = vander @ coeffs
        y = np.minimum(y, base)
    return base


def detect_trigger_times(
    time: np.ndarray,
    signal: np.ndarray,
    max_triggers: Optional[int] = None,
    threshold_ratio: float = 0.25,
    baseline_degree: int = 50,
) -> np.ndarray:
    """Middle time of each pulse of `signal` above threshold_ratio * max.

    The slow drift of the channel is removed first. Returns at most
    `max_triggers` times (one per detector frame).
    """
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if baseline_degree:
        signal = signal - iterative_poly_baseline(signal, baseline_degree)
    above = signal > signal.max() * threshold_ratio
    edges = np.diff(above.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if above[0]:
        starts.insert(0, 0)
    if above[-1]:
        ends.append(len(above) - 1)
    times = np.array([(time[s] + time[e]) / 2 for s, e in zip(starts, ends)])
    return times if max_triggers is None else times[:max_triggers]
