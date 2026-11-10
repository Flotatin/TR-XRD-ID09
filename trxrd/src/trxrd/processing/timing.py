"""Time base of a run: detection of the detector triggers on an oscilloscope channel."""

from __future__ import annotations

import math
from dataclasses import dataclass
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


REGULAR = 0.05  # periods within 5 % of their median: the triggers come from one clock


@dataclass(frozen=True)
class Pulses:
    """Pulses of a trigger channel, in the order of the trace: start and end times (s)."""

    start: np.ndarray
    end: np.ndarray
    complete: np.ndarray  # False for a pulse cut by the start or the end of the trace
    step: float = 0.0  # sampling step of the trace: the times are known to this step

    @property
    def middle(self) -> np.ndarray:
        return (self.start + self.end) / 2


def find_pulses(time: np.ndarray, signal: np.ndarray, threshold_ratio: float = 0.25,
                baseline_degree: int = 50) -> Pulses:
    """Pulses of `signal` above threshold_ratio * max, after removing the slow drift of the channel."""
    time = np.asarray(time, dtype=float)
    raw = np.asarray(signal, dtype=float)
    none = Pulses(np.empty(0), np.empty(0), np.empty(0, dtype=bool))
    if raw.size == 0 or np.ptp(raw) == 0:  # a constant channel (the baseline would divide by zero)
        return none
    signal = raw - iterative_poly_baseline(raw, baseline_degree) if baseline_degree else raw
    if signal.max() <= 1e-3 * np.abs(raw).max():  # what the baseline leaves of a smooth channel, not pulses
        return none
    above = signal > signal.max() * threshold_ratio
    edges = np.diff(above.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if above[0]:
        starts.insert(0, 0)
    if above[-1]:
        ends.append(len(above) - 1)
    count = min(len(starts), len(ends))
    starts, ends = np.asarray(starts[:count], dtype=int), np.asarray(ends[:count], dtype=int)
    complete = np.ones(count, dtype=bool)
    if count and above[0]:
        complete[0] = False
    if count and above[-1]:
        complete[-1] = False
    return Pulses(time[starts], time[ends], complete, float(np.median(np.diff(time))) if time.size > 1 else 0.0)


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
    times = find_pulses(time, signal, threshold_ratio, baseline_degree).middle
    return times if max_triggers is None else times[:max_triggers]


@dataclass(frozen=True)
class PulseTiming:
    """Medians over the complete pulses of a trigger channel (s)."""

    width: float  # duration of a pulse (median)
    period: float  # from the start of a pulse to the start of the next (mean)
    gap: float  # from the end of a pulse to the start of the next: the inter-frame time of the v1 (median)
    count: int  # complete pulses measured
    regular: bool  # every period within REGULAR of the mean, or within the sampling step
    step: float = 0.0  # sampling step of the trace


def pulse_timing(pulses: Pulses) -> Optional[PulseTiming]:
    """Width, period and gap of the pulses; None with fewer than two complete pulses.

    Sampled every 1 µs (Bi01), triggers 17.1 µs apart are seen 17 or 18 µs
    apart: 6 % from one another, but from one clock. A period may thus differ
    from the mean by one sampling step (1.5 for the rounding). The period is
    the mean from the first pulse to the last, which averages this rounding
    out: 1.01347 ms for Water01 scan 74 (sampled every 1 µs), where the median
    gave 1.013 ms, 0.7 ms off after 1500 frames.
    """
    start, end = pulses.start[pulses.complete], pulses.end[pulses.complete]
    if start.size < 2:
        return None
    periods = np.diff(start)
    period = float((start[-1] - start[0]) / (start.size - 1))
    return PulseTiming(
        width=float(np.median(end - start)),
        period=period,
        gap=float(np.median(start[1:] - end[:-1])),
        count=int(start.size),
        regular=bool(np.all(np.abs(periods - period) <= max(REGULAR * period, 1.5 * pulses.step))),
        step=pulses.step,
    )


def frame_times(pulses: Pulses, frames: int, trace_end: float) -> np.ndarray:
    """Time of the first `frames` frames: the middle of their trigger pulse.

    When the trace stops while the triggers still come at a regular period
    (Jungfrau continuous mode: 100 ms of oscilloscope for 1600 frames), the
    later frames continue at that period. Triggers that stop well before the
    end of the trace, or irregular ones, are not continued.
    """
    times = pulses.middle[:frames]
    rhythm = pulse_timing(pulses)
    if (times.size < frames and rhythm is not None and rhythm.regular
            and trace_end - pulses.end[-1] < 2 * rhythm.period):
        times = np.concatenate([times, times[-1] + rhythm.period * np.arange(1, frames - times.size + 1)])
    return times
