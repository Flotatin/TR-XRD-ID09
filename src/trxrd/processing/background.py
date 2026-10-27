"""1D treatment of an integrated pattern: smoothing and background (SNIP)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from pybaselines import Baseline
from scipy.signal import savgol_filter

from trxrd.domain import Processing, Spectrum


def snip_baseline(intensity: np.ndarray, degree: int) -> np.ndarray:
    """SNIP background; `degree` sets the smoothing window (5*degree points)."""
    baseline, _ = Baseline().snip(
        np.asarray(intensity, dtype=float), filter_order=2, smooth_half_window=5 * int(degree)
    )
    return baseline


def smooth(two_theta: np.ndarray, intensity: np.ndarray, processing: Processing) -> np.ndarray:
    kind = processing.filter_type
    p1, p2 = processing.filter_params
    if kind == "svg":
        return savgol_filter(intensity, window_length=int(p1), polyorder=int(p2))
    if kind == "fft":
        spectrum = np.fft.fft(intensity)
        freq = np.fft.fftfreq(len(intensity), d=two_theta[1] - two_theta[0])
        spectrum[(np.abs(freq) > p1) & (np.abs(freq) < p2)] = 0
        return np.real(np.fft.ifft(spectrum))
    return np.asarray(intensity, dtype=float)


def treat(two_theta: np.ndarray, intensity: np.ndarray, processing: Processing) -> Tuple[np.ndarray, np.ndarray]:
    """Return (smoothed, baseline). The background is estimated on the raw signal."""
    return smooth(two_theta, intensity, processing), snip_baseline(intensity, processing.baseline_degree)


def apply(spectrum: Spectrum) -> Spectrum:
    """Refresh `spectrum.baseline` from its processing parameters (in place)."""
    _, spectrum.baseline = treat(spectrum.two_theta, spectrum.intensity, spectrum.processing)
    return spectrum
