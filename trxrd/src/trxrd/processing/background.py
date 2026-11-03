"""1D treatment of an integrated pattern: smoothing and background (SNIP)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from pybaselines import Baseline
from scipy.signal import savgol_filter

from trxrd.domain import Processing, Spectrum


def snip_baseline(intensity: np.ndarray, degree: int, max_half_window: Optional[int] = None) -> np.ndarray:
    """SNIP background; `degree` sets the smoothing window (5*degree points), `max_half_window` (points) the
    largest window (None: estimated by pybaselines)."""
    baseline, _ = Baseline().snip(
        np.asarray(intensity, dtype=float), max_half_window=max_half_window, filter_order=2,
        smooth_half_window=5 * int(degree),
    )
    return baseline


def window_points(two_theta: np.ndarray, processing: Processing) -> Optional[int]:
    """The SNIP window of `processing` in points of `two_theta` (None: automatic)."""
    if processing.baseline_window is None:
        return None
    step = float(np.median(np.abs(np.diff(np.asarray(two_theta, dtype=float)))))
    return max(int(round(processing.baseline_window / step)), 1)


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


def noise_gain(two_theta: np.ndarray, processing: Processing) -> float:
    """Factor by which the filter of `processing` scales white noise: norm of its impulse response.

    sqrt(sum of the squared Savitzky-Golay coefficients) for "svg", 1 without filter.
    """
    n = len(two_theta)
    impulse = np.zeros(n)
    impulse[n // 2] = 1.0
    return float(np.sqrt(np.sum(smooth(np.asarray(two_theta, dtype=float), impulse, processing) ** 2)))


def check(processing: Processing, points: int, step: Optional[float] = None) -> None:
    """ValueError (in French: shown to the user) when `processing` cannot be applied to `points` points
    (2θ step `step`, deg, to check the window of the background)."""
    p1, p2 = processing.filter_params
    if processing.filter_type == "svg":
        if int(p1) != p1 or int(p1) % 2 == 0 or int(p1) < 3:
            raise ValueError("Savitzky-Golay : la fenêtre doit être un nombre impair de points (≥ 3)")
        if int(p2) != p2 or not 0 <= int(p2) < int(p1):
            raise ValueError("Savitzky-Golay : l'ordre doit être un entier plus petit que la fenêtre")
        if int(p1) > points:
            raise ValueError(f"Savitzky-Golay : fenêtre plus grande que le spectre ({points} points)")
    elif processing.filter_type == "fft":
        if not 0 <= p1 <= p2:
            raise ValueError("FFT : il faut 0 ≤ coupure basse ≤ coupure haute")
    elif processing.filter_type != "none":
        raise ValueError(f"Filtre inconnu : {processing.filter_type}")
    if int(processing.baseline_degree) != processing.baseline_degree or processing.baseline_degree < 0:
        raise ValueError("Le lissage du fond doit être un entier positif ou nul")
    if processing.baseline_window is not None and not processing.baseline_window > 0:
        raise ValueError("La fenêtre du fond doit être positive (vide : automatique)")
    if processing.baseline_window is not None and step and processing.baseline_window / step > points / 2:
        raise ValueError(f"Fenêtre du fond plus grande que la moitié du spectre ({points * step / 2:.2f}°)")


def corrected(spectrum: Spectrum) -> Tuple[np.ndarray, np.ndarray]:
    """(filtered - baseline, raw - baseline) of `spectrum`; the baseline is computed if missing."""
    if spectrum.baseline is None:
        apply(spectrum)
    raw = np.asarray(spectrum.intensity, dtype=float) - spectrum.baseline
    filtered = smooth(np.asarray(spectrum.two_theta, dtype=float), np.asarray(spectrum.intensity, dtype=float),
                      spectrum.processing)
    return filtered - spectrum.baseline, raw


def treat(two_theta: np.ndarray, intensity: np.ndarray, processing: Processing) -> Tuple[np.ndarray, np.ndarray]:
    """Return (smoothed, baseline). The background is estimated on the raw signal."""
    baseline = snip_baseline(intensity, processing.baseline_degree, window_points(two_theta, processing))
    return smooth(two_theta, intensity, processing), baseline


def apply(spectrum: Spectrum) -> Spectrum:
    """Refresh `spectrum.baseline` from its processing parameters (in place)."""
    _, spectrum.baseline = treat(spectrum.two_theta, spectrum.intensity, spectrum.processing)
    return spectrum
