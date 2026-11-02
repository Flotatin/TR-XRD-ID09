"""Peak detection on a background-corrected pattern.

Every threshold is expressed in units of the LOCAL noise (signal-to-noise
ratio) and every width in degrees, never relative to the strongest peak or
in number of points: a weak peak next to a strong one is kept, and the result
does not depend on the 2θ sampling. Shoulders (peaks without their own
maximum, frequent when phases overlap) are found on the second derivative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, peak_widths, savgol_coeffs, savgol_filter

MAD_TO_SIGMA = 1.4826  # for a Gaussian, sigma = 1.4826 * median absolute deviation


@dataclass
class DetectedPeak:
    position: float  # 2θ (deg), sub-sample
    height: float  # smoothed corrected intensity at the peak
    prominence: float  # height above the surrounding minima (= height for a shoulder)
    fwhm: float  # deg, NaN for a shoulder
    snr: float  # prominence / local noise
    shoulder: bool = False


@dataclass
class DetectionOptions:
    min_snr: float = 5.0  # height above the local level, in local-noise sigmas (of the smoothed signal)
    smooth_points: int = 7  # Savitzky-Golay window (points) used before searching maxima
    curvature_points: int = 5  # Savitzky-Golay window (points) of the second derivative (shoulders)
    min_fwhm: float = 0.0  # deg
    max_fwhm: float = 1.5  # deg (wider bumps are background or amorphous signal)
    noise_window: float = 1.5  # deg, window of the local noise and of the local level
    peak_window: float = 0.6  # deg, how far the prominence of a peak is looked for
    edge_margin: float = 0.25  # deg: filters and background are unreliable this close to the ends
    shoulders: bool = True
    max_peaks: Optional[int] = None  # keep the strongest ones
    windows: Optional[Sequence[Tuple[float, float]]] = None  # 2θ zones where peaks are kept
    exclude: Optional[Sequence[Tuple[float, float]]] = None  # 2θ zones ignored (gasket, diamond, detector gaps...)


def _odd(n: float, minimum: int = 3) -> int:
    n = max(int(round(n)), minimum)
    return n if n % 2 else n + 1


def local_noise(y: np.ndarray, step: float, window_deg: float = 1.5) -> np.ndarray:
    """Standard deviation of the noise at every point.

    The signal is removed with a short Savitzky-Golay filter; for white noise
    the residual variance is (1 - h0) sigma², h0 being the central filter
    coefficient. A running median of |residual| then gives a robust local
    estimate (Poisson noise grows with the intensity).
    """
    y = np.asarray(y, dtype=float)
    window = 7
    h0 = savgol_coeffs(window, 2)[window // 2]
    residual = y - savgol_filter(y, window, 2)
    residual -= np.median(residual)
    size = _odd(max(5 * window, window_deg / step))
    noise = MAD_TO_SIGMA / np.sqrt(1 - h0) * median_filter(np.abs(residual), size=size, mode="nearest")
    floor = MAD_TO_SIGMA / np.sqrt(1 - h0) * np.median(np.abs(residual))
    return np.maximum(noise, 0.2 * floor + 1e-12)


def _refine(x: np.ndarray, y: np.ndarray, index: int) -> float:
    """Sub-sample position of a maximum by parabolic interpolation."""
    if index <= 0 or index >= len(y) - 1:
        return float(x[index])
    y0, y1, y2 = y[index - 1], y[index], y[index + 1]
    denom = y0 - 2 * y1 + y2
    if denom >= 0:
        return float(x[index])
    shift = 0.5 * (y0 - y2) / denom
    return float(np.interp(index + shift, np.arange(len(x)), x))


def detect_peaks(x: np.ndarray, y: np.ndarray, options: Optional[DetectionOptions] = None) -> List[DetectedPeak]:
    """Peaks of y(x); y must already be background-corrected (not smoothed)."""
    options = options or DetectionOptions()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 * options.smooth_points:
        return []
    step = float(np.median(np.diff(x)))
    noise = local_noise(y, step, options.noise_window)
    smooth_window = _odd(options.smooth_points, 5)
    smoothed = savgol_filter(y, smooth_window, 3)
    smooth_noise = noise * np.sqrt(np.sum(savgol_coeffs(smooth_window, 3) ** 2))
    # residual background left by the baseline: running median of the signal
    level = median_filter(smoothed, size=_odd(options.noise_window / step), mode="nearest")
    threshold = options.min_snr * smooth_noise

    # A noise maximum easily reaches a large prominence over a long distance, but
    # rarely a large height above the local level: the height is the criterion,
    # the (local) prominence only rejects bumps on the flank of another peak.
    index, props = find_peaks(smoothed, prominence=0.5 * threshold, wlen=_odd(options.peak_window / step))
    widths = peak_widths(smoothed, index, rel_height=0.5)[0] * step if index.size else np.array([])
    heights = smoothed[index] - level[index]
    keep = (heights >= threshold[index]) & (widths >= options.min_fwhm) & (widths <= options.max_fwhm)
    found: List[DetectedPeak] = []
    for i, height, prominence, fwhm in zip(index[keep], heights[keep], props["prominences"][keep], widths[keep]):
        found.append(DetectedPeak(_refine(x, smoothed, i), float(height), float(prominence), float(fwhm),
                                  float(height / smooth_noise[i])))

    if options.shoulders:
        found.extend(_shoulders(x, y, smoothed - level, noise, smooth_noise, step, found, options))

    lo, hi = x[0] + options.edge_margin, x[-1] - options.edge_margin
    found = [p for p in found if lo <= p.position <= hi]
    if options.windows:
        found = [p for p in found if any(lo <= p.position <= hi for lo, hi in options.windows)]
    if options.exclude:
        found = [p for p in found if not any(lo <= p.position <= hi for lo, hi in options.exclude)]
    if options.max_peaks is not None:
        found = sorted(found, key=lambda p: p.snr, reverse=True)[: options.max_peaks]
    return sorted(found, key=lambda p: p.position)


def _shoulders(x, y, smoothed, noise, smooth_noise, step, maxima, options) -> List[DetectedPeak]:
    """Local maxima of -y'' that are not already peak maxima."""
    window = _odd(options.curvature_points, 5)
    curvature = -savgol_filter(y, window, 3, deriv=2)
    curvature_noise = noise * np.sqrt(np.sum(savgol_coeffs(window, 3, deriv=2) ** 2))
    index, _ = find_peaks(curvature, height=options.min_snr * curvature_noise)
    shoulders = []
    for i in index:
        position = _refine(x, curvature, i)
        nearest = min(maxima, key=lambda p: abs(p.position - position), default=None)
        if nearest is not None and abs(nearest.position - position) < max(0.5 * nearest.fwhm, 3 * step):
            continue  # the curvature maximum of an ordinary peak
        if smoothed[i] < options.min_snr * smooth_noise[i]:
            continue  # curvature of the noise or of the background
        shoulders.append(DetectedPeak(position, float(smoothed[i]), float(smoothed[i]), np.nan,
                                      float(smoothed[i] / smooth_noise[i]), shoulder=True))
    return shoulders
