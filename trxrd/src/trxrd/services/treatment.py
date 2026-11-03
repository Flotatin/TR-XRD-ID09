"""Use case: the 1D treatment of the spectra (background, filter) and the peak detection, as the user sees them.

The treatment is set spectrum by spectrum, or propagated to every spectrum
or to a selection. The signal shown by default is the filtered signal minus
the background: the peaks are searched on it (when the detection takes the
filter of the spectrum), while the fit always uses the raw intensity minus
the background (docs/PASSATION.md §5, point 1). `detection` gives exactly
what the automatic analysis detects, with the threshold, to check it.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np

from trxrd.domain import Processing, Run, Spectrum
from trxrd.processing import background
from trxrd.processing.peaks import DetectedPeak, detect_peaks, detection_levels
from trxrd.services.analysis import AnalysisSettings
from trxrd.services.identification import analysis_window, detection_signals

FILTERS = ("svg", "fft", "none")


def corrected(spectrum: Spectrum) -> Tuple[np.ndarray, np.ndarray]:
    """(filtered - background, raw - background) of `spectrum` (its background is computed if missing)."""
    return background.corrected(spectrum)


def check(processing: Processing, spectrum: Spectrum) -> None:
    """ValueError, with a message for the user, when `processing` cannot be applied to `spectrum`."""
    x = np.asarray(spectrum.two_theta, dtype=float)
    step = float(np.median(np.abs(np.diff(x)))) if x.size > 1 else None
    background.check(processing, x.size, step)


def set_processing(run: Run, indices: Iterable[int], processing: Processing) -> List[int]:
    """Give `processing` to the spectra `indices` and recompute their background; returns the indices.

    Their fits are kept: they were made on the former background (refit them).
    """
    indices = list(indices)
    for index in indices:
        check(processing, run.spectra[index])
    for index in indices:
        spectrum = run.spectra[index]
        spectrum.processing = dataclasses.replace(processing)  # each spectrum its own copy
        background.apply(spectrum)
    return indices


@dataclass
class Detection:
    """Peaks detected in one spectrum, as the automatic analysis detects them, and why."""

    x: np.ndarray  # 2θ (deg) of the analysed window
    baseline: np.ndarray  # background (raw intensity units)
    smoothed: np.ndarray  # signal the maxima are searched on (background-corrected)
    threshold: np.ndarray  # a peak must rise above this curve (local level + min_snr x noise)
    peaks: List[DetectedPeak]
    excluded: List[bool]  # in an excluded zone: fitted as free peaks, assigned to no phase
    broad: List[bool]  # broader than the search accepts: neither explains nor costs anything
    exclude: List[Tuple[float, float]]
    broad_fwhm: Optional[float]  # deg

    @property
    def used(self) -> int:
        """Peaks the phase search works with."""
        return sum(not e and not b for e, b in zip(self.excluded, self.broad))


def detection(run: Run, index: int, settings: Optional[AnalysisSettings] = None) -> Detection:
    settings = settings or AnalysisSettings()
    options = settings.options()
    window = analysis_window(run, options)
    spectrum = run.spectra[index]
    x, raw, smoothed, gain = detection_signals(spectrum, window, options.detection)
    peaks = detect_peaks(x, raw, options.detection, smoothed, gain)
    levels = detection_levels(x, raw, options.detection, smoothed, gain)
    all_x = np.asarray(spectrum.two_theta, dtype=float)
    inside = (all_x >= window[0]) & (all_x <= window[1])
    zones = list(options.exclude or [])
    broad_fwhm = options.search.max_fwhm
    return Detection(
        x=x,
        baseline=np.asarray(spectrum.baseline, dtype=float)[inside],
        smoothed=levels.smoothed,
        threshold=levels.level + levels.threshold,
        peaks=peaks,
        excluded=[any(lo <= p.position <= hi for lo, hi in zones) for p in peaks],
        broad=[broad_fwhm is not None and np.isfinite(p.fwhm) and p.fwhm > broad_fwhm for p in peaks],
        exclude=zones,
        broad_fwhm=broad_fwhm,
    )
