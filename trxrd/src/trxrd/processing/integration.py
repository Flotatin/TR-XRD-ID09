"""Azimuthal integration of detector images with pyFAI."""

from __future__ import annotations

import os
import tempfile
from typing import Optional, Tuple

import numpy as np

from trxrd.domain import Calibration


def apply_flip(image: np.ndarray, flip_axis: Optional[int]) -> np.ndarray:
    """flip_axis: None (no flip), 0 or 1 (that axis), 2 (both axes)."""
    if flip_axis is None:
        return image
    if flip_axis == 2:
        return np.flip(image)
    return np.flip(image, axis=flip_axis)


def azimuthal_integrator(calibration: Calibration):
    """pyFAI integrator built from the .poni text embedded in the calibration."""
    import pyFAI

    if not calibration.poni:
        raise ValueError("Calibration has no .poni content")
    fd, tmp = tempfile.mkstemp(suffix=".poni")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(calibration.poni)
        ai = pyFAI.load(tmp)
    finally:
        os.remove(tmp)
    ai.wavelength = calibration.wavelength_m  # the run energy is authoritative
    return ai


def pixel_two_theta_width(ai, two_theta_range: Tuple[float, float], shape=None) -> float:
    """Smallest 2θ step between neighbouring pixels inside the range (deg)."""
    two_theta = ai.center_array(shape or ai.detector.shape, unit="2th_deg")
    step = np.hypot(*np.gradient(two_theta))
    inside = (two_theta >= two_theta_range[0]) & (two_theta <= two_theta_range[1])
    return float(step[inside].min()) if inside.any() else float(step.min())


class Integrator:
    """Integrates images into I(2theta) on the calibration's 2theta grid.

    With `points_per_degree=None` the grid step is set to the smallest
    angular size of a pixel in the 2θ range: no resolution is lost and the
    bins are not emptier than the pixels. The value is written back into the
    calibration so that it is saved with the run.
    """

    def __init__(self, calibration: Calibration):
        self.calibration = calibration
        self.ai = azimuthal_integrator(calibration)
        if calibration.points_per_degree is None:
            shape = calibration.mask.shape if calibration.mask is not None else None
            width = pixel_two_theta_width(self.ai, calibration.two_theta_range, shape)
            calibration.points_per_degree = int(np.ceil(1.0 / width))

    def integrate(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        calib = self.calibration
        result = self.ai.integrate1d(
            apply_flip(np.asarray(image), calib.flip_axis),
            calib.n_points,
            mask=calib.mask,
            unit="2th_deg",
            radial_range=tuple(calib.two_theta_range),
        )
        return np.asarray(result.radial, dtype=float), np.asarray(result.intensity, dtype=float)
