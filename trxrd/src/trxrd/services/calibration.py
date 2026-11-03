"""Use case: calibrate the detector geometry on a LaB6 / CeO2 image, save the .poni, use it for runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from trxrd.domain import Calibration, wavelength_to_energy
from trxrd.io import esrf
from trxrd.io.calibration import read_mask
from trxrd.processing.geometry import (
    CALIBRANTS, ControlPoints, Geometry, RefinementResult, angle_maps, calibrant_rings, extract_rings, follow_ring,
    refine, ring_overlay, usable_pixels,
)

PathLike = Union[str, Path]

# (window around the rings in deg, parameters kept fixed): rotations are freed only once every visible
# ring holds points; with a beam centre off the detector, distance, centre and tilt are strongly coupled
STAGES = (
    (0.3, ("wavelength", "rot1", "rot2", "rot3")),
    (0.2, ("wavelength", "rot3")),
    (0.1, ("wavelength",)),
    (0.1, ("wavelength",)),
)


def load_calibration_image(path: PathLike, frames: Optional[Sequence[int]] = None) -> np.ndarray:
    """Calibrant image: mean of the frames of an HDF5 stack (all by default), or a single image file (fabio)."""
    path = Path(path)
    if path.suffix.lower() in (".h5", ".hdf5", ".nxs"):
        stack = esrf.ImageStack.open(path)
        total, count = None, 0
        for _, frame in stack.frames(frames):
            frame = np.asarray(frame, dtype=float)
            total = frame if total is None else total + frame
            count += 1
        if total is None:
            raise ValueError(f"No frame in {path}")
        return total / count
    import fabio

    return np.asarray(fabio.open(str(path)).data, dtype=float)


def load_mask(path: PathLike) -> np.ndarray:
    return read_mask(path)


def _brightest_near(image: np.ndarray, good: np.ndarray, row: int, col: int, radius: int = 4) -> Tuple[int, int]:
    """A click is never exactly on the ring: the brightest usable pixel around it."""
    r0, r1 = max(row - radius, 0), min(row + radius + 1, image.shape[0])
    c0, c1 = max(col - radius, 0), min(col + radius + 1, image.shape[1])
    patch = np.where(good[r0:r1, c0:c1], image[r0:r1, c0:c1], -np.inf)
    if not np.isfinite(patch).any():
        return row, col
    r, c = np.unravel_index(int(np.argmax(patch)), patch.shape)
    return r0 + int(r), c0 + int(c)


@dataclass
class CalibrationState:
    """What the calibration dialog works on."""

    image: np.ndarray
    geometry: Geometry
    calibrant: str = "LaB6"
    mask: Optional[np.ndarray] = None
    points: ControlPoints = field(default_factory=ControlPoints)
    result: Optional[RefinementResult] = None
    _maps: Optional[Tuple[Geometry, Tuple[np.ndarray, np.ndarray]]] = None

    def maps(self) -> Tuple[np.ndarray, np.ndarray]:
        """2θ, χ of the current geometry (cached: ~0.3 s for a Jungfrau 1M)."""
        if self._maps is None or self._maps[0] != self.geometry:
            self._maps = (self.geometry, angle_maps(self.geometry, self.image.shape))
        return self._maps[1]

    def rings(self) -> np.ndarray:
        return calibrant_rings(self.calibrant, self.geometry.wavelength)

    def pick(self, row: float, col: float, ring: int) -> int:
        """Points of the ring through the clicked pixel (number `ring`), added; returns how many."""
        two_theta, chi = self.maps()
        good = usable_pixels(self.image, self.mask)
        r = int(np.clip(round(row), 0, self.image.shape[0] - 1))
        c = int(np.clip(round(col), 0, self.image.shape[1] - 1))
        r, c = _brightest_near(self.image, good, r, c)
        found = follow_ring(self.image, good, two_theta, chi, r, c)
        self.points.add(found, ring)
        return len(found)

    def refine(self, fix: Sequence[str] = ("wavelength", "rot3")) -> RefinementResult:
        self.result = refine(self.geometry, self.points, self.calibrant, fix)
        self.geometry = self.result.geometry
        return self.result

    def extract(self, window: float = 0.1) -> int:
        """Replace the points by the rings found around their expected positions."""
        self.points = extract_rings(self.image, self.geometry, self.calibrant, self.mask, window, maps=self.maps())
        return len(self.points)

    def calibrate(self, stages=STAGES) -> RefinementResult:
        """From the clicked points (or the start geometry alone): refine, extract, refine... (see STAGES)."""
        if len(self.points) >= 6:
            self.refine(("wavelength", "rot1", "rot2", "rot3"))
        for window, fix in stages:
            if self.extract(window) < 6:
                raise ValueError("Pas assez de points sur les anneaux : pointer un anneau ou corriger la géométrie de départ")
            self.refine(fix)
        return self.result


def save_poni(geometry: Geometry, path: PathLike) -> Path:
    path = Path(path)
    path.write_text(geometry.to_poni_text(), encoding="utf-8")
    return path


def calibration_from_geometry(geometry: Geometry, mask: Optional[np.ndarray] = None, mask_file: str = "",
                              two_theta_range: Tuple[float, float] = (0.0, 40.0), poni_file: str = "") -> Calibration:
    """Calibration for build_run, the .poni text embedded."""
    return Calibration(
        energy_ev=float(wavelength_to_energy(geometry.wavelength)),
        poni=geometry.to_poni_text(),
        mask=None if mask is None else (np.asarray(mask) > 0).astype(np.uint8),
        two_theta_range=(float(two_theta_range[0]), float(two_theta_range[1])),
        poni_file=str(poni_file),
        mask_file=str(mask_file),
    )


__all__ = ["CALIBRANTS", "CalibrationState", "Geometry", "calibration_from_geometry", "load_calibration_image",
           "load_mask", "ring_overlay", "save_poni"]
