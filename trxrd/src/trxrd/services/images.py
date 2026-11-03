"""Use case: the raw detector image of a spectrum and its cake (2θ × azimuth), for display."""

from __future__ import annotations

import dataclasses
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np

from trxrd.domain import Run
from trxrd.io import esrf
from trxrd.processing.integration import Integrator, apply_flip
from trxrd.services import raw_data

PathLike = Union[str, Path]


@dataclass
class Cake:
    two_theta: np.ndarray  # deg, the 2θ grid of the run
    azimuth: np.ndarray  # deg
    intensity: np.ndarray  # [n_azimuth, n_two_theta], NaN where no pixel


class FrameSource:
    """Frames of the raw scan of a run (found from its source or its name) and their cakes.

    Building the pyFAI integrator takes about a second the first time: keep
    one FrameSource per run, and call it from a worker thread.
    """

    def __init__(self, run: Run, near: Iterable[PathLike] = ()) -> None:
        self.run = run
        self.image_file = raw_data.locate_raw_scan(run, near)
        self._stack = None
        self._integrator = None
        self._lock = threading.Lock()  # several worker threads may ask for frames of the same run

    @property
    def available(self) -> bool:
        return self.image_file is not None

    @property
    def problem(self) -> str:
        """Why no image can be shown ("" if they can)."""
        if self.image_file is None:
            return f"Images brutes introuvables pour {self.run.name} (fichier source déplacé, ou pas de RAW_DATA voisin)"
        if self.run.calibration is None or not self.run.calibration.poni:
            return "Le run n'a pas de calibration (.poni) : pas de cake"
        return ""

    def _frame_index(self, index: int) -> int:
        frame = self.run.spectra[index].frame
        return int(frame) if frame is not None and frame >= 0 else int(index)

    def _raw(self, index: int) -> np.ndarray:
        if self._stack is None:
            self._stack = esrf.ImageStack.open(self.image_file)
        return self._stack.frame(self._frame_index(index))

    def frame(self, index: int) -> np.ndarray:
        """Raw image of spectrum `index` (flipped like at integration), masked pixels NaN."""
        with self._lock:
            image = np.asarray(self._raw(index), dtype=float)
        calibration = self.run.calibration
        if calibration is not None:
            image = apply_flip(image, calibration.flip_axis)
            if calibration.mask is not None and calibration.mask.shape == image.shape:
                image = np.where(calibration.mask > 0, np.nan, image)
        return image

    def cake(self, index: int, n_azimuth: int = 360) -> Cake:
        if self.problem:
            raise ValueError(self.problem)
        with self._lock:
            if self._integrator is None:
                calibration = dataclasses.replace(self.run.calibration)  # the run is not modified
                if calibration.points_per_degree is None and len(self.run.spectra):
                    x = np.asarray(self.run.spectra[0].two_theta, dtype=float)  # same grid as the 1D patterns
                    calibration.points_per_degree = int(round((x.size - 1) / max(x[-1] - x[0], 1e-9)))
                self._integrator = Integrator(calibration)
            two_theta, azimuth, intensity = self._integrator.cake(self._raw(index), n_azimuth)
        intensity = np.where(intensity == 0, np.nan, intensity)  # pyFAI fills empty bins with 0
        return Cake(two_theta, azimuth, intensity)
