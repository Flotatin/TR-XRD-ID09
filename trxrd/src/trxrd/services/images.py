"""Use case: raw detector images for display: the image and cake of a spectrum, the preview of a raw scan."""

from __future__ import annotations

import dataclasses
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np

from trxrd.domain import Run
from trxrd.io import esrf
from trxrd.io.calibration import calibration_from_files, read_mask
from trxrd.processing.integration import Integrator, apply_flip, azimuthal_integrator
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


@dataclass
class Pattern:
    two_theta: np.ndarray  # deg
    intensity: np.ndarray  # NaN where no usable pixel
    coverage: np.ndarray  # deg of azimuth seen by usable pixels, per 2θ bin


class ScanPreview:
    """A raw scan before its integration, to choose the 2θ range: frames, 2θ of each pixel, pattern and cake.

    The pattern and the cake span every usable pixel of the detector; the run
    is integrated later on the chosen range only. Without a .poni file only
    the images are available. The 2θ map and the pyFAI integrator take about
    a second the first time: call it from a worker thread.
    """

    MAX_FRAMES = 32  # frames read for the maximum of a long scan, evenly spaced

    def __init__(self, image_file: PathLike, poni_file: Optional[PathLike] = None,
                 mask_file: Optional[PathLike] = None) -> None:
        self.image_file = Path(image_file)
        self.key = (str(image_file), str(poni_file or ""), str(mask_file or ""))
        self.stack = esrf.ImageStack.open(self.image_file)
        self.frames = len(self.stack)
        self.mask = read_mask(mask_file) if mask_file else None
        self.calibration = (calibration_from_files(poni_file, mask_file, two_theta_range=(0.0, 180.0))
                            if poni_file else None)
        self._maps: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._integrator: Optional[Integrator] = None
        self._coverage: Optional[np.ndarray] = None
        self._lock = threading.RLock()  # worker threads may overlap; pattern() takes it again in angle_maps()

    # ------------------------------------------------------------------ images

    def maximum_frames(self) -> List[int]:
        """Frames of the maximum: all of a short scan, MAX_FRAMES evenly spaced ones of a long one."""
        count = min(self.frames, self.MAX_FRAMES)
        return sorted({int(i) for i in np.linspace(0, self.frames - 1, count).round()})

    def image(self, frame: Optional[int] = None,
              progress: Optional[Callable[[int, int], bool]] = None) -> Optional[np.ndarray]:
        """Raw frame `frame`, or (None) the pixel-wise maximum over the scan: every line seen at any time.

        Returns None when `progress(done, total)` asks to stop.
        """
        if frame is not None:
            return np.asarray(self.stack.frame(int(np.clip(frame, 0, self.frames - 1))), dtype=np.float32)
        indices = self.maximum_frames()
        peak = None
        for done, (_, image) in enumerate(self.stack.frames(indices), start=1):
            image = np.asarray(image, dtype=np.float32)
            peak = image.copy() if peak is None else np.maximum(peak, image, out=peak)
            if progress is not None and not progress(done, len(indices)):
                return None
        return peak

    def _usable_mask(self, shape) -> Optional[np.ndarray]:
        return self.mask if self.mask is not None and self.mask.shape == tuple(shape) else None

    def shown(self, image: np.ndarray) -> np.ndarray:
        """`image` as it is integrated (flipped like at integration), masked pixels NaN."""
        flip = self.calibration.flip_axis if self.calibration is not None else None
        image = apply_flip(np.asarray(image, dtype=float), flip)
        mask = self._usable_mask(image.shape)
        return image if mask is None else np.where(mask > 0, np.nan, image)

    # ------------------------------------------------------------------ geometry (needs the .poni)

    def angle_maps(self, shape) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """(2θ, azimuth) of each pixel in degrees, NaN where masked; None without calibration."""
        if self.calibration is None:
            return None
        with self._lock:
            if self._maps is None or self._maps[0].shape != tuple(shape):
                ai = azimuthal_integrator(self.calibration)
                two_theta = np.asarray(ai.center_array(tuple(shape), unit="2th_deg"), dtype=float)
                azimuth = np.asarray(ai.center_array(tuple(shape), unit="chi_deg"), dtype=float)
                mask = self._usable_mask(shape)
                if mask is not None:
                    two_theta, azimuth = (np.where(mask > 0, np.nan, a) for a in (two_theta, azimuth))
                self._maps = two_theta, azimuth
            return self._maps

    def usable_range(self, shape) -> Optional[Tuple[float, float]]:
        """2θ range (deg) of the usable pixels."""
        maps = self.angle_maps(shape)
        if maps is None:
            return None
        return float(np.floor(np.nanmin(maps[0]) * 100) / 100), float(np.ceil(np.nanmax(maps[0]) * 100) / 100)

    def _full_integrator(self, shape) -> Integrator:
        if self.calibration is None:
            raise ValueError("Pas de calibration (.poni) : pas de diagramme en 2θ")
        usable = self.usable_range(shape)
        with self._lock:
            if self._integrator is None:
                calibration = dataclasses.replace(self.calibration, two_theta_range=usable, points_per_degree=None)
                self._integrator = Integrator(calibration)
            return self._integrator

    def _coverage_of(self, two_theta: np.ndarray, shape) -> np.ndarray:
        """Degrees of azimuth (1° bins) holding at least one usable pixel, for each 2θ bin."""
        if self._coverage is None or self._coverage.size != two_theta.size:
            tth, chi = self.angle_maps(shape)
            good = np.isfinite(tth)
            step = (two_theta[-1] - two_theta[0]) / max(two_theta.size - 1, 1)
            column = np.floor((tth[good] - (two_theta[0] - step / 2)) / step).astype(int)
            row = np.floor(chi[good] + 180.0).astype(int) % 360
            keep = (column >= 0) & (column < two_theta.size)
            seen = np.zeros((two_theta.size, 360), dtype=bool)
            seen[column[keep], row[keep]] = True
            self._coverage = seen.sum(axis=1).astype(float)
        return self._coverage

    def pattern(self, image: np.ndarray) -> Pattern:
        """I(2θ) of `image` over every usable pixel, with the azimuthal coverage of each 2θ bin."""
        integrator = self._full_integrator(image.shape)
        with self._lock:
            two_theta, intensity = integrator.integrate(image)
            coverage = self._coverage_of(two_theta, image.shape)
        return Pattern(two_theta, np.where(coverage > 0, intensity, np.nan), coverage)

    def cake(self, image: np.ndarray, n_azimuth: int = 360) -> Cake:
        integrator = self._full_integrator(image.shape)
        with self._lock:
            two_theta, azimuth, intensity = integrator.cake(image, n_azimuth)
        return Cake(two_theta, azimuth, np.where(intensity == 0, np.nan, intensity))  # empty bins are 0 in pyFAI
