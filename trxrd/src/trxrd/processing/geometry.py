"""Detector geometry from a calibrant image (LaB6, CeO2), in the spirit of Dioptas.

Pixel coordinates are (row, col) = (d1, d2) of pyFAI, in the image array as
read. The steps:

1. a start geometry: a .poni file, or a guess (distance, beam centre);
2. `ring_points`: points along a ring. In each azimuth sector the intensity
   is averaged versus 2θ (of the current geometry) and the centroid of the
   peak taken, then moved back to a sub-pixel position along the 2θ gradient.
   Averaging a sector copes with spotty rings; a smooth 2θ(χ) curve fitted
   through the sectors (harmonics up to 2χ, sigma clipping) removes the
   outliers and absorbs the errors of a poor start geometry;
3. `refine`: least squares of pyFAI (GeometryRefinement) on the points with
   their ring numbers, some parameters fixed;
4. `extract_rings`: `ring_points` around every expected ring of the refined
   geometry, then refine again.

pyFAI.goniometer (SingleGeometry.extract_cp) and the silx contour code are
not used: Windows application control blocks a silx library on some
machines.
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

CALIBRANTS = ("LaB6", "CeO2")
PARAMETERS = ("dist", "poni1", "poni2", "rot1", "rot2", "rot3", "wavelength")
HC_EV_M = 1.23984198e-6


@dataclass
class Geometry:
    """pyFAI geometry (distances in m, angles in rad) and the detector it refers to."""

    dist: float
    poni1: float
    poni2: float
    rot1: float = 0.0
    rot2: float = 0.0
    rot3: float = 0.0
    wavelength: float = 6.525e-11
    detector: str = "Jungfrau1M"
    detector_config: Dict = field(default_factory=lambda: {"pixel1": 75e-6, "pixel2": 75e-6, "max_shape": [1064, 1030],
                                                             "orientation": 2})

    @property
    def energy_ev(self) -> float:
        return HC_EV_M / self.wavelength

    @property
    def pixel_size(self) -> Tuple[float, float]:
        return float(self.detector_config.get("pixel1", 75e-6)), float(self.detector_config.get("pixel2", 75e-6))

    def integrator(self):
        from pyFAI.detectors import detector_factory
        from pyFAI.integrator.azimuthal import AzimuthalIntegrator

        detector = detector_factory(self.detector, config=dict(self.detector_config))
        return AzimuthalIntegrator(dist=self.dist, poni1=self.poni1, poni2=self.poni2, rot1=self.rot1, rot2=self.rot2,
                                   rot3=self.rot3, detector=detector, wavelength=self.wavelength)

    @classmethod
    def from_integrator(cls, ai) -> "Geometry":
        config = ai.detector.get_config() if hasattr(ai.detector, "get_config") else {}
        name = getattr(ai.detector, "aliases", None)
        return cls(float(ai.dist), float(ai.poni1), float(ai.poni2), float(ai.rot1), float(ai.rot2), float(ai.rot3),
                   float(ai.wavelength), (name[0] if name else ai.detector.__class__.__name__).replace(" ", ""),
                   dict(config))

    @classmethod
    def from_poni_text(cls, text: str) -> "Geometry":
        import pyFAI

        fd, path = tempfile.mkstemp(suffix=".poni")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
            return cls.from_integrator(pyFAI.load(path))
        finally:
            os.remove(path)

    def to_poni_text(self) -> str:
        fd, path = tempfile.mkstemp(suffix=".poni")
        os.close(fd)
        os.remove(path)  # pyFAI appends to an existing file
        try:
            self.integrator().write(path)
            with open(path, encoding="utf-8") as handle:
                return handle.read()
        finally:
            if os.path.exists(path):
                os.remove(path)

    @classmethod
    def from_guess(cls, distance_m: float, center_col: float, center_row: float, wavelength_m: float,
                   detector: str = "Jungfrau1M", detector_config: Optional[Dict] = None) -> "Geometry":
        """No tilt, direct beam at pixel (center_col, center_row), which may lie off the detector."""
        geometry = cls(distance_m, 0.0, 0.0, wavelength=wavelength_m, detector=detector)
        if detector_config is not None:
            geometry.detector_config = dict(detector_config)
        ai = geometry.integrator()
        ai.setFit2D(distance_m * 1e3, center_col, center_row)
        return replace(geometry, poni1=float(ai.poni1), poni2=float(ai.poni2), rot1=float(ai.rot1),
                       rot2=float(ai.rot2), rot3=float(ai.rot3))

    def beam_center(self) -> Tuple[float, float]:
        """(col, row) of the direct beam, in pixels (fit2d convention)."""
        fit2d = self.integrator().getFit2D()
        return float(fit2d["centerX"]), float(fit2d["centerY"])


def calibrant_rings(calibrant: str, wavelength_m: float, two_theta_max: float = 180.0) -> np.ndarray:
    """2θ (deg) of the calibrant rings, sorted, coincident lines merged."""
    from pyFAI.calibrant import get_calibrant

    if calibrant not in CALIBRANTS:
        raise ValueError(f"unknown calibrant {calibrant!r}, expected one of {CALIBRANTS}")
    reference = get_calibrant(calibrant)
    reference.wavelength = wavelength_m
    rings = np.degrees(np.asarray([t for t in reference.get_2th() if t is not None and np.isfinite(t)], dtype=float))
    rings = np.unique(np.round(rings[rings <= two_theta_max], 6))
    return rings[np.r_[True, np.diff(rings) > 1e-4]]


def angle_maps(geometry: Geometry, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """2θ and χ (deg) at the centre of every pixel."""
    ai = geometry.integrator()
    return ai.center_array(shape, unit="2th_deg"), ai.center_array(shape, unit="chi_deg")


def usable_pixels(image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Pixels to use: not masked, finite, not a negative detector artefact."""
    good = np.isfinite(image) & (image > -1.0)
    if mask is not None:
        good &= np.asarray(mask) == 0
    return good


def _harmonic_inliers(chi: np.ndarray, y: np.ndarray, weight: np.ndarray, order: int = 2, clip: float = 3.0) -> np.ndarray:
    x = np.radians(chi)
    design = np.column_stack([np.ones_like(x)] + [f(k * x) for k in range(1, order + 1) for f in (np.cos, np.sin)])
    if len(y) <= design.shape[1] + 2:
        design = design[:, :1]  # too few points: a constant
    keep = np.ones(len(y), bool)
    for _ in range(6):
        coef, *_ = np.linalg.lstsq(design[keep] * weight[keep, None], y[keep] * weight[keep], rcond=None)
        residual = y - design @ coef
        sigma = 1.4826 * np.median(np.abs(residual[keep])) + 1e-6
        new = np.abs(residual) < clip * sigma
        if (new == keep).all():
            break
        keep = new
    return keep


def ring_points(image: np.ndarray, good: np.ndarray, two_theta: np.ndarray, chi: np.ndarray, ring_two_theta: float,
                window: float, chi_step: float = 1.0, two_theta_step: Optional[float] = None, min_snr: float = 5.0,
                min_sectors: int = 5) -> np.ndarray:
    """(N, 2) sub-pixel (row, col) on the ring seen near `ring_two_theta` (±window, deg, of this geometry)."""
    band = good & (np.abs(two_theta - ring_two_theta) < window)
    rows, cols = np.nonzero(band)
    if rows.size < 20:
        return np.zeros((0, 2))
    gradient = np.gradient(two_theta)
    if two_theta_step is None:  # half a pixel in 2θ
        two_theta_step = 0.5 * float(np.median(np.hypot(gradient[0][rows, cols], gradient[1][rows, cols])))
    t, c, v = two_theta[rows, cols], chi[rows, cols], np.asarray(image, dtype=float)[rows, cols]
    n_bins = max(int(np.ceil(2 * window / two_theta_step)), 10)
    t_bin = np.clip(((t - (ring_two_theta - window)) / (2 * window) * n_bins).astype(int), 0, n_bins - 1)
    centers = ring_two_theta - window + (np.arange(n_bins) + 0.5) * (2 * window / n_bins)
    sector = np.floor((c + 180.0) / chi_step).astype(int)
    edges = np.r_[np.arange(n_bins // 5), np.arange(n_bins - n_bins // 5, n_bins)]
    found = []
    order = np.argsort(sector, kind="stable")
    bounds = np.flatnonzero(np.r_[True, np.diff(sector[order]) != 0, True])
    for start, stop in zip(bounds[:-1], bounds[1:]):
        members = order[start:stop]
        if members.size < 20:
            continue
        counts = np.bincount(t_bin[members], minlength=n_bins)
        if (counts > 0).sum() < 0.6 * n_bins:
            continue
        profile = np.bincount(t_bin[members], weights=v[members], minlength=n_bins) / np.maximum(counts, 1)
        profile[counts == 0] = np.nan
        e = edges[np.isfinite(profile[edges])]
        if e.size < 4:
            continue
        slope, offset = np.polyfit(centers[e], profile[e], 1)
        signal = np.nan_to_num(profile - (slope * centers + offset), nan=0.0)
        noise = float(np.std(signal[e])) + 1e-9
        k = int(np.argmax(signal))
        snr = signal[k] / noise
        if snr < min_snr or k in (0, n_bins - 1):
            continue
        lo, hi = k, k
        while lo > 0 and signal[lo - 1] > signal[k] / 2:
            lo -= 1
        while hi < n_bins - 1 and signal[hi + 1] > signal[k] / 2:
            hi += 1
        peak = float(np.sum(centers[lo:hi + 1] * signal[lo:hi + 1]) / np.sum(signal[lo:hi + 1]))
        middle = (sector[members[0]] + 0.5) * chi_step - 180.0
        j = members[np.argmin(np.abs(t[members] - peak) / two_theta_step + np.abs(c[members] - middle) / chi_step)]
        r, q = rows[j], cols[j]
        g1, g2 = gradient[0][r, q], gradient[1][r, q]
        norm = g1 * g1 + g2 * g2
        step = (peak - two_theta[r, q]) / norm if norm > 0 else 0.0
        found.append((r + step * g1, q + step * g2, peak, middle, snr))
    if len(found) < min_sectors:
        return np.zeros((0, 2))
    found = np.asarray(found)
    keep = _harmonic_inliers(found[:, 3], found[:, 2], np.minimum(found[:, 4], 50.0) / 50.0)
    return found[keep, :2]


def _profile_peak(t: np.ndarray, v: np.ndarray, center: float, window: float, step: float, min_snr: float):
    """Centroid (deg) and SNR of the peak of the mean intensity versus 2θ within center ± window; None if none."""
    n_bins = max(int(np.ceil(2 * window / step)), 10)
    inside = np.abs(t - center) < window
    if inside.sum() < 10:
        return None
    t, v = t[inside], v[inside]
    index = np.clip(((t - (center - window)) / (2 * window) * n_bins).astype(int), 0, n_bins - 1)
    counts = np.bincount(index, minlength=n_bins)
    if (counts > 0).sum() < 0.6 * n_bins:
        return None
    profile = np.bincount(index, weights=v, minlength=n_bins) / np.maximum(counts, 1)
    profile[counts == 0] = np.nan
    x = center - window + (np.arange(n_bins) + 0.5) * (2 * window / n_bins)
    edges = np.r_[np.arange(n_bins // 5), np.arange(n_bins - n_bins // 5, n_bins)]
    edges = edges[np.isfinite(profile[edges])]
    if edges.size < 4:
        return None
    slope, offset = np.polyfit(x[edges], profile[edges], 1)
    signal = np.nan_to_num(profile - (slope * x + offset), nan=0.0)
    k = int(np.argmax(signal))
    snr = signal[k] / (float(np.std(signal[edges])) + 1e-9)
    if snr < min_snr or k in (0, n_bins - 1):
        return None
    lo, hi = k, k
    while lo > 0 and signal[lo - 1] > signal[k] / 2:
        lo -= 1
    while hi < n_bins - 1 and signal[hi + 1] > signal[k] / 2:
        hi += 1
    return float(np.sum(x[lo:hi + 1] * signal[lo:hi + 1]) / np.sum(signal[lo:hi + 1])), float(snr)


def follow_ring(image: np.ndarray, good: np.ndarray, two_theta: np.ndarray, chi: np.ndarray, row: int, col: int,
                window: float = 0.15, reach: float = 1.5, chi_step: float = 2.0, min_snr: float = 4,
                max_gap: int = 10) -> np.ndarray:
    """Points of the ring through the clicked pixel, even when the geometry is poor ("click on a ring").

    From the clicked sector, the ring is followed sector by sector in both
    azimuth directions; the 2θ window of each sector is centred on the peak
    found in the previous one, so the ring may drift (up to `reach` deg) away
    from the iso-2θ lines of the current geometry.
    """
    start_t, start_c = float(two_theta[row, col]), float(chi[row, col])
    band = good & (np.abs(two_theta - start_t) < reach)
    rows, cols = np.nonzero(band)
    if rows.size < 20:
        return np.zeros((0, 2))
    gradient = np.gradient(two_theta)
    step = 0.5 * float(np.median(np.hypot(gradient[0][rows, cols], gradient[1][rows, cols])))
    t, c, v = two_theta[rows, cols], chi[rows, cols], np.asarray(image, dtype=float)[rows, cols]
    sector = np.floor((c - start_c) / chi_step + 0.5).astype(int)  # 0 = sector of the click
    members = {}
    for k, s in enumerate(sector):
        members.setdefault(int(s), []).append(k)
    found = []
    for direction in (1, -1):
        center, gap, s = start_t, 0, 0 if direction == 1 else -1
        while gap <= max_gap and abs(s) * chi_step <= 360:
            index = np.asarray(members.get(s, []), dtype=int)
            peak = _profile_peak(t[index], v[index], center, window, step, min_snr) if index.size else None
            if peak is None:
                gap += 1
            else:
                gap, center = 0, peak[0]
                j = index[np.argmin(np.abs(t[index] - center) / step + np.abs(c[index] - (start_c + s * chi_step)) / chi_step)]
                r, q = rows[j], cols[j]
                g1, g2 = gradient[0][r, q], gradient[1][r, q]
                norm = g1 * g1 + g2 * g2
                shift = (center - two_theta[r, q]) / norm if norm > 0 else 0.0
                found.append((r + shift * g1, q + shift * g2))
            s += direction
    return np.asarray(found, dtype=float).reshape(-1, 2)


@dataclass
class ControlPoints:
    """Points on the rings: (row, col) and the ring number (0 = first ring of the calibrant)."""

    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    rings: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    def __len__(self) -> int:
        return int(len(self.rings))

    def add(self, points: np.ndarray, ring: int) -> None:
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        self.points = np.vstack([self.points, points])
        self.rings = np.r_[self.rings, np.full(len(points), int(ring))]

    def remove_ring(self, ring: int) -> None:
        keep = self.rings != ring
        self.points, self.rings = self.points[keep], self.rings[keep]

    def ring_counts(self) -> Dict[int, int]:
        values, counts = np.unique(self.rings, return_counts=True)
        return {int(v): int(n) for v, n in zip(values, counts)}

    def as_pyfai_array(self) -> np.ndarray:
        return np.column_stack([self.points, self.rings]) if len(self) else np.zeros((0, 3))


@dataclass
class RefinementResult:
    geometry: Geometry
    rms_deg: float
    residuals_deg: np.ndarray  # one per point
    n_points: int
    n_rings: int


def residuals(geometry: Geometry, points: ControlPoints, rings: np.ndarray) -> np.ndarray:
    """2θ(point) - 2θ(its ring), deg."""
    if not len(points):
        return np.zeros(0)
    ai = geometry.integrator()
    return np.degrees(ai.tth(points.points[:, 0], points.points[:, 1])) - rings[points.rings]


def refine(geometry: Geometry, points: ControlPoints, calibrant: str, fix: Sequence[str] = ("wavelength", "rot3")
           ) -> RefinementResult:
    """Least squares of the geometry on the control points (pyFAI GeometryRefinement); `fix`: parameters kept."""
    from pyFAI.calibrant import get_calibrant
    from pyFAI.geometryRefinement import GeometryRefinement

    unknown = set(fix) - set(PARAMETERS)
    if unknown:
        raise ValueError(f"unknown parameters {sorted(unknown)}; expected among {PARAMETERS}")
    if len(set(points.rings.tolist())) < 1 or len(points) < 6:
        raise ValueError("at least 6 points on the rings are needed")
    reference = get_calibrant(calibrant)
    reference.wavelength = geometry.wavelength
    ai = geometry.integrator()
    fit = GeometryRefinement(points.as_pyfai_array(), calibrant=reference, detector=ai.detector,
                             wavelength=geometry.wavelength, dist=geometry.dist, poni1=geometry.poni1,
                             poni2=geometry.poni2, rot1=geometry.rot1, rot2=geometry.rot2, rot3=geometry.rot3)
    with contextlib.redirect_stdout(io.StringIO()):  # pyFAI prints its iterations
        fit.refine2(fix=list(fix))
    refined = replace(geometry, dist=float(fit.dist), poni1=float(fit.poni1), poni2=float(fit.poni2),
                      rot1=float(fit.rot1), rot2=float(fit.rot2), rot3=float(fit.rot3),
                      wavelength=float(fit.wavelength))
    rings = calibrant_rings(calibrant, refined.wavelength)
    error = residuals(refined, points, rings)
    return RefinementResult(refined, float(np.sqrt(np.mean(error ** 2))), error, len(points),
                            len(set(points.rings.tolist())))


def extract_rings(image: np.ndarray, geometry: Geometry, calibrant: str, mask: Optional[np.ndarray] = None,
                  window: float = 0.1, max_rings: Optional[int] = None, maps=None, **options) -> ControlPoints:
    """ring_points around every expected ring the detector sees (`maps`: precomputed angle_maps)."""
    image = np.asarray(image, dtype=float)
    two_theta, chi = maps if maps is not None else angle_maps(geometry, image.shape)
    good = usable_pixels(image, mask)
    seen = two_theta[good]
    lo, hi = float(seen.min()) + window, float(seen.max()) - window
    points = ControlPoints()
    for k, ring in enumerate(calibrant_rings(calibrant, geometry.wavelength)):
        if max_rings is not None and k >= max_rings:
            break
        if lo < ring < hi:
            found = ring_points(image, good, two_theta, chi, ring, window, **options)
            if len(found):
                points.add(found, k)
    return points


def ring_overlay(two_theta: np.ndarray, rings: np.ndarray, width: Optional[float] = None) -> np.ndarray:
    """int array: k+1 on the pixels within half a pixel of ring k, 0 elsewhere (to draw the expected rings)."""
    if width is None:
        gradient = np.gradient(two_theta)
        width = 0.6 * float(np.median(np.hypot(gradient[0], gradient[1])))
    overlay = np.zeros(two_theta.shape, dtype=np.int32)
    for k, ring in enumerate(rings):
        overlay[np.abs(two_theta - ring) < width] = k + 1
    return overlay
