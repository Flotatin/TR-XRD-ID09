"""Presenter of the calibration dialog (no Qt): image, calibrant, start geometry, clicked rings, refinement.

View expected (duck typing):
show_image(image), show_rings(overlay int array), show_points(points (N, 2), rings (N,)),
show_ring_choices(labels), show_geometry(text), show_start(distance_mm, center_col, center_row, energy_kev),
show_message(text), set_busy(text or None), ask_save_path(suggested) -> path or None,
on_click(row, col) set by the presenter.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from trxrd.gui.presenters.viewer import ImmediateRunner
from trxrd.services.calibration import (
    CALIBRANTS, CalibrationState, Geometry, load_calibration_image, load_mask, ring_overlay, save_poni,
)

DEFAULT_ENERGY_KEV = 19.0
HC_KEV_M = 1.23984198e-9
DEFAULT_FIX = ("wavelength", "rot3")


class CalibrationPresenter:
    def __init__(self, view, runner=None, poni_file: Optional[str] = None, mask_file: Optional[str] = None) -> None:
        self.view = view
        self.runner = runner or ImmediateRunner()
        self.state: Optional[CalibrationState] = None
        self.image_file = ""
        self.mask_file = mask_file or ""
        self.saved_poni: Optional[Path] = None
        self.ring = 0
        self.fix = list(DEFAULT_FIX)
        self._geometry: Optional[Geometry] = None  # start geometry, until an image is loaded
        self._mask = None
        self._calibrant = CALIBRANTS[0]
        view.on_click = self.click
        if poni_file:
            self.load_poni(poni_file)
        if mask_file:
            self.load_mask(mask_file)

    # ------------------------------------------------------------------ state

    @property
    def geometry(self) -> Optional[Geometry]:
        return self.state.geometry if self.state is not None else self._geometry

    def _set_geometry(self, geometry: Geometry) -> None:
        if self.state is not None:
            self.state.geometry = geometry
        self._geometry = geometry
        self._show_start()
        self._redraw()

    def _show_start(self) -> None:
        g = self.geometry
        if g is not None:
            col, row = g.beam_center()
            self.view.show_start(g.dist * 1e3, col, row, HC_KEV_M / g.wavelength)

    def _run(self, text: str, task, done) -> None:
        self.view.set_busy(text)

        def finished(result) -> None:
            self.view.set_busy(None)
            done(result)

        def failed(message: str) -> None:
            self.view.set_busy(None)
            self.view.show_message(message)

        self.runner.run(lambda progress: task(), finished, failed)

    def _redraw(self) -> None:
        """Expected rings of the current geometry over the image (2θ maps: in the background)."""
        if self.state is None:
            return
        state = self.state

        def task():
            rings = state.rings()
            return rings, ring_overlay(state.maps()[0], rings)

        def done(result) -> None:
            rings, overlay = result
            self.view.show_rings(overlay)
            self.view.show_ring_choices([f"{k} : {t:.3f}°" for k, t in enumerate(rings[:30])])
            self.view.show_points(state.points.points, state.points.rings)

        self._run("Calcul des anneaux…", task, done)

    # ------------------------------------------------------------------ inputs

    def load_image(self, path: str, frames: Optional[Sequence[int]] = None) -> None:
        def done(image) -> None:
            self.image_file = str(path)
            geometry = self._geometry or Geometry.from_guess(0.160, image.shape[1] / 2, image.shape[0] / 2,
                                                             HC_KEV_M / DEFAULT_ENERGY_KEV)
            self.state = CalibrationState(image, geometry, self._calibrant, self._mask)
            self._show_image()
            self._show_start()
            self.view.show_message(f"{Path(path).name} : {image.shape[1]} × {image.shape[0]} pixels. "
                                   "Choisir le numéro d'un anneau, puis cliquer dessus dans l'image.")
            self._redraw()

        self._run(f"Lecture de {Path(path).name}…", lambda: load_calibration_image(path, frames), done)

    def _show_image(self) -> None:
        image, mask = self.state.image, self.state.mask
        if mask is not None and mask.shape == image.shape:
            image = np.where(mask > 0, np.nan, image)  # masked pixels drawn dark
        self.view.show_image(image)

    def load_poni(self, path: str) -> None:
        try:
            geometry = Geometry.from_poni_text(Path(path).read_text(encoding="utf-8", errors="replace"))
        except Exception as exc:  # unreadable file: say so, keep the current geometry
            self.view.show_message(f"Fichier .poni illisible : {type(exc).__name__}: {exc}")
            return
        self._set_geometry(geometry)
        self.view.show_message(f"Géométrie de départ : {Path(path).name}")

    def load_mask(self, path: str) -> None:
        try:
            self._mask = load_mask(path)
        except Exception as exc:
            self.view.show_message(f"Masque illisible : {type(exc).__name__}: {exc}")
            return
        self.mask_file = str(path)
        if self.state is not None:
            self.state.mask = self._mask
            self._show_image()

    def set_guess(self, distance_mm: float, center_col: float, center_row: float, energy_kev: float) -> None:
        """Start geometry without tilt (the beam centre may lie off the detector)."""
        template = self.geometry or Geometry(0.16, 0.0, 0.0)
        self._set_geometry(Geometry.from_guess(distance_mm * 1e-3, center_col, center_row, HC_KEV_M / energy_kev,
                                               template.detector, template.detector_config))

    def set_energy(self, energy_kev: float) -> None:
        if self.geometry is not None:
            self._set_geometry(dataclasses.replace(self.geometry, wavelength=HC_KEV_M / energy_kev))

    def set_calibrant(self, name: str) -> None:
        if name not in CALIBRANTS:
            raise ValueError(f"unknown calibrant {name!r}")
        self._calibrant = name
        if self.state is not None:
            self.state.calibrant = name
            self.state.points.points, self.state.points.rings = np.zeros((0, 2)), np.zeros(0, dtype=int)
        self._redraw()

    def set_ring(self, ring: int) -> None:
        self.ring = int(ring)

    def set_fixed(self, parameters: Sequence[str]) -> None:
        self.fix = list(parameters)

    # ------------------------------------------------------------------ actions

    def click(self, row: float, col: float) -> None:
        if self.state is None:
            self.view.show_message("Charger d'abord l'image du calibrant.")
            return
        state, ring = self.state, self.ring

        def done(count: int) -> None:
            self.view.show_points(state.points.points, state.points.rings)
            counts = ", ".join(f"anneau {k} : {n}" for k, n in state.points.ring_counts().items())
            self.view.show_message(f"{count} points trouvés sur l'anneau {ring} ({counts})" if count else
                                   f"Aucun anneau trouvé là (anneau {ring}) : cliquer plus près d'un anneau visible.")

        self._run("Recherche de l'anneau…", lambda: state.pick(row, col, ring), done)

    def clear_points(self) -> None:
        if self.state is not None:
            self.state.points.points, self.state.points.rings = np.zeros((0, 2)), np.zeros(0, dtype=int)
            self.view.show_points(self.state.points.points, self.state.points.rings)

    def refine(self) -> None:
        """Least squares on the current points only."""
        if self.state is None:
            return
        state, fix = self.state, list(self.fix)
        self._run("Affinement…", lambda: state.refine(fix), self._refined)

    def calibrate(self) -> None:
        """Clicked points (if any) -> refine -> extract every ring -> refine..., as Dioptas' "Calibrate"."""
        if self.state is None:
            return
        state = self.state
        self._run("Calibration…", state.calibrate, self._refined)

    def _refined(self, result) -> None:
        g = result.geometry
        col, row = g.beam_center()
        self.view.show_geometry(
            f"distance {g.dist * 1e3:.3f} mm\ncentre du faisceau x {col:.1f} px, y {row:.1f} px\n"
            f"rot1 {g.rot1 * 1e3:.2f} mrad, rot2 {g.rot2 * 1e3:.2f} mrad, rot3 {g.rot3 * 1e3:.2f} mrad\n"
            f"λ {g.wavelength * 1e10:.5f} Å ({HC_KEV_M / g.wavelength:.3f} keV)\n"
            f"écart quadratique {result.rms_deg:.4f}° sur {result.n_points} points, {result.n_rings} anneaux")
        self._show_start()
        self._redraw()

    def save(self, path: Optional[str] = None) -> Optional[Path]:
        if self.geometry is None:
            return None
        if path is None:
            suggested = str(Path(self.image_file).with_suffix(".poni")) if self.image_file else "calibration.poni"
            path = self.view.ask_save_path(suggested)
            if not path:
                return None
        self.saved_poni = save_poni(self.geometry, path)
        self.view.show_message(f"Enregistré : {self.saved_poni}")
        return self.saved_poni
