"""Presenter of the raw-scan preview of the "new run" dialog (no Qt).

Before integrating, the user looks at the raw image (one frame, or the
maximum over the scan: every line seen at any time of the compression), its
cake and its pattern over the whole detector, and chooses the 2θ range: the
pixels outside it are shaded, and the azimuthal coverage of the rings tells
where the detector edge makes the pattern noisy.

View expected (duck typing):
show_preview(PreviewData), show_range(low, high, outside or None, text), show_message(text),
set_busy(text or None); callbacks on_frame(frame or None) and on_mode(mode) set by the presenter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from trxrd.gui.presenters import display
from trxrd.gui.presenters.viewer import ImmediateRunner
from trxrd.services.images import ScanPreview

NO_SCAN = "Choisir un scan : son image et son diagramme sur tout le détecteur s'afficheront ici."
CACHED_IMAGES = 4  # raw images kept for the current scan (maximum, last frames)


def _file(path) -> Optional[str]:
    return str(path) if path and Path(path).is_file() else None


class PreviewPresenter:
    MODES = ("image", "cake")

    def __init__(self, view, runner=None, two_theta_range: Tuple[float, float] = (8.0, 25.0)) -> None:
        self.view = view
        self.runner = runner or ImmediateRunner()
        self.scan: Optional[str] = None
        self.poni: Optional[str] = None
        self.mask: Optional[str] = None
        self.frame: Optional[int] = None  # None: maximum over the scan
        self.mode = "image"
        self.range = tuple(sorted(map(float, two_theta_range)))
        self.data: Optional[display.PreviewData] = None
        self.closed = False
        self._preview: Optional[ScanPreview] = None
        self._images = {}  # (scan, frame) -> raw image
        self._ticket = 0
        view.on_frame = self.set_frame
        view.on_mode = self.set_mode
        self.refresh()

    # ------------------------------------------------------------------ what the user chose

    def set_scan(self, image_file) -> None:
        scan = _file(image_file)
        if scan != self.scan:
            self.scan = scan
            self._images.clear()
            self.refresh()

    def set_calibration(self, poni_file, mask_file=None) -> None:
        """Files that do not exist (yet: a path being typed) count as none."""
        poni, mask = _file(poni_file), _file(mask_file)
        if (poni, mask) != (self.poni, self.mask):
            self.poni, self.mask = poni, mask
            self.refresh()

    def set_frame(self, frame: Optional[int]) -> None:
        frame = None if frame is None else max(int(frame), 0)
        if frame != self.frame:
            self.frame = frame
            self.refresh()

    def set_mode(self, mode: str) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unknown mode {mode!r}")
        if mode != self.mode:
            self.mode = mode
            self.refresh()

    def set_range(self, low: float, high: float) -> None:
        self.range = tuple(sorted((float(low), float(high))))
        self.show_range()

    def close(self) -> None:
        """The dialog is closed: results still on their way are dropped."""
        self.closed = True
        self._ticket += 1

    # ------------------------------------------------------------------ display

    def refresh(self) -> None:
        if self.closed:
            return
        self._ticket += 1
        if self.scan is None:
            self.data = None
            self.view.show_message(NO_SCAN)
            return
        ticket, key = self._ticket, (self.scan, self.frame)
        scan, poni, mask, frame, mode = self.scan, self.poni, self.mask, self.frame, self.mode
        preview, image = self._preview, self._images.get(key)

        def task(progress):
            def keep_going(done: int, total: int) -> bool:
                return ticket == self._ticket and progress(done, total)  # a newer request stops this one

            source = preview if preview is not None and preview.key == (scan, poni or "", mask or "") else \
                ScanPreview(scan, poni, mask)
            raw = image if image is not None else source.image(frame, keep_going)
            if raw is None:
                return None
            return source, raw, display.preview_data(source, raw, frame, mode)

        if image is not None:
            self.view.set_busy("Calcul de l'aperçu…")
        else:
            self.view.set_busy("Lecture de l'image…" if frame is not None else "Maximum des images…")
        self.runner.run(task, lambda result: self._show(ticket, key, result),
                        lambda message: self._failed(ticket, message),
                        lambda done, total: self._progress(ticket, done, total))

    def _progress(self, ticket: int, done: int, total: int) -> None:
        if ticket == self._ticket and not self.closed:
            self.view.set_busy(f"Maximum des images : {done} / {total}")

    def _show(self, ticket: int, key, result) -> None:
        if self.closed or ticket != self._ticket or result is None:
            return  # an older request, or the dialog is gone
        self._preview, raw, self.data = result
        if len(self._images) >= CACHED_IMAGES and key not in self._images:
            self._images.pop(next(iter(self._images)))
        self._images[key] = raw
        self.view.set_busy(None)
        self.view.show_preview(self.data)
        self.show_range()

    def _failed(self, ticket: int, message: str) -> None:
        if self.closed or ticket != self._ticket:
            return
        self.data = None
        self.view.set_busy(None)
        self.view.show_message(message)

    def show_range(self) -> None:
        if self.data is None or self.closed:
            return
        low, high = self.range
        outside = None
        if self.data.two_theta_map is not None and self.data.cake is None:  # on the raw image
            outside = display.outside_range(self.data.two_theta_map, low, high)
        self.view.show_range(low, high, outside, display.range_text(self.data, low, high))
