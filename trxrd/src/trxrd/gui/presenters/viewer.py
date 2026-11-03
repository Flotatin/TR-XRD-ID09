"""Presenters of the viewer (no Qt: testable with fake views).

Each presenter listens to the Session (and to the display settings), turns
the domain objects into display data (`display.py`) and hands them to a
passive view. A view reports the user's gestures through callbacks that its
presenter sets (`view.on_pick`, `view.on_select`); it never changes the data.

Views expected (duck typing):
- spectrum: show_spectrum(SpectrumCurves, reset: bool), clear()
- pressure: show_pressure(PressureData), set_marker(x), clear(), on_pick(x)
- map: show_map(MapData), set_marker(index), clear(), on_pick(index)
- table: show_rows(columns, rows)
- navigator: show_spectra(columns, rows), set_current(index, count), on_select(index)
- window: set_title, set_status, set_units(available, current), set_busy(text or None),
  ask_open_path() / ask_save_path(suggested) -> path or None, confirm_discard() -> bool,
  show_error(title, text)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

from trxrd.gui.presenters import display
from trxrd.services.session import Event, Session

Task = Callable[[Callable[[int, int], bool]], object]  # task(progress) -> result


class ImmediateRunner:
    """Runs a task at once, in the caller's thread (tests, scripts). The GUI uses a thread pool."""

    def run(self, task: Task, on_done: Callable[[object], None], on_error: Callable[[str], None]) -> None:
        try:
            result = task(lambda done, total: True)
        except Exception as exc:  # reported to the user, like the threaded runner does
            on_error(f"{type(exc).__name__}: {exc}")
        else:
            on_done(result)


class DisplaySettings:
    """GUI-only state shared by the presenters: the x axis unit."""

    def __init__(self, unit: str = "2th_deg") -> None:
        self.unit = unit
        self._listeners: List[Callable[["DisplaySettings"], None]] = []

    def subscribe(self, listener: Callable[["DisplaySettings"], None]) -> None:
        self._listeners.append(listener)

    def set_unit(self, unit: str) -> None:
        if unit not in display.UNITS:
            raise ValueError(f"unknown unit {unit!r}")
        if unit != self.unit:
            self.unit = unit
            for listener in list(self._listeners):
                listener(self)

    def unit_for(self, run) -> str:
        """The chosen unit, or 2θ when the run has no energy (no Q, no d)."""
        return self.unit if self.unit in display.available_units(run) else "2th_deg"


class SpectrumPresenter:
    def __init__(self, session: Session, settings: DisplaySettings, view) -> None:
        self.session, self.settings, self.view = session, settings, view
        self._reset = True  # new run or new unit: the view zooms out; otherwise it keeps its zoom
        session.subscribe(Event.RUN_CHANGED, self._new_axes)
        session.subscribe(Event.SPECTRUM_CHANGED, self.refresh)
        session.subscribe(Event.DATA_CHANGED, self.refresh)
        settings.subscribe(lambda s: (self._new_axes(), self.refresh()))

    def _new_axes(self, *_) -> None:
        self._reset = True

    def refresh(self, *_) -> None:
        run = self.session.run
        if run is None or not run.spectra:
            self.view.clear()
            return
        curves = display.spectrum_curves(run, self.session.index, self.settings.unit_for(run))
        self.view.show_spectrum(curves, self._reset)
        self._reset = False


class PressurePresenter:
    def __init__(self, session: Session, view) -> None:
        self.session, self.view = session, view
        self._x = []
        view.on_pick = self.pick
        session.subscribe(Event.RUN_CHANGED, self.refresh)
        session.subscribe(Event.DATA_CHANGED, self.refresh)
        session.subscribe(Event.SPECTRUM_CHANGED, self.marker)

    def refresh(self, *_) -> None:
        run = self.session.run
        if run is None or not run.spectra:
            self._x = []
            self.view.clear()
            return
        data = display.pressure_data(run)
        self._x = data.x
        self.view.show_pressure(data)
        self.marker()

    def marker(self, *_) -> None:
        if len(self._x):
            self.view.set_marker(float(self._x[self.session.index]))

    def pick(self, x: float) -> None:
        index = display.nearest_spectrum(self._x, x)
        if index is not None:
            self.session.select(index)


class MapPresenter:
    def __init__(self, session: Session, settings: DisplaySettings, view) -> None:
        self.session, self.settings, self.view = session, settings, view
        view.on_pick = self.pick
        session.subscribe(Event.RUN_CHANGED, self.refresh)
        session.subscribe(Event.DATA_CHANGED, self.refresh)
        session.subscribe(Event.SPECTRUM_CHANGED, self.marker)
        settings.subscribe(self.refresh)

    def refresh(self, *_) -> None:
        run = self.session.run
        data = display.intensity_map(run, self.settings.unit_for(run)) if run is not None else None
        if data is None:
            self.view.clear()
            return
        self.view.show_map(data)
        self.marker()

    def marker(self, *_) -> None:
        if self.session.run is not None and self.session.run.spectra:
            self.view.set_marker(self.session.index)

    def pick(self, index: int) -> None:
        self.session.select(index)


class TablesPresenter:
    """Phases and peaks of the current spectrum."""

    def __init__(self, session: Session, phases_view, peaks_view) -> None:
        self.session, self.phases_view, self.peaks_view = session, phases_view, peaks_view
        session.subscribe(Event.SPECTRUM_CHANGED, self.refresh)
        session.subscribe(Event.DATA_CHANGED, self.refresh)

    def refresh(self, *_) -> None:
        spectrum = self.session.spectrum
        self.phases_view.show_rows(display.PHASE_COLUMNS, display.phase_rows(spectrum))
        self.peaks_view.show_rows(display.PEAK_COLUMNS, display.peak_rows(self.session.run, spectrum))


class NavigatorPresenter:
    """List of the spectra and slider."""

    def __init__(self, session: Session, view) -> None:
        self.session, self.view = session, view
        view.on_select = session.select
        session.subscribe(Event.RUN_CHANGED, self.refresh)
        session.subscribe(Event.DATA_CHANGED, self.refresh)
        session.subscribe(Event.SPECTRUM_CHANGED, self.current)

    def refresh(self, *_) -> None:
        run = self.session.run
        self.view.show_spectra(display.SPECTRUM_COLUMNS, display.spectrum_rows(run) if run is not None else [])
        self.current()

    def current(self, *_) -> None:
        run = self.session.run
        self.view.set_current(self.session.index, len(run) if run is not None else 0)


class MainPresenter:
    """Window title and status, files (open in the background, save), navigation, unit."""

    FILE_FILTER = "Runs (*.trxrd.h5 *.h5 *.CEDX *.CEDXUp *.cedx);;Tous les fichiers (*)"

    def __init__(self, session: Session, settings: DisplaySettings, window, runner=None, version: str = "") -> None:
        self.session, self.settings, self.window = session, settings, window
        self.runner = runner or ImmediateRunner()
        self.version = version
        for event in (Event.RUN_CHANGED, Event.SPECTRUM_CHANGED, Event.DATA_CHANGED, Event.SAVED):
            session.subscribe(event, self.update)
        settings.subscribe(self.update)
        self.update()

    # ------------------------------------------------------------------ state shown by the window

    def update(self, *_) -> None:
        run = self.session.run
        title = f"TR-XRD v2 {self.version}".strip()
        if run is not None:
            title += f" — {run.name}{' *' if self.session.dirty else ''}"
        self.window.set_title(title)
        path = str(self.session.path) if self.session.path is not None else None
        self.window.set_status(display.summary_text(run, self.session.index, path, self.session.dirty))
        self.window.set_units(display.available_units(run), self.settings.unit_for(run))

    # ------------------------------------------------------------------ files

    def open(self, path: Optional[str] = None) -> None:
        if not self.may_discard():
            return
        path = path or self.window.ask_open_path()
        if not path:
            return
        self.window.set_busy(f"Ouverture de {Path(path).name}…")
        self.runner.run(lambda progress: Session.read(path), self._opened, self._failed("Ouverture impossible"))

    def _opened(self, result) -> None:
        self.window.set_busy(None)
        self.session.attach(*result)

    def _failed(self, title: str) -> Callable[[str], None]:
        def report(message: str) -> None:
            self.window.set_busy(None)
            self.window.show_error(title, message)
        return report

    def save(self) -> None:
        if self.session.run is None:
            return
        if self.session.path is None:
            self.save_as()
            return
        self._save(self.session.path)

    def save_as(self) -> None:
        if self.session.run is None:
            return
        suggested = str(self.session.path) if self.session.path else f"{self.session.run.name}.trxrd.h5"
        path = self.window.ask_save_path(suggested)
        if path:
            self._save(Path(path))

    def _save(self, path: Path) -> None:
        try:
            self.session.save(path)
        except Exception as exc:  # disk full, file locked...
            self.window.show_error("Enregistrement impossible", f"{type(exc).__name__}: {exc}")

    def close(self) -> None:
        if self.session.run is not None and self.may_discard():
            self.session.close()

    def may_discard(self) -> bool:
        """True when the current run may be replaced or closed (saved, or the user agrees to lose it)."""
        return self.session.run is None or not self.session.dirty or self.window.confirm_discard()

    # ------------------------------------------------------------------ navigation, display

    def step(self, delta: int) -> None:
        self.session.select(self.session.index + delta)

    def first(self) -> None:
        self.session.select(0)

    def last(self) -> None:
        if self.session.run is not None:
            self.session.select(len(self.session.run) - 1)

    def set_unit(self, unit: str) -> None:
        self.settings.set_unit(unit)
