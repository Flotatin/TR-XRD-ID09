"""Presenters of the viewer (no Qt: testable with fake views).

Each presenter listens to the Session (and to the display settings), turns
the domain objects into display data (`display.py`) and hands them to a
passive view. A view reports the user's gestures through callbacks that its
presenter sets (`view.on_pick`, `view.on_select`); it never changes the data.

Views expected (duck typing):
- spectrum: show_spectrum(SpectrumCurves, reset: bool), show_detection(DetectionMarks or None), clear()
  (zones: see zones.py)
- evolution: show_evolution(EvolutionData), set_marker(x), clear(), show_interval(region, segments),
  on_pick(x), on_quantity(key, relative), on_interval(start, stop)
- rates: show_rates(RateData), show_interval(IntervalData or None, region), set_marker(x), clear(),
  on_pick(x), on_options(by, half_window), on_interval(start, stop), on_interval_toggle(on)
- map: show_map(TimeMapData), show_interval(region), set_marker(x), clear(), on_pick(x), on_interval(start, stop)
- cake: show_cake(CakeData), show_image(array), show_message(text), clear(), on_mode(mode)
- table: show_rows(columns, rows)
- navigator: set_current(index, count, text), on_select(index)
- list of the spectra: show_spectra(columns, rows), set_current(index, selection), on_select(index),
  on_selection(indices)
- window: set_title, set_status, set_units(available, current), set_busy(text or None),
  set_undo(undo label or None, redo label or None), show_message(text),
  ask_open_path() / ask_save_path(suggested) -> path or None, confirm_discard() -> bool,
  show_error(title, text), show_info(title, text), ask_new_run(browse) -> NewRunRequest or None,
  start_progress(text, on_cancel), set_progress(done, total), end_progress()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from trxrd.gui.presenters import display
from trxrd.services import rates, raw_data
from trxrd.services.images import FrameSource
from trxrd.services.run_builder import build_and_save
from trxrd.services.session import Event, Session

Task = Callable[[Callable[[int, int], bool]], object]  # task(progress) -> result


class ImmediateRunner:
    """Runs a task at once, in the caller's thread (tests, scripts). The GUI uses a thread pool."""

    def __init__(self) -> None:
        self.cancelled = False  # set it to stop the task at its next progress() (tests)

    def run(self, task: Task, on_done: Callable[[object], None], on_error: Callable[[str], None],
            on_progress: Optional[Callable[[int, int], None]] = None):
        def progress(done: int, total: int) -> bool:
            if on_progress is not None:
                on_progress(done, total)
            return not self.cancelled

        try:
            result = task(progress)
        except Exception as exc:  # reported to the user, like the threaded runner does
            on_error(f"{type(exc).__name__}: {exc}")
        else:
            on_done(result)
        return self


@dataclass
class NewRunRequest:
    """What the "new run" dialog returns: raw scans to integrate and how."""

    scans: List[Path]  # image files (scan_jf1m_0000.h5)
    folder: Path  # where the .trxrd.h5 files are written
    poni_file: str
    mask_file: Optional[str] = None
    two_theta_range: Tuple[float, float] = (0.0, 40.0)
    phase_paths: List[str] = field(default_factory=list)


class DisplaySettings:
    """GUI-only state shared by the presenters: the x axis unit, raw or filtered - background spectrum."""

    def __init__(self, unit: str = "2th_deg") -> None:
        self.unit = unit
        self.raw = False  # spectrum shown raw with its background (default: filtered - background)
        self._listeners: List[Callable[["DisplaySettings"], None]] = []

    def subscribe(self, listener: Callable[["DisplaySettings"], None]) -> None:
        self._listeners.append(listener)

    def _notify(self) -> None:
        for listener in list(self._listeners):
            listener(self)

    def set_unit(self, unit: str) -> None:
        if unit not in display.UNITS:
            raise ValueError(f"unknown unit {unit!r}")
        if unit != self.unit:
            self.unit = unit
            self._notify()

    def set_raw(self, raw: bool) -> None:
        if bool(raw) != self.raw:
            self.raw = bool(raw)
            self._notify()

    def unit_for(self, run) -> str:
        """The chosen unit, or 2θ when the run has no energy (no Q, no d)."""
        return self.unit if self.unit in display.available_units(run) else "2th_deg"


class RateSettings:
    """dP/dt settings shared by the dDAC plots: curves per sample or per phase, smoothing, time zone."""

    def __init__(self) -> None:
        self.by = "family"
        self.half_window = 2  # spectra on each side of the sliding straight line
        # zone of the dDAC plots (ms, or spectra without times): mean dP/dt, spectra chosen; None: off
        self.interval: Optional[Tuple[float, float]] = None
        self._listeners: List[Callable[["RateSettings"], None]] = []

    def subscribe(self, listener: Callable[["RateSettings"], None]) -> None:
        self._listeners.append(listener)

    def update(self, **changes) -> None:
        """Change `by`, `half_window` and/or `interval` (None turns it off); listeners hear of real changes."""
        unknown = set(changes) - {"by", "half_window", "interval"}
        if unknown:
            raise ValueError(f"unknown settings {sorted(unknown)}")
        if "by" in changes and changes["by"] not in rates.GROUPINGS:
            raise ValueError(f"unknown grouping {changes['by']!r}")
        if "half_window" in changes:
            changes["half_window"] = max(int(changes["half_window"]), 1)
        if changes.get("interval") is not None:
            changes["interval"] = tuple(sorted(float(t) for t in changes["interval"]))
        changed = {k: v for k, v in changes.items() if getattr(self, k) != v}
        for key, value in changed.items():
            setattr(self, key, value)
        if changed:
            for listener in list(self._listeners):
                listener(self)


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
        curves = display.spectrum_curves(run, self.session.index, self.settings.unit_for(run), self.settings.raw)
        self.view.show_spectrum(curves, self._reset)
        self._reset = False


class EvolutionPresenter:
    """P, V, a, c/a... of every phase versus time, with an oscilloscope channel (dDAC view of the v1)."""

    def __init__(self, session: Session, view, rate_settings: Optional[RateSettings] = None) -> None:
        self.session, self.view = session, view
        self.rates = rate_settings or RateSettings()
        self.quantity, self.relative = "P", False
        self._x = []
        self._shows_pressure = False  # the straight lines of the mean dP/dt only go over P itself
        view.on_pick = self.pick
        view.on_quantity = self.set_quantity
        view.on_interval = lambda start, stop: self.rates.update(interval=(start, stop))
        session.subscribe(Event.RUN_CHANGED, self.refresh)
        session.subscribe(Event.DATA_CHANGED, self.refresh)
        session.subscribe(Event.SPECTRUM_CHANGED, self.marker)
        self.rates.subscribe(self.interval)

    def set_quantity(self, quantity: str, relative: bool = False) -> None:
        """Show `quantity` (key of display.QUANTITIES), divided by the card value when `relative`."""
        if quantity not in display.QUANTITIES:
            raise ValueError(f"unknown quantity {quantity!r}")
        self.quantity, self.relative = quantity, bool(relative)
        self.refresh()

    def refresh(self, *_) -> None:
        run = self.session.run
        if run is None or not run.spectra:
            self._x = []
            self.view.clear()
            return
        data = display.evolution_data(run, self.quantity, self.relative)
        self._x = data.x
        self._shows_pressure = data.quantity == "P" and not data.relative
        self.view.show_evolution(data)
        self.marker()
        self.interval()

    def interval(self, *_) -> None:
        """The time zone, and the straight lines of the mean dP/dt when P itself is shown."""
        if self.session.run is None:
            return
        found = display.interval_data(self.session.run, self.rates.interval, self.rates.by, self.rates.half_window)
        segments = found.segments if found is not None and self._shows_pressure else []
        self.view.show_interval(self.rates.interval, segments)

    def marker(self, *_) -> None:
        if len(self._x):
            self.view.set_marker(float(self._x[self.session.index]))

    def pick(self, x: float) -> None:
        index = display.nearest_spectrum(self._x, x)
        if index is not None:
            self.session.select(index)


class RatePresenter:
    """dP/dt of every sample (or phase) versus time, and its mean over an interval chosen on the plots."""

    def __init__(self, session: Session, view, rate_settings: Optional[RateSettings] = None) -> None:
        self.session, self.view = session, view
        self.rates = rate_settings or RateSettings()
        self._x = []
        view.on_pick = self.pick
        view.on_options = lambda by, half_window: self.rates.update(by=by, half_window=half_window)
        view.on_interval = lambda start, stop: self.rates.update(interval=(start, stop))
        view.on_interval_toggle = self.toggle_interval
        session.subscribe(Event.RUN_CHANGED, self._new_run)
        session.subscribe(Event.DATA_CHANGED, self.refresh)
        session.subscribe(Event.SPECTRUM_CHANGED, self.marker)
        self.rates.subscribe(self._settings_changed)
        self._options = (self.rates.by, self.rates.half_window)

    def _new_run(self, *_) -> None:
        self.rates.update(interval=None)  # the times of the previous run mean nothing here
        self.refresh()

    def _settings_changed(self, settings: RateSettings) -> None:
        if (settings.by, settings.half_window) != self._options:
            self.refresh()
        else:
            self.interval()

    def toggle_interval(self, on: bool) -> None:
        """On: the zone starts over the selection, or over all the spectra; off: no zone, no spectrum chosen."""
        run = self.session.run
        if on and run is not None and len(run):
            chosen = self.session.selection if len(self.session.selection) > 1 else range(len(run))
            self.rates.update(interval=display.region_of(display.spectrum_axis(run)[0], chosen))
        else:
            self.rates.update(interval=None)

    def refresh(self, *_) -> None:
        run = self.session.run
        self._options = (self.rates.by, self.rates.half_window)
        if run is None or not run.spectra:
            self._x = []
            self.view.clear()
            return
        self._x = display.spectrum_axis(run)[0]
        self.view.show_rates(display.rate_data(run, self.rates.half_window, self.rates.by))
        self.marker()
        self.interval()

    def interval(self, *_) -> None:
        self.view.show_interval(
            display.interval_data(self.session.run, self.rates.interval, self.rates.by, self.rates.half_window),
            self.rates.interval)

    def marker(self, *_) -> None:
        if len(self._x):
            self.view.set_marker(float(self._x[self.session.index]))

    def pick(self, x: float) -> None:
        index = display.nearest_spectrum(self._x, x)
        if index is not None:
            self.session.select(index)


class MapPresenter:
    """Integrated patterns versus time (x, shared with the evolution plot) and 2θ / Q / d (y)."""

    def __init__(self, session: Session, settings: DisplaySettings, view,
                 rate_settings: Optional[RateSettings] = None) -> None:
        self.session, self.settings, self.view = session, settings, view
        self.rates = rate_settings or RateSettings()
        self._x = []
        view.on_pick = self.pick
        view.on_interval = lambda start, stop: self.rates.update(interval=(start, stop))
        session.subscribe(Event.RUN_CHANGED, self.refresh)
        session.subscribe(Event.DATA_CHANGED, self.refresh)
        session.subscribe(Event.SPECTRUM_CHANGED, self.marker)
        settings.subscribe(self.refresh)
        self.rates.subscribe(self.interval)

    def refresh(self, *_) -> None:
        run = self.session.run
        data = display.time_map(run, self.settings.unit_for(run)) if run is not None else None
        if data is None:
            self._x = []
            self.view.clear()
            self.interval()
            return
        self._x = display.spectrum_axis(run)[0]
        self.view.show_map(data)
        self.marker()
        self.interval()

    def interval(self, *_) -> None:
        self.view.show_interval(self.rates.interval)

    def marker(self, *_) -> None:
        if len(self._x):
            self.view.set_marker(float(self._x[self.session.index]))

    def pick(self, x: float) -> None:
        index = display.nearest_spectrum(self._x, x)
        if index is not None:
            self.session.select(index)


class CakePresenter:
    """Cake (2θ × azimuth) or raw detector image of the current spectrum, computed in the background.

    The newest request wins: when the user scrolls through the spectra, the
    results of the older requests are dropped.
    """

    MODES = ("cake", "image")

    def __init__(self, session: Session, settings: DisplaySettings, view, runner=None) -> None:
        self.session, self.settings, self.view = session, settings, view
        self.runner = runner or ImmediateRunner()
        self.mode = "cake"
        self.visible = True
        self._source = None
        self._ticket = 0
        view.on_mode = self.set_mode
        session.subscribe(Event.RUN_CHANGED, self._new_run)
        session.subscribe(Event.SPECTRUM_CHANGED, self.refresh)
        settings.subscribe(self.refresh)

    def _new_run(self, *_) -> None:
        self._source = None

    def set_mode(self, mode: str) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.refresh()

    def set_visible(self, visible: bool) -> None:
        self.visible = bool(visible)
        self.refresh()

    def refresh(self, *_) -> None:
        run = self.session.run
        if not self.visible:
            return
        if run is None or not run.spectra:
            self.view.clear()
            return
        index, unit, mode = self.session.index, self.settings.unit_for(run), self.mode
        near = [self.session.path] if self.session.path is not None else []
        self._ticket += 1
        ticket = self._ticket

        def task(progress):
            if self._source is None or self._source.run is not run:
                self._source = FrameSource(run, near)
            source = self._source
            if source.problem and not (mode == "image" and source.available):
                return "message", source.problem
            if mode == "image":
                return "image", source.frame(index)
            return "cake", display.cake_data(source.cake(index), unit, run)

        self.runner.run(task, lambda result: self._show(ticket, result),
                        lambda message: self._show(ticket, ("message", message)))

    def _show(self, ticket: int, result) -> None:
        if ticket != self._ticket:
            return  # an older spectrum
        kind, data = result
        if kind == "cake":
            self.view.show_cake(data)
        elif kind == "image":
            self.view.show_image(data)
        else:
            self.view.show_message(data)


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
    """Slider of the spectra, and where we are: the current spectrum and the spectra chosen by the zone."""

    def __init__(self, session: Session, view) -> None:
        self.session, self.view = session, view
        view.on_select = session.select
        for event in (Event.SPECTRUM_CHANGED, Event.SELECTION_CHANGED):  # a new run emits SPECTRUM_CHANGED too
            session.subscribe(event, self.refresh)

    def refresh(self, *_) -> None:
        run, index, selection = self.session.run, self.session.index, self.session.selection
        count = len(run) if run is not None else 0
        self.view.set_current(index, count, display.position_text(index, count, selection))


class SpectraListPresenter:
    """List of the spectra (tab "Spectres"): one row goes to a spectrum, several choose them.

    Moving the current spectrum (slider, keys, plots) never changes the
    selection. One row clears a selection, as in any list; a zone stays,
    since it takes several rows to move it.
    """

    def __init__(self, session: Session, view, rate_settings: Optional[RateSettings] = None) -> None:
        self.session, self.view = session, view
        self.rates = rate_settings or RateSettings()
        self._choosing = False
        view.on_select = session.select
        view.on_selection = self.chosen
        session.subscribe(Event.RUN_CHANGED, self.refresh)
        session.subscribe(Event.DATA_CHANGED, self.refresh)
        session.subscribe(Event.SPECTRUM_CHANGED, self.current)
        session.subscribe(Event.SELECTION_CHANGED, self._selection_moved)

    def chosen(self, rows: List[int]) -> None:
        self._choosing = True  # the list shows it already
        try:
            if len(rows) > 1:
                self.session.set_selection(rows)
            elif self.rates.interval is None:
                self.session.set_selection([])
            else:
                self.current()  # the zone stays, and the list shows it again
        finally:
            self._choosing = False

    def _selection_moved(self, *_) -> None:
        if not self._choosing:
            self.current()

    def refresh(self, *_) -> None:
        run = self.session.run
        self.view.show_spectra(display.SPECTRUM_COLUMNS, display.spectrum_rows(run) if run is not None else [])
        self.current()

    def current(self, *_) -> None:
        self.view.set_current(self.session.index, self.session.selection)


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
        self.window.set_undo(self.session.undo_label, self.session.redo_label)

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

    # ------------------------------------------------------------------ new runs from raw data

    def new_run(self) -> None:
        """Ask which raw scans to integrate, build and save them in the background, open the last one."""
        if not self.may_discard():
            return
        request = self.window.ask_new_run(raw_data)  # the dialog browses RAW_DATA with these functions
        if request is None or not request.scans:
            return
        scans = list(request.scans)

        def task(progress):
            paths = []
            for k, scan in enumerate(scans):
                def frames(done, total, k=k):  # one progress bar over every scan
                    return progress(int(1000 * (k + done / max(total, 1))), 1000 * len(scans))
                paths.append(build_and_save(scan, request.folder, request.poni_file, request.mask_file,
                                            two_theta_range=request.two_theta_range,
                                            phase_paths=request.phase_paths, progress=frames))
            return paths, Session.read(paths[-1])

        self.window.start_progress(f"Intégration de {len(scans)} scan(s)…", self.cancel)
        self._job = self.runner.run(task, self._built, self._build_failed, self.window.set_progress)

    def cancel(self) -> None:
        if getattr(self, "_job", None) is not None:
            self._job.cancelled = True

    def _built(self, result) -> None:
        paths, opened = result
        self.window.end_progress()
        self._job = None
        self.session.attach(*opened)
        if len(paths) > 1:
            self.window.show_info("Runs créés", "\n".join(str(p) for p in paths))

    def _build_failed(self, message: str) -> None:
        self.window.end_progress()
        self._job = None
        if message.startswith("Cancelled"):
            self.window.set_status("Création arrêtée : les runs déjà terminés sont enregistrés.")
        else:
            self.window.show_error("Création du run impossible", message)

    def undo(self) -> None:
        label = self.session.undo()
        if label:
            self.window.show_message(f"Annulé : {label}")

    def redo(self) -> None:
        label = self.session.redo()
        if label:
            self.window.show_message(f"Rétabli : {label}")

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
