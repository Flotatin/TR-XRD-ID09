"""Live mode from the GUI (no Qt): look at the raw data folder at regular times, open each new scan.

A new scan is built, analysed and saved at once, in a worker thread and in its
own file. Only the switch on screen waits: unsaved changes of the run shown,
or a dialog, are never swept away.

Window expected (duck typing): ask_live(LiveSettings) -> LiveSettings or None, start_live(period_ms), stop_live(),
set_live(text or None), busy() -> bool, show_error(title, text); callback on_live_tick().
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from trxrd.domain import Run
from trxrd.gui.presenters.viewer import ImmediateRunner
from trxrd.services import live
from trxrd.services.live import LiveSettings, Template, Watch
from trxrd.services.session import Session

WAITING = "en attente du scan suivant"


class LivePresenter:
    PERIOD = 2000  # ms between two looks at the folder

    def __init__(self, session: Session, window, runner=None) -> None:
        self.session, self.window = session, window
        self.runner = runner or ImmediateRunner()
        self.settings: Optional[LiveSettings] = None
        self.watch: Optional[Watch] = None
        self.template: Optional[Template] = None  # kept if the run is closed meanwhile
        self._looking = False
        self._building: Optional[Path] = None
        self._job = None
        self._ready: Optional[Tuple[Run, Path]] = None  # built, waiting for the run on screen to be free
        window.on_live_tick = self.tick

    @property
    def running(self) -> bool:
        return self.watch is not None

    def toggle(self, on: bool) -> bool:
        """Start or stop; True when it runs afterwards (the user may have cancelled)."""
        if on and not self.running:
            self.start()
        elif not on and self.running:
            self.stop()
        return self.running

    def start(self) -> None:
        run = self.session.run
        if run is None or not run.spectra:
            self.window.show_error("Mode Live", "Ouvrir d'abord un run : le mode Live reprend sa calibration, ses phases "
                                                "et ses réglages.")
            return
        if run.calibration is None or not run.calibration.poni:
            self.window.show_error("Mode Live", "Ce run n'a pas de calibration (.poni) : impossible d'intégrer les "
                                                "prochains scans.")
            return
        path = self.session.path
        near = [path] if path is not None else []
        settings = self.window.ask_live(LiveSettings(live.dataset_folder(run, near) or Path(""),
                                                     path.parent if path is not None else Path("")))
        if settings is None:
            return
        self.settings, self.template = settings, live.template_of(run)
        self.watch = Watch(settings.folder)
        self.window.start_live(self.PERIOD)
        self.window.set_live(f"Live : {WAITING} dans {settings.folder}")

    def stop(self) -> None:
        if self._job is not None:
            self._job.cancelled = True
        self.watch = self.settings = self._ready = None
        self.window.stop_live()
        self.window.set_live(None)

    # ------------------------------------------------------------------ at each look

    def tick(self) -> None:
        if self.watch is None:
            return
        self._deliver()
        if self._looking or self._building is not None:
            return
        self._looking = True
        watch = self.watch
        self.runner.run(lambda progress: watch.poll(), lambda scan: self._looked(watch, scan),
                        lambda message: self._look_failed(watch, message))

    def _looked(self, watch: Watch, scan: Optional[Path]) -> None:
        self._looking = False
        if watch is self.watch and scan is not None:
            self._build(watch, scan)

    def _look_failed(self, watch: Watch, message: str) -> None:
        self._looking = False
        if watch is self.watch:
            self.window.set_live(f"Live : dossier illisible ({message})")

    def _build(self, watch: Watch, scan: Path) -> None:
        run = self.session.run
        if run is not None and run.calibration is not None and run.calibration.poni:
            self.template = live.template_of(run)  # what the user has just changed counts
        settings, template, label = self.settings, self.template, scan.parent.name
        self._building = scan
        self.window.set_live(f"Live : {label}…")
        self._job = self.runner.run(lambda progress: live.process(scan, template, settings, progress),
                                    lambda result: self._built(watch, result),
                                    lambda message: self._failed(watch, scan, message),
                                    lambda done, total: self._progress(watch, label, done, total))

    def _progress(self, watch: Watch, label: str, done: int, total: int) -> None:
        if watch is self.watch:
            self.window.set_live(f"Live : {label}… {100 * done // max(total, 1)} %")

    def _built(self, watch: Watch, result: Tuple[Run, Path]) -> None:
        self._building = self._job = None
        if watch is self.watch:
            self._ready = result
            self._deliver()

    def _failed(self, watch: Watch, scan: Path, message: str) -> None:
        self._building = self._job = None
        if watch is not self.watch or message.startswith("Cancelled"):
            return
        if watch.failed(scan):
            self.window.set_live(f"Live : {scan.parent.name} pas encore lisible, nouvel essai…")
        else:
            self.window.set_live(f"Live : {scan.parent.name} ignoré ({message}) ; {WAITING}")

    def _deliver(self) -> None:
        """The run built goes on screen, as soon as this loses nothing."""
        if self._ready is None:
            return
        run, path = self._ready
        if self.session.dirty:
            self.window.set_live(f"Live : {run.name} enregistré. Il s'ouvrira dès que le run affiché sera enregistré.")
        elif self.window.busy():
            self.window.set_live(f"Live : {run.name} enregistré. Il s'ouvrira à la fin de l'opération en cours.")
        else:
            self._ready = None
            self.session.attach(run, path)
            self.window.set_live(f"Live : {run.name} ouvert à {datetime.now():%H:%M:%S}, {WAITING}")
