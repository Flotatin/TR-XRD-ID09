"""Application state shared by the GUI presenters, independent of Qt.

The GUI never keeps its own copy of the data: views read the session and
react to its events; user actions call session/services methods.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable, DefaultDict, List, Optional, Union

from trxrd.domain import Calibration, Run, Spectrum
from trxrd.io import h5store, legacy
from trxrd.services import run_builder

PathLike = Union[str, Path]
Listener = Callable[["Session"], None]


class Event:
    RUN_CHANGED = "run_changed"  # another run loaded / created / closed
    SPECTRUM_CHANGED = "spectrum_changed"  # selection moved to another spectrum
    DATA_CHANGED = "data_changed"  # current run modified (fit, note...)
    SAVED = "saved"


class Session:
    def __init__(self) -> None:
        self.run: Optional[Run] = None
        self.path: Optional[Path] = None
        self.index = 0
        self.dirty = False
        self._listeners: DefaultDict[str, List[Listener]] = defaultdict(list)

    # ------------------------------------------------------------------ events

    def subscribe(self, event: str, listener: Listener) -> None:
        self._listeners[event].append(listener)

    def _emit(self, event: str) -> None:
        for listener in list(self._listeners[event]):
            listener(self)

    # ------------------------------------------------------------------ run lifecycle

    def open(self, path: PathLike) -> Run:
        """Open a run file; a legacy dill file is converted in memory and will be
        saved next to it as `<name>.trxrd.h5` (the old file is never overwritten)."""
        path = Path(path)
        if legacy.is_legacy_file(path):
            run = legacy.legacy_to_run(legacy.load_legacy(path), name=path.stem)
            self._set_run(run, path.with_name(f"{path.stem}.trxrd.h5"), dirty=True)
        else:
            self._set_run(h5store.load_run(path), path, dirty=False)
        return self.run

    def new_from_scan(self, scan_file: PathLike, calibration: Calibration, folder: PathLike, **options) -> Run:
        run = run_builder.build_run(scan_file, calibration, **options)
        self._set_run(run, run_builder.output_path(run, folder), dirty=True)
        return run

    def save(self, path: Optional[PathLike] = None) -> Path:
        if self.run is None:
            raise RuntimeError("No run to save")
        self.path = h5store.save_run(self.run, Path(path) if path else self.path)
        self.dirty = False
        self._emit(Event.SAVED)
        return self.path

    def close(self) -> None:
        self._set_run(None, None, dirty=False)

    def _set_run(self, run: Optional[Run], path: Optional[Path], dirty: bool) -> None:
        self.run, self.path, self.index, self.dirty = run, path, 0, dirty
        self._emit(Event.RUN_CHANGED)
        self._emit(Event.SPECTRUM_CHANGED)

    # ------------------------------------------------------------------ navigation / edition

    @property
    def spectrum(self) -> Optional[Spectrum]:
        if self.run is None or not self.run.spectra:
            return None
        return self.run.spectra[self.index]

    def select(self, index: int) -> None:
        if self.run is None or not self.run.spectra:
            return
        index = max(0, min(int(index), len(self.run.spectra) - 1))
        if index != self.index:
            self.index = index
            self._emit(Event.SPECTRUM_CHANGED)

    def modified(self) -> None:
        """To be called by services after they changed the current run."""
        self.dirty = True
        self._emit(Event.DATA_CHANGED)
