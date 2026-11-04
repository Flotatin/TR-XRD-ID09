"""Application state shared by the GUI presenters, independent of Qt.

The GUI never keeps its own copy of the data: views read the session and
react to its events; user actions call session/services methods.
"""

from __future__ import annotations

import copy
import dataclasses
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, DefaultDict, Dict, Iterable, List, Optional, Tuple, Union

from trxrd.domain import Calibration, Run, Spectrum
from trxrd.io import h5store, legacy
from trxrd.services import run_builder

PathLike = Union[str, Path]
Listener = Callable[["Session"], None]
MAX_UNDO = 30  # edits kept


@dataclass
class Snapshot:
    """State of some spectra before an edit, to undo it."""

    label: str  # what was done, for the menu
    spectra: Dict[int, Spectrum]
    ui_state: Optional[Dict[str, Any]] = None  # the settings of the run, for an edit that changes them


def _kept(spectrum: Spectrum) -> Spectrum:
    """Copy of what an edit changes. The arrays are shared: edits replace them, they never write into them."""
    return dataclasses.replace(spectrum, processing=dataclasses.replace(spectrum.processing),
                               phases=copy.deepcopy(spectrum.phases), free_peaks=copy.deepcopy(spectrum.free_peaks))


class Event:
    RUN_CHANGED = "run_changed"  # another run loaded / created / closed
    SPECTRUM_CHANGED = "spectrum_changed"  # selection moved to another spectrum
    SELECTION_CHANGED = "selection_changed"  # spectra chosen together (zone of spectra), for actions on several
    DATA_CHANGED = "data_changed"  # current run modified (fit, note...)
    SAVED = "saved"


class Session:
    def __init__(self) -> None:
        self.run: Optional[Run] = None
        self.path: Optional[Path] = None
        self.index = 0
        self.selection: List[int] = []  # spectra chosen together; the current one may be outside
        self.dirty = False
        self._undo: List[Snapshot] = []
        self._redo: List[Snapshot] = []
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
        self.attach(*self.read(path))
        return self.run

    @staticmethod
    def read(path: PathLike) -> Tuple[Run, Path, bool]:
        """Load a run file without touching the session (safe in a worker thread):
        the run, the path it will be saved to, and whether it differs from that file."""
        path = Path(path)
        if legacy.is_legacy_file(path):
            run = legacy.legacy_to_run(legacy.load_legacy(path), name=path.stem)
            return run, path.with_name(f"{path.stem}.trxrd.h5"), True
        return h5store.load_run(path), path, False

    def attach(self, run: Run, path: Optional[Path], dirty: bool = False) -> None:
        """Make `run` the current run (from the GUI thread: the listeners update the views)."""
        self._set_run(run, path, dirty)

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
        self.selection = []
        self._undo, self._redo = [], []
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

    def set_selection(self, indices) -> None:
        """Spectra chosen together (sorted, without duplicates, inside the run)."""
        count = len(self.run.spectra) if self.run is not None else 0
        selection = sorted({int(i) for i in indices if 0 <= int(i) < count})
        if selection != self.selection:
            self.selection = selection
            self._emit(Event.SELECTION_CHANGED)

    def modified(self) -> None:
        """To be called by services after they changed the current run."""
        self.dirty = True
        self._emit(Event.DATA_CHANGED)

    # ------------------------------------------------------------------ undo

    def remember(self, indices: Iterable[int], label: str, merge: bool = False, state: bool = False) -> bool:
        """Keep the spectra `indices` as they are, before an edit called `label`; False if nothing new was kept.

        `merge`: successive edits with this label on the same spectra (a value
        changed step by step) are undone together. `state`: the edit changes
        the settings kept in the run too (excluded zones).
        """
        indices = list(indices)
        self._redo = []
        if merge and self._undo and self._undo[-1].label == label and set(self._undo[-1].spectra) == set(indices):
            return False
        kept_state = copy.deepcopy(self.run.ui_state) if state else None
        self._undo.append(Snapshot(label, {i: _kept(self.run.spectra[i]) for i in indices}, kept_state))
        del self._undo[:-MAX_UNDO]
        return True

    def forget(self) -> None:
        """Drop the last kept state: the edit failed and changed nothing."""
        if self._undo:
            self._undo.pop()

    @property
    def undo_label(self) -> Optional[str]:
        return self._undo[-1].label if self._undo else None

    @property
    def redo_label(self) -> Optional[str]:
        return self._redo[-1].label if self._redo else None

    def _swap(self, snapshot: Snapshot) -> Snapshot:
        """Put `snapshot` into the run; returns what it replaced."""
        replaced = Snapshot(snapshot.label, {i: self.run.spectra[i] for i in snapshot.spectra})
        for index, spectrum in snapshot.spectra.items():
            self.run.spectra[index] = spectrum
        if snapshot.ui_state is not None:
            replaced.ui_state, self.run.ui_state = self.run.ui_state, snapshot.ui_state
        return replaced

    def undo(self) -> Optional[str]:
        """Undo the last edit; returns its label (None: nothing to undo)."""
        if not self._undo:
            return None
        snapshot = self._undo.pop()
        self._redo.append(self._swap(snapshot))
        self.modified()
        return snapshot.label

    def redo(self) -> Optional[str]:
        if not self._redo:
            return None
        snapshot = self._redo.pop()
        self._undo.append(self._swap(snapshot))
        self.modified()
        return snapshot.label
