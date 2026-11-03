"""Manual work on the phases from the GUI (no Qt): add, move, remove, fit, propagate, copy; every edit can be undone.

Short edits run at once; fitting a series runs in a worker thread on copies
of the spectra, with a progress bar and a stop button. The settings
(temperature, detection) are those of the automatic analysis, kept in the
run.

View expected (duck typing): show_model(ModelChoices), show_message(text), clear(); callbacks on_add(phase, pressure),
on_remove(phase), on_pressure(phase, pressure), on_fit(), on_fit_series(scope: "selection" | "all", follow), on_copy(),
on_paste(), on_clear().
Window: start_progress(text, on_cancel), set_progress(done, total), end_progress(), show_message(text),
show_error(title, text).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from trxrd.gui.presenters.analysis import phase_label
from trxrd.gui.presenters.viewer import ImmediateRunner
from trxrd.services import analysis, editing
from trxrd.services.editing import FitSeries
from trxrd.services.session import Event, Session


@dataclass
class ModelChoices:
    """What the Phases tab offers for the current spectrum."""

    cards: List[Tuple[str, str]]  # (name, label) of the phases of the run
    on_spectrum: Dict[str, float]  # pressure of the phases of the current spectrum, in their order
    selected: int  # spectra chosen together
    count: int
    copied_from: Optional[int]  # spectrum of the copied model


class EditingPresenter:
    def __init__(self, session: Session, view, window, runner=None) -> None:
        self.session, self.view, self.window = session, view, window
        self.runner = runner or ImmediateRunner()
        self.clipboard: Optional[editing.Model] = None
        self.running = False
        self._job = None
        view.on_add = self.add
        view.on_remove = self.remove
        view.on_pressure = self.move
        view.on_fit = self.fit
        view.on_fit_series = self.fit_series
        view.on_copy = self.copy
        view.on_paste = self.paste
        view.on_clear = self.clear
        session.subscribe(Event.RUN_CHANGED, self._new_run)
        for event in (Event.SPECTRUM_CHANGED, Event.DATA_CHANGED, Event.SELECTION_CHANGED):
            session.subscribe(event, self.refresh)

    def _new_run(self, *_) -> None:
        self.clipboard = None
        self.refresh()

    def refresh(self, *_) -> None:
        run, spectrum = self.session.run, self.session.spectrum
        if spectrum is None:
            self.view.clear()
            return
        cards = [(name, phase_label(name, card.p_range)) for name, card in run.phases.items()]
        pressures = {f.phase: float(f.pressure) for f in spectrum.phases if np.isfinite(f.pressure)}
        self.view.show_model(ModelChoices(cards, pressures, len(self.session.selection), len(run),
                                          self.clipboard.source if self.clipboard is not None else None))

    def _settings(self) -> analysis.AnalysisSettings:
        return analysis.saved_settings(self.session.run)

    def _edit(self, indices: Iterable[int], label: str, action: Callable[[], object], merge: bool = False) -> None:
        """Do `action` on the spectra `indices`, which can then be undone; its errors are shown, not raised."""
        indices = list(indices)
        self.session.remember(indices, label, merge)
        try:
            action()
        except (ValueError, KeyError) as exc:
            self.view.show_message(str(exc))
            return
        self.view.show_message("")
        self.session.modified()

    # ------------------------------------------------------------------ phases of the current spectrum

    def _ready(self) -> bool:
        if self.session.spectrum is None or self.running:
            return False
        if self.session.run.calibration is None:
            self.view.show_message("Le run n'a pas de calibration (énergie inconnue).")
            return False
        return True

    def add(self, phase: str, pressure: float) -> None:
        if self._ready():
            run, index, settings = self.session.run, self.session.index, self._settings()
            self._edit([index], f"ajout de {phase}", lambda: editing.add_phase(run, index, phase, pressure, settings))

    def move(self, phase: str, pressure: float) -> None:
        """New pressure of a phase of the spectrum: its lines follow, its peaks go to what is detected there."""
        if self._ready():
            run, index, settings = self.session.run, self.session.index, self._settings()
            self._edit([index], f"pression de {phase}",
                       lambda: editing.add_phase(run, index, phase, pressure, settings), merge=True)

    def remove(self, phase: str) -> None:
        if self._ready():
            run, index = self.session.run, self.session.index
            self._edit([index], f"retrait de {phase}", lambda: editing.remove_phase(run, index, phase))

    def fit(self) -> None:
        if not self._ready():
            return
        run, index = self.session.run, self.session.index
        if not self.session.spectrum.phases:
            self.view.show_message("Aucune phase sur ce spectre : en ajouter une avant d'ajuster.")
            return
        settings = self._settings()
        self._edit([index], "ajustement du spectre", lambda: editing.fit_current(run, index, settings))

    # ------------------------------------------------------------------ series

    def fit_series(self, scope: str = "selection", follow: bool = True) -> None:
        """Fit the selection or every spectrum in the background, from the current spectrum outwards."""
        if scope not in ("selection", "all"):
            raise ValueError(f"unknown scope {scope!r}")
        if not self._ready():
            return
        run, start, settings = self.session.run, self.session.index, self._settings()
        indices = list(self.session.selection) if scope == "selection" else list(range(len(run)))
        if not indices:
            self.view.show_message("Aucun spectre sélectionné : les choisir dans la liste (Maj + clic, Ctrl + clic).")
            return
        self.running = True
        self.window.start_progress(f"Ajustement de {len(indices)} spectre(s)…", self.cancel)
        self._job = self.runner.run(lambda progress: editing.fit_series(run, start, indices, settings, follow, progress),
                                    self._fitted, self._failed, self.window.set_progress)

    def cancel(self) -> None:
        if self._job is not None:
            self._job.cancelled = True  # the spectra already fitted are kept

    def _fitted(self, result: FitSeries) -> None:
        self.running, self._job = False, None
        self.window.end_progress()
        if self.session.run is not result.run:
            return  # that run was closed meanwhile
        self.session.remember(result.indices, f"ajustement de {len(result.indices)} spectre(s)")
        result.apply_to(self.session.run)
        self.session.modified()
        text = f"{len(result.indices)} spectre(s) ajusté(s) en {result.elapsed:.1f} s"
        self.window.show_message(text + (", arrêt demandé" if result.stopped else ""))

    def _failed(self, message: str) -> None:
        self.running, self._job = False, None
        self.window.end_progress()
        self.window.show_error("Ajustement impossible", message)

    # ------------------------------------------------------------------ model of a spectrum on others

    def copy(self) -> None:
        if self.session.spectrum is None:
            return
        if not self.session.spectrum.phases:
            self.view.show_message("Aucune phase à copier sur ce spectre.")
            return
        self.clipboard = editing.copy_model(self.session.run, self.session.index)
        self.view.show_message(f"Modèle du spectre {self.session.index} copié.")
        self.refresh()

    def _selection(self) -> List[int]:
        """The selection, or the current spectrum alone when nothing else is chosen."""
        return list(self.session.selection) or [self.session.index]

    def paste(self) -> None:
        if self.clipboard is None or self.session.spectrum is None:
            return
        run, indices, model = self.session.run, self._selection(), self.clipboard
        self._edit(indices, f"modèle collé sur {len(indices)} spectre(s)", lambda: editing.paste_model(run, model, indices))

    def clear(self) -> None:
        if self.session.spectrum is None:
            return
        run, indices = self.session.run, self._selection()
        self._edit(indices, f"phases retirées de {len(indices)} spectre(s)", lambda: editing.clear_model(run, indices))
