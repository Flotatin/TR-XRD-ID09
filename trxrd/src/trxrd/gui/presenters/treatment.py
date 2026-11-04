"""Presenters of the treatment of the spectra and of the peak detection (no Qt).

Treatment: the background and the filter of the current spectrum are
changed at once (case by case), then propagated to every spectrum or to the
selection (zone of spectra). Detection: the peaks the automatic analysis
detects in the current spectrum, with its settings (kept in the run), drawn
on the spectrum with the threshold, to check the detection before the
phase search.

Views expected (duck typing):
- processing: show_processing(Processing, selected, count), show_message(text), clear();
  callbacks on_change(Processing), on_propagate(scope: "all" | "selection")
- detection: show_options(AnalysisSettings), show_detection(columns, rows, summary), clear();
  callbacks on_options(**changes), on_visible(shown)
- spectrum: show_detection(DetectionMarks or None)
"""

from __future__ import annotations

import dataclasses

from trxrd.domain import Processing
from trxrd.gui.presenters import display
from trxrd.gui.presenters.viewer import DisplaySettings
from trxrd.services import analysis, treatment
from trxrd.services.session import Event, Session

DETECTION_OPTIONS = ("min_snr", "max_fwhm", "noise_window", "broad_fwhm", "shoulders", "spectrum_filter")


class ProcessingPresenter:
    def __init__(self, session: Session, view) -> None:
        self.session, self.view = session, view
        view.on_change = self.change
        view.on_propagate = self.propagate
        for event in (Event.RUN_CHANGED, Event.SPECTRUM_CHANGED, Event.DATA_CHANGED, Event.SELECTION_CHANGED):
            session.subscribe(event, self.refresh)

    def refresh(self, *_) -> None:
        run, spectrum = self.session.run, self.session.spectrum
        if spectrum is None:
            self.view.clear()
            return
        self.view.show_processing(spectrum.processing, len(self.session.selection), len(run))

    def change(self, processing: Processing) -> None:
        """New background or filter for the current spectrum."""
        if self.session.spectrum is None:
            return
        try:
            treatment.check(processing, self.session.spectrum)
        except ValueError as exc:
            self.view.show_message(str(exc))
            return
        self.session.remember([self.session.index], "traitement du spectre", merge=True)
        treatment.set_processing(self.session.run, [self.session.index], processing)
        self.view.show_message("")
        self.session.modified()

    def propagate(self, scope: str) -> None:
        """The treatment of the current spectrum to every spectrum ("all") or to the selection."""
        run, spectrum = self.session.run, self.session.spectrum
        if spectrum is None:
            return
        if scope not in ("all", "selection"):
            raise ValueError(f"unknown scope {scope!r}")
        indices = range(len(run)) if scope == "all" else self.session.selection
        if not indices:
            self.view.show_message(display.NO_SELECTION)
            return
        self.session.remember(indices, "traitement propagé")
        done = treatment.set_processing(run, indices, spectrum.processing)
        self.session.modified()
        self.view.show_message(f"Traitement du spectre {self.session.index} appliqué à {len(done)} spectre(s).")


class DetectionPresenter:
    def __init__(self, session: Session, settings: DisplaySettings, view, spectrum_view) -> None:
        self.session, self.settings, self.view, self.spectrum_view = session, settings, view, spectrum_view
        self.visible = True
        self.detection = None
        view.on_options = self.set_options
        view.on_visible = self.set_visible
        for event in (Event.RUN_CHANGED, Event.SPECTRUM_CHANGED, Event.DATA_CHANGED):
            session.subscribe(event, self.refresh)
        settings.subscribe(self.refresh)

    def set_visible(self, shown: bool) -> None:
        self.visible = bool(shown)
        self.refresh()

    def set_options(self, **changes) -> None:
        """Detection settings of the automatic analysis (kept in the run)."""
        unknown = set(changes) - set(DETECTION_OPTIONS)
        if unknown:
            raise ValueError(f"unknown detection options {sorted(unknown)}")
        run = self.session.run
        if run is None:
            return
        settings = analysis.saved_settings(run)
        new = dataclasses.replace(settings, **changes)
        if new != settings:
            run.ui_state[analysis.UI_KEY] = new.to_dict()
            self.session.modified()

    def refresh(self, *_) -> None:
        run = self.session.run
        if run is None or not run.spectra:
            self.detection = None
            self.view.clear()
            self.spectrum_view.show_detection(None)
            return
        settings = analysis.saved_settings(run)
        self.view.show_options(settings)
        self.detection = treatment.detection(run, self.session.index, settings)
        self.view.show_detection(display.DETECTION_COLUMNS, display.detection_rows(self.detection),
                                 display.detection_summary(self.detection))
        marks = None
        if self.visible:
            marks = display.detection_marks(run, self.detection, self.settings.unit_for(run), self.settings.raw)
        self.spectrum_view.show_detection(marks)
