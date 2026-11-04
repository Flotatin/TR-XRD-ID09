"""Automatic analysis of the current run from the GUI (no Qt): choices, background work, merge, static peaks.

The dialog shows the candidate phases of the run and the last settings of
this run (excluded zones included, kept in the run file). The analysis runs
in a worker thread on a copy of the spectra, with a progress bar and a stop
button; the spectra analysed are then put into the run. Static peaks that
are no line of the sample (gasket, diamond) are proposed as excluded zones,
and the analysis can be run again with them.

Window expected (duck typing), besides MainPresenter's:
ask_analysis(AnalysisChoices, library_phases) -> AnalysisRequest or None, ask_yes_no(title, text) -> bool,
show_message(text) (status bar, temporary).
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from trxrd.gui.presenters.viewer import ImmediateRunner
from trxrd.services import analysis
from trxrd.services.analysis import Analysis, AnalysisSettings, Zone
from trxrd.services.identification import StaticPeak
from trxrd.services.session import Session


@dataclass
class AnalysisChoices:
    """What the analysis dialog shows."""

    phases: List[Tuple[str, str, bool]]  # (name, label, checked): candidate phases of the run
    settings: AnalysisSettings  # last settings of this run
    count: int  # spectra of the run
    current: int  # current spectrum
    analysed: int  # spectra that already hold phases
    two_theta: Tuple[float, float]  # range of the patterns (deg)
    selection: List[int] = field(default_factory=list)  # spectra of the zone: what the analysis offers first


@dataclass
class AnalysisRequest:
    """What the dialog returns."""

    settings: AnalysisSettings
    indices: List[int]


def phase_label(name: str, p_range: Optional[Zone]) -> str:
    return f"{name}  ({p_range[0]:g}–{p_range[1]:g} GPa)" if p_range is not None else name


def parse_zones(text: str) -> List[Zone]:
    """'18.479:18.679; 20,1-20,3' -> [(18.479, 18.679), (20.1, 20.3)] (decimal comma or point)."""
    zones = []
    text = re.sub(r"\s*[:–-]\s*", ":", text.strip())
    for part in re.split(r"[;\s]+", text):
        if not part:
            continue
        try:
            low, high = sorted(float(b) for b in part.replace(",", ".").split(":"))
        except ValueError:
            raise ValueError(f"Zone « {part} » : écrire début:fin en degrés, par exemple 18.48:18.68") from None
        zones.append((low, high))
    return zones


def zones_text(zones: Sequence[Zone]) -> str:
    return "; ".join(f"{low:g}:{high:g}" for low, high in zones)


def summary_text(result: Analysis) -> str:
    searched = len(result.results) - result.tracked
    text = (f"Analyse automatique : {len(result.indices)} spectre(s) en {result.elapsed:.1f} s "
            f"({result.tracked} suivis, {searched} avec recherche)")
    if result.stopped:
        text += ", arrêtée par l'utilisateur"
    counts = result.phase_counts()
    found = ", ".join(f"{name} {count}" for name, count in counts.items())
    return f"{text} ; phases : {found}" if found else f"{text} ; aucune phase trouvée"


def static_text(static: Sequence[StaticPeak]) -> str:
    lines = [f"- {p.position:.3f}° (vu dans {p.fraction:.0%} des spectres, ±{p.spread:.3f}°) → zone "
             f"{p.zone()[0]:g}:{p.zone()[1]:g}" for p in static]
    return ("Ces pics ne bougent pas et ne sont des raies d'aucune phase trouvée : joint ou diamant ?\n"
            + "\n".join(lines) + "\n\nLes exclure et relancer l'analyse ?")


class AnalysisPresenter:
    def __init__(self, session: Session, window, runner=None) -> None:
        self.session, self.window = session, window
        self.runner = runner or ImmediateRunner()
        self.running = False
        self.last: Optional[Analysis] = None
        self._job = None

    def choices(self) -> AnalysisChoices:
        run = self.session.run
        settings = analysis.saved_settings(run)
        chosen = set(settings.phases) if settings.phases is not None else None
        phases = [(name, phase_label(name, card.p_range), chosen is None or name in chosen)
                  for name, card in run.phases.items()]
        lows = [float(s.two_theta[0]) for s in run.spectra if len(s.two_theta)]
        highs = [float(s.two_theta[-1]) for s in run.spectra if len(s.two_theta)]
        return AnalysisChoices(phases, settings, len(run), self.session.index,
                               sum(1 for s in run.spectra if s.phases),
                               (min(lows), max(highs)) if lows else (0.0, 0.0), list(self.session.selection))

    def start(self) -> None:
        """Action "Analyse automatique…": ask, confirm overwriting earlier analyses, run in the background."""
        run = self.session.run
        if run is None or not run.spectra or self.running:
            return
        if run.calibration is None:
            self.window.show_error("Analyse impossible", "Le run n'a pas de calibration (énergie inconnue).")
            return
        request = self.window.ask_analysis(self.choices(), analysis.library_phases)
        if request is None or not request.indices:
            return
        already = sum(1 for i in request.indices if run.spectra[i].phases)
        if already and not self.window.ask_yes_no(
                "Remplacer l'analyse ?",
                f"{already} des {len(request.indices)} spectres choisis ont déjà des phases : l'analyse "
                "automatique les remplace. Le fichier ne change qu'à l'enregistrement."):
            return
        self.launch(request.settings, request.indices)

    def launch(self, settings: AnalysisSettings, indices: Sequence[int]) -> None:
        run, indices = self.session.run, list(indices)
        self.running = True
        self.window.start_progress(f"Analyse automatique de {len(indices)} spectre(s)…", self.cancel)
        self._job = self.runner.run(lambda progress: analysis.analyse(run, indices, settings, progress),
                                    self._done, self._failed, self.window.set_progress)

    def cancel(self) -> None:
        if self._job is not None:
            self._job.cancelled = True  # the spectra already analysed are kept

    def _done(self, result: Analysis) -> None:
        self.running, self._job = False, None
        self.window.end_progress()
        if self.session.run is not result.run:
            return  # that run was closed meanwhile
        self.session.remember(result.indices, "analyse automatique")
        result.apply_to(self.session.run)
        self.last = result
        self.session.modified()
        summary = summary_text(result)
        self.window.show_message(summary)
        if result.static and self.window.ask_yes_no("Pics fixes", f"{summary}\n\n{static_text(result.static)}"):
            zones = [peak.zone() for peak in result.static]
            self.launch(dataclasses.replace(result.settings, exclude=list(result.settings.exclude) + zones),
                        result.indices)

    def _failed(self, message: str) -> None:
        self.running, self._job = False, None
        self.window.end_progress()
        self.window.show_error("Analyse impossible", message)
