"""Zones from the GUI (no Qt): fit zone of the spectra, excluded zones of the run, zone of spectra of the dDAC plots.

The fit zone belongs to each spectrum, then goes to the selection or to all of
them: given to all, it is also the range of the automatic analysis. The
excluded zones belong to the run. Both are drawn on the spectrum, edited or
removed, and can be undone.

The zone of spectra is the time zone of the dDAC plots, as in the v1: it is
the interval of the mean dP/dt, and it chooses the spectra, like the list of
the spectra: the selection of the actions on several spectra and of the
automatic analysis.

View expected (duck typing): show_zones(ZoneChoices), show_message(text), clear(); callbacks on_edit(on),
on_fit_zone(low, high), on_whole(), on_propagate("selection" | "all"), on_row(row), on_remove().
Spectrum view: show_zones(ZoneMarks or None, editing, chosen); callbacks on_zone_drawn(start, stop, fit_zone),
on_fit_zone(start, stop), on_zone_moved(row, start, stop), on_zone_picked(row), all in the display unit.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from trxrd.gui.presenters import display
from trxrd.gui.presenters.editing import apply_edit
from trxrd.gui.presenters.viewer import DisplaySettings, RateSettings
from trxrd.services import analysis, editing
from trxrd.services.analysis import Zone
from trxrd.services.session import Event, Session

CHOOSE_A_ZONE = "Choisir d'abord une zone : la cliquer sur le spectre, ou dans le tableau."
TOO_NARROW = "Zone trop étroite, ou hors du spectre."


@dataclass
class ZoneChoices:
    """What the Zones tab shows for the current spectrum (2θ, deg)."""

    limits: Zone  # the whole pattern
    fit: Zone
    excluded: List[Zone] = field(default_factory=list)
    chosen: Optional[int] = None  # row of the excluded zone chosen
    selected: int = 0  # spectra chosen together
    count: int = 0
    note: str = ""


class ZonesPresenter:
    def __init__(self, session: Session, settings: DisplaySettings, view, spectrum_view) -> None:
        self.session, self.settings, self.view, self.spectrum_view = session, settings, view, spectrum_view
        self.editing = False
        self.chosen: Optional[int] = None
        view.on_edit = self.set_editing
        view.on_fit_zone = lambda low, high: self.set_fit_zone((low, high))
        view.on_whole = lambda: self.set_fit_zone(None)
        view.on_propagate = self.propagate
        view.on_row = self.pick
        view.on_remove = self.remove
        spectrum_view.on_zone_drawn = self.drawn
        spectrum_view.on_fit_zone = lambda start, stop: self.set_fit_zone(self._in_two_theta(start, stop))
        spectrum_view.on_zone_moved = self.moved
        spectrum_view.on_zone_picked = self.pick
        session.subscribe(Event.RUN_CHANGED, self._new_run)
        for event in (Event.SPECTRUM_CHANGED, Event.DATA_CHANGED, Event.SELECTION_CHANGED):
            session.subscribe(event, self.refresh)
        settings.subscribe(self.refresh)

    def _new_run(self, *_) -> None:
        self.chosen = None
        self.refresh()

    def set_editing(self, on: bool) -> None:
        self.editing = bool(on)
        self.chosen = None
        self.refresh()

    def refresh(self, *_) -> None:
        run, spectrum = self.session.run, self.session.spectrum
        if spectrum is None:
            self.view.clear()
            self.spectrum_view.show_zones(None, False, None)
            return
        zones = analysis.saved_settings(run).exclude
        if self.chosen is not None and self.chosen >= len(zones):
            self.chosen = None
        marks = display.zone_marks(run, self.session.index, zones, self.settings.unit_for(run))
        self.spectrum_view.show_zones(marks, self.editing, self.chosen)
        limits = display.pattern_limits(spectrum)
        self.view.show_zones(ZoneChoices(
            limits, spectrum.fit_range or limits, list(zones), self.chosen, len(self.session.selection), len(run),
            display.zone_note(editing.peaks_outside(spectrum), editing.peaks_excluded(spectrum, zones))))

    def _in_two_theta(self, start: float, stop: float) -> Zone:
        unit = self.settings.unit_for(self.session.run)
        return tuple(sorted(display.from_unit(value, unit, self.session.run) for value in (start, stop)))

    # ------------------------------------------------------------------ fit zone

    def set_fit_zone(self, zone: Optional[Zone]) -> None:
        """Fit zone of the current spectrum; None: the whole pattern."""
        if self.session.spectrum is None:
            return
        run, index = self.session.run, self.session.index
        if not apply_edit(self.session, self.view, [index], "zone d'ajustement",
                          lambda: editing.set_fit_range(run, [index], zone)):
            self.refresh()  # the fields and the lines go back to the zone that stays

    def propagate(self, scope: str) -> None:
        """The fit zone of the current spectrum on the others; on all of them, it is the range analysed too."""
        spectrum = self.session.spectrum
        if spectrum is None:
            return
        run, zone = self.session.run, spectrum.fit_range
        everywhere = scope == "all"
        indices = list(range(len(run))) if everywhere else list(self.session.selection) or [self.session.index]
        whole = zone is None or tuple(zone) == display.pattern_limits(spectrum)

        def action() -> None:
            editing.set_fit_range(run, indices, zone)
            if everywhere:
                self._save(window=None if whole else tuple(zone))

        if apply_edit(self.session, self.view, indices, f"zone d'ajustement de {len(indices)} spectre(s)", action,
                      state=everywhere):
            self.view.show_message(f"Zone donnée à {len(indices)} spectre(s)"
                                   + (", et à l'analyse automatique." if everywhere else "."))

    # ------------------------------------------------------------------ excluded zones

    def _save(self, **changes) -> None:
        run = self.session.run
        run.ui_state[analysis.UI_KEY] = dataclasses.replace(analysis.saved_settings(run), **changes).to_dict()

    def _set_excluded(self, label: str, zones: Sequence[Zone], chosen: Optional[Zone] = None) -> None:
        merged = analysis.merged_zones(zones)
        if apply_edit(self.session, self.view, [], label, lambda: self._save(exclude=merged), state=True):
            self.pick(self._row_of(merged, chosen))

    @staticmethod
    def _row_of(zones: Sequence[Zone], zone: Optional[Zone]) -> Optional[int]:
        """Row of the zone that holds `zone` now that it may have joined a neighbour."""
        if zone is None:
            return None
        middle = sum(zone) / 2
        return next((row for row, (low, high) in enumerate(zones) if low <= middle <= high), None)

    def _inside(self, zone: Zone) -> Optional[Zone]:
        """`zone` kept inside the pattern; None when too little is left of it."""
        limits = display.pattern_limits(self.session.spectrum)
        low, high = max(min(zone), limits[0]), min(max(zone), limits[1])
        if high - low < editing.MIN_ZONE:
            self.view.show_message(TOO_NARROW)
            return None
        return low, high

    def drawn(self, start: float, stop: float, fit_zone: bool = False) -> None:
        """A zone drawn on the spectrum: the fit zone, or a new excluded zone."""
        if self.session.spectrum is None:
            return
        zone = self._in_two_theta(start, stop)
        if fit_zone:
            self.set_fit_zone(zone)
            return
        zone = self._inside(zone)
        if zone is not None:
            self._set_excluded("zone exclue", [*analysis.saved_settings(self.session.run).exclude, zone], zone)

    def moved(self, row: int, start: float, stop: float) -> None:
        if self.session.spectrum is None:
            return
        zones = list(analysis.saved_settings(self.session.run).exclude)
        zone = self._inside(self._in_two_theta(start, stop))
        if zone is None:
            self.refresh()  # the zone goes back where it was
        elif 0 <= row < len(zones):
            zones[row] = zone
            self._set_excluded("zone exclue déplacée", zones, zone)

    def pick(self, row: Optional[int]) -> None:
        self.chosen = row
        self.refresh()

    def remove(self) -> None:
        if self.session.spectrum is None:
            return
        zones = list(analysis.saved_settings(self.session.run).exclude)
        if self.chosen is None or self.chosen >= len(zones):
            self.view.show_message(CHOOSE_A_ZONE)
            return
        del zones[self.chosen]
        self._set_excluded("zone exclue retirée", zones)


class SpectraZonePresenter:
    """The zone of the dDAC plots chooses the spectra, and spectra chosen in the list move the zone.

    The zone removed, the selection goes with it. Spectra apart stay as they
    are chosen: the zone only spans them.
    """

    def __init__(self, session: Session, rates: RateSettings) -> None:
        self.session, self.rates = session, rates
        self._region = rates.interval
        self._following = False  # one follows the other: no echo
        rates.subscribe(self._zone_moved)
        session.subscribe(Event.SELECTION_CHANGED, self._selection_moved)

    def _x(self):
        return display.spectrum_axis(self.session.run)[0]

    def _zone_moved(self, *_) -> None:
        region = self.rates.interval
        if region == self._region:
            return  # the settings of dP/dt changed, not the zone
        self._region = region
        if self._following or self.session.run is None:
            return
        self._following = True
        try:
            self.session.set_selection([] if region is None else display.spectra_in(self._x(), region))
        finally:
            self._following = False

    def _selection_moved(self, *_) -> None:
        if self._following or self.rates.interval is None or self.session.run is None or not self.session.selection:
            return
        self._following = True
        try:
            self.rates.update(interval=display.region_of(self._x(), self.session.selection))
        finally:
            self._following = False
