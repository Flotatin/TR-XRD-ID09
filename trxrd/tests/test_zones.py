"""Zones: fit zone of the spectra, excluded zones of the run, zone of spectra of the dDAC plots."""

import copy

import numpy as np
import pytest

from synthetic import GASKET
from trxrd.gui.presenters import display
from trxrd.gui.presenters.viewer import (
    DisplaySettings, EvolutionPresenter, MapPresenter, NavigatorPresenter, RatePresenter, RateSettings,
)
from trxrd.gui.presenters.zones import CHOOSE_A_ZONE, TOO_NARROW, SpectraZonePresenter, ZonesPresenter
from trxrd.services import analysis, editing
from trxrd.services.analysis import AnalysisSettings
from trxrd.services.session import Session

ICE_VII = "H2O_Ice_VII"


class Recorder:
    """Fake view: records every call, `last(name)` gives the arguments of the last one."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def method(*args):
            self.calls.append((name, args))
        return method

    def last(self, name):
        return next(args for n, args in reversed(self.calls) if n == name)


@pytest.fixture
def run(compression_run):
    return compression_run  # spectra 3, 4, 5: ice VII at 2.6, 3.1, 3.7 GPa; a gasket line in all of them


def excluded(run):
    return analysis.saved_settings(run).exclude


# ------------------------------------------------------------------ services


def test_overlapping_zones_become_one():
    assert analysis.merged_zones([(5.0, 4.0), (1.0, 2.0), (3.5, 4.5), (2.0, 2.5)]) == [(1.0, 2.5), (3.5, 5.0)]
    assert analysis.merged_zones([]) == []


def test_fit_range_stays_in_the_pattern(run):
    x = run.spectra[5].two_theta
    assert editing.set_fit_range(run, [4, 5], (x.max() + 5, 15.0)) == [4, 5]
    assert run.spectra[5].fit_range == (15.0, float(x.max())) and run.spectra[4].fit_range == (15.0, float(x.max()))
    editing.set_fit_range(run, [5], None)
    assert run.spectra[5].fit_range == (float(x.min()), float(x.max()))


def test_a_refused_fit_range_changes_no_spectrum(run):
    before = run.spectra[5].fit_range
    with pytest.raises(ValueError, match="trop étroite"):
        editing.set_fit_range(run, [5], (15.0, 15.01))
    with pytest.raises(ValueError, match="hors du spectre"):
        editing.set_fit_range(run, [5], (500.0, 600.0))
    assert run.spectra[5].fit_range == before


def test_peaks_left_out_of_the_fit_zone_no_longer_count(run):
    """Their old positions would keep weighing on the cell, and on P, without being fitted."""
    editing.add_phase(run, 5, ICE_VII, 3.7)
    editing.fit_current(run, 5)
    peaks = run.spectra[5].phases[0].peaks
    assert len(peaks) >= 2
    kept = peaks[0]
    editing.set_fit_range(run, [5], (kept.center - 0.5, kept.center + 0.5))
    assert editing.peaks_outside(run.spectra[5]) == len(peaks) - 1

    pressures = []
    for shift in (0.0, 0.3):
        work = copy.deepcopy(run)
        spectrum = work.spectra[5]
        away = spectrum.phases[0].peaks[1]
        away.center += shift
        position = away.center
        editing.fit_current(work, 5)
        fit = spectrum.phases[0]
        assert fit.peaks[1].center == position  # not fitted
        assert abs(spectrum.model[np.argmin(abs(spectrum.two_theta - position))]) < 50.0  # not drawn
        pressures.append(fit.pressure)
    assert pressures[0] == pressures[1] and abs(pressures[0] - 3.7) < 0.1  # from the line inside the zone alone


def test_moving_a_phase_does_not_widen_the_fit_zone(run):
    zone = (15.0, 17.0)
    editing.set_fit_range(run, [5], zone)
    fit = editing.add_phase(run, 5, ICE_VII, 3.7)
    assert run.spectra[5].fit_range == zone and fit.peaks
    assert all(zone[0] <= p.center <= zone[1] for p in fit.peaks)
    editing.add_phase(run, 5, ICE_VII, 3.5)  # the wheel of the pressure
    assert run.spectra[5].fit_range == zone


def test_a_peak_placed_by_hand_still_widens_the_fit_zone(run):
    editing.set_fit_range(run, [5], (15.0, 17.0))
    editing.add_peak(run, 5, GASKET)
    assert run.spectra[5].fit_range[1] > GASKET


def test_no_line_goes_to_a_peak_of_an_excluded_zone(run):
    first = editing.add_phase(run, 5, ICE_VII, 3.7).peaks
    assert len(first) >= 2
    hidden = first[0]
    settings = AnalysisSettings(exclude=[(hidden.center - 0.1, hidden.center + 0.1)])
    again = editing.add_phase(run, 5, ICE_VII, 3.7, settings).peaks
    assert [p.hkl for p in again] == [p.hkl for p in first[1:]]  # the others are placed as before


def test_the_settings_of_an_edit_can_be_undone(run):
    session = Session()
    session.attach(run, None)
    run.ui_state["other"] = 1
    session.remember([], "zone", state=True)
    run.ui_state["other"] = 2
    session.undo()
    assert run.ui_state["other"] == 1
    session.redo()
    assert run.ui_state["other"] == 2
    session.remember([5], "sans réglages")
    run.ui_state["other"] = 3
    session.undo()  # an edit of the spectra leaves the settings alone
    assert run.ui_state["other"] == 3


# ------------------------------------------------------------------ fit zone and excluded zones


@pytest.fixture
def zones(run):
    session, settings, view, spectrum = Session(), DisplaySettings(), Recorder(), Recorder()
    presenter = ZonesPresenter(session, settings, view, spectrum)
    session.attach(run, None)
    session.select(5)
    presenter.view, presenter.spectrum_view = view, spectrum
    return presenter


def test_zones_are_shown_for_the_current_spectrum(zones, run):
    choices = zones.view.last("show_zones")[0]
    x = run.spectra[5].two_theta
    assert choices.fit == choices.limits == (float(x.min()), float(x.max()))
    assert choices.count == len(run) and choices.excluded == [] and choices.chosen is None
    marks, edited, chosen = zones.spectrum_view.last("show_zones")
    assert marks.fit == marks.limits and marks.excluded == [] and not edited and chosen is None
    zones.set_editing(True)
    assert zones.spectrum_view.last("show_zones")[1] is True


def test_a_zone_drawn_is_excluded_for_the_whole_run_and_can_be_undone(zones, run):
    session = zones.session
    zones.drawn(GASKET + 0.1, GASKET - 0.1)  # drawn from right to left
    assert excluded(run) == [pytest.approx((GASKET - 0.1, GASKET + 0.1))] and session.dirty
    assert zones.chosen == 0 and session.undo_label == "zone exclue"
    session.select(2)
    assert zones.spectrum_view.last("show_zones")[0].excluded == [pytest.approx((GASKET - 0.1, GASKET + 0.1))]
    session.undo()
    assert excluded(run) == [] and zones.chosen is None
    assert zones.view.last("show_zones")[0].excluded == []
    session.redo()
    assert len(excluded(run)) == 1


def test_drawn_zones_join_when_they_overlap(zones, run):
    zones.drawn(18.0, 18.5)
    zones.drawn(18.3, 18.8)
    assert excluded(run) == [pytest.approx((18.0, 18.8))] and zones.chosen == 0


def test_a_slip_of_the_mouse_makes_no_zone(zones, run):
    undo = zones.session.undo_label
    zones.drawn(18.0, 18.01)
    assert excluded(run) == [] and zones.view.last("show_message") == (TOO_NARROW,)
    zones.drawn(500.0, 600.0)  # outside the pattern
    assert excluded(run) == [] and zones.session.undo_label == undo


def test_an_excluded_zone_is_moved_chosen_and_removed(zones, run):
    zones.drawn(18.0, 18.5)
    zones.drawn(20.0, 20.5)
    zones.moved(0, 18.1, 18.7)
    assert excluded(run)[0] == pytest.approx((18.1, 18.7)) and zones.chosen == 0
    zones.moved(0, 18.1, 18.11)  # refused: it goes back where it was
    assert excluded(run)[0] == pytest.approx((18.1, 18.7))
    zones.pick(None)
    zones.remove()
    assert zones.view.last("show_message") == (CHOOSE_A_ZONE,) and len(excluded(run)) == 2
    zones.pick(1)
    assert zones.spectrum_view.last("show_zones")[2] == 1
    zones.remove()
    assert excluded(run) == [pytest.approx((18.1, 18.7))] and zones.chosen is None


def test_the_fit_zone_drawn_belongs_to_the_spectrum(zones, run):
    zones.drawn(17.0, 12.0, fit_zone=True)
    assert run.spectra[5].fit_range == (12.0, 17.0) and run.spectra[4].fit_range is None
    assert excluded(run) == [] and zones.session.undo_label == "zone d'ajustement"
    zones.session.undo()
    assert run.spectra[5].fit_range is None
    zones.view.on_fit_zone(13.0, 16.0)
    assert run.spectra[5].fit_range == (13.0, 16.0)
    zones.view.on_whole()
    assert run.spectra[5].fit_range == display.pattern_limits(run.spectra[5])


def test_a_refused_fit_zone_leaves_nothing_to_undo(zones, run):
    zones.set_fit_zone((12.0, 17.0))
    undo = zones.session.undo_label
    count = len(zones.view.calls)
    zones.view.on_fit_zone(15.0, 15.01)
    assert zones.session.undo_label == undo and "trop étroite" in zones.view.last("show_message")[0]
    assert ("show_zones" in [name for name, _ in zones.view.calls[count:]]  # the field shows the zone that stays
            and zones.view.last("show_zones")[0].fit == (12.0, 17.0))


def test_the_fit_zone_goes_to_the_selection_then_to_all_and_to_the_analysis(zones, run):
    session = zones.session
    zones.set_fit_zone((12.0, 17.0))
    session.set_selection([3, 4, 5])
    zones.propagate("selection")
    assert [s.fit_range for s in run.spectra[3:]] == [(12.0, 17.0)] * 3 and run.spectra[0].fit_range is None
    assert analysis.saved_settings(run).window is None
    assert "3 spectre(s)" in zones.view.last("show_message")[0]
    zones.propagate("all")
    assert [s.fit_range for s in run.spectra] == [(12.0, 17.0)] * len(run)
    assert analysis.saved_settings(run).window == (12.0, 17.0)
    session.undo()
    assert analysis.saved_settings(run).window is None and run.spectra[0].fit_range is None
    assert run.spectra[3].fit_range == (12.0, 17.0)


def test_a_zone_given_to_all_survives_the_automatic_analysis(zones, run):
    """Otherwise the analysis would fit the whole pattern again, and the zone drawn would be lost."""
    zones.set_fit_zone((12.0, 17.0))
    zones.propagate("all")
    settings = analysis.saved_settings(run)
    result = analysis.analyse(run, range(len(run)), settings)
    result.apply_to(run)
    assert [s.fit_range for s in run.spectra] == [(12.0, 17.0)] * len(run)
    for spectrum in run.spectra:
        assert all(12.0 <= p.center <= 17.0 for fit in spectrum.phases for p in fit.peaks)
    assert any(spectrum.phases for spectrum in run.spectra)


def test_the_whole_pattern_given_to_all_frees_the_analysis(zones, run):
    zones.set_fit_zone((12.0, 17.0))
    zones.propagate("all")
    zones.set_fit_zone(None)
    zones.propagate("all")
    assert analysis.saved_settings(run).window is None


def test_zones_drawn_in_d_are_stored_in_two_theta(zones, run):
    zones.settings.set_unit("d_A")
    low, high = (float(display.to_unit(np.array(t), "d_A", run)) for t in (18.0, 19.0))
    assert low > high  # d decreases with 2θ
    marks = zones.spectrum_view.last("show_zones")[0]
    assert marks.limits[0] < marks.limits[1]
    zones.drawn(low, high)
    assert excluded(run) == [pytest.approx((18.0, 19.0))]
    marks = zones.spectrum_view.last("show_zones")[0]
    assert marks.excluded == [pytest.approx((high, low))]  # low end first, whatever the unit


def test_the_tab_says_what_the_zones_leave_behind(zones, run):
    editing.add_phase(run, 5, ICE_VII, 3.7)
    spectrum = run.spectra[5]
    first = spectrum.phases[0].peaks[0]
    zones.set_fit_zone((first.center - 0.5, first.center + 0.5))
    note = zones.view.last("show_zones")[0].note
    assert "hors de la zone d'ajustement" in note and "zone exclue" not in note
    zones.set_fit_zone(None)
    zones.drawn(first.center - 0.1, first.center + 0.1)
    note = zones.view.last("show_zones")[0].note
    assert "1 pic(s) de phases dans une zone exclue" in note and "hors de la zone" not in note


def test_no_run_no_zone():
    session, view, spectrum = Session(), Recorder(), Recorder()
    presenter = ZonesPresenter(session, DisplaySettings(), view, spectrum)
    presenter.refresh()
    assert spectrum.last("show_zones") == (None, False, None) and view.last("clear") == ()
    presenter.drawn(1.0, 2.0)
    presenter.moved(0, 1.0, 2.0)
    presenter.remove()
    presenter.propagate("all")
    presenter.set_fit_zone((1.0, 2.0))


# ------------------------------------------------------------------ zone of spectra


def test_spectra_of_a_region_and_region_of_spectra():
    x = [0.0, 1.0, 2.0, 4.0, 8.0]
    assert display.spectra_in(x, (3.0, 0.5)) == [1, 2]
    assert display.region_of(x, [1, 2]) == (0.5, 3.0)
    assert display.region_of(x, [0, 4]) == (-0.75, 8.75)  # beyond the ends: half the median step
    assert display.region_of(x, []) is None and display.region_of([], [0]) is None
    assert display.region_of([3.0], [0]) == (2.5, 3.5)
    for chosen in ([0], [1, 2, 3], [4], [0, 1, 2, 3, 4]):
        assert display.spectra_in(x, display.region_of(x, chosen)) == chosen  # these spectra, no other


@pytest.fixture
def ddac(run):
    session, rates = Session(), RateSettings()
    navigator, evolution, rate_view, map_view = Recorder(), Recorder(), Recorder(), Recorder()
    NavigatorPresenter(session, navigator)
    EvolutionPresenter(session, evolution, rates)
    RatePresenter(session, rate_view, rates)
    MapPresenter(session, DisplaySettings(), map_view, rates)
    SpectraZonePresenter(session, rates)
    session.attach(run, None)
    return session, rates, navigator, evolution, rate_view, map_view


def test_where_we_are():
    assert display.position_text(0, 0, []) == ""
    assert display.position_text(2, 6, []) == "Spectre 2 / 5"
    assert display.position_text(2, 6, [4]) == "Spectre 2 / 5 — sélection : spectre 4"
    assert display.position_text(2, 6, [3, 4, 5]) == "Spectre 2 / 5 — sélection : 3 à 5 (3 spectres)"


def test_the_zone_chooses_the_spectra_and_shows_on_the_three_plots(ddac):
    session, rates, navigator, evolution, rate_view, map_view = ddac
    evolution.on_interval(0.6, 3.4)  # ms; drawn or dragged on the plot above
    assert session.selection == [1, 2, 3] and rates.interval == (0.6, 3.4)
    assert navigator.last("set_current")[2].endswith("sélection : 1 à 3 (3 spectres)")
    assert map_view.last("show_interval") == ((0.6, 3.4),) and evolution.last("show_interval")[0] == (0.6, 3.4)
    assert rate_view.last("show_interval")[1] == (0.6, 3.4)
    map_view.on_interval(1.6, 4.4)  # the same zone from the map
    assert session.selection == [2, 3, 4] and evolution.last("show_interval")[0] == (1.6, 4.4)


def test_the_zone_stays_while_the_current_spectrum_moves(ddac):
    """Browsing outside the zone must not lose it, as in the v1."""
    session, rates, navigator, *_ = ddac
    rates.update(interval=(0.6, 3.4))
    session.select(5)
    assert session.selection == [1, 2, 3] and rates.interval == (0.6, 3.4)
    assert navigator.last("set_current")[:2] == (5, len(session.run))


def test_no_zone_no_selection(ddac):
    """Nothing shows a selection without a zone: left behind, it would act on spectra nobody has in mind."""
    session, rates, navigator, evolution, rate_view, _ = ddac
    rates.update(interval=(0.6, 3.4))
    assert session.selection == [1, 2, 3]
    rate_view.on_interval_toggle(False)
    assert session.selection == [] and navigator.last("set_current")[2] == "Spectre 0 / 5"


def test_changing_the_dpdt_settings_keeps_the_selection(ddac):
    session, rates, *_ = ddac
    rates.update(interval=(0.6, 3.4))
    rates.update(half_window=4)
    assert session.selection == [1, 2, 3]


def test_the_zone_starts_over_all_the_spectra(ddac):
    session, rates, navigator, evolution, rate_view, _ = ddac
    rate_view.on_interval_toggle(True)
    assert rates.interval == (-0.5, 5.5) and session.selection == list(range(len(session.run)))
    rate_view.on_interval_toggle(False)
    assert rates.interval is None and evolution.last("show_interval")[0] is None


def test_a_zone_with_no_spectrum_chooses_none(ddac):
    session, rates, *_ = ddac
    rates.update(interval=(1.2, 1.4))
    assert session.selection == [] and rates.interval == (1.2, 1.4)


def test_the_zone_works_without_times(run):
    for spectrum in run.spectra:
        spectrum.time = np.nan
    session, rates, evolution, rate_view = Session(), RateSettings(), Recorder(), Recorder()
    EvolutionPresenter(session, evolution, rates)
    RatePresenter(session, rate_view, rates)
    SpectraZonePresenter(session, rates)
    session.attach(run, None)
    evolution.on_interval(1.5, 3.5)  # in spectra
    assert session.selection == [2, 3]
    assert evolution.last("show_interval")[0] == (1.5, 3.5)
    assert rate_view.last("show_interval") == (None, (1.5, 3.5))  # no mean dP/dt, the zone is there


def test_a_new_run_has_no_zone(ddac, run):
    session, rates, navigator, evolution, rate_view, map_view = ddac
    rates.update(interval=(0.6, 3.4))
    session.attach(run, None)
    assert rates.interval is None and session.selection == []
    assert map_view.last("show_interval") == (None,)
