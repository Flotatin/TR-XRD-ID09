"""Manual work on the phases of the spectra (services.editing), free peaks and undo."""

import numpy as np
import pytest

from synthetic import GASKET
from trxrd.domain import Processing
from trxrd.services import analysis, editing
from trxrd.services.analysis import AnalysisSettings
from trxrd.services.session import Session

ICE_VII = "H2O_Ice_VII"
ICE_VI = "H2O_Ice_VI"


@pytest.fixture
def run(compression_run):
    return compression_run  # spectra 3, 4, 5: ice VII at 2.6, 3.1, 3.7 GPa; a gasket line in all of them


def test_a_phase_is_placed_on_the_detected_peaks_then_fitted(run):
    fit = editing.add_phase(run, 5, ICE_VII, 3.6)
    spectrum = run.spectra[5]
    assert spectrum.phases == [fit] and fit.pressure == 3.6 and not np.isfinite(fit.pressure_err)
    expected = {r.hkl: t for t, r in run.phases[ICE_VII].two_theta_lines(3.6, run.calibration.energy_ev)}
    assert {p.hkl for p in fit.peaks} <= set(expected) and len(fit.peaks) >= 2
    for peak in fit.peaks:
        assert abs(peak.center - expected[peak.hkl]) < 0.1 and peak.height > 0 and peak.width > 0
    assert spectrum.model is not None and spectrum.model.max() > 0  # curve of the peaks, before any fit

    editing.fit_current(run, 5)
    assert spectrum.phases[0].pressure == pytest.approx(3.7, abs=0.03)
    assert np.isfinite(spectrum.phases[0].pressure_err)

    moved = editing.add_phase(run, 5, ICE_VII, 3.65)  # the pressure of the phase changed by hand
    assert len(spectrum.phases) == 1 and spectrum.phases[0] is moved and moved.pressure == 3.65
    editing.remove_phase(run, 5, ICE_VII)
    assert spectrum.phases == [] and not spectrum.model.any()
    with pytest.raises(ValueError, match="Aucune phase"):
        editing.fit_current(run, 5)


def test_lines_without_a_peak_and_peaks_of_other_phases(run):
    far = editing.add_phase(run, 5, ICE_VII, 8.0)  # nothing there: only the strong lines, at their expected place
    card = run.phases[ICE_VII]
    strong = [r.hkl for t, r in card.two_theta_lines(8.0, run.calibration.energy_ev)
              if r.intensity >= editing.STRONG_LINE and 8 <= t <= 25]
    assert sorted(p.hkl for p in far.peaks) == sorted(strong) and all(np.isnan(p.height) for p in far.peaks)

    editing.remove_phase(run, 5, ICE_VII)
    first = editing.add_phase(run, 5, ICE_VII, 3.7)
    taken = {round(p.center, 2) for p in first.peaks}
    second = editing.add_phase(run, 5, "H2O_Ice_Ic", 0.0)  # another phase: it never takes a peak already explained
    assert not taken & {round(p.center, 2) for p in second.peaks}


def test_a_series_follows_from_the_current_spectrum_outwards(run):
    editing.add_phase(run, 4, ICE_VII, 3.1)
    editing.fit_current(run, 4)
    steps = []
    series = editing.fit_series(run, 4, [3, 5], progress=lambda done, total: steps.append((done, total)) or True)
    assert steps == [(1, 3), (2, 3), (3, 3)] and series.indices == [3, 4, 5] and not series.stopped
    assert not run.spectra[3].phases and not run.spectra[5].phases  # run untouched...
    series.apply_to(run)  # ...until the result is put into it
    found = {i: run.spectra[i].phases[0].pressure for i in (3, 4, 5)}
    assert found == pytest.approx({3: 2.6, 4: 3.1, 5: 3.7}, abs=0.03)

    stopped = editing.fit_series(run, 4, [3, 5], progress=lambda done, total: done < 2)
    assert stopped.stopped and stopped.indices == [4, 5]

    alone = editing.fit_series(run, 4, [0, 3], follow=False)  # each from its own model; 0 has none
    assert alone.indices == [3, 4]
    editing.clear_model(run, [4])
    with pytest.raises(ValueError, match="rien à propager"):
        editing.fit_series(run, 4, [3])


def test_model_copied_onto_a_selection(run):
    editing.add_phase(run, 4, ICE_VII, 3.1)
    editing.fit_current(run, 4)
    model = editing.copy_model(run, 4)
    assert editing.paste_model(run, model, [3, 5]) == [3, 5]
    assert [f.phase for f in run.spectra[5].phases] == [ICE_VII]
    assert run.spectra[5].phases[0] is not run.spectra[3].phases[0]  # each spectrum its own copy
    assert run.spectra[5].phases[0].pressure == run.spectra[4].phases[0].pressure  # not fitted yet
    assert run.spectra[5].model is not None
    assert editing.clear_model(run, [3, 5]) == [3, 5]
    assert run.spectra[3].phases == [] and run.spectra[3].model is None


def test_free_peaks_are_kept_and_in_the_fitted_curve(run):
    zone = (GASKET - 0.1, GASKET + 0.1)
    result = analysis.analyse(run, range(len(run)), AnalysisSettings(pressure_range=(0.0, 10.0), exclude=[zone]))
    result.apply_to(run)
    spectrum = run.spectra[5]
    gasket = [p for p in spectrum.free_peaks if zone[0] <= p.center <= zone[1]]
    assert len(gasket) == 1 and gasket[0].height > 1000
    at = np.argmin(abs(spectrum.two_theta - GASKET))
    assert spectrum.model[at] > 0.5 * (spectrum.intensity - spectrum.baseline)[at]  # the curve goes up on it

    pressure = spectrum.phases[0].pressure
    editing.fit_current(run, 5)  # the free peaks stay
    assert len(spectrum.free_peaks) == len(result.spectra[5].free_peaks)
    assert spectrum.phases[0].pressure == pytest.approx(pressure, abs=0.01)


def test_undo_and_redo(run):
    session = Session()
    session.attach(run, None)
    assert session.undo() is None and session.redo() is None and session.undo_label is None
    session.remember([5], "phase ajoutée")
    editing.add_phase(run, 5, ICE_VII, 3.7)
    session.modified()
    assert session.undo_label == "phase ajoutée" and session.redo_label is None

    assert session.undo() == "phase ajoutée"
    assert run.spectra[5].phases == [] and session.redo_label == "phase ajoutée" and session.undo_label is None
    assert session.redo() == "phase ajoutée"
    assert [f.phase for f in run.spectra[5].phases] == [ICE_VII]
    session.undo()

    session.remember([5], "traitement", merge=True)  # a value changed step by step: undone at once
    run.spectra[5].processing = Processing(2, "svg", (11, 2))
    session.remember([5], "traitement", merge=True)
    run.spectra[5].processing = Processing(3, "svg", (13, 2))
    assert session.undo() == "traitement" and session.undo_label is None
    assert run.spectra[5].processing == Processing()

    session.remember([5], "a")
    assert session.redo_label is None  # a new edit forgets what was undone
    for k in range(40):
        session.remember([5], f"edit {k}")
    assert len(session._undo) == 30
    session.attach(run, None)
    assert session.undo_label is None  # another run
