"""Manual work on the phases of the spectra (services.editing), free peaks and undo."""

import dataclasses

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
    far = editing.add_phase(run, 5, ICE_VII, 8.0)  # nothing detected near its lines: they get no peak
    assert far.peaks == [] and not editing.has_peaks(run.spectra[5])
    assert [line.used for line in editing.phase_lines(run, 5, ICE_VII)] == [False, False]  # but they are shown
    with pytest.raises(ValueError, match="Aucun pic"):
        editing.fit_current(run, 5)
    moved = editing.add_phase(run, 5, ICE_VII, 3.72)  # the wheel brings the lines onto the detected peaks
    assert [p.hkl for p in moved.peaks] == [(1, 1, 0), (2, 0, 0)]
    close = editing.add_phase(run, 5, ICE_VII, 4.3)  # (200) moves faster: only (110) is still explained
    assert [p.hkl for p in close.peaks] == [(1, 1, 0)]
    assert [line.used for line in editing.phase_lines(run, 5, ICE_VII)] == [True, False]

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


def test_single_peaks(run):
    editing.add_phase(run, 5, ICE_VII, 3.7)
    spectrum = run.spectra[5]
    fit = spectrum.phases[0]
    assert [p.hkl for p in fit.peaks] == [(1, 1, 0), (2, 0, 0)]
    line = fit.peaks[1].center
    editing.remove_peak(run, 5, (ICE_VII, 1))
    assert [p.hkl for p in fit.peaks] == [(1, 1, 0)]

    back = editing.add_peak(run, 5, line + 0.05, ICE_VII)  # a click near a line without a peak: it takes that line
    assert back == (ICE_VII, 1) and editing.get_peak(spectrum, back).hkl == (2, 0, 0)
    assert editing.get_peak(spectrum, back).center == pytest.approx(line, abs=0.01)  # on the detected peak
    free = editing.add_peak(run, 5, GASKET + 0.03, ICE_VII)  # no line there: a free peak
    peak = editing.get_peak(spectrum, free)
    assert free == (None, 0) and peak.hkl is None and peak.center == pytest.approx(GASKET, abs=0.02) and peak.height > 1000
    assert spectrum.model[np.argmin(abs(spectrum.two_theta - GASKET))] > 1000  # in the curve at once
    assert editing.peak_refs(spectrum) == [(ICE_VII, 0), (ICE_VII, 1), (None, 0)]

    empty = editing.add_peak(run, 5, 12.0)  # nothing detected: where the user clicked
    assert editing.get_peak(spectrum, empty).center == 12.0 and editing.get_peak(spectrum, empty).height >= 1.0
    editing.remove_peak(run, 5, empty)

    editing.move_peak(run, 5, back, line + 0.02, 500.0)
    moved = editing.get_peak(spectrum, back)
    assert moved.center == line + 0.02 and moved.height == 500.0 and np.isnan(moved.center_err)

    at_gasket = np.argmin(abs(spectrum.two_theta - GASKET))
    assert editing.toggle_peak(run, 5, free) is False
    assert spectrum.model[at_gasket] < 200  # out of the curve
    before = editing.get_peak(spectrum, free).height
    editing.fit_current(run, 5)
    assert editing.get_peak(spectrum, free).height == before and not editing.get_peak(spectrum, free).enabled  # and of the fit
    assert spectrum.phases[0].pressure == pytest.approx(3.7, abs=0.03)
    assert editing.toggle_peak(run, 5, free) is True

    with pytest.raises(ValueError, match="n'existe plus"):
        editing.remove_peak(run, 5, (None, 9))
    with pytest.raises(ValueError, match="pas sur ce spectre"):
        editing.move_peak(run, 5, (ICE_VI, 0), 10.0)


def test_a_peak_added_by_hand_is_covered_and_fitted_wherever_it_is(run):
    """An old analysis may have a narrow fit range, and the gasket line is far from every line of the phase."""
    editing.add_phase(run, 5, ICE_VII, 3.7)
    spectrum = run.spectra[5]
    spectrum.fit_range = (15.0, 17.0)  # the (200) line and the gasket are outside
    free = editing.add_peak(run, 5, GASKET)
    assert spectrum.fit_range[0] == 15.0 and spectrum.fit_range[1] > GASKET + 0.4
    at = np.argmin(abs(spectrum.two_theta - GASKET))
    assert spectrum.model[at] > 1000  # in the curve
    started = editing.get_peak(spectrum, free).height
    editing.fit_current(run, 5)
    fitted = editing.get_peak(spectrum, free)
    assert fitted.height != started and np.isfinite(fitted.height_err)  # fitted, far from the ice lines
    assert fitted.center == pytest.approx(GASKET, abs=0.03)

    only_free = run.spectra[3]  # without any phase, free peaks alone can be fitted
    editing.add_peak(run, 3, GASKET)
    editing.fit_current(run, 3)
    assert np.isfinite(only_free.free_peaks[0].height_err)


def test_parameters_of_a_peak(run):
    from trxrd.physics import profiles

    editing.add_phase(run, 5, ICE_VII, 3.7)
    spectrum = run.spectra[5]
    ref = (ICE_VII, 0)
    peak = editing.get_peak(spectrum, ref)
    assert peak.model == "PearsonIV"
    full = profiles.fwhm(peak.model, peak.width, peak.shape)
    editing.set_peak(run, 5, ref, model="PseudoVoigt")  # same position, height and full width
    assert peak.model == "PseudoVoigt" and peak.shape == (0.5,)
    assert profiles.fwhm(peak.model, peak.width, peak.shape) == pytest.approx(full, rel=0.02)
    editing.set_peak(run, 5, ref, model="SplitLorentzian")
    assert peak.shape == (peak.width,)  # symmetric to start with
    editing.set_peak(run, 5, ref, model="Gaussian")
    assert peak.shape == ()
    assert profiles.fwhm(peak.model, peak.width, peak.shape) == pytest.approx(full, rel=0.02)

    editing.set_peak(run, 5, ref, model="PearsonIV", center=16.1, height=2000.0, width=0.08, shape=(1.5, 0.4))
    assert (peak.center, peak.height, peak.width, peak.shape) == (16.1, 2000.0, 0.08, (1.5, 0.4))
    editing.set_peak(run, 5, ref, shape=(100.0, -9.0))  # kept inside the bounds of the model
    assert peak.shape == (20.0, -5.0)
    assert spectrum.model.max() > 1000  # the curve follows
    with pytest.raises(ValueError, match="positive"):
        editing.set_peak(run, 5, ref, width=0.0)
    with pytest.raises(ValueError, match="forme"):
        editing.set_peak(run, 5, ref, shape=(1.0,))


def test_fit_of_one_peak_alone(run):
    editing.add_phase(run, 5, ICE_VII, 3.7)
    spectrum = run.spectra[5]
    fit = spectrum.phases[0]
    other = dataclasses.replace(fit.peaks[1])
    editing.set_peak(run, 5, (ICE_VII, 0), center=fit.peaks[0].center + 0.05, height=100.0)
    fitted = editing.fit_peak(run, 5, (ICE_VII, 0), AnalysisSettings(fit_window=8.0))
    assert fit.peaks[0] is fitted and fitted.height > 1000 and np.isfinite(fitted.center_err)
    moved = fit.peaks[1]
    assert (moved.center, moved.height, moved.width) == (other.center, other.height, other.width)  # the others stay
    assert fit.pressure == pytest.approx(3.7, abs=0.05)  # and the phase follows

    free = editing.add_peak(run, 5, GASKET)
    assert editing.fit_peak(run, 5, free).center == pytest.approx(GASKET, abs=0.02)

    spectrum.two_theta = spectrum.two_theta[:3]  # far too few points: an error, the peak stays
    before = fit.peaks[0]
    with pytest.raises(ValueError, match="impossible"):
        spectrum.intensity, spectrum.baseline = spectrum.intensity[:3], spectrum.baseline[:3]
        editing.fit_peak(run, 5, (ICE_VII, 0))
    assert fit.peaks[0] is before


def test_fit_variation_bounds_the_widths(run):
    from trxrd.physics import profiles

    editing.add_phase(run, 5, ICE_VII, 3.7)
    spectrum = run.spectra[5]
    start = spectrum.phases[0].peaks[0].width
    editing.set_peak(run, 5, (ICE_VII, 0), width=start * 3)  # far too broad
    tight = AnalysisSettings(fit_variation=0.1)
    assert tight.options().fit.variation == 0.1
    fitted = editing.fit_peak(run, 5, (ICE_VII, 0), tight)
    assert fitted.width == pytest.approx(start * 3 / 1.1, rel=1e-3)  # it stops at the bound
    assert profiles.model(fitted.model).function is not None
    settings = AnalysisSettings(peak_model="Gaussian", fit_center_tolerance=0.1)
    assert settings.options().fit.model == "Gaussian" and settings.options().fit.center_tolerance == 0.1
    assert AnalysisSettings().options().fit.variation is None


def test_lines_of_a_phase_chosen_one_by_one(run):
    editing.add_phase(run, 5, ICE_VII, 3.7)
    spectrum = run.spectra[5]
    lines = editing.phase_lines(run, 5, ICE_VII)
    assert [line.label for line in lines] == ["110", "200"] and all(line.used for line in lines)
    assert lines[0].two_theta < lines[1].two_theta and lines[0].intensity == 100.0
    assert editing.phase_lines(run, 5, ICE_VI) == []  # not on this spectrum
    pressure = spectrum.phases[0].pressure
    spectrum.phases[0].pressure = 99.0  # outside the equation of state: no line, no error
    assert editing.phase_lines(run, 5, ICE_VII) == []
    spectrum.phases[0].pressure = pressure

    position = spectrum.phases[0].peaks[1].center
    editing.set_line(run, 5, ICE_VII, (2, 0, 0), False)
    assert [p.hkl for p in spectrum.phases[0].peaks] == [(1, 1, 0)]
    assert [line.used for line in editing.phase_lines(run, 5, ICE_VII)] == [True, False]
    editing.set_line(run, 5, ICE_VII, (2, 0, 0), True)
    editing.set_line(run, 5, ICE_VII, (2, 0, 0), True)  # already there: not twice
    assert [p.hkl for p in spectrum.phases[0].peaks] == [(1, 1, 0), (2, 0, 0)]
    assert spectrum.phases[0].peaks[1].center == pytest.approx(position, abs=0.01)  # back on the detected peak
    assert spectrum.phases[0].peaks[1].name.endswith("_200")

    with pytest.raises(ValueError, match="Pas de raie"):
        editing.set_line(run, 5, ICE_VII, (9, 9, 9), True)
    with pytest.raises(ValueError, match="pas sur ce spectre"):
        editing.set_line(run, 5, ICE_VI, (1, 0, 0), True)


def test_misfit_compares_two_fits(run):
    assert editing.misfit(run.spectra[5]) is None
    editing.add_phase(run, 5, ICE_VII, 3.7)
    before = editing.misfit(run.spectra[5])
    editing.fit_current(run, 5)
    assert 0 < editing.misfit(run.spectra[5]) < before


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
