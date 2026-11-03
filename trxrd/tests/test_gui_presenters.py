"""Presenters of the viewer, with fake views: no Qt, no window."""

import numpy as np
import pytest

from trxrd.gui.presenters import display
from trxrd.gui.presenters.viewer import (
    DisplaySettings, EvolutionPresenter, MainPresenter, MapPresenter, NavigatorPresenter, SpectrumPresenter,
    TablesPresenter,
)
from trxrd.io.h5store import save_run
from trxrd.physics import eos
from trxrd.services.session import Session


class Recorder:
    """Fake view: records every call, `last(name)` gives the arguments of the last one."""

    def __init__(self, **answers):
        self.calls, self.answers = [], answers

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def method(*args):
            self.calls.append((name, args))
            return self.answers.get(name)
        return method

    def last(self, name):
        return next(args for n, args in reversed(self.calls) if n == name)

    def count(self, name):
        return sum(n == name for n, _ in self.calls)


@pytest.fixture
def viewer(tmp_path, sample_run):
    session, settings = Session(), DisplaySettings()
    views = {k: Recorder() for k in ("spectrum", "evolution", "map", "phases", "peaks", "navigator")}
    window = Recorder(confirm_discard=False)
    presenters = [
        SpectrumPresenter(session, settings, views["spectrum"]),
        EvolutionPresenter(session, views["evolution"]),
        MapPresenter(session, settings, views["map"]),
        TablesPresenter(session, views["phases"], views["peaks"]),
        NavigatorPresenter(session, views["navigator"]),
    ]
    main = MainPresenter(session, settings, window, version="2.0.0a1")
    path = save_run(sample_run, tmp_path / "run.trxrd.h5")
    return session, settings, views, window, main, path, presenters


def test_open_shows_every_panel(viewer, sample_run):
    session, settings, views, window, main, path, _ = viewer
    assert window.last("set_title") == ("TR-XRD v2 2.0.0a1",)
    main.open(str(path))
    assert session.run is not None and not session.dirty
    assert window.last("set_title") == ("TR-XRD v2 2.0.0a1 — water02_0001_scan0027",)
    assert window.count("show_error") == 0 and window.last("set_busy") == (None,)

    curves, reset = views["spectrum"].last("show_spectrum")
    assert reset and curves.x_label == "2θ (°)" and curves.title.startswith("Spectre 0 / 4")
    np.testing.assert_allclose(curves.fit, sample_run.spectra[0].model + sample_run.spectra[0].baseline)
    assert [l.phase for l in curves.lines] == ["H2O_Ice_VII"] and "110" in curves.lines[0].labels

    data = views["evolution"].last("show_evolution")[0]
    assert data.x_label == "t (ms)" and data.y_label == "P (GPa)"
    assert [s.phase for s in data.series] == ["H2O_Ice_VII", "H2O_Ice_VI"]
    np.testing.assert_allclose(data.series[0].y, [2.0, 3.0, 4.0])
    assert {t.name for t in data.traces} == {"Channel2", "Channel3"} and [t.trigger for t in data.traces] == [True, False]
    assert data.span == (0.0, 4.0) and data.trace_span == (0.0, 1000.0)  # the whole oscilloscope record
    assert views["evolution"].last("set_marker") == (0.0,)

    patterns = views["map"].last("show_map")[0]
    assert patterns.image.shape == (400, 5) and patterns.x_label == "t (ms)" and patterns.y_label == "2θ (°)"
    np.testing.assert_allclose(patterns.x, [0, 1, 2, 3, 4])  # one column per spectrum, on the time axis
    assert views["map"].last("set_marker") == (0.0,)
    assert views["navigator"].last("show_spectra")[1][2] == ("2", "2", "Ice VII 4.00, Ice VI 1.50")
    assert views["navigator"].last("set_current") == (0, 5)
    assert views["phases"].last("show_rows")[1][0][:3] == ("H2O_Ice_VII", "2.000", "0.050")
    assert [r[1] for r in views["peaks"].last("show_rows")[1]] == ["110", "200"]


def test_navigation_from_every_view(viewer):
    session, settings, views, window, main, path, _ = viewer
    main.open(str(path))
    views["navigator"].on_select(3)
    assert session.index == 3 and views["spectrum"].last("show_spectrum")[1] is False  # zoom kept
    assert views["map"].last("set_marker") == (3.0,)  # ms
    views["evolution"].on_pick(1.9)  # ms: nearest spectrum is #2 (t = 2 ms)
    assert session.index == 2 and views["evolution"].last("set_marker") == (2.0,)
    views["map"].on_pick(3.8)
    main.step(+1)  # already the last one
    assert session.index == 4
    main.first()
    assert session.index == 0
    main.last()
    assert session.index == 4 and views["phases"].last("show_rows")[1] == []


def test_units(viewer, sample_run):
    session, settings, views, window, main, path, _ = viewer
    main.open(str(path))
    main.set_unit("q_A^-1")
    curves, reset = views["spectrum"].last("show_spectrum")
    assert reset and curves.x_label == "Q (Å⁻¹)"
    lam = display.wavelength(sample_run)
    np.testing.assert_allclose(curves.x[0], 4 * np.pi * np.sin(np.radians(8.0) / 2) / (lam * 1e10))
    assert window.last("set_units") == (display.UNITS, "q_A^-1")
    main.set_unit("d_A")
    data = views["map"].last("show_map")[0]
    assert data.y_label == "d (Å)" and np.all(np.diff(data.y) > 0) and data.image.shape == (400, 5)
    # no energy: only 2θ
    session.run.calibration = None
    session.modified()
    assert views["spectrum"].last("show_spectrum")[0].x_label == "2θ (°)"
    assert window.last("set_units") == (("2th_deg",), "2th_deg")


def test_open_errors_and_unsaved_changes(viewer, tmp_path):
    session, settings, views, window, main, path, _ = viewer
    main.open(str(tmp_path / "missing.trxrd.h5"))
    assert window.last("show_error")[0] == "Ouverture impossible" and session.run is None
    assert window.last("set_busy") == (None,)

    main.open(str(path))
    session.run.note = "edited"
    session.modified()
    assert window.last("set_title")[0].endswith(" *")
    main.close()  # the fake window refuses to lose the changes
    assert session.run is not None
    main.open(str(path))
    assert session.run.note == "edited"
    window.answers["confirm_discard"] = True
    main.close()
    assert session.run is None and views["spectrum"].count("clear") >= 1 and views["map"].count("clear") >= 1


def test_save_and_save_as(viewer, tmp_path):
    session, settings, views, window, main, path, _ = viewer
    main.open(str(path))
    session.run.note = "edited"
    session.modified()
    main.save()
    assert not session.dirty and Session().open(path).note == "edited"
    window.answers["ask_save_path"] = str(tmp_path / "copy.trxrd.h5")
    main.save_as()
    assert session.path == tmp_path / "copy.trxrd.h5" and (tmp_path / "copy.trxrd.h5").exists()


def test_legacy_file_opens_unsaved(viewer, tmp_path):
    from conftest import DATA

    session, settings, views, window, main, path, _ = viewer
    main.open(str(DATA / "legacy_sample.CEDX"))
    assert session.dirty and session.path.name == "legacy_sample.trxrd.h5"
    assert views["spectrum"].last("show_spectrum")[0].lines[0].phase == "Ga_I"


def test_display_without_times_uses_the_index(sample_run):
    for spectrum in sample_run.spectra:
        spectrum.time = np.nan
    data = display.evolution_data(sample_run)
    assert data.x_label == "spectre" and list(data.x) == [0, 1, 2, 3, 4] and data.traces == []
    assert display.nearest_spectrum(data.x, 2.6) == 3 and data.trace_span is None
    assert display.time_map(sample_run, "2th_deg").x_label == "spectre"


def test_new_runs_from_raw_data(viewer, raw_root, tmp_path):
    from conftest import DATA, LIBRARY
    from trxrd.gui.presenters.viewer import NewRunRequest

    session, settings, views, window, main, path, _ = viewer
    scan = raw_root / "water02" / "water02_0001" / "scan0003" / "scan_jf1m_0000.h5"
    request = NewRunRequest([scan, scan], tmp_path / "analyse", str(DATA / "calib_THT8_correct_samedi.poni"),
                            str(DATA / "scan_jf1m_0000.mask"), (8.0, 25.0), [str(LIBRARY / "Ga")])
    window.answers["ask_new_run"] = request
    main.new_run()
    assert window.last("ask_new_run")[0].find_datasets  # the dialog gets the browsing functions
    progress = [args for name, args in window.calls if name == "set_progress"]
    assert progress[-1] == (2000, 2000) and window.count("end_progress") == 1
    assert session.run.name == "water02_0001_scan0003" and len(session.run) == 3 and not session.dirty
    assert session.path == tmp_path / "analyse" / "water02_0001_scan0003.trxrd.h5"
    assert window.last("show_info")[0] == "Runs créés" and views["navigator"].last("set_current") == (0, 3)


def test_new_run_cancelled_or_failing(viewer, raw_root, tmp_path):
    from conftest import DATA
    from trxrd.gui.presenters.viewer import NewRunRequest

    session, settings, views, window, main, path, _ = viewer
    scan = raw_root / "water02" / "water02_0001" / "scan0003" / "scan_jf1m_0000.h5"
    window.answers["ask_new_run"] = NewRunRequest([scan], tmp_path / "out", str(DATA / "calib_THT8_correct_samedi.poni"))
    main.runner.cancelled = True  # the user presses "Arrêter"
    main.new_run()
    assert session.run is None and "arrêtée" in window.last("set_status")[0] and window.count("show_error") == 0
    main.runner.cancelled = False
    window.answers["ask_new_run"] = NewRunRequest([scan], tmp_path / "out", str(tmp_path / "missing.poni"))
    main.new_run()
    assert window.last("show_error")[0] == "Création du run impossible" and session.run is None
    window.answers["ask_new_run"] = None  # dialog closed
    main.new_run()
    assert window.count("start_progress") == 2


class DeferredRunner:
    """Keeps the tasks; the test runs them in any order (out-of-order results of a thread pool)."""

    def __init__(self):
        self.jobs = []

    def run(self, task, on_done, on_error, on_progress=None):
        self.jobs.append((task, on_done, on_error))

    def finish(self, k):
        task, on_done, on_error = self.jobs[k]
        on_done(task(lambda done, total: True))


def test_cake_of_the_current_spectrum(tmp_path, raw_root, sample_run):
    from conftest import DATA
    from trxrd.gui.presenters.viewer import CakePresenter
    from trxrd.services.run_builder import build_and_save

    scan = raw_root / "water02" / "water02_0001" / "scan0003" / "scan_jf1m_0000.h5"
    built = build_and_save(scan, tmp_path / "analyse", DATA / "calib_THT8_correct_samedi.poni",
                           DATA / "scan_jf1m_0000.mask", two_theta_range=(8, 25))
    session, settings, view = Session(), DisplaySettings(), Recorder()
    presenter = CakePresenter(session, settings, view)
    session.open(built)
    cake = view.last("show_cake")[0]
    assert cake.x_label == "2θ (°)" and cake.image.shape[1] == len(session.run.spectra[0].two_theta)
    assert 0 < cake.image.shape[0] <= 360 and len(cake.azimuth) == cake.image.shape[0]  # covered sector only
    np.testing.assert_allclose(cake.x, session.run.spectra[0].two_theta, atol=1e-6)  # same axis as the spectrum
    assert np.nanmax(cake.image) > 0
    settings.set_unit("q_A^-1")
    assert view.last("show_cake")[0].x_label == "Q (Å⁻¹)" and np.all(np.diff(view.last("show_cake")[0].x) > 0)
    presenter.set_mode("image")
    assert view.last("show_image")[0].ndim == 2
    presenter.set_visible(False)
    count = len(view.calls)
    session.select(2)
    assert len(view.calls) == count  # nothing computed while hidden

    session.attach(sample_run, tmp_path / "elsewhere.trxrd.h5")  # no raw scan anywhere
    presenter.set_visible(True)
    assert "introuvables" in view.last("show_message")[0]


def test_cake_newest_request_wins(tmp_path, raw_root):
    from conftest import DATA
    from trxrd.gui.presenters.viewer import CakePresenter
    from trxrd.services.run_builder import build_and_save

    scan = raw_root / "water02" / "water02_0001" / "scan0003" / "scan_jf1m_0000.h5"
    built = build_and_save(scan, tmp_path / "analyse", DATA / "calib_THT8_correct_samedi.poni", two_theta_range=(8, 25))
    session, view, runner = Session(), Recorder(), DeferredRunner()
    CakePresenter(session, DisplaySettings(), view, runner)
    session.open(built)
    session.select(2)
    runner.finish(1)  # spectrum 2 comes back first
    runner.finish(0)  # then the stale spectrum 0
    assert view.count("show_cake") == 1


def test_cake_keeps_the_covered_azimuth_sector_in_one_piece():
    azimuth = np.arange(-179.5, 180, 1.0)
    image = np.full((360, 4), np.nan)
    image[:60] = 1.0  # -180..-120
    image[280:] = 2.0  # 100..180: the same sector, cut at ±180°
    az, cropped = display.compact_azimuth(azimuth, image)
    assert cropped.shape == (140, 4) and np.isfinite(cropped).all()
    assert az[0] == pytest.approx(100.5) and az[-1] == pytest.approx(239.5) and np.allclose(np.diff(az), 1.0)
    full = np.ones((360, 4))
    assert display.compact_azimuth(azimuth, full)[1].shape == (360, 4)


def test_quantities_versus_time(sample_run):
    ice6 = sample_run.spectra[2].phase_fit("H2O_Ice_VI")
    ice6.a, ice6.a_err, ice6.c, ice6.c_err = 6.0, 0.003, 5.4, 0.004  # tetragonal: a and c are free
    keys = [key for key, _ in display.available_quantities(sample_run)]
    assert keys == ["P", "V", "a", "c", "c/a"]  # ice VII is cubic: no b, c or c/a of its own

    ratio = display.evolution_data(sample_run, "c/a")
    assert [s.phase for s in ratio.series] == ["H2O_Ice_VI"] and ratio.y_label == "c/a"
    assert ratio.series[0].y[0] == pytest.approx(0.9)
    assert ratio.series[0].error[0] == pytest.approx(0.9 * np.hypot(0.003 / 6.0, 0.004 / 5.4))

    card = sample_run.phases["H2O_Ice_VII"]
    volume = display.evolution_data(sample_run, "V", relative=True)
    assert volume.y_label == "V/V₀" and volume.relative and volume.can_be_relative
    np.testing.assert_allclose(volume.series[0].y, np.array([36.0, 37.0, 38.0]) / card.v0_at(eos.T_REF))
    np.testing.assert_allclose(volume.series[0].error, 0.1 / card.v0_at(eos.T_REF))
    relative_a = display.evolution_data(sample_run, "a", relative=True)
    assert relative_a.y_label == "a/a₀" and relative_a.series[0].y[0] == pytest.approx(3.3 / card.a)

    pressure = display.evolution_data(sample_run, "P", relative=True)  # P has no card value
    assert not pressure.relative and not pressure.can_be_relative and pressure.y_label == "P (GPa)"
    assert display.evolution_data(sample_run, "b").quantity == "P"  # no b in this run: back to P
    with pytest.raises(ValueError):
        display.evolution_data(sample_run, "Z")


def test_evolution_presenter_changes_the_quantity(viewer):
    session, settings, views, window, main, path, presenters = viewer
    main.open(str(path))
    views["evolution"].on_quantity("V", True)
    data = views["evolution"].last("show_evolution")[0]
    assert data.quantity == "V" and data.y_label == "V/V₀" and [s.phase for s in data.series] == ["H2O_Ice_VII"]
    session.select(1)
    assert views["evolution"].last("set_marker") == (1.0,)
    with pytest.raises(ValueError):
        views["evolution"].on_quantity("Z", False)


def test_map_columns_follow_the_time_axis(sample_run):
    grid, columns = display.time_columns(np.array([0.0, 1.0, 2.0, 3.0]))
    np.testing.assert_allclose(grid, [0, 1, 2, 3])
    assert list(columns) == [0, 1, 2, 3]
    grid, columns = display.time_columns(np.array([0.0, 1.0, 2.0, 10.0]))  # a gap: uniform columns, nearest spectrum
    np.testing.assert_allclose(grid, np.arange(11.0))
    assert list(columns) == [0, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3]
    grid, columns = display.time_columns(np.array([2.0, 0.0, 1.0]))  # not sorted
    assert list(grid) == [0.0, 1.0, 2.0] and list(columns) == [1, 2, 0]
    assert list(display.time_columns(np.array([5.0]))[1]) == [0]

    sample_run.spectra[4].time = 10e-3  # s
    data = display.time_map(sample_run, "2th_deg")
    assert data.image.shape == (400, 11) and list(data.columns[-3:]) == [4, 4, 4]
    np.testing.assert_allclose(data.image[:, 0], sample_run.spectra[0].corrected)


def test_image_scales_and_levels():
    image = np.array([[np.nan, -5.0, 1.0], [10.0, 100.0, 1000.0]])
    np.testing.assert_array_equal(display.scaled_image(image, "linear"), image)
    logged = display.scaled_image(image, "log")
    assert np.isnan(logged[0, 0]) and logged[1, 2] == pytest.approx(3.0) and np.isfinite(logged[0, 1])
    assert logged[0, 1] == logged[0, 2] == pytest.approx(np.log10(np.percentile([1, 10, 100, 1000], 5)))  # floor
    rooted = display.scaled_image(image, "sqrt")
    assert rooted[0, 1] == 0.0 and rooted[1, 1] == pytest.approx(10.0) and np.isnan(rooted[0, 0])
    with pytest.raises(ValueError):
        display.scaled_image(image, "cubic")

    hot = np.zeros((100, 100))
    hot[:, :50] = 1.0
    hot[0, 0] = 1e6  # one hot pixel
    low, high = display.robust_levels(hot)
    assert low == 0.0 and high == 1.0
    assert display.robust_levels(np.full((3, 3), 7.0)) == (7.0, 8.0)
    assert display.robust_levels(np.full((3, 3), np.nan)) == (0.0, 1.0)


def test_dpdt_display(sample_run):
    data = display.rate_data(sample_run, half_window=1)
    assert data.y_label == "dP/dt (GPa/ms)" and [s.name for s in data.series] == ["H2O"] and data.message == ""
    assert data.series[0].color == 0 and data.series[0].y[0] == pytest.approx(0.375)  # colour of ice VII
    assert len(display.rate_data(sample_run, 1, by="phase").series) == 1  # ice VI: a single spectrum
    found = display.interval_data(sample_run, (2.0, 0.0), by="phase")
    assert (found.start, found.stop) == (0.0, 2.0) and found.segments == [(0.0, 2.0, 2.0, 4.0, 0)]
    assert found.lines == ["H2O_Ice_VII : pente 1 GPa/ms ; ΔP/Δt = 1 GPa/ms (ΔP = +2.000 GPa en 2 ms, 3 spectres)"]  # exact line
    assert found.table.splitlines()[1].startswith("H2O_Ice_VII\t")
    assert display.interval_data(sample_run, (0.0, 2.0)).lines == [
        "H2O (Ice_VII, Ice_VI) : pente 0.38 ± 0.36 GPa/ms ; ΔP/Δt = 0.375 GPa/ms (ΔP = +0.750 GPa en 2 ms, 3 spectres)"]
    assert display.interval_data(sample_run, None) is None
    assert display.with_error(-25.3127, 0.4231) == "-25.31 ± 0.42" and display.with_error(-25.3, 3.2) == "-25.3 ± 3.2"
    assert display.with_error(2.0, np.nan) == "2" and display.with_error(np.nan, 1.0) == "—"
    for spectrum in sample_run.spectra:
        spectrum.time = np.nan
    assert "base de temps" in display.rate_data(sample_run).message
    assert display.interval_data(sample_run, (0.0, 1.0)) is None


def test_dpdt_presenter_and_shared_interval(tmp_path, sample_run):
    from trxrd.gui.presenters.viewer import RatePresenter, RateSettings

    session, settings = Session(), RateSettings()
    evolution, rates_view = Recorder(), Recorder()
    EvolutionPresenter(session, evolution, settings)
    RatePresenter(session, rates_view, settings)
    session.open(save_run(sample_run, tmp_path / "run.trxrd.h5"))
    data = rates_view.last("show_rates")[0]
    assert data.by == "family" and data.half_window == 2 and [s.name for s in data.series] == ["H2O"]
    assert rates_view.last("show_interval") == (None,) and evolution.last("show_interval") == (None, [])

    rates_view.on_interval_toggle(True)  # the whole range of the spectra
    assert settings.interval == (0.0, 4.0)
    found = rates_view.last("show_interval")[0]
    assert (found.start, found.stop) == (0.0, 4.0) and [m.name for m in found.means] == ["H2O"]
    region, segments = evolution.last("show_interval")
    assert region == (0.0, 4.0) and len(segments) == 1  # the straight line over P(t)
    evolution.on_interval(2.0, 0.5)  # dragged on the evolution plot
    assert settings.interval == (0.5, 2.0) and rates_view.last("show_interval")[0].start == 0.5
    evolution.on_quantity("V", False)
    assert evolution.last("show_interval") == ((0.5, 2.0), [])  # no P line over V(t)

    count = rates_view.count("show_rates")
    rates_view.on_options("phase", 3)
    data = rates_view.last("show_rates")[0]
    assert rates_view.count("show_rates") == count + 1 and data.by == "phase" and data.half_window == 3
    rates_view.on_pick(3.1)
    assert session.index == 3 and rates_view.last("set_marker") == (3.0,)

    session.open(save_run(sample_run, tmp_path / "other.trxrd.h5"))  # a new run: no interval
    assert settings.interval is None and rates_view.last("show_interval") == (None,)
    rates_view.on_interval_toggle(False)
    with pytest.raises(ValueError):
        settings.update(by="sample")
    with pytest.raises(ValueError):
        settings.update(colour="red")
    settings.update(half_window=0)
    assert settings.half_window == 1


def test_raw_scan_preview_before_integration(tmp_path):
    pytest.importorskip("pyFAI")
    from conftest import DATA
    from trxrd.gui.presenters.preview import NO_SCAN, PreviewPresenter

    view = Recorder()
    presenter = PreviewPresenter(view, two_theta_range=(25.0, 8.0))
    assert view.last("show_message") == (NO_SCAN,) and presenter.range == (8.0, 25.0)
    presenter.set_calibration(str(tmp_path / "being typed"), None)  # not a file yet: nothing to do
    presenter.set_scan(DATA / "scan_jf1m_0000.h5")
    data = view.last("show_preview")[0]
    assert data.title == "maximum de 16 images sur 16" and data.x is None and ".poni" in data.message
    assert view.last("show_range") == (8.0, 25.0, None, "")
    assert [args for name, args in view.calls if name == "set_busy"][-1] == (None,)

    presenter.set_calibration(DATA / "calib_THT8_correct_samedi.poni", DATA / "scan_jf1m_0000.mask")
    data = view.last("show_preview")[0]
    low, high, outside, text = view.last("show_range")
    assert data.x is not None and outside.shape == data.image.shape and "8.00 → 25.00°" in text
    presenter.set_range(20.0, 12.0)  # dragged, or typed in the dialog
    assert view.last("show_range")[:2] == (12.0, 20.0) and view.count("show_preview") == 2  # nothing recomputed

    presenter.set_mode("cake")
    assert view.last("show_preview")[0].cake is not None and view.last("show_range")[2] is None  # no shading on a cake
    presenter.set_frame(2)
    assert view.last("show_preview")[0].title == "image n° 2 (de 0 à 15)"
    with pytest.raises(ValueError):
        presenter.set_mode("map")

    presenter.set_calibration(tmp_path / "missing.poni", None)
    presenter.set_frame(None)
    assert view.last("show_preview")[0].x is None  # back to the images only
    (tmp_path / "broken.poni").write_text("not a calibration")
    presenter.set_calibration(tmp_path / "broken.poni", None)
    assert "ValueError" in view.last("show_message")[0]
    presenter.set_scan(None)
    assert view.last("show_message") == (NO_SCAN,)


def test_raw_scan_preview_newest_request_wins_and_closing_drops_results():
    from conftest import DATA
    from trxrd.gui.presenters.preview import PreviewPresenter

    view, runner = Recorder(), DeferredRunner()
    presenter = PreviewPresenter(view, runner)
    presenter.set_scan(DATA / "scan_jf1m_0000.h5")
    presenter.set_frame(1)
    runner.finish(1)  # frame 1 comes back first
    runner.finish(0)  # then the stale maximum
    assert view.count("show_preview") == 1 and view.last("show_preview")[0].title.startswith("image n° 1")
    presenter.set_frame(2)
    presenter.close()  # the dialog is closed before the result
    runner.finish(2)
    assert view.count("show_preview") == 1


def test_automatic_analysis_from_the_gui(compression_run):
    from synthetic import GASKET, X
    from trxrd.gui.presenters.analysis import AnalysisPresenter, AnalysisRequest
    from trxrd.services import analysis
    from trxrd.services.analysis import AnalysisSettings

    session, window = Session(), Recorder(ask_yes_no=True)
    session.attach(compression_run, None)
    changes = []
    session.subscribe("data_changed", lambda *_: changes.append(1))
    presenter = AnalysisPresenter(session, window)
    choices = presenter.choices()
    assert [name for name, _, checked in choices.phases if checked] == list(compression_run.phases)
    assert choices.count == 6 and choices.analysed == 0 and choices.two_theta == (8.0, float(X[-1]))
    assert dict((n, label) for n, label, _ in choices.phases)["H2O_Ice_VII"] == "H2O_Ice_VII  (-1–70 GPa)"

    window.answers["ask_analysis"] = AnalysisRequest(AnalysisSettings(pressure_range=(0.0, 10.0)), list(range(6)))
    presenter.start()
    title, text = window.last("ask_yes_no")
    assert title == "Pics fixes" and f"{GASKET:.1f}" in text  # the gasket line is proposed...
    assert window.count("start_progress") == 2 and window.count("end_progress") == 2  # ...and the analysis run again
    assert window.count("ask_yes_no") == 1 and changes == [1, 1] and session.dirty
    saved = analysis.saved_settings(session.run)
    assert len(saved.exclude) == 1 and saved.exclude[0][0] < GASKET < saved.exclude[0][1]
    assert presenter.last.tracked >= 1 and all(s.phases for s in session.run.spectra)
    assert window.last("show_message")[0].endswith("phases : H2O_Ice_VI 3, H2O_Ice_VII 4")

    assert presenter.choices().settings.exclude == saved.exclude and presenter.choices().analysed == 6
    window.answers["ask_yes_no"] = False  # "6 spectra already hold phases: replace them?" -> no
    presenter.start()
    assert window.count("start_progress") == 2 and "déjà des phases" in window.last("ask_yes_no")[1]

    window.answers["ask_yes_no"] = True
    presenter.runner.cancelled = True  # the user presses "Arrêter" at once
    presenter.start()
    presenter.runner.cancelled = False
    assert "arrêtée" in window.last("show_message")[0] and presenter.last.indices == [0]

    window.answers["ask_analysis"] = AnalysisRequest(AnalysisSettings(phases=["missing"]), [0])
    presenter.start()
    assert window.last("show_error") == ("Analyse impossible", "ValueError: Aucune phase candidate : en cocher au moins une")
    assert not presenter.running
    window.answers["ask_analysis"] = None  # dialog closed
    presenter.start()
    assert window.count("start_progress") == 4


def test_analysis_needs_a_run_with_its_energy(sample_run):
    from trxrd.gui.presenters.analysis import AnalysisPresenter, parse_zones, zones_text

    session, window = Session(), Recorder()
    presenter = AnalysisPresenter(session, window)
    presenter.start()  # no run: nothing
    sample_run.calibration = None
    session.attach(sample_run, None)
    presenter.start()
    assert window.last("show_error")[0] == "Analyse impossible" and window.count("ask_analysis") == 0

    assert parse_zones(" 18.479:18.679; 20,1 - 20,3\n21.5:21.2 ") == [(18.479, 18.679), (20.1, 20.3), (21.2, 21.5)]
    assert parse_zones("") == [] and zones_text([(18.479, 18.679), (20.1, 20.3)]) == "18.479:18.679; 20.1:20.3"
    with pytest.raises(ValueError, match="18.5"):
        parse_zones("18.5")
