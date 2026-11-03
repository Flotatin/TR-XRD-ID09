"""Presenters of the viewer, with fake views: no Qt, no window."""

import numpy as np
import pytest

from trxrd.gui.presenters import display
from trxrd.gui.presenters.viewer import (
    DisplaySettings, MainPresenter, MapPresenter, NavigatorPresenter, PressurePresenter, SpectrumPresenter,
    TablesPresenter,
)
from trxrd.io.h5store import save_run
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
    views = {k: Recorder() for k in ("spectrum", "pressure", "map", "phases", "peaks", "navigator")}
    window = Recorder(confirm_discard=False)
    presenters = [
        SpectrumPresenter(session, settings, views["spectrum"]),
        PressurePresenter(session, views["pressure"]),
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

    data = views["pressure"].last("show_pressure")[0]
    assert data.x_label == "t (ms)" and [s.phase for s in data.series] == ["H2O_Ice_VII", "H2O_Ice_VI"]
    np.testing.assert_allclose(data.series[0].pressure, [2.0, 3.0, 4.0])
    assert {t.name for t in data.traces} == {"Channel2", "Channel3"} and [t.trigger for t in data.traces] == [True, False]
    assert views["pressure"].last("set_marker") == (0.0,)

    image = views["map"].last("show_map")[0].image
    assert image.shape == (5, 400)
    assert views["navigator"].last("show_spectra")[1][2] == ("2", "2", "Ice VII 4.00, Ice VI 1.50")
    assert views["navigator"].last("set_current") == (0, 5)
    assert views["phases"].last("show_rows")[1][0][:3] == ("H2O_Ice_VII", "2.000", "0.050")
    assert [r[1] for r in views["peaks"].last("show_rows")[1]] == ["110", "200"]


def test_navigation_from_every_view(viewer):
    session, settings, views, window, main, path, _ = viewer
    main.open(str(path))
    views["navigator"].on_select(3)
    assert session.index == 3 and views["spectrum"].last("show_spectrum")[1] is False  # zoom kept
    assert views["map"].last("set_marker") == (3,)
    views["pressure"].on_pick(1.9)  # ms: nearest spectrum is #2 (t = 2 ms)
    assert session.index == 2 and views["pressure"].last("set_marker") == (2.0,)
    views["map"].on_pick(4)
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
    assert data.x_label == "d (Å)" and np.all(np.diff(data.x) > 0) and data.image.shape == (5, 400)
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
    data = display.pressure_data(sample_run)
    assert data.x_label == "spectre" and list(data.x) == [0, 1, 2, 3, 4] and data.traces == []
    assert display.nearest_spectrum(data.x, 2.6) == 3


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
