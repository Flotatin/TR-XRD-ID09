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
