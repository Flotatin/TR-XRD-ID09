"""The real window, offscreen (QT_QPA_PLATFORM=offscreen in conftest): it builds, opens a run, navigates."""

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from trxrd.gui.presenters.viewer import ImmediateRunner  # noqa: E402
from trxrd.io.h5store import save_run  # noqa: E402


@pytest.fixture
def viewer(tmp_path, sample_run):
    from trxrd.gui.app import create_viewer
    from trxrd.gui.views.main_window import application

    app = application(["test"])
    viewer = create_viewer(runner=ImmediateRunner())
    viewer.window.confirm_discard = lambda: True  # no modal dialog in tests
    viewer.path = save_run(sample_run, tmp_path / "run.trxrd.h5")
    yield viewer
    viewer.window.close()
    app.processEvents()


def test_window_opens_a_run_and_navigates(viewer):
    window = viewer.window
    viewer.main.open(str(viewer.path))
    assert window.windowTitle() == "TR-XRD v2 2.0.0a1 — water02_0001_scan0027"
    x, y = window.spectrum.data.getData()
    assert len(x) == 400 and np.allclose(y, viewer.session.run.spectra[0].intensity)
    assert window.spectrum.fit.getData()[0] is not None
    assert len(window.spectrum._ticks) == 2  # ice VII: ticks + their tooltips
    assert window.navigator.table.rowCount() == 5 and window.phases.rowCount() == 1
    assert window.peaks.rowCount() == 2 and window.map.image.image.shape == (5, 400)
    assert window.pressure.channel.currentText() == "Channel3"  # the non-trigger channel first

    window.actions.get("next").trigger()
    window.actions.get("next").trigger()
    assert viewer.session.index == 2 and window.navigator.slider.value() == 2
    assert window.phases.rowCount() == 2 and window.pressure.marker.value() == pytest.approx(2.0)
    window.navigator.slider.setValue(4)
    assert viewer.session.index == 4 and window.map.marker.value() == 4
    window.actions.get("first").trigger()
    assert viewer.session.index == 0 and window.navigator.table.currentRow() == 0

    window.actions.get("unit_q").trigger()
    assert window.units.currentData() == "q_A^-1" and window.actions.get("unit_q").isChecked()
    assert window.spectrum.data.getData()[0][0] < 1.5  # Å⁻¹
    window.units.setCurrentIndex(window.units.findData("d_A"))
    assert viewer.settings.unit == "d_A"


def test_close_and_reopen(viewer):
    viewer.main.open(str(viewer.path))
    viewer.window.actions.get("close").trigger()
    assert viewer.session.run is None and viewer.window.navigator.table.rowCount() == 0
    assert viewer.window.windowTitle() == "TR-XRD v2 2.0.0a1"
    viewer.main.open(str(viewer.path))
    assert viewer.window.navigator.table.rowCount() == 5


def test_shortcuts_are_unique():
    from trxrd.gui.app import create_viewer
    from trxrd.gui.views.main_window import application

    application(["test"])
    viewer = create_viewer(runner=ImmediateRunner())
    keys = [s for info in viewer.window.actions.info.values() for s in info.shortcuts]
    assert len(keys) == len(set(keys))
    assert viewer.window.actions.find("F3") == "save"  # as in the v1
    assert {row[1] for row in viewer.window.actions.shortcut_table()} >= {"Ouvrir…", "Spectre suivant"}
    viewer.window.close()


def test_thread_runner_reports_in_the_gui_thread(tmp_path, sample_run):
    import time

    from PySide6.QtCore import QThread
    from trxrd.gui.tasks import ThreadRunner
    from trxrd.gui.views.main_window import application
    from trxrd.services.session import Session

    app = application(["test"])
    path = save_run(sample_run, tmp_path / "run.trxrd.h5")
    results, threads = [], []
    runner = ThreadRunner()
    runner.run(lambda progress: (progress(1, 1), Session.read(path))[1],
               lambda result: (results.append(result), threads.append(QThread.currentThread())), results.append)
    runner.run(lambda progress: 1 / 0, results.append, results.append)
    deadline = time.monotonic() + 10
    while len(results) < 2 and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert len(results) == 2 and not runner.busy
    run, saved_to, dirty = next(r for r in results if isinstance(r, tuple))
    assert run.name == sample_run.name and not dirty
    assert "ZeroDivisionError" in next(r for r in results if isinstance(r, str))
    assert threads[0] is app.thread()
