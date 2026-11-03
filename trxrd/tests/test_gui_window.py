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
    assert window.peaks.rowCount() == 2 and window.map.image.image.shape == (400, 5)  # [2θ, time]
    assert window.evolution.channel.currentText() == "Channel3"  # the non-trigger channel first

    window.actions.get("next").trigger()
    window.actions.get("next").trigger()
    assert viewer.session.index == 2 and window.navigator.slider.value() == 2
    assert window.phases.rowCount() == 2 and window.evolution.marker.value() == pytest.approx(2.0)
    window.navigator.slider.setValue(4)
    assert viewer.session.index == 4 and window.map.marker.value() == pytest.approx(4.0)  # ms
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


def test_new_run_dialog_browses_raw_data(raw_root, tmp_path):
    from PySide6.QtCore import QSettings
    from conftest import DATA
    from trxrd.gui.views.main_window import application
    from trxrd.gui.views.new_run import NewRunDialog
    from trxrd.services import raw_data

    application(["test"])
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    settings.setValue("new_run/raw_root", str(raw_root))
    dialog = NewRunDialog(raw_data, settings=settings, runner=ImmediateRunner())
    assert dialog.tree.topLevelItemCount() == 1 and "1 jeux" in dialog.message.text()
    dialog.tree.setCurrentItem(dialog.tree.topLevelItem(0).child(0))
    assert dialog.table.rowCount() == 1 and dialog.table.item(0, 3).text() == "timescan 0.1 1"
    dialog.images_only.setChecked(False)
    assert dialog.table.rowCount() == 2
    ok = dialog.buttons.button(dialog.buttons.StandardButton.Ok)
    dialog.table.selectRow(1)
    assert not ok.isEnabled()  # no calibration, no folder yet
    dialog.poni.setText(str(DATA / "calib_THT8_correct_samedi.poni"))
    dialog.folder.setText(str(tmp_path / "analyse"))
    assert ok.isEnabled()
    request = dialog.request()
    assert request.scans == [raw_root / "water02" / "water02_0001" / "scan0003" / "scan_jf1m_0000.h5"]
    assert request.two_theta_range == (8.0, 25.0) and [p.split("phases")[-1][1:] for p in request.phase_paths] == ["H2O", "Sn"]
    dialog.accept()
    assert NewRunDialog(raw_data, settings=settings, runner=ImmediateRunner()).poni.text() ==         str(DATA / "calib_THT8_correct_samedi.poni")


def test_new_run_dialog_previews_the_scan_and_its_two_theta_range(raw_root, tmp_path):
    pytest.importorskip("pyFAI")
    from PySide6.QtCore import QSettings
    from conftest import DATA
    from trxrd.gui.views.main_window import application
    from trxrd.gui.views.new_run import NewRunDialog
    from trxrd.services import raw_data

    application(["test"])
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    for key, value in (("raw_root", raw_root), ("poni", DATA / "calib_THT8_correct_samedi.poni"),
                       ("mask", DATA / "scan_jf1m_0000.mask")):
        settings.setValue(f"new_run/{key}", str(value))
    dialog = NewRunDialog(raw_data, settings=settings, runner=ImmediateRunner())
    view = dialog.preview_view
    assert "Choisir un scan" in view.text.text()
    dialog.tree.setCurrentItem(dialog.tree.topLevelItem(0).child(0))
    dialog.table.selectRow(0)
    assert view.title.text() == "maximum de 3 images sur 3" and view.curve.xData is not None
    assert view.range.region.getRegion() == (8.0, 25.0) and view.shade.image is not None

    view.range.region.setRegion((10.0, 20.0))  # as the user's drag does
    assert (dialog.tth_min.value(), dialog.tth_max.value()) == (10.0, 20.0)
    assert dialog.request().two_theta_range == (10.0, 20.0) and "10.00 → 20.00°" in view.text.text()
    dialog.tth_max.setValue(22.5)  # typed
    assert view.range.region.getRegion() == (10.0, 22.5)
    view.whole.click()
    low, high = dialog.preview.data.usable
    assert dialog.tth_min.value() == pytest.approx(low) and dialog.tth_max.value() == pytest.approx(high)

    view.mode.setCurrentIndex(1)
    assert dialog.preview.data.cake is not None and view.cake_range.region.isVisible() and not view.shade.isVisible()
    view.which.setCurrentIndex(1)
    view.frame.setValue(2)
    assert view.title.text() == "image n° 2 (de 0 à 2)"
    dialog.reject()
    assert dialog.preview.closed


def test_image_panel_scale_colours_and_levels(tmp_path):
    from PySide6.QtCore import QSettings
    from trxrd.gui.views.images import ImagePanel
    from trxrd.gui.views.main_window import application

    application(["test"])
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    panel = ImagePanel("test", scale="log", settings=settings)
    image = np.outer(np.arange(1, 11), np.ones(20)) ** 3  # 1 .. 1000
    image[0, 0] = np.nan  # masked pixel
    panel.set_image(image)
    assert panel.auto.isChecked()  # setting the levels from the program is not a user choice
    assert panel.levels[1] == pytest.approx(3.0, abs=0.01) and np.isfinite(panel.image.image).all()  # log10
    panel.scale.setCurrentIndex(panel.scale.findData("linear"))
    assert panel.levels[1] == pytest.approx(1000.0, rel=0.01)
    panel.set_levels(10.0, 500.0)
    assert not panel.auto.isChecked()
    panel.set_image(image * 2)  # next spectrum: the levels chosen by the user stay
    assert panel.levels == (10.0, 500.0) and tuple(panel.image.levels) == (10.0, 500.0)
    panel.reset_levels()
    assert panel.levels[1] == pytest.approx(2000.0, rel=0.01)
    panel.colors.setCurrentIndex(panel.colors.findData("gray_r"))
    lut = panel.image.lut(panel.image.image) if callable(panel.image.lut) else panel.image.lut
    assert tuple(lut[0][:3]) == (255, 255, 255)  # reversed grey: low values in white
    panel.show_histogram.setChecked(False)
    assert not panel.histogram.isVisibleTo(panel)

    again = ImagePanel("test", settings=settings)  # the choices are remembered
    assert again.scale.currentData() == "linear" and again.colors.currentData() == "gray_r"
    assert not again.show_histogram.isChecked()


def test_evolution_view_and_aligned_plots(viewer):
    window = viewer.window
    viewer.main.open(str(viewer.path))
    evolution = window.evolution
    assert [evolution.quantity.itemData(k) for k in range(evolution.quantity.count())] == ["P", "V", "a"]
    assert not evolution.relative.isEnabled()  # P has no card value
    evolution.quantity.setCurrentIndex(evolution.quantity.findData("V"))
    assert evolution.plot.getPlotItem().getAxis("left").labelText == "V (Å³)" and evolution.relative.isEnabled()
    evolution.relative.setChecked(True)
    assert evolution.plot.getPlotItem().getAxis("left").labelText == "V/V₀"
    x_range = evolution.plot.getPlotItem().vb.viewRange()[0]
    assert x_range[0] < 0.0 and 4.0 < x_range[1] < 5.0  # the spectra (0..4 ms)
    assert window.map.plot.getPlotItem().vb.viewRange()[0] == pytest.approx(x_range)  # same time axis
    evolution.span.setCurrentIndex(evolution.span.findData("trace"))
    assert evolution.plot.getPlotItem().vb.viewRange()[0][1] > 1000.0  # the whole oscilloscope record (1 s)
    window.map.on_pick(3.2)
    assert viewer.session.index == 3

    window.show()
    window.map.panel.show_histogram.setChecked(True)
    window.cake.panel.show_histogram.setChecked(True)
    assert evolution.spacer.isVisible() and window.spectrum.spacer.isVisible()
    window.map.panel.show_histogram.setChecked(False)
    assert not evolution.spacer.isVisible()
    window.actions.get("cake").trigger()  # cake hidden: no margin left under the spectrum
    assert not window.cake.isVisible() and not window.spectrum.spacer.isVisible()


def test_dpdt_view_with_its_interval(viewer):
    from PySide6.QtGui import QGuiApplication

    window = viewer.window
    viewer.main.open(str(viewer.path))
    dpdt = window.dpdt
    assert dpdt.by.currentData() == "family" and dpdt.window.value() == 2 and len(dpdt._items) == 2  # bars + curve
    assert not dpdt.interval.region.isVisible() and not dpdt.copy.isEnabled()
    dpdt.toggle.setChecked(True)
    assert dpdt.interval.region.getRegion() == pytest.approx((0.0, 4.0))
    assert window.evolution.interval.region.getRegion() == pytest.approx((0.0, 4.0))  # the same interval above
    assert "H2O (Ice_VII, Ice_VI) : pente" in dpdt.result.text() and len(window.evolution._segments) == 1
    window.evolution.interval.region.setRegion((0.5, 2.5))  # dragged on the evolution plot
    assert dpdt.interval.region.getRegion() == pytest.approx((0.5, 2.5)) and "de 0.5 à 2.5 ms" in dpdt.result.text()
    dpdt.copy.click()
    assert QGuiApplication.clipboard().text().startswith("courbe\t")
    dpdt.by.setCurrentIndex(dpdt.by.findData("phase"))
    assert "H2O_Ice_VII :" in dpdt.result.text()
    dpdt.toggle.setChecked(False)
    assert not dpdt.interval.region.isVisible() and dpdt.result.text() == "" and not window.evolution._segments
