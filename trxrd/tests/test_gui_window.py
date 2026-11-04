"""The real window, offscreen (QT_QPA_PLATFORM=offscreen in conftest): it builds, opens a run, navigates."""

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import pyqtgraph as pg  # noqa: E402

from trxrd.gui.presenters.viewer import ImmediateRunner  # noqa: E402
from trxrd.io.h5store import save_run  # noqa: E402

pg.setConfigOption("mouseRateLimit", -1)  # the mouse events of these tests follow each other within microseconds


@pytest.fixture
def viewer(tmp_path, sample_run):
    from trxrd.gui.builder import create_viewer
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
    first = viewer.session.run.spectra[0]
    x, y = window.spectrum.data.getData()
    from scipy.signal import savgol_filter
    assert len(x) == 400 and np.allclose(y, savgol_filter(first.intensity, 9, 2) - first.baseline)  # filtered - background
    assert window.spectrum.fit.getData()[0] is not None
    window.actions.get("raw").trigger()  # Ctrl+B: the raw signal, with its background
    assert np.allclose(window.spectrum.data.getData()[1], first.intensity)
    assert np.allclose(window.spectrum.baseline.getData()[1], first.baseline)
    assert [label.text for sample, label in window.spectrum.legend.items][0] == "brut"
    window.actions.get("raw").trigger()
    assert len(window.spectrum._ticks) == 2  # ice VII: ticks + their tooltips
    assert window.navigator.table.rowCount() == 5 and window.phases.table.rowCount() == 1
    assert window.peaks.table.rowCount() == 2 and window.map.image.image.shape == (400, 5)  # [2θ, time]
    assert window.evolution.channel.currentText() == "Channel3"  # the non-trigger channel first

    window.actions.get("next").trigger()
    window.actions.get("next").trigger()
    assert viewer.session.index == 2 and window.navigator.slider.value() == 2
    assert window.phases.table.rowCount() == 2 and window.evolution.marker.value() == pytest.approx(2.0)
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
    from trxrd.gui.builder import create_viewer
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
    assert dpdt.interval.region.getRegion() == pytest.approx((-0.5, 4.5))
    assert window.evolution.interval.region.getRegion() == pytest.approx((-0.5, 4.5))  # the same interval above
    assert window.map.interval.region.getRegion() == pytest.approx((-0.5, 4.5))  # and on the map
    assert "H2O (Ice_VII, Ice_VI) : pente" in dpdt.result.text() and len(window.evolution._segments) == 1
    window.evolution.interval.region.setRegion((0.5, 2.5))  # dragged on the evolution plot
    assert dpdt.interval.region.getRegion() == pytest.approx((0.5, 2.5)) and "de 0.5 à 2.5 ms" in dpdt.result.text()
    dpdt.copy.click()
    assert QGuiApplication.clipboard().text().startswith("courbe\t")
    dpdt.by.setCurrentIndex(dpdt.by.findData("phase"))
    assert "H2O_Ice_VII :" in dpdt.result.text()
    dpdt.toggle.setChecked(False)
    assert not dpdt.interval.region.isVisible() and dpdt.result.text() == "" and not window.evolution._segments


def test_analysis_dialog(viewer):
    from conftest import LIBRARY
    from trxrd.gui.presenters.analysis import AnalysisChoices
    from trxrd.gui.views.analysis import AnalysisDialog
    from trxrd.physics import eos
    from trxrd.services import analysis
    from trxrd.services.analysis import AnalysisSettings

    settings = AnalysisSettings(phases=["H2O_Ice_VI"], pressure_range=(0.5, 12.0), exclude=[(18.48, 18.68)],
                                max_fwhm=0.8, noise_window=2.5, shoulders=False, spectrum_filter=False)
    choices = AnalysisChoices([("H2O_Ice_VI", "H2O_Ice_VI", True), ("H2O_Ice_VII", "H2O_Ice_VII", False)],
                              settings, count=6, current=2, analysed=3, two_theta=(8.0, 25.0))
    dialog = AnalysisDialog(choices, analysis.library_phases)
    assert dialog.zones.text() == "18.48:18.68" and (dialog.pmin.value(), dialog.pmax.value()) == (0.5, 12.0)
    request = dialog.request()
    assert request.indices == list(range(6)) and request.settings.phases == ["H2O_Ice_VI"]
    assert request.settings.exclude == [(18.48, 18.68)] and request.settings.window is None
    assert request.settings.temperature == eos.T_REF and request.settings.max_step is None
    kept = request.settings  # set in the Détection tab, not shown here: kept
    assert (kept.max_fwhm, kept.noise_window, kept.shoulders, kept.spectrum_filter) == (0.8, 2.5, False, False)
    dialog.current.setChecked(True)
    assert dialog.request().indices == [2]
    dialog.first.setValue(1)
    dialog.last.setValue(3)
    assert dialog.between.isChecked() and dialog.request().indices == [1, 2, 3]
    dialog.limit.setChecked(True)
    assert dialog.request().settings.window == (8.0, 25.0)

    dialog.add_folder(str(LIBRARY / "Sn"))
    assert "Sn_Beta" in dialog.phase_names() and dialog.request().settings.library == [str(LIBRARY / "Sn")]
    dialog.add_folder(str(LIBRARY / "H2O"))  # a card of the run: kept, with the P range of the library
    assert dialog.phase_names(checked_only=False).count("H2O_Ice_VI") == 1
    assert dialog.phases.item(0).text() == "H2O_Ice_VI  (-0.2–3 GPa)"
    dialog.zones.setText("18.5")
    dialog.accept()
    assert dialog.answer is None and "18.5" in dialog.message.text()  # stays open
    dialog.zones.setText("")
    dialog._check_all(True)
    assert dialog.request().settings.phases is None  # every phase, those added later too
    dialog._check_all(False)
    dialog.accept()
    assert dialog.answer is None and "Cocher" in dialog.message.text()


def test_automatic_analysis_action(viewer, compression_run):
    from trxrd.gui.presenters.analysis import AnalysisRequest
    from trxrd.services.analysis import AnalysisSettings

    window = viewer.window
    assert window.actions.get("auto").shortcut().toString() == "Ctrl+R"
    viewer.session.attach(compression_run, None)
    asked = []
    window.ask_analysis = lambda choices, library: asked.append(choices) or AnalysisRequest(
        AnalysisSettings(pressure_range=(0.0, 10.0)), list(range(6)))
    window.ask_yes_no = lambda title, text: False  # do not exclude the gasket line this time
    window.actions.get("auto").trigger()
    assert asked[0].count == 6 and all(s.phases for s in viewer.session.run.spectra)
    assert window.windowTitle().endswith("*") and window.statusBar().currentMessage().startswith("Analyse automatique")
    assert window.phases.table.rowCount() >= 1  # the views show the new fits


def test_treatment_tab_and_selection_of_spectra(viewer, compression_run):
    from PySide6.QtCore import QItemSelection, QItemSelectionModel
    from trxrd.domain import Processing

    window, session = viewer.window, viewer.session
    session.attach(compression_run, None)
    panel = window.processing
    assert panel.isEnabled() and panel.filter.currentData() == "svg" and (panel.p1.value(), panel.p2.value()) == (9, 2)
    panel.p1.setValue(15)  # applied to the current spectrum at once
    assert session.run.spectra[0].processing == Processing(1, "svg", (15.0, 2.0)) and window.windowTitle().endswith("*")
    assert session.run.spectra[1].processing == Processing()
    panel.p1.setValue(16)
    assert "impair" in panel.message.text() and session.run.spectra[0].processing.filter_params == (15.0, 2.0)
    panel.filter.setCurrentIndex(panel.filter.findData("fft"))
    assert panel.p1_label.text().startswith("Coupure basse") and session.run.spectra[0].processing.filter_type == "fft"
    panel.filter.setCurrentIndex(panel.filter.findData("svg"))
    panel.p1.setValue(15)

    table = window.navigator.table  # Shift + click from spectrum 2 to spectrum 4
    rows = QItemSelection(table.model().index(2, 0), table.model().index(4, table.columnCount() - 1))
    table.selectionModel().select(rows, QItemSelectionModel.SelectionFlag.ClearAndSelect)
    assert session.selection == [2, 3, 4] and session.index == 2 and panel.selected.text().endswith("(3)")
    panel.p1.setValue(13)  # the current spectrum (2)...
    panel.selected.click()  # ...then the selection
    assert [s.processing.filter_params for s in session.run.spectra] == [(15.0, 2.0), (9, 2), (13.0, 2.0),
                                                                          (13.0, 2.0), (13.0, 2.0), (9, 2)]
    assert panel.message.text() == "Traitement du spectre 2 appliqué à 3 spectre(s)."
    session.select(4)  # inside the selection: it stays
    assert window.navigator.selected_rows() == [2, 3, 4] and session.selection == [2, 3, 4]
    window.actions.get("next").trigger()  # outside: the current spectrum alone
    assert session.index == 5 and window.navigator.selected_rows() == [5] and session.selection == [5]
    panel.all.click()
    assert all(s.processing == Processing(1, "svg", (9, 2)) for s in session.run.spectra)  # that of spectrum 5
    panel.window.setValue(1.0)  # background window for broad peaks
    assert session.spectrum.processing.baseline_window == 1.0
    panel.window.setValue(0.0)
    assert session.spectrum.processing.baseline_window is None and panel.window.text() == "auto"


def test_detection_tab_draws_the_peaks(viewer, compression_run):
    window = viewer.window
    viewer.session.attach(compression_run, None)
    tab = window.detection
    count = tab.table.rowCount()
    assert count >= 3 and len(window.spectrum.detected.data) == count and "pour la recherche" in tab.summary.text()
    assert window.spectrum.threshold.getData()[0] is not None
    tab.min_snr.setValue(60.0)
    assert 0 < tab.table.rowCount() < count and window.windowTitle().endswith("*")
    tab.visible.setChecked(False)
    assert len(window.spectrum.detected.data) == 0 and tab.table.rowCount() > 0  # the table stays
    tab.visible.setChecked(True)
    window.actions.get("raw").trigger()
    assert len(window.spectrum.detected.data) == tab.table.rowCount()

    from trxrd.services import analysis
    tab.broad_fwhm.setValue(0.0)  # liquids: broad peaks count in the search
    assert analysis.saved_settings(viewer.session.run).broad_fwhm is None and tab.broad_fwhm.text() == "jamais"
    tab.broad_fwhm.setValue(0.8)
    assert analysis.saved_settings(viewer.session.run).broad_fwhm == 0.8


def test_phases_tab_and_undo(viewer, compression_run):
    window, session = viewer.window, viewer.session
    ice = "H2O_Ice_VII"
    session.attach(compression_run, None)
    session.select(4)
    panel, undo, redo = window.phases, window.actions.get("undo"), window.actions.get("redo")
    assert panel.card.count() == len(compression_run.phases) and not undo.isEnabled() and not redo.isEnabled()
    panel.card.setCurrentIndex(panel.card.findData(ice))
    assert panel.add.isEnabled() and not panel.remove.isEnabled()
    panel.pressure.setValue(3.05)
    assert session.spectrum.phases == []  # not on the spectrum: it only chooses the starting pressure
    panel.add.click()
    assert not panel.add.isEnabled() and panel.remove.isEnabled() and panel.table.rowCount() == 1
    assert len(window.spectrum._ticks) == 2 and undo.text() == f"&Annuler : ajout de {ice}"
    panel.pressure.setValue(3.1)  # on the spectrum: its lines follow
    assert session.spectrum.phases[0].pressure == 3.1
    window.actions.get("fit").trigger()
    assert session.spectrum.phases[0].pressure == pytest.approx(3.1, abs=0.03)
    assert panel.pressure.value() == pytest.approx(session.spectrum.phases[0].pressure, abs=0.005)

    undo.trigger()
    assert redo.isEnabled() and session.spectrum.phases[0].pressure == 3.1 and panel.pressure.value() == 3.1
    assert window.statusBar().currentMessage() == "Annulé : ajustement du spectre"
    redo.trigger()

    panel.follow.setChecked(False)
    panel.fit_all.click()  # each spectrum from its own model: only spectrum 4 has one
    assert [bool(s.phases) for s in session.run.spectra] == [False] * 4 + [True, False]
    panel.follow.setChecked(True)
    window.actions.get("copy_model").trigger()
    assert panel.paste.isEnabled()
    session.select(3)
    session.set_selection([3, 5])  # the current spectrum is always in the selection of the list
    window.actions.get("paste_model").trigger()
    assert session.run.spectra[5].phases and panel.fit_selection.text() == "Ajuster la sélection (2)"
    panel.fit_selection.click()
    assert session.run.spectra[5].phases[0].pressure == pytest.approx(3.7, abs=0.03)
    panel.clear_model.click()
    assert not session.run.spectra[5].phases
    panel.table.selectRow(0)
    assert panel.card.currentData() == ice
    panel.remove.click()
    assert session.spectrum.phases == [] and panel.table.rowCount() == 0


def test_expected_lines_stay_in_sight_when_zooming(viewer):
    window = viewer.window
    viewer.main.open(str(viewer.path))
    viewer.session.select(2)  # ice VII and ice VI
    view = window.spectrum
    box = view.band.getViewBox()
    assert len(view._ticks) == 4 and all(item.getViewBox() is box for item in view._ticks)
    assert view.band.maximumHeight() == view.AXIS_HEIGHT + 2 * view.ROW_HEIGHT  # one row per phase
    assert box.viewRange()[1] == [0.0, 2.0]

    view.plot.setYRange(0, 1)  # on the small peaks: the band keeps its own scale
    assert box.viewRange()[1] == [0.0, 2.0]
    view.plot.setXRange(15, 17, padding=0)  # but it shares the x axis
    assert box.viewRange()[0] == pytest.approx([15, 17])

    labels = [label.text for sample, label in view.legend.items]
    assert "H2O_Ice_VII  4.00 GPa" in labels and "fond" not in labels  # no background drawn: not in the legend
    window.actions.get("raw").trigger()
    assert "fond" in [label.text for sample, label in view.legend.items]
    window.actions.get("raw").trigger()
    assert sum("GPa" in label.text for sample, label in view.legend.items) == 2  # not repeated at each refresh

    assert view._guides and all(guide.isVisible() for guide in view._guides)
    window.actions.get("guides").trigger()  # Ctrl+G
    assert not any(guide.isVisible() for guide in view._guides)
    viewer.session.select(1)
    assert view._guides and not any(guide.isVisible() for guide in view._guides)  # stays off on the next spectrum
    window.actions.get("guides").trigger()
    assert all(guide.isVisible() for guide in view._guides)


def test_zooming_out_stops_at_the_natural_limits(viewer):
    window = viewer.window
    viewer.main.open(str(viewer.path))

    def zoomed_out(plot):
        plot.getViewBox().scaleBy((50, 50))
        return plot.getViewBox().viewRange()

    x, y = zoomed_out(window.spectrum.plot)
    assert x[0] >= 8.0 - 0.6 and x[1] <= 25.0 + 0.6  # the pattern and 3 % of its width
    assert zoomed_out(window.spectrum.band)[0] == pytest.approx(x)  # linked plots stay with it
    assert window.spectrum.band.getViewBox().viewRange()[1] == [0.0, 1.0]

    x = zoomed_out(window.evolution.plot)[0]
    assert x[0] >= -0.2 and x[1] <= 4.2  # ms, the spectra
    assert zoomed_out(window.map.plot)[0] == pytest.approx(x)
    assert zoomed_out(window.dpdt.plot)[0] == pytest.approx(x)
    window.evolution.span.setCurrentIndex(1)  # the whole oscilloscope record is allowed then
    x = zoomed_out(window.evolution.plot)[0]
    assert x[1] > 900 and window.map.plot.getViewBox().viewRange()[0] == pytest.approx(x)


def test_image_limits_with_locked_proportions(viewer):
    """In a wide window the whole image stays visible: strict limits would crop it."""
    from trxrd.gui.views.images import ImagePanel
    from trxrd.gui.views.main_window import application

    app = application(["test"])
    panel = ImagePanel("test", histogram=False)
    panel.resize(900, 300)
    panel.show()
    app.processEvents()  # the plot gets its real size
    box = panel.plot.getViewBox()
    box.setAspectLocked(True)
    panel.set_image(np.arange(10000.0).reshape(100, 100))
    panel.plot.enableAutoRange()
    app.processEvents()
    box.scaleBy((50, 50))
    app.processEvents()
    x, y = box.viewRange()
    assert y[0] <= 0.5 and y[1] >= 99.5 and y[1] - y[0] < 110 and x[1] - x[0] > 100  # all of it, not more than needed
    panel.clear()
    assert box.state["limits"]["xLimits"] == [None, None]
    panel.close()


def test_long_texts_do_not_widen_the_window(viewer):
    """The window left a screen of 2560 pixels: the path in the status bar and the message of the dP/dt plot."""
    window = viewer.window
    width = window.minimumSizeHint().width()
    window.set_status("F:/ESRF/" + "un dossier très long/" * 40)
    window.dpdt.message.setText("dP/dt : pas assez de pressions ajustées (3 spectres au moins par courbe). " * 3)
    window.cake.message.setText("Images brutes introuvables, fichier source déplacé. " * 5)
    assert window.minimumSizeHint().width() == width
    assert window.status_label.toolTip().startswith("F:/ESRF/")  # the whole path stays readable


def test_peaks_with_the_mouse_on_the_spectrum(viewer, compression_run):
    from PySide6.QtCore import QPointF, Qt
    from synthetic import GASKET
    from trxrd.gui.views.main_window import application
    from trxrd.services import editing

    class Click:
        def __init__(self, scene_position, modifiers):
            self.position, self.keys = scene_position, modifiers

        def button(self):
            return Qt.MouseButton.LeftButton

        def modifiers(self):
            return self.keys

        def scenePos(self):
            return self.position

    app = application(["test"])
    window, session = viewer.window, viewer.session
    window.show()
    session.attach(compression_run, None)
    session.select(5)
    window.phases.card.setCurrentIndex(window.phases.card.findData("H2O_Ice_VII"))
    window.phases.pressure.setValue(3.7)
    window.phases.add.click()
    app.processEvents()
    view, panel, delete = window.spectrum, window.peaks, window.actions.get("delete_peak")
    assert view.handles is None and not delete.isEnabled()

    window.actions.get("edit_peaks").trigger()  # Ctrl+E
    assert panel.edit.isChecked() and delete.isEnabled() and len(view.handles) == 2
    assert panel.table.rowCount() == 2
    view.plot.setYRange(0, 5000)
    app.processEvents()
    box = view.plot.getPlotItem().vb
    where = box.mapViewToScene(QPointF(GASKET, 1000.0))
    view._clicked(Click(where, Qt.KeyboardModifier.NoModifier))  # a plain click does nothing
    assert len(view.handles) == 2
    view._clicked(Click(where, Qt.KeyboardModifier.ControlModifier))
    assert len(view.handles) == 3 and panel.table.rowCount() == 3 and panel.table.currentRow() == 2
    assert window.actions.get("undo").text() == "&Annuler : pic ajouté"

    handle = view.handles[0]
    handle.setPos(handle.pos().x() + 0.02, 1500.0)  # dragged
    handle.sigPositionChangeFinished.emit(handle)
    app.processEvents()
    first = session.spectrum.phases[0].peaks[0]
    assert first.height == 1500.0 and viewer.peaks.selected == ("H2O_Ice_VII", 0) and len(view.handles) == 3

    view._handle_clicked((None, 0))
    app.processEvents()
    assert viewer.peaks.selected == (None, 0) and panel.table.selectionModel().selectedRows()[0].row() == 2
    delete.trigger()  # Suppr
    assert session.spectrum.free_peaks == [] and len(view.handles) == 2
    window.actions.get("undo").trigger()
    assert len(session.spectrum.free_peaks) == 1

    panel.table.selectRow(1)  # chosen in the table
    assert viewer.peaks.selected == ("H2O_Ice_VII", 1)
    panel.toggle.click()
    assert not session.spectrum.phases[0].peaks[1].enabled
    window.actions.get("fit").trigger()
    assert session.spectrum.phases[0].pressure == pytest.approx(3.7, abs=0.05)  # the (110) line alone

    panel.edit.setChecked(False)
    assert view.handles is None and not delete.isEnabled() and not window.actions.get("edit_peaks").isChecked()
    assert editing.peak_refs(session.spectrum)  # the peaks stay


def test_custom_fit_of_a_peak_in_the_window(viewer, compression_run):
    from trxrd.services import analysis

    window, session = viewer.window, viewer.session
    session.attach(compression_run, None)
    session.select(5)
    window.phases.card.setCurrentIndex(window.phases.card.findData("H2O_Ice_VII"))
    window.phases.pressure.setValue(3.7)
    window.phases.add.click()
    view, panel = window.spectrum, window.peaks
    assert len(view._fills) == 2 and not panel.fit.isEnabled() and panel.title.text() == "Aucun pic choisi"
    assert not view.residual.isHidden()
    window.actions.get("residual").trigger()  # Ctrl+D
    assert view.residual.isHidden()
    window.actions.get("residual").trigger()
    window.actions.get("fills").trigger()  # Ctrl+P
    assert not any(item.isVisible() for item in view._fills)
    window.actions.get("fills").trigger()

    panel.table.selectRow(0)
    assert panel.title.text().startswith("H2O_Ice_VII (110)") and panel.fit.isEnabled()
    assert panel.zoom_data.getData()[0] is not None and panel.zoom_left.isVisible()
    assert panel.model.currentText() == "PearsonIV" and panel.shape_labels[0].text() == "m"
    assert not panel.shapes[1].isHidden()
    panel.width_box.setValue(0.05)
    assert session.spectrum.phases[0].peaks[0].width == 0.05
    panel.shapes[0].setValue(1.8)
    assert session.spectrum.phases[0].peaks[0].shape[0] == 1.8
    panel.model.setCurrentText("Gaussian")
    assert session.spectrum.phases[0].peaks[0].model == "Gaussian"
    assert panel.shapes[0].isHidden() and panel.shape_labels[0].isHidden()
    panel.model.setCurrentText("PseudoVoigt")
    assert panel.shape_labels[0].text().startswith("α") and panel.shapes[1].isHidden()

    window.actions.get("fit_peak").trigger()  # Maj+R
    assert np.isfinite(session.spectrum.phases[0].peaks[0].center_err) and panel.message.text() == ""
    panel.fit_window.setValue(8.0)
    panel.variation.setValue(120.0)
    kept = analysis.saved_settings(session.run)
    assert kept.fit_window == 8.0 and kept.fit_variation == pytest.approx(1.2)
    assert panel.variation.value() == 120.0  # shown back

    panel.select_row(None)
    viewer.peaks.selected = None
    viewer.peaks.refresh()
    panel.model.setCurrentText("Moffat")
    assert analysis.saved_settings(session.run).peak_model == "Moffat" and panel.model.currentText() == "Moffat"


def test_lines_of_the_phase_and_keys_of_the_v1(viewer, compression_run):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QKeySequence
    from synthetic import GASKET
    from trxrd.gui.views.main_window import application

    class Click:
        def __init__(self, position):
            self.position = position

        def button(self):
            return Qt.MouseButton.LeftButton

        def modifiers(self):
            return Qt.KeyboardModifier.NoModifier

        def scenePos(self):
            return self.position

    app = application(["test"])
    window, session = viewer.window, viewer.session
    window.show()
    session.attach(compression_run, None)
    session.select(5)
    phases, view = window.phases, window.spectrum
    phases.card.setCurrentIndex(phases.card.findData("H2O_Ice_VII"))
    assert phases.lines.count() == 0  # not on the spectrum yet
    phases.pressure.setValue(3.7)
    window.actions.get("add_phase").trigger()  # Maj+A
    assert session.spectrum.phases[0].phase == "H2O_Ice_VII" and phases.lines.count() == 2
    assert phases.lines.item(1).text().startswith("200") and phases.lines.item(1).checkState() == Qt.CheckState.Checked

    phases.lines.item(1).setCheckState(Qt.CheckState.Unchecked)  # a box of the lines
    assert [p.hkl for p in session.spectrum.phases[0].peaks] == [(1, 1, 0)]
    assert phases.lines.item(1).checkState() == Qt.CheckState.Unchecked
    phases.lines.item(1).setCheckState(Qt.CheckState.Checked)
    assert len(session.spectrum.phases[0].peaks) == 2 and window.peaks.table.rowCount() == 2

    window.actions.get("edit_peaks").trigger()
    view.plot.setYRange(0, 5000)
    app.processEvents()
    box = view.plot.getPlotItem().vb
    view._clicked(Click(box.mapViewToScene(QPointF(GASKET, 1000.0))))  # a click puts the cursor
    assert view.cross_x.isVisible() and view.cross_x.value() == pytest.approx(GASKET, abs=0.05)
    assert session.spectrum.free_peaks == []
    window.actions.get("confirm_peak").trigger()  # C
    assert len(session.spectrum.free_peaks) == 1
    view._clicked(Click(box.mapViewToScene(QPointF(GASKET + 0.1, 800.0))))
    window.actions.get("replace_peak").trigger()  # R
    assert session.spectrum.free_peaks[0].center == pytest.approx(GASKET + 0.1, abs=0.05)
    assert session.spectrum.free_peaks[0].height == pytest.approx(800.0, rel=0.05)
    window.actions.get("delete_peak").trigger()  # U
    assert session.spectrum.free_peaks == []
    window.actions.get("edit_peaks").trigger()
    assert not view.cross_x.isVisible()

    window.actions.get("remove_phase").trigger()  # D
    assert session.spectrum.phases == [] and phases.lines.count() == 0
    keys = {key: window.actions.get(key).shortcuts() for key in ("fit", "fit_selection", "fit_peak", "copy_model", "paste_model")}
    assert QKeySequence("Shift+F") in keys["fit"] and QKeySequence("Shift+T") in keys["fit_selection"]
    assert QKeySequence("Shift+R") in keys["fit_peak"] and QKeySequence("Ctrl+C") in keys["copy_model"]
    assert QKeySequence("Ctrl+V") in keys["paste_model"]
    window.actions.get("show_detection").trigger()  # M
    assert window.tabs.currentWidget() is window.detection


def test_the_oscilloscope_channel_chosen_stays(viewer):
    window, session = viewer.window, viewer.session
    viewer.main.open(str(viewer.path))
    view = window.evolution
    assert view.channel.currentText() == "Channel3"  # by default the channel that is not the trigger
    for choice in ("Channel2", "aucun", "Channel3"):
        view.channel.setCurrentText(choice)
        session.modified()  # any refresh
        session.select(1)
        assert view.channel.currentText() == choice and view.wanted_channel == choice
    view.channel.setCurrentText("aucun")
    session.modified()
    assert len(view.trace.getData()[0] if view.trace.getData()[0] is not None else []) == 0  # nothing drawn

    view.channel.setCurrentText("Channel2")
    window.actions.get("close").trigger()  # the choice survives the run
    assert view.channel.count() == 0
    viewer.main.open(str(viewer.path))
    assert view.channel.currentText() == "Channel2"
    view.wanted_channel = "ch9"  # a run without that channel: the default, and the choice is kept for the next run
    session.modified()
    assert view.channel.currentText() == "Channel3" and view.wanted_channel == "ch9"


def _shown(viewer, compression_run):
    """The window on screen (offscreen), with the compression run: the mouse needs real geometry."""
    window = viewer.window
    viewer.session.attach(compression_run, None)
    window.resize(1600, 950)
    window.show()
    QApplication.processEvents()
    return window


def _point(plot, x):
    """Point of the viewport at x (data coordinates), at mid height."""
    box = plot.getPlotItem().vb
    low, high = box.viewRange()[1]
    return plot.mapFromScene(box.mapViewToScene(QPointF(x, (low + high) / 2)))


def _mouse(plot, kind, point, buttons, modifiers):
    """A mouse event with its button held: QTest.mouseMove moves the cursor, and pyqtgraph would see no drag."""
    viewport = plot.viewport()
    left = Qt.MouseButton.LeftButton
    button = Qt.MouseButton.NoButton if kind == QEvent.Type.MouseMove else left
    position = QPointF(point)
    QApplication.sendEvent(viewport, QMouseEvent(kind, position, QPointF(viewport.mapToGlobal(point)), button, buttons,
                                                 modifiers))
    QApplication.processEvents()


def _settle():
    for _ in range(3):  # the callbacks wait for the end of the mouse event
        QApplication.processEvents()


def _drag(plot, start, stop, modifiers=Qt.KeyboardModifier.NoModifier):
    """A real left drag on `plot`, from x = start to x = stop."""
    left, none = Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton
    points = [_point(plot, start + k * (stop - start) / 4) for k in range(5)]
    _mouse(plot, QEvent.Type.MouseMove, points[0], none, modifiers)
    _mouse(plot, QEvent.Type.MouseButtonPress, points[0], left, modifiers)
    for point in points[1:]:
        _mouse(plot, QEvent.Type.MouseMove, point, left, modifiers)
    _mouse(plot, QEvent.Type.MouseButtonRelease, points[-1], none, modifiers)
    _settle()


def _click(plot, x, modifiers=Qt.KeyboardModifier.NoModifier):
    left, none = Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton
    point = _point(plot, x)
    _mouse(plot, QEvent.Type.MouseMove, point, none, modifiers)
    _mouse(plot, QEvent.Type.MouseButtonPress, point, left, modifiers)
    _mouse(plot, QEvent.Type.MouseButtonRelease, point, none, modifiers)
    _settle()


def test_zones_with_the_mouse_on_the_spectrum(viewer, compression_run):
    from trxrd.services import analysis

    window = _shown(viewer, compression_run)
    plot = window.spectrum.plot
    shift = Qt.KeyboardModifier.ShiftModifier

    before = plot.getPlotItem().vb.viewRange()[0]
    _drag(plot, 17.0, 18.0, shift)  # not in the zone mode: the view moves, nothing is drawn
    assert analysis.saved_settings(compression_run).exclude == []
    assert plot.getPlotItem().vb.viewRange()[0] != pytest.approx(before)
    plot.getPlotItem().vb.setXRange(*before, padding=0)

    window.actions.get("edit_zones").trigger()  # the key Z
    assert window.zones.edit.isChecked() and window.tabs.currentWidget() is window.zones
    assert window.actions.get("delete_peak").isEnabled()
    assert all(line.isVisible() for line in window.spectrum.fit_edges)
    _drag(plot, 18.4, 18.8, shift)
    (low, high), = analysis.saved_settings(compression_run).exclude
    assert (low, high) == pytest.approx((18.4, 18.8), abs=0.05)
    assert window.zones.table.rowCount() == 1 and window.zones.remove.isEnabled()
    assert len(window.spectrum._excluded) == 1 and not window.spectrum.sketch.isVisible()

    view = plot.getPlotItem().vb.viewRange()[0]
    _drag(plot, 11.0, 20.0, shift | Qt.KeyboardModifier.ControlModifier)
    spectrum = viewer.session.spectrum
    assert spectrum.fit_range == pytest.approx((11.0, 20.0), abs=0.05)
    assert len(analysis.saved_settings(compression_run).exclude) == 1
    assert plot.getPlotItem().vb.viewRange()[0] == pytest.approx(view)  # drawing does not move the view
    assert all(item.isVisible() for item in window.spectrum.outside)  # grey on both sides
    assert window.zones.low.value() == pytest.approx(spectrum.fit_range[0], abs=1e-3)

    window.spectrum.fit_edges[1].setValue(19.0)  # what a drag does, then its end
    window.spectrum.fit_edges[1].sigPositionChangeFinished.emit(window.spectrum.fit_edges[1])
    QApplication.processEvents()
    assert spectrum.fit_range[1] == pytest.approx(19.0)

    _click(plot, 18.6)  # chooses the zone...
    assert viewer.zones.chosen == 0 and window.zones.table.selectionModel().selectedRows()
    window.actions.get("delete_peak").trigger()  # ...and Del removes it
    assert analysis.saved_settings(compression_run).exclude == [] and window.spectrum._excluded == []
    window.actions.get("undo").trigger()
    assert len(analysis.saved_settings(compression_run).exclude) == 1 and len(window.spectrum._excluded) == 1

    window.actions.get("edit_zones").trigger()
    assert not window.zones.edit.isChecked() and not window.actions.get("delete_peak").isEnabled()
    assert not any(line.isVisible() for line in window.spectrum.fit_edges)
    assert len(window.spectrum._excluded) == 1 and not window.spectrum._excluded[0].movable  # shown, not movable


def test_an_excluded_zone_dragged_with_the_mouse(viewer, compression_run):
    from trxrd.services import analysis

    window = _shown(viewer, compression_run)
    window.zones.edit.setChecked(True)
    viewer.zones.drawn(18.0, 18.4)
    QApplication.processEvents()
    _drag(window.spectrum.plot, 18.2, 18.7)
    (low, high), = analysis.saved_settings(compression_run).exclude
    assert (low, high) == pytest.approx((18.5, 18.9), abs=0.05)


def test_delete_still_removes_a_peak_with_both_modes(viewer, compression_run):
    from trxrd.services import editing

    window = _shown(viewer, compression_run)
    editing.add_phase(compression_run, 0, "H2O_Ice_VI", 1.0)
    viewer.session.modified()
    window.peaks.edit.setChecked(True)
    window.zones.edit.setChecked(True)
    window.peaks.edit.setChecked(False)
    assert window.actions.get("delete_peak").isEnabled()  # the zones still use the key
    window.zones.edit.setChecked(False)
    assert not window.actions.get("delete_peak").isEnabled()
    window.peaks.edit.setChecked(True)
    viewer.peaks.pick(("H2O_Ice_VI", 0))
    count = len(compression_run.spectra[0].phases[0].peaks)
    window.actions.get("delete_peak").trigger()
    assert len(compression_run.spectra[0].phases[0].peaks) == count - 1


@pytest.mark.parametrize("plot_of", [lambda w: w.evolution.plot, lambda w: w.dpdt.plot, lambda w: w.map.plot],
                         ids=["evolution", "dpdt", "map"])
def test_zone_of_spectra_drawn_on_a_ddac_plot(viewer, compression_run, plot_of):
    window = _shown(viewer, compression_run)
    assert not window.dpdt.toggle.isChecked()
    _drag(plot_of(window), 0.6, 3.4, Qt.KeyboardModifier.ShiftModifier)
    assert viewer.session.selection == [1, 2, 3] and window.dpdt.toggle.isChecked()
    assert window.navigator.selected_rows() == [1, 2, 3]
    for view in (window.evolution, window.dpdt, window.map):
        assert view.interval.region.isVisible()
        assert view.interval.region.getRegion() == pytest.approx((0.6, 3.4), abs=0.1)


@pytest.mark.parametrize("plot_of", [lambda w: w.evolution.plot, lambda w: w.dpdt.plot, lambda w: w.map.plot],
                         ids=["evolution", "dpdt", "map"])
def test_zone_of_spectra_dragged_with_the_mouse(viewer, compression_run, plot_of):
    """On the evolution plot the oscilloscope has its own view box, over the one of the zone."""
    window = _shown(viewer, compression_run)
    viewer.session.set_selection([1, 2])
    window.dpdt.toggle.setChecked(True)
    assert window.evolution.interval.region.getRegion() == pytest.approx((0.5, 2.5))
    view = window.evolution.plot.getPlotItem().vb.viewRange()[0]
    _drag(plot_of(window), 1.5, 3.5)  # its body, two spectra further
    assert viewer.session.selection == [3, 4] and window.navigator.selected_rows() == [3, 4]
    assert window.map.interval.region.getRegion() == pytest.approx((2.5, 4.5), abs=0.1)
    assert window.evolution.plot.getPlotItem().vb.viewRange()[0] == pytest.approx(view)  # the plot stays
    _drag(plot_of(window), 4.5, 5.5)  # its right edge
    assert viewer.session.selection == [3, 4, 5]


def test_zone_of_spectra_survives_browsing_in_the_real_list(viewer, compression_run):
    window = _shown(viewer, compression_run)
    _drag(window.evolution.plot, 0.6, 3.4, Qt.KeyboardModifier.ShiftModifier)
    window.actions.get("last").trigger()
    assert viewer.session.index == 5 and viewer.session.selection == [1, 2, 3]
    assert window.navigator.selected_rows() == [1, 2, 3]
    window.navigator.table.selectRow(0)  # a click on one row of the list
    QApplication.processEvents()
    assert viewer.session.index == 0 and viewer.session.selection == [1, 2, 3]
    assert window.navigator.selected_rows() == [1, 2, 3]
    window.dpdt.toggle.setChecked(False)
    window.actions.get("next").trigger()  # no zone any more: the list works as before
    assert viewer.session.index == 1
    window.actions.get("last").trigger()
    assert viewer.session.selection == [5] and window.navigator.selected_rows() == [5]


def test_the_zone_is_seen_over_the_map(viewer, compression_run):
    window = _shown(viewer, compression_run)
    assert window.map.interval.region.zValue() > window.map.image.zValue()
