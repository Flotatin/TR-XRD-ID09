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
    assert window.peaks.rowCount() == 2 and window.map.image.image.shape == (400, 5)  # [2θ, time]
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
