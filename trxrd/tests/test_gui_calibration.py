"""Calibration dialog: presenter with a fake view (no Qt), and the real dialog offscreen."""

import h5py
import numpy as np
import pytest

pytest.importorskip("pyFAI")

from test_calibration_engine import TRUE, _click, make_lab6  # noqa: E402
from test_gui_presenters import Recorder  # noqa: E402
from trxrd.gui.presenters.calibration import CalibrationPresenter  # noqa: E402
from trxrd.services.calibration import save_poni  # noqa: E402


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    import dataclasses
    import fabio

    folder = tmp_path_factory.mktemp("calibration")
    image, mask, two_theta = make_lab6()
    with h5py.File(folder / "scan_jf1m_0000.h5", "w") as f:  # a LIMA-like stack of 2 frames
        f["entry_0000/measurement/data"] = np.stack([image, image]).astype(np.float32)
    fabio.edfimage.EdfImage(data=mask.astype(np.uint8)).write(str(folder / "det.edf"))
    start = dataclasses.replace(TRUE, dist=TRUE.dist * 1.05, poni1=TRUE.poni1 + 10 * 75e-6, rot1=0.0, rot2=0.0)
    save_poni(start, folder / "start.poni")
    return folder, two_theta, mask


def test_presenter_click_calibrate_save(files):
    folder, two_theta, mask = files
    view = Recorder(ask_save_path=str(folder / "new.poni"))
    presenter = CalibrationPresenter(view, poni_file=str(folder / "start.poni"), mask_file=str(folder / "det.edf"))
    assert view.last("show_start")[0] == pytest.approx(TRUE.dist * 1.05e3)
    view.on_click(10, 10)  # no image yet
    assert "Charger d'abord" in view.last("show_message")[0]

    presenter.load_image(str(folder / "scan_jf1m_0000.h5"))
    assert view.last("show_image")[0].shape == (1064, 1030)
    assert view.last("show_rings")[0].max() > 0 and view.last("show_ring_choices")[0][0].startswith("0 : 9.0")
    for ring in (0, 2):
        presenter.set_ring(ring)
        view.on_click(*_click(two_theta, mask, ring))
        assert "points trouvés sur l'anneau" in view.last("show_message")[0]
    assert set(view.last("show_points")[1].tolist()) == {0, 2}
    presenter.calibrate()
    text = view.last("show_geometry")[0]
    assert "distance 165.2" in text and "anneaux" in text
    assert view.last("set_busy") == (None,)

    path = presenter.save()
    assert path == folder / "new.poni" and path.exists() and presenter.mask_file.endswith("det.edf")
    presenter.clear_points()
    assert len(view.last("show_points")[0]) == 0


def test_presenter_guess_energy_calibrant_and_errors(files, tmp_path):
    folder, two_theta, mask = files
    view = Recorder()
    presenter = CalibrationPresenter(view)
    presenter.set_guess(160.0, 1200.0, 500.0, 19.0)
    distance, col, row, energy = view.last("show_start")
    assert distance == pytest.approx(160.0) and (col, row) == pytest.approx((1200.0, 500.0)) and energy == pytest.approx(19.0)
    presenter.set_energy(20.0)
    assert view.last("show_start")[3] == pytest.approx(20.0)
    presenter.set_calibrant("CeO2")
    with pytest.raises(ValueError):
        presenter.set_calibrant("Si")
    (tmp_path / "bad.poni").write_text("not a poni")
    presenter.load_poni(str(tmp_path / "bad.poni"))
    assert view.last("show_start")[0] == pytest.approx(160.0)  # unchanged
    presenter.load_image(str(tmp_path / "missing.h5"))
    assert view.last("set_busy") == (None,) and "Error" in view.last("show_message")[0]


def test_calibration_dialog_offscreen(files, tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings
    from trxrd.gui.presenters.viewer import ImmediateRunner
    from trxrd.gui.views.calibration import CalibrationDialog
    from trxrd.gui.views.main_window import application

    folder, two_theta, mask = files
    application(["test"])
    dialog = CalibrationDialog(settings=QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat))
    dialog.presenter = CalibrationPresenter(dialog, ImmediateRunner(), str(folder / "start.poni"), str(folder / "det.edf"))
    dialog.presenter.load_image(str(folder / "scan_jf1m_0000.h5"))
    assert dialog.image.image.shape == (1064, 1030) and dialog.rings.image.shape == (1064, 1030, 4)
    assert dialog.ring.count() >= 6 and dialog.distance.value() == pytest.approx(TRUE.dist * 1.05e3, abs=1e-3)
    dialog.ring.setCurrentIndex(2)
    assert dialog.presenter.ring == 2
    row, col = _click(two_theta, mask, 2)
    dialog.on_click(row, col)
    assert len(dialog.points.data) > 10
    dialog.fixed["rot1"].setChecked(True)
    assert "rot1" in dialog.presenter.fix
    dialog.presenter.calibrate()
    assert "distance 165.2" in dialog.result.text()
