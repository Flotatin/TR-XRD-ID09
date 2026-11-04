import h5py
import numpy as np
import pytest

from trxrd import XRDRun
from trxrd.io.esrf import locate_bliss_master
from trxrd.io.h5store import load_run, save_run
from trxrd.services.run_builder import TimeMode, build_run


def test_locate_bliss_master(bliss_dataset):
    master, scan = locate_bliss_master(bliss_dataset)
    assert master.name == "water02_0001.h5" and scan == "3.1"


def test_build_run_from_raw_scan(tmp_path, bliss_dataset, tuto_calibration):
    run = build_run(bliss_dataset, tuto_calibration)
    assert run.name == "water02_0001_scan0003"
    assert len(run) == 3
    first = run.spectra[0]
    # auto sampling: one bin per smallest pixel (0.021° at 8-25° for this geometry)
    assert tuto_calibration.points_per_degree == 48
    assert first.two_theta.size == tuto_calibration.n_points == 816
    assert 8.0 <= first.two_theta[0] and first.two_theta[-1] <= 25.0
    assert first.baseline is not None and np.isfinite(first.intensity).all()
    np.testing.assert_allclose(run.times, [0.002, 0.005, 0.008], atol=1e-5)
    assert run.oscilloscope.trigger_channel == "Channel2"
    assert run.source.scan == "3.1"

    path = save_run(run, tmp_path / "out.h5")
    loaded = load_run(path)
    np.testing.assert_array_equal(loaded.calibration.mask, tuto_calibration.mask)
    assert loaded.calibration.poni == tuto_calibration.poni
    assert XRDRun(path).intensity.shape == (3, 816)
    assert loaded.calibration.points_per_degree == 48


def test_build_run_without_oscilloscope(tuto_scan, tuto_calibration):
    run = build_run(tuto_scan, tuto_calibration, frames=[0, 5], time_mode=TimeMode.PERIOD, frame_period=0.5)
    assert [s.frame for s in run.spectra] == [0, 5]
    np.testing.assert_allclose(run.times, [0.0, 2.5])
    assert run.oscilloscope is None


def test_cli_build(tmp_path, bliss_dataset):
    from trxrd.cli import main
    from conftest import DATA, LIBRARY

    out = tmp_path / "analysis"
    code = main([
        "build", str(bliss_dataset),
        "--poni", str(DATA / "calib_THT8_correct_samedi.poni"), "--mask", str(DATA / "scan_jf1m_0000.mask"),
        "--tth-min", "8", "--tth-max", "25", "--phases", str(LIBRARY / "Ga"), "-o", str(out),
    ])
    assert code == 0
    run = XRDRun(out / "water02_0001_scan0003.trxrd.h5")
    assert len(run) == 3 and "Ga_I" in run.phases


def test_optional_oscilloscope_channel(bliss_dataset, tuto_calibration):
    run = build_run(bliss_dataset, tuto_calibration, frames=[0])
    assert set(run.oscilloscope.channels) == {"Channel2", "Channel3"}  # no ch2 in this scan
    with h5py.File(bliss_dataset.parents[1] / "water02_0001.h5", "a") as master:
        master["3.1/measurement/ch2"] = np.cos(master["3.1/measurement/ch5_time"][()])
    run = build_run(bliss_dataset, tuto_calibration, frames=[0])
    assert set(run.oscilloscope.channels) == {"Channel2", "Channel3", "ch2"}


def test_browse_raw_data(raw_root):
    from trxrd.domain import Run, Source
    from trxrd.services import raw_data

    [dataset] = raw_data.find_datasets(raw_root)
    assert (dataset.sample, dataset.name) == ("water02", "water02_0001") and dataset.master is not None
    scans = raw_data.list_scans(dataset)
    assert [(s.number, s.title, s.oscilloscope, s.image_file is not None) for s in scans] == [
        (1, "ascan piezoy 0 1 40 0.3", False, False), (3, "timescan 0.1 1", True, True)]
    assert [s.label for s in raw_data.list_scans(dataset, images_only=True)] == ["scan0003"]

    old_file = raw_root.parent / "Fichier_CED_DRX" / "water02_scan0003.CEDX"  # where v1 runs live
    image = raw_data.locate_raw_scan(Run("water02_scan0003"), [old_file])
    assert image == scans[1].image_file
    assert raw_data.locate_raw_scan(Run("water02_0001_scan0003"), [raw_root]) == image
    assert raw_data.locate_raw_scan(Run("water02_scan0099"), [old_file]) is None
    assert raw_data.locate_raw_scan(Run("x", source=Source(raw_file=str(image))), []) == image
    assert raw_data.parse_run_name("Water05_0001_scan0009") == ("Water05", "0001", 9)


def test_build_and_save(tmp_path, raw_root):
    from conftest import DATA, LIBRARY
    from trxrd.services.run_builder import Cancelled, build_and_save

    scan = raw_root / "water02" / "water02_0001" / "scan0003" / "scan_jf1m_0000.h5"
    steps = []
    path = build_and_save(scan, tmp_path / "analyse", DATA / "calib_THT8_correct_samedi.poni", DATA / "scan_jf1m_0000.mask",
                          two_theta_range=(8, 25), phase_paths=[LIBRARY / "Ga"], progress=lambda d, n: steps.append(d) or True)
    assert path == tmp_path / "analyse" / "water02_0001_scan0003.trxrd.h5" and steps == [1, 2, 3]
    run = load_run(path)
    assert len(run) == 3 and "Ga_I" in run.phases and run.source.raw_file == str(scan.resolve())
    with pytest.raises(Cancelled):
        build_and_save(scan, tmp_path / "stopped", DATA / "calib_THT8_correct_samedi.poni", progress=lambda d, n: d < 2)
    assert not (tmp_path / "stopped").exists()


def test_the_treatment_is_kept_whole(tmp_path, bliss_dataset):
    """The background window, for broad peaks, used to be left out."""
    from conftest import DATA
    from trxrd.domain import Processing
    from trxrd.io.calibration import calibration_from_files
    from trxrd.services.run_builder import build_run

    calibration = calibration_from_files(DATA / "calib_THT8_correct_samedi.poni", DATA / "scan_jf1m_0000.mask",
                                         two_theta_range=(8, 25))
    processing = Processing(3, "svg", (11, 3), baseline_window=1.0)
    run = build_run(bliss_dataset, calibration, processing=processing)
    assert run.spectra[0].processing == processing and run.spectra[0].processing is not processing
    assert run.spectra[0].processing is not run.spectra[1].processing
