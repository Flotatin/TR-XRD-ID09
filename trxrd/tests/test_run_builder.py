import h5py
import numpy as np
import pytest

from trxrd import XRDRun
from trxrd.io.esrf import locate_bliss_master
from trxrd.io.h5store import load_run, save_run
from trxrd.services.run_builder import TimeMode, build_run


@pytest.fixture
def bliss_dataset(tmp_path, tuto_scan):
    """Minimal ESRF layout: <ds>/<ds>.h5 (oscilloscope) + <ds>/scan0003/scan_jf1m_0000.h5 (3 frames)."""
    dataset = tmp_path / "water02_0001"
    scan_dir = dataset / "scan0003"
    scan_dir.mkdir(parents=True)
    with h5py.File(tuto_scan, "r") as src, h5py.File(scan_dir / "scan_jf1m_0000.h5", "w") as dst:
        dst.attrs["default"] = "entry_0000"
        entry = dst.create_group("entry_0000")
        entry.attrs["default"] = "plot"
        plot = entry.create_group("plot")
        plot.attrs["signal"] = "data"
        plot.create_dataset("data", data=src["entry_0000/ESRF-ID09/jf1m/data"][:3], compression="gzip")
    t = np.linspace(0, 0.01, 2000)
    trigger = sum(((t > c - 1e-4) & (t < c + 1e-4)).astype(float) for c in (0.002, 0.005, 0.008))
    with h5py.File(dataset / "water02_0001.h5", "w") as master:
        measurement = master.create_group("3.1/measurement")
        measurement["ch5_time"] = t
        measurement["ch5"] = trigger
        measurement["ch6"] = np.sin(t)
    return scan_dir / "scan_jf1m_0000.h5"


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
