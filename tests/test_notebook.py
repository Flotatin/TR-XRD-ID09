import numpy as np
import pytest

from trxrd import XRDRun
from trxrd.io.h5store import save_run


@pytest.fixture
def saved(tmp_path, sample_run):
    return save_run(sample_run, tmp_path / "water02_0001_scan0027.trxrd.h5")


def test_summary_has_legacy_columns(saved):
    run = XRDRun(saved)
    summary = run.summary
    assert len(run) == len(summary) == 5
    np.testing.assert_allclose(summary["P_H2O_Ice_VII"], [2.0, 3.0, 4.0, np.nan, np.nan])
    np.testing.assert_allclose(summary["V_H2O_Ice_VII"][:3], [36.0, 37.0, 38.0])
    assert summary["State_H2O_Ice_VI"][2] == "in_noise"
    np.testing.assert_allclose(summary["time"], np.arange(5) * 1e-3)


def test_pressure_and_peaks(saved):
    run = XRDRun(saved)
    pressure = run.pressure(with_errors=True)
    assert list(pressure.columns) == ["H2O_Ice_VII", "H2O_Ice_VII_err", "H2O_Ice_VI", "H2O_Ice_VI_err"]
    assert pressure.index.name == "time"
    peaks = run.peaks
    assert len(peaks) == 7
    assert peaks[peaks["name"] == "Ice7_pD200"]["enabled"].tolist() == [False] * 3
    assert {"time", "center", "center_err", "h", "k", "l"} <= set(peaks.columns)


def test_spectrum_and_arrays(saved):
    run = XRDRun(saved)
    first = run.spectrum(0)
    np.testing.assert_allclose(first["corrected"], first["intensity"] * 0.5)
    assert "model" in first
    assert run.intensity.shape == (5, 400)
    assert run.oscilloscope.columns.tolist() == ["time", "Channel2", "Channel3"]
    assert run.phase_card("H2O_Ice_VII").k0 == 21.23
    assert run.to_run().note.startswith("Pressure ramp")


def test_find_and_plots(saved, tmp_path):
    (tmp_path / "not_a_run.h5").write_bytes(b"")
    runs = XRDRun.find(tmp_path, "*water*")
    assert [r.name for r in runs] == ["water02_0001_scan0027"]
    run = runs[0]
    run.plot_spectrum(0)
    run.plot_pressure()
    run.plot_map()
    assert "5 spectra" in repr(run)
