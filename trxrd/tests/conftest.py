import contextlib
import io
import os
import sys
import types
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

DATA = ROOT / "tests" / "data"
LIBRARY = ROOT / "resources" / "phases"
OLD_PROGRAM = ROOT.parent / "src"  # old program, only present before the repository split


@pytest.fixture(scope="session")
def tuto_scan():
    return DATA / "scan_jf1m_0000.h5"


@pytest.fixture(scope="session")
def tuto_calibration():
    pytest.importorskip("pyFAI")
    from trxrd.io.calibration import calibration_from_files

    return calibration_from_files(
        DATA / "calib_THT8_correct_samedi.poni", DATA / "scan_jf1m_0000.mask", two_theta_range=(8.0, 25.0)
    )


@pytest.fixture(scope="session")
def legacy():
    """The old CL_FD_Update module, for numerical comparisons; skipped once the
    old program is no longer next to the new one."""
    if not (OLD_PROGRAM / "cedapp").is_dir():
        pytest.skip("old program not available")
    sys.path.append(str(OLD_PROGRAM))
    for hardware in ("lecroyscope", "cedapp.drx.Oscilloscope_LeCroy_QTinterface"):
        sys.modules.setdefault(hardware, types.ModuleType(hardware))
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    import matplotlib

    use_backend, matplotlib.use = matplotlib.use, lambda *args, **kwargs: None  # old code forces Qt5Agg
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            from cedapp.drx import CL_FD_Update
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"old program not importable: {exc}")
    finally:
        matplotlib.use = use_backend
    return CL_FD_Update


@pytest.fixture
def sample_run():
    """A synthetic run exercising every part of the format."""
    from trxrd.domain import (
        Calibration, Oscilloscope, Peak, PhaseFit, Processing, Run, Source, Spectrum,
    )
    from trxrd.io.jcpds import read_jcpds

    ice7 = read_jcpds(LIBRARY / "H2O" / "Ice-VII.jcpds")
    ice6 = read_jcpds(LIBRARY / "H2O" / "Ice-VI.jcpds")
    x = np.linspace(8, 25, 400)
    spectra = []
    for i in range(5):
        y = np.random.default_rng(i).random(x.size) + 10
        spectrum = Spectrum(
            two_theta=x,
            intensity=y,
            time=1e-3 * i,
            frame=i,
            label=f"frame_{i}",
            processing=Processing(baseline_degree=i % 3, filter_type="svg", filter_params=(9, 2)),
            baseline=None if i == 4 else y * 0.5,
            model=None if i >= 3 else y * 0.1,
            fit_range=(10.0, 20.0) if i == 1 else None,
        )
        if i < 3:
            spectrum.phases.append(
                PhaseFit(
                    phase=ice7.name, a=3.3, a_err=0.01, volume=36.0 + i, volume_err=0.1,
                    pressure=2.0 + i, pressure_err=0.05,
                    peaks=[
                        Peak("Ice7_pD110", "PearsonIV", 16.3, 0.01, 100.0, 1.0, 0.1, 0.01, (1.1, 0.0), (1, 1, 0)),
                        Peak("Ice7_pD200", "PseudoVoigt", 23.0, 0.02, 20.0, 1.0, 0.1, 0.01, (0.5,), (2, 0, 0), enabled=False),
                    ],
                )
            )
        if i == 2:
            spectrum.phases.append(PhaseFit(phase=ice6.name, state="in_noise", pressure=1.5, peaks=[Peak("free", "Gaussian", 12.0)]))
        spectra.append(spectrum)
    return Run(
        name="water02_0001_scan0027",
        spectra=spectra,
        calibration=Calibration(
            energy_ev=19000.0, poni="Distance: 0.16\nWavelength: 6.5e-11\n",
            mask=(np.random.default_rng(0).random((64, 48)) > 0.9).astype(np.uint8),
            two_theta_range=(8.0, 25.0), points_per_degree=50, flip_axis=1, sigma_distance_m=1e-4,
        ),
        phases={ice7.name: ice7, ice6.name: ice6},
        source=Source(raw_file="", raw_dataset="", master_file="/data/x.h5", scan="27.1"),
        oscilloscope=Oscilloscope(time=np.linspace(0, 1, 50), channels={"Channel2": np.ones(50), "Channel3": np.zeros(50)}, trigger_channel="Channel2"),
        note="Pressure ramp — essai n°2 [P_H2O_Ice_VII] -> 0.1",
        ui_state={"zones": [[10, 12]], "theme": "light"},
    )


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


@pytest.fixture
def raw_root(tmp_path, bliss_dataset):
    """RAW_DATA/water02/water02_0001/... with a titled scan, an alignment scan without images and a Fichier_CED_DRX."""
    import shutil

    raw = tmp_path / "campaign" / "RAW_DATA"
    shutil.copytree(bliss_dataset.parents[1], raw / "water02" / "water02_0001")
    with h5py.File(raw / "water02" / "water02_0001" / "water02_0001.h5", "a") as master:
        master["3.1/title"] = "timescan 0.1 1"
        master["3.1/start_time"] = "2026-01-22T22:18:41"
        master.create_group("1.1/measurement")["piezoy"] = np.zeros(3)
        master["1.1/title"] = "ascan piezoy 0 1 40 0.3"
    (tmp_path / "campaign" / "Fichier_CED_DRX").mkdir()
    return raw
