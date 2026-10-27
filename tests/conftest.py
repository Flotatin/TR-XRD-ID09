import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

TUTO = ROOT / "TUTO"
LIBRARY = ROOT / "resources" / "bibdrx"


@pytest.fixture(scope="session")
def tuto_scan():
    return TUTO / "scan_jf1m_0000.h5"


@pytest.fixture(scope="session")
def tuto_calibration():
    pytest.importorskip("pyFAI")
    from trxrd.io.calibration import calibration_from_files

    return calibration_from_files(
        TUTO / "calib_THT8_correct_samedi.poni", TUTO / "scan_jf1m_0000.mask", two_theta_range=(8.0, 25.0)
    )


@pytest.fixture(scope="session")
def legacy():
    """The old CL_FD_Update module (skipped when its heavy dependencies are missing)."""
    from trxrd.io.legacy import _import_legacy_modules

    warnings.filterwarnings("ignore", category=SyntaxWarning)
    try:
        modules = _import_legacy_modules()
        return modules["CL_FD_Update"]
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"legacy code not importable: {exc}")


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
            two_theta_range=(8.0, 25.0), flip_axis=1, sigma_distance_m=1e-4,
        ),
        phases={ice7.name: ice7, ice6.name: ice6},
        source=Source(raw_file="", raw_dataset="", master_file="/data/x.h5", scan="27.1"),
        oscilloscope=Oscilloscope(time=np.linspace(0, 1, 50), channels={"Channel2": np.ones(50), "Channel3": np.zeros(50)}, trigger_channel="Channel2"),
        note="Pressure ramp — essai n°2 [P_H2O_Ice_VII] -> 0.1",
        ui_state={"zones": [[10, 12]], "theme": "light"},
    )
