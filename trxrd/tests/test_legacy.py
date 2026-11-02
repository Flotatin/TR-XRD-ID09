"""Conversion of the old dill files. Needs neither the old program nor dill/lmfit:
data/legacy_sample.CEDX was produced once by data/make_legacy_sample.py."""

import io
import os
import pickle
import subprocess
import sys
import textwrap
import types
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import DATA, ROOT
from trxrd import XRDRun
from trxrd.io.legacy import LegacyObject, _LegacyUnpickler, convert_file, is_legacy_file, legacy_to_run, load_legacy

SAMPLE = DATA / "legacy_sample.CEDX"
EXPECTED = np.load(DATA / "legacy_sample_expected.npz")


def test_convert_keeps_data_and_results(tmp_path):
    run = XRDRun(convert_file(SAMPLE, tmp_path / "out.trxrd.h5"))

    assert run.name == "legacy_sample" and len(run) == 16
    spectrum = run.spectrum(0)
    np.testing.assert_allclose(spectrum["intensity"], EXPECTED["intensity"])
    np.testing.assert_allclose(spectrum["baseline"], EXPECTED["baseline"])
    # fitted curve rebuilt with the new profiles == the one lmfit computed in the old program
    np.testing.assert_allclose(spectrum["model"], EXPECTED["model"], atol=1e-9)

    summary = run.summary
    assert summary["P_Ga_I"][0] == pytest.approx(float(EXPECTED["pressure"]))
    assert summary["V_Ga_I"][0] == pytest.approx(float(EXPECTED["volume"]))
    assert np.isnan(summary["P_Ga_I"][2])
    peaks = run.peaks[run.peaks["spectrum"] == 0]
    assert len(peaks) == int(EXPECTED["n_peaks"]) and peaks["has_hkl"].all()
    assert run.phase_card("Ga_I").k0 == float(EXPECTED["k0"])
    assert run.note == "[P_Ga_I] -> 0.2"
    np.testing.assert_allclose(run.time, np.arange(16) * 1e-3)

    full = run.to_run()
    np.testing.assert_array_equal(full.calibration.mask, EXPECTED["mask"])
    assert full.calibration.poni == ""  # the .poni path of the beamline PC is not reachable here
    assert full.calibration.poni_file == "D:/ESRF/old_session/calib.poni"


def test_conversion_needs_no_old_code_nor_dill_nor_lmfit(tmp_path):
    script = textwrap.dedent(
        f"""
        import sys
        for name in ["cedapp", "Bibli_python", "dill", "lmfit", "pyFAI", "fabio", "PyQt5", "PySide6",
                     "pyqtgraph", "matplotlib", "tkinter", "lecroyscope", "pynverse", "peakutils", "deap"]:
            sys.modules[name] = None  # importing any of these now fails
        sys.path.insert(0, {str(ROOT / "src")!r})
        from trxrd.io.legacy import convert_file
        print(convert_file({str(SAMPLE)!r}, {str(tmp_path / "isolated.h5")!r}))
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env={**os.environ, "PYTHONWARNINGS": "ignore"})
    assert result.returncode == 0, result.stderr
    assert XRDRun(tmp_path / "isolated.h5").summary["P_Ga_I"][0] == pytest.approx(float(EXPECTED["pressure"]))


@pytest.mark.parametrize("module", ["CL_FD_Update", "Bibli_python.CL_FD_Update", "src.cedapp.drx.CL_FD_Update"])
def test_any_historical_module_path(module):
    """Classes are matched by nothing but their name: every old import path works."""
    parts = module.split(".")
    packages = [".".join(parts[: i + 1]) for i in range(len(parts))]
    saved = {name: sys.modules.get(name) for name in packages}
    for name in packages:
        sys.modules[name] = types.ModuleType(name)
    fake = sys.modules[module]
    for name in ("CED_DRX", "Spectre"):
        setattr(fake, name, type(name, (), {"__module__": module}))
    try:
        obj = fake.CED_DRX()
        obj.Spectra = [fake.Spectre()]
        obj.Spectra[0].wnb = np.arange(3.0)
        data = pickle.dumps(obj)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    restored = _LegacyUnpickler(io.BytesIO(data)).load()
    assert isinstance(restored, LegacyObject) and type(restored).__name__ == "CED_DRX"
    np.testing.assert_array_equal(restored.Spectra[0].wnb, [0.0, 1.0, 2.0])


def test_spectroscopy_gauges_are_dropped(caplog):
    x = np.linspace(10, 20, 5)
    ruby = SimpleNamespace(name="Ruby", pics=[], P=3.0)
    old = SimpleNamespace(Spectra=[SimpleNamespace(wnb=x, spec=x, Gauges=[ruby])])
    run = legacy_to_run(old, "ruby_run")
    assert run.spectra[0].phases == [] and run.phases == {}
    assert "Ruby" in caplog.text


@pytest.mark.parametrize("time_index", ["Channel2", "None", np.arange(3), None])
def test_trigger_channel_of_any_kind(time_index):
    """Jungfrau "continuous" runs stored an array of frame indices (or "None") instead of a channel name."""
    import pandas as pd

    x = np.linspace(10, 20, 5)
    old = SimpleNamespace(
        Spectra=[SimpleNamespace(wnb=x, spec=x)],
        data_oscillo=pd.DataFrame({"Time": np.arange(4.0), "Channel2": np.ones(4), "Channel3": np.zeros(4)}),
        time_index=time_index,
    )
    oscilloscope = legacy_to_run(old, "continuous").oscilloscope
    assert set(oscilloscope.channels) == {"Channel2", "Channel3"}
    assert oscilloscope.trigger_channel == ("Channel2" if isinstance(time_index, str) and time_index == "Channel2" else "")


@pytest.mark.parametrize("card_rows, expected", [(None, 0.6), ([["VERSION", "4"], ["Vmin", "0.5"]], 0.5)])
def test_fallback_vmin_of_the_old_program_is_not_kept(card_rows, expected):
    """The v1 set Vmin = 0.8 when its inversion failed: ice VII could not go above ~7 GPa."""
    import pandas as pd
    from trxrd.io.legacy import _phase_card

    ice7 = SimpleNamespace(
        Dhkl=pd.DataFrame({"Dhkl": [2.44023], "I": [100.0], "h": [1], "k": [1], "l": [0]}),
        symmetrie="CUBIC", A=3.451, B=None, C=None, ALPHA=90.0, BETA=90.0, GAMMA=90.0, V0=41.4802,
        K0=20.15, K0P=4.0, Vmin=0.8, file=None if card_rows is None else pd.DataFrame(card_rows),
    )
    card = _phase_card("H2O_Ice_VII", ice7)
    assert card.v_min_ratio == expected
    assert np.isfinite(card.two_theta_table(np.array([5.0, 10.0, 14.0]), 19000.0)).all()


def test_legacy_extensions():
    assert is_legacy_file("run24.CEDXUp") and is_legacy_file("a.cedx") and not is_legacy_file("a.h5")
    assert load_legacy(SAMPLE).Spectra[0].E > 0


def test_cli_convert_folder(tmp_path, capsys):
    from trxrd.cli import main

    folder = tmp_path / "Fichier_CED_DRX"
    folder.mkdir()
    (folder / "run1.CEDX").write_bytes(SAMPLE.read_bytes())
    (folder / "broken.CEDX").write_bytes(b"not a pickle")
    assert main(["convert", str(folder), "-o", str(tmp_path / "converted")]) == 1  # one failure reported
    assert (tmp_path / "converted" / "run1.trxrd.h5").exists()
    assert "FAILED" in capsys.readouterr().err
    assert main(["info", str(tmp_path / "converted" / "run1.trxrd.h5")]) == 0


def test_library_gives_the_pressure_range_of_the_run_cards():
    """v1 cards keep their EoS, but Sn beta/gamma get the ranges of the library (hysteresis 8-12 GPa)."""
    from conftest import LIBRARY
    from trxrd.cli import _add_library
    from trxrd.domain import Run

    import dataclasses

    card = legacy_to_run(load_legacy(SAMPLE), "x").phases["Ga_I"]
    run = Run("sn", phases={"Sn_Gamma": dataclasses.replace(card, name="Sn_Gamma", p_range=(9.0, 100.0), k0=1.0)})
    _add_library(run, [LIBRARY / "Sn"])
    assert run.phases["Sn_Gamma"].p_range == (8.0, 100.0) and run.phases["Sn_Gamma"].k0 == 1.0
    assert run.phases["Sn_Beta"].p_range == (-1.0, 12.0)


def test_cli_info_on_a_legacy_file(capsys):
    from trxrd.cli import main

    assert main(["info", str(SAMPLE)]) == 1
    assert "trxrd convert" in capsys.readouterr().err


def test_cli_output_on_a_cp1252_stream(monkeypatch):
    """Windows pipes and redirections are cp1252: the 2θ of the help must not crash the program."""
    from trxrd.cli import main

    stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stdout)
    with pytest.raises(SystemExit) as stop:
        main(["compare", "--help"])
    assert stop.value.code == 0
    stdout.seek(0)
    assert "tolerance (deg)" in stdout.read()


def test_cli_compare_with_the_legacy_analysis(tmp_path, capsys):
    from trxrd.cli import main

    reference = tmp_path / "run.CEDX"
    reference.write_bytes(SAMPLE.read_bytes())
    assert main(["compare", str(reference), "--pmax", "5", "--frames", "0:3"]) == 0
    out = capsys.readouterr().out
    assert "reference: Ga_I" in out and "same phases in" in out
    assert reference.read_bytes() == SAMPLE.read_bytes()  # the reference is never modified
