import contextlib
import io
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import LIBRARY, TUTO
from trxrd import XRDRun
from trxrd.io.legacy import convert_file, load_legacy


@pytest.fixture(scope="module")
def legacy_run(legacy):
    """A CED_DRX built and analysed with the old code, like the GUI used to do."""
    pytest.importorskip("dill")
    from cedapp.drx.Calibration import Calib_DRX
    from cedapp.drx.gauge import Element

    with contextlib.redirect_stdout(io.StringIO()):
        calib = Calib_DRX(
            file_mask=str(TUTO / "scan_jf1m_0000.mask"),
            file_poni=str(TUTO / "calib_THT8_correct_samedi.poni"),
            theta_range=[8, 25],
        )
        run = legacy.CED_DRX(str(TUTO / "scan_jf1m_0000.h5"), calib, calib.energy)
        card = pd.read_csv(LIBRARY / "Ga" / "Ga-I.jcpds", sep=":", header=None, engine="python")
        library = legacy.DRX(E=calib.energy)
        ref = legacy.Element_Bibli(file=card, E=calib.energy)
        ref.save_var = [i < 6 for i in range(len(ref.Dhkl))]  # reflections ticked in the GUI
        library.Bibli_elements[ref.name] = ref
        for spectrum in run.Spectra[:2]:
            element = Element(ref, ref.name)
            element.bit_fit = True
            spectrum.Gauges = [element]
            spectrum.bit_fit = True
            spectrum.Calcul_study(mini=True)
    run.Time_spectrum = list(np.arange(len(run.Spectra)) * 1e-3)
    run.note = "[P_Ga_I] -> 0.2"
    return run


def dump(obj, path, module_alias=None):
    """dill.dump, optionally pretending the classes lived in an old module path."""
    import dill

    from cedapp.drx import CL_FD_Update

    classes = [CL_FD_Update.CED_DRX, CL_FD_Update.Spectre]
    if module_alias:
        sys.modules[module_alias] = CL_FD_Update
        for cls in classes:
            cls.__module__ = module_alias
    try:
        with open(path, "wb") as handle:
            dill.dump(obj, handle)
    finally:
        for cls in classes:
            cls.__module__ = "cedapp.drx.CL_FD_Update"
        sys.modules.pop(module_alias, None)


@pytest.mark.parametrize("alias", [None, "CL_FD_Update", "Bibli_python.CL_FD_Update"])
def test_load_legacy_whatever_the_old_module_path(tmp_path, legacy_run, alias):
    path = tmp_path / "old.CEDX"
    dump(legacy_run, path, alias)
    assert len(load_legacy(path).Spectra) == 16


def test_convert_keeps_data_and_results(tmp_path, legacy_run):
    src = tmp_path / "ESRFid09_2026_Ga02_run8.CEDXUp"
    dump(legacy_run, src)
    run = XRDRun(convert_file(src))

    assert run.name == "ESRFid09_2026_Ga02_run8" and len(run) == 16
    spectrum = run.spectrum(0)
    np.testing.assert_allclose(spectrum["intensity"], legacy_run.Spectra[0].spec)
    np.testing.assert_allclose(spectrum["baseline"], legacy_run.Spectra[0].blfit)

    element = legacy_run.Spectra[0].Gauges[0]
    summary = run.summary
    assert summary["P_Ga_I"][0] == pytest.approx(element.P)
    assert summary["V_Ga_I"][0] == pytest.approx(element.V)
    assert np.isnan(summary["P_Ga_I"][2])
    assert len(run.peaks[run.peaks["spectrum"] == 0]) == len(element.pics)
    assert run.peaks["has_hkl"].all()
    assert run.phase_card("Ga_I").k0 == element.Element_ref.K0
    assert run.note == "[P_Ga_I] -> 0.2"
    np.testing.assert_allclose(run.time, np.arange(16) * 1e-3)

    full = run.to_run()
    assert "Distance" in full.calibration.poni
    np.testing.assert_array_equal(full.calibration.mask, legacy_run.calib.mask)


def test_cli_convert_folder(tmp_path, legacy_run, capsys):
    from trxrd.cli import main

    folder = tmp_path / "Fichier_CED_DRX"
    folder.mkdir()
    dump(legacy_run, folder / "run1.CEDX")
    (folder / "broken.CEDX").write_bytes(b"not a pickle")
    assert main(["convert", str(folder), "-o", str(tmp_path / "converted")]) == 1  # one failure reported
    assert (tmp_path / "converted" / "run1.trxrd.h5").exists()
    assert "FAILED" in capsys.readouterr().err
    assert main(["info", str(tmp_path / "converted" / "run1.trxrd.h5")]) == 0
