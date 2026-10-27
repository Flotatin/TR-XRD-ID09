"""Regenerate legacy_sample.CEDX with the OLD program (only needed to rebuild the test data).

Run from the repository that still contains the old code:

    python trxrd/tests/data/make_legacy_sample.py

It builds a CED_DRX from scan_jf1m_0000.h5 the way the old GUI did, adds a
Ga-I phase on two spectra (one skewed Pearson IV peak), saves it with dill,
and stores in legacy_sample_expected.npz the values the converter must find,
including the fitted curve evaluated by lmfit.
"""

import contextlib
import io
import sys
import types
import warnings
from pathlib import Path

import dill
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
for missing in ("lecroyscope", "cedapp.drx.Oscilloscope_LeCroy_QTinterface"):
    sys.modules.setdefault(missing, types.ModuleType(missing))

with contextlib.redirect_stdout(io.StringIO()):
    from cedapp.drx import CL_FD_Update as CL
    from cedapp.drx.Calibration import Calib_DRX
    from cedapp.drx.gauge import Element

    calib = Calib_DRX(
        file_mask=str(HERE / "scan_jf1m_0000.mask"),
        file_poni=str(HERE / "calib_THT8_correct_samedi.poni"),
        theta_range=[8, 25],
    )
    run = CL.CED_DRX(str(HERE / "scan_jf1m_0000.h5"), calib, calib.energy)
    card = pd.read_csv(ROOT / "resources" / "bibdrx" / "Ga" / "Ga-I.jcpds", sep=":", header=None, engine="python")
    ref = CL.Element_Bibli(file=card, E=calib.energy)
    ref.save_var = [i < 6 for i in range(len(ref.Dhkl))]  # reflections ticked in the GUI
    for spectrum in run.Spectra[:2]:
        element = Element(ref, ref.name)
        element.bit_fit = True
        element.pics[0].Update(coef_spe=[1.4, 0.7], sigma=0.05, ampH=30.0)
        spectrum.Gauges = [element]
        spectrum.bit_fit = True
        spectrum.Calcul_study(mini=True)
run.Time_spectrum = list(np.arange(len(run.Spectra)) * 1e-3)
run.note = "[P_Ga_I] -> 0.2"
calib.file_poni = calib.file_mask = "D:/ESRF/old_session/calib.poni"  # as on the beamline PC: unreachable here

with open(HERE / "legacy_sample.CEDX", "wb") as handle:
    dill.dump(run, handle)

first = run.Spectra[0]
element = first.Gauges[0]
np.savez_compressed(
    HERE / "legacy_sample_expected.npz",
    intensity=first.spec,
    baseline=first.blfit,
    model=sum(p.model.eval(p.model.make_params(), x=first.wnb) for p in element.pics),
    pressure=element.P,
    volume=element.V,
    n_peaks=len(element.pics),
    k0=ref.K0,
    mask=calib.mask,
)
print("written", HERE / "legacy_sample.CEDX")
