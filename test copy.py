import h5py

import silx
import pandas as pd
from silx.io.h5py_utils import File
from pathlib import Path
import sys


SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cedapp.drx import Calibration
from cedapp.drx import CL_FD_Update as CL
from cedapp.drx import gauge as gauge

import matplotlib.pyplot as plt
import fabio

import numpy as np
import matplotlib.pyplot as plt

# =========================
# Chargement du spectre
# =========================
Ga_ref=CL.Element_Bibli( pd.read_csv(r"C:\Users\ADM_DEMBELEF\Documents\GitHub\TR-XRD-ID09\resources\bibdrx\KCl\KCl_B1.jcpds", sep=":", header=None, engine="python"),E=19e3)
Ga_ref.save_var=[True]*len(Ga_ref.name_dhkl)
Ga = gauge.Element(name=Ga_ref.name,Element_ref=Ga_ref)

print(Ga_ref.Dhkl)
Ga.bit_fit=True
Ga.Calcul(verbose=True)