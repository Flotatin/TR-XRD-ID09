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


import matplotlib.pyplot as plt
import fabio
scan_lbl='33.1'
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Chargement du spectre
# =========================
CED = CL.LOAD_CEDd(r"f:\ESRF\2026 Janvier HC-6664 id09\CED_File\Water05_scan0013.CEDX")
sp = CED.Spectra[9]

x = np.asarray(sp.wnb, dtype=float)
y = np.asarray(sp.spec, dtype=float)

# =========================
# Paramètres "Dioptas-like"
# =========================
smooth_frac = 0.01            # 2 % de la longueur du signal
n_iter = 20                   # itérations de moyenne mobile
deg_list = [0, 4, 8, 12, 16, 20]  # ordres polynomiaux à tester

# Fenêtre moyenne mobile
smooth_width = int(smooth_frac * len(y))
if smooth_width % 2 == 0:
    smooth_width += 1
smooth_width = max(smooth_width, 5)

kernel = np.ones(smooth_width) / smooth_width

# =========================
# Lissage itératif
# =========================
y_s = y.copy()
for _ in range(n_iter):
    y_s = np.convolve(y_s, kernel, mode="same")

# Estimation robuste du bruit (pour pondération)
res = y - y_s
sigma_r = np.median(np.abs(res)) * 1.4826
if not np.isfinite(sigma_r) or sigma_r <= 0:
    sigma_r = np.std(res) if np.std(res) > 0 else 1.0

# Poids : on réduit l'influence des pics
weights = np.ones_like(y)
weights[res > 3.0 * sigma_r] = 0.2

# =========================
# Affichage
# =========================
fig, axes = plt.subplots(len(deg_list), 1, figsize=(10, 12), sharex=True)

for ax, deg in zip(axes, deg_list):
    coeffs = np.polyfit(x, y_s, deg=deg, w=weights)
    baseline = np.polyval(coeffs, x)

    ax.plot(x, y, color="lightgray", lw=1, label="Brut")
    ax.plot(x, y_s, color="tab:blue", lw=1, label="Lissé")
    ax.plot(x, baseline, "g--", lw=2, label=f"Baseline poly deg {deg}")
    ax.plot(x, y - baseline, "k", lw=1, label="Corrigé")
    ax.axhline(0)
    ax.set_title(f"Polynomial degree = {deg}")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(loc="best")

axes[-1].set_xlabel(r"$\lambda$ (nm)")
plt.tight_layout()
plt.show()



"""

loaded_file_DRX=r'/data/visitor/hc6664/id09/20260121/RAW_DATA/Water12/Water12_0001/scan0003/scan_jf1m_0000.h5'
with File(r'/data/visitor/hc6664/id09/20260121/RAW_DATA/Water12/Water12_0001/Water12_0001.h5') as f:
    print(f[scan_lbl]["measurement"].keys())
    t1 = f[scan_lbl]["measurement"]["ch5_time"][:].ravel()
    y1 = f[scan_lbl]["measurement"]["ch5"][:].ravel()
    t2 = f[scan_lbl]["measurement"]["ch6_time"][:].ravel()
    y2 = f[scan_lbl]["measurement"]["ch6"][:].ravel()

if not (len(t1) == len(y1) == len(y2)):
    raise ValueError(
    f"Tailles incohérentes : "
    f"t1={len(t1)}, y1={len(y1)}, y2={len(y2)}"
)

data_oscillo = pd.DataFrame(
    {
    "Time": t1,
    "Channel2": y1,
    "Channel3": y2,
    }
    )
    
plt.plot(t2,y2)
plt.plot(t1,y1)
plt.show()
'''
calib = Calibration.Calib_DRX(
            file_mask=r'/data/visitor/hc6664/id09/20260121/dioptas_files/scan_jf1m_0000.mask',
            file_poni=r'/data/visitor/hc6664/id09/20260121/dioptas_files/calib_ddac2_correct.poni',
            theta_range=[8,30],
            energy=19000,
        )


img_data=fabio.open(loaded_file_DRX)


tth, intens = Calibration.Integrate_DRX(img_data.getframe(50).data, calib.mask, calib.ai, theta_range=calib.theta_range)

print('start')
CEDX = CL.CED_DRX(loaded_file_DRX
    ,
    calib,
    19000,
    data_oscillo=data_oscillo,
    time_index="Channel2"
)
CEDX.Print(Oscilo=True)

'''
"""