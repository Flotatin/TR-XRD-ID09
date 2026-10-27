# trxrd

Traitement de la DRX résolue en temps (ID09, DAC dynamique) : intégration des images, ajustement des pics,
maille, volume et pression, sauvegarde en HDF5 ouvert et lecture en notebook.

Nouveau programme, qui remplace à terme l'application historique `cedapp`. L'architecture, le format de
fichier et la feuille de route sont décrits dans [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Installation

```bash
pip install -e "./trxrd[notebook,dev]"     # depuis la racine du dépôt (ou ".[notebook,dev]" depuis trxrd/)
python -m pytest trxrd                      # 78 tests
```

## Ligne de commande

```bash
# Scan brut ESRF -> fichier d'analyse (intégration, fond, oscilloscope, temps)
trxrd build RAW_DATA/water02/water02_0001/scan0027/scan_jf1m_0000.h5 \
      --poni calib.poni --mask det.mask --tth-min 8 --tth-max 25 \
      --phases trxrd/resources/phases/H2O -o analyse/

# Anciens fichiers dill (.CEDX, .CEDXUp) -> .trxrd.h5 (l'ancien programme n'est pas nécessaire)
trxrd convert "F:/ESRF/2024 Juin HC-5719 id09/Fichier_CED_DRX" -o converti/

trxrd info analyse/water02_0001_scan0027.trxrd.h5
```

## Python / notebook

```python
from trxrd import XRDRun
from trxrd.io.h5store import load_run, save_run
from trxrd.services.fitting import fit_spectra

run = load_run("analyse/water02_0001_scan0027.trxrd.h5")
fit_spectra(run, range(len(run)), {"H2O_Ice_VII": 2.0})      # pression de départ en GPa
save_run(run, "analyse/water02_0001_scan0027.trxrd.h5")

r = XRDRun("analyse/water02_0001_scan0027.trxrd.h5")
r.summary                                   # P_<phase>, sigma_P_<phase>, V_<phase>... par spectre
r.plot_spectrum(10, unit="q_A^-1")          # axes : "2th_deg", "q_A^-1", "d_A"
r.plot_pressure()
```
