# trxrd — TR-XRD v2

**Nouvelle version (v2)** du programme. L'ancienne version (v1, `cedapp`, PyQt5, fichiers `.CEDX`) reste à la
racine du dépôt, figée, pour relire les anciennes données ; son dernier état est le commit `6f579f5` de `main`.

Traitement de la DRX résolue en temps (ID09, DAC dynamique) : intégration des images, ajustement des pics,
maille, volume et pression, sauvegarde en HDF5 ouvert et lecture en notebook.

L'architecture, le format de fichier, les choix d'algorithmes et leur validation sont décrits dans
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) ; l'état du projet et la liste des tâches restantes dans
[docs/PASSATION.md](docs/PASSATION.md).

## Installation

Un environnement virtuel dédié, séparé de celui de la v1 : PyQt5 (v1) et PySide6 (v2) ne cohabitent pas.

```bash
python -m venv .venv-v2
.venv-v2\Scripts\activate                    # Windows ; Linux/macOS : source .venv-v2/bin/activate
pip install -e "./trxrd[notebook,gui,dev]"  # depuis la racine du dépôt
python -m pytest trxrd                      # 121 tests
```

## Ligne de commande

```bash
# Scan brut ESRF -> fichier d'analyse (intégration, fond, oscilloscope, temps)
trxrd build RAW_DATA/water02/water02_0001/scan0027/scan_jf1m_0000.h5 \
      --poni calib.poni --mask det.mask --tth-min 8 --tth-max 25 \
      --phases trxrd/resources/phases/H2O -o analyse/

# Analyse automatique : détection des pics -> phases et pressions -> ajustement Pearson IV -> P(t)
# (les pics fixes, joint/diamant, sont signalés avec la zone --exclude à utiliser)
trxrd auto analyse/water02_0001_scan0027.trxrd.h5 --library trxrd/resources/phases/H2O trxrd/resources/phases/Sn \
      --pmax 10 --exclude 18.48:18.68

# Réanalyser un fichier de la v1 à l'aveugle et comparer avec l'analyse manuelle (rien n'est modifié)
trxrd compare "F:/ESRF/2026 Janvier HC-6664 id09/Fichier_CED_DRX/Water05_scan0009.CEDX" \
      --library trxrd/resources/phases/H2O trxrd/resources/phases/Sn --pmax 10 --exclude 18.479:18.679

# Anciens fichiers dill (.CEDX, .CEDXUp) -> .trxrd.h5 (l'ancien programme n'est pas nécessaire)
trxrd convert "F:/ESRF/2024 Juin HC-5719 id09/Fichier_CED_DRX" -o converti/

trxrd info analyse/water02_0001_scan0027.trxrd.h5
```

## Python / notebook

```python
from trxrd import XRDRun
from trxrd.io.h5store import load_run, save_run
from trxrd.services.fitting import fit_spectra
from trxrd.services.identification import auto_analyse

run = load_run("analyse/water02_0001_scan0027.trxrd.h5")
auto_analyse(run, range(len(run)), ["H2O_Ice_VI", "H2O_Ice_VII"])   # tout automatique
# ou, phases connues : fit_spectra(run, range(len(run)), {"H2O_Ice_VII": 2.0})  (P de départ en GPa)
save_run(run, "analyse/water02_0001_scan0027.trxrd.h5")

r = XRDRun("analyse/water02_0001_scan0027.trxrd.h5")
r.summary                                   # P_<phase>, sigma_P_<phase>, V_<phase>... par spectre
r.plot_spectrum(10, unit="q_A^-1")          # axes : "2th_deg", "q_A^-1", "d_A"
r.plot_pressure()
```
