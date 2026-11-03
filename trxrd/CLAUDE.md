# TR-XRD v2 (`trxrd`) — notes for Claude

New version of the time-resolved XRD program (ESRF ID09, dynamic DAC). The user (Florian) writes in
French: answer and write docs in French; code, identifiers and docstrings in English. The legacy
program (v1, `cedapp`, repo root) is frozen: never modify it.

Full design, file format, algorithm choices and their validation: `docs/ARCHITECTURE.md` (read it
before changing an algorithm or the HDF5 format).

Project status, decisions already taken by the user, lessons learned and the prioritised to-do list
(in French, keep it up to date): @docs/PASSATION.md

## Setup and checks

```bash
python -m venv .venv-v2
.venv-v2\Scripts\activate                                     # Windows (Linux/macOS: source .venv-v2/bin/activate)
pip install -e "./trxrd[notebook,gui,dev]"                    # from the repository root
python -m pytest trxrd                                        # must stay green
```

Keep v1 in its own environment (`pip install -r requirements.txt`: PyQt5, numpy<2, dill...): mixing
PyQt5 and PySide6 in one environment breaks Qt.

## Rules enforced by tests

- Layers (`tests/test_architecture.py`): `physics` (numpy/scipy only) <- `domain` <- `io` / `processing`
  <- `services` <- `cli`, `notebook`, `gui`. `io` and `processing` never import each other; only
  `gui` may import PySide6/pyqtgraph; the GUI goes through `services` (Session), never h5py/pyFAI.
- Every third-party import must be declared in `pyproject.toml` and every declared dependency used.
  Not used any more, do not bring back: lmfit (use `scipy.optimize.curve_fit`), dill, deap, tkinter,
  PyQt5, lecroyscope, peakutils, pynverse.
- New behaviour = new test. Synthetic ID09-like patterns: `tests/synthetic.py`.

## Conventions

- Storage axis 2θ in degrees (Q/d only for display: `physics.lattice.convert_two_theta`), energy in
  eV, pressure in GPa, lengths in Å.
- Peak profiles (`physics/profiles.py`): `f(x, center, height, width, *shape)`, center = maximum.
  Default model Pearson IV (asymmetric synchrotron / dynamic peaks).
- Fits use raw intensity minus background (never the smoothed signal), in local windows around the
  peaks; unassigned peaks are fitted as free peaks.
- The `.trxrd.h5` format is versioned (`io/h5store.py`, `FORMAT_VERSION`): compatible additions bump
  the minor version, update `docs/ARCHITECTURE.md` §5 and the round-trip tests.

## Validating on real data

The user's archives hold v1 files (`Fichier_CED_DRX/*.CEDX`) analysed by hand. To compare the
automatic analysis with them (nothing is modified):

```bash
trxrd compare "F:/ESRF/.../Fichier_CED_DRX/Water05_scan0009.CEDX" \
      --library trxrd/resources/phases/H2O trxrd/resources/phases/Sn --pmax 10 --exclude 18.479:18.679
```

Follow the `note:` lines (static gasket/diamond peaks -> `--exclude`; they are printed after the analysis,
static lines of the sample are screened out). `compare` searches and fits at the temperature of the reference
(`--temperature` overrides). Reference result on Water05 scan 9: same phases in 15/16 spectra, ice
pressures within 0.03 GPa, Sn within 0.17 GPa. Whole January 2026 campaign (143 runs): 81% of the spectra
with the same phases (`docs/ARCHITECTURE.md` §6.8); check the reference too (single-line Sn fits, copied
values). Never commit the user's data (raw scans, .CEDX) to the repository.

## Where things are

| Need | Module |
|---|---|
| Data model | `domain/model.py` |
| Read / write runs | `io/h5store.py`; notebook reader `notebook.py` (`XRDRun`) |
| Raw ESRF data, calibration, phase cards | `io/esrf.py`, `io/calibration.py`, `io/jcpds.py` |
| v1 files | `io/legacy.py` (no old code needed) |
| Peak detection / phase search | `processing/peaks.py`, `processing/phase_search.py` |
| Peak fit, cell, pressure | `processing/peakfit.py`, `processing/cell.py`, `services/fitting.py` |
| Automatic analysis with peak tracking | `services/identification.py` |
| GUI state (Qt-free) | `services/session.py` |
| GUI (`trxrd-gui`): display data and presenters without Qt, passive views, actions, threads | `gui/presenters/`, `gui/views/`, `gui/actions.py`, `gui/tasks.py`, `gui/app.py` |
| Raw data browsing, raw image and cake of a spectrum | `services/raw_data.py`, `services/images.py` |
| GUI image display (histogram, log/linear/sqrt, colour map) | `gui/views/images.py` (`ImagePanel`) |
| Quantities versus time (P, V, a, c/a, X/X0), map of the patterns versus time | `gui/presenters/display.py` (`evolution_data`, `time_map`) |
| dP/dt per sample (family) or phase, mean dP/dt over an interval (slope and ΔP/Δt) | `processing/rates.py`, `services/rates.py` |
| Detector calibration (LaB6/CeO2, no pyFAI.goniometer: silx DLL blocked on Windows) | `processing/geometry.py`, `services/calibration.py` |

## Next steps

See `docs/PASSATION.md` §6 (P1: GUI, viewer + run creation + cake + calibration + dDAC panel done, next: automatic
analysis from the GUI, mask editing, then interactive fitting and phase management; P2: remaining validation items; P4: v1 features not ported yet;
P5: algorithm ideas; P6: repository split).
