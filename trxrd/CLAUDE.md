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

Follow the `note:` lines (static gasket/diamond peaks -> `--exclude`). Reference result on Water05
scan 9: same phases in 15/16 spectra, ice pressures within 0.03 GPa, Sn within 0.16 GPa.
Never commit the user's data (raw scans, .CEDX) to the repository.

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
| GUI state (for the coming PySide6 GUI) | `services/session.py` |

## Next steps

See `docs/PASSATION.md` §6 (P1: PySide6 viewer on `services.Session`; P2: validation on the user's
archives with `trxrd compare`; P3: Sn offset; P4: v1 features not ported yet; P5: algorithm ideas; P6:
v1 tag and repository split).
