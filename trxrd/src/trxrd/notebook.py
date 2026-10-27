"""Read-only access to saved runs for notebooks and scripts.

    from trxrd import XRDRun

    run = XRDRun("water02_0001_scan0027.trxrd.h5")
    run.summary                       # one row per spectrum, legacy-like columns (P_<phase>, ...)
    run.peaks                         # one row per fitted peak (2θ, d, Q)
    run.plot_pressure(); run.plot_spectrum(12, unit="q_A^-1")

    runs = XRDRun.find(r"F:\\ESRF\\2025 Juin", "*water*")   # many runs at once

Needs only h5py, numpy, pandas (matplotlib for the plot helpers): no
pyFAI, no GUI.

Axis units: "2th_deg" (2θ in degrees), "q_A^-1" (Q = 4π sin θ / λ in Å⁻¹),
"d_A" (d-spacing in Å).
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import h5py
import numpy as np
import pandas as pd

from trxrd.domain import PhaseCard, Run, energy_to_wavelength
from trxrd.io import h5store
from trxrd.io.h5utils import read_scalar, read_str, read_table
from trxrd.physics.lattice import AXIS_LABELS, convert_two_theta

PathLike = Union[str, Path]

# legacy column prefix -> PhaseFit field
_SUMMARY_FIELDS = {
    "P": "pressure",
    "sigma_P": "pressure_err",
    "V": "volume",
    "sigma_V": "volume_err",
    "a": "a",
    "b": "b",
    "c": "c",
    "T": "temperature",
    "state": "state",
}


class XRDRun:
    def __init__(self, path: PathLike):
        self.path = Path(path)
        with h5py.File(self.path, "r") as f:
            h5store.check_format(f)
            entry = f[h5store.ENTRY]
            self.name = read_str(entry, "title")
            self.note = read_str(entry, "note")
            self.created = read_str(entry, "created")
            self.energy_ev = (
                float(read_scalar(entry["calibration"], "energy_ev")) if "calibration" in entry else np.nan
            )
            spectra = entry["spectra"]
            self.time = spectra["time"][()]
            self.frame = spectra["frame"][()]
            self._n_points = spectra["n_points"][()]
            self._phases_table = pd.DataFrame(read_table(entry["fits"]["phases"]))
            self._peaks_table = pd.DataFrame(read_table(entry["fits"]["peaks"]))
            self.phases: List[str] = list(entry["phases"].keys())

    # ------------------------------------------------------------------ discovery

    @classmethod
    def find(cls, folder: PathLike, pattern: str = "*", recursive: bool = True) -> List["XRDRun"]:
        """Every run file under `folder` whose name matches `pattern`."""
        folder = Path(folder)
        candidates = folder.rglob(f"{pattern}.h5") if recursive else folder.glob(f"{pattern}.h5")
        return [cls(p) for p in sorted(candidates) if h5store.is_run_file(p)]

    def __len__(self) -> int:
        return len(self.time)

    def __repr__(self) -> str:
        used = sorted(set(self._phases_table.get("phase", [])))
        return f"<XRDRun {self.name!r}: {len(self)} spectra, phases={used}, file={self.path.name}>"

    # ------------------------------------------------------------------ arrays

    def _read(self, name: str) -> Optional[np.ndarray]:
        with h5py.File(self.path, "r") as f:
            group = f[h5store.ENTRY]["spectra"]
            return group[name][()] if name in group else None

    @cached_property
    def two_theta(self) -> np.ndarray:
        """1D (shared axis) or 2D (one row per spectrum, NaN padded)."""
        return self._read("two_theta")

    @cached_property
    def intensity(self) -> np.ndarray:
        return self._read("intensity")

    @cached_property
    def baseline(self) -> Optional[np.ndarray]:
        return self._read("baseline")

    @cached_property
    def model(self) -> Optional[np.ndarray]:
        return self._read("model")

    @property
    def wavelength_m(self) -> float:
        return energy_to_wavelength(self.energy_ev) if np.isfinite(self.energy_ev) else np.nan

    def axis(self, unit: str = "2th_deg", index: Optional[int] = None) -> np.ndarray:
        """The x axis in `unit` (row `index` if the spectra have their own 2θ axes)."""
        x = self.two_theta
        if x.ndim == 2:
            if index is None:
                raise ValueError("Spectra do not share a 2θ axis: give `index`")
            x = x[index, : int(self._n_points[index])]
        return convert_two_theta(x, unit, self.wavelength_m)

    def spectrum(self, index: int) -> pd.DataFrame:
        """Columns two_theta, q, d, intensity, baseline, corrected, model."""
        n = int(self._n_points[index])
        x = self.two_theta if self.two_theta.ndim == 1 else self.two_theta[index, :n]
        data = {"two_theta": x, "q": self.axis("q_A^-1", index), "d": self.axis("d_A", index)}
        data["intensity"] = self.intensity[index, :n]
        if self.baseline is not None:
            data["baseline"] = self.baseline[index, :n]
            data["corrected"] = data["intensity"] - data["baseline"]
        if self.model is not None:
            data["model"] = self.model[index, :n]
        return pd.DataFrame(data)

    @cached_property
    def oscilloscope(self) -> Optional[pd.DataFrame]:
        with h5py.File(self.path, "r") as f:
            entry = f[h5store.ENTRY]
            if "oscilloscope" not in entry:
                return None
            group = entry["oscilloscope"]
            names = [str(n) for n in group.attrs.get("channels", [])]
            return pd.DataFrame({"time": group["time"][()], **{n: group[n][()] for n in names}})

    def phase_card(self, name: str) -> PhaseCard:
        with h5py.File(self.path, "r") as f:
            return h5store.read_phases(f[h5store.ENTRY]["phases"])[name]

    # ------------------------------------------------------------------ tables

    @property
    def phase_fits(self) -> pd.DataFrame:
        """One row per (spectrum, phase): cell, volume, pressure, temperature with errors."""
        table = self._phases_table.copy()
        if len(table):
            table.insert(1, "time", self.time[table["spectrum"].to_numpy(dtype=int)])
        return table

    @property
    def peaks(self) -> pd.DataFrame:
        """One row per peak; `center` is 2θ (deg), `d` (Å) and `q` (Å⁻¹) are derived from it."""
        table = self._peaks_table.copy()
        if len(table):
            table.insert(1, "time", self.time[table["spectrum"].to_numpy(dtype=int)])
            position = table.columns.get_loc("center") + 1
            table.insert(position, "d", convert_two_theta(table["center"], "d_A", self.wavelength_m))
            table.insert(position + 1, "q", convert_two_theta(table["center"], "q_A^-1", self.wavelength_m))
        return table

    @property
    def summary(self) -> pd.DataFrame:
        """Wide table, one row per spectrum, columns `<P|sigma_P|V|...>_<phase>` like the old Summary."""
        summary = pd.DataFrame({"spectrum": np.arange(len(self)), "time": self.time})
        fits = self._phases_table
        for phase in dict.fromkeys(fits.get("phase", [])):
            rows = fits[fits["phase"] == phase]
            for prefix, field in _SUMMARY_FIELDS.items():
                column = pd.Series(np.nan, index=summary.index, dtype=object if field == "state" else float)
                column.iloc[rows["spectrum"].to_numpy(dtype=int)] = rows[field].to_numpy()
                summary[f"{prefix}_{phase}"] = column
        return summary

    def pressure(self, phases: Optional[Sequence[str]] = None, with_errors: bool = False) -> pd.DataFrame:
        """P(t) per phase, indexed by time (spectrum number if the run has no time base)."""
        summary = self.summary
        index = summary["time"] if np.isfinite(self.time).any() else summary["spectrum"]
        phases = phases or [c[2:] for c in summary.columns if c.startswith("P_")]
        columns: Dict[str, np.ndarray] = {}
        for phase in phases:
            columns[phase] = summary[f"P_{phase}"].to_numpy()
            if with_errors:
                columns[f"{phase}_err"] = summary[f"sigma_P_{phase}"].to_numpy()
        return pd.DataFrame(columns, index=pd.Index(index.to_numpy(), name=index.name))

    def to_run(self) -> Run:
        """Full domain object (e.g. to edit and save again)."""
        return h5store.load_run(self.path)

    # ------------------------------------------------------------------ plots

    def plot_spectrum(self, index: int, ax=None, lines: bool = True, unit: str = "2th_deg"):
        import matplotlib.pyplot as plt

        ax = ax or plt.subplots()[1]
        data = self.spectrum(index)
        x = self.axis(unit, index)
        ax.plot(x, data["intensity"], color="0.5", lw=1, label="data")
        if "baseline" in data:
            ax.plot(x, data["baseline"], "--", color="0.3", lw=1, label="background")
            if "model" in data and np.isfinite(data["model"]).any():
                ax.plot(x, data["model"] + data["baseline"], color="C3", lw=1.5, label="fit")
        if lines:
            peaks = self._peaks_table[self._peaks_table["spectrum"] == index] if len(self._peaks_table) else []
            for i, (phase, group) in enumerate(peaks.groupby("phase", sort=False) if len(peaks) else []):
                for center in convert_two_theta(group["center"], unit, self.wavelength_m):
                    ax.axvline(center, color=f"C{i}", lw=0.8, alpha=0.7)
                ax.plot([], [], color=f"C{i}", label=phase)
        ax.set_xlabel(AXIS_LABELS[unit])
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title(f"{self.name} — spectrum {index}" + (f" (t = {self.time[index]:.4g} s)" if np.isfinite(self.time[index]) else ""))
        ax.legend()
        return ax

    def plot_pressure(self, ax=None, phases: Optional[Sequence[str]] = None):
        import matplotlib.pyplot as plt

        ax = ax or plt.subplots()[1]
        table = self.pressure(phases, with_errors=True)
        for phase in [c for c in table.columns if not c.endswith("_err")]:
            valid = np.isfinite(table[phase].to_numpy(dtype=float))
            ax.errorbar(table.index[valid], table[phase][valid], yerr=table[f"{phase}_err"][valid], fmt="o-", ms=3, label=phase)
        ax.set_xlabel("time (s)" if table.index.name == "time" else "spectrum")
        ax.set_ylabel("P (GPa)")
        ax.set_title(self.name)
        ax.legend()
        return ax

    def plot_map(self, ax=None, unit: str = "2th_deg"):
        """Intensity map (x in `unit`) × spectrum (needs a shared 2θ axis)."""
        import matplotlib.pyplot as plt

        ax = ax or plt.subplots()[1]
        y = np.arange(len(self))
        ax.pcolormesh(self.axis(unit), y, self.intensity, shading="auto")
        ax.set_xlabel(AXIS_LABELS[unit])
        ax.set_ylabel("spectrum")
        ax.set_title(self.name)
        return ax
