"""dP/dt of a run, per sample or per phase: local rate (sliding straight line) and mean rate over an interval.

Per sample ("family"): the phases of one substance (H2O_Ice_VII, H2O_Ice_VI,
H2O_Ice_Ih) give one pressure curve, continuous across the transitions; where
two of them coexist in a spectrum, their pressures are averaged. The pressure
marker (Sn_Beta, then Sn_bct) is another sample. The v1 averaged every phase
together: the sample and the marker do not sit at the same pressure, and the
average jumped whenever a phase appeared or disappeared.

Times are in ms, rates in GPa/ms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from trxrd.domain import Run
from trxrd.processing.phase_search import family
from trxrd.processing.rates import linear_fit, local_fit

GROUPINGS = ("family", "phase")

Entry = Tuple[int, float, str, float]  # (spectrum, time in ms, phase, pressure in GPa)


@dataclass
class PressureCurve:
    name: str  # "H2O" (sample) or the phase name
    phases: List[str]
    index: np.ndarray  # spectra
    time: np.ndarray  # ms, increasing
    pressure: np.ndarray  # GPa


@dataclass
class RateCurve(PressureCurve):
    rate: np.ndarray = field(default_factory=lambda: np.array([]))  # dP/dt (GPa/ms) at each point
    rate_err: np.ndarray = field(default_factory=lambda: np.array([]))  # 1σ


@dataclass
class MeanRate:
    """Mean dP/dt of one curve over [start, stop], two ways.

    - `rate`: slope of the straight line through every spectrum of the
      interval. It uses all of them and has an error, but it is the mean rate
      only where P(t) is straight: over a curved P(t) it differs from ΔP/Δt.
    - `secant`: ΔP/Δt between the first and the last spectrum of the
      interval, on the pressure smoothed by the sliding straight line (the
      noise of one spectrum does not go into it).

    The two agree when P(t) is straight over the interval.
    """

    name: str
    phases: List[str]
    start: float  # ms
    stop: float
    n: int  # spectra in the interval
    rate: float  # GPa/ms
    rate_err: float  # 1σ, from the scatter around the line (NaN with 2 points)
    p_start: float  # GPa, the straight line at `start`
    p_stop: float
    rms: float  # GPa, scatter of the points around the line
    first: float  # ms, first spectrum in the interval
    last: float
    p_first: float  # GPa, smoothed pressure there
    p_last: float

    @property
    def delta_p(self) -> float:
        """Change of the (smoothed) pressure between the first and the last spectrum."""
        return self.p_last - self.p_first

    @property
    def delta_t(self) -> float:
        return self.last - self.first

    @property
    def secant(self) -> float:
        return self.delta_p / self.delta_t if self.delta_t > 0 else np.nan


def run_entries(run: Run) -> List[Entry]:
    """Every fitted pressure of the run with the time of its spectrum (spectra without a time are left out)."""
    entries = []
    for i, spectrum in enumerate(run.spectra):
        if not np.isfinite(spectrum.time):
            continue
        for fit in spectrum.phases:
            if np.isfinite(fit.pressure):
                entries.append((i, spectrum.time * 1e3, fit.phase, float(fit.pressure)))
    return entries


def pressure_curves(entries: Iterable[Entry], by: str = "family") -> List[PressureCurve]:
    """One pressure curve per sample (`by="family"`) or per phase, in order of first appearance."""
    if by not in GROUPINGS:
        raise ValueError(f"unknown grouping {by!r} (expected one of {GROUPINGS})")
    groups = {}  # key -> (name, phases, {spectrum: (time, [pressures])})
    for spectrum, time, phase, pressure in entries:
        key = family(phase) if by == "family" else phase
        name = phase.split("_", 1)[0] if by == "family" else phase
        _, phases, points = groups.setdefault(key, (name, [], {}))
        if phase not in phases:
            phases.append(phase)
        points.setdefault(spectrum, (time, []))[1].append(pressure)
    curves = []
    for name, phases, points in groups.values():
        rows = sorted((time, spectrum, float(np.mean(values))) for spectrum, (time, values) in points.items())
        curves.append(PressureCurve(name, phases, np.array([r[1] for r in rows], dtype=int),
                                    np.array([r[0] for r in rows], dtype=float), np.array([r[2] for r in rows], dtype=float)))
    return curves


def dpdt(source, half_window: int = 2, by: str = "family") -> List[RateCurve]:
    """dP/dt at every spectrum: straight line over ±`half_window` spectra of the same curve.

    `source`: a Run, or entries (spectrum, time in ms, phase, pressure).
    """
    curves = pressure_curves(run_entries(source) if isinstance(source, Run) else source, by)
    out = []
    for curve in curves:
        fit = local_fit(curve.time, curve.pressure, half_window)
        out.append(RateCurve(curve.name, curve.phases, curve.index, curve.time, curve.pressure, fit.slope, fit.slope_err))
    return out


def mean_dpdt(source, start: float, stop: float, by: str = "family", half_window: int = 2) -> List[MeanRate]:
    """Mean dP/dt of every curve between `start` and `stop` (ms), see MeanRate.

    `half_window`: smoothing of the pressures at the ends (as for `dpdt`).
    Curves with fewer than 2 spectra in the interval are left out.
    """
    start, stop = sorted((float(start), float(stop)))
    curves = pressure_curves(run_entries(source) if isinstance(source, Run) else source, by)
    out = []
    for curve in curves:
        inside = np.flatnonzero((curve.time >= start) & (curve.time <= stop))
        fit = linear_fit(curve.time[inside], curve.pressure[inside])
        if fit.n < 2 or not np.isfinite(fit.slope):
            continue
        smoothed = local_fit(curve.time, curve.pressure, half_window).value
        first, last = inside[0], inside[-1]
        out.append(MeanRate(curve.name, curve.phases, start, stop, fit.n, fit.slope, fit.slope_err,
                            float(fit.at(start)), float(fit.at(stop)), fit.rms, float(curve.time[first]),
                            float(curve.time[last]), float(smoothed[first]), float(smoothed[last])))
    return out


TABLE_COLUMNS = ("courbe", "phases", "t début (ms)", "t fin (ms)", "spectres", "pente (GPa/ms)", "σ pente (GPa/ms)",
                 "ΔP/Δt (GPa/ms)", "ΔP (GPa)", "Δt (ms)", "P premier (GPa)", "P dernier (GPa)", "écart à la droite (GPa)")


def rates_table(rates: Sequence[MeanRate]) -> str:
    """Tab-separated table of mean rates (header included), to paste into a spreadsheet."""
    lines = ["\t".join(TABLE_COLUMNS)]
    for r in rates:
        values = (r.start, r.stop, r.n, r.rate, r.rate_err, r.secant, r.delta_p, r.delta_t, r.p_first, r.p_last, r.rms)
        lines.append("\t".join([r.name, ", ".join(r.phases)] + [f"{v:.6g}" for v in values]))
    return "\n".join(lines)
