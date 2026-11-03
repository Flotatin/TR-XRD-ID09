"""What the viewer shows, computed from the domain objects (no Qt, testable alone).

Curves of the current spectrum (data, background, fit) and the expected
lines of its phases, P(t) per phase, oscilloscope traces, intensity map and
table rows. Every x value is in the display unit: 2θ (deg), Q (Å⁻¹) or d (Å).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from trxrd.domain import Run, Spectrum, energy_to_wavelength
from trxrd.physics import eos
from trxrd.physics.lattice import AXIS_LABELS, convert_two_theta

UNITS = ("2th_deg", "q_A^-1", "d_A")


def wavelength(run: Optional[Run]) -> Optional[float]:
    """λ (m) of the run, None without a usable energy (then only 2θ can be shown)."""
    energy = run.calibration.energy_ev if run is not None and run.calibration is not None else np.nan
    return energy_to_wavelength(energy) if np.isfinite(energy) and energy > 0 else None


def available_units(run: Optional[Run]) -> Tuple[str, ...]:
    return UNITS if wavelength(run) else ("2th_deg",)


def to_unit(two_theta, unit: str, run: Optional[Run]) -> np.ndarray:
    if unit == "2th_deg" or wavelength(run) is None:
        return np.asarray(two_theta, dtype=float)
    return convert_two_theta(two_theta, unit, wavelength(run))


def phase_order(run: Optional[Run]) -> List[str]:
    """Phases in a stable order: a phase keeps its colour from one spectrum to the next."""
    return run.phase_names() if run is not None else []


def _number(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if value is not None and np.isfinite(value) else ""


# ---------------------------------------------------------------------------- current spectrum


@dataclass
class Lines:
    """Expected reflections of one phase at its fitted pressure."""

    phase: str
    color: int  # index in phase_order
    x: np.ndarray
    intensity: np.ndarray  # relative intensity of the card (%)
    labels: List[str]  # "hkl" of each line
    pressure: float


@dataclass
class SpectrumCurves:
    title: str
    x_label: str
    x: np.ndarray
    intensity: np.ndarray
    baseline: Optional[np.ndarray] = None
    fit: Optional[np.ndarray] = None  # baseline + fitted model
    lines: List[Lines] = field(default_factory=list)


def expected_lines(run: Run, spectrum: Spectrum, unit: str, min_intensity: float = 1.0) -> List[Lines]:
    order = phase_order(run)
    lo, hi = float(np.nanmin(spectrum.two_theta)), float(np.nanmax(spectrum.two_theta))
    energy = run.calibration.energy_ev if run.calibration is not None else np.nan
    out = []
    for fit in spectrum.phases:
        card = run.phases.get(fit.phase)
        if card is None or not np.isfinite(fit.pressure) or not np.isfinite(energy):
            continue
        temperature = fit.temperature if np.isfinite(fit.temperature) else eos.T_REF
        positions = card.two_theta_table([fit.pressure], energy, temperature)[0]
        keep = [k for k, (x, r) in enumerate(zip(positions, card.reflections))
                if np.isfinite(x) and lo <= x <= hi and r.intensity >= min_intensity]
        out.append(Lines(
            phase=fit.phase,
            color=order.index(fit.phase) if fit.phase in order else len(order),
            x=to_unit(positions[keep], unit, run),
            intensity=np.array([card.reflections[k].intensity for k in keep], dtype=float),
            labels=[card.reflections[k].label for k in keep],
            pressure=float(fit.pressure),
        ))
    return out


def spectrum_curves(run: Run, index: int, unit: str) -> SpectrumCurves:
    spectrum = run.spectra[index]
    baseline = None if spectrum.baseline is None else np.asarray(spectrum.baseline, dtype=float)
    fit = None
    if spectrum.model is not None:
        fit = np.asarray(spectrum.model, dtype=float) + (baseline if baseline is not None else 0.0)
    time = f"  t = {spectrum.time * 1e3:.4g} ms" if np.isfinite(spectrum.time) else ""
    return SpectrumCurves(
        title=f"Spectre {index} / {len(run) - 1}{time}",
        x_label=AXIS_LABELS[unit],
        x=to_unit(spectrum.two_theta, unit, run),
        intensity=np.asarray(spectrum.intensity, dtype=float),
        baseline=baseline,
        fit=fit,
        lines=expected_lines(run, spectrum, unit),
    )


# ---------------------------------------------------------------------------- P(t), oscilloscope


@dataclass
class Series:
    phase: str
    color: int
    index: np.ndarray  # spectra where the phase was fitted
    x: np.ndarray
    pressure: np.ndarray
    error: np.ndarray


@dataclass
class Trace:
    name: str
    x: np.ndarray  # same axis as the pressures (ms)
    y: np.ndarray
    trigger: bool


@dataclass
class PressureData:
    x_label: str
    x: np.ndarray  # one value per spectrum: time (ms) or index
    series: List[Series]
    traces: List[Trace]


def spectrum_axis(run: Run) -> Tuple[np.ndarray, str, bool]:
    """x of each spectrum on the P(t) axis: time in ms, or the index when times are missing."""
    times = run.times
    if len(times) and np.isfinite(times).all():
        return times * 1e3, "t (ms)", True
    return np.arange(len(run), dtype=float), "spectre", False


def pressure_data(run: Run) -> PressureData:
    x, label, timed = spectrum_axis(run)
    order = phase_order(run)
    series = []
    for k, phase in enumerate(order):
        rows = [(i, f.pressure, f.pressure_err) for i, s in enumerate(run.spectra) for f in s.phases
                if f.phase == phase and np.isfinite(f.pressure)]
        if rows:
            index = np.array([r[0] for r in rows])
            err = np.array([r[2] for r in rows], dtype=float)
            series.append(Series(phase, k, index, x[index], np.array([r[1] for r in rows], dtype=float),
                                 np.where(np.isfinite(err), err, 0.0)))
    traces = []
    if timed and run.oscilloscope is not None and len(run.oscilloscope.time):
        t = np.asarray(run.oscilloscope.time, dtype=float) * 1e3
        for name, values in run.oscilloscope.channels.items():
            traces.append(Trace(name, t, np.asarray(values, dtype=float), name == run.oscilloscope.trigger_channel))
    return PressureData(label, x, series, traces)


def nearest_spectrum(x_of_spectra: np.ndarray, x: float) -> Optional[int]:
    if not len(x_of_spectra) or not np.isfinite(x):
        return None
    return int(np.nanargmin(np.abs(np.asarray(x_of_spectra) - x)))


# ---------------------------------------------------------------------------- map


@dataclass
class MapData:
    x_label: str
    x: np.ndarray  # uniform grid in the display unit
    image: np.ndarray  # [n_spectra, n_x], background subtracted when known


def intensity_map(run: Run, unit: str) -> Optional[MapData]:
    if not run.spectra:
        return None
    reference = np.asarray(run.spectra[0].two_theta, dtype=float)
    rows = []
    for spectrum in run.spectra:
        y = np.asarray(spectrum.corrected, dtype=float)
        x = np.asarray(spectrum.two_theta, dtype=float)
        rows.append(y if x.shape == reference.shape and np.allclose(x, reference) else np.interp(reference, x, y))
    image = np.vstack(rows)
    converted = to_unit(reference, unit, run)
    if converted[0] > converted[-1]:  # d decreases with 2θ
        converted, image = converted[::-1], image[:, ::-1]
    grid = np.linspace(converted[0], converted[-1], converted.size)
    if unit != "2th_deg":  # Q and d are not uniform in 2θ: resample
        image = np.vstack([np.interp(grid, converted, row) for row in image])
    return MapData(AXIS_LABELS[unit], grid, image)


# ---------------------------------------------------------------------------- tables

SPECTRUM_COLUMNS = ("#", "t (ms)", "Phases")
PHASE_COLUMNS = ("Phase", "P (GPa)", "σP (GPa)", "V (Å³)", "a (Å)", "b (Å)", "c (Å)", "T (K)", "État", "Pics")
PEAK_COLUMNS = ("Phase", "hkl", "2θ (°)", "σ 2θ (°)", "d (Å)", "Hauteur", "Largeur (°)", "Profil", "Actif")


def _short(phase: str) -> str:
    return phase.replace("H2O_Ice_", "Ice ")


def spectrum_rows(run: Run) -> List[Tuple[str, str, str]]:
    rows = []
    for i, spectrum in enumerate(run.spectra):
        time = f"{spectrum.time * 1e3:.4g}" if np.isfinite(spectrum.time) else ""
        phases = ", ".join(f"{_short(f.phase)} {_number(f.pressure, 2)}" for f in spectrum.phases)
        rows.append((str(i), time, phases))
    return rows


def phase_rows(spectrum: Optional[Spectrum]) -> List[Tuple[str, ...]]:
    if spectrum is None:
        return []
    return [
        (f.phase, _number(f.pressure), _number(f.pressure_err), _number(f.volume, 2), _number(f.a, 4),
         _number(f.b, 4), _number(f.c, 4), _number(f.temperature, 0), f.state, str(len(f.peaks)))
        for f in spectrum.phases
    ]


def peak_rows(run: Run, spectrum: Optional[Spectrum]) -> List[Tuple[str, ...]]:
    if spectrum is None:
        return []
    lam = wavelength(run)
    rows = []
    for fit in spectrum.phases:
        for peak in fit.peaks:
            d = ""
            if lam is not None and np.isfinite(peak.center) and peak.center > 0:
                d = _number(lam * 1e10 / (2 * math.sin(math.radians(peak.center) / 2)), 4)
            hkl = "".join(str(i) for i in peak.hkl) if peak.hkl is not None else ""
            rows.append((fit.phase, hkl, _number(peak.center, 4), _number(peak.center_err, 4), d,
                         _number(peak.height, 1), _number(peak.width, 4), peak.model, "oui" if peak.enabled else "non"))
    return rows


def summary_text(run: Optional[Run], index: int, path: Optional[str], dirty: bool) -> str:
    if run is None:
        return "Aucun run ouvert"
    state = "modifié, non enregistré" if dirty else "enregistré"
    return f"{run.name} — {len(run)} spectres — {path or 'pas encore de fichier'} ({state})"


def phases_text(spectra: Sequence[Spectrum]) -> str:
    return ", ".join(sorted({f.phase for s in spectra for f in s.phases}))
