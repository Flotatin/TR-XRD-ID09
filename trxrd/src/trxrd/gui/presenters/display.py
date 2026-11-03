"""What the viewer shows, computed from the domain objects (no Qt, testable alone).

Curves of the current spectrum (data, background, fit) and the expected
lines of its phases; P, V, a, c/a... of each phase versus time with the
oscilloscope traces; dP/dt and its mean over an interval; map of the
integrated patterns versus time; image scaling; table rows. Every 2θ value
is shown in the display unit: 2θ (deg), Q (Å⁻¹) or d (Å).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from trxrd.domain import PhaseCard, PhaseFit, Run, Spectrum, energy_to_wavelength
from trxrd.physics import eos, lattice
from trxrd.physics.lattice import AXIS_LABELS, convert_two_theta
from trxrd.services import rates

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


# ---------------------------------------------------------------------------- quantities versus time, oscilloscope


@dataclass(frozen=True)
class Quantity:
    key: str
    label: str  # axis label
    relative_label: Optional[str] = None  # divided by the value of the card (X/X₀); None: no reference


QUANTITIES = {q.key: q for q in (
    Quantity("P", "P (GPa)"),
    Quantity("V", "V (Å³)", "V/V₀"),
    Quantity("a", "a (Å)", "a/a₀"),
    Quantity("b", "b (Å)", "b/b₀"),
    Quantity("c", "c (Å)", "c/c₀"),
    Quantity("c/a", "c/a", "(c/a)/(c/a)₀"),
    Quantity("b/a", "b/a", "(b/a)/(b/a)₀"),
    Quantity("T", "T (K)"),
)}


def _ratio(top: float, top_err: float, bottom: float, bottom_err: float) -> Tuple[float, float]:
    if not (np.isfinite(top) and np.isfinite(bottom)) or bottom == 0:
        return np.nan, np.nan
    value = top / bottom
    terms = [(e / v) ** 2 for e, v in ((top_err, top), (bottom_err, bottom)) if np.isfinite(e) and v]
    return value, abs(value) * math.sqrt(sum(terms))  # errors taken as independent


def quantity_value(fit: PhaseFit, card: Optional[PhaseCard], key: str) -> Tuple[float, float]:
    """(value, 1σ error) of one quantity for one fitted phase; NaN when it does not apply.

    b, c and the axial ratios only exist where the crystal system leaves them
    free (b of a cubic phase is a, its c/a is 1: not shown).
    """
    free = lattice.free_lengths(card.symmetry) if card is not None else ("a", "b", "c")
    lengths = {"a": (fit.a, fit.a_err), "b": (fit.b, fit.b_err), "c": (fit.c, fit.c_err)}
    if key == "P":
        return fit.pressure, fit.pressure_err
    if key == "T":
        return fit.temperature, fit.temperature_err
    if key == "V":
        return fit.volume, fit.volume_err
    if key in lengths:
        return lengths[key] if key in free else (np.nan, np.nan)
    if key in ("c/a", "b/a"):
        top = key[0]
        return _ratio(*lengths[top], *lengths["a"]) if top in free else (np.nan, np.nan)
    raise ValueError(f"unknown quantity {key!r}")


def reference_value(card: Optional[PhaseCard], key: str, temperature: float = np.nan) -> float:
    """Value of the card at ambient pressure (V₀ at the temperature of the fit), NaN if none."""
    if card is None or QUANTITIES[key].relative_label is None:
        return np.nan
    if key == "V":
        return card.v0_at(temperature if np.isfinite(temperature) else eos.T_REF)
    if key in ("a", "b", "c"):
        return float(getattr(card, key))
    return float(getattr(card, key[0])) / card.a  # c/a, b/a


@dataclass
class Series:
    phase: str
    color: int
    index: np.ndarray  # spectra where the phase has a value
    x: np.ndarray
    y: np.ndarray
    error: np.ndarray  # 1σ, 0 when unknown


@dataclass
class Trace:
    name: str
    x: np.ndarray  # same axis as the spectra (ms)
    y: np.ndarray
    trigger: bool


@dataclass
class EvolutionData:
    """One quantity of every phase versus time, and the oscilloscope (dDAC view of the v1)."""

    x_label: str
    y_label: str
    x: np.ndarray  # one value per spectrum: time (ms) or index
    series: List[Series]
    traces: List[Trace]
    quantity: str  # key of QUANTITIES actually shown
    relative: bool
    quantities: List[Tuple[str, str]]  # (key, label) of the quantities this run has

    @property
    def can_be_relative(self) -> bool:
        """The quantity has a card value to divide by (not P, not T)."""
        return QUANTITIES[self.quantity].relative_label is not None

    @property
    def span(self) -> Tuple[float, float]:
        """x range of the spectra."""
        return (float(np.nanmin(self.x)), float(np.nanmax(self.x))) if len(self.x) else (0.0, 1.0)

    @property
    def trace_span(self) -> Optional[Tuple[float, float]]:
        """x range of the whole oscilloscope record, None without one."""
        if not self.traces:
            return None
        return (float(min(np.nanmin(t.x) for t in self.traces)), float(max(np.nanmax(t.x) for t in self.traces)))


def spectrum_axis(run: Run) -> Tuple[np.ndarray, str, bool]:
    """x of each spectrum on the time axis: time in ms, or the index when times are missing."""
    times = run.times
    if len(times) and np.isfinite(times).all():
        return times * 1e3, "t (ms)", True
    return np.arange(len(run), dtype=float), "spectre", False


def available_quantities(run: Run) -> List[Tuple[str, str]]:
    """Quantities with at least one value in the run (P always first)."""
    found = {"P"}
    for spectrum in run.spectra:
        for fit in spectrum.phases:
            card = run.phases.get(fit.phase)
            found.update(key for key in QUANTITIES
                         if key not in found and np.isfinite(quantity_value(fit, card, key)[0]))
    return [(key, q.label) for key, q in QUANTITIES.items() if key in found]


def evolution_data(run: Run, quantity: str = "P", relative: bool = False) -> EvolutionData:
    """`quantity` of every phase per spectrum; relative: divided by the card value (V/V₀, a/a₀...).

    A quantity the run does not have falls back to P.
    """
    if quantity not in QUANTITIES:
        raise ValueError(f"unknown quantity {quantity!r}")
    quantities = available_quantities(run)
    if quantity not in dict(quantities):
        quantity = "P"
    info = QUANTITIES[quantity]
    relative = bool(relative) and info.relative_label is not None
    x, label, timed = spectrum_axis(run)
    series = []
    for k, phase in enumerate(phase_order(run)):
        card = run.phases.get(phase)
        rows = []
        for i, spectrum in enumerate(run.spectra):
            fit = spectrum.phase_fit(phase)
            if fit is None:
                continue
            value, error = quantity_value(fit, card, quantity)
            if relative:
                reference = reference_value(card, quantity, fit.temperature)
                value, error = value / reference, error / reference
            if np.isfinite(value):
                rows.append((i, value, error))
        if rows:
            index = np.array([r[0] for r in rows])
            error = np.array([r[2] for r in rows], dtype=float)
            series.append(Series(phase, k, index, x[index], np.array([r[1] for r in rows], dtype=float),
                                 np.where(np.isfinite(error), np.abs(error), 0.0)))
    traces = []
    if timed and run.oscilloscope is not None and len(run.oscilloscope.time):
        t = np.asarray(run.oscilloscope.time, dtype=float) * 1e3
        for name, values in run.oscilloscope.channels.items():
            traces.append(Trace(name, t, np.asarray(values, dtype=float), name == run.oscilloscope.trigger_channel))
    y_label = info.relative_label if relative else info.label
    return EvolutionData(label, y_label, x, series, traces, quantity, relative, quantities)


def nearest_spectrum(x_of_spectra: np.ndarray, x: float) -> Optional[int]:
    if not len(x_of_spectra) or not np.isfinite(x):
        return None
    return int(np.nanargmin(np.abs(np.asarray(x_of_spectra) - x)))


# ---------------------------------------------------------------------------- dP/dt

RATE_LABEL = "dP/dt (GPa/ms)"


@dataclass
class RateSeries:
    name: str  # sample ("H2O") or phase
    phases: List[str]
    color: int  # colour of its first phase
    x: np.ndarray  # ms
    y: np.ndarray  # GPa/ms
    error: np.ndarray  # 1σ, 0 when unknown


@dataclass
class RateData:
    x_label: str
    y_label: str
    series: List[RateSeries]
    message: str = ""  # why nothing is shown
    by: str = "family"  # "family" (one curve per sample) or "phase"
    half_window: int = 2


def _color_of(run: Run, phases: Sequence[str]) -> int:
    order = phase_order(run)
    return order.index(phases[0]) if phases and phases[0] in order else len(order)


def has_time_base(run: Optional[Run]) -> bool:
    return run is not None and len(run) > 0 and bool(np.isfinite(run.times).all())


def rate_data(run: Run, half_window: int = 2, by: str = "family") -> RateData:
    """dP/dt of every sample (or phase): straight line over ±half_window spectra."""
    if not has_time_base(run):
        return RateData("spectre", RATE_LABEL, [], "dP/dt : ce run n'a pas de base de temps.", by, half_window)
    series = []
    for curve in rates.dpdt(run, half_window, by):
        keep = np.isfinite(curve.rate)
        if keep.any():
            series.append(RateSeries(curve.name, curve.phases, _color_of(run, curve.phases), curve.time[keep],
                                     curve.rate[keep], np.where(np.isfinite(curve.rate_err[keep]), curve.rate_err[keep], 0.0)))
    message = "" if series else "dP/dt : pas assez de pressions ajustées (3 spectres au moins par courbe)."
    return RateData("t (ms)", RATE_LABEL, series, message, by, half_window)


def with_error(value: float, error: float) -> str:
    """'-25.31 ± 0.42': the value rounded to two significant digits of its error."""
    if not np.isfinite(value):
        return "—"
    if not (np.isfinite(error) and error > 0):
        return f"{value:.4g}"
    digits = max(0, 1 - int(math.floor(math.log10(error))))
    return f"{value:.{digits}f} ± {error:.{digits}f}"


def _short_phases(phases: Sequence[str]) -> str:
    return ", ".join(p.split("_", 1)[1] if "_" in p else p for p in phases)


@dataclass
class IntervalData:
    """Mean dP/dt of every curve over the chosen interval (ms)."""

    start: float
    stop: float
    means: List[rates.MeanRate]
    colors: List[int]

    @property
    def lines(self) -> List[str]:
        """One line of text per curve."""
        out = []
        for r in self.means:
            phases = f" ({_short_phases(r.phases)})" if len(r.phases) > 1 or r.phases[0] != r.name else ""
            secant = f"{r.secant:.4g}" if np.isfinite(r.secant) else "—"
            out.append(f"{r.name}{phases} : pente {with_error(r.rate, r.rate_err)} GPa/ms ; ΔP/Δt = {secant} GPa/ms "
                       f"(ΔP = {r.delta_p:+.3f} GPa en {r.delta_t:.4g} ms, {r.n} spectres)")
        return out

    @property
    def segments(self) -> List[Tuple[float, float, float, float, int]]:
        """(t start, t stop, P start, P stop, colour) of the straight line of each curve, for the P(t) plot."""
        return [(r.start, r.stop, r.p_start, r.p_stop, color) for r, color in zip(self.means, self.colors)]

    @property
    def table(self) -> str:
        return rates.rates_table(self.means)


def interval_data(run: Optional[Run], interval: Optional[Tuple[float, float]], by: str = "family",
                  half_window: int = 2) -> Optional[IntervalData]:
    """Mean dP/dt over the interval: slope of the straight line, and ΔP/Δt of the smoothed pressure."""
    if interval is None or not has_time_base(run):
        return None
    start, stop = sorted(interval)
    found = rates.mean_dpdt(run, start, stop, by, half_window)
    return IntervalData(start, stop, found, [_color_of(run, r.phases) for r in found])


# ---------------------------------------------------------------------------- maps


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
    grid, image = _resample(reference, np.vstack(rows), unit, run)  # Q and d are not uniform in 2θ
    return MapData(AXIS_LABELS[unit], grid, image)


MAX_COLUMNS = 4000


def time_columns(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform grid over the spectra times and the spectrum shown in each column (the nearest one).

    An image needs evenly spaced columns: when the spectra are not evenly
    spaced (gaps, static spectra), each column shows the nearest spectrum.
    """
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="stable")
    ordered = x[order]
    steps = np.diff(ordered)
    positive = steps[steps > 0]
    if len(x) < 2 or not positive.size:
        return ordered, order
    step = float(np.median(positive))
    if np.allclose(steps, step, rtol=0.02, atol=0):
        return ordered, order
    count = min(int(round((ordered[-1] - ordered[0]) / step)) + 1, MAX_COLUMNS)
    grid = np.linspace(ordered[0], ordered[-1], count)
    nearest = np.searchsorted((ordered[1:] + ordered[:-1]) / 2, grid)
    return grid, order[nearest]


@dataclass
class TimeMapData:
    """Integrated patterns side by side: time (or index) along x, 2θ / Q / d along y (dDAC view of the v1)."""

    x_label: str
    x: np.ndarray  # uniform grid of the columns (ms or index)
    columns: np.ndarray  # spectrum shown in each column
    y_label: str
    y: np.ndarray  # uniform grid in the display unit
    image: np.ndarray  # [n_y, n_x], background subtracted when known


def time_map(run: Optional[Run], unit: str) -> Optional[TimeMapData]:
    patterns = intensity_map(run, unit) if run is not None else None
    if patterns is None:
        return None
    x, label, _ = spectrum_axis(run)
    grid, columns = time_columns(x)
    return TimeMapData(label, grid, columns, patterns.x_label, patterns.x, patterns.image[columns].T)


# ---------------------------------------------------------------------------- image display

SCALES = ("linear", "log", "sqrt")


def scaled_image(image, scale: str) -> np.ndarray:
    """Display copy of an image: linear, log10 or square root (NaN kept: masked pixels, empty azimuths).

    In log, values below a floor (5th percentile of the positive values) are
    raised to it: background-subtracted maps have negative values.
    """
    image = np.asarray(image, dtype=float)
    if scale == "linear":
        return image
    if scale == "log":
        finite = image[np.isfinite(image)]
        positive = finite[finite > 0]
        floor = max(float(np.percentile(positive, 5)), 1e-6) if positive.size else 1.0
        return np.log10(np.clip(image, floor, None))
    if scale == "sqrt":
        return np.sqrt(np.clip(image, 0.0, None))
    raise ValueError(f"unknown scale {scale!r}")


def robust_levels(image: np.ndarray, low: float = 1.0, high: float = 99.8) -> Tuple[float, float]:
    """Colour levels from percentiles (NaN ignored): a few hot pixels must not darken the whole image."""
    finite = np.asarray(image, dtype=float)
    finite = finite[np.isfinite(finite)]
    lo, hi = (float(v) for v in np.percentile(finite, [low, high])) if finite.size else (0.0, 1.0)
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


# ---------------------------------------------------------------------------- cake


@dataclass
class CakeData:
    x_label: str
    x: np.ndarray  # uniform grid in the display unit, as the spectrum axis
    azimuth: np.ndarray  # deg
    image: np.ndarray  # [n_azimuth, n_x]


def _resample(two_theta: np.ndarray, image: np.ndarray, unit: str, run: Optional[Run]):
    """Columns of `image` (on a 2θ grid) on a uniform grid of the display unit."""
    converted = to_unit(two_theta, unit, run)
    if converted[0] > converted[-1]:  # d decreases with 2θ
        converted, image = converted[::-1], image[:, ::-1]
    if unit == "2th_deg" or wavelength(run) is None:
        return converted, image
    grid = np.linspace(converted[0], converted[-1], converted.size)
    return grid, np.vstack([np.interp(grid, converted, row) for row in image])


def compact_azimuth(azimuth: np.ndarray, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Only the azimuth sector the detector covers, in one piece.

    The Jungfrau 1M sees ~140° of the rings; pyFAI cuts the azimuth at ±180°,
    which may split that sector in two. The rows are rolled so that the
    largest empty gap is at the ends, then cut off (the azimuth goes past
    180° where needed).
    """
    filled = np.isfinite(image).any(axis=1)
    if filled.all() or not filled.any():
        return azimuth, image
    n = filled.size
    gap, best, best_end = 0, 0, 0
    for k, empty in enumerate(np.concatenate([~filled, ~filled])):  # circular: the gap may wrap
        gap = gap + 1 if empty else 0
        if gap > best:
            best, best_end = gap, k
    start = (best_end + 1) % n  # first covered row after the largest gap
    azimuth = np.concatenate([azimuth[start:], azimuth[:start] + 360.0])
    image = np.roll(image, -start, axis=0)
    rows = np.flatnonzero(np.isfinite(image).any(axis=1))
    return azimuth[rows[0]:rows[-1] + 1], image[rows[0]:rows[-1] + 1]


def cake_data(cake, unit: str, run: Optional[Run]) -> CakeData:
    """cake: services.images.Cake (2θ grid, azimuth, intensity)."""
    azimuth, intensity = compact_azimuth(np.asarray(cake.azimuth, dtype=float), np.asarray(cake.intensity, dtype=float))
    x, image = _resample(np.asarray(cake.two_theta, dtype=float), intensity, unit, run)
    label = AXIS_LABELS[unit] if wavelength(run) is not None else AXIS_LABELS["2th_deg"]
    return CakeData(label, x, azimuth, image)


# ---------------------------------------------------------------------------- preview of a raw scan (new run)


@dataclass
class PreviewData:
    """A raw scan in the "new run" dialog, over the whole detector, to choose the 2θ range of the integration."""

    image: np.ndarray  # [rows, columns] as integrated (flipped), masked pixels NaN
    title: str  # which image: one frame or the maximum over the scan
    frames: int
    two_theta_map: Optional[np.ndarray] = None  # deg per pixel, NaN where masked; None without .poni
    usable: Optional[Tuple[float, float]] = None  # 2θ range of the usable pixels
    x: Optional[np.ndarray] = None  # 2θ (deg) of the pattern over the usable range
    intensity: Optional[np.ndarray] = None
    coverage: Optional[np.ndarray] = None  # deg of azimuth seen, per 2θ bin
    cake: Optional[CakeData] = None
    message: str = ""


def preview_data(preview, image: np.ndarray, frame: Optional[int], mode: str = "image") -> PreviewData:
    """preview: services.images.ScanPreview; image: its raw frame `frame`, or its maximum (frame None)."""
    count = preview.frames
    if frame is None:
        title = f"maximum de {len(preview.maximum_frames())} images sur {count}"
    else:
        title = f"image n° {min(int(frame), count - 1)} (de 0 à {count - 1})"
    data = PreviewData(preview.shown(image), title, count)
    maps = preview.angle_maps(image.shape)
    if maps is None:
        data.message = "Choisir un fichier .poni pour voir le diagramme en 2θ et la plage intégrée."
        return data
    data.two_theta_map, data.usable = maps[0], preview.usable_range(image.shape)
    pattern = preview.pattern(image)
    data.x, data.intensity, data.coverage = pattern.two_theta, pattern.intensity, pattern.coverage
    if mode == "cake":
        data.cake = cake_data(preview.cake(image), "2th_deg", None)
    return data


def outside_range(two_theta_map: np.ndarray, low: float, high: float) -> np.ndarray:
    """Usable pixels outside [low, high]: they are not integrated (masked pixels, NaN, are not counted)."""
    return (two_theta_map < low) | (two_theta_map > high)


def range_text(data: Optional[PreviewData], low: float, high: float) -> str:
    if data is None or data.x is None:
        return ""
    parts = [f"Plage intégrée : {low:.2f} → {high:.2f}° ({high - low:.2f}°)"]
    inside = (data.x >= low) & (data.x <= high) & (data.coverage > 0)
    if inside.any():
        seen = data.coverage[inside]
        parts.append(f"anneaux vus sur {seen.min():.0f} à {seen.max():.0f}° d'azimut")
    if low < data.usable[0] or high > data.usable[1]:
        parts.append(f"au-delà des pixels utilisables ({data.usable[0]:.2f} → {data.usable[1]:.2f}°) : bords vides")
    return " ; ".join(parts)


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
