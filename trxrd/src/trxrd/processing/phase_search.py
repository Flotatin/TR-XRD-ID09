"""Which phases, at which pressures, explain the detected peaks?

Deterministic search that replaces the former genetic algorithm:

1. For every phase, 2θ of its lines on a fine pressure grid (vectorised).
   The pressures where the phase alone explains the peaks best, each giving a
   different set of explained peaks, become its candidates.
2. Every combination of up to `max_phases` phases × candidate pressures is
   scored exactly once (no duplicates, nothing random: same input, same
   answer).
3. The best combinations are refined by least squares on the matched peak
   positions (scipy.optimize.least_squares), then ranked again.

Cost of a combination (lower is better), in units of "one unexplained strong peak":

    Σ_peaks  s_j · min(1, (d_j / tol)²)        distance to the nearest line, 1 if none within tol
  + cost_missing_line · Σ_strong lines not observed (relative intensity)
  + cost_chance · Σ_phases p_chance · Σ_j s_j        (peaks a phase would explain by pure chance)
  + cost_phase · n_phases                        (a phase must explain enough to be added)

p_chance = n_lines_in_window · 2·tol / window width: a phase with a forest of
lines (ice XXI: 72 lines between 8 and 25°) explains any pattern by chance
and must be credited only for what it explains beyond that.
  + ((P_max − P_min) / max_pressure_spread)²     (optional: phases of one sample see similar P)
  + cost_family_change · (families added or removed w.r.t. the previous spectrum)
  + cost_phase_change · (phases added or removed)    (a polymorph change inside a family is cheap)

The family of a phase is the start of its name ("H2O_Ice_VII" -> "H2O",
"Sn_Beta" -> "Sn"): the sample does not vanish, it transforms.

s_j = min(1, snr_j / snr_full) weighs each detected peak by its significance.
Weak lines of a card may explain a peak (texture, spotty rings) but only the
lines above `min_line_intensity` count as missing when they are not seen.

    tables = LineTables(run.phases, energy_ev, (8, 25), SearchOptions(pressure_range=(0, 20)))
    best = search_composition(detected_peaks, tables)[0]   # best.pressures = {"H2O_Ice_VII": 2.49, ...}
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from trxrd.domain import PhaseCard
from trxrd.physics import eos
from trxrd.processing.peaks import DetectedPeak


@dataclass
class SearchOptions:
    pressure_range: Tuple[float, float] = (0.0, 30.0)  # GPa
    use_card_ranges: bool = True  # also limit each phase to the P range written in its card
    pressure_step: float = 0.01  # GPa, grid of the first pass
    tolerance: float = 0.1  # deg: a line explains a peak closer than this
    max_phases: int = 3
    candidates_per_phase: int = 5
    min_line_intensity: float = 5.0  # % of the strongest line: weaker lines may explain a peak but are never required
    max_combined_phases: int = 12  # only the phases explaining most alone enter combinations of 2+ phases
    snr_full: float = 10.0  # detected peaks with this SNR or more weigh 1
    cost_missing_line: float = 0.5
    cost_chance: float = 1.0
    cost_phase: float = 0.5
    max_pressure_spread: Optional[float] = 1.5  # GPa, soft (None: phases at any pressure)
    max_pressure_step: Optional[float] = None  # GPa, with `previous`: |P - P_previous| allowed
    cost_phase_change: float = 0.2
    cost_family_change: float = 1.0
    exclude: Optional[Sequence[Tuple[float, float]]] = None  # 2θ zones where lines can be neither seen nor missed
    temperature: float = eos.T_REF
    keep: int = 10  # combinations refined and returned


def family(name: str) -> str:
    """'H2O_Ice_VII' -> 'H2O': polymorphs of one substance."""
    return str(name).split("_", 1)[0]


@dataclass
class Match:
    peak: int  # index in the detected peaks
    phase: str
    reflection: int  # index in card.reflections
    two_theta: float  # expected position at the solution pressure


@dataclass
class Solution:
    pressures: Dict[str, float]  # phase -> pressure (GPa)
    cost: float
    matches: List[Match] = field(default_factory=list)
    unexplained: List[int] = field(default_factory=list)  # detected peaks explained by no line
    missing: Dict[str, List[int]] = field(default_factory=dict)  # strong reflections not observed

    @property
    def phases(self) -> List[str]:
        return list(self.pressures)


@dataclass
class _Phase:
    card: PhaseCard
    reflections: np.ndarray  # indices into card.reflections
    weights: np.ndarray  # relative intensities (strongest = 1), 0 for the lines that are not required
    relative: np.ndarray  # relative intensities (strongest = 1) of every line
    pressures: np.ndarray  # grid
    lines: np.ndarray  # 2θ [n_pressures, n_reflections], NaN outside the window


class LineTables:
    """2θ of every line of every phase on the pressure grid, for one energy and
    2θ window. Independent of the detected peaks: build it once per run."""

    def __init__(self, cards: Dict[str, PhaseCard], energy_ev: float, window: Tuple[float, float],
                 options: Optional[SearchOptions] = None):
        self.options = options or SearchOptions()
        self.energy_ev = energy_ev
        self.window = tuple(window)
        self.phases: Dict[str, _Phase] = {}
        for name, card in cards.items():
            phase = self._tabulate(card)
            if phase is not None:
                self.phases[name] = phase

    def bounds(self, card: PhaseCard) -> Tuple[float, float]:
        lo, hi = self.options.pressure_range
        if self.options.use_card_ranges and card.p_range is not None:
            lo, hi = max(lo, card.p_range[0]), min(hi, card.p_range[1])
        return lo, hi

    def _tabulate(self, card: PhaseCard) -> Optional[_Phase]:
        lo, hi = self.bounds(card)
        if hi < lo or not card.reflections:
            return None
        intensity = np.array([r.intensity for r in card.reflections], dtype=float)
        selected = np.flatnonzero(intensity > 0)
        pressures = np.arange(lo, hi + 0.5 * self.options.pressure_step, self.options.pressure_step)
        lines = card.two_theta_table(pressures, self.energy_ev, self.options.temperature)[:, selected]
        lines[(lines < self.window[0]) | (lines > self.window[1])] = np.nan
        for lo, hi in self.options.exclude or ():
            lines[(lines >= lo) & (lines <= hi)] = np.nan
        valid = np.isfinite(lines).any(axis=1)
        if not valid.any():
            return None
        relative = intensity[selected] / intensity[selected].max()
        weights = np.where(relative < self.options.min_line_intensity / 100, 0.0, relative)
        return _Phase(card, selected, weights, relative, pressures[valid], lines[valid])


class _Problem:
    def __init__(self, peaks, tables: LineTables, options, previous):
        self.options = options
        self.tables = tables
        self.previous = previous or {}
        self.x = np.array([p.position for p in peaks], dtype=float)
        lo, hi = tables.window
        excluded = sum(max(0.0, min(hi, b) - max(lo, a)) for a, b in options.exclude or ())
        self.width = max(hi - lo - excluded, 1e-9)
        self.strength = np.minimum(1.0, np.array([p.snr for p in peaks], dtype=float) / options.snr_full)
        self.phases: Dict[str, _Phase] = {}
        for name, phase in tables.phases.items():
            lo, hi = self._pressure_bounds(name, phase.card)
            rows = (phase.pressures >= lo - 1e-9) & (phase.pressures <= hi + 1e-9)
            if rows.any():
                self.phases[name] = _Phase(phase.card, phase.reflections, phase.weights, phase.relative,
                                           phase.pressures[rows], phase.lines[rows])

    def _pressure_bounds(self, name: str, card: PhaseCard) -> Tuple[float, float]:
        lo, hi = self.tables.bounds(card)
        step = self.options.max_pressure_step
        if name in self.previous and step is not None:
            lo, hi = max(lo, self.previous[name] - step), min(hi, self.previous[name] + step)
        return lo, hi

    # -------------------------------------------------------------- per-phase pass

    def distances(self, lines: np.ndarray) -> np.ndarray:
        """|peak - nearest line| for every peak; lines: [..., n_lines] -> [..., n_peaks]."""
        if self.x.size == 0:
            return np.zeros(lines.shape[:-1] + (0,))
        diff = np.abs(self.x[None, :, None] - lines.reshape(-1, 1, lines.shape[-1]))
        nearest = np.nanmin(np.where(np.isnan(diff), np.inf, diff), axis=2)
        return nearest.reshape(lines.shape[:-1] + (self.x.size,))

    def missing_cost(self, phase: _Phase, lines: np.ndarray) -> np.ndarray:
        """Σ weights of the lines (inside the window) with no detected peak within tol."""
        if self.x.size == 0:
            return np.nansum(np.where(np.isfinite(lines), phase.weights, 0.0), axis=-1)
        gap = np.min(np.abs(lines[..., None] - self.x), axis=-1)
        missing = np.isfinite(lines) & ~(gap <= self.options.tolerance)
        return np.sum(np.where(missing, phase.weights, 0.0), axis=-1)

    def penalty(self, phase: _Phase, lines: np.ndarray) -> np.ndarray:
        """Cost a phase brings by itself: its missing strong lines and its chance matches."""
        n_lines = np.sum(np.isfinite(lines), axis=-1)
        chance = np.minimum(1.0, n_lines * 2 * self.options.tolerance / self.width) * np.sum(self.strength)
        return self.options.cost_missing_line * self.missing_cost(phase, lines) + self.options.cost_chance * chance

    def candidates(self, name: str) -> List[Tuple[float, np.ndarray, float]]:
        """(pressure, distances, own penalty) of the best distinct assignments of one phase."""
        phase = self.phases[name]
        tol = self.options.tolerance
        distances = self.distances(phase.lines)
        explained = distances <= tol
        cost = np.sum(self.strength * np.where(explained, (distances / tol) ** 2, 1.0), axis=1)
        missing = self.penalty(phase, phase.lines)
        cost = cost + missing
        picked, signatures = [], set()
        for i in np.argsort(cost, kind="stable"):
            if not explained[i].any():
                continue
            signature = explained[i].tobytes()
            if signature in signatures:
                continue
            signatures.add(signature)
            picked.append((float(phase.pressures[i]), distances[i], float(missing[i])))
            if len(picked) == self.options.candidates_per_phase:
                break
        return picked

    # -------------------------------------------------------------- combinations

    def combination_cost(self, names: Sequence[str], pressures: Sequence[float], distances, penalties) -> float:
        options = self.options
        nearest = np.min(np.vstack(distances), axis=0) if len(distances) else np.full(self.x.size, np.inf)
        ratio = nearest / options.tolerance
        cost = float(np.sum(self.strength * np.where(ratio <= 1, ratio**2, 1.0)))
        cost += float(sum(penalties)) + options.cost_phase * len(names)
        if options.max_pressure_spread and len(pressures) > 1:
            cost += ((max(pressures) - min(pressures)) / options.max_pressure_spread) ** 2
        cost += self.change_cost(names)
        return cost

    def change_cost(self, names) -> float:
        if not self.previous:
            return 0.0
        options = self.options
        families = {family(n) for n in names} ^ {family(n) for n in self.previous}
        return options.cost_phase_change * len(set(names) ^ set(self.previous)) + options.cost_family_change * len(families)

    def subset_costs(self, names, candidates):
        """combination_cost of every product of the candidates of `names` at once."""
        options = self.options
        grids = np.meshgrid(*[np.arange(len(c)) for c in candidates], indexing="ij")
        index = [g.ravel() for g in grids]
        distances = [np.array([c[1] for c in cand])[i] for cand, i in zip(candidates, index)]  # [n_combo, n_peaks]
        nearest = np.minimum.reduce(distances) if self.x.size else np.zeros((index[0].size, 0))
        ratio = nearest / options.tolerance
        cost = np.sum(self.strength * np.where(ratio <= 1, ratio**2, 1.0), axis=1)
        cost += sum(np.array([c[2] for c in cand])[i] for cand, i in zip(candidates, index))
        cost += options.cost_phase * len(names)
        pressures = np.stack([np.array([c[0] for c in cand])[i] for cand, i in zip(candidates, index)], axis=1)
        if options.max_pressure_spread and len(names) > 1:
            cost += ((pressures.max(axis=1) - pressures.min(axis=1)) / options.max_pressure_spread) ** 2
        cost += self.change_cost(names)
        return zip(cost.tolist(), pressures.tolist())

    def lines_at(self, name: str, pressure: float) -> np.ndarray:
        """2θ of the lines at any pressure, interpolated in the (fine) table."""
        phase = self.phases[name]
        grid = phase.pressures
        if grid.size == 1:
            return phase.lines[0].copy()
        i = int(np.clip(np.searchsorted(grid, pressure) - 1, 0, grid.size - 2))
        frac = (pressure - grid[i]) / (grid[i + 1] - grid[i])
        return phase.lines[i] + frac * (phase.lines[i + 1] - phase.lines[i])

    def evaluate(self, pressures: Dict[str, float]) -> Solution:
        names = list(pressures)
        all_lines = [self.lines_at(n, p) for n, p in pressures.items()]
        distances = [self.distances(lines[None, :])[0] for lines in all_lines]
        penalties = [float(self.penalty(self.phases[n], lines[None, :])[0]) for n, lines in zip(names, all_lines)]
        cost = self.combination_cost(names, list(pressures.values()), distances, penalties)
        matches, unexplained = [], []
        tol = self.options.tolerance
        for j, x in enumerate(self.x):
            # when reflections of several phases overlap, the peak goes to the line expected
            # strongest (a 100 % line of one phase rather than a 5 % line of another)
            best = None
            for name, lines in zip(names, all_lines):
                if np.isfinite(lines).any():
                    k = int(np.nanargmin(np.abs(lines - x)))
                    gap = abs(lines[k] - x)
                    if gap <= tol:
                        score = self.phases[name].relative[k] * (1 - 0.5 * (gap / tol) ** 2)
                        if best is None or score > best[0]:
                            best = (score, name, k, lines[k])
            if best is None:
                unexplained.append(j)
            else:
                matches.append(Match(j, best[1], int(self.phases[best[1]].reflections[best[2]]), float(best[3])))
        missing_lines = {}
        for name, lines in zip(names, all_lines):
            observed = {m.reflection for m in matches if m.phase == name}
            phase = self.phases[name]
            strong = phase.weights >= 0.2
            missing_lines[name] = [int(r) for r, line, s in zip(phase.reflections, lines, strong)
                                   if s and np.isfinite(line) and r not in observed]
        return Solution(dict(pressures), cost, matches, unexplained, missing_lines)

    def refine(self, solution: Solution) -> Solution:
        """Least squares on the matched positions, matching kept fixed."""
        names = [n for n in solution.pressures if any(m.phase == n for m in solution.matches)]
        if not names:
            return solution
        bounds = [self._pressure_bounds(n, self.phases[n].card) for n in names]
        by_phase = {n: [(m.peak, list(self.phases[n].reflections).index(m.reflection)) for m in solution.matches if m.phase == n] for n in names}

        def residuals(values):
            out = []
            for name, pressure in zip(names, values):
                lines = self.lines_at(name, pressure)
                for peak, k in by_phase[name]:
                    out.append(self.x[peak] - lines[k] if np.isfinite(lines[k]) else self.options.tolerance)
            return np.asarray(out) / self.options.tolerance

        start = np.clip([solution.pressures[n] for n in names], [b[0] for b in bounds], [b[1] for b in bounds])
        lower = [b[0] for b in bounds]
        upper = [b[1] + 1e-9 for b in bounds]
        refined = least_squares(residuals, start, bounds=(lower, upper), diff_step=1e-6)
        pressures = dict(solution.pressures)
        pressures.update({n: float(p) for n, p in zip(names, refined.x)})
        candidate = self.evaluate(pressures)
        return candidate if candidate.cost <= solution.cost else solution


def search_composition(
    peaks: Sequence[DetectedPeak],
    tables: LineTables,
    previous: Optional[Dict[str, float]] = None,
) -> List[Solution]:
    """Ranked solutions (best first). `previous` (phase -> pressure of the
    preceding spectrum) favours continuity along a time series."""
    options = tables.options
    problem = _Problem(peaks, tables, options, previous)
    candidates = {name: problem.candidates(name) for name in problem.phases}
    candidates = {name: c for name, c in candidates.items() if c}

    # phases that explain the most on their own enter the multi-phase combinations
    best_alone = {n: min(problem.combination_cost([n], [c[0]], [c[1]], [c[2]]) for c in candidates[n]) for n in candidates}
    names = sorted(candidates)
    combined = sorted(sorted(candidates, key=lambda n: best_alone[n])[: options.max_combined_phases])
    scored = []
    for size in range(1, min(options.max_phases, len(names)) + 1):
        for subset in itertools.combinations(names if size == 1 else combined, size):
            for cost, pressures in problem.subset_costs(subset, [candidates[n] for n in subset]):
                scored.append((cost, subset, pressures))
    scored.sort(key=lambda item: item[0])

    solutions: List[Solution] = []
    seen = set()
    for cost, subset, pressures in scored[: options.keep * 3]:
        solution = problem.refine(problem.evaluate(dict(zip(subset, pressures))))
        key = tuple((n, round(p, 3)) for n, p in sorted(solution.pressures.items()))
        if key not in seen:
            seen.add(key)
            solutions.append(solution)
    solutions.sort(key=lambda s: s.cost)
    return solutions[: options.keep]
