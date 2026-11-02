"""Use case: compare an automatic analysis with a reference one (e.g. a v1 run analysed by hand)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

from trxrd.domain import Run


@dataclass
class SpectrumComparison:
    index: int
    reference: Dict[str, float]  # phase -> pressure (GPa)
    found: Dict[str, float]

    @property
    def same_phases(self) -> bool:
        return set(self.reference) == set(self.found)


@dataclass
class ComparisonSummary:
    spectra: List[SpectrumComparison]

    @property
    def agreement(self) -> int:
        """Number of spectra with exactly the same phases."""
        return sum(s.same_phases for s in self.spectra)

    def max_pressure_difference(self) -> Dict[str, float]:
        """Largest |P_found - P_reference| per phase, over the spectra where both have it."""
        out: Dict[str, float] = {}
        for s in self.spectra:
            for phase, pressure in s.found.items():
                if phase in s.reference and np.isfinite(pressure) and np.isfinite(s.reference[phase]):
                    out[phase] = max(out.get(phase, 0.0), abs(pressure - s.reference[phase]))
        return out


def _pressures(run: Run, index: int) -> Dict[str, float]:
    return {f.phase: f.pressure for f in run.spectra[index].phases}


def compare_runs(reference: Run, analysed: Run, indices: Optional[Iterable[int]] = None) -> ComparisonSummary:
    """Spectrum by spectrum: same phases? pressure differences?"""
    if len(reference) != len(analysed):
        raise ValueError(f"{len(reference)} spectra in the reference, {len(analysed)} analysed")
    indices = range(len(reference)) if indices is None else indices
    return ComparisonSummary([SpectrumComparison(i, _pressures(reference, i), _pressures(analysed, i)) for i in indices])
