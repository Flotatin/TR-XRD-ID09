"""Domain model of a time-resolved XRD run.

Plain dataclasses only: no Qt, no HDF5, no lmfit. Every other layer
(storage, processing, GUI, notebook) speaks this vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from trxrd.physics import eos, lattice

HC_EV_M = 1.239841984e-6  # h*c in eV.m : lambda[m] = HC_EV_M / E[eV]


def energy_to_wavelength(energy_ev: float) -> float:
    return HC_EV_M / float(energy_ev)


def wavelength_to_energy(wavelength_m: float) -> float:
    return HC_EV_M / float(wavelength_m)


@dataclass
class Calibration:
    """Detector geometry and integration settings.

    The .poni content is embedded as text so a saved run never depends on
    the original calibration file being reachable.
    """

    energy_ev: float
    poni: str = ""
    mask: Optional[np.ndarray] = None  # pyFAI convention: non-zero = masked pixel
    two_theta_range: Tuple[float, float] = (0.0, 40.0)
    points_per_degree: Optional[int] = None  # None: matched to the detector pixel size at integration
    flip_axis: Optional[int] = None  # None, 0, 1, or 2 (= both axes)
    sigma_distance_m: float = 0.0
    poni_file: str = ""
    mask_file: str = ""

    @property
    def wavelength_m(self) -> float:
        return energy_to_wavelength(self.energy_ev)

    @property
    def distance_m(self) -> Optional[float]:
        """Sample-detector distance read from the embedded .poni text."""
        for line in self.poni.splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "Distance":
                return float(value)
        return None

    @property
    def n_points(self) -> int:
        if self.points_per_degree is None:
            raise ValueError("points_per_degree not resolved yet (see processing.integration.Integrator)")
        lo, hi = self.two_theta_range
        return max(int(round((hi - lo) * self.points_per_degree)), 2)


@dataclass
class Reflection:
    d: float  # Angstrom, at ambient conditions
    intensity: float
    h: int
    k: int
    l: int

    @property
    def hkl(self) -> Tuple[int, int, int]:
        return (self.h, self.k, self.l)

    @property
    def label(self) -> str:
        return "".join(str(i) for i in self.hkl)


@dataclass
class PhaseCard:
    """Reference description of a crystalline phase (content of a .jcpds card)."""

    name: str
    symmetry: str
    a: float
    b: Optional[float] = None
    c: Optional[float] = None
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0
    v0: Optional[float] = None
    k0: float = 0.0
    k0p: float = 4.0
    alpha_t: Optional[float] = None  # volumetric thermal expansion (1/K)
    reflections: List[Reflection] = field(default_factory=list)
    t_range: Optional[Tuple[float, float]] = None
    p_range: Optional[Tuple[float, float]] = None
    v_min_ratio: float = 0.6
    z_max: Optional[int] = None
    fu: Optional[int] = None
    source_text: str = ""  # original .jcpds text, embedded in saved runs
    source_file: str = ""

    def __post_init__(self) -> None:
        self.symmetry = self.symmetry.strip().upper()
        if self.b is None:
            self.b = self.a
        if self.c is None:
            self.c = self.a
        if self.v0 is None:
            self.v0 = lattice.cell_volume(self.symmetry, self.cell)

    @property
    def cell(self) -> lattice.Cell:
        return lattice.Cell(self.a, self.b, self.c, self.alpha, self.beta, self.gamma)

    def v0_at(self, temperature: float = eos.T_REF) -> float:
        return eos.v0_at_temperature(self.v0, self.alpha_t, temperature)

    def pressure_from_volume(self, volume: float, temperature: float = eos.T_REF) -> float:
        return float(eos.birch_murnaghan(volume, self.v0_at(temperature), self.k0, self.k0p))

    def volume_from_pressure(self, pressure: float, temperature: float = eos.T_REF) -> float:
        return eos.volume_from_pressure(
            pressure, self.v0_at(temperature), self.k0, self.k0p, v_min_ratio=self.v_min_ratio
        )

    def cell_at(self, pressure: float, temperature: float = eos.T_REF) -> lattice.Cell:
        """Isotropic compression of the reference cell to the volume given by the EoS."""
        return lattice.scale_cell_to_volume(
            self.symmetry, self.cell, self.volume_from_pressure(pressure, temperature)
        )

    def two_theta_lines(
        self, pressure: float, energy_ev: float, temperature: float = eos.T_REF, max_lines: int = 30
    ) -> List[Tuple[float, Reflection]]:
        """Expected (2theta in degrees, reflection) at the given P and T."""
        cell = self.cell_at(pressure, temperature)
        wavelength = energy_to_wavelength(energy_ev)
        lines = []
        for ref in self.reflections[:max_lines]:
            d = lattice.d_spacing(self.symmetry, cell, ref.hkl)
            lines.append((lattice.two_theta_from_d(d, wavelength), ref))
        return lines


@dataclass
class Peak:
    """One fitted (or initialised) diffraction peak."""

    name: str
    model: str = "PearsonIV"  # PseudoVoigt, PearsonIV, Gaussian, Moffat, SplitLorentzian
    center: float = np.nan  # 2theta (deg)
    center_err: float = np.nan
    height: float = np.nan
    height_err: float = np.nan
    width: float = np.nan  # profile "sigma" parameter (deg)
    width_err: float = np.nan
    shape: Tuple[float, ...] = ()  # model-specific: fraction | (expon, skew) | beta | sigma_r
    hkl: Optional[Tuple[int, int, int]] = None
    inter: float = 3.0  # relative freedom of the bounds when refitting
    enabled: bool = True


class FitState:
    FITTED = "fitted"
    IN_NOISE = "in_noise"  # phase present in the model but no measurable signal
    DISABLED = "disabled"


@dataclass
class PhaseFit:
    """A phase identified in one spectrum, with its refined cell and pressure."""

    phase: str  # key in Run.phases
    state: str = FitState.FITTED
    a: float = np.nan
    a_err: float = np.nan
    b: float = np.nan
    b_err: float = np.nan
    c: float = np.nan
    c_err: float = np.nan
    volume: float = np.nan
    volume_err: float = np.nan
    pressure: float = np.nan
    pressure_err: float = np.nan
    temperature: float = np.nan
    temperature_err: float = np.nan
    peaks: List[Peak] = field(default_factory=list)


@dataclass
class Processing:
    """1D treatment applied before fitting."""

    baseline_degree: int = 1
    filter_type: str = "svg"  # svg | fft | none
    filter_params: Tuple[float, float] = (9, 2)


@dataclass
class Spectrum:
    """One integrated pattern I(2theta) and everything derived from it."""

    two_theta: np.ndarray
    intensity: np.ndarray
    time: float = np.nan  # s, relative to the run time base
    frame: int = -1  # index in the raw image stack, -1 if not from a stack
    label: str = ""
    processing: Processing = field(default_factory=Processing)
    baseline: Optional[np.ndarray] = None
    model: Optional[np.ndarray] = None  # total fitted curve (baseline excluded)
    fit_range: Optional[Tuple[float, float]] = None
    phases: List[PhaseFit] = field(default_factory=list)

    @property
    def corrected(self) -> np.ndarray:
        if self.baseline is None:
            return np.asarray(self.intensity)
        return np.asarray(self.intensity) - np.asarray(self.baseline)

    def phase_fit(self, phase: str) -> Optional[PhaseFit]:
        for fit in self.phases:
            if fit.phase == phase:
                return fit
        return None


@dataclass
class Source:
    """Where the raw data come from (information + optional HDF5 external link)."""

    raw_file: str = ""
    raw_dataset: str = ""  # path of the image stack inside raw_file
    master_file: str = ""  # BLISS master file (scalars, oscilloscope channels)
    scan: str = ""  # BLISS scan label, e.g. "27.1"


@dataclass
class Oscilloscope:
    time: np.ndarray
    channels: Dict[str, np.ndarray] = field(default_factory=dict)
    trigger_channel: str = ""


@dataclass
class Run:
    """A full experiment: a sequence of spectra plus everything needed to reproduce it."""

    name: str
    spectra: List[Spectrum] = field(default_factory=list)
    calibration: Optional[Calibration] = None
    phases: Dict[str, PhaseCard] = field(default_factory=dict)
    source: Source = field(default_factory=Source)
    oscilloscope: Optional[Oscilloscope] = None
    note: str = ""
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    ui_state: Dict[str, Any] = field(default_factory=dict)  # free JSON-able GUI state

    def __len__(self) -> int:
        return len(self.spectra)

    @property
    def times(self) -> np.ndarray:
        return np.array([s.time for s in self.spectra], dtype=float)

    def phase_names(self) -> List[str]:
        """Phases used in fits (in order of first appearance), then unused library phases."""
        names: List[str] = []
        for spectrum in self.spectra:
            for fit in spectrum.phases:
                if fit.phase not in names:
                    names.append(fit.phase)
        names.extend(name for name in self.phases if name not in names)
        return names
