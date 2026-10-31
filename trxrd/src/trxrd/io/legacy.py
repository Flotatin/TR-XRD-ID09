"""Conversion of the legacy dill files (.CEDX / .CEDXUp) into the HDF5 run format.

The old program does NOT need to be installed: every class or function of
the old code (and of lmfit, pyFAI, PyQt...) found in a file is replaced by a
passive `LegacyObject` that only keeps the pickled attributes. Only numpy,
pandas and a few builtins are rebuilt for real, and no pickled function is
ever executed. This narrows the risks of unpickling, but convert only files
you trust anyway.
"""

from __future__ import annotations

import logging
import pickle
import re
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from trxrd.domain import (
    Calibration,
    FitState,
    Oscilloscope,
    Peak,
    PhaseCard,
    PhaseFit,
    Processing,
    Reflection,
    Run,
    Source,
    Spectrum,
)
from trxrd.io.jcpds import parse_jcpds
from trxrd.physics import profiles

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

LEGACY_EXTENSIONS = (".CEDX", ".CEDXUp", ".CEDUp", ".CEDd")
_STATES = {"Y": FitState.FITTED, "IN_NOISE": FitState.IN_NOISE, "N": FitState.DISABLED}


def is_legacy_file(path: PathLike) -> bool:
    return Path(path).suffix.lower() in {ext.lower() for ext in LEGACY_EXTENSIONS}


# ----------------------------------------------------------------------------- unpickling


class LegacyObject:
    """Passive stand-in for anything of the old program: state lands in __dict__."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return LegacyObject()

    def __setstate__(self, state):
        if isinstance(state, tuple) and len(state) == 2 and all(p is None or isinstance(p, dict) for p in state):
            state = {**(state[0] or {}), **(state[1] or {})}  # (__dict__, __slots__) form
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_state"] = state

    # list and dict subclasses are refilled with APPENDS / SETITEMS
    def append(self, item):
        self.__dict__.setdefault("_items", []).append(item)

    def extend(self, items):
        self.__dict__.setdefault("_items", []).extend(items)

    def __setitem__(self, key, value):
        self.__dict__.setdefault("_mapping", {})[key] = value


_STUBS: Dict[Tuple[str, str], type] = {}


def _stub(module: str, name: str) -> type:
    if (module, name) not in _STUBS:
        attrs = {"__module__": module, "__qualname__": name}
        _STUBS[(module, name)] = type(name.rsplit(".", 1)[-1], (LegacyObject,), attrs)
    return _STUBS[(module, name)]


def _safe_getattr(obj, name, *default):
    try:
        return getattr(obj, name, *default)
    except AttributeError:
        return _stub(type(obj).__module__, name)()


def _create_array(f, args, state, npdict=None):
    """dill's numpy array reduction (`f` is numpy's own reconstructor)."""
    array = f(*args)
    array.__setstate__(state)
    return array


_TRUSTED_ROOTS = {"numpy", "pandas", "collections", "copyreg", "_codecs", "datetime"}
_SAFE_TYPES = {
    t.__name__.lower(): t
    for t in (object, set, frozenset, list, dict, tuple, complex, bytearray, bytes, slice, range, int, float, str, bool)
}
_SAFE_TYPES.update({"none": type(None), "ellipsis": type(Ellipsis), "notimplemented": type(NotImplemented)})


def _load_type(name: str):
    """dill's reference to a builtin type ('slice', 'NoneType', 'SliceType'...): data types only."""
    key = name.lower()
    key = key[:-4] if key.endswith("type") and key[:-4] in _SAFE_TYPES else key
    return _SAFE_TYPES.get(key) or _stub("dill._dill", name)


_DILL = {
    "_create_array": _create_array,
    "_get_attr": _safe_getattr,
    "_import_module": lambda name, *args: types.ModuleType(name),
    "_load_type": _load_type,
}

try:  # pandas' own unpickler fixes objects saved with older pandas versions
    from pandas.compat.pickle_compat import Unpickler as _BaseUnpickler
    from pandas.compat.pickle_compat import _class_locations_map as _PANDAS_MOVES
except ImportError:  # pragma: no cover - depends on the pandas version
    _BaseUnpickler, _PANDAS_MOVES = pickle.Unpickler, {}


class _LegacyUnpickler(_BaseUnpickler):
    def find_class(self, module: str, name: str):
        module, name = _PANDAS_MOVES.get((module, name), (module, name))
        if module in ("builtins", "__builtin__"):
            if name == "getattr":
                return _safe_getattr
            return _SAFE_TYPES.get(name) or _stub(module, name)
        if module in ("dill._dill", "dill.dill"):
            return _DILL.get(name) or _stub(module, name)
        if module.split(".")[0] in _TRUSTED_ROOTS:
            return super().find_class(module, name)
        return _stub(module, name)


def load_legacy(path: PathLike) -> Any:
    """Read an old dill file into LegacyObjects, whatever the module path it was
    saved from (CL_FD_Update, Bibli_python..., src.cedapp.drx..., cedapp.drx...)."""
    with open(path, "rb") as handle:
        return _LegacyUnpickler(handle).load()


# ----------------------------------------------------------------------------- legacy -> domain


def _f(value: Any) -> float:
    try:
        value = np.asarray(value, dtype=float).ravel()
        return float(value[0]) if value.size else np.nan
    except (TypeError, ValueError):
        return np.nan


def _half_width(bounds: Any) -> float:
    try:
        lo, hi = bounds
        return abs(float(hi) - float(lo)) / 2
    except (TypeError, ValueError):
        return np.nan


def _jcpds_text(element_ref: Any) -> str:
    frame = getattr(element_ref, "file", None)
    if frame is None:
        return ""
    lines = []
    for _, row in frame.iterrows():
        values = [str(v).strip() for v in row.tolist() if v is not None and str(v) != "nan"]
        if values:
            lines.append(values[0] + ": " + ":".join(values[1:]))
    return "\n".join(lines)


def _phase_card(name: str, ref: Any) -> PhaseCard:
    """PhaseCard from the old Element_Bibli attributes (what was really used)."""
    reflections: List[Reflection] = []
    dhkl = getattr(ref, "Dhkl", None)
    if dhkl is not None:
        for d, i, h, k, l in dhkl[["Dhkl", "I", "h", "k", "l"]].itertuples(index=False):
            reflections.append(Reflection(float(d), float(i), int(h), int(k), int(l)))
    text = _jcpds_text(ref)
    try:
        return PhaseCard(
            name=name,
            symmetry=str(ref.symmetrie),
            a=float(ref.A),
            b=None if ref.B is None else float(ref.B),
            c=None if ref.C is None else float(ref.C),
            alpha=float(ref.ALPHA or 90.0),
            beta=float(ref.BETA or 90.0),
            gamma=float(ref.GAMMA or 90.0),
            v0=None if ref.V0 is None else float(ref.V0),
            k0=float(ref.K0 or 0.0),
            k0p=float(4.0 if ref.K0P is None else ref.K0P),
            alpha_t=getattr(ref, "ALPHAKT", None),
            reflections=reflections,
            t_range=getattr(ref, "T_range", None),
            p_range=getattr(ref, "P_range", None),
            v_min_ratio=float(getattr(ref, "Vmin", 0.6) or 0.6),
            source_text=text,
        )
    except (AttributeError, TypeError, ValueError):
        card = parse_jcpds(text)
        card.name = name
        return card


def _hkl(element: Any, index: int) -> Optional[tuple]:
    try:
        label = element.pic_ref[index][0]
    except (AttributeError, IndexError, TypeError):
        return None
    match = re.match(r"D([-+]?\d)([-+]?\d)([-+]?\d)$", str(label))
    return tuple(int(g) for g in match.groups()) if match else None


def _peak(pic: Any, hkl: Optional[tuple]) -> Peak:
    center_err = getattr(pic, "sigma_ctr_total", None) or getattr(pic, "sigma_ctr_stat", None)
    return Peak(
        name=str(pic.name),
        model=str(getattr(pic, "model_fit", "PseudoVoigt")),
        center=_f(pic.ctr[0]),
        center_err=_f(center_err) if center_err else _half_width(pic.ctr[1]),
        height=_f(pic.ampH[0]),
        height_err=_half_width(pic.ampH[1]),
        width=_f(pic.sigma[0]),
        width_err=_half_width(pic.sigma[1]),
        shape=tuple(_f(c[0]) for c in getattr(pic, "coef_spe", [])),
        hkl=hkl,
        inter=_f(getattr(pic, "inter", 3.0)),
    )


def _phase_fit(gauge: Any) -> PhaseFit:
    g = lambda attr: _f(getattr(gauge, attr, np.nan))  # noqa: E731
    return PhaseFit(
        phase=str(gauge.name),
        state=_STATES.get(str(getattr(gauge, "state", "Y")), FitState.FITTED),
        a=g("a"), a_err=g("sigma_a"), b=g("b"), b_err=g("sigma_b"), c=g("c"), c_err=g("sigma_c"),
        volume=g("V"), volume_err=g("sigma_V"),
        pressure=g("P"), pressure_err=_f(getattr(gauge, "sigma_P", getattr(gauge, "sigmaP", np.nan))),
        temperature=g("T"), temperature_err=g("sigma_T"),
        peaks=[_peak(p, _hkl(gauge, j)) for j, p in enumerate(getattr(gauge, "pics", []))],
    )


def _model_curve(fits: List[PhaseFit], x: np.ndarray) -> Optional[np.ndarray]:
    """Fitted curve rebuilt from the peak parameters (same conventions as the old lmfit models)."""
    peaks = [p for f in fits for p in f.peaks if np.isfinite([p.center, p.height, p.width]).all()]
    if not peaks:
        return None
    return sum(profiles.evaluate(p.model, x, p.center, p.height, p.width, p.shape) for p in peaks)


def _poni_text(calib: Any) -> str:
    """The integrator itself was never saved: re-read the .poni if it is still there."""
    poni_file = getattr(calib, "file_poni", None)
    if poni_file and Path(poni_file).is_file():
        return Path(poni_file).read_text(encoding="utf-8", errors="replace")
    return ""


def _calibration(obj: Any, energy: float) -> Optional[Calibration]:
    calib = getattr(obj, "calib", None)
    if calib is None:
        return None
    theta_range = getattr(calib, "theta_range", None) or (0.0, 40.0)
    mask = getattr(calib, "mask", None)
    return Calibration(
        energy_ev=float(getattr(calib, "energy", None) or energy),
        poni=_poni_text(calib),
        mask=None if mask is None else (np.asarray(mask) > 0).astype(np.uint8),
        two_theta_range=(float(theta_range[0]), float(theta_range[-1])),
        sigma_distance_m=float(getattr(obj, "sigma_dist", 0.0) or 0.0),
        poni_file=str(getattr(calib, "file_poni", "") or ""),
        mask_file=str(getattr(calib, "file_mask", "") or ""),
    )


def _oscilloscope(obj: Any) -> Optional[Oscilloscope]:
    frame = getattr(obj, "data_oscillo", None)
    if frame is None or not hasattr(frame, "columns") or frame.shape[1] < 2:
        return None
    columns = list(frame.columns)
    trigger = getattr(obj, "time_index", "")
    return Oscilloscope(
        time=frame[columns[0]].to_numpy(dtype=float),
        channels={str(c): frame[c].to_numpy(dtype=float) for c in columns[1:]},
        trigger_channel=str(trigger) if trigger in columns else "",
    )


def legacy_to_run(obj: Any, name: str) -> Run:
    if not hasattr(obj, "Spectra"):
        raise TypeError(f"{type(obj).__name__} is not a CED_DRX-like object (no .Spectra)")
    times = getattr(obj, "Time_spectrum", None)
    times = [] if times is None else list(times)
    energy = next((s.E for s in obj.Spectra if getattr(s, "E", None)), np.nan)
    phases: Dict[str, PhaseCard] = {}
    spectra = []
    skipped = set()
    for i, legacy in enumerate(obj.Spectra):
        x = np.asarray(legacy.wnb, dtype=float)
        gauges = []
        for gauge in getattr(legacy, "Gauges", []) or []:
            ref = getattr(gauge, "Element_ref", None)
            if ref is None:  # spectroscopy gauge (ruby, Sm...): not part of the new program
                skipped.add(str(getattr(gauge, "name", "?")))
                continue
            gauges.append(gauge)
            phase_name = str(gauge.name)  # old files mix str and numpy.str_ names
            if phase_name not in phases:
                phases[phase_name] = _phase_card(phase_name, ref)
        fits = [_phase_fit(g) for g in gauges]
        index_x = getattr(legacy, "indexX", None)
        fit_range = None
        if index_x is not None and len(index_x):
            fit_range = (float(np.min(x[index_x])), float(np.max(x[index_x])))
        param_f = getattr(legacy, "param_f", (9, 2))
        baseline = getattr(legacy, "blfit", None)
        spectra.append(
            Spectrum(
                two_theta=x,
                intensity=np.asarray(legacy.spec, dtype=float),
                time=float(times[i]) if i < len(times) and times[i] is not None else np.nan,
                frame=i,
                label=f"drx_{i}",
                processing=Processing(
                    baseline_degree=int(getattr(legacy, "deg_baseline", 1) or 0),
                    filter_type=str(getattr(legacy, "type_filtre", "svg")),
                    filter_params=(float(param_f[0]), float(param_f[1])),
                ),
                baseline=None if baseline is None else np.asarray(baseline, dtype=float),
                model=_model_curve(fits, x),
                fit_range=fit_range,
                phases=fits,
            )
        )
    if skipped:
        logger.warning("%s: spectroscopy gauges ignored: %s", name, ", ".join(sorted(skipped)))
    data_drx = getattr(obj, "data_drx", None)
    return Run(
        name=name,
        spectra=spectra,
        calibration=_calibration(obj, energy),
        phases=phases,
        source=Source(raw_file=str(data_drx) if isinstance(data_drx, (str, Path)) else ""),
        oscilloscope=_oscilloscope(obj),
        note=str(getattr(obj, "note", "") or ""),
    )


def convert_file(src: PathLike, dst: Optional[PathLike] = None) -> Path:
    """Convert one legacy file; default output: same folder, `<name>.trxrd.h5`."""
    from trxrd.io.h5store import save_run

    src = Path(src)
    dst = Path(dst) if dst else src.with_name(src.stem + ".trxrd.h5")
    run = legacy_to_run(load_legacy(src), name=src.stem)
    return save_run(run, dst, link_raw=False)
