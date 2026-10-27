"""Conversion of the legacy dill files (.CEDX / .CEDXUp) into the HDF5 run format.

This is the only module of the package allowed to import the old code
(`cedapp.drx`): unpickling needs the old classes. Everything else works on
the domain model.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import tempfile
import types
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

LEGACY_EXTENSIONS = (".CEDX", ".CEDXUp", ".CEDUp", ".CEDd")
_LEGACY_PACKAGE = "cedapp.drx"
_LEGACY_MODULES = ("CL_FD_Update", "gauge", "pic", "Calibration", "pressure_law")
_STATES = {"Y": FitState.FITTED, "IN_NOISE": FitState.IN_NOISE, "N": FitState.DISABLED}


def is_legacy_file(path: PathLike) -> bool:
    return Path(path).suffix.lower() in {ext.lower() for ext in LEGACY_EXTENSIONS}


# ----------------------------------------------------------------------------- unpickling


def _import_legacy_modules() -> Dict[str, types.ModuleType]:
    """Import the old modules without their hardware-only parts.

    The LeCroy oscilloscope tools (lecroyscope, tkinter) are not needed to
    read files and may be missing; the old code also forces a Qt matplotlib
    backend on import, which is neutralised here.
    """
    import importlib

    stubs = {"lecroyscope": "lecroyscope", "tkinter": f"{_LEGACY_PACKAGE}.Oscilloscope_LeCroy_QTinterface"}
    for dependency, module in stubs.items():
        try:
            importlib.import_module(dependency)
        except ImportError:
            sys.modules.setdefault(module, types.ModuleType(module))
    import matplotlib

    use_backend, matplotlib.use = matplotlib.use, lambda *args, **kwargs: None
    modules = {}
    try:
        with redirect_stdout(io.StringIO()):  # the old modules print on import
            for name in _LEGACY_MODULES:
                try:
                    modules[name] = importlib.import_module(f"{_LEGACY_PACKAGE}.{name}")
                except ImportError as exc:
                    logger.debug("legacy module %s unavailable: %s", name, exc)
    finally:
        matplotlib.use = use_backend
    return modules


def load_legacy(path: PathLike) -> Any:
    """Unpickle an old dill file whatever the module path it was saved from
    (CL_FD_Update, Bibli_python.CL_FD_Update, src.cedapp.drx..., cedapp.drx...)."""
    import dill

    modules = _import_legacy_modules()

    class _Unpickler(dill.Unpickler):
        def find_class(self, module: str, name: str):
            short = module.rsplit(".", 1)[-1]
            if short in modules:
                for candidate in [modules[short], *modules.values()]:
                    if hasattr(candidate, name):
                        return getattr(candidate, name)
            return super().find_class(module, name)

    with open(path, "rb") as handle, redirect_stdout(io.StringIO()):
        return _Unpickler(handle).load()


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
    import re

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


def _model_curve(gauges: List[Any], x: np.ndarray) -> Optional[np.ndarray]:
    total = np.zeros_like(x, dtype=float)
    found = False
    for gauge in gauges:
        for pic in getattr(gauge, "pics", []):
            try:
                total += pic.model.eval(pic.model.make_params(), x=x)
                found = True
            except Exception:  # an unpicklable/odd lmfit model must not block the conversion
                continue
    return total if found else None


def _poni_text(calib: Any) -> str:
    ai = getattr(calib, "ai", None)
    if ai is not None:
        fd, tmp = tempfile.mkstemp(suffix=".poni")
        os.close(fd)
        try:
            os.remove(tmp)
            ai.write(tmp)
            return Path(tmp).read_text(encoding="utf-8")
        except Exception:
            pass
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
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
    for i, legacy in enumerate(obj.Spectra):
        x = np.asarray(legacy.wnb, dtype=float)
        gauges = list(getattr(legacy, "Gauges", []) or [])
        for gauge in gauges:
            ref = getattr(gauge, "Element_ref", None)
            if ref is not None and gauge.name not in phases:
                phases[gauge.name] = _phase_card(gauge.name, ref)
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
                model=_model_curve(gauges, x),
                fit_range=fit_range,
                phases=[_phase_fit(g) for g in gauges],
            )
        )
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
