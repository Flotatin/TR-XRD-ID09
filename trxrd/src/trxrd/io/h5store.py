"""Persistence of a Run in a self-describing HDF5 (NeXus-flavoured) file.

Layout (format "trxrd-run", see docs/ARCHITECTURE.md for the full spec)::

    /                       NXroot   file_format, format_version, creator
    /entry                  NXentry  title, note, created, ui_state (JSON)
      source/               NXcollection  raw_file, raw_dataset, master_file, scan
        raw_data            -> ExternalLink to the raw image stack (optional)
      calibration/          NXcollection  energy_ev, poni (text), mask, ...
      phases/<name>/        NXcollection  card parameters (attrs) + jcpds text + reflections table
      oscilloscope/         NXdata   time + channels
      spectra/              NXdata   intensity[n, npt], two_theta, baseline, model, time, ...
        processing/         table    one row per spectrum
      fits/phases/          table    one row per (spectrum, phase)
      fits/peaks/           table    one row per peak, `fit` = row in fits/phases
      results/              NXdata   P(t) per phase (derived, for quick plotting)

Plain arrays and tables only: the file opens in silx view / h5web / MATLAB /
any h5py script, without this package and without pickling.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import h5py
import numpy as np

from trxrd import __version__
from trxrd.domain import (
    Calibration,
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
from trxrd.io.h5utils import (
    nan_if_none,
    none_if_nan,
    nx_group,
    read_scalar,
    read_str,
    read_table,
    write_array,
    write_str,
    write_strings,
    write_table,
)

PathLike = Union[str, Path]

FORMAT_NAME = "trxrd-run"
FORMAT_VERSION = "1.1"  # 1.1: processing/baseline_window
ENTRY = "entry"

PHASE_FIT_FIELDS = (
    "a", "a_err", "b", "b_err", "c", "c_err",
    "volume", "volume_err", "pressure", "pressure_err", "temperature", "temperature_err",
)
PEAK_FIELDS = ("center", "center_err", "height", "height_err", "width", "width_err", "inter")


class FormatError(ValueError):
    pass


# ----------------------------------------------------------------------------- save


def save_run(run: Run, path: PathLike, *, link_raw: bool = True) -> Path:
    """Write `run` to `path` atomically (temporary file + rename)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with h5py.File(tmp, "w") as f:
        f.attrs["NX_class"] = "NXroot"
        f.attrs["default"] = ENTRY
        f.attrs["file_format"] = FORMAT_NAME
        f.attrs["format_version"] = FORMAT_VERSION
        f.attrs["creator"] = f"trxrd {__version__}"
        entry = nx_group(f, ENTRY, "NXentry")
        write_str(entry, "title", run.name)
        write_str(entry, "note", run.note)
        write_str(entry, "created", run.created)
        write_str(entry, "ui_state", json.dumps(run.ui_state))
        _write_source(entry, run.source, link_raw)
        if run.calibration is not None:
            _write_calibration(entry, run.calibration)
        _write_phases(entry, run.phases)
        if run.oscilloscope is not None:
            _write_oscilloscope(entry, run.oscilloscope)
        _write_spectra(entry, run.spectra)
        _write_fits(entry, run.spectra)
        _write_results(entry, run)
        entry.attrs["default"] = "results" if "results" in entry else "spectra"
    os.replace(tmp, path)
    return path


def _write_source(entry: h5py.Group, source: Source, link_raw: bool) -> None:
    group = nx_group(entry, "source", "NXcollection")
    for name in ("raw_file", "raw_dataset", "master_file", "scan"):
        write_str(group, name, getattr(source, name))
    if link_raw and source.raw_file and source.raw_dataset:
        group["raw_data"] = h5py.ExternalLink(str(Path(source.raw_file).resolve()), source.raw_dataset)


def _write_calibration(entry: h5py.Group, calib: Calibration) -> None:
    group = nx_group(entry, "calibration", "NXcollection")
    group["energy_ev"] = float(calib.energy_ev)
    group["energy_ev"].attrs["units"] = "eV"
    group["wavelength_m"] = calib.wavelength_m
    group["wavelength_m"].attrs["units"] = "m"
    write_str(group, "poni", calib.poni)
    write_str(group, "poni_file", calib.poni_file)
    write_str(group, "mask_file", calib.mask_file)
    group["two_theta_range"] = np.asarray(calib.two_theta_range, dtype=float)
    group["points_per_degree"] = -1 if calib.points_per_degree is None else int(calib.points_per_degree)
    group["flip_axis"] = -1 if calib.flip_axis is None else int(calib.flip_axis)
    group["sigma_distance_m"] = float(calib.sigma_distance_m)
    if calib.mask is not None:
        dataset = write_array(group, "mask", np.asarray(calib.mask, dtype=np.uint8))
        dataset.attrs["convention"] = "pyFAI: non-zero = masked pixel"


def _write_phases(entry: h5py.Group, phases: Dict[str, PhaseCard]) -> None:
    root = nx_group(entry, "phases", "NXcollection")
    root.attrs["order"] = [str(k) for k in phases]  # HDF5 lists groups alphabetically
    for key, card in phases.items():
        group = nx_group(root, str(key), "NXcollection")
        group.attrs["name"] = str(card.name)
        group.attrs["symmetry"] = str(card.symmetry)
        for attr in ("a", "b", "c", "alpha", "beta", "gamma", "v0", "k0", "k0p", "alpha_t", "v_min_ratio"):
            group.attrs[attr] = nan_if_none(getattr(card, attr))
        for attr in ("t_range", "p_range"):
            value = getattr(card, attr)
            group.attrs[attr] = np.asarray(value if value is not None else (np.nan, np.nan), dtype=float)
        for attr in ("z_max", "fu"):
            value = getattr(card, attr)
            group.attrs[attr] = -1 if value is None else int(value)
        write_str(group, "jcpds", card.source_text)
        write_str(group, "source_file", card.source_file)
        refl = card.reflections
        write_table(
            group,
            "reflections",
            {
                "d": np.array([r.d for r in refl], dtype=float),
                "intensity": np.array([r.intensity for r in refl], dtype=float),
                "h": np.array([r.h for r in refl], dtype=int),
                "k": np.array([r.k for r in refl], dtype=int),
                "l": np.array([r.l for r in refl], dtype=int),
            },
        )


def _write_oscilloscope(entry: h5py.Group, osc: Oscilloscope) -> None:
    names = list(osc.channels)
    signal = osc.trigger_channel if osc.trigger_channel in osc.channels else (names[0] if names else "time")
    group = nx_group(entry, "oscilloscope", "NXdata", signal=signal, axes="time")
    group.attrs["channels"] = [str(n) for n in names]
    group.attrs["trigger_channel"] = str(osc.trigger_channel)
    group["time"] = np.asarray(osc.time, dtype=float)
    for name, values in osc.channels.items():
        group[name] = np.asarray(values, dtype=float)


def _stack(rows: List[Optional[np.ndarray]], width: int) -> Optional[np.ndarray]:
    if all(row is None for row in rows):
        return None
    out = np.full((len(rows), width), np.nan)
    for i, row in enumerate(rows):
        if row is not None:
            row = np.asarray(row, dtype=float)
            out[i, : row.size] = row
    return out


def _write_spectra(entry: h5py.Group, spectra: List[Spectrum]) -> None:
    group = nx_group(entry, "spectra", "NXdata", signal="intensity")
    n_points = np.array([np.asarray(s.two_theta).size for s in spectra], dtype=int)
    width = int(n_points.max()) if len(spectra) else 0
    shared_axis = len(spectra) > 0 and all(
        n == n_points[0] and np.array_equal(s.two_theta, spectra[0].two_theta) for n, s in zip(n_points, spectra)
    )
    if shared_axis:
        group["two_theta"] = np.asarray(spectra[0].two_theta, dtype=float)
        group.attrs["axes"] = [".", "two_theta"]
    else:
        write_array(group, "two_theta", _stack([s.two_theta for s in spectra], width) if spectra else np.zeros((0, 0)))
        group.attrs["axes"] = [".", "."]
    group["two_theta"].attrs["units"] = "deg"
    write_array(group, "intensity", _stack([s.intensity for s in spectra], width) if spectra else np.zeros((0, 0)))
    for name in ("baseline", "model"):
        stacked = _stack([getattr(s, name) for s in spectra], width)
        if stacked is not None:
            write_array(group, name, stacked)
    group["n_points"] = n_points
    group["time"] = np.array([s.time for s in spectra], dtype=float)
    group["time"].attrs["units"] = "s"
    group["frame"] = np.array([s.frame for s in spectra], dtype=int)
    write_strings(group, "label", [s.label for s in spectra])
    group["fit_range"] = np.array(
        [s.fit_range if s.fit_range is not None else (np.nan, np.nan) for s in spectra], dtype=float
    ).reshape(len(spectra), 2)
    write_table(
        group,
        "processing",
        {
            "baseline_degree": np.array([s.processing.baseline_degree for s in spectra], dtype=int),
            "filter_type": [s.processing.filter_type for s in spectra],
            "filter_p1": np.array([s.processing.filter_params[0] for s in spectra], dtype=float),
            "filter_p2": np.array([s.processing.filter_params[1] for s in spectra], dtype=float),
            "baseline_window": np.array([np.nan if s.processing.baseline_window is None else s.processing.baseline_window
                                         for s in spectra], dtype=float),
        },
    )


def _window(processing: Dict[str, np.ndarray], i: int) -> Optional[float]:
    """SNIP window of spectrum i (format 1.1); files 1.0 have none: automatic."""
    values = processing.get("baseline_window")
    return None if values is None or not np.isfinite(values[i]) else float(values[i])


def _write_fits(entry: h5py.Group, spectra: List[Spectrum]) -> None:
    fits = nx_group(entry, "fits", "NXcollection")
    phase_rows: Dict[str, list] = {k: [] for k in ("spectrum", "phase", "state", *PHASE_FIT_FIELDS)}
    peak_rows: Dict[str, list] = {
        k: [] for k in ("fit", "spectrum", "phase", "name", "model", *PEAK_FIELDS, "shape", "hkl", "enabled")
    }
    fit_index = 0
    for i, spectrum in enumerate(spectra):
        for fit in spectrum.phases:
            phase_rows["spectrum"].append(i)
            phase_rows["phase"].append(fit.phase)
            phase_rows["state"].append(fit.state)
            for field in PHASE_FIT_FIELDS:
                phase_rows[field].append(nan_if_none(getattr(fit, field)))
            for peak in fit.peaks:
                peak_rows["fit"].append(fit_index)
                peak_rows["spectrum"].append(i)
                peak_rows["phase"].append(fit.phase)
                peak_rows["name"].append(peak.name)
                peak_rows["model"].append(peak.model)
                for field in PEAK_FIELDS:
                    peak_rows[field].append(nan_if_none(getattr(peak, field)))
                peak_rows["shape"].append(tuple(float(v) for v in peak.shape))
                peak_rows["hkl"].append(peak.hkl)
                peak_rows["enabled"].append(bool(peak.enabled))
            fit_index += 1

    table = {k: phase_rows[k] for k in ("spectrum", "phase", "state")}
    table["spectrum"] = np.asarray(table["spectrum"], dtype=int)
    table.update({k: np.asarray(phase_rows[k], dtype=float) for k in PHASE_FIT_FIELDS})
    write_table(fits, "phases", table)

    n_shape = max((len(s) for s in peak_rows["shape"]), default=0)
    shapes = np.full((len(peak_rows["shape"]), n_shape), np.nan)
    for row, values in enumerate(peak_rows["shape"]):
        shapes[row, : len(values)] = values
    hkl = np.array([h if h is not None else (0, 0, 0) for h in peak_rows["hkl"]], dtype=int).reshape(-1, 3)
    table = {
        "fit": np.asarray(peak_rows["fit"], dtype=int),
        "spectrum": np.asarray(peak_rows["spectrum"], dtype=int),
        "phase": peak_rows["phase"],
        "name": peak_rows["name"],
        "model": peak_rows["model"],
    }
    table.update({k: np.asarray(peak_rows[k], dtype=float) for k in PEAK_FIELDS})
    table.update({f"shape_{j}": shapes[:, j] for j in range(n_shape)})
    table.update(
        {
            "has_hkl": np.array([h is not None for h in peak_rows["hkl"]], dtype=bool),
            "h": hkl[:, 0],
            "k": hkl[:, 1],
            "l": hkl[:, 2],
            "enabled": np.asarray(peak_rows["enabled"], dtype=bool),
        }
    )
    write_table(fits, "peaks", table)


def _write_results(entry: h5py.Group, run: Run) -> None:
    """Derived P(t) per phase, laid out so that silx/h5web plot it directly."""
    names = [n for n in run.phase_names() if any(s.phase_fit(n) for s in run.spectra)]
    if not names:
        return
    times = run.times
    use_time = bool(np.isfinite(times).any())
    axis = times if use_time else np.arange(len(run.spectra), dtype=float)
    group = nx_group(entry, "results", "NXdata", axes="time" if use_time else "spectrum")
    group[("time" if use_time else "spectrum")] = axis
    signals = []
    for name in names:
        pressure = np.full(len(run.spectra), np.nan)
        error = np.full(len(run.spectra), np.nan)
        for i, spectrum in enumerate(run.spectra):
            fit = spectrum.phase_fit(name)
            if fit is not None:
                pressure[i], error[i] = fit.pressure, fit.pressure_err
        key = f"pressure_{name}"
        group[key] = pressure
        group[key].attrs["units"] = "GPa"
        group[key + "_errors"] = error
        signals.append(key)
    group.attrs["signal"] = signals[0]
    if len(signals) > 1:
        group.attrs["auxiliary_signals"] = signals[1:]


# ----------------------------------------------------------------------------- load


def is_run_file(path: PathLike) -> bool:
    try:
        with h5py.File(path, "r") as f:
            return f.attrs.get("file_format") == FORMAT_NAME
    except (OSError, ValueError):
        return False


def check_format(f: h5py.File) -> str:
    if f.attrs.get("file_format") != FORMAT_NAME:
        raise FormatError(f"{f.filename} is not a {FORMAT_NAME} file")
    version = str(f.attrs.get("format_version", "0"))
    if version.split(".")[0] != FORMAT_VERSION.split(".")[0]:
        raise FormatError(f"{f.filename}: unsupported format version {version} (expected {FORMAT_VERSION})")
    return version


def load_run(path: PathLike) -> Run:
    with h5py.File(path, "r") as f:
        check_format(f)
        entry = f[ENTRY]
        spectra = _read_spectra(entry["spectra"])
        _read_fits(entry["fits"], spectra)
        return Run(
            name=read_str(entry, "title"),
            note=read_str(entry, "note"),
            created=read_str(entry, "created"),
            ui_state=json.loads(read_str(entry, "ui_state", "{}") or "{}"),
            source=_read_source(entry["source"]),
            calibration=_read_calibration(entry["calibration"]) if "calibration" in entry else None,
            phases=read_phases(entry["phases"]),
            oscilloscope=_read_oscilloscope(entry["oscilloscope"]) if "oscilloscope" in entry else None,
            spectra=spectra,
        )


def _read_source(group: h5py.Group) -> Source:
    return Source(**{name: read_str(group, name) for name in ("raw_file", "raw_dataset", "master_file", "scan")})


def _read_calibration(group: h5py.Group) -> Calibration:
    flip = int(read_scalar(group, "flip_axis", -1))
    points_per_degree = int(read_scalar(group, "points_per_degree", -1))
    return Calibration(
        energy_ev=float(read_scalar(group, "energy_ev")),
        poni=read_str(group, "poni"),
        mask=group["mask"][()] if "mask" in group else None,
        two_theta_range=tuple(float(v) for v in group["two_theta_range"][()]),
        points_per_degree=None if points_per_degree < 0 else points_per_degree,
        flip_axis=None if flip < 0 else flip,
        sigma_distance_m=float(read_scalar(group, "sigma_distance_m", 0.0)),
        poni_file=read_str(group, "poni_file"),
        mask_file=read_str(group, "mask_file"),
    )


def read_phases(root: h5py.Group) -> Dict[str, PhaseCard]:
    phases: Dict[str, PhaseCard] = {}
    for key in [str(k) for k in root.attrs.get("order", list(root))]:
        group = root[key]
        attrs = group.attrs
        refl = read_table(group["reflections"])
        ranges = {
            attr: (None if np.isnan(attrs[attr]).any() else tuple(float(v) for v in attrs[attr]))
            for attr in ("t_range", "p_range")
        }
        ints = {attr: (None if int(attrs[attr]) < 0 else int(attrs[attr])) for attr in ("z_max", "fu")}
        phases[key] = PhaseCard(
            name=str(attrs["name"]),
            symmetry=str(attrs["symmetry"]),
            **{a: none_if_nan(attrs[a]) for a in ("a", "b", "c", "v0", "alpha_t")},
            **{a: float(attrs[a]) for a in ("alpha", "beta", "gamma", "k0", "k0p", "v_min_ratio")},
            **ranges,
            **ints,
            reflections=[
                Reflection(float(d), float(i), int(h), int(k), int(l))
                for d, i, h, k, l in zip(refl["d"], refl["intensity"], refl["h"], refl["k"], refl["l"])
            ],
            source_text=read_str(group, "jcpds"),
            source_file=read_str(group, "source_file"),
        )
    return phases


def _read_oscilloscope(group: h5py.Group) -> Oscilloscope:
    names = [str(n) for n in group.attrs.get("channels", [])]
    return Oscilloscope(
        time=group["time"][()],
        channels={name: group[name][()] for name in names},
        trigger_channel=str(group.attrs.get("trigger_channel", "")),
    )


def _row(data: Optional[np.ndarray], i: int, n: int) -> Optional[np.ndarray]:
    if data is None:
        return None
    row = data[i, :n]
    return None if np.isnan(row).all() else row


def _read_spectra(group: h5py.Group) -> List[Spectrum]:
    n_points = group["n_points"][()]
    two_theta = group["two_theta"][()]
    intensity = group["intensity"][()]
    baseline = group["baseline"][()] if "baseline" in group else None
    model = group["model"][()] if "model" in group else None
    times = group["time"][()]
    frames = group["frame"][()]
    labels = group["label"].asstr()[()]
    fit_ranges = group["fit_range"][()]
    proc = read_table(group["processing"])
    spectra = []
    for i, n in enumerate(n_points):
        axis = two_theta if two_theta.ndim == 1 else two_theta[i, :n]
        fit_range = fit_ranges[i]
        spectra.append(
            Spectrum(
                two_theta=np.array(axis, dtype=float),
                intensity=intensity[i, :n].copy(),
                time=float(times[i]),
                frame=int(frames[i]),
                label=str(labels[i]),
                processing=Processing(
                    baseline_degree=int(proc["baseline_degree"][i]),
                    filter_type=str(proc["filter_type"][i]),
                    filter_params=(float(proc["filter_p1"][i]), float(proc["filter_p2"][i])),
                    baseline_window=_window(proc, i),
                ),
                baseline=_row(baseline, i, n),
                model=_row(model, i, n),
                fit_range=None if np.isnan(fit_range).any() else (float(fit_range[0]), float(fit_range[1])),
            )
        )
    return spectra


def _read_fits(group: h5py.Group, spectra: List[Spectrum]) -> None:
    phases = read_table(group["phases"])
    fits: List[PhaseFit] = []
    for row in range(len(phases["spectrum"])):
        fit = PhaseFit(
            phase=str(phases["phase"][row]),
            state=str(phases["state"][row]),
            **{field: float(phases[field][row]) for field in PHASE_FIT_FIELDS},
        )
        fits.append(fit)
        spectra[int(phases["spectrum"][row])].phases.append(fit)

    peaks = read_table(group["peaks"])
    shape_cols = sorted((c for c in peaks if c.startswith("shape_")), key=lambda c: int(c.split("_")[1]))
    for row in range(len(peaks["fit"])):
        shape = tuple(float(peaks[c][row]) for c in shape_cols if not np.isnan(peaks[c][row]))
        hkl = (int(peaks["h"][row]), int(peaks["k"][row]), int(peaks["l"][row])) if peaks["has_hkl"][row] else None
        fits[int(peaks["fit"][row])].peaks.append(
            Peak(
                name=str(peaks["name"][row]),
                model=str(peaks["model"][row]),
                **{field: float(peaks[field][row]) for field in PEAK_FIELDS},
                shape=shape,
                hkl=hkl,
                enabled=bool(peaks["enabled"][row]),
            )
        )
