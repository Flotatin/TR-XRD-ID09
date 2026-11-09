"""Readers for ESRF raw data: LIMA image stacks and BLISS master files.

Raw files are only ever opened read-only.

Typical ID09 layout::

    RAW_DATA/<sample>/<sample>_<dataset>/<sample>_<dataset>.h5      <- BLISS master (ch5, ch6, ...)
    RAW_DATA/<sample>/<sample>_<dataset>/scan0027/scan_jf1m_0000.h5 <- LIMA images of scan 27
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

try:  # compression filters of the detector files (bitshuffle/LZ4...)
    import hdf5plugin  # noqa: F401
except ImportError:  # pragma: no cover - only needed for compressed detector data
    hdf5plugin = None

PathLike = Union[str, Path]


def find_default_signal(f: h5py.File) -> str:
    """Path of the NeXus default plottable dataset, else of the first 3D dataset."""
    try:
        group = f
        while "default" in group.attrs:
            group = group[str(group.attrs["default"])]
        if "signal" in group.attrs:
            return group[str(group.attrs["signal"])].name
    except KeyError:
        pass
    found = []
    f.visititems(lambda name, obj: found.append(name) if isinstance(obj, h5py.Dataset) and obj.ndim == 3 else None)
    if not found:
        raise KeyError(f"No image stack found in {f.filename}")
    return "/" + found[0]


@dataclass
class ImageStack:
    """Lazy access to the frames of a LIMA HDF5 file (or any 3D HDF5 dataset)."""

    path: Path
    dataset: str

    @classmethod
    def open(cls, path: PathLike, dataset: Optional[str] = None) -> "ImageStack":
        path = Path(path)
        if dataset is None:
            with h5py.File(path, "r") as f:
                dataset = find_default_signal(f)
        return cls(path, dataset)

    def __len__(self) -> int:
        with h5py.File(self.path, "r") as f:
            return int(f[self.dataset].shape[0])

    def frame(self, index: int) -> np.ndarray:
        with h5py.File(self.path, "r") as f:
            return f[self.dataset][index]

    def frames(self, indices: Optional[Sequence[int]] = None) -> Iterator[Tuple[int, np.ndarray]]:
        with h5py.File(self.path, "r") as f:
            data = f[self.dataset]
            for i in range(data.shape[0]) if indices is None else indices:
                yield i, data[i]


def read_exposure(path: PathLike, dataset: Optional[str] = None) -> Optional[float]:
    """Exposure time (s) of each frame, as set on the detector; None if the file does not say.

    LIMA writes it in the detector group: <detector>/acquisition/exposure_time,
    and as text in <detector>/header/acq_expo_time. The images are often
    reached through <detector>/plot/data: the groups above them are searched.
    """
    with h5py.File(path, "r") as f:
        group = f[dataset or find_default_signal(f)].parent
        while True:
            for key in ("acquisition/exposure_time", "header/acq_expo_time"):
                if key not in group:
                    continue
                value = group[key][()]
                try:
                    value = float(value.decode() if isinstance(value, bytes) else value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value) and value > 0:
                    return value
            if group.name == "/":
                return None
            group = group.parent


# ----------------------------------------------------------------------------- BLISS master


def locate_bliss_master(scan_file: PathLike) -> Tuple[Path, str]:
    """From .../<dataset>/scan0027/scan_jf1m_0000.h5 return (.../<dataset>/<dataset>.h5, "27.1")."""
    scan_file = Path(scan_file)
    scan_dir = scan_file.parent
    match = re.fullmatch(r"scan(\d+)", scan_dir.name)
    if not match:
        raise ValueError(f"{scan_file} is not inside a 'scanNNNN' folder")
    dataset_dir = scan_dir.parent
    return dataset_dir / f"{dataset_dir.name}.h5", f"{int(match.group(1))}.1"


def read_bliss_channels(
    master_file: PathLike, scan: str, channels: Dict[str, str], time_counter: str,
    optional: Sequence[str] = (),
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Read `measurement` counters of one scan.

    channels maps the name used in the run (e.g. "trigger") to the BLISS
    counter (e.g. "ch5"); time_counter is e.g. "ch5_time". The channels named
    in `optional` are skipped when the scan does not have them. All the
    oscilloscope channels share one time base (ch5_time and ch6_time differ
    by 19 ns for a 160 ns step).
    """
    with h5py.File(master_file, "r") as f:
        measurement = f[scan]["measurement"]
        time = np.asarray(measurement[time_counter][()], dtype=float).ravel()
        data = {name: np.asarray(measurement[counter][()], dtype=float).ravel() for name, counter in channels.items()
                if counter in measurement or name not in optional}
    for name, values in data.items():
        if values.size != time.size:
            raise ValueError(f"Channel {name!r} has {values.size} points, time has {time.size}")
    return time, data


@dataclass
class BlissScan:
    """One scan of a BLISS master file, as listed to the user."""

    label: str  # "27.1"
    number: int
    title: str  # BLISS command, e.g. "timescan 0.1 1"
    start_time: str
    counters: Tuple[str, ...]  # names in measurement/
    frames: int  # images of the detector (0: no image)


def _text(value) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def read_bliss_scans(master_file: PathLike, detector: str = "jf1m") -> List[BlissScan]:
    """Every scan of a master file (title, start time, counters, number of detector frames)."""
    scans = []
    with h5py.File(master_file, "r") as f:
        for label in f:
            match = re.fullmatch(r"(\d+)\.1", label)
            if not match:
                continue
            entry = f[label]
            measurement = entry.get("measurement")
            counters = tuple(measurement.keys()) if measurement is not None else ()
            frames = 0
            if detector in counters:
                try:  # external link to the LIMA file: may be missing
                    frames = int(measurement[detector].shape[0])
                except (KeyError, OSError):
                    frames = 0
            scans.append(BlissScan(
                label=label,
                number=int(match.group(1)),
                title=_text(entry["title"][()]) if "title" in entry else "",
                start_time=_text(entry["start_time"][()]) if "start_time" in entry else "",
                counters=counters,
                frames=frames,
            ))
    return sorted(scans, key=lambda s: s.number)
