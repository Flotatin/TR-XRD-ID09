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
from typing import Dict, Iterator, Optional, Sequence, Tuple, Union

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
    master_file: PathLike, scan: str, channels: Dict[str, str], time_counter: str
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Read `measurement` counters of one scan.

    channels maps the name used in the run (e.g. "trigger") to the BLISS
    counter (e.g. "ch5"); time_counter is e.g. "ch5_time".
    """
    with h5py.File(master_file, "r") as f:
        measurement = f[scan]["measurement"]
        time = np.asarray(measurement[time_counter][()], dtype=float).ravel()
        data = {name: np.asarray(measurement[counter][()], dtype=float).ravel() for name, counter in channels.items()}
    for name, values in data.items():
        if values.size != time.size:
            raise ValueError(f"Channel {name!r} has {values.size} points, time has {time.size}")
    return time, data
