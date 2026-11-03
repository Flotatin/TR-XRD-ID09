"""Use case: browse the raw data of a campaign and find the raw scan of a run.

ID09 layout (read-only)::

    RAW_DATA/<sample>/<sample>_0001/<sample>_0001.h5            BLISS master: one entry per scan,
                                                                 oscilloscope ch5 (trigger), ch6, ch2
    RAW_DATA/<sample>/<sample>_0001/scan0009/scan_jf1m_0000.h5   Jungfrau images of scan 9
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Union

from trxrd.domain import Run
from trxrd.io import esrf

PathLike = Union[str, Path]
IMAGE_PATTERN = "scan_jf1m_*.h5"


@dataclass
class Dataset:
    sample: str
    name: str  # e.g. Water05_0001
    folder: Path

    @property
    def master(self) -> Optional[Path]:
        path = self.folder / f"{self.name}.h5"
        return path if path.is_file() else None


@dataclass
class RawScan:
    number: int
    folder: Path  # .../scan0009 (may not exist: scan without images)
    image_file: Optional[Path]
    title: str = ""
    start_time: str = ""
    frames: int = 0
    oscilloscope: bool = False  # trigger channel (ch5) recorded

    @property
    def label(self) -> str:
        return f"scan{self.number:04d}"


def find_datasets(raw_root: PathLike) -> List[Dataset]:
    """<sample>/<dataset> folders under RAW_DATA that hold scans or a master file."""
    datasets = []
    root = Path(raw_root)
    for sample in sorted(p for p in root.iterdir() if p.is_dir()):
        for folder in sorted(p for p in sample.iterdir() if p.is_dir()):
            if (folder / f"{folder.name}.h5").is_file() or any(folder.glob("scan[0-9]*")):
                datasets.append(Dataset(sample.name, folder.name, folder))
    return datasets


def _image_file(scan_folder: Path) -> Optional[Path]:
    files = sorted(scan_folder.glob(IMAGE_PATTERN)) if scan_folder.is_dir() else []
    return files[0] if files else None


def list_scans(dataset: Dataset, images_only: bool = False) -> List[RawScan]:
    """Scans of the master file merged with the scanNNNN folders (images)."""
    scans = {}
    if dataset.master is not None:
        for entry in esrf.read_bliss_scans(dataset.master):
            folder = dataset.folder / f"scan{entry.number:04d}"
            scans[entry.number] = RawScan(entry.number, folder, _image_file(folder), entry.title, entry.start_time,
                                          entry.frames, "ch5" in entry.counters)
    for folder in dataset.folder.glob("scan[0-9]*"):
        match = re.fullmatch(r"scan(\d+)", folder.name)
        if match and int(match.group(1)) not in scans:
            scans[int(match.group(1))] = RawScan(int(match.group(1)), folder, _image_file(folder))
    out = [scans[k] for k in sorted(scans)]
    return [s for s in out if s.image_file is not None] if images_only else out


def parse_run_name(name: str):
    """'Water05_scan0009' or 'Water05_0001_scan0009' -> ('Water05', '0001' or None, 9), else None."""
    match = re.fullmatch(r"(.+?)(?:_(\d{4}))?_scan(\d+)", name)
    if not match:
        return None
    return match.group(1), match.group(2), int(match.group(3))


def locate_raw_scan(run: Run, near: Iterable[PathLike] = ()) -> Optional[Path]:
    """Image file of the run: its recorded source if it exists, else found by its name in a RAW_DATA folder.

    `near`: files or folders of the campaign (e.g. the run file, in
    <campaign>/Fichier_CED_DRX/): RAW_DATA is looked for in them and in their parents.
    """
    if run.source.raw_file and Path(run.source.raw_file).is_file():
        return Path(run.source.raw_file)
    parsed = parse_run_name(run.name)
    if parsed is None:
        return None
    sample, dataset, number = parsed
    for start in near:
        start = Path(start)
        for folder in [start, *start.parents][:4]:
            raw = folder / "RAW_DATA" if folder.name != "RAW_DATA" else folder
            pattern = f"{sample}_{dataset}" if dataset else f"{sample}_*"
            for candidate in sorted((raw / sample).glob(f"{pattern}/scan{number:04d}")) if (raw / sample).is_dir() else []:
                image = _image_file(candidate)
                if image is not None:
                    return image
    return None


def default_phase_library() -> Optional[Path]:
    """trxrd/resources/phases of the source tree (editable install), if present."""
    folder = Path(__file__).resolve().parents[3] / "resources" / "phases"
    return folder if folder.is_dir() else None
