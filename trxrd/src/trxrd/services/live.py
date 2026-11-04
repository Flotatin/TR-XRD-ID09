"""Use case: Live mode. During an experiment, follow the raw data and analyse each new scan as soon as it ends.

As in the v1, a new scan takes what the run on screen has: calibration, phases,
treatment. Each scan here holds many spectra, so where the v1 fitted the kept
gauges on one spectrum, the automatic analysis follows them through the scan,
with the settings of the run.

A scan is left alone while its file grows. Nothing is ever overwritten: the
raw data are only read, and an analysis already saved is opened as it is.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from trxrd.domain import Calibration, PhaseCard, Processing, Run
from trxrd.io.h5store import load_run, save_run
from trxrd.services import analysis
from trxrd.services.raw_data import IMAGE_PATTERN, locate_raw_scan
from trxrd.services.run_builder import Cancelled, build_run, default_run_name

PathLike = Union[str, Path]
Progress = Callable[[int, int], bool]
MAX_TRIES = 3  # a scan that cannot be read yet is tried again this many times


@dataclass
class LiveSettings:
    folder: Path  # what is watched: a dataset, a sample or RAW_DATA
    output: Path  # where the .trxrd.h5 files are written
    analyse: bool = True  # automatic analysis of each new scan; otherwise integrated only


@dataclass
class Template:
    """What a new scan takes from the run on screen (a copy: that run may change meanwhile)."""

    calibration: Calibration
    phases: Dict[str, PhaseCard]
    processing: Processing
    ui_state: Dict[str, Any]


def template_of(run: Run) -> Template:
    """The candidate phases are those chosen for the analysis, or else those the user worked with."""
    settings = analysis.saved_settings(run)
    used = sorted({fit.phase for spectrum in run.spectra for fit in spectrum.phases})
    if settings.phases is None and used:
        settings = dataclasses.replace(settings, phases=used)
    ui_state = copy.deepcopy(run.ui_state)
    ui_state[analysis.UI_KEY] = settings.to_dict()
    processing = run.spectra[0].processing if run.spectra else Processing()
    return Template(dataclasses.replace(run.calibration), dict(run.phases), dataclasses.replace(processing), ui_state)


def dataset_folder(run: Run, near: Iterable[PathLike] = ()) -> Optional[Path]:
    """Dataset of the raw scan of `run`: where the next scans will land."""
    image = locate_raw_scan(run, near)
    return image.parents[1] if image is not None else None


def _modified(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:  # gone since it was listed
        return -1.0


def newest_scan(folder: PathLike) -> Optional[Path]:
    """Newest image file under `folder`: a dataset, a sample or RAW_DATA."""
    folder = Path(folder)
    files = [file for depth in ("", "*/", "*/*/") for file in folder.glob(f"{depth}scan[0-9]*/{IMAGE_PATTERN}")]
    return max(files, key=_modified, default=None)


class Watch:
    """Says when a new scan is complete: newer than those seen, and unchanged since the last look."""

    def __init__(self, folder: PathLike) -> None:
        self.folder = Path(folder)
        self.seen = newest_scan(self.folder)  # what is there at the start is not new
        self._looked: Optional[Tuple[Path, int, int]] = None
        self._tries: Dict[Path, int] = {}

    def poll(self) -> Optional[Path]:
        scan = newest_scan(self.folder)
        if scan is None or scan == self.seen:
            self._looked = None
            return None
        stat = scan.stat()
        looked = (scan, stat.st_size, stat.st_mtime_ns)
        if looked != self._looked:
            self._looked = looked  # it may still be growing
            return None
        self.seen, self._looked = scan, None
        return scan

    def failed(self, scan: Path) -> bool:
        """`scan` could not be read, perhaps still being written: True if it will be tried again."""
        self._tries[scan] = self._tries.get(scan, 0) + 1
        if self._tries[scan] >= MAX_TRIES:
            return False
        self.seen = None
        return True


def output_path(scan: PathLike, settings: LiveSettings) -> Path:
    return Path(settings.output) / f"{default_run_name(scan)}.trxrd.h5"


def process(scan: PathLike, template: Template, settings: LiveSettings,
            progress: Optional[Progress] = None) -> Tuple[Run, Path]:
    """The run of `scan`: integrated, analysed if asked, saved. Call it from a worker thread.

    A scan analysed before is opened, not overwritten. Raises Cancelled, and
    saves nothing, when `progress` asks to stop.
    """
    path = output_path(scan, settings)
    if path.exists():
        return load_run(path), path
    stopped: List[int] = []
    stages = 2 if settings.analyse else 1

    def stage(number: int) -> Progress:
        def step(done: int, total: int) -> bool:
            keep_going = progress is None or progress(number * total + done, stages * total)
            if not keep_going:
                stopped.append(done)
            return keep_going
        return step

    run = build_run(scan, dataclasses.replace(template.calibration), processing=template.processing,
                    phases=dict(template.phases), progress=stage(0))
    run.ui_state = copy.deepcopy(template.ui_state)
    if settings.analyse and run.phases and not stopped:
        analysis.analyse(run, range(len(run)), analysis.saved_settings(run), stage(1)).apply_to(run)
    if stopped:
        raise Cancelled(f"stopped after {stopped[0]} spectra")
    path.parent.mkdir(parents=True, exist_ok=True)
    return run, save_run(run, path)
