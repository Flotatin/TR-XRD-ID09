"""Long computations in a worker thread; results come back in the GUI thread.

Same interface as presenters.viewer.ImmediateRunner:
run(task, on_done, on_error), where task(progress) returns the result and
progress(done, total) -> keep going? (the `progress` of build_run, fit_spectra,
auto_analyse). The views and the Session are only touched from on_done /
on_error, which Qt runs in the GUI thread.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

logger = logging.getLogger(__name__)


class _Relay(QObject):
    """Lives in the GUI thread: signals emitted by the worker are queued to it."""

    done = Signal(object)
    failed = Signal(str)
    progressed = Signal(int, int)

    def __init__(self, on_done, on_error, on_progress, finished) -> None:
        super().__init__()
        self._on_done, self._on_error, self._on_progress, self._finished = on_done, on_error, on_progress, finished
        self.cancelled = False
        self.done.connect(self._done)
        self.failed.connect(self._failed)
        self.progressed.connect(self._progressed)

    @Slot(object)
    def _done(self, result) -> None:
        self._finished(self)
        self._on_done(result)

    @Slot(str)
    def _failed(self, message: str) -> None:
        self._finished(self)
        self._on_error(message)

    @Slot(int, int)
    def _progressed(self, done: int, total: int) -> None:
        if self._on_progress is not None:
            self._on_progress(done, total)


class _Job(QRunnable):
    def __init__(self, task, relay: _Relay) -> None:
        super().__init__()
        self.task, self.relay = task, relay

    def _emit(self, signal: str, *values) -> bool:
        """False when nobody is left to hear it: the window was closed while the task ran."""
        try:
            getattr(self.relay, signal).emit(*values)
        except RuntimeError:
            return False
        return True

    def run(self) -> None:
        def progress(done: int, total: int) -> bool:
            return self._emit("progressed", int(done), int(total)) and not self.relay.cancelled

        try:
            result = self.task(progress)
        except Exception as exc:
            logger.exception("background task failed")
            self._emit("failed", f"{type(exc).__name__}: {exc}")
        else:
            self._emit("done", result)


class ThreadRunner:
    def __init__(self, pool: Optional[QThreadPool] = None) -> None:
        self.pool = pool or QThreadPool.globalInstance()
        self._running = set()  # keeps the relays alive until they report

    def run(self, task, on_done: Callable[[object], None], on_error: Callable[[str], None],
            on_progress: Optional[Callable[[int, int], None]] = None) -> _Relay:
        relay = _Relay(on_done, on_error, on_progress, self._running.discard)
        self._running.add(relay)
        self.pool.start(_Job(task, relay))
        return relay  # relay.cancelled = True asks the task to stop at its next progress()

    @property
    def busy(self) -> bool:
        return bool(self._running)
