"""Entry point of `trxrd-gui`: the executable has no console, so errors must go somewhere.

Without this module an error at start, even an import error, leaves no trace.
It stays light (the GUI, builder.py, is imported inside `main`). Each start, the
warnings and the errors go to a log file, and errors are shown in a message box.

Keep the name `trxrd.gui.app:main`: it is written in trxrd-gui.exe, and Smart App
Control (Windows) refuses an executable with a new content ("Device Guard").
"""

from __future__ import annotations

import logging
import os
import platform
import sys
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Sequence, Set

logger = logging.getLogger("trxrd")
MAX_LOG_BYTES = 1_000_000
_shown: Set[str] = set()  # an error raised at every repaint must not open a message box each time
_handler: Optional[logging.Handler] = None


def log_path() -> Path:
    return Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "trxrd" / "trxrd-gui.log"


def start_log() -> None:
    """Warnings and errors of every module go to the log file; a full disk must not stop the program."""
    global _handler
    if _handler is not None:
        return
    try:
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        _handler = RotatingFileHandler(path, maxBytes=MAX_LOG_BYTES, backupCount=1, encoding="utf-8")
    except OSError:
        return
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(_handler)
    logger.setLevel(logging.INFO)


def stop_log() -> None:
    global _handler
    if _handler is not None:
        logging.getLogger().removeHandler(_handler)
        _handler.close()
        _handler = None


def show_error(text: str) -> None:
    """A message box, even when Qt could not be started."""
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox

        if QApplication.instance() is not None:
            QMessageBox.critical(None, "TR-XRD v2", text)
            return
    except Exception:  # Qt itself may be what fails
        logger.exception("no Qt message box")
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, text, "TR-XRD v2", 0x10)


def report(kind, value, trace) -> None:
    """Log an uncaught error and show it (sys.excepthook: also the errors raised in Qt slots)."""
    if issubclass(kind, KeyboardInterrupt):
        sys.__excepthook__(kind, value, trace)
        return
    logger.critical("uncaught error\n%s", "".join(traceback.format_exception(kind, value, trace)))
    message = f"{kind.__name__}: {value}"
    if message not in _shown:
        _shown.add(message)
        show_error(f"{message}\n\nDétails dans {log_path()}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    start_log()
    logger.info("start: %s, Python %s, %s", sys.argv, platform.python_version(), sys.executable)
    sys.excepthook = report
    try:
        from trxrd.gui.builder import run

        code = run(argv)
    except Exception:
        report(*sys.exc_info())
        return 1
    logger.info("end: exit code %s", code)
    return code
