from __future__ import annotations

from typing import Optional

from PyQt5.QtWidgets import QApplication


def update_progress_dialog(
    progress_dialog,
    label_text: Optional[str] = None,
    step: Optional[int] = None,
) -> None:
    if progress_dialog is None:
        return
    if progress_dialog.wasCanceled():
        return
    if label_text is not None:
        progress_dialog.setLabelText(label_text)
    if step is not None:
        progress_dialog.setValue(step)
    QApplication.processEvents()
