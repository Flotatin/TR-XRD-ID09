"""PySide6 + pyqtgraph interface of TR-XRD v2 (MVP over services.Session).

presenters/: no Qt, testable with fake views; views/: passive widgets;
builder.py: builds the window and wires actions to the presenters; app.py: entry point, errors to a log file.
Start it with `trxrd-gui [run file]` or `python -m trxrd.gui`.
"""

import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")  # pyqtgraph must not pick another Qt binding
