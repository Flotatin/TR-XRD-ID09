"""Entry point for launching the CED application from the repo root."""
 
from pathlib import Path
import sys


SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cedapp.main import main

if __name__ == "__main__":
    raise SystemExit(main())
