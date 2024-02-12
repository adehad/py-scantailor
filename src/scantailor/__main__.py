"""Main Entrypoint for GUI Application.

No relative imports so PyInstaller doesn't get confused.
"""
from __future__ import annotations

from scantailor.main import main

if __name__ == "__main__":
    raise SystemExit(main())
