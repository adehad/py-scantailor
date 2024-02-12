"""Project Metadata."""
from __future__ import annotations

try:
    import importlib.metadata

    __version__ = importlib.metadata.version("scantailor")
except Exception:
    __version__ = "unknown"
