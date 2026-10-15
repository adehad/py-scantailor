"""Project Metadata."""

APP_NAME = "pyScanTailor"

try:
    import importlib.metadata

    __version__ = importlib.metadata.version("py-scantailor")
except Exception:
    __version__ = "unknown"
