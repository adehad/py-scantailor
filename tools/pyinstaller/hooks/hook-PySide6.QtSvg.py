"""Trim PySide6.QtSvg to what ScanTailor uses; the rules are in _qt.py."""

from _qt import trimmed_hook

hook = trimmed_hook("PySide6.QtSvg")
