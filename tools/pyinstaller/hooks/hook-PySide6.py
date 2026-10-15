"""Trim PySide6 to what ScanTailor uses; the rules are in _qt.py."""

from _qt import trimmed_hook

hook = trimmed_hook("PySide6")

# PySide6/__init__ imports QtNetwork only to preload OpenSSL from an "openssl"
# folder that the wheels do not ship; the app has no network code.
excludedimports = ["PySide6.QtNetwork"]
