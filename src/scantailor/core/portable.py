"""Portable mode: keep settings, profiles and logs in a folder beside the program.

A folder named ``config`` next to the executable of a frozen build switches it
on, as in ScanTailor Advanced. Without that folder the program uses the usual
per-user locations. A build run from source has no executable of its own, so it
is never portable.
"""

import functools
import sys
import tempfile
from pathlib import Path

from loguru import logger
from PySide6 import QtCore

PORTABLE_CONFIG_DIR = "config"


def application_folder() -> Path | None:
    """Locate the folder holding the program's executable.

    Returns:
        pathlib.Path | None: The folder of the frozen executable, or None when running
            from source, where ``sys.executable`` is the Python interpreter.
    """
    if not getattr(sys, "frozen", False):
        return None
    return Path(sys.executable).resolve().parent


def find_portable_folder(program_folder: Path | None) -> Path | None:
    """Find the portable configuration folder beside a program.

    Args:
        program_folder (pathlib.Path | None): Folder of the executable; None means
            there is no executable to look beside.

    Returns:
        pathlib.Path | None: The ``config`` folder if it exists and this user can create
            files in it, else None. A folder that cannot be written is logged and
            ignored, so the program still saves its settings in the usual place.
    """
    if program_folder is None:
        return None
    folder = program_folder / PORTABLE_CONFIG_DIR
    if not folder.is_dir():
        return None
    try:
        with tempfile.TemporaryDirectory(dir=folder):
            pass
    except OSError as error:
        logger.warning("Not portable: cannot write to {}: {}", folder, error)
        return None
    return folder


@functools.cache
def portable_folder() -> Path | None:
    """Return the portable configuration folder of this process, if any.

    The answer is worked out once, so every part of the program agrees on where
    its files are even if the folder appears or disappears while it runs.

    Returns:
        pathlib.Path | None: The ``config`` folder beside the executable, or None when
            the program is not portable.
    """
    return find_portable_folder(application_folder())


def use_portable_qsettings(folder: Path | None = None) -> bool:
    """Make QSettings keep its values in an INI file in the portable folder.

    Must run before the first QSettings object is made. The file is
    ``<folder>/<organisation>/<application>.ini``.

    Args:
        folder (pathlib.Path | None): The portable folder; None looks it up.

    Returns:
        bool: True if the program is portable and QSettings now uses the folder.
    """
    folder = folder or portable_folder()
    if folder is None:
        return False
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
    QtCore.QSettings.setPath(
        QtCore.QSettings.Format.IniFormat,
        QtCore.QSettings.Scope.UserScope,
        str(folder),
    )
    return True
