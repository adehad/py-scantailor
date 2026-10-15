"""Shared helpers for the project hooks that trim what an upstream hook collects.

PyInstaller runs one hook per module, the highest-priority one, so a project hook
named like an upstream one replaces it. These helpers run the replaced hook and
filter its result, keeping the upstream logic instead of copying it.
"""

from collections.abc import Callable, Iterable
from pathlib import Path, PurePath
from runpy import run_path
from typing import Any

import _pyinstaller_hooks_contrib
import PyInstaller
from PyInstaller.utils.hooks import get_hook_config

_UPSTREAM_DIRS = (
    Path(_pyinstaller_hooks_contrib.__file__).parent / "stdhooks",
    Path(PyInstaller.__file__).parent / "hooks",
)
CONFIG_SECTION = "scantailor"


def upstream_hook(module: str) -> dict[str, Any]:
    """Run the hook this project replaces and return what it defines.

    Args:
        module (str): The hooked module, such as ``cv2``.

    Returns:
        dict[str, Any]: The upstream hook's globals (``binaries``, ``datas``,
        ``hiddenimports``...).

    Raises:
        FileNotFoundError: Neither PyInstaller nor its contributed hooks have one.
    """
    for folder in _UPSTREAM_DIRS:
        hook = folder / f"hook-{module}.py"
        if hook.is_file():
            return run_path(str(hook))
    raise FileNotFoundError(f"no upstream hook for {module}")


def keep(
    entries: Iterable[tuple[str, str]], wanted: Callable[[PurePath, PurePath], bool]
) -> list[tuple[str, str]]:
    """Filter a hook's (source file, destination folder) pairs.

    Args:
        entries (Iterable[tuple[str, str]]): The hook's ``binaries`` or ``datas``.
        wanted (Callable[[PurePath, PurePath], bool]): Called with the source
            file and the destination folder; True keeps the entry.

    Returns:
        list[tuple[str, str]]: The entries kept.
    """
    return [
        (source, dest)
        for source, dest in entries
        if wanted(PurePath(source), PurePath(dest))
    ]


def config(hook_api: Any, key: str) -> Any:
    """Read a value app.spec passed through ``hooksconfig["scantailor"]``."""
    value = get_hook_config(hook_api, CONFIG_SECTION, key)
    if value is None:
        raise KeyError(f'app.spec sets no hooksconfig["{CONFIG_SECTION}"]["{key}"]')
    return value
