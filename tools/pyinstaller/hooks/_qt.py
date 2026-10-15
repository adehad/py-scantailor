"""What the PySide6 hooks keep: the Qt modules, plugins and translations in use.

ScanTailor draws with QtWidgets on the raster engine, reads and writes page
images with Pillow and OpenCV rather than Qt, and loads its forms with QtUiTools.
Each hook-PySide6.*.py beside this file runs the upstream hook and keeps only the
entries these rules allow.
"""

from collections.abc import Callable
from pathlib import PurePath
from typing import Any

from _trim import config, keep, upstream_hook

QT_MODULES = frozenset({"QtCore", "QtGui", "QtSvg", "QtUiTools", "QtWidgets"})

# Plugin folders dropped whole, each with why the app can do without it.
DROPPED_PLUGIN_TYPES = frozenset(
    {
        "generic",  # TUIO touch input over the network
        "networkinformation",  # QtNetwork, which the app does not use
        "tls",  # QtNetwork
        "egldeviceintegrations",  # only for the eglfs platform, dropped below
    }
)
# Plugins dropped from folders that otherwise stay.
DROPPED_PLUGINS = frozenset(
    {
        "qtvirtualkeyboardplugin",  # on-screen keyboard; pulls in Qt Quick and QML
        "qdirect2d",  # experimental Windows platform
        "qminimal",
        "qminimalegl",
        "qlinuxfb",
        "qeglfs",
        "qvnc",
        "qvkkhrdisplay",
    }
)
# Image formats Qt itself reads: the SVG icons and Windows .ico window icons.
# Pages go through Pillow and OpenCV, never through a Qt image plugin.
KEPT_IMAGE_FORMATS = frozenset({"qsvg", "qico"})
# Software OpenGL fallback for Windows machines without a GPU driver; the app
# never creates an OpenGL context.
DROPPED_LIBRARIES = frozenset({"opengl32sw"})


def _plugin_name(source: PurePath) -> str:
    return source.name.split(".")[0].removeprefix("lib")


def _wanted_binary(source: PurePath, dest: PurePath) -> bool:
    if _plugin_name(source) in DROPPED_LIBRARIES:
        return False
    if "plugins" not in dest.parts:
        return True
    plugin_type = dest.name
    name = _plugin_name(source)
    if plugin_type in DROPPED_PLUGIN_TYPES or name in DROPPED_PLUGINS:
        return False
    return plugin_type != "imageformats" or name in KEPT_IMAGE_FORMATS


def _translation_filter(languages: list[str]) -> Callable[[PurePath, PurePath], bool]:
    """Keep the qtbase catalogues of the shipped languages and other data."""

    def wanted(source: PurePath, _dest: PurePath) -> bool:
        if source.suffix != ".qm":
            return True
        prefix, _, locale = source.stem.partition("_")
        return prefix == "qtbase" and any(
            locale == language or locale.startswith(f"{language}_")
            for language in languages
        )

    return wanted


def _wanted_import(name: str) -> bool:
    package, _, module = name.partition(".")
    return package != "PySide6" or not module.startswith("Qt") or module in QT_MODULES


def trimmed_hook(module: str) -> Callable[[Any], None]:
    """Build a hook that runs the upstream PySide6 hook and trims its result.

    Args:
        module (str): The hooked module, such as ``PySide6.QtGui``.

    Returns:
        Callable[[Any], None]: The ``hook(hook_api)`` function for the hook file.
    """

    def hook(hook_api: Any) -> None:
        upstream = upstream_hook(module)
        languages = config(hook_api, "qt_languages")
        hook_api.add_imports(*filter(_wanted_import, upstream.get("hiddenimports", [])))
        hook_api.add_binaries(keep(upstream.get("binaries", []), _wanted_binary))
        hook_api.add_datas(
            keep(upstream.get("datas", []), _translation_filter(languages))
        )

    return hook
