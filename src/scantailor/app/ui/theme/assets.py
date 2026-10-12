"""Glyph files referenced from the stylesheet, written once per process.

A stylesheet ``url()`` cannot recolour an SVG or read it from a palette, so each
look gets its own copies painted in the token colours.
"""

import atexit
import functools
import pathlib
import shutil
import tempfile

from PySide6 import QtGui

from scantailor.app.ui.icons import tinted_svg

from .tokens import Tokens

# Lucide has no floating-dock glyph; this one keeps its 24 px grid and 2 px stroke.
_FLOAT_GLYPH = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" '
    'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<rect x="3" y="9" width="12" height="12" rx="2"/><path d="M9 9V5a2 2 0 0 1 2-2h8'
    'a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-4"/></svg>'
)
_DIM_GLYPHS = {
    "chevron_down": "chevron-down",
    "chevron_up": "chevron-up",
    "chevron_right": "chevron-right",
    "close": "x",
}


@functools.cache
def _asset_root() -> pathlib.Path:
    root = pathlib.Path(tempfile.mkdtemp(prefix="scantailor-theme-"))
    atexit.register(lambda: shutil.rmtree(root, ignore_errors=True))
    return root


def write_assets(t: Tokens, name: str) -> dict[str, str]:
    """Write the stylesheet glyphs for a look.

    Args:
        t (Tokens): Colours for the look.
        name (str): Look name, used as the sub-directory.

    Returns:
        dict[str, str]: Glyph name to a forward-slash path usable in QSS url().
    """
    folder = _asset_root() / name
    folder.mkdir(parents=True, exist_ok=True)
    dim = QtGui.QColor(t.text_dim)
    files = {key: tinted_svg(glyph, dim) for key, glyph in _DIM_GLYPHS.items()}
    files["float"] = _FLOAT_GLYPH.replace("currentColor", dim.name()).encode()
    files["check"] = tinted_svg("check", QtGui.QColor(t.accent_text))
    urls: dict[str, str] = {}
    for key, data in files.items():
        target = folder / f"{key}.svg"
        target.write_bytes(data)
        urls[key] = target.as_posix()
    return urls
