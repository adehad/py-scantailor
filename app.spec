# -*- mode: python -*-
"""PyInstaller spec: the ScanTailor window and scantailor-cli in one folder.

Run ``pdm run exe``, which compiles the translations first.
"""

import sys
import typing
from pathlib import Path

from PyInstaller.utils.hooks import copy_metadata

if typing.TYPE_CHECKING:
    from PyInstaller.building.api import COLLECT, EXE, PYZ
    from PyInstaller.building.build_main import Analysis
    from PyInstaller.building.osx import BUNDLE

    SPECPATH: str

ROOT = Path(SPECPATH)
SRC = ROOT / "src"
PACKAGE = SRC / "scantailor"
ENTRY_POINTS = ROOT / "tools" / "pyinstaller"
ICON = PACKAGE / "resources" / "win32" / "scantailor.ico"


def _beside_package(files: list[Path]) -> list[tuple[str, str]]:
    """Place each file at its path relative to src/, where the code looks for it."""
    return [(str(file), str(file.parent.relative_to(SRC))) for file in files]


translations = sorted((PACKAGE / "translations").glob("scantailor_*.qm"))
if not translations:
    raise SystemExit(
        "No compiled translations; run tools/build_translations.py (pdm run exe "
        "does this) before PyInstaller."
    )

# The dialogs load their .ui files and the translation loader its .qm files from
# beside the modules; fonts.py reads the Korean fallback from disk. Inter and the
# icons are compiled into resources_rc. The package metadata carries __version__.
datas = [
    *_beside_package(sorted(PACKAGE.rglob("*.ui"))),
    *_beside_package(translations),
    *_beside_package(sorted((PACKAGE / "resources" / "fonts" / "noto-sans-kr").iterdir())),
    *copy_metadata("py-scantailor"),
]

# QtSvg gives PyInstaller's hook the qsvg image-format plugin that draws the glyphs.
#
# The numba kernels cannot ship precompiled: numba keys its cache by the source
# path and, in a frozen build, by a hash of the executable, and writes it to the
# user's cache directory (%LOCALAPPDATA%\numba, ~/.cache/numba). numba_warmup
# compiles every kernel once on tiny input, so the first run pays the compile in
# the background and every later run loads the cache.
hiddenimports = ["PySide6.QtSvg", "scantailor.imageproc.numba_warmup"]

# The dev environment also holds the docs and test tools; PyInstaller's hooks for
# numba and scikit-image would otherwise sweep them in (about 30 MB).
excludes = ["_pytest", "alabaster", "babel", "docutils", "pytest", "pytz", "sphinx"]

icon = str(ICON) if sys.platform == "win32" else None


def _analyse(script: str) -> "Analysis":
    return Analysis(
        [str(ENTRY_POINTS / script)],
        pathex=[str(SRC)],
        datas=datas,
        hiddenimports=hiddenimports,
        excludes=excludes,
    )


gui = _analyse("scantailor_gui.py")
cli = _analyse("scantailor_cli.py")

gui_exe = EXE(
    PYZ(gui.pure),
    gui.scripts,
    exclude_binaries=True,
    name="ScanTailor",
    upx=False,
    console=False,
    icon=icon,
)
cli_exe = EXE(
    PYZ(cli.pure),
    cli.scripts,
    exclude_binaries=True,
    name="scantailor-cli",
    upx=False,
    console=True,
    icon=icon,
)

# UPX stays off everywhere: it corrupts some Qt and Control Flow Guard DLLs.
coll = COLLECT(
    gui_exe,
    gui.binaries,
    gui.datas,
    cli_exe,
    cli.binaries,
    cli.datas,
    upx=False,
    name="ScanTailor",
)

if sys.platform == "darwin":
    BUNDLE(coll, name="ScanTailor.app", bundle_identifier=None)
