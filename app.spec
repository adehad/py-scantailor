# -*- mode: python -*-
"""PyInstaller spec: the ScanTailor window and scantailor-cli in one folder.

On macOS the folder is also wrapped in ScanTailor.app.

Run ``pdm run exe``, which compiles the translations first.
"""

import sys
import typing
from importlib import metadata
from pathlib import Path

from packaging.version import Version
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
RESOURCES = PACKAGE / "resources"

BUNDLE_IDENTIFIER = "io.github.adehad.py-scantailor"
PROJECT_UTI = f"{BUNDLE_IDENTIFIER}.project"
# Launch Services offers the app for a file only when the bundle declares the
# file's type, here a UTI exported for the .ScanTailor extension. The Finder
# then hands the file over as an open event, which main.FileOpenHandler takes.
MACOS_INFO_PLIST = {
    "LSBackgroundOnly": False,
    "NSHighResolutionCapable": True,
    "CFBundleDocumentTypes": [
        {
            "CFBundleTypeName": "ScanTailor Project",
            "CFBundleTypeRole": "Editor",
            "LSHandlerRank": "Owner",
            "LSItemContentTypes": [PROJECT_UTI],
        }
    ],
    "UTExportedTypeDeclarations": [
        {
            "UTTypeIdentifier": PROJECT_UTI,
            "UTTypeDescription": "ScanTailor Project",
            "UTTypeConformsTo": ["public.data"],
            "UTTypeTagSpecification": {
                "public.filename-extension": ["ScanTailor"],
                "public.mime-type": ["application/x-scantailor-project"],
            },
        }
    ],
}


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
hiddenimports = ["PySide6.QtSvg"]

# The dev environment also holds the docs and test tools, which PyInstaller's
# hooks would otherwise sweep in.
excludes = ["_pytest", "alabaster", "babel", "docutils", "pytest", "pytz", "sphinx"]

exe_icon = None
if sys.platform == "win32":
    exe_icon = str(RESOURCES / "win32" / "scantailor.ico")


def _analyse(script: str) -> "Analysis":
    return Analysis(
        [str(ENTRY_POINTS / script)],
        pathex=[str(SRC)],
        datas=datas,
        hiddenimports=hiddenimports,
        excludes=excludes,
    )


def _without_video_backend(binaries: list) -> list:
    """Drop OpenCV's ffmpeg plugin (30 MB on Windows).

    OpenCV loads it only when a VideoCapture or VideoWriter opens a file, and
    nothing in the application reads or writes video.
    """
    return [entry for entry in binaries if "opencv_videoio_ffmpeg" not in entry[0]]


gui = _analyse("scantailor_gui.py")
cli = _analyse("scantailor_cli.py")

gui_exe = EXE(
    PYZ(gui.pure),
    gui.scripts,
    exclude_binaries=True,
    name="ScanTailor",
    upx=False,
    console=False,
    icon=exe_icon,
)
cli_exe = EXE(
    PYZ(cli.pure),
    cli.scripts,
    exclude_binaries=True,
    name="scantailor-cli",
    upx=False,
    console=True,
    icon=exe_icon,
)

# UPX stays off everywhere: it corrupts some Qt and Control Flow Guard DLLs.
coll = COLLECT(
    gui_exe,
    _without_video_backend(gui.binaries),
    gui.datas,
    cli_exe,
    _without_video_backend(cli.binaries),
    cli.datas,
    upx=False,
    name="ScanTailor",
)

# The bundle copies the whole folder into Contents/MacOS, so the CLI ships at
# ScanTailor.app/Contents/MacOS/scantailor-cli beside the libraries it shares.
# There is no .icns in the resources; PyInstaller converts the PNG with Pillow.
# BUNDLE takes its console flag from the COLLECT, which is the console CLI's, and
# that would mark the window app LSBackgroundOnly: no Dock icon and no menu bar.
if sys.platform == "darwin":
    BUNDLE(
        coll,
        name="ScanTailor.app",
        icon=str(RESOURCES / "ScanTailor.png"),
        bundle_identifier=BUNDLE_IDENTIFIER,
        version=Version(metadata.version("py-scantailor")).base_version,
        info_plist=MACOS_INFO_PLIST,
    )
