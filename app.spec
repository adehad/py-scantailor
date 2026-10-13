# -*- mode: python -*-
"""PyInstaller file for generating app binaries."""


import typing

if typing.TYPE_CHECKING:
    from PyInstaller.building.api import COLLECT, EXE, PYZ
    from PyInstaller.building.build_main import Analysis
    from PyInstaller.building.osx import BUNDLE

block_cipher = None

a = Analysis(
    ["src/scantailor/__main__.py"],
    pathex=["."],
    binaries=None,
    # fonts.py reads the Korean fallback from disk next to the package; Inter and
    # the icons are compiled into resources_rc.
    datas=[
        (
            "src/scantailor/resources/fonts/noto-sans-kr",
            "scantailor/resources/fonts/noto-sans-kr",
        ),
    ],
    # QtSvg gives PyInstaller's hook the qsvg image-format plugin that draws the
    # glyphs; resources_rc is the compiled Qt resource module.
    #
    # The numba kernels cannot ship precompiled: numba keys its cache by the
    # source path and, in a frozen build, by a hash of the executable, and writes
    # it to the user's cache directory (%LOCALAPPDATA%\numba, ~/.cache/numba).
    # numba_warmup compiles every kernel once on tiny input, so the first run
    # pays the compile in the background (one kernel took 2.5 s in a frozen
    # test build, 0.3 s once cached) and every later run loads the cache.
    hiddenimports=[
        "PySide6.QtSvg",
        "scantailor.app._ui.resources_rc",
        "scantailor.imageproc.numba_warmup",
    ],
    hookspath=None,
    runtime_hooks=None,
    excludes=None,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="ScanTailor",
    debug=False,
    strip=False,
    upx=True,
    console=False,
    # icon="src\\scantailor\\resources\\appicon.svg,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name="ScanTailor",
)

app = BUNDLE(
    coll,
    name="ScanTailor.app",
    # icon="src/scantailor/resources/appicon.svg",
    bundle_identifier=None,
)
