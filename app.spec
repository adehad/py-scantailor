# -*- mode: python -*-
"""PyInstaller file for generating app binaries."""
from __future__ import annotations

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
    datas=[],
    hiddenimports=[],
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
