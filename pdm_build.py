"""pdm-backend build hook: compile the translations into the wheel.

The .qm catalogues are gitignored build output, so a wheel built from a clean
checkout or from the sdist would otherwise hold only the .ts sources. The sdist
keeps just the sources, and an editable install leaves the catalogues to the
application, which compiles them into its cache on first use.

PySide6 is not a build requirement, as it would add about 100 MB to every
isolated build. When the build environment lacks it the wheel ships without
catalogues and the application compiles them on first start instead.
"""

import importlib.util
import pathlib
import sys
from types import ModuleType
from typing import Any

_PACKAGE = pathlib.Path("scantailor") / "translations"


def _catalogues(root: pathlib.Path) -> ModuleType:
    """Load the compiler by path; importing the package would need its dependencies."""
    path = root / "src" / _PACKAGE / "catalogues.py"
    spec = importlib.util.spec_from_file_location("_scantailor_catalogues", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pdm_build_update_files(context: Any, files: dict[str, pathlib.Path]) -> None:
    """Add a compiled .qm for every .ts source to a wheel, when lrelease is at hand.

    Args:
        context (Any): pdm-backend build context; its ``target`` is "sdist",
            "wheel" or "editable".
        files (dict[str, pathlib.Path]): Archive paths mapped to the files that
            fill them; the catalogues are added to it.
    """
    if context.target != "wheel":
        return
    root = pathlib.Path(context.root)
    catalogues = _catalogues(root)
    try:
        lrelease = catalogues.find_lrelease()
    except FileNotFoundError:
        sys.stderr.write(
            "PySide6 is not in the build environment; the wheel ships without "
            "compiled translations, which the application compiles on first start\n"
        )
        return
    target_dir = pathlib.Path(context.ensure_build_dir()) / _PACKAGE
    written = catalogues.compile_all(root / "src" / _PACKAGE, target_dir, lrelease)
    for catalogue in written:
        files[(_PACKAGE / catalogue.name).as_posix()] = catalogue
