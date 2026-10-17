"""Compile the .ts translation sources into the .qm catalogues Qt loads.

Only the standard library is imported here: the wheel build hook (pdm_build.py)
loads this file by path, in a build environment that has PySide6 but none of
the application's other dependencies.
"""

import hashlib
import importlib.util
import os
import pathlib
import shutil
import subprocess
import tempfile

SOURCE_PATTERN = "scantailor_*.ts"
_LRELEASE_NAME = "lrelease"


def find_lrelease() -> pathlib.Path:
    """Find the lrelease executable that PySide6 ships inside its package.

    Returns:
        pathlib.Path: Path to lrelease.

    Raises:
        FileNotFoundError: If PySide6 is not installed or holds no lrelease.
    """
    spec = importlib.util.find_spec("PySide6")
    folders = [] if spec is None else list(spec.submodule_search_locations or [])
    for folder in folders:
        found = shutil.which(_LRELEASE_NAME, path=folder)
        if found is not None:
            return pathlib.Path(found)
    raise FileNotFoundError(f"{_LRELEASE_NAME} not found in the PySide6 package")


def translation_sources(source_dir: pathlib.Path) -> list[pathlib.Path]:
    """List the .ts sources of a folder in name order.

    Args:
        source_dir (pathlib.Path): Folder holding the scantailor_*.ts files.

    Returns:
        list[pathlib.Path]: The sources; empty when the folder has none.
    """
    return sorted(source_dir.glob(SOURCE_PATTERN))


def sources_digest(source_dir: pathlib.Path) -> str:
    """Fingerprint the .ts sources, so compiled catalogues can be matched to them.

    Args:
        source_dir (pathlib.Path): Folder holding the scantailor_*.ts files.

    Returns:
        str: 16 hexadecimal digits that change when any source or its name does.
    """
    digest = hashlib.sha256()
    for source in translation_sources(source_dir):
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()[:16]


def compile_translation(
    source: pathlib.Path, target_dir: pathlib.Path, lrelease: pathlib.Path
) -> pathlib.Path:
    """Compile one .ts file into a .qm file of the same stem.

    The catalogue is written under a temporary name and then renamed, so a
    process loading it never sees half a file.

    Args:
        source (pathlib.Path): The .ts file to compile.
        target_dir (pathlib.Path): Folder to write the .qm file into.
        lrelease (pathlib.Path): Path to the lrelease executable.

    Returns:
        pathlib.Path: The written .qm file.

    Raises:
        subprocess.CalledProcessError: If lrelease rejects the source file.
    """
    target = target_dir / f"{source.stem}.qm"
    descriptor, partial = tempfile.mkstemp(
        prefix=f".{source.stem}.", suffix=".qm", dir=target_dir
    )
    os.close(descriptor)
    try:
        subprocess.run(
            [str(lrelease), "-silent", str(source), "-qm", partial],
            check=True,
            capture_output=True,
        )
        os.replace(partial, target)
    finally:
        pathlib.Path(partial).unlink(missing_ok=True)
    return target


def compile_all(
    source_dir: pathlib.Path, target_dir: pathlib.Path, lrelease: pathlib.Path
) -> list[pathlib.Path]:
    """Compile every .ts file in a folder.

    Args:
        source_dir (pathlib.Path): Folder holding the scantailor_*.ts files.
        target_dir (pathlib.Path): Folder to write the .qm files into; it is
            created when missing.
        lrelease (pathlib.Path): Path to the lrelease executable.

    Returns:
        list[pathlib.Path]: The .qm files written, sorted by name.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    return [
        compile_translation(source, target_dir, lrelease)
        for source in translation_sources(source_dir)
    ]
