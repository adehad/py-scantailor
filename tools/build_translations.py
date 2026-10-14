"""Compile the bundled .ts translation sources into the .qm files the app loads.

The .qm files are build output and stay gitignored; run this after editing a .ts file
or on a fresh clone before launching the application.
"""

import argparse
import pathlib
import shutil
import subprocess
import sys

from loguru import logger

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRANSLATIONS_DIR = REPO_ROOT / "src" / "scantailor" / "translations"
_LRELEASE_NAME = "pyside6-lrelease"


def find_lrelease() -> pathlib.Path:
    """Find the lrelease tool installed next to the running interpreter.

    Returns:
        pathlib.Path: Path to pyside6-lrelease in the same environment.

    Raises:
        FileNotFoundError: If the tool is not installed in that environment.
    """
    scripts_dir = pathlib.Path(sys.executable).parent
    found = shutil.which(_LRELEASE_NAME, path=str(scripts_dir))
    if found is None:
        raise FileNotFoundError(f"{_LRELEASE_NAME} not found beside {sys.executable}")
    return pathlib.Path(found)


def compile_translation(
    source: pathlib.Path, target_dir: pathlib.Path, lrelease: pathlib.Path
) -> pathlib.Path:
    """Compile one .ts file into a .qm file of the same stem.

    Args:
        source (pathlib.Path): The .ts file to compile.
        target_dir (pathlib.Path): Folder to write the .qm file into.
        lrelease (pathlib.Path): Path to the pyside6-lrelease executable.

    Returns:
        pathlib.Path: The written .qm file.

    Raises:
        subprocess.CalledProcessError: If lrelease rejects the source file.
    """
    target = target_dir / f"{source.stem}.qm"
    subprocess.run([str(lrelease), str(source), "-qm", str(target)], check=True)
    logger.info(f"Compiled {source.name} to {target.name}")
    return target


def compile_all(
    source_dir: pathlib.Path, target_dir: pathlib.Path, lrelease: pathlib.Path
) -> list[pathlib.Path]:
    """Compile every .ts file in a folder.

    Args:
        source_dir (pathlib.Path): Folder holding the scantailor_*.ts files.
        target_dir (pathlib.Path): Folder to write the .qm files into.
        lrelease (pathlib.Path): Path to the pyside6-lrelease executable.

    Returns:
        list[pathlib.Path]: The .qm files written, sorted by name.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    return [
        compile_translation(source, target_dir, lrelease)
        for source in sorted(source_dir.glob("scantailor_*.ts"))
    ]


def main(argv: list[str] | None = None) -> int:
    """Compile the translations from the command line.

    Args:
        argv (list[str] | None): Arguments without the program name; None uses
            sys.argv.

    Returns:
        int: Process exit code, 0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-dir", type=pathlib.Path, default=TRANSLATIONS_DIR)
    parser.add_argument("--out-dir", type=pathlib.Path, default=TRANSLATIONS_DIR)
    args = parser.parse_args(argv)
    written = compile_all(args.source_dir, args.out_dir, find_lrelease())
    logger.info(f"Wrote {len(written)} translation files to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
