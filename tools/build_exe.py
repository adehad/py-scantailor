"""Freeze ScanTailor with PyInstaller from the runtime dependencies only.

PyInstaller's hooks sweep in whatever optional imports they find installed, so
a build from the development environment bundles test and docs tools. This
script keeps a separate virtual environment (build/venv) synced from pdm.lock
with the runtime dependencies plus the installer group, removing anything else,
and runs PyInstaller from it. Local builds then match the release build.

Usage:
    pdm run exe [PyInstaller options]
"""

import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Final

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
BUILD_VENV: Final = REPO_ROOT / "build" / "venv"
INSTALLER_GROUP: Final = "installer"


def other_groups(pyproject: Path = REPO_ROOT / "pyproject.toml") -> list[str]:
    """Name every dependency group but the installer's.

    pdm refuses ``--prod`` together with a dependency group, so the build
    environment selects the installer group and leaves out all the others.
    """
    groups = tomllib.loads(pyproject.read_text(encoding="utf-8"))["dependency-groups"]
    return sorted(set(groups) - {INSTALLER_GROUP})


def venv_python(venv: Path) -> Path:
    """Return the interpreter of a virtual environment on this platform."""
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def sync_build_venv(venv: Path = BUILD_VENV) -> Path:
    """Create the build environment if needed and sync it to the lock file.

    Args:
        venv (Path): Where the environment lives.

    Returns:
        Path: Its Python interpreter.
    """
    python = venv_python(venv)
    if not python.is_file():
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    pdm = shutil.which("pdm")
    if pdm is None:
        raise SystemExit("pdm is not on PATH; it syncs the build environment")
    environment = {
        **os.environ,
        "VIRTUAL_ENV": str(venv),
        "PDM_IGNORE_SAVED_PYTHON": "1",
        "PDM_IGNORE_ACTIVE_VENV": "0",
    }
    subprocess.run(
        [
            pdm,
            "sync",
            "--group",
            INSTALLER_GROUP,
            "--without",
            ",".join(other_groups()),
            "--clean",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
    )
    return python


def main(argv: list[str] | None = None) -> int:
    """Sync the build environment and run PyInstaller on app.spec.

    Args:
        argv (list[str] | None): Extra PyInstaller options.

    Returns:
        int: PyInstaller's exit code.
    """
    python = sync_build_venv()
    command = [
        str(python),
        "-m",
        "PyInstaller",
        str(REPO_ROOT / "app.spec"),
        "--noconfirm",
        "--clean",
        *(sys.argv[1:] if argv is None else argv),
    ]
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
