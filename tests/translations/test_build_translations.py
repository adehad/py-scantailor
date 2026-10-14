"""Tests for tools/build_translations.py, which compiles .ts sources into .qm files."""

import pathlib
import subprocess
from types import ModuleType

import pytest

from scantailor.translations import TRANSLATIONS_DIR


def test_compile_all_writes_one_qm_per_bundled_ts_file(
    build_script: ModuleType, tmp_path: pathlib.Path
) -> None:
    """Every bundled .ts file yields a .qm with the same stem, template included."""
    written = build_script.compile_all(
        TRANSLATIONS_DIR, tmp_path, build_script.find_lrelease()
    )

    expected_stems = {p.stem for p in TRANSLATIONS_DIR.glob("scantailor_*.ts")}
    assert {p.stem for p in written} == expected_stems
    assert all(p.stat().st_size > 0 for p in written)


def test_compile_all_creates_missing_output_folder(
    build_script: ModuleType, tmp_path: pathlib.Path
) -> None:
    """A not-yet-existing output folder is created rather than failing the build."""
    target = tmp_path / "nested" / "qm"

    build_script.compile_all(TRANSLATIONS_DIR, target, build_script.find_lrelease())

    assert target.is_dir()


def test_compile_translation_rejects_missing_source_file(
    build_script: ModuleType, tmp_path: pathlib.Path
) -> None:
    """A missing .ts source makes lrelease fail, and the error reaches the caller."""
    missing = tmp_path / "scantailor_missing.ts"

    with pytest.raises(subprocess.CalledProcessError):
        build_script.compile_translation(
            missing, tmp_path, build_script.find_lrelease()
        )


def test_find_lrelease_raises_when_tool_is_absent_from_interpreter_folder(
    build_script: ModuleType, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing lrelease beside the interpreter raises FileNotFoundError."""
    monkeypatch.setattr(build_script.sys, "executable", str(tmp_path / "python.exe"))

    with pytest.raises(FileNotFoundError, match="pyside6-lrelease"):
        build_script.find_lrelease()


def test_main_returns_zero_and_writes_files_to_requested_folder(
    build_script: ModuleType, tmp_path: pathlib.Path
) -> None:
    """The command line compiles into --out-dir and exits 0 on success."""
    exit_code = build_script.main(["--out-dir", str(tmp_path)])

    assert exit_code == 0
    assert (tmp_path / "scantailor_de.qm").is_file()
