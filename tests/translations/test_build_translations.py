"""Tests for compiling the .ts sources into .qm catalogues, and the build tool."""

import pathlib
import subprocess
from types import ModuleType

import pytest

from scantailor.translations import TRANSLATIONS_DIR, catalogues
from scantailor.translations.catalogues import (
    compile_all,
    compile_translation,
    find_lrelease,
    sources_digest,
)


def test_compile_all_writes_one_qm_per_bundled_ts_file(tmp_path: pathlib.Path) -> None:
    """Every bundled .ts file yields a .qm with the same stem, template included."""
    written = compile_all(TRANSLATIONS_DIR, tmp_path, find_lrelease())

    expected_stems = {p.stem for p in TRANSLATIONS_DIR.glob("scantailor_*.ts")}
    assert {p.stem for p in written} == expected_stems
    assert all(p.stat().st_size > 0 for p in written)


def test_compile_all_creates_missing_output_folder(tmp_path: pathlib.Path) -> None:
    """A not-yet-existing output folder is created rather than failing the build."""
    target = tmp_path / "nested" / "qm"

    compile_all(TRANSLATIONS_DIR, target, find_lrelease())

    assert target.is_dir()


def test_compile_translation_rejects_missing_source_and_leaves_no_partial_file(
    tmp_path: pathlib.Path,
) -> None:
    """A missing .ts source makes lrelease fail, and nothing is left behind."""
    missing = tmp_path / "scantailor_missing.ts"

    with pytest.raises(subprocess.CalledProcessError):
        compile_translation(missing, tmp_path, find_lrelease())

    assert list(tmp_path.iterdir()) == []


def test_find_lrelease_raises_when_pyside6_is_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without PySide6 there is no lrelease, which is a FileNotFoundError."""
    monkeypatch.setattr(catalogues.importlib.util, "find_spec", lambda _name: None)

    with pytest.raises(FileNotFoundError, match="lrelease"):
        find_lrelease()


def test_sources_digest_changes_when_a_source_changes(tmp_path: pathlib.Path) -> None:
    """Editing one catalogue source gives the folder a new digest."""
    source = tmp_path / "scantailor_de.ts"
    source.write_bytes((TRANSLATIONS_DIR / "scantailor_de.ts").read_bytes())
    before = sources_digest(tmp_path)

    source.write_bytes(source.read_bytes() + b"\n")

    assert sources_digest(tmp_path) != before


def test_main_returns_zero_and_writes_files_to_requested_folder(
    build_script: ModuleType, tmp_path: pathlib.Path
) -> None:
    """The command line compiles into --out-dir and exits 0 on success."""
    exit_code = build_script.main(["--out-dir", str(tmp_path)])

    assert exit_code == 0
    assert (tmp_path / "scantailor_de.qm").is_file()
