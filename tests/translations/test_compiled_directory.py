"""Tests for finding, or compiling on first use, the catalogues the app loads."""

import importlib.util
import pathlib
import shutil
from types import ModuleType, SimpleNamespace

import pytest
from PySide6 import QtCore, QtWidgets

import scantailor.translations as translations
from scantailor.translations import (
    TRANSLATIONS_DIR,
    compiled_directory,
    load_translation,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture
def source_checkout(tmp_path: pathlib.Path) -> pathlib.Path:
    """A translations folder as a fresh clone has it: German .ts, no .qm."""
    folder = tmp_path / "translations"
    folder.mkdir()
    shutil.copy(TRANSLATIONS_DIR / "scantailor_de.ts", folder)
    return folder


@pytest.fixture
def failing_compiler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any compilation fail the test."""

    def compile_all(*_args: object) -> None:
        pytest.fail("compiled although catalogues were available")

    monkeypatch.setattr(translations, "compile_all", compile_all)


def test_sources_without_catalogues_are_compiled_into_the_cache(
    source_checkout: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """A checkout without .qm files gets them in the cache, not in the package."""
    cache = tmp_path / "cache"

    found = compiled_directory(source_checkout, cache)

    assert found.parent == cache
    assert (found / "scantailor_de.qm").is_file()
    assert not (source_checkout / "scantailor_de.qm").exists()


def test_a_second_start_reuses_the_cached_catalogues(
    source_checkout: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catalogues already compiled for these sources are not compiled again."""
    cache = tmp_path / "cache"
    first = compiled_directory(source_checkout, cache)
    monkeypatch.setattr(translations, "compile_all", pytest.fail)

    assert compiled_directory(source_checkout, cache) == first


def test_an_edited_source_is_compiled_again_into_a_new_folder(
    source_checkout: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """The cache folder is named after the sources, so an edit is never stale."""
    cache = tmp_path / "cache"
    first = compiled_directory(source_checkout, cache)
    source = source_checkout / "scantailor_de.ts"
    source.write_bytes(source.read_bytes() + b"\n")

    second = compiled_directory(source_checkout, cache)

    assert second != first
    assert (second / "scantailor_de.qm").is_file()


@pytest.mark.usefixtures("failing_compiler")
def test_catalogues_beside_the_sources_are_used_as_they_are(
    source_checkout: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """A wheel ships .qm beside the .ts, which needs no compiling."""
    (source_checkout / "scantailor_de.qm").write_bytes(b"compiled")

    assert compiled_directory(source_checkout, tmp_path / "cache") == source_checkout


@pytest.mark.usefixtures("failing_compiler")
def test_a_folder_of_catalogues_only_is_used_as_it_is(tmp_path: pathlib.Path) -> None:
    """A release bundle has .qm files and no sources."""
    (tmp_path / "scantailor_de.qm").write_bytes(b"compiled")

    assert compiled_directory(tmp_path, tmp_path / "cache") == tmp_path


def test_without_lrelease_the_folder_is_returned_and_english_stays(
    source_checkout: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed compilation is logged and leaves the UI in English, not crashing."""

    def missing() -> pathlib.Path:
        raise FileNotFoundError("lrelease not found in the PySide6 package")

    monkeypatch.setattr(translations, "find_lrelease", missing)

    assert compiled_directory(source_checkout, tmp_path / "cache") == source_checkout


def test_a_rejected_source_is_returned_as_the_folder(
    source_checkout: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """A .ts lrelease cannot read leaves the folder, without catalogues, in place."""
    (source_checkout / "scantailor_xx.ts").write_text("not xml", encoding="utf-8")

    assert compiled_directory(source_checkout, tmp_path / "cache") == source_checkout


def test_load_translation_translates_a_source_checkout(
    qapp: QtWidgets.QApplication,
    source_checkout: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    removed_translators: None,
) -> None:
    """Loading from a folder of .ts sources alone gives the German UI."""
    monkeypatch.setattr(translations, "_default_cache_root", lambda: tmp_path / "c")

    load_translation(qapp, QtCore.QLocale("de"), source_checkout)

    assert qapp.translate("AboutDialog", "About") == "Über"


def _build_hook() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "pdm_build", _REPO_ROOT / "pdm_build.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _context(target: str, build_dir: pathlib.Path) -> SimpleNamespace:
    build_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        target=target, root=_REPO_ROOT, ensure_build_dir=lambda: build_dir
    )


def test_the_wheel_build_hook_adds_a_catalogue_for_every_source(
    tmp_path: pathlib.Path,
) -> None:
    """The wheel gets scantailor/translations/<name>.qm for each bundled .ts."""
    files: dict[str, pathlib.Path] = {}

    _build_hook().pdm_build_update_files(_context("wheel", tmp_path), files)

    expected = {
        f"scantailor/translations/{source.stem}.qm"
        for source in TRANSLATIONS_DIR.glob("scantailor_*.ts")
    }
    assert set(files) == expected
    assert all(path.is_file() for path in files.values())


def test_the_wheel_build_hook_skips_the_catalogues_without_pyside6(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A build environment without PySide6 still builds; the app compiles them."""
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    files: dict[str, pathlib.Path] = {}

    _build_hook().pdm_build_update_files(_context("wheel", tmp_path), files)

    assert files == {}
    assert "PySide6 is not in the build environment" in capsys.readouterr().err


@pytest.mark.parametrize("target", ["sdist", "editable"])
def test_the_build_hook_leaves_sdists_and_editable_installs_alone(
    target: str, tmp_path: pathlib.Path
) -> None:
    """The sdist ships the sources; an editable install compiles at first use."""
    files: dict[str, pathlib.Path] = {}

    _build_hook().pdm_build_update_files(_context(target, tmp_path), files)

    assert files == {}
