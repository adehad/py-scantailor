"""Sidecar files of earlier versions are read, then deleted after a save."""

from pathlib import Path

import pytest

from scantailor.core.pdf_source import SOURCE_FILE_NAME, PdfSource

SOURCE = PdfSource(pdf_path=Path("book.pdf"), dpi=300, pages={"a.png": 0, "b.png": 1})


def _write_sidecar(directory: Path, source: PdfSource = SOURCE) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    sidecar = directory / SOURCE_FILE_NAME
    sidecar.write_text(source.model_dump_json(), encoding="utf-8")
    return sidecar


def test_discarding_removes_the_files_of_every_folder(tmp_path: Path) -> None:
    first = _write_sidecar(tmp_path / "one")
    second = _write_sidecar(tmp_path / "two")

    PdfSource.discard_sidecars([tmp_path / "one" / "a.png", tmp_path / "two" / "a.png"])

    assert not first.exists()
    assert not second.exists()


def test_discarding_where_there_is_no_file_does_nothing(tmp_path: Path) -> None:
    PdfSource.discard_sidecars([tmp_path / "a.png"])
    PdfSource.discard_sidecars([])


def test_a_sidecar_that_cannot_be_deleted_is_left_without_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sidecar = _write_sidecar(tmp_path)

    def refuse(self: Path, missing_ok: bool = False) -> None:
        raise PermissionError(self)

    monkeypatch.setattr(Path, "unlink", refuse)

    PdfSource.discard_sidecars([tmp_path / "a.png"])

    assert sidecar.exists()


def test_a_damaged_sidecar_gives_nothing(tmp_path: Path) -> None:
    (tmp_path / SOURCE_FILE_NAME).write_text("{", encoding="utf-8")

    assert PdfSource.find([tmp_path / "a.png"]) is None


def test_finding_leaves_the_file_where_it_is(tmp_path: Path) -> None:
    sidecar = _write_sidecar(tmp_path)

    assert PdfSource.find([tmp_path / "a.png"]) == SOURCE
    assert sidecar.exists()
