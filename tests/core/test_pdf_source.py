"""Rendered-page sources: earlier sidecars, and matching pages by folder."""

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

    assert PdfSource.find([tmp_path / "a.png"]) == []


def test_finding_reads_the_sidecar_and_leaves_the_file_where_it_is(
    tmp_path: Path,
) -> None:
    sidecar = _write_sidecar(tmp_path)

    found = PdfSource.find([tmp_path / "a.png"])

    assert found == [SOURCE.model_copy(update={"pages_dir": tmp_path})]
    assert sidecar.exists()


def test_finding_returns_one_source_per_folder_that_has_a_sidecar(
    tmp_path: Path,
) -> None:
    other = SOURCE.model_copy(update={"pdf_path": Path("other.pdf")})
    _write_sidecar(tmp_path / "one")
    _write_sidecar(tmp_path / "two", other)

    found = PdfSource.find([tmp_path / "one" / "a.png", tmp_path / "two" / "a.png"])

    assert [source.pdf_path for source in found] == [
        Path("book.pdf"),
        Path("other.pdf"),
    ]


def test_a_page_matches_only_the_source_whose_folder_holds_it(tmp_path: Path) -> None:
    first = PdfSource(
        pdf_path=Path("one.pdf"),
        dpi=300,
        pages_dir=tmp_path / "one_pages",
        pages={"page_0001.png": 0},
    )
    second = PdfSource(
        pdf_path=Path("two.pdf"),
        dpi=300,
        pages_dir=tmp_path / "two_pages",
        pages={"page_0001.png": 4},
    )

    assert first.page_index(tmp_path / "one_pages" / "page_0001.png") == 0
    assert second.page_index(tmp_path / "two_pages" / "page_0001.png") == 4
    assert first.page_index(tmp_path / "two_pages" / "page_0001.png") is None


def test_a_source_without_a_folder_matches_pages_by_name_alone(tmp_path: Path) -> None:
    legacy = PdfSource(pdf_path=Path("old.pdf"), dpi=300, pages={"page_0002.png": 1})

    assert legacy.pages_dir is None
    assert legacy.page_index(tmp_path / "anywhere" / "page_0002.png") == 1
    assert legacy.page_index(tmp_path / "anywhere" / "page_0009.png") is None
