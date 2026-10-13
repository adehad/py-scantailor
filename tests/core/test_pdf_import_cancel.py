"""Cancel and progress of ``import_pdf``, and what a cancel leaves behind."""

import threading
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pikepdf
import pytest

from scantailor.core.batch import spawn_pool
from scantailor.core.pdf_import import import_pdf
from scantailor.core.pdf_pool import PdfCancelled

PAGE_COUNT = 4


def _write_pdf(path: Path, pages: int = PAGE_COUNT, shade: str = "0") -> Path:
    pdf = pikepdf.Pdf.new()
    for _ in range(pages):
        page = pdf.add_blank_page(page_size=(72, 72))
        page.Contents = pdf.make_stream(f"{shade} g 10 10 20 20 re f".encode())
    pdf.save(path)
    return path


@pytest.fixture(scope="module")
def pool() -> Iterator[ProcessPoolExecutor]:
    with spawn_pool(2) as executor:
        yield executor


def _leftovers(parent: Path) -> list[str]:
    return sorted(p.name for p in parent.iterdir())


def test_progress_is_reported_after_every_page(tmp_path: Path) -> None:
    source = _write_pdf(tmp_path / "book.pdf")
    seen: list[tuple[int, int]] = []

    import_pdf(source, tmp_path / "pages", dpi=72, on_page=lambda *a: seen.append(a))

    assert seen == [(n, PAGE_COUNT) for n in range(1, PAGE_COUNT + 1)]


def test_cancel_during_the_import_stops_it_and_leaves_no_files(tmp_path: Path) -> None:
    source = _write_pdf(tmp_path / "book.pdf")
    cancel = threading.Event()
    rendered: list[int] = []

    def after_page(done: int, _total: int) -> None:
        rendered.append(done)
        cancel.set()

    with pytest.raises(PdfCancelled):
        import_pdf(
            source, tmp_path / "pages", dpi=72, cancel=cancel, on_page=after_page
        )

    assert rendered == [1]
    assert not (tmp_path / "pages").exists()
    assert _leftovers(tmp_path) == ["book.pdf"]


def test_cancel_set_before_the_start_renders_nothing(tmp_path: Path) -> None:
    source = _write_pdf(tmp_path / "book.pdf")
    cancel = threading.Event()
    cancel.set()
    seen: list[int] = []

    with pytest.raises(PdfCancelled):
        import_pdf(
            source,
            tmp_path / "pages",
            dpi=72,
            cancel=cancel,
            on_page=lambda done, _total: seen.append(done),
        )

    assert seen == []
    assert _leftovers(tmp_path) == ["book.pdf"]


def test_cancelled_second_import_leaves_the_earlier_import_untouched(
    tmp_path: Path,
) -> None:
    pages = tmp_path / "pages"
    first = import_pdf(_write_pdf(tmp_path / "book.pdf", shade="0"), pages, dpi=72)
    before = {path.name: path.read_bytes() for path in first}
    cancel = threading.Event()

    with pytest.raises(PdfCancelled):
        import_pdf(
            _write_pdf(tmp_path / "book.pdf", shade="1"),
            pages,
            dpi=72,
            cancel=cancel,
            on_page=lambda *_: cancel.set(),
        )

    assert {p.name: p.read_bytes() for p in pages.iterdir()} == before


def test_failed_second_import_leaves_the_earlier_import_untouched(
    tmp_path: Path,
) -> None:
    pages = tmp_path / "pages"
    first = import_pdf(_write_pdf(tmp_path / "book.pdf"), pages, dpi=72)
    before = {path.name: path.read_bytes() for path in first}

    def fail_after_first_page(_done: int, _total: int) -> None:
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        import_pdf(
            _write_pdf(tmp_path / "other.pdf", shade="1"),
            pages,
            dpi=72,
            on_page=fail_after_first_page,
        )

    assert {p.name: p.read_bytes() for p in pages.iterdir()} == before
    assert _leftovers(tmp_path) == ["book.pdf", "other.pdf", "pages"]


def test_completed_second_import_replaces_the_pages(tmp_path: Path) -> None:
    pages = tmp_path / "pages"
    import_pdf(_write_pdf(tmp_path / "book.pdf", shade="0"), pages, dpi=72)
    before = (pages / "page_0001.png").read_bytes()

    import_pdf(_write_pdf(tmp_path / "book.pdf", shade="1"), pages, dpi=72)

    assert (pages / "page_0001.png").read_bytes() != before
    assert _leftovers(tmp_path) == ["book.pdf", "pages"]


def test_pool_import_reports_every_page_and_names_the_pages_in_order(
    tmp_path: Path, pool: ProcessPoolExecutor
) -> None:
    source = _write_pdf(tmp_path / "book.pdf", pages=6)
    seen: list[tuple[int, int]] = []

    paths = import_pdf(
        source,
        tmp_path / "pool",
        dpi=72,
        executor=pool,
        max_workers=2,
        on_page=lambda *a: seen.append(a),
    )

    assert [p.name for p in paths] == [f"page_{n:04d}.png" for n in range(1, 7)]
    assert seen[-1] == (6, 6)
    assert [done for done, _ in seen] == sorted(done for done, _ in seen)


def test_pool_import_cancelled_after_the_first_run_leaves_no_files(
    tmp_path: Path, pool: ProcessPoolExecutor
) -> None:
    source = _write_pdf(tmp_path / "book.pdf", pages=8)
    cancel = threading.Event()

    with pytest.raises(PdfCancelled):
        import_pdf(
            source,
            tmp_path / "pool",
            dpi=72,
            executor=pool,
            max_workers=2,
            cancel=cancel,
            on_page=lambda *_: cancel.set(),
        )

    assert _leftovers(tmp_path) == ["book.pdf"]
