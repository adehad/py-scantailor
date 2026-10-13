"""Progress and cancel of ``export_pdf``, and the bookmarks and labels it carries."""

import threading
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pikepdf
import pytest
from pikepdf import Dictionary, Name, OutlineItem

from scantailor.core.batch import spawn_pool
from scantailor.core.pdf_export import export_pdf
from scantailor.core.pdf_pool import PdfCancelled

PAGES = 4


def _pages(count: int = PAGES) -> list[tuple[np.ndarray, float]]:
    return [(np.full((20, 30), 255, dtype=np.uint8), 72.0) for _ in range(count)]


@pytest.fixture(scope="module")
def pool() -> Iterator[ProcessPoolExecutor]:
    with spawn_pool(2) as executor:
        yield executor


def _source_with_outline(path: Path, pages: int = PAGES) -> Path:
    pdf = pikepdf.Pdf.new()
    for _ in range(pages):
        pdf.add_blank_page(page_size=(100, 100))
    with pdf.open_outline() as outline:
        chapter = OutlineItem("Chapter", 1)
        chapter.children.append(OutlineItem("Section", 2, "XYZ", left=5, top=90))
        closed = OutlineItem("Appendix", 3)
        closed.is_closed = True
        closed.children.append(OutlineItem("Note", 3))
        outline.root.extend([OutlineItem("Cover", 0), chapter, closed])
    labels = pikepdf.Array(
        [
            0,
            Dictionary(S=Name.r),
            2,
            Dictionary(S=Name.D, St=1, P="p-"),
        ]
    )
    pdf.Root.PageLabels = Dictionary(Nums=labels)
    pdf.save(path)
    return path


def _titles(item: OutlineItem, pdf: pikepdf.Pdf) -> tuple[str, int | None, list]:
    page = None
    if item.destination is not None:
        page = [p.objgen for p in pdf.pages].index(item.destination[0].objgen)
    return (str(item.title), page, [_titles(c, pdf) for c in item.children])


def _outline_of(path: Path) -> list[tuple]:
    with pikepdf.open(path) as pdf, pdf.open_outline() as outline:
        return [_titles(item, pdf) for item in outline.root]


def test_progress_covers_every_page_then_the_save(tmp_path: Path) -> None:
    seen: list[tuple[int, int, str]] = []

    export_pdf(_pages(), tmp_path / "out.pdf", on_progress=lambda *a: seen.append(a))

    assert seen == [(n, PAGES, "Writing PDF") for n in range(1, PAGES + 1)] + [
        (PAGES, PAGES, "Saving PDF")
    ]


def test_pool_progress_has_an_encoding_phase_before_the_writing_phase(
    tmp_path: Path, pool: ProcessPoolExecutor
) -> None:
    seen: list[tuple[int, int, str]] = []

    export_pdf(
        _pages(),
        tmp_path / "out.pdf",
        executor=pool,
        max_workers=2,
        on_progress=lambda *a: seen.append(a),
    )

    phases = [phase for _, _, phase in seen]
    assert phases == ["Encoding pages"] * PAGES + ["Writing PDF"] * PAGES + [
        "Saving PDF"
    ]
    assert [done for done, _, phase in seen if phase == "Encoding pages"] == [
        1,
        2,
        3,
        4,
    ]


def test_cancel_while_writing_stops_without_creating_the_file(tmp_path: Path) -> None:
    cancel = threading.Event()
    out = tmp_path / "out.pdf"

    def stop_after_second_page(done: int, _total: int, _phase: str) -> None:
        if done == 2:
            cancel.set()

    with pytest.raises(PdfCancelled):
        export_pdf(_pages(), out, cancel=cancel, on_progress=stop_after_second_page)

    assert not out.exists()


def test_cancel_set_before_the_start_writes_nothing(tmp_path: Path) -> None:
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(PdfCancelled):
        export_pdf(_pages(), tmp_path / "out.pdf", cancel=cancel)

    assert list(tmp_path.iterdir()) == []


def test_bookmarks_keep_their_titles_nesting_and_pages(tmp_path: Path) -> None:
    source = _source_with_outline(tmp_path / "source.pdf")

    export_pdf(_pages(), tmp_path / "out.pdf", outline_source=source)

    assert _outline_of(tmp_path / "out.pdf") == [
        ("Cover", 0, []),
        ("Chapter", 1, [("Section", 2, [])]),
        ("Appendix", 3, [("Note", 3, [])]),
    ]


def test_a_collapsed_bookmark_stays_collapsed(tmp_path: Path) -> None:
    source = _source_with_outline(tmp_path / "source.pdf")

    export_pdf(_pages(), tmp_path / "out.pdf", outline_source=source)

    with pikepdf.open(tmp_path / "out.pdf") as pdf, pdf.open_outline() as outline:
        assert [item.is_closed for item in outline.root] == [False, False, True]


def test_page_labels_are_carried_over(tmp_path: Path) -> None:
    source = _source_with_outline(tmp_path / "source.pdf")

    export_pdf(_pages(), tmp_path / "out.pdf", outline_source=source)

    with pikepdf.open(tmp_path / "out.pdf") as pdf:
        assert [pdf.pages[i].label for i in range(PAGES)] == ["i", "ii", "p-1", "p-2"]


def test_nothing_is_carried_when_the_page_count_differs(tmp_path: Path) -> None:
    source = _source_with_outline(tmp_path / "source.pdf")

    export_pdf(_pages(PAGES - 1), tmp_path / "out.pdf", outline_source=source)

    with pikepdf.open(tmp_path / "out.pdf") as pdf:
        assert "/PageLabels" not in pdf.Root
        assert "/Outlines" not in pdf.Root


def test_a_bookmark_to_a_named_destination_is_resolved(tmp_path: Path) -> None:
    pdf = pikepdf.Pdf.new()
    for _ in range(2):
        pdf.add_blank_page(page_size=(100, 100))
    target = Dictionary(D=pikepdf.Array([pdf.pages[1].obj, Name.Fit]))
    pdf.Root.Dests = Dictionary(Target=target)
    with pdf.open_outline() as outline:
        outline.root.append(OutlineItem("Second", Name.Target))
    source = tmp_path / "named.pdf"
    pdf.save(source)

    export_pdf(_pages(2), tmp_path / "out.pdf", outline_source=source)

    assert _outline_of(tmp_path / "out.pdf") == [("Second", 1, [])]


def test_a_bookmark_pointing_nowhere_is_dropped_but_its_children_stay(
    tmp_path: Path,
) -> None:
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(100, 100))
    with pdf.open_outline() as outline:
        group = OutlineItem("Group")
        group.children.append(OutlineItem("Child", 0))
        outline.root.extend([group, OutlineItem("Lost", Name.Missing)])
    source = tmp_path / "odd.pdf"
    pdf.save(source)

    export_pdf(_pages(1), tmp_path / "out.pdf", outline_source=source)

    assert _outline_of(tmp_path / "out.pdf") == [("Group", None, [("Child", 0, [])])]


def test_a_source_without_either_structure_adds_none(tmp_path: Path) -> None:
    plain = pikepdf.Pdf.new()
    for _ in range(PAGES):
        plain.add_blank_page(page_size=(100, 100))
    plain.save(tmp_path / "plain.pdf")

    export_pdf(_pages(), tmp_path / "out.pdf", outline_source=tmp_path / "plain.pdf")

    with pikepdf.open(tmp_path / "out.pdf") as pdf:
        assert "/PageLabels" not in pdf.Root
        assert "/Outlines" not in pdf.Root


def test_an_unreadable_outline_source_does_not_fail_the_export(tmp_path: Path) -> None:
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf")

    export_pdf(_pages(), tmp_path / "out.pdf", outline_source=broken)

    with pikepdf.open(tmp_path / "out.pdf") as pdf:
        assert len(pdf.pages) == PAGES
