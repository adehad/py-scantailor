"""Changing the reading order renames the outputs of split pages."""

from pathlib import Path

from scantailor.core.batch import output_file_names, rename_split_outputs
from scantailor.core.models import ImageId, PageId, SubPage

SOURCE = Path("/scans/book.tif")
LEFT = PageId(image_id=ImageId(file_path=SOURCE), sub_page=SubPage.LEFT_PAGE)
RIGHT = PageId(image_id=ImageId(file_path=SOURCE), sub_page=SubPage.RIGHT_PAGE)
SINGLE = PageId(
    image_id=ImageId(file_path=Path("/scans/cover.tif")),
    sub_page=SubPage.SINGLE_PAGE,
)


def _write(folder: Path, name: str, content: bytes) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_bytes(content)


def test_each_half_keeps_its_content_under_the_new_number(tmp_path: Path) -> None:
    pages = [SINGLE, LEFT, RIGHT]
    old = output_file_names(pages, "LTR")
    new = output_file_names(pages, "RTL")
    _write(tmp_path, old[LEFT], b"left")
    _write(tmp_path, old[RIGHT], b"right")
    _write(tmp_path, old[SINGLE], b"cover")

    renamed = rename_split_outputs(tmp_path, pages, "LTR", "RTL")

    assert len(renamed) == 2
    assert (tmp_path / new[LEFT]).read_bytes() == b"left"
    assert (tmp_path / new[RIGHT]).read_bytes() == b"right"
    assert not (tmp_path / old[LEFT]).exists()
    assert (tmp_path / old[SINGLE]).read_bytes() == b"cover"


def test_going_back_restores_the_original_names(tmp_path: Path) -> None:
    pages = [LEFT, RIGHT]
    old = output_file_names(pages, "LTR")
    _write(tmp_path, old[LEFT], b"left")
    _write(tmp_path, old[RIGHT], b"right")

    rename_split_outputs(tmp_path, pages, "LTR", "RTL")
    rename_split_outputs(tmp_path, pages, "RTL", "LTR")

    assert (tmp_path / old[LEFT]).read_bytes() == b"left"
    assert (tmp_path / old[RIGHT]).read_bytes() == b"right"


def test_the_layer_folders_are_renamed_too(tmp_path: Path) -> None:
    pages = [LEFT]
    old = output_file_names(pages, "LTR")
    new = output_file_names(pages, "RTL")
    _write(tmp_path / "foreground", old[LEFT], b"fg")

    rename_split_outputs(tmp_path, pages, "LTR", "RTL")

    assert (tmp_path / "foreground" / new[LEFT]).read_bytes() == b"fg"


def test_a_page_that_was_never_written_is_skipped(tmp_path: Path) -> None:
    assert rename_split_outputs(tmp_path, [LEFT, RIGHT], "LTR", "RTL") == []
