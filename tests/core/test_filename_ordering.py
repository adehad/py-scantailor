"""Tests for natural file name ordering and the page order it gives a project."""

from pathlib import Path

from PIL import Image

from scantailor.core.filename_ordering import smart_filename_key, sort_smart
from scantailor.core.image_io import list_image_pages


def _names(paths: list[Path]) -> list[str]:
    return [path.name for path in paths]


def test_numbers_compare_by_value_not_by_text(tmp_path: Path):
    paths = [tmp_path / name for name in ("p10.png", "p2.png", "p1.png", "p100.png")]

    assert _names(sort_smart(paths)) == ["p1.png", "p2.png", "p10.png", "p100.png"]


def test_digit_sorts_before_letter_at_the_same_position(tmp_path: Path):
    paths = [tmp_path / name for name in ("a.png", "1.png", "b1.png")]

    assert _names(sort_smart(paths)) == ["1.png", "a.png", "b1.png"]


def test_mixed_case_names_order_by_character_code_after_numbers(tmp_path: Path):
    paths = [tmp_path / name for name in ("b2.png", "B10.png", "B2.png", "a1.png")]

    assert _names(sort_smart(paths)) == ["B2.png", "B10.png", "a1.png", "b2.png"]


def test_names_equal_apart_from_leading_zeros_fall_back_to_plain_order(tmp_path: Path):
    paths = [tmp_path / name for name in ("p1.png", "p01.png", "p001.png")]

    assert _names(sort_smart(paths)) == ["p001.png", "p01.png", "p1.png"]


def test_shorter_name_precedes_one_it_is_a_prefix_of(tmp_path: Path):
    paths = [tmp_path / name for name in ("scan-2b.png", "scan-2.png")]

    assert _names(sort_smart(paths)) == ["scan-2.png", "scan-2b.png"]


def test_files_are_grouped_by_folder_before_name(tmp_path: Path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    paths = [second / "p1.png", first / "p10.png", first / "p2.png", second / "p0.png"]

    assert sort_smart(paths) == [
        first / "p2.png",
        first / "p10.png",
        second / "p0.png",
        second / "p1.png",
    ]


def test_same_name_in_two_folders_stays_two_distinct_ordered_entries(tmp_path: Path):
    paths = [tmp_path / "b" / "p1.png", tmp_path / "a" / "p1.png"]

    ordered = sort_smart(paths)

    assert ordered == [tmp_path / "a" / "p1.png", tmp_path / "b" / "p1.png"]
    assert smart_filename_key(ordered[0]) < smart_filename_key(ordered[1])


def test_sorting_does_not_change_the_input(tmp_path: Path):
    paths = [tmp_path / "p2.png", tmp_path / "p1.png"]

    sort_smart(paths)

    assert _names(paths) == ["p2.png", "p1.png"]


def test_relative_and_absolute_spellings_of_a_file_share_a_key():
    assert smart_filename_key("p1.png") == smart_filename_key(Path("p1.png").absolute())


def test_non_ascii_digits_compare_by_value(tmp_path: Path):
    paths = [tmp_path / name for name in ("p10.png", "p٣.png", "p٢.png")]

    assert _names(sort_smart(paths)) == ["p٢.png", "p٣.png", "p10.png"]


def test_multi_page_tiff_keeps_page_order_inside_the_file_order(tmp_path: Path):
    for name, frames in (("scan10.tif", 3), ("scan2.tif", 2)):
        pages = [Image.new("L", (20 + index, 30)) for index in range(frames)]
        pages[0].save(tmp_path / name, save_all=True, append_images=pages[1:])

    entries = [
        image_id
        for path in sort_smart(tmp_path.glob("*.tif"))
        for image_id, _metadata in list_image_pages(path)
    ]

    assert [(entry.file_path.name, entry.page) for entry in entries] == [
        ("scan2.tif", 1),
        ("scan2.tif", 2),
        ("scan10.tif", 1),
        ("scan10.tif", 2),
        ("scan10.tif", 3),
    ]
