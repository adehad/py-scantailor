"""Tests for relinking directory prefixes of image paths."""

from pathlib import Path

from scantailor.core.relinker import Relinker, image_directories


def test_path_without_a_mapped_ancestor_is_unchanged(tmp_path: Path):
    relinker = Relinker()
    relinker.relink(tmp_path / "old", tmp_path / "new")

    unrelated = tmp_path / "other" / "a.png"
    assert relinker.substitute(unrelated) == unrelated


def test_every_file_below_a_relinked_directory_moves_with_it(tmp_path: Path):
    relinker = Relinker()
    relinker.relink(tmp_path / "old", tmp_path / "new")

    assert relinker.substitute(tmp_path / "old" / "a.png") == tmp_path / "new" / "a.png"
    nested = relinker.substitute(tmp_path / "old" / "sub" / "b.png")
    assert nested == tmp_path / "new" / "sub" / "b.png"


def test_nearest_mapped_ancestor_wins(tmp_path: Path):
    relinker = Relinker()
    relinker.relink(tmp_path / "old", tmp_path / "new")
    relinker.relink(tmp_path / "old" / "sub", tmp_path / "elsewhere")

    nested = relinker.substitute(tmp_path / "old" / "sub" / "b.png")
    assert nested == tmp_path / "elsewhere" / "b.png"
    assert relinker.substitute(tmp_path / "old" / "a.png") == tmp_path / "new" / "a.png"


def test_relinking_a_path_to_itself_removes_the_mapping(tmp_path: Path):
    relinker = Relinker()
    relinker.relink(tmp_path / "old", tmp_path / "new")
    relinker.relink(tmp_path / "old", tmp_path / "old")

    assert relinker.is_empty()


def test_unlink_forgets_a_mapping_and_ignores_unknown_paths(tmp_path: Path):
    relinker = Relinker()
    relinker.relink(tmp_path / "old", tmp_path / "new")

    relinker.unlink(tmp_path / "unknown")
    assert relinker.target_of(tmp_path / "old") == tmp_path / "new"
    relinker.unlink(tmp_path / "old")
    assert relinker.target_of(tmp_path / "old") is None


def test_image_directories_lists_each_folder_once_in_sorted_order(tmp_path: Path):
    files = [
        tmp_path / "b" / "1.png",
        tmp_path / "a" / "2.png",
        tmp_path / "b" / "3.png",
    ]

    assert image_directories(files, only_missing=False) == [
        tmp_path / "a",
        tmp_path / "b",
    ]


def test_image_directories_can_keep_only_folders_of_missing_files(tmp_path: Path):
    present = tmp_path / "here" / "1.png"
    present.parent.mkdir()
    present.write_bytes(b"x")
    gone = tmp_path / "gone" / "2.png"

    assert image_directories([present, gone], only_missing=True) == [tmp_path / "gone"]
