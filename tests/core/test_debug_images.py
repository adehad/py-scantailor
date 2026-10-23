"""The collector that stores a render's intermediate images in temporary files."""

import gc

import cv2
import numpy as np

from scantailor.core import debug_images
from scantailor.core.debug_images import DebugImages, as_displayable


def test_added_image_is_written_to_a_png_that_decodes_to_the_same_pixels():
    gray = np.arange(64, dtype=np.uint8).reshape(8, 8)
    with DebugImages() as dbg:
        dbg.add(gray, "gray150")

        (item,) = dbg.images
        assert item.label == "gray150"
        assert item.path.suffix == ".png"
        decoded = cv2.imread(str(item.path), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(decoded, gray)


def test_images_keep_the_order_they_were_added_in():
    with DebugImages() as dbg:
        for label in ("bw150", "content", "despeckled"):
            dbg.add(np.zeros((4, 4), dtype=np.uint8), label)

        assert [item.label for item in dbg.images] == [
            "bw150",
            "content",
            "despeckled",
        ]


def test_bool_mask_is_drawn_black_where_true_on_white():
    mask = np.array([[True, False]])

    assert as_displayable(mask).tolist() == [[0, 255]]


def test_float_image_is_stretched_to_the_full_gray_range():
    values = np.array([[-1.0, 0.0, 1.0]])

    assert as_displayable(values).tolist() == [[0, 128, 255]]


def test_flat_float_image_becomes_black_instead_of_dividing_by_zero():
    assert as_displayable(np.full((2, 2), 3.5)).tolist() == [[0, 0], [0, 0]]


def test_collector_makes_no_directory_until_the_first_image():
    dbg = DebugImages()

    assert dbg.directory is None
    assert dbg.images == ()


def test_close_deletes_the_files_and_empties_the_collector():
    dbg = DebugImages()
    dbg.add(np.zeros((4, 4), dtype=np.uint8), "mask")
    directory = dbg.directory
    assert directory is not None and directory.is_dir()

    dbg.close()

    assert not directory.exists()
    assert dbg.images == ()


def test_leaving_the_with_block_deletes_the_files():
    with DebugImages() as dbg:
        dbg.add(np.zeros((4, 4), dtype=np.uint8), "mask")
        directory = dbg.directory

    assert directory is not None and not directory.exists()


def test_collector_that_nobody_closed_deletes_its_files_when_collected():
    dbg = DebugImages()
    dbg.add(np.zeros((4, 4), dtype=np.uint8), "mask")
    directory = dbg.directory
    assert directory is not None

    del dbg
    gc.collect()

    assert not directory.exists()


def test_closed_collector_records_into_a_new_directory():
    dbg = DebugImages()
    dbg.add(np.zeros((4, 4), dtype=np.uint8), "first")
    first = dbg.directory
    dbg.close()

    dbg.add(np.zeros((4, 4), dtype=np.uint8), "second")

    assert dbg.directory != first
    assert [item.label for item in dbg.images] == ["second"]
    dbg.close()


def test_image_whose_write_raises_is_skipped_and_the_render_goes_on(monkeypatch):
    def failing_write(*_args, **_kwargs):
        raise cv2.error("disk full")

    with DebugImages() as dbg:
        dbg.add(np.zeros((4, 4), dtype=np.uint8), "first")
        monkeypatch.setattr(cv2, "imwrite", failing_write)
        dbg.add(np.zeros((4, 4), dtype=np.uint8), "lost")

        assert [item.label for item in dbg.images] == ["first"]


def test_image_is_skipped_when_no_temp_directory_can_be_made(monkeypatch):
    def failing_mkdtemp(*_args, **_kwargs):
        raise OSError("read-only temp")

    monkeypatch.setattr(debug_images.tempfile, "mkdtemp", failing_mkdtemp)
    dbg = DebugImages()

    dbg.add(np.zeros((4, 4), dtype=np.uint8), "lost")

    assert dbg.images == ()
    assert dbg.directory is None
