"""The batch geometry pass decodes a page shrunk to the resolution detection reads."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from scantailor.core import batch, pipeline
from scantailor.core.image_io import load_image
from scantailor.core.models import Dpi, ImageId, PageId, SubPage
from scantailor.core.page_source import LoadedPage, load_page, shrink_page
from scantailor.core.pipeline import (
    DETECTION_DPI,
    PipelineOptions,
    detect_geometry,
    fix_aggregate_page_size,
)
from scantailor.core.stage_sequence import FilterStage, StageSequence
from tests.core.dark_pages import printed_page


def _page_on_a_bed(scale: int) -> NDArray[np.uint8]:
    """A printed page turned 2 degrees on a dark bed, ``scale`` times 300 dpi."""
    page = printed_page(angle=2.0)
    bed = np.full((page.shape[0] + 200, page.shape[1] + 200), 60, dtype=np.uint8)
    bed[100:-100, 100:-100] = page
    height, width = bed.shape
    return cv2.resize(
        bed, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC
    )


def _save(path: Path, pixels: NDArray[np.uint8], dpi: int | None) -> Path:
    extra = {} if dpi is None else {"dpi": (dpi, dpi)}
    Image.fromarray(pixels).save(path, **extra)
    return path


def _image(path: Path) -> ImageId:
    return ImageId(file_path=path)


def test_a_600_dpi_jpeg_decodes_at_half_size_and_reports_its_full_size(
    tmp_path: Path,
) -> None:
    pixels = np.full((1200, 800), 200, dtype=np.uint8)
    path = _save(tmp_path / "page.jpg", pixels, 600)

    shrunk, metadata = load_image(_image(path), min_dpi=300)

    assert shrunk.shape == (600, 400)
    assert (metadata.width, metadata.height) == (800, 1200)
    assert metadata.dpi == Dpi.uniform(600)


def test_a_600_dpi_jpeg_of_odd_size_still_decodes_at_half_size(
    tmp_path: Path,
) -> None:
    path = _save(tmp_path / "page.jpg", np.full((1201, 801), 200, np.uint8), 600)

    shrunk, _ = load_image(_image(path), min_dpi=300)

    assert shrunk.shape == (601, 401)


def test_a_1200_dpi_jpeg_decodes_at_a_quarter_size(tmp_path: Path) -> None:
    path = _save(tmp_path / "page.jpg", np.zeros((800, 600), np.uint8), 1200)

    shrunk, _ = load_image(_image(path), min_dpi=300)

    assert shrunk.shape == (200, 150)


def test_a_600_dpi_png_is_decoded_in_full_then_averaged_to_half_size(
    tmp_path: Path,
) -> None:
    pixels = _page_on_a_bed(1)[:400, :300]
    path = _save(tmp_path / "page.png", pixels, 600)

    shrunk, metadata = load_image(_image(path), min_dpi=300)

    np.testing.assert_array_equal(
        shrunk, cv2.resize(pixels, (150, 200), interpolation=cv2.INTER_AREA)
    )
    assert (metadata.width, metadata.height) == (300, 400)


def test_a_page_under_twice_the_least_resolution_decodes_in_full(
    tmp_path: Path,
) -> None:
    path = _save(tmp_path / "page.jpg", np.zeros((400, 300), np.uint8), 500)

    shrunk, _ = load_image(_image(path), min_dpi=300)

    assert shrunk.shape == (400, 300)


def test_a_page_without_a_resolution_shrinks_by_the_one_assumed_for_it(
    tmp_path: Path,
) -> None:
    path = _save(tmp_path / "page.png", np.zeros((400, 300), np.uint8), None)

    page = load_page(_image(path), min_dpi=300, default_dpi=1200)

    assert page.pixels.shape == (100, 75)
    assert page.size == (300, 400)


def test_the_batch_geometry_pass_decodes_a_600_dpi_page_at_300_dpi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _save(tmp_path / "page.jpg", _page_on_a_bed(2), 600)
    decoded: list[tuple[int, ...]] = []
    load = batch.load_page

    def recording_load(*args, **kwargs) -> LoadedPage:
        page = load(*args, **kwargs)
        decoded.append(page.pixels.shape)
        return page

    monkeypatch.setattr(batch, "load_page", recording_load)

    with ThreadPoolExecutor(1) as pool:
        result = batch.run_batch(
            StageSequence(),
            [PageId(image_id=_image(path), sub_page=SubPage.SINGLE_PAGE)],
            tmp_path / "out",
            end_stage=FilterStage.SELECT_CONTENT,
            executor=pool,  # type: ignore[arg-type]
        )

    height, width = _page_on_a_bed(2).shape
    assert result.failed == {}
    assert decoded == [(height // 2, width // 2)]


@pytest.mark.slow
def test_geometry_from_a_shrunk_decode_matches_the_full_decode(tmp_path: Path) -> None:
    """Angles agree to 1e-6 degrees; boxes within a 300 dpi pixel and a half."""
    path = _save(tmp_path / "page.png", _page_on_a_bed(2), 600)
    page = PageId(image_id=_image(path), sub_page=SubPage.SINGLE_PAGE)
    options = PipelineOptions(dpi=600.0, end_stage=FilterStage.PAGE_LAYOUT)
    full, shrunk = StageSequence(), StageSequence()

    fix_aggregate_page_size([(load_page(page.image_id).pixels, page)], full, options)
    loaded = load_page(page.image_id, min_dpi=DETECTION_DPI)
    detect_geometry(loaded.pixels, page, shrunk, options, full_size=loaded.full_size)

    full_layout = full.page_split_filter.get_params(page).page_layout
    shrunk_layout = shrunk.page_split_filter.get_params(page).page_layout
    assert full_layout is not None and shrunk_layout is not None
    assert shrunk_layout.outline == full_layout.outline
    assert shrunk.deskew_filter.get_params(page).deskew_angle_deg == pytest.approx(
        full.deskew_filter.get_params(page).deskew_angle_deg, abs=1e-6
    )
    full_box = full.select_content_filter.get_params(page).content_box
    shrunk_box = shrunk.select_content_filter.get_params(page).content_box
    for side in ("x", "y", "width", "height"):
        assert getattr(shrunk_box, side) == pytest.approx(
            getattr(full_box, side), abs=3
        )


def test_page_split_on_a_shrunk_spread_stores_the_layout_in_full_size_pixels() -> None:
    left, right = _page_on_a_bed(1), _page_on_a_bed(1)
    spread = np.hstack([left, right])
    height, width = spread.shape
    shrunk = cv2.resize(spread, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
    page = PageId(
        image_id=ImageId(file_path=Path("spread.png")), sub_page=SubPage.SINGLE_PAGE
    )
    full, reduced = StageSequence(), StageSequence()

    full.page_split_filter.settle(spread, page, dpi=600.0)
    reduced.page_split_filter.settle(shrunk, page, dpi=600.0, full_size=(width, height))

    full_params = full.page_split_filter.get_params(page)
    reduced_params = reduced.page_split_filter.get_params(page)
    assert reduced_params.dependencies == full_params.dependencies
    assert reduced_params.page_layout is not None
    assert full_params.page_layout is not None
    assert reduced_params.page_layout.outline == (0.0, 0.0, float(width), float(height))
    assert reduced_params.page_layout.layout_type == full_params.page_layout.layout_type
    full_cuts = np.array(full_params.page_layout.cutter1)
    assert np.abs(np.array(reduced_params.page_layout.cutter1) - full_cuts).max() <= 4


def test_a_shrunk_source_gives_no_full_size_view() -> None:
    state = pipeline._PageState(
        np.zeros((100, 80), np.uint8), 600.0, full_size=(160, 200)
    )

    with pytest.raises(ValueError, match="shrunk"):
        state.view()


def test_a_quarter_turn_swaps_the_full_size_of_a_shrunk_source() -> None:
    state = pipeline._PageState(
        np.zeros((100, 80), np.uint8), 600.0, full_size=(160, 200)
    )

    state.replace_source(np.zeros((80, 100), np.uint8), 90)

    assert state.full_size == (200, 160)
    assert state.source_scale == (0.5, 0.5)


@pytest.mark.parametrize("suffix", [".jpg", ".png"])
def test_shrinking_a_full_decode_gives_the_pixels_of_a_shrunk_decode(
    tmp_path: Path, suffix: str
) -> None:
    pixels = _page_on_a_bed(1)[:401, :301]
    image = _image(_save(tmp_path / f"page{suffix}", pixels, 600))

    shrunk = shrink_page(load_page(image), image, min_dpi=DETECTION_DPI)

    decoded = load_page(image, min_dpi=DETECTION_DPI)
    np.testing.assert_array_equal(shrunk.pixels, decoded.pixels)
    assert (shrunk.dpi, shrunk.size) == (decoded.dpi, decoded.size)
