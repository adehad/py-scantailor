"""Project DPI takes precedence over the file's when a page is loaded and processed."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scantailor.core.batch import PageDetected, PageDone, run_batch
from scantailor.core.image_io import load_image, write_image
from scantailor.core.models import Dpi, ImageId, PageId
from scantailor.core.page_source import DpiOverrides, dpi_overrides_of, load_page
from scantailor.core.project import ImageMetadata, Project
from scantailor.core.stage_sequence import StageSequence
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.select_content.content_box import ContentBox

FILE_DPI = 72
FIXED_DPI = 300


def _write_source(path: Path, dpi: int) -> ImageId:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.full((400, 300), 220, dtype=np.uint8)
    pixels[100:300, 80:220] = 40
    write_image(path, pixels, Dpi.uniform(dpi))
    return ImageId(file_path=path)


def _configure(stages: StageSequence, page: PageId) -> None:
    stages.page_split_filter.set_layout_type(page, LayoutType.SINGLE_PAGE_UNCUT)
    stages.deskew_filter.set_manual_angle(page, 0.0)
    stages.select_content_filter.set_manual_content(
        page, ContentBox(x=80, y=100, width=140, height=200)
    )
    stages.output_filter.set_output_dpi(page, Dpi.uniform(FIXED_DPI))


class TestLoadPage:
    def test_override_wins_over_the_dpi_stored_in_the_file(self, tmp_path: Path):
        image_id = _write_source(tmp_path / "a.tif", FILE_DPI)

        page = load_page(image_id, {image_id: Dpi.uniform(FIXED_DPI)})

        assert page.dpi == Dpi.uniform(FIXED_DPI)
        assert page.resolution() == Dpi.uniform(FIXED_DPI)

    def test_pages_without_an_override_keep_the_file_dpi(self, tmp_path: Path):
        image_id = _write_source(tmp_path / "a.tif", FILE_DPI)
        other = ImageId(file_path=tmp_path / "other.tif")

        with_unrelated = load_page(image_id, {other: Dpi.uniform(FIXED_DPI)})

        assert with_unrelated.dpi == Dpi.uniform(FILE_DPI)
        assert load_page(image_id).dpi == Dpi.uniform(FILE_DPI)

    def test_null_override_is_ignored(self, tmp_path: Path):
        image_id = _write_source(tmp_path / "a.tif", FILE_DPI)

        page = load_page(image_id, {image_id: Dpi()})

        assert page.dpi == Dpi.uniform(FILE_DPI)

    def test_pixels_are_the_same_with_and_without_override(self, tmp_path: Path):
        image_id = _write_source(tmp_path / "a.tif", FILE_DPI)

        fixed = load_page(image_id, {image_id: Dpi.uniform(FIXED_DPI)})

        np.testing.assert_array_equal(fixed.pixels, load_image(image_id)[0])

    def test_file_without_dpi_falls_back_to_the_default_on_both_axes(
        self, tmp_path: Path
    ):
        path = tmp_path / "nodpi.png"
        Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(path)

        page = load_page(ImageId(file_path=path))

        assert page.dpi.is_null()
        assert page.resolution(default=123.0) == Dpi.uniform(123)


class TestDpiOverridesOf:
    def test_only_images_with_a_defined_project_dpi_are_included(self, tmp_path: Path):
        fixed, undefined = (ImageId(file_path=tmp_path / n) for n in ("a", "b"))
        project = Project()
        project.add_image(fixed, ImageMetadata(width=3, height=3, dpi=Dpi.uniform(300)))
        project.add_image(undefined, ImageMetadata(width=3, height=3))

        assert dpi_overrides_of(project) == {fixed: Dpi.uniform(300)}

    def test_no_project_gives_no_overrides(self):
        assert dpi_overrides_of(None) == {}


def _run_batch(
    tmp_path: Path, name: str, image_id: ImageId, overrides: DpiOverrides | None
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Batch one page; return the written image's size and DPI."""
    page = PageId(image_id=image_id)
    stages = StageSequence()
    _configure(stages, page)
    events = []
    result = run_batch(
        stages,
        [page],
        tmp_path / name,
        max_workers=1,
        on_progress=events.append,
        dpi_overrides=overrides,
    )
    assert result.failed == {}
    assert [type(e) for e in events] == [PageDetected, PageDone]
    with Image.open(result.written[page]) as image:
        dpi = (round(image.info["dpi"][0]), round(image.info["dpi"][1]))
        return image.size, dpi


@pytest.mark.slow
class TestBatchUsesProjectDpi:
    def test_fixed_dpi_gives_the_output_of_a_file_saved_at_that_dpi(
        self, tmp_path: Path
    ):
        low = _write_source(tmp_path / "in" / "low.tif", FILE_DPI)
        high = _write_source(tmp_path / "in" / "high.tif", FIXED_DPI)

        fixed = _run_batch(tmp_path, "fixed", low, {low: Dpi.uniform(FIXED_DPI)})
        reference = _run_batch(tmp_path, "reference", high, None)

        assert fixed == reference
        assert fixed[1] == (FIXED_DPI, FIXED_DPI)

    def test_unfixed_page_is_sized_by_the_file_dpi(self, tmp_path: Path):
        low = _write_source(tmp_path / "in" / "low.tif", FILE_DPI)

        unfixed = _run_batch(tmp_path, "unfixed", low, None)
        fixed = _run_batch(tmp_path, "fixed", low, {low: Dpi.uniform(FIXED_DPI)})

        assert unfixed[0][0] > fixed[0][0]
