"""The TIFF compression choice reaches the files a batch writes."""

import uuid
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scantailor.core.batch import (
    _PageJob,
    _PhaseConfig,
    _render_page,
    collect_effective_stage_params,
    collect_stage_defaults,
    run_batch,
)
from scantailor.core.image_io import write_image
from scantailor.core.models import Dpi, ImageId, PageId, SubPage
from scantailor.core.settings import (
    ApplicationSettings,
    TiffCompression,
    TiffCompressions,
)
from scantailor.core.stage_sequence import StageSequence
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.select_content.content_box import ContentBox

COMPRESSION_TAG = 259
SOURCE_DPI = 300


def _make_page(tmp_path: Path, mode: ColorMode) -> tuple[StageSequence, PageId]:
    source = tmp_path / "in" / "a.tif"
    source.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.full((400, 300), 220, dtype=np.uint8)
    pixels[100:300, 80:220] = 40
    write_image(source, pixels, Dpi.uniform(SOURCE_DPI))
    page = PageId(image_id=ImageId(file_path=source), sub_page=SubPage.SINGLE_PAGE)
    stages = StageSequence()
    stages.page_split_filter.set_layout_type(page, LayoutType.SINGLE_PAGE_UNCUT)
    stages.deskew_filter.set_manual_angle(page, 0.0)
    stages.select_content_filter.set_manual_content(
        page, ContentBox(x=80, y=100, width=140, height=200)
    )
    stages.output_filter.set_color_mode(page, mode)
    return stages, page


def _render_in_process(
    tmp_path: Path, mode: ColorMode, compressions: TiffCompressions
) -> int:
    stages, page = _make_page(tmp_path, mode)
    config = _PhaseConfig(
        token=uuid.uuid4().hex,
        stage_params=collect_effective_stage_params(stages, [page]),
        defaults=collect_stage_defaults(stages),
        default_dpi=SOURCE_DPI,
        tiff_compressions=compressions,
    )
    path, _info, _files = _render_page(
        _PageJob(page, config, tmp_path / "out" / "a.tif")
    )
    with Image.open(path) as written:
        return written.tag_v2[COMPRESSION_TAG]


class TestWorkerWrites:
    @pytest.mark.parametrize(
        "method",
        [
            TiffCompression.NONE,
            TiffCompression.LZW,
            TiffCompression.DEFLATE,
            TiffCompression.CCITT_FAX4,
        ],
    )
    def test_black_and_white_output_uses_the_black_and_white_setting(
        self, tmp_path: Path, method: TiffCompression
    ) -> None:
        tag = _render_in_process(
            tmp_path,
            ColorMode.BLACK_AND_WHITE,
            TiffCompressions(bw=method, color=TiffCompression.JPEG),
        )

        assert tag == method.value

    @pytest.mark.parametrize(
        "method",
        [
            TiffCompression.NONE,
            TiffCompression.LZW,
            TiffCompression.DEFLATE,
            TiffCompression.JPEG,
        ],
    )
    def test_grey_output_uses_the_colour_setting(
        self, tmp_path: Path, method: TiffCompression
    ) -> None:
        tag = _render_in_process(
            tmp_path,
            ColorMode.COLOR_GRAYSCALE,
            TiffCompressions(bw=TiffCompression.NONE, color=method),
        )

        assert tag == method.value


@pytest.mark.slow
class TestRunBatchCompression:
    def test_choice_given_to_the_batch_reaches_the_worker_processes(
        self, tmp_path: Path
    ) -> None:
        stages, page = _make_page(tmp_path, ColorMode.COLOR_GRAYSCALE)

        result = run_batch(
            stages,
            [page],
            tmp_path / "out",
            max_workers=1,
            tiff_compressions=TiffCompressions(color=TiffCompression.DEFLATE),
        )

        assert result.failed == {}
        with Image.open(result.written[page]) as written:
            assert written.tag_v2[COMPRESSION_TAG] == TiffCompression.DEFLATE.value

    def test_without_a_choice_the_application_settings_are_used(
        self, tmp_path: Path
    ) -> None:
        ApplicationSettings.set_config_path(tmp_path / "settings.json")
        try:
            ApplicationSettings.get_instance().tiff_color_compression = (
                TiffCompression.NONE
            )
            stages, page = _make_page(tmp_path, ColorMode.COLOR_GRAYSCALE)

            result = run_batch(stages, [page], tmp_path / "out", max_workers=1)
        finally:
            ApplicationSettings.reset_instance()

        with Image.open(result.written[page]) as written:
            assert written.tag_v2[COMPRESSION_TAG] == TiffCompression.NONE.value
