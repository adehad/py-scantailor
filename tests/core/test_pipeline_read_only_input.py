"""Every stage and colour mode runs on a read-only page and leaves it unchanged."""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from scantailor.core.models import Dpi, ImageId, PageId, SubPage
from scantailor.core.pipeline import PipelineOptions, process_page
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.despeckle import DespeckleLevel
from scantailor.filters.output.splitting_options import SplittingOptions

_PAGE_ID = PageId(
    image_id=ImageId(file_path=Path("read_only.tiff")), sub_page=SubPage.SINGLE_PAGE
)


def _colour_page() -> NDArray[np.uint8]:
    """A light, unevenly lit page with text-like bars, specks and a picture."""
    rng = np.random.default_rng(7)
    height, width = 360, 280
    shade = np.linspace(200, 240, width, dtype=np.float32)[None, :, None]
    page = np.broadcast_to(shade, (height, width, 3)).astype(np.uint8)
    for row in range(40, 200, 18):
        page[row : row + 8, 30:250] = (20, 25, 30)
    page[230:320, 40:200] = rng.integers(0, 256, (90, 160, 3), dtype=np.uint8)
    specks = rng.integers(0, height * width, 60)
    page.reshape(-1, 3)[specks] = 0
    return page


def _read_only(page: NDArray[np.uint8]) -> NDArray[np.uint8]:
    page = page.copy()
    page.flags.writeable = False
    return page


@pytest.mark.parametrize("gray", [False, True], ids=["colour", "gray"])
@pytest.mark.parametrize("color_mode", list(ColorMode))
def test_every_stage_runs_on_a_read_only_page_and_leaves_it_unchanged(
    color_mode: ColorMode, gray: bool
) -> None:
    """Detection, every Output path and split layers never write the input."""
    page = _colour_page()
    if gray:
        page = np.ascontiguousarray(page[..., 1])
    source = _read_only(page)
    sequence = StageSequence()
    output = sequence.output_filter
    params = output.get_params(_PAGE_ID)
    output.set_params(
        _PAGE_ID,
        params.model_copy(
            update={
                "output_dpi": Dpi.uniform(150),
                "color_mode": color_mode,
                "despeckle_level": DespeckleLevel.AGGRESSIVE,
                "color_common": params.color_common.model_copy(
                    update={"normalize_illumination": True, "wiener_coef": 0.3}
                ),
                "splitting": SplittingOptions(split_output=True),
            }
        ),
    )

    for end_stage in FilterStage:
        result = process_page(
            source,
            _PAGE_ID,
            sequence,
            PipelineOptions(dpi=150.0, end_stage=end_stage),
        )
        assert result.image is not None

    np.testing.assert_array_equal(source, page)
