"""Whole-page Output results against digests taken before the stage was parallelised."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from scantailor.core.models import Dpi, ImageId, PageId, SubPage
from scantailor.core.pipeline import PipelineOptions, process_page
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.output.binarization import BinarizationMethod
from scantailor.filters.output.color_mode import ColorMode
from tests.dewarping.synthetic import PAGE_DPI, straight_text_page

_NOISE = 5
_OUTPUT_DPI = 200

# (colour mode, binarisation) -> first 16 hex digits of the SHA-256 of the output.
_REFERENCES = {
    (ColorMode.BLACK_AND_WHITE, BinarizationMethod.OTSU): "b6178d3fc976eea0",
    (ColorMode.BLACK_AND_WHITE, BinarizationMethod.WOLF): "6eaab6efb187439b",
    (ColorMode.COLOR_GRAYSCALE, BinarizationMethod.OTSU): "b3a6e63bf13b4c6c",
}


def _noisy_text_page() -> NDArray[np.uint8]:
    page = straight_text_page()
    rng = np.random.default_rng(3)
    noise = rng.integers(-_NOISE, _NOISE + 1, page.shape)
    return np.clip(page.astype(np.int16) + noise, 0, 255).astype(np.uint8)


@pytest.mark.parametrize(("mode", "method"), sorted(_REFERENCES, key=str))
def test_output_pixels_match_the_stored_reference(
    mode: ColorMode, method: BinarizationMethod
) -> None:
    """Speed-ups to smoothing, binarisation and despeckling must not move a pixel."""
    page_id = PageId(
        image_id=ImageId(file_path=Path("reference.tiff")),
        sub_page=SubPage.SINGLE_PAGE,
    )
    stages = StageSequence()
    params = stages.output_filter.get_params(page_id)
    stages.output_filter.set_params(
        page_id,
        params.model_copy(
            update={
                "output_dpi": Dpi.uniform(_OUTPUT_DPI),
                "color_mode": mode,
                "binarization": params.binarization.model_copy(
                    update={"method": method}
                ),
            }
        ),
    )

    result = process_page(
        _noisy_text_page(),
        page_id,
        stages,
        PipelineOptions(end_stage=FilterStage.OUTPUT, dpi=float(PAGE_DPI)),
    )

    digest = hashlib.sha256(np.ascontiguousarray(result.image).tobytes())
    assert digest.hexdigest()[:16] == _REFERENCES[(mode, method)]
