"""A batch stores the polarity its workers measured at Output."""

from pathlib import Path

import numpy as np
import pytest

from scantailor.core.batch import run_batch
from scantailor.core.image_io import load_image, write_image
from scantailor.core.models import Dpi, ImageId, PageId, SubPage
from scantailor.core.settings import BlackOnWhiteDetection
from scantailor.core.stage_sequence import StageSequence
from tests.core.dark_pages import dark_page


@pytest.fixture
def dark_source(tmp_path: Path) -> PageId:
    path = tmp_path / "in" / "dark.tif"
    path.parent.mkdir()
    write_image(path, dark_page(400, 500), Dpi.uniform(300))
    return PageId(image_id=ImageId(file_path=path), sub_page=SubPage.SINGLE_PAGE)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("detection", "stored"),
    [(BlackOnWhiteDetection(), False), (BlackOnWhiteDetection(at_output=False), True)],
    ids=["measured at output", "not measured at output"],
)
def test_batch_stores_the_polarity_output_worked_in(
    tmp_path: Path,
    dark_source: PageId,
    detection: BlackOnWhiteDetection,
    stored: bool,
):
    stages = StageSequence()

    result = run_batch(
        stages,
        [dark_source],
        tmp_path / "out",
        max_workers=1,
        black_on_white_detection=detection,
    )

    assert result.failed == {}
    assert stages.output_filter.get_params(dark_source).black_on_white is stored
    written, _ = load_image(ImageId(file_path=result.written[dark_source]))
    assert (np.mean(written) < 128) is not stored
