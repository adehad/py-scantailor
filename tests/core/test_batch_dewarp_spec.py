"""A batch records how a page was dewarped, so that its text can follow."""

import numpy as np
import pytest

from scantailor.core.batch import _dewarp_spec
from scantailor.core.models import ImageId, PageId
from scantailor.core.pipeline import PipelineResult
from scantailor.filters.output.dewarping_options import DewarpingMode, DewarpingOptions
from scantailor.filters.output.params import DistortionModelSpec, Params

TOP = ((0.0, 10.0), (100.0, 8.0))
BOTTOM = ((0.0, 90.0), (100.0, 92.0))


def _result(model: DistortionModelSpec | None) -> PipelineResult:
    return PipelineResult(
        image=np.zeros((4, 4), dtype=np.uint8),
        page_id=PageId(image_id=ImageId(file_path="/a.png")),
        distortion_model=model,
    )


def _params(mode: DewarpingMode, **extra: object) -> Params:
    return Params(dewarping=DewarpingOptions(mode=mode, **extra), depth_perception=2.5)


def test_a_model_the_pipeline_found_is_recorded_with_the_depth() -> None:
    found = DistortionModelSpec(top=TOP, bottom=BOTTOM)

    spec = _dewarp_spec(_result(found), _params(DewarpingMode.AUTO))

    assert spec is not None
    assert (spec.top, spec.bottom, spec.depth_perception) == (TOP, BOTTOM, 2.5)
    assert spec.post_deskew_angle == 0.0


def test_a_manual_model_comes_from_the_parameters() -> None:
    params = _params(DewarpingMode.MANUAL).with_distortion_model(
        DistortionModelSpec(top=TOP, bottom=BOTTOM)
    )

    spec = _dewarp_spec(_result(None), params)

    assert spec is not None
    assert spec.top == TOP


def test_the_post_deskew_angle_is_recorded_for_the_modes_that_apply_it() -> None:
    params = _params(
        DewarpingMode.MANUAL, post_deskew=True, post_deskew_angle=1.5
    ).with_distortion_model(DistortionModelSpec(top=TOP, bottom=BOTTOM))

    spec = _dewarp_spec(_result(None), params)

    assert spec is not None
    assert spec.post_deskew_angle == pytest.approx(1.5)


def test_without_any_model_there_is_no_spec() -> None:
    assert _dewarp_spec(_result(None), _params(DewarpingMode.AUTO)) is None
