"""The generator despeckles at the stored strength, whether a step or in between."""

import numpy as np
import pytest

from scantailor.filters.output.binarization import BinarizationOptions
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.despeckle import DespeckleLevel
from scantailor.filters.output.generator import generate_output
from scantailor.filters.output.params import Params
from scantailor.filters.output.picture_shape import PictureShape, PictureShapeOptions

_PLAIN = BinarizationOptions(
    normalize_illumination=False,
    savitzky_golay_smoothing=False,
    morphological_smoothing=False,
)


def _page() -> np.ndarray:
    page = np.full((320, 320), 240, dtype=np.uint8)
    page[120:180, 60:260] = 20
    page[220:280, 60:260] = 20
    page[20:22, 20:22] = 20
    return page


def _params(mode: ColorMode) -> Params:
    return Params(
        color_mode=mode,
        binarization=_PLAIN,
        picture_shape=PictureShapeOptions(shape=PictureShape.OFF),
    )


@pytest.fixture
def strengths_seen(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    seen: list[float] = []

    def spy(
        binary: np.ndarray, dpi: int, strength: float, dbg: object = None
    ) -> np.ndarray:
        seen.append(strength)
        return binary.copy()

    monkeypatch.setattr("scantailor.filters.output.generator.despeckle_image", spy)
    return seen


@pytest.mark.parametrize("mode", [ColorMode.BLACK_AND_WHITE, ColorMode.MIXED])
def test_an_in_between_strength_reaches_the_despeckler_without_a_rewrite(
    mode: ColorMode, strengths_seen: list[float]
) -> None:
    params = _params(mode).with_despeckle_factor(1.7)

    generate_output(_page(), params)

    assert strengths_seen == [1.7]


def test_a_named_level_despeckles_at_its_preset(strengths_seen: list[float]) -> None:
    params = _params(ColorMode.BLACK_AND_WHITE).with_despeckle_level(
        DespeckleLevel.AGGRESSIVE
    )

    generate_output(_page(), params)

    assert strengths_seen == [3.0]


def test_a_zero_strength_skips_despeckling(strengths_seen: list[float]) -> None:
    params = _params(ColorMode.BLACK_AND_WHITE).with_despeckle_factor(0.0)

    generate_output(_page(), params)

    assert strengths_seen == []
