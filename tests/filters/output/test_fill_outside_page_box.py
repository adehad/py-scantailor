"""Fill outside page box: the option, its render flag, persistence and effect."""

import numpy as np
import pytest

from scantailor.filters.output.binarization import BinarizationMethod
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.color_options import ColorCommonOptions
from scantailor.filters.output.generator import generate_output
from scantailor.filters.output.params import Params
from scantailor.filters.output.render_params import RenderParams

HEIGHT, WIDTH = 120, 160
OUTLINE = [(40.0, 20.0), (120.0, 20.0), (120.0, 100.0), (40.0, 100.0)]
CONTENT = [(50.0, 30.0), (110.0, 30.0), (110.0, 90.0), (50.0, 90.0)]
CONTENT_BEYOND_THE_PAGE = [(5.0, 10.0), (155.0, 10.0), (155.0, 110.0), (5.0, 110.0)]
PAPER = 200
BEYOND_THE_PAGE = 240
OFFCUT = 150
INK = 30


def _scan() -> np.ndarray:
    """A page of paper with text, lying in a larger and lighter area with an offcut."""
    scan = np.full((HEIGHT, WIDTH), BEYOND_THE_PAGE, dtype=np.uint8)
    scan[20:100, 40:120] = PAPER
    scan[40:44, 55:105] = INK
    scan[60:64, 55:105] = INK
    scan[:, :30] = OFFCUT
    return scan


def _generate(
    common: ColorCommonOptions, content: list[tuple[float, float]] = CONTENT
) -> np.ndarray:
    params = Params(color_mode=ColorMode.COLOR_GRAYSCALE, color_common=common)
    return generate_output(
        _scan(), params, content_polygon=content, page_outline=OUTLINE
    ).image


class TestOption:
    def test_off_by_default(self):
        assert ColorCommonOptions().fill_outside_page_box is False

    def test_survives_a_json_round_trip(self):
        params = Params(
            color_common=ColorCommonOptions(
                fill_outside_page_box=True, fill_offcut=False
            )
        )

        restored = Params.model_validate_json(params.model_dump_json())

        assert restored.color_common.fill_outside_page_box is True
        assert restored.color_common.fill_offcut is False

    def test_params_saved_without_the_field_load_with_it_off(self):
        stored = Params().model_dump(mode="json")
        del stored["color_common"]["fill_outside_page_box"]

        assert Params.model_validate(stored).color_common.fill_outside_page_box is False

    @pytest.mark.parametrize("method", list(BinarizationMethod))
    def test_every_binarization_method_survives_a_json_round_trip(
        self, method: BinarizationMethod
    ):
        params = Params()
        params = params.with_binarization_method(method)

        restored = Params.model_validate_json(params.model_dump_json())

        assert restored.binarization.method is method

    @pytest.mark.parametrize(
        ("text", "method"),
        [
            ("fox", BinarizationMethod.FOX),
            ("window", BinarizationMethod.WINDOW),
            ("grad", BinarizationMethod.GRAD),
        ],
    )
    def test_new_methods_are_named_as_upstream_names_them(
        self, text: str, method: BinarizationMethod
    ):
        assert BinarizationMethod.from_string(text) is method
        assert str(method) == text


class TestRenderFlag:
    def test_flag_follows_the_option(self):
        common = ColorCommonOptions(fill_outside_page_box=True)

        render = RenderParams.from_params(Params(color_common=common))

        assert render.fill_outside_page_box is True

    def test_it_replaces_offcut_filling_when_both_are_set(self):
        common = ColorCommonOptions(fill_offcut=True, fill_outside_page_box=True)

        render = RenderParams.from_params(Params(color_common=common))

        assert render.fill_offcut is False

    def test_offcut_filling_stays_on_without_it(self):
        render = RenderParams.from_params(Params())

        assert render.fill_offcut is True
        assert render.fill_outside_page_box is False


class TestGeneratorBehaviour:
    def test_offcut_fill_paints_outside_the_outline_with_the_page_background(self):
        result = _generate(
            ColorCommonOptions(fill_offcut=True), CONTENT_BEYOND_THE_PAGE
        )

        assert result[60, 10] == PAPER

    def test_the_page_outline_does_not_clip_with_the_option_on(self):
        result = _generate(
            ColorCommonOptions(fill_offcut=False, fill_outside_page_box=True),
            CONTENT_BEYOND_THE_PAGE,
        )

        assert result[60, 10] == OFFCUT

    def test_background_is_taken_from_the_whole_rectangle_with_the_option_on(self):
        with_option = _generate(
            ColorCommonOptions(fill_offcut=False, fill_outside_page_box=True)
        )
        without = _generate(ColorCommonOptions(fill_offcut=True))

        # A margin inside the outline: the surroundings are the lighter majority.
        assert without[25, 45] == PAPER
        assert with_option[25, 45] == BEYOND_THE_PAGE

    def test_content_is_untouched_either_way(self):
        with_option = _generate(
            ColorCommonOptions(fill_offcut=False, fill_outside_page_box=True)
        )
        without = _generate(ColorCommonOptions(fill_offcut=True))

        content = (slice(30, 90), slice(50, 110))
        assert np.array_equal(with_option[content], without[content])

    def test_option_wins_over_offcut_fill_when_both_are_set(self):
        both = _generate(
            ColorCommonOptions(fill_offcut=True, fill_outside_page_box=True)
        )
        only_option = _generate(
            ColorCommonOptions(fill_offcut=False, fill_outside_page_box=True)
        )

        assert np.array_equal(both, only_option)

    def test_margins_are_still_filled_when_fill_margins_is_on(self):
        result = _generate(
            ColorCommonOptions(
                fill_offcut=False, fill_outside_page_box=True, fill_margins=True
            )
        )

        # Between the content edge and the outline the margin gets the fill colour.
        assert result[25, 80] != INK
        assert result[25, 80] == BEYOND_THE_PAGE
