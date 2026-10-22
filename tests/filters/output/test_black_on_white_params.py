"""The black-on-white choice of a page and when detection may change it."""

from pathlib import Path

import pytest

from scantailor.core.models import Dpi, ImageId, PageId, SubPage
from scantailor.filters.output.params import Params
from scantailor.filters.output.settings import Settings

PAGE = PageId(image_id=ImageId(file_path=Path("a.png")), sub_page=SubPage.SINGLE_PAGE)


class TestStoredParams:
    def test_new_params_are_black_on_white_and_not_chosen_by_hand(self):
        params = Params()

        assert params.black_on_white is True
        assert params.black_on_white_set_manually is False

    def test_stored_white_on_black_without_the_flag_loads_as_chosen_by_hand(self):
        stored = Params().with_black_on_white(False).model_dump(mode="json")
        del stored["black_on_white_set_manually"]

        params = Params.model_validate(stored)

        assert params.black_on_white is False
        assert params.black_on_white_set_manually is True

    def test_stored_black_on_white_without_the_flag_stays_open_to_detection(self):
        stored = Params().model_dump(mode="json")
        del stored["black_on_white_set_manually"]

        assert Params.model_validate(stored).black_on_white_set_manually is False

    @pytest.mark.parametrize("manual", [False, True])
    def test_flag_survives_a_json_round_trip(self, manual: bool):
        params = Params().model_copy(
            update={"black_on_white": False, "black_on_white_set_manually": manual}
        )

        reloaded = Params.model_validate_json(params.model_dump_json())

        assert reloaded == params

    def test_choosing_by_hand_sets_the_flag(self):
        params = Params().with_manual_black_on_white(True)

        assert params.black_on_white is True
        assert params.black_on_white_set_manually is True


class TestSetDetectedBlackOnWhite:
    def test_page_without_params_gets_the_defaults_with_the_measured_polarity(self):
        settings = Settings()
        settings.set_default_dpi(Dpi.uniform(300))

        changed = settings.set_detected_black_on_white(PAGE, False)

        assert changed
        stored = settings.get_params(PAGE)
        assert stored.black_on_white is False
        assert stored.output_dpi == Dpi.uniform(300)
        assert not stored.black_on_white_set_manually

    def test_matching_polarity_stores_nothing(self):
        settings = Settings()

        assert not settings.set_detected_black_on_white(PAGE, True)
        assert not settings.is_params_set(PAGE)

    def test_polarity_chosen_by_hand_is_kept(self):
        settings = Settings()
        settings.set_params(PAGE, Params().with_manual_black_on_white(True))

        assert not settings.set_detected_black_on_white(PAGE, False)
        assert settings.get_params(PAGE).black_on_white is True

    def test_detected_polarity_follows_the_page_back(self):
        settings = Settings()
        settings.set_detected_black_on_white(PAGE, False)

        assert settings.set_detected_black_on_white(PAGE, True)
        assert settings.get_params(PAGE).black_on_white is True
