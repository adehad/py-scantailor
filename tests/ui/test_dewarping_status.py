"""The Output panel's dewarping status line with the post-deskew angle."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.options_widgets.output import (
    OptionsWidget,
    dewarping_status_text,
)
from scantailor.app.ui.stage_panels import StagePanels
from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.core.pipeline import OutputExtras
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.output.dewarping_options import DewarpingMode, DewarpingOptions
from scantailor.filters.output.filter import Filter
from scantailor.filters.output.params import Params
from tests.ui.locale_numbers import shown_numbers, without_numbers

PAGE = PageId(
    image_id=ImageId(file_path=Path("page.tif")), sub_page=SubPage.SINGLE_PAGE
)


def _params(
    mode: DewarpingMode, angle: float = 0.0, post_deskew: bool = True
) -> Params:
    return Params(
        dewarping=DewarpingOptions(
            mode=mode, post_deskew=post_deskew, post_deskew_angle=angle
        )
    )


def _words_and_numbers(text: str | None) -> tuple[str, list[float]]:
    assert text is not None
    return without_numbers(text), shown_numbers(text)


def _status_label(options: QtWidgets.QWidget) -> QtWidgets.QLabel:
    label = options.findChild(QtWidgets.QLabel, "dewarpingStatusLabel")
    assert label is not None
    return label


@pytest.fixture
def make_options(qtbot: QtBot) -> Callable[[Params], tuple[OptionsWidget, Filter]]:
    """Return a function that shows one page's params in a new options widget."""

    def make(params: Params) -> tuple[OptionsWidget, Filter]:
        filter_ = Filter()
        filter_.set_params(PAGE, params)
        widget = OptionsWidget(filter_)
        qtbot.addWidget(widget)
        widget.set_current_page(PAGE)
        return widget, filter_

    return make


class TestStatusText:
    def test_manual_mode_shows_the_skew_found_which_is_the_negated_rotation(
        self,
    ) -> None:
        options = DewarpingOptions(mode=DewarpingMode.MANUAL, post_deskew_angle=1.234)

        assert _words_and_numbers(dewarping_status_text(options)) == (
            "Manual (deskew: °)",
            pytest.approx([-1.23]),
        )

    def test_marginal_mode_shows_the_angle_too(self) -> None:
        options = DewarpingOptions(mode=DewarpingMode.MARGINAL, post_deskew_angle=-0.5)

        assert _words_and_numbers(dewarping_status_text(options)) == (
            "Marginal (deskew: °)",
            pytest.approx([0.5]),
        )

    def test_a_tiny_angle_rounds_to_zero_without_a_minus_sign(self) -> None:
        options = DewarpingOptions(mode=DewarpingMode.MANUAL, post_deskew_angle=0.001)

        text = dewarping_status_text(options)

        assert _words_and_numbers(text) == ("Manual (deskew: °)", [0.0])
        assert QtCore.QLocale().negativeSign() not in str(text)

    def test_post_deskew_turned_off_says_so_instead_of_an_angle(self) -> None:
        options = DewarpingOptions(
            mode=DewarpingMode.MANUAL, post_deskew=False, post_deskew_angle=2.0
        )

        assert dewarping_status_text(options) == "Manual (deskew disabled)"

    @pytest.mark.parametrize("mode", [DewarpingMode.OFF, DewarpingMode.AUTO])
    def test_off_and_auto_have_no_status_line(self, mode: DewarpingMode) -> None:
        assert dewarping_status_text(DewarpingOptions(mode=mode)) is None


class TestOptionsPanel:
    def test_a_manual_page_shows_its_status_line(
        self, make_options: Callable[[Params], tuple[OptionsWidget, Filter]]
    ) -> None:
        widget, _ = make_options(_params(DewarpingMode.MANUAL, angle=-2.5))

        label = _status_label(widget)
        assert label.isVisibleTo(widget)
        assert _words_and_numbers(label.text()) == (
            "Manual (deskew: °)",
            pytest.approx([2.5]),
        )

    def test_an_auto_page_hides_the_status_line(
        self, make_options: Callable[[Params], tuple[OptionsWidget, Filter]]
    ) -> None:
        widget, _ = make_options(_params(DewarpingMode.AUTO, angle=-2.5))

        assert not _status_label(widget).isVisibleTo(widget)

    def test_unticking_post_deskew_updates_the_status_line(
        self, make_options: Callable[[Params], tuple[OptionsWidget, Filter]]
    ) -> None:
        widget, _ = make_options(_params(DewarpingMode.MARGINAL, angle=1.0))
        check = widget.findChild(QtWidgets.QCheckBox, "postDeskewCB")
        assert check is not None

        check.setChecked(False)

        assert _status_label(widget).text() == "Marginal (deskew disabled)"

    def test_choosing_manual_shows_the_status_line(
        self, make_options: Callable[[Params], tuple[OptionsWidget, Filter]]
    ) -> None:
        widget, _ = make_options(_params(DewarpingMode.OFF))
        button = widget.findChild(QtWidgets.QPushButton, "dewarpManualBtn")
        assert button is not None

        button.click()

        label = _status_label(widget)
        assert label.isVisibleTo(widget)
        assert _words_and_numbers(label.text()) == ("Manual (deskew: °)", [0.0])


def test_a_render_that_measures_the_angle_updates_the_status_line(
    make_stage_panels: Callable[[FilterStage, StageSequence], StagePanels],
) -> None:
    sequence = StageSequence()
    sequence.output_filter.set_params(PAGE, _params(DewarpingMode.MANUAL))
    panels = make_stage_panels(FilterStage.OUTPUT, sequence)
    panels.set_page(PAGE)

    sequence.output_filter.settings.set_post_deskew_angle(PAGE, -1.75)
    panels.set_output_extras(
        OutputExtras(np.full((100, 80), 200, dtype=np.uint8), None, 300)
    )

    assert _words_and_numbers(_status_label(panels.options).text()) == (
        "Manual (deskew: °)",
        pytest.approx([1.75]),
    )
