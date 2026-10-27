"""Under a decimal-comma locale every number on screen uses the comma.

These run in every pass, whatever --qt-locale says, so a label that formats
numbers without the locale fails here even when CI runs in English.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6 import QtCore, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.image_views.deskew import format_angle
from scantailor.app.ui.options_widgets.deskew import OptionsWidget as DeskewOptions
from scantailor.app.ui.options_widgets.output import OptionsWidget as OutputOptions
from scantailor.app.ui.options_widgets.output import dewarping_status_text
from scantailor.app.ui.units_provider import UnitsProvider
from scantailor.app.ui.utils import coordinate_separator, format_number
from scantailor.app.ui.widgets.status_bar_panel import StatusBarPanel
from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.core.settings import Units
from scantailor.filters.deskew.dependencies import Dependencies
from scantailor.filters.deskew.filter import Filter as DeskewFilter
from scantailor.filters.output.dewarping_options import DewarpingMode, DewarpingOptions
from scantailor.filters.output.filter import Filter as OutputFilter
from scantailor.filters.output.params import Params
from tests.ui.locale_numbers import shown_numbers, without_numbers

PAGE = PageId(
    image_id=ImageId(file_path=Path("/page.tif")), sub_page=SubPage.SINGLE_PAGE
)


@pytest.fixture
def swedish() -> Iterator[QtCore.QLocale]:
    """Make Swedish Qt's default locale for one test, then restore the previous."""
    previous = QtCore.QLocale()
    locale = QtCore.QLocale(
        QtCore.QLocale.Language.Swedish, QtCore.QLocale.Country.Sweden
    )
    QtCore.QLocale.setDefault(locale)
    yield locale
    QtCore.QLocale.setDefault(previous)


@pytest.fixture
def english() -> Iterator[QtCore.QLocale]:
    """Make US English Qt's default locale for one test, then restore the previous."""
    previous = QtCore.QLocale()
    locale = QtCore.QLocale(
        QtCore.QLocale.Language.English, QtCore.QLocale.Country.UnitedStates
    )
    QtCore.QLocale.setDefault(locale)
    yield locale
    QtCore.QLocale.setDefault(previous)


class TestSwedishShowsADecimalComma:
    def test_a_label_number_uses_the_comma_and_no_group_separator(
        self, swedish: QtCore.QLocale
    ) -> None:
        assert format_number(2.7) == "2,7"
        assert format_number(1234.5) == "1234,5"
        assert format_number(2.0, 1) == "2,0"
        assert format_number(51) == "51"

    def test_the_status_bar_size_uses_the_comma(
        self, qtbot: QtBot, swedish: QtCore.QLocale
    ) -> None:
        panel = StatusBarPanel(UnitsProvider(Units.CENTIMETERS))
        qtbot.addWidget(panel)

        panel.set_page_size(QtCore.QSizeF(600, 800), 300)

        assert panel.phys_size_label.text() == "5,1 x 6,8 cm"

    def test_the_cursor_position_separates_its_numbers_with_a_semicolon(
        self, qtbot: QtBot, swedish: QtCore.QLocale
    ) -> None:
        panel = StatusBarPanel(UnitsProvider(Units.CENTIMETERS))
        qtbot.addWidget(panel)
        panel.set_page_size(QtCore.QSizeF(600, 800), 300)

        panel.set_cursor_position(QtCore.QPointF(300.4, 400.2))

        assert panel.mouse_pos_label.text() == "2,6; 3,4"

    def test_the_dewarping_status_angle_uses_the_comma(
        self, swedish: QtCore.QLocale
    ) -> None:
        options = DewarpingOptions(mode=DewarpingMode.MANUAL, post_deskew_angle=-1.25)

        assert dewarping_status_text(options) == "Manual (deskew: 1,25°)"

    def test_the_depth_perception_label_uses_the_comma(
        self, qtbot: QtBot, swedish: QtCore.QLocale
    ) -> None:
        filter_ = OutputFilter()
        filter_.set_params(PAGE, Params(depth_perception=2.5))
        widget = OutputOptions(filter_)
        qtbot.addWidget(widget)
        widget.set_current_page(PAGE)
        label = widget.findChild(QtWidgets.QLabel, "depthPerceptionLabel")
        assert label is not None

        assert label.text() == "2,5"

    def test_the_deskew_spin_box_and_view_label_agree_on_the_comma(
        self, qtbot: QtBot, swedish: QtCore.QLocale
    ) -> None:
        filter_ = DeskewFilter()
        filter_.settings.store_detection(PAGE, Dependencies(), rotation=-1.5)
        widget = DeskewOptions(filter_)
        qtbot.addWidget(widget)
        widget.set_current_page(PAGE)
        spin_box = widget.findChild(QtWidgets.QDoubleSpinBox, "angleSpinBox")
        assert spin_box is not None

        assert spin_box.text() == "1,50°"
        assert format_angle(1.5) == "1,50°"


class TestEnglishShowsADecimalPoint:
    def test_a_label_number_uses_the_point_and_no_group_separator(
        self, english: QtCore.QLocale
    ) -> None:
        assert format_number(2.7) == "2.7"
        assert format_number(1234.5) == "1234.5"

    def test_the_cursor_position_separates_its_numbers_with_a_comma(
        self, english: QtCore.QLocale
    ) -> None:
        assert coordinate_separator() == ", "


class TestReadingNumbersBack:
    @pytest.mark.parametrize("locale_fixture", ["swedish", "english"])
    def test_shown_numbers_reads_what_format_number_wrote(
        self, request: pytest.FixtureRequest, locale_fixture: str
    ) -> None:
        request.getfixturevalue(locale_fixture)
        text = f"{format_number(-1.25)} x {format_number(6.8)} cm"

        assert shown_numbers(text) == pytest.approx([-1.25, 6.8])
        assert without_numbers(text) == " x  cm"
