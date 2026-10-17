"""The status bar panel formats pages, sizes and positions as C++ does."""

from pathlib import Path

import pytest
from PySide6 import QtCore
from pytestqt.qtbot import QtBot

from scantailor.app.ui.units_provider import UnitsProvider
from scantailor.app.ui.widgets.status_bar_panel import StatusBarPanel
from scantailor.core.models import Dpi, ImageId, PageId, SubPage
from scantailor.core.settings import Units

LONG_NAME = "abcdefghijklmnopqrstuvwxyz.png"


@pytest.fixture
def provider() -> UnitsProvider:
    return UnitsProvider(Units.MILLIMETERS)


@pytest.fixture
def panel(qtbot: QtBot, provider: UnitsProvider) -> StatusBarPanel:
    widget = StatusBarPanel(provider)
    qtbot.addWidget(widget)
    widget.show()
    return widget


def page(name: str, sub_page: SubPage = SubPage.SINGLE_PAGE) -> PageId:
    return PageId(image_id=ImageId(file_path=Path("scans") / name), sub_page=sub_page)


class TestPage:
    def test_page_number_and_count_are_shown(self, panel: StatusBarPanel) -> None:
        panel.update_page(2, 14, page("scan.png"))

        assert panel.page_no_label.text() == "p. 2 / 14"

    def test_a_short_file_name_is_shown_without_its_suffix(
        self, panel: StatusBarPanel
    ) -> None:
        panel.update_page(1, 1, page("scan.v2.png"))

        assert panel.page_info_label.text() == "scan.v2"

    def test_a_long_file_name_keeps_its_last_thirteen_characters(
        self, panel: StatusBarPanel
    ) -> None:
        panel.update_page(1, 1, page(LONG_NAME))

        assert panel.page_info_label.text() == "...nopqrstuvwxyz"

    def test_the_halves_of_a_spread_are_marked(self, panel: StatusBarPanel) -> None:
        panel.update_page(1, 2, page("scan.png", SubPage.LEFT_PAGE))
        assert panel.page_info_label.text() == "scan [L]"

        panel.update_page(2, 2, page("scan.png", SubPage.RIGHT_PAGE))
        assert panel.page_info_label.text() == "scan [R]"

    def test_a_long_name_of_a_half_keeps_eleven_characters_before_the_marker(
        self, panel: StatusBarPanel
    ) -> None:
        panel.update_page(1, 2, page(LONG_NAME, SubPage.LEFT_PAGE))

        assert panel.page_info_label.text() == "pqrstuvwxyz [L]"

    def test_clear_hides_everything(self, panel: StatusBarPanel) -> None:
        panel.update_page(1, 1, page("scan.png"))
        panel.set_page_size(QtCore.QSizeF(600, 800), 300)
        panel.set_cursor_position(QtCore.QPointF(10, 10))

        panel.clear()

        for label in (
            panel.page_no_label,
            panel.page_info_label,
            panel.phys_size_label,
            panel.mouse_pos_label,
        ):
            assert label.text() == ""
            assert not label.isVisible()


class TestSize:
    @pytest.mark.parametrize(
        ("units", "text"),
        [
            (Units.PIXELS, "600 x 800 px"),
            (Units.MILLIMETERS, "51 x 68 mm"),
            (Units.CENTIMETERS, "5.1 x 6.8 cm"),
            (Units.INCHES, "2 x 2.7 in"),
        ],
    )
    def test_size_is_rounded_for_the_unit(
        self, panel: StatusBarPanel, provider: UnitsProvider, units: Units, text: str
    ) -> None:
        provider.set_units(units)
        panel.set_page_size(QtCore.QSizeF(600, 800), 300)

        assert panel.phys_size_label.text() == text

    def test_changing_the_unit_updates_a_shown_size(
        self, panel: StatusBarPanel, provider: UnitsProvider
    ) -> None:
        panel.set_page_size(QtCore.QSizeF(600, 800), 300)

        provider.set_units(Units.INCHES)

        assert panel.phys_size_label.text() == "2 x 2.7 in"

    def test_an_unknown_resolution_hides_the_size_and_the_position(
        self, panel: StatusBarPanel
    ) -> None:
        panel.set_page_size(QtCore.QSizeF(600, 800), 0)
        panel.set_cursor_position(QtCore.QPointF(5, 5))

        assert panel.phys_size_label.text() == ""
        assert panel.mouse_pos_label.text() == ""

    def test_no_page_hides_the_size(self, panel: StatusBarPanel) -> None:
        panel.set_page_size(QtCore.QSizeF(600, 800), 300)

        panel.set_page_size(None, 0)

        assert panel.phys_size_label.text() == ""


class TestCursor:
    @pytest.mark.parametrize(
        ("units", "text"),
        [
            (Units.PIXELS, "301, 401"),
            (Units.MILLIMETERS, "26, 34"),
            (Units.INCHES, "1.1, 1.4"),
            (Units.CENTIMETERS, "2.6, 3.4"),
        ],
    )
    def test_position_is_rounded_up_for_the_unit(
        self, panel: StatusBarPanel, provider: UnitsProvider, units: Units, text: str
    ) -> None:
        provider.set_units(units)
        panel.set_page_size(QtCore.QSizeF(600, 800), 300)

        panel.set_cursor_position(QtCore.QPointF(300.4, 400.2))

        assert panel.mouse_pos_label.text() == text

    def test_leaving_the_view_clears_the_position(self, panel: StatusBarPanel) -> None:
        panel.set_page_size(QtCore.QSizeF(600, 800), 300)
        panel.set_cursor_position(QtCore.QPointF(30, 40))

        panel.set_cursor_position(None)

        assert panel.mouse_pos_label.text() == ""

    def test_changing_the_unit_updates_the_position(
        self, panel: StatusBarPanel, provider: UnitsProvider
    ) -> None:
        panel.set_page_size(QtCore.QSizeF(600, 800), 300)
        panel.set_cursor_position(QtCore.QPointF(300, 400))
        before = panel.mouse_pos_label.text()

        provider.set_units(Units.INCHES)

        assert panel.mouse_pos_label.text() != before


class TestSeparators:
    def test_no_separator_leads_the_first_visible_field(
        self, panel: StatusBarPanel
    ) -> None:
        panel.update_page(1, 3, page("scan.png"))

        visible = [separator.isVisible() for separator, _label in panel._fields]
        assert visible == [False, False, False, False, True]


class TestSourceDpi:
    def test_a_uniform_resolution_is_shown_once(self, panel: StatusBarPanel) -> None:
        panel.set_source_dpi(Dpi.uniform(300))

        assert panel.source_dpi_label.text() == "300 dpi"
        assert panel.source_dpi_label.isVisible()

    def test_unequal_axes_are_both_shown(self, panel: StatusBarPanel) -> None:
        panel.set_source_dpi(Dpi(horizontal=300, vertical=600))

        assert panel.source_dpi_label.text() == "300 x 600 dpi"

    def test_a_page_unlike_most_pages_also_shows_their_resolution(
        self, panel: StatusBarPanel
    ) -> None:
        panel.set_source_dpi(Dpi.uniform(600), usual=Dpi.uniform(300))

        assert panel.source_dpi_label.text() == "600 dpi (most pages 300 dpi)"

    def test_a_page_like_most_pages_shows_only_its_own(
        self, panel: StatusBarPanel
    ) -> None:
        panel.set_source_dpi(Dpi.uniform(300), usual=Dpi.uniform(300))

        assert panel.source_dpi_label.text() == "300 dpi"

    @pytest.mark.parametrize("dpi", [None, Dpi(), Dpi(horizontal=1, vertical=1)])
    def test_an_unknown_resolution_hides_the_field(
        self, panel: StatusBarPanel, dpi: Dpi | None
    ) -> None:
        panel.set_source_dpi(Dpi.uniform(300))

        panel.set_source_dpi(dpi)

        assert panel.source_dpi_label.text() == ""
        assert not panel.source_dpi_label.isVisible()

    def test_the_field_explains_itself(self, panel: StatusBarPanel) -> None:
        assert "most pages" in panel.source_dpi_label.toolTip()
        assert panel.source_dpi_label.statusTip() != ""

    def test_clear_hides_it(self, panel: StatusBarPanel) -> None:
        panel.set_source_dpi(Dpi.uniform(300))

        panel.clear()

        assert panel.source_dpi_label.text() == ""
