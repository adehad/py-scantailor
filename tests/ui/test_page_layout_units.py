"""The Margins options show and edit values in the chosen unit."""

from pathlib import Path

import pytest
from PySide6.QtWidgets import QDoubleSpinBox, QToolButton
from pytestqt.qtbot import QtBot

from scantailor.app.ui.options_widgets import page_layout
from scantailor.app.ui.units_provider import UnitsProvider
from scantailor.core.models import ImageId, Margins, PageId, SubPage
from scantailor.core.settings import Units
from scantailor.core.stage_sequence import StageSequence

PAGE = PageId(image_id=ImageId(file_path=Path("a.png")), sub_page=SubPage.SINGLE_PAGE)


@pytest.fixture
def provider() -> UnitsProvider:
    return UnitsProvider(Units.MILLIMETERS)


@pytest.fixture
def stages() -> StageSequence:
    sequence = StageSequence()
    filter_ = sequence.page_layout_filter
    params = filter_.get_params(PAGE).with_margins(
        Margins(top=25.4, bottom=50.8, left=12.7, right=76.2)
    )
    filter_.set_params(PAGE, params)
    return sequence


@pytest.fixture
def widget(
    qtbot: QtBot, stages: StageSequence, provider: UnitsProvider
) -> page_layout.OptionsWidget:
    options = page_layout.OptionsWidget(
        stages.page_layout_filter, units_provider=provider
    )
    qtbot.addWidget(options)
    options.set_page_dpi(300, 600)
    options.set_current_page(PAGE)
    return options


def spin(widget: page_layout.OptionsWidget, side: str) -> QDoubleSpinBox:
    box = widget.findChild(QDoubleSpinBox, f"{side}MarginSpinBox")
    assert box is not None
    return box


def shown(widget: page_layout.OptionsWidget) -> dict[str, float]:
    return {
        side: spin(widget, side).value() for side in ("top", "bottom", "left", "right")
    }


class TestDisplay:
    def test_millimetres_show_the_stored_values(
        self, widget: page_layout.OptionsWidget
    ) -> None:
        assert shown(widget) == {
            "top": 25.4,
            "bottom": 50.8,
            "left": 12.7,
            "right": 76.2,
        }

    def test_inches_show_the_same_margins_in_inches(
        self, widget: page_layout.OptionsWidget, provider: UnitsProvider
    ) -> None:
        provider.set_units(Units.INCHES)

        assert shown(widget) == {"top": 1.0, "bottom": 2.0, "left": 0.5, "right": 3.0}

    def test_centimetres_show_the_same_margins_in_centimetres(
        self, widget: page_layout.OptionsWidget, provider: UnitsProvider
    ) -> None:
        provider.set_units(Units.CENTIMETERS)

        assert shown(widget) == {
            "top": 2.54,
            "bottom": 5.08,
            "left": 1.27,
            "right": 7.62,
        }

    def test_pixels_use_the_resolution_of_each_axis(
        self, widget: page_layout.OptionsWidget, provider: UnitsProvider
    ) -> None:
        provider.set_units(Units.PIXELS)

        shown_px = shown(widget)
        assert shown_px["left"] == pytest.approx(150.0)
        assert shown_px["right"] == pytest.approx(900.0)
        assert shown_px["top"] == pytest.approx(600.0)
        assert shown_px["bottom"] == pytest.approx(1200.0)

    def test_precision_follows_the_unit(
        self, widget: page_layout.OptionsWidget, provider: UnitsProvider
    ) -> None:
        top = spin(widget, "top")
        assert (top.decimals(), top.singleStep()) == (1, 1.0)

        provider.set_units(Units.INCHES)

        assert (top.decimals(), top.singleStep()) == (2, 0.01)

    def test_a_new_page_is_shown_in_the_current_unit(
        self,
        widget: page_layout.OptionsWidget,
        stages: StageSequence,
        provider: UnitsProvider,
    ) -> None:
        provider.set_units(Units.INCHES)
        other = PageId(
            image_id=ImageId(file_path=Path("b.png")), sub_page=SubPage.SINGLE_PAGE
        )
        filter_ = stages.page_layout_filter
        filter_.set_params(
            other, filter_.get_params(other).with_margins(Margins.uniform(50.8))
        )

        widget.set_current_page(other)

        assert shown(widget) == {"top": 2.0, "bottom": 2.0, "left": 2.0, "right": 2.0}


class TestEditing:
    def test_a_value_typed_in_inches_is_stored_in_millimetres(
        self,
        widget: page_layout.OptionsWidget,
        stages: StageSequence,
        provider: UnitsProvider,
        qtbot: QtBot,
    ) -> None:
        provider.set_units(Units.INCHES)

        with qtbot.waitSignal(widget.margins_changed) as signal:
            spin(widget, "left").setValue(1.0)

        stored = stages.page_layout_filter.get_params(PAGE).hard_margins_mm
        assert stored.left == pytest.approx(25.4)
        assert signal.args == [PAGE, stored]

    def test_untouched_margins_keep_their_millimetre_values(
        self,
        widget: page_layout.OptionsWidget,
        stages: StageSequence,
        provider: UnitsProvider,
    ) -> None:
        filter_ = stages.page_layout_filter
        odd = Margins(top=10.0, bottom=11.11, left=12.34, right=13.57)
        filter_.set_params(PAGE, filter_.get_params(PAGE).with_margins(odd))
        widget.set_current_page(PAGE)
        provider.set_units(Units.INCHES)

        spin(widget, "top").setValue(1.0)

        stored = filter_.get_params(PAGE).hard_margins_mm
        assert stored == Margins(top=25.4, bottom=11.11, left=12.34, right=13.57)

    def test_switching_units_does_not_change_the_stored_margins(
        self,
        widget: page_layout.OptionsWidget,
        stages: StageSequence,
        provider: UnitsProvider,
    ) -> None:
        before = stages.page_layout_filter.get_params(PAGE).hard_margins_mm

        for units in (Units.INCHES, Units.PIXELS, Units.MILLIMETERS):
            provider.set_units(units)

        assert stages.page_layout_filter.get_params(PAGE).hard_margins_mm == before
        assert shown(widget)["top"] == 25.4

    def test_linked_margins_move_together_in_any_unit(
        self,
        widget: page_layout.OptionsWidget,
        stages: StageSequence,
        provider: UnitsProvider,
    ) -> None:
        provider.set_units(Units.INCHES)
        link = widget.findChild(QToolButton, "topBottomLink")
        assert link is not None
        link.setChecked(True)

        spin(widget, "top").setValue(2.0)

        stored = stages.page_layout_filter.get_params(PAGE).hard_margins_mm
        assert stored.top == pytest.approx(50.8)
        assert stored.bottom == pytest.approx(50.8)
        assert spin(widget, "bottom").value() == 2.0

    def test_pixels_edit_with_the_axis_resolution(
        self,
        widget: page_layout.OptionsWidget,
        stages: StageSequence,
        provider: UnitsProvider,
    ) -> None:
        provider.set_units(Units.PIXELS)

        spin(widget, "right").setValue(300.0)

        stored = stages.page_layout_filter.get_params(PAGE).hard_margins_mm
        assert stored.right == pytest.approx(25.4)

    def test_pixels_without_a_resolution_leave_values_unconverted(
        self,
        widget: page_layout.OptionsWidget,
        provider: UnitsProvider,
    ) -> None:
        widget.set_page_dpi(0, 0)

        provider.set_units(Units.PIXELS)

        assert shown(widget)["top"] == 25.4

    def test_the_range_covers_the_same_distance_in_every_unit(
        self, widget: page_layout.OptionsWidget, provider: UnitsProvider
    ) -> None:
        provider.set_units(Units.INCHES)

        top = spin(widget, "top")
        assert top.maximum() == pytest.approx(9999.0 / 25.4, abs=0.01)
        assert top.minimum() == pytest.approx(-9999.0 / 25.4, abs=0.01)
