"""The Deskew angle as the user sees it: counter-clockwise positive, editable in
Auto, and "?" until the page has an angle."""

import pathlib

import pytest
from PySide6 import QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.options_widgets.deskew import OptionsWidget
from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.filters.deskew.dependencies import Dependencies
from scantailor.filters.deskew.filter import Filter
from scantailor.filters.deskew.params import (
    AutoManualMode,
    Params,
    shown_deskew_angle,
    stored_deskew_angle,
)
from tests.ui.locale_numbers import shown_numbers


def _page(name: str = "page") -> PageId:
    return PageId(
        image_id=ImageId(file_path=pathlib.Path(f"/{name}.tiff")),
        sub_page=SubPage.SINGLE_PAGE,
    )


@pytest.fixture
def filter_() -> Filter:
    return Filter()


@pytest.fixture
def widget(qtbot: QtBot, filter_: Filter) -> OptionsWidget:
    options = OptionsWidget(filter_)
    qtbot.addWidget(options)
    options.set_current_page(_page())
    return options


def _angle(widget: OptionsWidget) -> QtWidgets.QDoubleSpinBox:
    spin_box = widget.findChild(QtWidgets.QDoubleSpinBox, "angleSpinBox")
    assert spin_box is not None
    return spin_box


def _button(widget: OptionsWidget, name: str) -> QtWidgets.QPushButton:
    button = widget.findChild(QtWidgets.QPushButton, name)
    assert button is not None
    return button


def _detect(filter_: Filter, rotation: float) -> None:
    filter_.settings.store_detection(_page(), Dependencies(), rotation=rotation)


class TestSignConversion:
    @pytest.mark.parametrize("stored", [1.5, -2.25, 45.0, -45.0])
    def test_shown_and_stored_angles_have_opposite_signs_and_round_trip(
        self, stored: float
    ):
        assert shown_deskew_angle(stored) == -stored
        assert stored_deskew_angle(shown_deskew_angle(stored)) == stored

    def test_a_zero_angle_is_shown_without_a_minus_sign(self):
        assert str(shown_deskew_angle(0.0)) == "0.0"
        assert str(stored_deskew_angle(0.0)) == "0.0"


class TestTypedAngleRoundTrip:
    def test_a_typed_counter_clockwise_angle_is_stored_clockwise_and_shown_as_typed(
        self, qtbot: QtBot, widget: OptionsWidget, filter_: Filter
    ):
        _detect(filter_, 0.0)
        widget.refresh()

        with qtbot.waitSignal(widget.angle_changed) as changed:
            _angle(widget).setValue(1.5)

        assert filter_.get_deskew_angle(_page()) == pytest.approx(-1.5)
        assert changed.args == [_page(), pytest.approx(-1.5), True]

        widget.set_current_page(_page("other"))
        widget.set_current_page(_page())
        assert _angle(widget).value() == pytest.approx(1.5)
        assert shown_numbers(_angle(widget).text()) == pytest.approx([1.5])

    def test_the_project_json_keeps_the_stored_clockwise_angle(
        self, widget: OptionsWidget, filter_: Filter
    ):
        _detect(filter_, 0.0)
        widget.refresh()

        _angle(widget).setValue(2.0)

        saved = filter_.get_params(_page()).model_dump(mode="json")
        assert saved["deskew_angle_deg"] == pytest.approx(-2.0)

    def test_a_detected_angle_is_shown_with_the_opposite_sign(
        self, widget: OptionsWidget, filter_: Filter
    ):
        _detect(filter_, 3.0)

        widget.refresh()

        assert _angle(widget).value() == pytest.approx(-3.0)
        assert shown_numbers(_angle(widget).text()) == pytest.approx([-3.0])


class TestEditingInAuto:
    def test_the_spin_box_is_editable_once_an_automatic_angle_is_detected(
        self, widget: OptionsWidget, filter_: Filter
    ):
        _detect(filter_, 1.0)

        widget.refresh()

        assert filter_.get_params(_page()).mode is AutoManualMode.AUTO
        assert _button(widget, "autoBtn").isChecked()
        assert _angle(widget).isEnabled()

    def test_editing_an_automatic_angle_switches_the_page_to_manual(
        self, qtbot: QtBot, widget: OptionsWidget, filter_: Filter
    ):
        _detect(filter_, 1.0)
        widget.refresh()

        with qtbot.waitSignal(widget.angle_changed) as changed:
            _angle(widget).setValue(-0.75)

        stored = filter_.get_params(_page())
        assert stored.mode is AutoManualMode.MANUAL
        assert stored.deskew_angle_deg == pytest.approx(0.75)
        assert _button(widget, "manualBtn").isChecked()
        assert not _button(widget, "autoBtn").isChecked()
        assert changed.args[2] is True

    def test_a_manual_angle_stays_editable(
        self, widget: OptionsWidget, filter_: Filter
    ):
        filter_.settings.set_params(
            _page(), Params(deskew_angle_deg=2.0, mode=AutoManualMode.MANUAL)
        )

        widget.refresh()

        assert _angle(widget).isEnabled()
        assert _angle(widget).value() == pytest.approx(-2.0)


class TestUnknownAngle:
    def test_a_page_detection_has_not_reached_shows_a_question_mark(
        self, widget: OptionsWidget
    ):
        assert _angle(widget).text() == "?"
        assert not _angle(widget).isEnabled()

    def test_a_new_widget_shows_a_question_mark_before_any_page(
        self, qtbot: QtBot, filter_: Filter
    ):
        options = OptionsWidget(filter_)
        qtbot.addWidget(options)

        assert _angle(options).text() == "?"

    def test_asking_for_auto_again_shows_a_question_mark_until_detection_finishes(
        self, widget: OptionsWidget, filter_: Filter
    ):
        filter_.settings.set_params(
            _page(), Params(deskew_angle_deg=2.0, mode=AutoManualMode.MANUAL)
        )
        widget.refresh()

        _button(widget, "autoBtn").click()

        assert _angle(widget).text() == "?"
        assert not _angle(widget).isEnabled()

        _detect(filter_, 0.5)
        widget.refresh()

        assert shown_numbers(_angle(widget).text()) == pytest.approx([-0.5])
        assert _angle(widget).isEnabled()

    def test_a_detected_angle_at_the_limit_is_shown_as_a_number(
        self, widget: OptionsWidget, filter_: Filter
    ):
        _detect(filter_, 45.0)

        widget.refresh()

        assert _angle(widget).value() == pytest.approx(-45.0)
        assert shown_numbers(_angle(widget).text()) == pytest.approx([-45.0])

    def test_showing_the_question_mark_does_not_store_an_angle(
        self, widget: OptionsWidget, filter_: Filter
    ):
        widget.refresh()

        assert not filter_.settings.is_params_set(_page())
