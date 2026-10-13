"""The trim controls of the Fix Orientation options and the view that shows the result.

Behaviour follows upstream ``fix_orientation/OptionsWidget.cpp``.
"""

from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.image_views.fix_orientation import FixOrientationImageView
from scantailor.app.ui.options_widgets.fix_orientation import OptionsWidget
from scantailor.core.models import ImageId, OrthogonalRotation
from scantailor.core.transformation import ImageTransformation
from scantailor.filters.fix_orientation.filter import Filter
from scantailor.filters.fix_orientation.image_trim import ImageTrim

IMAGE = ImageId(file_path=Path("/scans/a.tiff"))
OTHER = ImageId(file_path=Path("/scans/b.tiff"))


@pytest.fixture
def filter_() -> Filter:
    return Filter()


@pytest.fixture
def widget(qtbot: QtBot, filter_: Filter) -> OptionsWidget:
    options = OptionsWidget(filter_)
    qtbot.addWidget(options)
    options.set_current_image(IMAGE)
    options.set_image_size(300, 400)
    return options


def _child(widget: QtWidgets.QWidget, cls: type, name: str):
    found = widget.findChild(cls, name)
    assert found is not None, name
    return found


def _spins(widget: QtWidgets.QWidget) -> dict[str, QtWidgets.QSpinBox]:
    return {
        side: _child(widget, QtWidgets.QSpinBox, f"trim{side}Spin")
        for side in ("Left", "Right", "Top", "Bottom")
    }


class TestControls:
    def test_the_trim_group_has_the_upstream_title_and_texts(
        self, widget: OptionsWidget
    ):
        group = _child(widget, QtWidgets.QGroupBox, "trimGroupBox")
        check = _child(widget, QtWidgets.QCheckBox, "trimEnabledCheck")
        reset = _child(widget, QtWidgets.QPushButton, "resetTrimBtn")

        assert group.title() == "Crop scan (trim)"
        assert check.text() == "Enable manual trim"
        assert reset.text() == "Reset trim"

    def test_the_corner_labels_are_plain_text(self, widget: OptionsWidget):
        texts = {
            name: _child(widget, QtWidgets.QLabel, name).text()
            for name in (
                "labelTrimTopLeft",
                "labelTrimTopRight",
                "labelTrimCenter",
                "labelTrimBottomLeft",
                "labelTrimBottomRight",
            )
        }

        assert texts == {
            "labelTrimTopLeft": "|=",
            "labelTrimTopRight": "=|",
            "labelTrimCenter": "-|-",
            "labelTrimBottomLeft": "|___",
            "labelTrimBottomRight": "___|",
        }

    def test_the_reset_button_carries_the_undo_icon(self, widget: OptionsWidget):
        reset = _child(widget, QtWidgets.QPushButton, "resetTrimBtn")

        assert not reset.icon().isNull()

    def test_everything_but_the_check_box_is_disabled_until_the_trim_is_enabled(
        self, widget: OptionsWidget
    ):
        for spin in _spins(widget).values():
            assert not spin.isEnabled()
        assert not _child(widget, QtWidgets.QLabel, "labelTrimCenter").isEnabled()
        assert not _child(widget, QtWidgets.QPushButton, "resetTrimBtn").isEnabled()
        assert _child(widget, QtWidgets.QCheckBox, "trimEnabledCheck").isEnabled()

    def test_enabling_the_trim_enables_the_controls(self, widget: OptionsWidget):
        _child(widget, QtWidgets.QCheckBox, "trimEnabledCheck").setChecked(True)

        for spin in _spins(widget).values():
            assert spin.isEnabled()
        assert _child(widget, QtWidgets.QPushButton, "resetTrimBtn").isEnabled()
        assert _child(widget, QtWidgets.QLabel, "labelTrimTopLeft").isEnabled()


class TestEditing:
    def test_enabling_stores_the_trim_and_announces_it(
        self, qtbot: QtBot, widget: OptionsWidget, filter_: Filter
    ):
        with qtbot.waitSignal(widget.trim_changed) as signal:
            _child(widget, QtWidgets.QCheckBox, "trimEnabledCheck").setChecked(True)

        assert signal.args == [IMAGE]
        assert filter_.get_trim(IMAGE) == ImageTrim(enabled=True)

    def test_a_margin_edit_is_stored_while_enabled(
        self, widget: OptionsWidget, filter_: Filter
    ):
        _child(widget, QtWidgets.QCheckBox, "trimEnabledCheck").setChecked(True)
        spins = _spins(widget)

        spins["Left"].setValue(12)
        spins["Top"].setValue(34)
        spins["Right"].setValue(5)
        spins["Bottom"].setValue(6)

        assert filter_.get_trim(IMAGE) == ImageTrim(
            enabled=True, left=12, top=34, right=5, bottom=6
        )

    def test_a_margin_edit_while_disabled_stores_nothing(
        self, qtbot: QtBot, widget: OptionsWidget, filter_: Filter
    ):
        spins = _spins(widget)
        spins["Left"].setEnabled(True)

        with qtbot.assertNotEmitted(widget.trim_changed):
            spins["Left"].setValue(12)

        assert filter_.get_trim(IMAGE) == ImageTrim()

    def test_disabling_removes_the_stored_trim_but_keeps_the_numbers_shown(
        self, qtbot: QtBot, widget: OptionsWidget, filter_: Filter
    ):
        check = _child(widget, QtWidgets.QCheckBox, "trimEnabledCheck")
        check.setChecked(True)
        _spins(widget)["Left"].setValue(12)

        with qtbot.waitSignal(widget.trim_changed):
            check.setChecked(False)

        assert filter_.get_trim(IMAGE) == ImageTrim()
        assert _spins(widget)["Left"].value() == 12
        check.setChecked(True)
        assert filter_.get_trim(IMAGE).left == 12

    def test_reset_zeroes_the_margins_and_keeps_the_trim_enabled(
        self, widget: OptionsWidget, filter_: Filter
    ):
        _child(widget, QtWidgets.QCheckBox, "trimEnabledCheck").setChecked(True)
        for spin in _spins(widget).values():
            spin.setValue(20)

        _child(widget, QtWidgets.QPushButton, "resetTrimBtn").click()

        assert all(spin.value() == 0 for spin in _spins(widget).values())
        assert filter_.get_trim(IMAGE) == ImageTrim(enabled=True)

    def test_reset_restores_the_maximums_the_other_margins_had_taken(
        self, widget: OptionsWidget
    ):
        _child(widget, QtWidgets.QCheckBox, "trimEnabledCheck").setChecked(True)
        spins = _spins(widget)
        spins["Right"].setValue(100)
        assert spins["Left"].maximum() < 268

        _child(widget, QtWidgets.QPushButton, "resetTrimBtn").click()

        assert spins["Left"].maximum() == 268

    def test_get_current_trim_reports_what_the_controls_show(
        self, widget: OptionsWidget
    ):
        assert widget.get_current_trim() == ImageTrim()
        _child(widget, QtWidgets.QCheckBox, "trimEnabledCheck").setChecked(True)
        _spins(widget)["Bottom"].setValue(9)

        assert widget.get_current_trim() == ImageTrim(enabled=True, bottom=9)


class TestImageSwitching:
    def test_the_controls_show_the_stored_trim_of_the_selected_image(
        self, widget: OptionsWidget, filter_: Filter
    ):
        filter_.set_trim(
            OTHER, ImageTrim(enabled=True, left=1, top=2, right=3, bottom=4)
        )

        widget.set_current_image(OTHER)

        assert _child(widget, QtWidgets.QCheckBox, "trimEnabledCheck").isChecked()
        assert [spin.value() for spin in _spins(widget).values()] == [1, 3, 2, 4]
        assert _spins(widget)["Left"].isEnabled()

    def test_showing_a_trim_does_not_store_or_announce_anything(
        self, qtbot: QtBot, widget: OptionsWidget, filter_: Filter
    ):
        filter_.set_trim(OTHER, ImageTrim(enabled=True, left=7))

        with qtbot.assertNotEmitted(widget.trim_changed):
            widget.set_current_image(OTHER)

        assert filter_.get_trim(IMAGE) == ImageTrim()
        assert filter_.get_trim(OTHER).left == 7

    def test_an_image_without_a_trim_shows_a_disabled_zeroed_group(
        self, widget: OptionsWidget, filter_: Filter
    ):
        _child(widget, QtWidgets.QCheckBox, "trimEnabledCheck").setChecked(True)
        _spins(widget)["Left"].setValue(30)

        widget.set_current_image(OTHER)

        assert not _child(widget, QtWidgets.QCheckBox, "trimEnabledCheck").isChecked()
        assert _spins(widget)["Left"].value() == 0
        assert not _spins(widget)["Left"].isEnabled()

    def test_a_stored_margin_larger_than_the_previous_images_limit_is_not_clamped(
        self, widget: OptionsWidget, filter_: Filter
    ):
        widget.set_image_size(100, 100)
        filter_.set_trim(OTHER, ImageTrim(enabled=True, left=250))

        widget.set_current_image(OTHER)

        assert _spins(widget)["Left"].value() == 250


class TestMaximums:
    """Opposite margins may not leave less than 32 pixels, as in upstream."""

    def test_each_margin_may_reach_the_image_size_less_32_minus_its_opposite(
        self, widget: OptionsWidget
    ):
        spins = _spins(widget)
        _child(widget, QtWidgets.QCheckBox, "trimEnabledCheck").setChecked(True)

        # 300 x 400 image: horizontal 300 - 32 = 268, vertical 400 - 32 = 368.
        assert (spins["Left"].maximum(), spins["Right"].maximum()) == (268, 268)
        assert (spins["Top"].maximum(), spins["Bottom"].maximum()) == (368, 368)

        spins["Right"].setValue(100)
        spins["Bottom"].setValue(68)

        assert spins["Left"].maximum() == 168
        assert spins["Top"].maximum() == 300

    def test_a_margin_cannot_exceed_its_maximum(self, widget: OptionsWidget):
        spins = _spins(widget)
        _child(widget, QtWidgets.QCheckBox, "trimEnabledCheck").setChecked(True)

        spins["Left"].setValue(10_000)

        assert spins["Left"].value() == 268

    def test_an_image_smaller_than_the_minimum_allows_no_trim(
        self, widget: OptionsWidget
    ):
        widget.set_image_size(20, 20)

        assert all(spin.maximum() == 0 for spin in _spins(widget).values())


class TestView:
    @staticmethod
    def _view(qtbot: QtBot) -> FixOrientationImageView:
        view = FixOrientationImageView()
        qtbot.addWidget(view)
        view.resize(400, 400)
        image = np.zeros((400, 300), dtype=np.uint8)
        view.set_image(image, transformation=ImageTransformation.from_size(300, 400))
        return view

    def test_the_view_shows_only_the_trimmed_region(self, qtbot: QtBot):
        view = self._view(qtbot)

        view.set_trim(ImageTrim(enabled=True, left=10, top=20, right=30, bottom=40))

        rect = view.virtual_display_area
        assert (rect.width(), rect.height()) == (260, 340)

    def test_a_quarter_turn_swaps_the_trimmed_region_with_the_page(self, qtbot: QtBot):
        view = self._view(qtbot)
        view.set_trim(ImageTrim(enabled=True, left=10, top=20, right=30, bottom=40))

        view.set_rotation(OrthogonalRotation(degrees=90))

        rect = view.virtual_display_area
        assert (rect.width(), rect.height()) == (340, 260)

    def test_disabling_the_trim_shows_the_whole_image_again(self, qtbot: QtBot):
        view = self._view(qtbot)
        view.set_trim(ImageTrim(enabled=True, left=10))

        view.set_trim(ImageTrim())

        rect = view.virtual_display_area
        assert (rect.width(), rect.height()) == (300, 400)
