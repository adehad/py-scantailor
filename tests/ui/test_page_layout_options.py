"""Margins panel: size lock, match-all, source DPI, and the per-side Apply dialog."""

from pathlib import Path

import pytest
from PySide6 import QtWidgets
from PySide6.QtWidgets import QCheckBox, QComboBox, QLineEdit, QPushButton
from pytestqt.qtbot import QtBot

from scantailor.app.ui.dialogs.apply_to import (
    MARGIN_SIDE_ICONS,
    MARGIN_SIDE_OPTIONS,
    MARGINS_OPTIONS_TITLE,
    MARGINS_TITLE,
    ApplyToDialog,
)
from scantailor.app.ui.icons import is_palette_icon
from scantailor.app.ui.options_widgets import page_layout
from scantailor.app.ui.units_provider import UnitsProvider
from scantailor.core.models import Dpi, ImageId, PageId, SubPage
from scantailor.core.propagation import ApplyScope
from scantailor.core.settings import Units
from scantailor.core.stage_sequence import StageSequence
from scantailor.filters.page_layout.alignment import Alignment
from scantailor.filters.page_layout.params import ContentSize, Params


def _page(name: str) -> PageId:
    return PageId(
        image_id=ImageId(file_path=Path(f"{name}.png")),
        sub_page=SubPage.SINGLE_PAGE,
    )


PAGE = _page("a")
OTHER = _page("b")


@pytest.fixture
def stages() -> StageSequence:
    sequence = StageSequence()
    settings = sequence.page_layout_filter.settings
    settings.set_params(
        PAGE, Params(content_size_mm=ContentSize(width_mm=100, height_mm=100))
    )
    settings.set_params(
        OTHER, Params(content_size_mm=ContentSize(width_mm=300, height_mm=400))
    )
    return sequence


@pytest.fixture
def widget(qtbot: QtBot, stages: StageSequence) -> page_layout.OptionsWidget:
    options = page_layout.OptionsWidget(
        stages.page_layout_filter, units_provider=UnitsProvider(Units.MILLIMETERS)
    )
    qtbot.addWidget(options)
    options.set_current_page(PAGE)
    return options


def child(widget: QtWidgets.QWidget, kind: type, name: str):
    found = widget.findChild(kind, name)
    assert found is not None, name
    return found


class TestLockAggregateSize:
    def test_checkbox_has_the_upstream_text_and_starts_unchecked(
        self, widget: page_layout.OptionsWidget
    ) -> None:
        box = child(widget, QCheckBox, "freezeAggregateHardSizeCb")

        assert box.text() == "Lock aggregate size for matching"
        assert not box.isChecked()

    def test_checking_it_freezes_the_current_aggregate_size(
        self, widget: page_layout.OptionsWidget, stages: StageSequence
    ) -> None:
        settings = stages.page_layout_filter.settings

        child(widget, QCheckBox, "freezeAggregateHardSizeCb").click()
        settings.set_params(
            OTHER, Params(content_size_mm=ContentSize(width_mm=1, height_mm=1))
        )

        assert settings.is_aggregate_hard_size_frozen()
        assert settings.get_aggregate_hard_size_mm().width_mm == 320.0

    def test_unchecking_it_releases_the_size(
        self, widget: page_layout.OptionsWidget, stages: StageSequence
    ) -> None:
        box = child(widget, QCheckBox, "freezeAggregateHardSizeCb")
        box.click()

        box.click()

        assert not stages.page_layout_filter.settings.is_aggregate_hard_size_frozen()

    def test_toggling_asks_for_a_redraw(
        self, widget: page_layout.OptionsWidget, qtbot: QtBot
    ) -> None:
        with qtbot.waitSignal(widget.aggregate_size_changed):
            child(widget, QCheckBox, "freezeAggregateHardSizeCb").click()

    def test_the_box_follows_the_settings_when_a_page_is_shown(
        self, widget: page_layout.OptionsWidget, stages: StageSequence
    ) -> None:
        stages.page_layout_filter.settings.set_aggregate_hard_size_frozen(True)

        widget.set_current_page(OTHER)

        assert child(widget, QCheckBox, "freezeAggregateHardSizeCb").isChecked()


class TestMatchSizeToAllPages:
    def test_button_has_the_upstream_text_and_no_icon(
        self, widget: page_layout.OptionsWidget
    ) -> None:
        button = child(widget, QPushButton, "matchSizeToAllBtn")

        assert button.text() == "Match size to all pages"
        assert button.icon().isNull()

    def test_clicking_turns_matching_on_for_the_current_page(
        self, widget: page_layout.OptionsWidget, stages: StageSequence
    ) -> None:
        filter_ = stages.page_layout_filter
        filter_.set_alignment(PAGE, Alignment(is_null=True))
        widget.set_current_page(PAGE)

        child(widget, QPushButton, "matchSizeToAllBtn").click()

        assert filter_.get_alignment(PAGE).aligned_with_others()
        assert child(widget, QCheckBox, "alignWithOthersCB").isChecked()

    def test_clicking_asks_for_every_page_to_follow(
        self, widget: page_layout.OptionsWidget, qtbot: QtBot
    ) -> None:
        with (
            qtbot.waitSignal(widget.match_size_to_all_requested),
            qtbot.waitSignal(widget.aggregate_size_changed),
        ):
            child(widget, QPushButton, "matchSizeToAllBtn").click()

    def test_clicking_keeps_the_chosen_position(
        self, widget: page_layout.OptionsWidget, stages: StageSequence
    ) -> None:
        filter_ = stages.page_layout_filter
        before = filter_.get_alignment(PAGE)

        child(widget, QPushButton, "matchSizeToAllBtn").click()

        after = filter_.get_alignment(PAGE)
        assert (after.vertical, after.horizontal) == (
            before.vertical,
            before.horizontal,
        )


class TestSourceDpi:
    @pytest.fixture
    def shown(self, widget: page_layout.OptionsWidget) -> page_layout.OptionsWidget:
        widget.set_source_image(2480, 3508, Dpi.uniform(300))
        return widget

    def test_fields_show_the_resolution_of_the_image(
        self, shown: page_layout.OptionsWidget
    ) -> None:
        assert child(shown, QLineEdit, "sourceXDpi").text() == "300"
        assert child(shown, QLineEdit, "sourceYDpi").text() == "300"

    def test_a_preset_resolution_selects_its_combo_entry(
        self, shown: page_layout.OptionsWidget
    ) -> None:
        assert child(shown, QComboBox, "sourceDpiCombo").currentText() == "300 x 300"

    def test_other_resolutions_select_custom(
        self, widget: page_layout.OptionsWidget
    ) -> None:
        widget.set_source_image(2480, 3508, Dpi(horizontal=300, vertical=301))

        assert child(widget, QComboBox, "sourceDpiCombo").currentText() == "Custom"

    def test_an_undefined_resolution_leaves_the_fields_empty(
        self, widget: page_layout.OptionsWidget
    ) -> None:
        widget.set_source_image(2480, 3508, Dpi())

        assert child(widget, QLineEdit, "sourceXDpi").text() == ""
        assert child(widget, QLineEdit, "sourceYDpi").text() == ""

    def test_choosing_a_preset_reports_it(
        self, shown: page_layout.OptionsWidget, qtbot: QtBot
    ) -> None:
        combo = child(shown, QComboBox, "sourceDpiCombo")

        with qtbot.waitSignal(shown.source_dpi_changed) as signal:
            combo.setCurrentIndex(3)
            combo.activated.emit(3)

        assert signal.args == [600, 600]
        assert child(shown, QLineEdit, "sourceXDpi").text() == "600"

    def test_typing_a_resolution_reports_it_when_editing_finishes(
        self, shown: page_layout.OptionsWidget, qtbot: QtBot
    ) -> None:
        x_field = child(shown, QLineEdit, "sourceXDpi")
        x_field.setText("400")

        with qtbot.waitSignal(shown.source_dpi_changed) as signal:
            x_field.editingFinished.emit()

        assert signal.args == [400, 300]

    def test_an_unchanged_resolution_is_not_reported(
        self, shown: page_layout.OptionsWidget, qtbot: QtBot
    ) -> None:
        with qtbot.assertNotEmitted(shown.source_dpi_changed):
            child(shown, QLineEdit, "sourceXDpi").editingFinished.emit()

    def test_an_incomplete_resolution_is_not_reported(
        self, shown: page_layout.OptionsWidget, qtbot: QtBot
    ) -> None:
        x_field = child(shown, QLineEdit, "sourceXDpi")
        x_field.clear()

        with qtbot.assertNotEmitted(shown.source_dpi_changed):
            x_field.editingFinished.emit()

    def test_an_undefined_resolution_is_not_reported(
        self, shown: page_layout.OptionsWidget, qtbot: QtBot
    ) -> None:
        x_field = child(shown, QLineEdit, "sourceXDpi")
        x_field.setText("0")

        with qtbot.assertNotEmitted(shown.source_dpi_changed):
            x_field.editingFinished.emit()

    def test_a_doubtful_resolution_is_flagged_with_the_upstream_hint(
        self, shown: page_layout.OptionsWidget
    ) -> None:
        y_field = child(shown, QLineEdit, "sourceYDpi")
        y_field.setText("100")

        y_field.editingFinished.emit()

        assert y_field.toolTip() == (
            "DPI is too small. Even if it's correct, you are not going to get "
            "acceptable results with it."
        )
        assert child(shown, QLineEdit, "sourceXDpi").toolTip() == ""
        assert "color" in y_field.styleSheet()
        assert child(shown, QLineEdit, "sourceXDpi").styleSheet() == ""

    def test_a_reasonable_resolution_clears_the_hint(
        self, shown: page_layout.OptionsWidget
    ) -> None:
        y_field = child(shown, QLineEdit, "sourceYDpi")
        y_field.setText("100")
        y_field.editingFinished.emit()

        y_field.setText("300")
        y_field.editingFinished.emit()

        assert y_field.toolTip() == ""
        assert y_field.styleSheet() == ""

    def test_a_reported_resolution_is_used_for_pixel_units(
        self, qtbot: QtBot, stages: StageSequence
    ) -> None:
        provider = UnitsProvider(Units.PIXELS)
        options = page_layout.OptionsWidget(
            stages.page_layout_filter, units_provider=provider
        )
        qtbot.addWidget(options)
        options.set_current_page(PAGE)
        options.set_page_dpi(300, 300)
        options.set_source_image(2480, 3508, Dpi.uniform(300))
        left = child(options, QtWidgets.QDoubleSpinBox, "leftMarginSpinBox")
        before = left.value()

        x_field = child(options, QLineEdit, "sourceXDpi")
        x_field.setText("600")
        x_field.editingFinished.emit()

        assert left.value() == pytest.approx(before * 2, abs=1.0)


class TestFixDpiButton:
    def test_button_has_the_upstream_text_and_asks_for_the_dialog(
        self, widget: page_layout.OptionsWidget, qtbot: QtBot
    ) -> None:
        button = child(widget, QPushButton, "fixDpiBtn")

        with qtbot.waitSignal(widget.fix_dpi_requested):
            button.click()

        assert button.text() == "Fix all..."
        assert button.icon().isNull()


class TestApplyMarginsDialog:
    @pytest.fixture
    def dialog(self, qtbot: QtBot) -> ApplyToDialog:
        pages = [PAGE, OTHER]
        dialog = ApplyToDialog(
            PAGE,
            pages,
            title=MARGINS_TITLE,
            options=MARGIN_SIDE_OPTIONS,
            option_icons=MARGIN_SIDE_ICONS,
            options_title=MARGINS_OPTIONS_TITLE,
        )
        qtbot.addWidget(dialog)
        return dialog

    def test_each_side_has_a_checked_box_in_upstream_order(
        self, dialog: ApplyToDialog
    ) -> None:
        group = child(dialog, QtWidgets.QGroupBox, "optionsGroup")
        boxes = group.findChildren(QCheckBox)

        assert group.title() == "Apply margins"
        assert [box.text() for box in boxes] == ["Left", "Right", "Top", "Bottom"]
        assert all(box.isChecked() for box in boxes)
        assert dialog.checked_options() == {"left", "right", "top", "bottom"}

    def test_each_side_carries_its_edge_arrow(self, dialog: ApplyToDialog) -> None:
        for side, glyph in MARGIN_SIDE_ICONS.items():
            box = child(dialog, QCheckBox, f"{side}Option")
            assert is_palette_icon(box.icon()), glyph

        assert MARGIN_SIDE_ICONS == {
            "left": "arrow-left-to-line",
            "right": "arrow-right-to-line",
            "top": "arrow-up-to-line",
            "bottom": "arrow-down-to-line",
        }

    def test_unchecked_sides_drop_out_of_the_choice(
        self, dialog: ApplyToDialog
    ) -> None:
        child(dialog, QCheckBox, "leftOption").setChecked(False)
        child(dialog, QCheckBox, "bottomOption").setChecked(False)

        assert dialog.checked_options() == {"right", "top"}

    def test_the_scope_choice_is_unaffected_by_the_sides(
        self, dialog: ApplyToDialog
    ) -> None:
        dialog.scope = ApplyScope.ALL_PAGES
        child(dialog, QCheckBox, "topOption").setChecked(False)

        assert dialog.pages() == [PAGE, OTHER]

    def test_dialogs_without_options_have_no_group(self, qtbot: QtBot) -> None:
        dialog = ApplyToDialog(PAGE, [PAGE])
        qtbot.addWidget(dialog)

        assert dialog.findChild(QtWidgets.QGroupBox, "optionsGroup") is None
        assert dialog.checked_options() == set()

    def test_options_without_a_title_stay_in_a_column(self, qtbot: QtBot) -> None:
        dialog = ApplyToDialog(PAGE, [PAGE], options={"a": "A"})
        qtbot.addWidget(dialog)

        assert dialog.findChild(QtWidgets.QGroupBox, "optionsGroup") is None
        assert child(dialog, QCheckBox, "aOption").text() == "A"
