"""Oblique (shear) controls of the Deskew stage: options widget, view and dialogs."""

import math
import pathlib
from collections.abc import Iterator

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.dialogs.apply_to import (
    DESKEW_OPTIONS,
    DESKEW_TITLE,
    ApplyToDialog,
)
from scantailor.app.ui.dialogs.default_params import DefaultParamsDialog
from scantailor.app.ui.image_views.deskew import DeskewImageView
from scantailor.app.ui.options_widgets.deskew import OptionsWidget
from scantailor.app.ui.units_provider import UnitsProvider
from scantailor.core.default_params import DefaultParams, DeskewDefaults
from scantailor.core.default_params_profiles import ProfileStore
from scantailor.core.models import Dpi, ImageId, PageId, SubPage
from scantailor.core.settings import ApplicationSettings
from scantailor.core.transformation import ImageTransformation
from scantailor.filters.deskew.dependencies import Dependencies
from scantailor.filters.deskew.filter import Filter
from scantailor.filters.deskew.params import MAX_OBLIQUE_DEG, AutoManualMode, Params

AUTO = AutoManualMode.AUTO
MANUAL = AutoManualMode.MANUAL
IMAGE_WIDTH = 600
IMAGE_HEIGHT = 800
DEPENDENCIES = Dependencies(
    page_outline=((0.0, 0.0), (600.0, 0.0), (600.0, 800.0), (0.0, 800.0)),
    rotation_degrees=0,
)
# A windowed view, a maximised one, and a narrow tall one.
VIEW_SIZES = [(640, 480), (1900, 1000), (420, 900)]


def _page(name: str = "page") -> PageId:
    return PageId(
        image_id=ImageId(file_path=pathlib.Path(f"/{name}.tiff")),
        sub_page=SubPage.SINGLE_PAGE,
    )


def _find[T: QtWidgets.QWidget](
    widget: QtWidgets.QWidget, kind: type[T], name: str
) -> T:
    child = widget.findChild(kind, name)
    assert child is not None, name
    return child


@pytest.fixture
def filter_() -> Filter:
    return Filter()


@pytest.fixture
def widget(qtbot: QtBot, filter_: Filter) -> OptionsWidget:
    options = OptionsWidget(filter_)
    qtbot.addWidget(options)
    options.set_current_page(_page())
    return options


def _spin(widget: OptionsWidget, name: str) -> QtWidgets.QDoubleSpinBox:
    return _find(widget, QtWidgets.QDoubleSpinBox, name)


def _button(widget: OptionsWidget, name: str) -> QtWidgets.QPushButton:
    return _find(widget, QtWidgets.QPushButton, name)


def _check(widget: OptionsWidget) -> QtWidgets.QCheckBox:
    return _find(widget, QtWidgets.QCheckBox, "autoObliqueCheckBox")


class TestOptionsWidgetControls:
    """The controls and strings of the upstream Deskew options."""

    def test_has_the_upstream_oblique_controls_and_strings(self, widget: OptionsWidget):
        assert _check(widget).text() == "Automatic oblique correction"
        assert _check(widget).toolTip() == (
            "When deskew mode is Auto, run automatic oblique (shear) correction. "
            "Disable for pages that only need rotation."
        )
        assert _find(widget, QtWidgets.QLabel, "obliqueLabel").text() == "Oblique (°):"
        assert (
            _find(widget, QtWidgets.QLabel, "obliqueModeLabel").text() == "Oblique mode"
        )
        assert _button(widget, "obliqueAutoBtn").text() == "Auto"
        assert _button(widget, "obliqueAutoBtn").toolTip() == (
            "Automatically estimate oblique (shear) correction."
        )
        assert _button(widget, "obliqueManualBtn").text() == "Manual"
        assert _button(widget, "obliqueManualBtn").toolTip() == (
            "Keep the oblique angle set in the spin box."
        )

    def test_the_oblique_spin_box_covers_five_degrees_either_way(
        self, widget: OptionsWidget
    ):
        spin = _spin(widget, "obliqueSpinBox")

        assert spin.minimum() == -MAX_OBLIQUE_DEG
        assert spin.maximum() == MAX_OBLIQUE_DEG
        assert spin.singleStep() == pytest.approx(0.1)


class TestFreshPage:
    """With nothing changed, a page is not sheared and the controls agree."""

    def test_default_off_shows_manual_zero_and_an_unchecked_box(
        self, widget: OptionsWidget
    ):
        assert not _check(widget).isChecked()
        assert _button(widget, "obliqueManualBtn").isChecked()
        assert not _button(widget, "obliqueAutoBtn").isChecked()
        assert _spin(widget, "obliqueSpinBox").value() == 0.0

    def test_default_on_shows_auto_and_a_checked_box(self, qtbot: QtBot):
        filter_ = Filter()
        filter_.auto_oblique_default = True
        options = OptionsWidget(filter_)
        qtbot.addWidget(options)
        options.set_current_page(_page())

        assert _check(options).isChecked()
        assert _button(options, "obliqueAutoBtn").isChecked()

    def test_stored_values_are_shown_when_a_page_is_selected(
        self, widget: OptionsWidget, filter_: Filter
    ):
        other = _page("other")
        filter_.settings.set_params(
            other, Params(oblique_angle_deg=-2.5, oblique_mode=MANUAL)
        )

        widget.set_current_page(other)

        assert _spin(widget, "obliqueSpinBox").value() == pytest.approx(-2.5)
        assert _button(widget, "obliqueManualBtn").isChecked()


class TestObliqueEditing:
    """Editing the oblique stores it, flips it to manual and says so."""

    def test_typing_an_angle_stores_a_manual_oblique(
        self, qtbot: QtBot, widget: OptionsWidget, filter_: Filter
    ):
        with qtbot.waitSignal(widget.oblique_changed) as changed:
            _spin(widget, "obliqueSpinBox").setValue(2.0)

        stored = filter_.get_params(_page())
        assert stored.oblique_angle_deg == 2.0
        assert stored.oblique_mode is MANUAL
        assert changed.args == [_page(), 2.0, True]

    def test_editing_an_automatic_oblique_switches_the_mode_button_to_manual(
        self, widget: OptionsWidget, filter_: Filter
    ):
        filter_.redetect_oblique(_page())
        widget.refresh()
        assert _button(widget, "obliqueAutoBtn").isChecked()

        _spin(widget, "obliqueSpinBox").setValue(1.0)

        assert _button(widget, "obliqueManualBtn").isChecked()

    def test_editing_the_oblique_leaves_the_rotation_alone(
        self, widget: OptionsWidget, filter_: Filter
    ):
        filter_.settings.set_params(_page(), Params(deskew_angle_deg=3.0, mode=AUTO))

        _spin(widget, "obliqueSpinBox").setValue(1.0)

        stored = filter_.get_params(_page())
        assert stored.deskew_angle_deg == 3.0
        assert stored.mode is AUTO

    def test_the_oblique_auto_button_asks_for_detection_and_keeps_the_rotation(
        self, qtbot: QtBot, widget: OptionsWidget, filter_: Filter
    ):
        filter_.set_manual_angle(_page(), 2.0)
        filter_.set_manual_oblique(_page(), 3.0)
        widget.refresh()

        with qtbot.waitSignal(widget.oblique_changed) as changed:
            _button(widget, "obliqueAutoBtn").click()

        stored = filter_.get_params(_page())
        assert stored.oblique_mode is AUTO and stored.redetect_oblique
        assert stored.mode is MANUAL and stored.deskew_angle_deg == 2.0
        assert changed.args[2] is False
        assert filter_.needs_detection(_page(), DEPENDENCIES)

    def test_the_oblique_manual_button_keeps_the_current_angle(
        self, qtbot: QtBot, widget: OptionsWidget, filter_: Filter
    ):
        filter_.settings.set_params(
            _page(), Params(oblique_angle_deg=1.7, oblique_mode=AUTO)
        )
        widget.refresh()

        with qtbot.waitSignal(widget.oblique_changed) as changed:
            _button(widget, "obliqueManualBtn").click()

        stored = filter_.get_params(_page())
        assert stored.oblique_mode is MANUAL
        assert stored.oblique_angle_deg == 1.7
        assert changed.args == [_page(), 1.7, True]

    def test_external_updates_do_not_loop_back(
        self, widget: OptionsWidget, filter_: Filter
    ):
        emitted: list[tuple] = []
        widget.oblique_changed.connect(lambda *args: emitted.append(args))

        widget.set_oblique(1.2, is_manual=True)

        assert _spin(widget, "obliqueSpinBox").value() == pytest.approx(1.2)
        assert emitted == []


class TestAutomaticObliqueDefault:
    """The check box is the default for new pages and acts on an automatic page."""

    def test_checking_it_changes_the_default_and_asks_the_page_to_detect(
        self, qtbot: QtBot, widget: OptionsWidget, filter_: Filter
    ):
        with qtbot.waitSignals(
            [widget.auto_oblique_default_changed, widget.oblique_changed]
        ):
            _check(widget).setChecked(True)

        assert filter_.auto_oblique_default
        stored = filter_.get_params(_page())
        assert stored.oblique_mode is AUTO and stored.redetect_oblique

    def test_unchecking_it_turns_the_oblique_off_for_an_automatic_page(
        self, widget: OptionsWidget, filter_: Filter
    ):
        filter_.auto_oblique_default = True
        filter_.redetect_oblique(_page())
        widget.refresh()
        assert _check(widget).isChecked()

        _check(widget).setChecked(False)

        stored = filter_.get_params(_page())
        assert not filter_.auto_oblique_default
        assert stored.oblique_mode is MANUAL
        assert stored.oblique_angle_deg == 0.0
        assert _button(widget, "obliqueManualBtn").isChecked()

    def test_a_manual_rotation_page_keeps_its_oblique_when_the_default_changes(
        self, widget: OptionsWidget, filter_: Filter
    ):
        filter_.settings.set_params(
            _page(),
            Params(mode=MANUAL, oblique_angle_deg=2.0, oblique_mode=MANUAL),
        )
        widget.refresh()

        _check(widget).setChecked(True)

        stored = filter_.get_params(_page())
        assert stored.oblique_angle_deg == 2.0
        assert stored.oblique_mode is MANUAL
        assert filter_.auto_oblique_default


class TestUpstreamUiReports:
    """The widget side of upstream issues 145, 156, 169 and 170."""

    def test_issue_156_the_oblique_mode_is_not_auto_while_the_box_is_unchecked(
        self, widget: OptionsWidget
    ):
        widget.refresh()

        assert not _check(widget).isChecked()
        assert not _button(widget, "obliqueAutoBtn").isChecked()

    def test_issue_169_auto_after_manual_stays_auto(
        self, qtbot: QtBot, widget: OptionsWidget, filter_: Filter
    ):
        _spin(widget, "angleSpinBox").setValue(1.0)
        assert _button(widget, "manualBtn").isChecked()

        with qtbot.waitSignal(widget.angle_changed) as changed:
            _button(widget, "autoBtn").click()
        # Detection runs after the click and refreshes the options.
        filter_.settings.store_detection(_page(), DEPENDENCIES, rotation=-0.5)
        widget.refresh()

        assert changed.args[2] is False
        assert filter_.get_params(_page()).mode is AUTO
        assert _button(widget, "autoBtn").isChecked()
        assert not _button(widget, "manualBtn").isChecked()
        assert _spin(widget, "angleSpinBox").value() == pytest.approx(-0.5)

    def test_issue_170_auto_deskew_does_not_touch_the_oblique(
        self, widget: OptionsWidget, filter_: Filter
    ):
        _check(widget).setChecked(False)
        assert filter_.get_params(_page()).oblique_mode is MANUAL

        _button(widget, "manualBtn").click()
        _button(widget, "autoBtn").click()

        stored = filter_.get_params(_page())
        assert stored.oblique_mode is MANUAL
        assert stored.oblique_angle_deg == 0.0
        assert _button(widget, "obliqueManualBtn").isChecked()

    def test_issue_170_auto_deskew_keeps_a_manual_oblique_angle(
        self, widget: OptionsWidget, filter_: Filter
    ):
        _spin(widget, "obliqueSpinBox").setValue(2.0)

        _button(widget, "autoBtn").click()

        stored = filter_.get_params(_page())
        assert stored.oblique_angle_deg == 2.0
        assert stored.oblique_mode is MANUAL
        assert _spin(widget, "obliqueSpinBox").value() == 2.0


@pytest.fixture
def make_view(qtbot: QtBot):
    def create(size: tuple[int, int] = (640, 480)) -> DeskewImageView:
        view = DeskewImageView()
        qtbot.addWidget(view)
        view.resize(*size)
        view.show()
        qtbot.waitExposed(view)
        page = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 255, dtype=np.uint8)
        view.set_image(
            page,
            transformation=ImageTransformation.from_size(
                IMAGE_WIDTH, IMAGE_HEIGHT, Dpi.uniform(300)
            ),
        )
        return view

    return create


def _point(point: QtCore.QPointF) -> QtCore.QPoint:
    return QtCore.QPoint(round(point.x()), round(point.y()))


def _drag(
    qtbot: QtBot, view: DeskewImageView, start: QtCore.QPointF, end: QtCore.QPointF
):
    viewport = view.viewport()
    assert viewport is not None
    left = QtCore.Qt.MouseButton.LeftButton
    qtbot.mousePress(viewport, left, pos=_point(start))
    for step in range(1, 5):
        qtbot.mouseMove(viewport, _point(start + (end - start) * step / 4))
    qtbot.mouseRelease(viewport, left, pos=_point(end))


class TestObliqueLever:
    """Two handles on arcs at the top and bottom of a circle at the right edge."""

    def test_at_zero_the_handles_are_above_each_other_on_the_vertical_axis(
        self, make_view
    ):
        view = make_view()

        top, bottom = view.oblique_handles()
        square = view.oblique_arc_square()

        assert top.x() == pytest.approx(bottom.x())
        assert top.x() == pytest.approx(square.center().x())
        assert top.y() < bottom.y()
        assert top.y() == pytest.approx(square.top())
        assert bottom.y() == pytest.approx(square.bottom())

    def test_the_handles_swing_with_the_angle_in_opposite_directions(self, make_view):
        view = make_view()
        view.set_oblique_angle(3.0)

        top, bottom = view.oblique_handles()
        square = view.oblique_arc_square()
        radius = square.height() / 2

        assert top.x() - square.center().x() == pytest.approx(
            radius * math.sin(math.radians(3.0))
        )
        assert bottom.x() - square.center().x() == pytest.approx(
            -radius * math.sin(math.radians(3.0))
        )

    @pytest.mark.parametrize("size", VIEW_SIZES)
    def test_issue_145_the_lever_stays_inside_the_window_whatever_its_size(
        self, make_view, size: tuple[int, int]
    ):
        view = make_view(size)
        viewport = view.viewport()
        assert viewport is not None
        window = QtCore.QRectF(viewport.rect())

        for angle in (-MAX_OBLIQUE_DEG, 0.0, MAX_OBLIQUE_DEG):
            view.set_oblique_angle(angle)
            for handle in view.oblique_handles():
                assert window.contains(handle)
        square = view.oblique_arc_square()
        assert window.contains(square.topRight() + QtCore.QPointF(-1, 1))

    @pytest.mark.parametrize("size", VIEW_SIZES)
    def test_issue_145_the_lever_shares_its_area_with_the_rotation_arcs(
        self, make_view, size: tuple[int, int]
    ):
        view = make_view(size)
        oblique = view.oblique_arc_square()
        rotation = view.rotation_arc_square()

        margin = DeskewImageView.HANDLE_RADIUS + 20
        viewport = view.viewport()
        assert viewport is not None
        assert oblique.center().y() == pytest.approx(rotation.center().y())
        assert oblique.right() == pytest.approx(viewport.width() - margin)
        assert oblique.top() >= margin - 1e-6
        assert oblique.bottom() <= viewport.height() - margin + 1e-6

    def test_issue_145_resizing_the_window_moves_the_lever_with_it(
        self, qtbot: QtBot, make_view
    ):
        view = make_view((1900, 1000))
        large = view.oblique_arc_square()

        view.resize(640, 480)
        qtbot.waitUntil(lambda: view.viewport().width() < 700)
        small = view.oblique_arc_square()

        assert small.right() < large.right()
        assert small.height() < large.height()

    def test_a_window_too_small_for_the_lever_has_none(self, make_view):
        view = make_view((50, 50))

        assert view.oblique_arc_square().isEmpty()
        assert view.oblique_handles()[0].isNull()

    def test_dragging_the_top_handle_right_sets_a_positive_angle(
        self, qtbot: QtBot, make_view
    ):
        view = make_view()
        top, _bottom = view.oblique_handles()
        radius = view.oblique_arc_square().height() / 2
        target = QtCore.QPointF(top.x() + radius * math.sin(math.radians(2.0)), top.y())

        with qtbot.waitSignal(view.manual_oblique_angle_set) as dragged:
            _drag(qtbot, view, top, target)

        assert dragged.args[0] == pytest.approx(2.0, abs=0.5)
        assert view.oblique_angle == pytest.approx(dragged.args[0])

    def test_dragging_the_bottom_handle_right_sets_a_negative_angle(
        self, qtbot: QtBot, make_view
    ):
        view = make_view()
        _top, bottom = view.oblique_handles()
        radius = view.oblique_arc_square().height() / 2
        target = QtCore.QPointF(
            bottom.x() + radius * math.sin(math.radians(2.0)), bottom.y()
        )

        with qtbot.waitSignal(view.manual_oblique_angle_set) as dragged:
            _drag(qtbot, view, bottom, target)

        assert dragged.args[0] == pytest.approx(-2.0, abs=0.5)

    def test_dragging_far_past_the_arc_is_limited_to_five_degrees(
        self, qtbot: QtBot, make_view
    ):
        view = make_view()
        top, _bottom = view.oblique_handles()
        square = view.oblique_arc_square()
        target = QtCore.QPointF(square.right(), top.y())

        with qtbot.waitSignal(view.manual_oblique_angle_set) as dragged:
            _drag(qtbot, view, top, target)

        assert dragged.args[0] == pytest.approx(MAX_OBLIQUE_DEG)
        assert view.oblique_angle == pytest.approx(MAX_OBLIQUE_DEG)

    def test_a_drag_of_a_rotation_handle_does_not_report_an_oblique(
        self, qtbot: QtBot, make_view
    ):
        view = make_view()
        reported: list[float] = []
        view.manual_oblique_angle_set.connect(reported.append)
        _left, right = view.rotation_handles()

        with qtbot.waitSignal(view.manual_deskew_angle_set):
            _drag(qtbot, view, right, right + QtCore.QPointF(0, 12))

        assert reported == []

    def test_the_page_shears_with_the_oblique_angle(self, make_view):
        view = make_view()
        before = view.image_to_virtual

        view.set_oblique_angle(4.0)

        assert view.image_to_virtual != before
        expected = ImageTransformation.from_size(
            IMAGE_WIDTH, IMAGE_HEIGHT, Dpi.uniform(300)
        )
        expected.set_post_oblique(4.0)
        mapped = view.image_to_virtual.map(QtCore.QPointF(0, IMAGE_HEIGHT))
        assert (mapped.x(), mapped.y()) == pytest.approx(
            expected.transform_point(0, IMAGE_HEIGHT), abs=1e-6
        )

    def test_the_oblique_survives_a_new_base_transformation(self, make_view):
        view = make_view()
        view.set_oblique_angle(2.0)
        view.set_deskew_angle(1.0)

        view.set_transformation(
            ImageTransformation.from_size(IMAGE_WIDTH, IMAGE_HEIGHT, Dpi.uniform(300))
        )

        expected = ImageTransformation.from_size(
            IMAGE_WIDTH, IMAGE_HEIGHT, Dpi.uniform(300)
        )
        expected.set_post_rotation(1.0)
        expected.set_post_oblique(2.0)
        mapped = view.image_to_virtual.map(QtCore.QPointF(300, 400))
        assert (mapped.x(), mapped.y()) == pytest.approx(
            expected.transform_point(300, 400), abs=1e-6
        )

    def test_setting_the_oblique_from_outside_does_not_report_it(self, make_view):
        view = make_view()
        reported: list[float] = []
        view.manual_oblique_angle_set.connect(reported.append)

        view.set_oblique_angle(1.5)

        assert reported == []
        assert view.oblique_angle == 1.5

    def test_the_oblique_is_limited_to_five_degrees(self, make_view):
        view = make_view()

        view.set_oblique_angle(30.0)

        assert view.oblique_angle == MAX_OBLIQUE_DEG

    def test_the_view_paints_with_the_lever(self, make_view):
        view = make_view()
        view.set_oblique_angle(2.0)

        image = view.viewport().grab().toImage()

        assert not image.isNull()


class TestApplyToDeskewOptions:
    """The Apply To dialog offers the two parts of the Deskew settings."""

    def test_the_dialog_offers_the_upstream_checkboxes_all_checked(self, qtbot: QtBot):
        page = _page()
        dialog = ApplyToDialog(page, [page], title=DESKEW_TITLE, options=DESKEW_OPTIONS)
        qtbot.addWidget(dialog)

        boxes = {box.text(): box for box in dialog.findChildren(QtWidgets.QCheckBox)}

        assert DESKEW_TITLE == "Apply Deskew"
        assert set(boxes) == {"Deskew angle and mode", "Oblique angle and mode"}
        assert dialog.option_checked("deskew")
        assert dialog.option_checked("oblique")

    def test_a_part_can_be_unchecked(self, qtbot: QtBot):
        page = _page()
        dialog = ApplyToDialog(page, [page], options=DESKEW_OPTIONS)
        qtbot.addWidget(dialog)

        next(
            box
            for box in dialog.findChildren(QtWidgets.QCheckBox)
            if box.text() == "Oblique angle and mode"
        ).setChecked(False)

        assert dialog.option_checked("deskew")
        assert not dialog.option_checked("oblique")


@pytest.fixture
def config(tmp_path: pathlib.Path) -> Iterator[pathlib.Path]:
    ApplicationSettings.set_config_path(tmp_path / "settings.json")
    UnitsProvider.reset_instance()
    yield tmp_path
    ApplicationSettings.reset_instance()
    UnitsProvider.reset_instance()


class TestDefaultParametersDialog:
    """The "Automatic oblique correction" default."""

    def _dialog(self, qtbot: QtBot, config: pathlib.Path) -> DefaultParamsDialog:
        dialog = DefaultParamsDialog(
            settings=ApplicationSettings(), store=ProfileStore(config / "profiles")
        )
        qtbot.addWidget(dialog)
        return dialog

    def test_the_checkbox_has_the_upstream_text_and_starts_unchecked(
        self, qtbot: QtBot, config: pathlib.Path
    ):
        dialog = self._dialog(qtbot, config)

        box = _find(dialog, QtWidgets.QCheckBox, "deskewAutoObliqueCB")

        assert box.text() == "Automatic oblique correction"
        assert not box.isChecked()

    def test_the_profile_value_is_shown(self, qtbot: QtBot, config: pathlib.Path):
        dialog = self._dialog(qtbot, config)

        dialog._load_deskew(DeskewDefaults(auto_oblique=True))

        assert _find(dialog, QtWidgets.QCheckBox, "deskewAutoObliqueCB").isChecked()

    def test_the_form_builds_the_default_back(self, qtbot: QtBot, config: pathlib.Path):
        dialog = self._dialog(qtbot, config)
        box = _find(dialog, QtWidgets.QCheckBox, "deskewAutoObliqueCB")

        box.setChecked(True)

        assert dialog._build_params().deskew.auto_oblique is True
        box.setChecked(False)
        assert dialog._build_params().deskew.auto_oblique is False
        assert isinstance(dialog._build_params(), DefaultParams)
