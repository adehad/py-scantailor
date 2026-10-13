"""The Page Layout view: content box drags, content moves, shared links, guides."""

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.image_views.page_layout import PageLayoutImageView
from scantailor.app.ui.image_views.page_layout_state import PageLayoutState
from scantailor.app.ui.options_widgets import page_layout as page_layout_options
from scantailor.core.content_mask import ContentMask
from scantailor.core.models import Dpi, Margins
from scantailor.core.transformation import ImageTransformation
from scantailor.filters.page_layout.filter import Filter

IMAGE_WIDTH = 600
IMAGE_HEIGHT = 800
MM_X, MM_Y = 0.1, 0.2
MARGIN = 60  # Virtual pixels; wider than the grab distance at every zoom used here
SOFT = 30
CONTROL = QtCore.Qt.KeyboardModifier.ControlModifier
SHIFT = QtCore.Qt.KeyboardModifier.ShiftModifier
ALT = QtCore.Qt.KeyboardModifier.AltModifier
INNER = QtCore.QRectF(200, 300, 200, 200)


@pytest.fixture
def view(qtbot: QtBot) -> PageLayoutImageView:
    widget = PageLayoutImageView()
    qtbot.addWidget(widget)
    widget.resize(300, 400)
    widget.show()
    qtbot.waitExposed(widget)
    widget.set_image(
        np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 255, dtype=np.uint8),
        transformation=ImageTransformation.from_size(
            IMAGE_WIDTH, IMAGE_HEIGHT, Dpi.uniform(300)
        ),
    )
    widget.set_margin_links(left_right=False, top_bottom=False)
    widget.set_mm_per_pixel(MM_X, MM_Y)
    middle = INNER.adjusted(-MARGIN, -MARGIN, MARGIN, MARGIN)
    widget.set_rectangles(INNER, middle, middle.adjusted(-SOFT, -SOFT, SOFT, SOFT))
    return widget


def _hover(
    view: PageLayoutImageView,
    position: QtCore.QPointF,
    modifier: QtCore.Qt.KeyboardModifier,
) -> None:
    """Move the pointer with exactly the given keys held."""
    viewport = view.viewport()
    assert viewport is not None
    QtWidgets.QApplication.sendEvent(
        viewport,
        QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseMove,
            position,
            viewport.mapToGlobal(position),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.MouseButton.NoButton,
            modifier,
        ),
    )


def _zoom_in_on_the_content(view: PageLayoutImageView) -> None:
    """Show the page at full size with its centre in the middle of the window."""
    view.set_zoom(1.0, QtCore.QPointF(150, 200))


def _drag(
    qtbot: QtBot,
    view: PageLayoutImageView,
    start: QtCore.QPointF,
    end: QtCore.QPointF,
    modifier: QtCore.Qt.KeyboardModifier = QtCore.Qt.KeyboardModifier.NoModifier,
) -> None:
    viewport = view.viewport()
    assert viewport is not None
    left = QtCore.Qt.MouseButton.LeftButton
    qtbot.mousePress(viewport, left, modifier, pos=start.toPoint())
    for step in range(1, 5):
        position = start + (end - start) * step / 4
        # qtbot.mouseMove cannot carry the keys held, which the view reads.
        QtWidgets.QApplication.sendEvent(
            viewport,
            QtGui.QMouseEvent(
                QtCore.QEvent.Type.MouseMove,
                position,
                viewport.mapToGlobal(position),
                QtCore.Qt.MouseButton.NoButton,
                left,
                modifier,
            ),
        )
    qtbot.mouseRelease(viewport, left, modifier, pos=end.toPoint())


class TestDraggingTheContentBox:
    def test_dragging_a_content_box_side_inwards_grows_that_margin(self, qtbot, view):
        _zoom_in_on_the_content(view)
        edge = view.virtual_to_widget_point(QtCore.QPointF(INNER.left(), 400))

        with qtbot.waitSignal(view.margins_set_locally) as dragged:
            _drag(qtbot, view, edge, edge + QtCore.QPointF(30, 0))

        margins: Margins = dragged.args[0]
        assert margins.left == pytest.approx((MARGIN + 30) * MM_X)
        assert margins.right == pytest.approx(MARGIN * MM_X)
        assert margins.top == pytest.approx(MARGIN * MM_Y)

    def test_dragging_a_content_box_side_outwards_shrinks_the_margin(self, qtbot, view):
        _zoom_in_on_the_content(view)
        edge = view.virtual_to_widget_point(QtCore.QPointF(INNER.right(), 400))

        with qtbot.waitSignal(view.margins_set_locally) as dragged:
            _drag(qtbot, view, edge, edge + QtCore.QPointF(20, 0))

        assert dragged.args[0].right == pytest.approx((MARGIN - 20) * MM_X)

    def test_the_side_stays_under_the_pointer_as_the_page_moves(self, qtbot, view):
        _zoom_in_on_the_content(view)
        edge = view.virtual_to_widget_point(QtCore.QPointF(INNER.left(), 400))
        target = edge + QtCore.QPointF(30, 0)

        with qtbot.waitSignal(view.margins_set_locally):
            _drag(qtbot, view, edge, target)

        after = view.virtual_to_widget_point(QtCore.QPointF(INNER.left(), 400))
        assert after.x() == pytest.approx(target.x(), abs=1.0)

    def test_the_margin_cannot_become_negative(self, qtbot, view):
        _zoom_in_on_the_content(view)
        edge = view.virtual_to_widget_point(QtCore.QPointF(INNER.left(), 400))

        with qtbot.waitSignal(view.margins_set_locally) as dragged:
            _drag(qtbot, view, edge, edge + QtCore.QPointF(-200, 0))

        assert dragged.args[0].left == pytest.approx(0)

    def test_a_corner_of_the_content_box_moves_both_of_its_margins(self, qtbot, view):
        _zoom_in_on_the_content(view)
        corner = view.virtual_to_widget_point(INNER.topLeft())

        with qtbot.waitSignal(view.margins_set_locally) as dragged:
            _drag(qtbot, view, corner, corner + QtCore.QPointF(10, 12))

        margins = dragged.args[0]
        assert margins.left == pytest.approx((MARGIN + 10) * MM_X)
        assert margins.top == pytest.approx((MARGIN + 12) * MM_Y)

    def test_a_linked_pair_changes_together(self, qtbot, view):
        view.set_margin_links(left_right=True, top_bottom=False)
        _zoom_in_on_the_content(view)
        edge = view.virtual_to_widget_point(QtCore.QPointF(INNER.left(), 400))

        with qtbot.waitSignal(view.margins_set_locally) as dragged:
            _drag(qtbot, view, edge, edge + QtCore.QPointF(15, 0))

        margins = dragged.args[0]
        assert margins.left == pytest.approx((MARGIN + 15) * MM_X)
        assert margins.right == pytest.approx((MARGIN + 15) * MM_X)
        assert margins.top == pytest.approx(MARGIN * MM_Y)

    def test_the_content_box_wins_over_the_margin_rectangle_where_they_meet(
        self, qtbot
    ):
        widget = PageLayoutImageView()
        qtbot.addWidget(widget)
        widget.resize(300, 400)
        widget.show()
        qtbot.waitExposed(widget)
        widget.set_image(
            np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 255, dtype=np.uint8),
            transformation=ImageTransformation.from_size(
                IMAGE_WIDTH, IMAGE_HEIGHT, Dpi.uniform(300)
            ),
        )
        widget.set_mm_per_pixel(MM_X, MM_Y)
        widget.set_rectangles(INNER, INNER, INNER.adjusted(-30, -30, 30, 30))
        edge = widget.virtual_to_widget_point(QtCore.QPointF(INNER.left(), 400))

        with qtbot.waitSignal(widget.margins_set_locally) as dragged:
            _drag(qtbot, widget, edge, edge + QtCore.QPointF(10, 0))

        assert dragged.args[0].left == pytest.approx(10 / widget.zoom_level * MM_X)

    def test_a_page_without_content_has_no_content_box_handles(self, qtbot, view):
        middle = QtCore.QRectF(100, 100, 400, 400)
        view.set_rectangles(QtCore.QRectF(), middle, middle)
        reported: list[Margins] = []
        view.margins_set_locally.connect(reported.append)
        inside = view.virtual_to_widget_point(QtCore.QPointF(300, 300))

        _drag(qtbot, view, inside, inside + QtCore.QPointF(10, 10), modifier=CONTROL)

        assert reported == []

    def test_a_click_without_moving_reports_nothing(self, qtbot, view):
        edge = view.virtual_to_widget_point(QtCore.QPointF(INNER.left(), 400))
        reported: list[Margins] = []
        view.margins_set_locally.connect(reported.append)

        _drag(qtbot, view, edge, edge)

        assert reported == []


class TestMovingTheContent:
    def _inside(self, view: PageLayoutImageView) -> QtCore.QPointF:
        return view.virtual_to_widget_point(INNER.center())

    def test_dragging_the_body_with_control_moves_the_content_vertically(
        self, qtbot, view
    ):
        _zoom_in_on_the_content(view)
        start = self._inside(view)

        with qtbot.waitSignal(view.margins_set_locally) as dragged:
            _drag(qtbot, view, start, start + QtCore.QPointF(25, 20), modifier=CONTROL)

        margins = dragged.args[0]
        assert margins.top == pytest.approx((MARGIN + 20) * MM_Y)
        assert margins.bottom == pytest.approx((MARGIN - 20) * MM_Y)
        assert margins.left == pytest.approx(MARGIN * MM_X)
        assert margins.right == pytest.approx(MARGIN * MM_X)

    def test_dragging_the_body_with_shift_moves_the_content_horizontally(
        self, qtbot, view
    ):
        _zoom_in_on_the_content(view)
        start = self._inside(view)

        with qtbot.waitSignal(view.margins_set_locally) as dragged:
            _drag(qtbot, view, start, start + QtCore.QPointF(25, 20), modifier=SHIFT)

        margins = dragged.args[0]
        assert margins.left == pytest.approx((MARGIN + 25) * MM_X)
        assert margins.right == pytest.approx((MARGIN - 25) * MM_X)
        assert margins.top == pytest.approx(MARGIN * MM_Y)

    def test_both_keys_together_move_the_content_freely(self, qtbot, view):
        _zoom_in_on_the_content(view)
        start = self._inside(view)

        with qtbot.waitSignal(view.margins_set_locally) as dragged:
            _drag(
                qtbot,
                view,
                start,
                start + QtCore.QPointF(-10, 15),
                modifier=CONTROL | SHIFT,
            )

        margins = dragged.args[0]
        assert margins.left == pytest.approx((MARGIN - 10) * MM_X)
        assert margins.top == pytest.approx((MARGIN + 15) * MM_Y)

    def test_without_a_key_the_body_does_not_move_the_content(self, qtbot, view):
        _zoom_in_on_the_content(view)
        start = self._inside(view)
        reported: list[Margins] = []
        view.margins_set_locally.connect(reported.append)

        _drag(qtbot, view, start, start + QtCore.QPointF(25, 20))

        assert reported == []

    def test_the_content_cannot_leave_the_hard_margins_rectangle(self, qtbot, view):
        _zoom_in_on_the_content(view)
        start = self._inside(view)

        with qtbot.waitSignal(view.margins_set_locally) as dragged:
            _drag(qtbot, view, start, start + QtCore.QPointF(500, 0), modifier=SHIFT)

        margins = dragged.args[0]
        assert margins.right == pytest.approx(0)
        assert margins.left > MARGIN * MM_X

    def test_the_content_moves_with_the_pointer_on_screen(self, qtbot, view):
        _zoom_in_on_the_content(view)
        start = self._inside(view)
        end = start + QtCore.QPointF(20, 0)

        with qtbot.waitSignal(view.margins_set_locally):
            _drag(qtbot, view, start, end, modifier=SHIFT)

        assert view.virtual_to_widget_point(INNER.center()).x() == pytest.approx(
            end.x(), abs=1.0
        )

    def test_the_hint_over_the_content_needs_a_key(self, view):
        inside = self._inside(view)

        _hover(view, inside, QtCore.Qt.KeyboardModifier.NoModifier)
        assert view.status_tip_text == view.default_status_tip

        _hover(view, inside, SHIFT)
        assert (
            view.status_tip_text == "Hold left mouse button to drag the page content."
        )

    def test_pressing_a_key_over_the_content_changes_the_hint_at_once(
        self, qtbot, view
    ):
        _hover(view, self._inside(view), QtCore.Qt.KeyboardModifier.NoModifier)
        assert view.status_tip_text == view.default_status_tip

        qtbot.keyPress(view, QtCore.Qt.Key.Key_Control)
        try:
            assert (
                view.status_tip_text
                == "Hold left mouse button to drag the page content."
            )
        finally:
            qtbot.keyRelease(view, QtCore.Qt.Key.Key_Control)
        assert view.status_tip_text == view.default_status_tip


class TestSharedState:
    def test_the_view_and_the_options_use_one_link_state(self, qtbot, view):
        options = page_layout_options.OptionsWidget(Filter())
        qtbot.addWidget(options)
        view.set_state(options.state)

        options.state.top_bottom_linked = True

        link = options.findChild(QtCore.QObject, "topBottomLink")
        assert link is not None and link.isChecked()
        assert view.state.top_bottom_linked

    def test_pressing_the_link_button_links_the_view(self, qtbot, view):
        options = page_layout_options.OptionsWidget(Filter())
        qtbot.addWidget(options)
        options.state.left_right_linked = False
        view.set_state(options.state)
        link = options.findChild(QtCore.QObject, "leftRightLink")
        assert link is not None

        link.click()

        assert view.state.left_right_linked

    def test_switching_a_link_on_makes_unequal_margins_equal_at_once(self, qtbot, view):
        middle = INNER.adjusted(-80, -MARGIN, 20, MARGIN)
        view.set_rectangles(INNER, middle, middle)

        with qtbot.waitSignal(view.margins_set_locally) as linked:
            view.state.left_right_linked = True

        margins = linked.args[0]
        assert margins.left == pytest.approx(20 * MM_X)
        assert margins.right == pytest.approx(20 * MM_X)
        assert margins.top == pytest.approx(MARGIN * MM_Y)

    def test_switching_a_link_on_leaves_equal_margins_alone(self, view):
        reported: list[Margins] = []
        view.margins_set_locally.connect(reported.append)

        view.state.left_right_linked = True

        assert reported == []

    def test_a_pair_that_differs_on_the_next_page_is_unlinked(self, qtbot):
        widget = page_layout_options.OptionsWidget(Filter())
        qtbot.addWidget(widget)
        widget.state.left_right_linked = True
        widget.state.top_bottom_linked = True
        from scantailor.core.models import ImageId, PageId

        page = PageId(image_id=ImageId(file_path="a.png"))
        widget._filter.set_params(
            page,
            widget._filter.get_params(page).with_margins(
                Margins(left=10, right=10, top=5, bottom=8)
            ),
        )

        widget.set_current_page(page)

        assert widget.state.left_right_linked
        assert not widget.state.top_bottom_linked

    def test_a_view_given_new_state_stops_following_the_old_one(self, qtbot, view):
        old = view.state
        view.set_state(PageLayoutState())
        reported: list[Margins] = []
        view.margins_set_locally.connect(reported.append)

        old.left_right_linked = True

        assert reported == []


class TestLinkPersistence:
    def test_both_pairs_start_linked_when_the_user_never_chose(self):
        state = PageLayoutState()

        assert state.left_right_linked
        assert state.top_bottom_linked

    def test_a_pair_the_user_unlinked_stays_unlinked_in_the_next_session(self, qtbot):
        options = page_layout_options.OptionsWidget(Filter())
        qtbot.addWidget(options)
        link = options.findChild(QtCore.QObject, "leftRightLink")
        assert link is not None

        link.click()

        assert not options.state.left_right_linked
        later = PageLayoutState()
        assert not later.left_right_linked
        assert later.top_bottom_linked

    def test_each_pair_is_stored_under_its_upstream_key(self, qtbot):
        options = page_layout_options.OptionsWidget(Filter())
        qtbot.addWidget(options)
        top_bottom = options.findChild(QtCore.QObject, "topBottomLink")
        assert top_bottom is not None

        top_bottom.click()

        settings = QtCore.QSettings()
        assert settings.value("margins/topBottomLinked", type=bool) is False
        assert not settings.contains("margins/leftRightLinked")

    def test_a_pair_released_by_unequal_margins_is_not_stored_as_unlinked(self, qtbot):
        from scantailor.core.models import ImageId, PageId

        widget = page_layout_options.OptionsWidget(Filter())
        qtbot.addWidget(widget)
        page = PageId(image_id=ImageId(file_path="a.png"))
        widget._filter.set_params(
            page,
            widget._filter.get_params(page).with_margins(
                Margins(left=10, right=12, top=5, bottom=5)
            ),
        )

        widget.set_current_page(page)

        assert not widget.state.left_right_linked
        assert PageLayoutState().left_right_linked


class TestGuides:
    def test_a_guide_is_kept_in_millimetres_from_the_centre_of_the_outer_rectangle(
        self, view
    ):
        centre = view.virtual_to_widget_point(QtCore.QPointF(300, 400))
        view._context_pos = centre + QtCore.QPointF(20, 0)

        view._add_guide(horizontal=False)

        (guide,) = view.state.guides
        assert not guide.is_horizontal
        assert guide.position_mm == pytest.approx(20 / view.zoom_level * MM_X)

    def test_a_guide_survives_a_rebuilt_view(self, qtbot, view):
        view.state.add_guide(is_horizontal=True, position_mm=2.0)
        other = PageLayoutImageView(state=view.state)
        qtbot.addWidget(other)

        assert [g.position_mm for g in other.state.guides] == [2.0]

    def test_a_guide_is_dragged_with_control_and_alt(self, qtbot, view):
        guide = view.state.add_guide(is_horizontal=False, position_mm=1.0)
        at = view.virtual_to_widget_point(QtCore.QPointF(300 + 1.0 / MM_X, 400))

        _drag(qtbot, view, at, at + QtCore.QPointF(15, 0), modifier=CONTROL | ALT)

        assert guide.position_mm == pytest.approx(1.0 + 15 / view.zoom_level * MM_X)

    def test_a_guide_does_not_move_without_the_keys(self, qtbot, view):
        guide = view.state.add_guide(is_horizontal=False, position_mm=1.0)
        at = view.virtual_to_widget_point(QtCore.QPointF(300 + 1.0 / MM_X, 400))

        _drag(qtbot, view, at, at + QtCore.QPointF(15, 0))

        assert guide.position_mm == pytest.approx(1.0)

    def test_a_guide_stays_within_the_outer_rectangle(self, qtbot, view):
        guide = view.state.add_guide(is_horizontal=False, position_mm=1.0)
        at = view.virtual_to_widget_point(QtCore.QPointF(300 + 1.0 / MM_X, 400))

        _drag(qtbot, view, at, at + QtCore.QPointF(5000, 0), modifier=CONTROL | ALT)

        limit = view._outer_rect.width() / 2 * MM_X
        assert guide.position_mm == pytest.approx(limit)

    def test_a_page_that_is_not_aligned_has_no_guides_to_pick(self, qtbot, view):
        guide = view.state.add_guide(is_horizontal=False, position_mm=1.0)
        view.set_aligned(False)
        at = view.virtual_to_widget_point(QtCore.QPointF(300 + 1.0 / MM_X, 400))

        _drag(qtbot, view, at, at + QtCore.QPointF(15, 0), modifier=CONTROL | ALT)

        assert guide.position_mm == pytest.approx(1.0)

    def test_hovering_a_guide_says_how_to_drag_it(self, qtbot, view):
        view.state.add_guide(is_horizontal=False, position_mm=1.0)
        at = view.virtual_to_widget_point(QtCore.QPointF(300 + 1.0 / MM_X, 400))

        qtbot.mouseMove(view.viewport(), at.toPoint())

        assert view.status_tip_text == "Drag the guide."

    def test_the_menu_offers_what_the_original_does(self, qtbot, view):
        view.state.add_guide(is_horizontal=True, position_mm=0.0)
        menu_texts = {
            view._add_horizontal_guide_action.text(),
            view._add_vertical_guide_action.text(),
            view._remove_all_guides_action.text(),
            view._remove_guide_action.text(),
            view._show_middle_rect_action.text(),
        }

        assert menu_texts == {
            "Add a horizontal guide",
            "Add a vertical guide",
            "Remove all the guides",
            "Remove this guide",
            "Show hard margins rectangle",
        }

    def test_hiding_the_hard_margins_rectangle_makes_its_sides_untouchable(
        self, qtbot, view
    ):
        _zoom_in_on_the_content(view)
        view.state.middle_rect_shown = False
        reported: list[Margins] = []
        view.margins_set_locally.connect(reported.append)
        side = view.virtual_to_widget_point(QtCore.QPointF(INNER.left() - MARGIN, 400))

        _drag(qtbot, view, side, side + QtCore.QPointF(10, 0))

        assert reported == []


class TestDoubleClickOnContent:
    @staticmethod
    def _mask_with_content_at(left: int, right: int) -> ContentMask:
        """A mask at half the page's resolution with a bar between two columns."""
        image = np.zeros((IMAGE_HEIGHT // 2, IMAGE_WIDTH // 2), dtype=bool)
        image[170:230, left // 2 : right // 2] = True
        return ContentMask(image, np.diag([0.5, 0.5, 1.0]))

    def test_the_content_is_moved_so_its_nearest_side_meets_the_guide(
        self, qtbot, view
    ):
        mask = self._mask_with_content_at(250, 290)
        view.set_content_mask_provider(lambda: mask)
        view.state.add_guide(is_horizontal=False, position_mm=10 * MM_X)
        _zoom_in_on_the_content(view)
        click = view.virtual_to_widget_point(QtCore.QPointF(270, 400))
        # The area searched is 30 mask pixels wide around the click.
        found = mask.find_content_in_area(270 // 2 - 15, 400 // 2 - 15, 30, 30)
        assert found is not None
        found_right = (found[0] + found[2]) * 2  # in page pixels
        shift = (300 + 10) - found_right

        with qtbot.waitSignal(view.margins_set_locally) as moved:
            qtbot.mouseDClick(
                view.viewport(),
                QtCore.Qt.MouseButton.LeftButton,
                pos=click.toPoint(),
            )

        margins = moved.args[0]
        assert margins.left == pytest.approx((MARGIN + shift) * MM_X, abs=0.2)
        assert margins.right == pytest.approx((MARGIN - shift) * MM_X, abs=0.2)

    def test_nothing_happens_without_guides(self, qtbot, view):
        mask = self._mask_with_content_at(250, 290)
        view.set_content_mask_provider(lambda: mask)
        reported: list[Margins] = []
        view.margins_set_locally.connect(reported.append)
        click = view.virtual_to_widget_point(QtCore.QPointF(270, 400))

        qtbot.mouseDClick(
            view.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=click.toPoint()
        )

        assert reported == []

    def test_nothing_happens_when_the_click_is_not_on_content(self, qtbot, view):
        mask = self._mask_with_content_at(250, 290)
        view.set_content_mask_provider(lambda: mask)
        view.state.add_guide(is_horizontal=False, position_mm=10 * MM_X)
        reported: list[Margins] = []
        view.margins_set_locally.connect(reported.append)
        click = view.virtual_to_widget_point(QtCore.QPointF(380, 480))

        qtbot.mouseDClick(
            view.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=click.toPoint()
        )

        assert reported == []

    def test_a_vertical_only_key_ignores_vertical_guides(self, qtbot, view):
        mask = self._mask_with_content_at(250, 290)
        view.set_content_mask_provider(lambda: mask)
        view.state.add_guide(is_horizontal=False, position_mm=10 * MM_X)
        reported: list[Margins] = []
        view.margins_set_locally.connect(reported.append)
        click = view.virtual_to_widget_point(QtCore.QPointF(270, 400))

        qtbot.mouseDClick(
            view.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            CONTROL,
            pos=click.toPoint(),
        )

        assert reported == []


def test_the_view_paints_without_content_or_state(qtbot):
    widget = PageLayoutImageView()
    qtbot.addWidget(widget)
    widget.resize(200, 200)
    widget.show()
    qtbot.waitExposed(widget)

    image: QtGui.QImage = widget.viewport().grab().toImage()

    assert not image.isNull()
