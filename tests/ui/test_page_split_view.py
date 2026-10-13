"""The Page Split view: tilted cutters, the tint of the pages and removed halves."""

import numpy as np
import pytest
from PySide6 import QtCore, QtGui
from pytestqt.qtbot import QtBot

from scantailor.app.ui.image_views.base import DEFAULT_STATUS_TIP
from scantailor.app.ui.image_views.page_split import PageSplitImageView
from scantailor.core.models import Dpi, SubPage
from scantailor.core.transformation import ImageTransformation
from scantailor.filters.page_split.page_layout import PageLayout

IMAGE_WIDTH = 600
IMAGE_HEIGHT = 800
FIT_ZOOM = 0.5  # A 300 by 400 window shows the page at half size


def _page() -> np.ndarray:
    return np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 255, dtype=np.uint8)


@pytest.fixture
def view(qtbot: QtBot) -> PageSplitImageView:
    widget = PageSplitImageView()
    qtbot.addWidget(widget)
    widget.resize(300, 400)
    widget.show()
    qtbot.waitExposed(widget)
    widget.set_image(
        _page(),
        transformation=ImageTransformation.from_size(
            IMAGE_WIDTH, IMAGE_HEIGHT, Dpi.uniform(300)
        ),
    )
    return widget


def _drag(
    qtbot: QtBot, view: PageSplitImageView, start: QtCore.QPointF, end: QtCore.QPointF
) -> None:
    viewport = view.viewport()
    assert viewport is not None
    left = QtCore.Qt.MouseButton.LeftButton
    qtbot.mousePress(viewport, left, pos=start.toPoint())
    for step in range(1, 5):
        qtbot.mouseMove(viewport, (start + (end - start) * step / 4).toPoint())
    qtbot.mouseRelease(viewport, left, pos=end.toPoint())


def _pixel(view: PageSplitImageView, image_x: float, image_y: float) -> QtGui.QColor:
    """Return the colour the view paints at a point of the page."""
    spot = view.image_to_widget().map(QtCore.QPointF(image_x, image_y)).toPoint()
    return view.viewport().grab().toImage().pixelColor(spot)


def _tilted_layout() -> PageLayout:
    return PageLayout.two_pages(IMAGE_WIDTH, IMAGE_HEIGHT, 300).model_copy(
        update={"cutter1": ((250.0, 0.0), (350.0, 800.0))}
    )


class TestTiltedCutters:
    def test_a_tilted_cutter_is_drawn_tilted_with_a_handle_at_each_end(self, view):
        view.set_page_layout(_tilted_layout())

        ((top, bottom),) = view.line_handle_positions()

        assert top.x() < bottom.x()
        # The handles sit a radius inside the page edges, so they see less of the tilt.
        inset = 2 * PageSplitImageView.HANDLE_RADIUS
        shown_height = IMAGE_HEIGHT * FIT_ZOOM - inset
        assert bottom.x() - top.x() == pytest.approx(100 * shown_height / IMAGE_HEIGHT)
        assert top.y() < bottom.y()

    def test_dragging_the_bottom_handle_leaves_the_top_where_it_was(self, qtbot, view):
        view.set_page_layout(PageLayout.two_pages(IMAGE_WIDTH, IMAGE_HEIGHT, 300))
        ((_top, bottom),) = view.line_handle_positions()

        with qtbot.waitSignal(view.page_layout_set_locally) as moved:
            _drag(qtbot, view, bottom, bottom + QtCore.QPointF(-40, -10))

        layout = moved.args[0]
        assert layout.cutter1 is not None
        assert layout.cutter1[0] == pytest.approx((300, 0))
        assert layout.cutter1[1] == pytest.approx((300 - 40 / FIT_ZOOM, IMAGE_HEIGHT))

    def test_dragging_the_body_of_a_tilted_cutter_keeps_its_slope(self, qtbot, view):
        view.set_page_layout(_tilted_layout())
        ((top, bottom),) = view.line_handle_positions()
        middle = (top + bottom) / 2

        with qtbot.waitSignal(view.page_layout_set_locally) as moved:
            _drag(qtbot, view, middle, middle + QtCore.QPointF(30, 0))

        layout = moved.args[0]
        assert layout.cutter1 is not None
        (top_x, top_y), (bottom_x, bottom_y) = layout.cutter1
        assert bottom_x - top_x == pytest.approx(100)
        assert top_x == pytest.approx(250 + 30 / FIT_ZOOM)
        assert (top_y, bottom_y) == (0, IMAGE_HEIGHT)

    def test_a_handle_does_not_jump_to_the_pointer_when_grabbed_off_centre(
        self, qtbot, view
    ):
        view.set_page_layout(PageLayout.two_pages(IMAGE_WIDTH, IMAGE_HEIGHT, 300))
        ((top, _bottom),) = view.line_handle_positions()
        grab = top + QtCore.QPointF(6, 3)

        with qtbot.waitSignal(view.page_layout_set_locally) as moved:
            _drag(qtbot, view, grab, grab + QtCore.QPointF(20, 0))

        assert moved.args[0].cutter1[0][0] == pytest.approx(300 + 20 / FIT_ZOOM)

    def test_the_two_cutters_of_a_page_with_offcut_can_each_be_tilted(
        self, qtbot, view
    ):
        view.set_page_layout(
            PageLayout.single_page_cut(IMAGE_WIDTH, IMAGE_HEIGHT, 100, 500)
        )
        _left, (right_top, _right_bottom) = view.line_handle_positions()

        with qtbot.waitSignal(view.page_layout_set_locally) as moved:
            _drag(qtbot, view, right_top, right_top + QtCore.QPointF(-30, 0))

        layout = moved.args[0]
        assert layout.cutter1 is not None and layout.cutter2 is not None
        assert layout.cutter1 == ((100, 0), (100, IMAGE_HEIGHT))
        assert layout.cutter2[0][0] == pytest.approx(500 - 30 / FIT_ZOOM)
        assert layout.cutter2[1][0] == pytest.approx(500)

    def test_a_cutter_cannot_be_pulled_across_the_other_at_either_end(
        self, qtbot, view
    ):
        view.set_page_layout(
            PageLayout.single_page_cut(IMAGE_WIDTH, IMAGE_HEIGHT, 100, 500)
        )
        (_l_top, l_bottom), _right = view.line_handle_positions()

        with qtbot.waitSignal(view.page_layout_set_locally) as moved:
            _drag(qtbot, view, l_bottom, l_bottom + QtCore.QPointF(400, 0))

        layout = moved.args[0]
        assert layout.cutter1[1][0] <= layout.cutter2[1][0]

    def test_a_cutter_given_off_the_page_edges_is_extended_to_them(self, view):
        view.set_page_layout(
            PageLayout.two_pages(IMAGE_WIDTH, IMAGE_HEIGHT, 300).model_copy(
                update={"cutter1": ((280.0, 100.0), (320.0, 700.0))}
            )
        )

        layout = view.page_layout

        assert layout is not None and layout.cutter1 is not None
        (top_x, top_y), (bottom_x, bottom_y) = layout.cutter1
        assert (top_y, bottom_y) == (0, IMAGE_HEIGHT)
        assert top_x == pytest.approx(280 - 100 * 40 / 600)
        assert bottom_x == pytest.approx(320 + 100 * 40 / 600)

    def test_the_hover_hint_names_the_line_and_handles(self, qtbot, view):
        view.set_page_layout(_tilted_layout())
        ((top, _bottom),) = view.line_handle_positions()

        qtbot.mouseMove(view.viewport(), top.toPoint())

        assert view.status_tip_text == "Drag the line or the handles."

    def test_the_default_hint_returns_away_from_the_cutters(self, qtbot, view):
        view.set_page_layout(_tilted_layout())
        ((top, _bottom),) = view.line_handle_positions()
        qtbot.mouseMove(view.viewport(), top.toPoint())

        qtbot.mouseMove(view.viewport(), QtCore.QPoint(10, 200))

        assert view.status_tip_text == DEFAULT_STATUS_TIP


class TestPageTint:
    def test_an_uncut_page_is_tinted_all_over(self, view):
        view.set_page_layout(None)
        bare = _pixel(view, 300, 400)

        view.set_page_layout(PageLayout.single_page_uncut(IMAGE_WIDTH, IMAGE_HEIGHT))

        assert _pixel(view, 300, 400) != bare
        assert _pixel(view, 20, 20) != bare

    def test_only_the_part_between_the_cutters_is_tinted_on_a_page_with_offcut(
        self, view
    ):
        view.set_page_layout(None)
        bare = _pixel(view, 50, 400)

        view.set_page_layout(
            PageLayout.single_page_cut(IMAGE_WIDTH, IMAGE_HEIGHT, 200, 400)
        )

        assert _pixel(view, 50, 400) == bare
        assert _pixel(view, 300, 400) != bare

    def test_the_two_pages_of_a_spread_have_different_tints(self, view):
        view.set_page_layout(PageLayout.two_pages(IMAGE_WIDTH, IMAGE_HEIGHT, 300))

        assert _pixel(view, 100, 400) != _pixel(view, 500, 400)

    def test_the_tint_follows_a_tilted_cutter(self, view):
        view.set_page_layout(_tilted_layout())
        straight = PageLayout.two_pages(IMAGE_WIDTH, IMAGE_HEIGHT, 300)

        # Left of the vertical line at the bottom, but right of the tilted one.
        point = (330, 700)
        tilted_colour = _pixel(view, *point)
        view.set_page_layout(straight)

        assert tilted_colour != _pixel(view, *point)


class TestRemovedPages:
    @pytest.fixture
    def split_view(self, view: PageSplitImageView) -> PageSplitImageView:
        view.set_page_layout(PageLayout.two_pages(IMAGE_WIDTH, IMAGE_HEIGHT, 300))
        return view

    def test_a_removed_half_is_tinted_differently_from_a_kept_one(self, split_view):
        kept_left = _pixel(split_view, 100, 100)
        kept_right = _pixel(split_view, 500, 100)

        split_view.set_removed_pages(left=True, right=False)

        assert _pixel(split_view, 100, 100) != kept_left
        assert _pixel(split_view, 500, 100) == kept_right

    def test_a_removed_half_offers_a_button_to_restore_it(self, qtbot, split_view):
        split_view.set_removed_pages(left=False, right=True)
        button = split_view._unremove_button_rect(SubPage.RIGHT_PAGE)
        assert button is not None
        assert split_view._unremove_button_rect(SubPage.LEFT_PAGE) is None

        with qtbot.waitSignal(split_view.unremove_requested) as clicked:
            qtbot.mouseClick(
                split_view.viewport(),
                QtCore.Qt.MouseButton.LeftButton,
                pos=button.center().toPoint(),
            )

        assert clicked.args == [SubPage.RIGHT_PAGE]
        assert split_view._unremove_button_rect(SubPage.RIGHT_PAGE) is None

    def test_the_restore_button_sits_in_the_middle_of_its_page(self, split_view):
        split_view.set_removed_pages(left=True, right=False)

        button = split_view._unremove_button_rect(SubPage.LEFT_PAGE)

        assert button is not None
        centre = split_view.widget_to_image().map(button.center())
        assert centre.x() == pytest.approx(150)
        assert centre.y() == pytest.approx(400)

    def test_hovering_the_restore_button_says_what_it_does(self, qtbot, split_view):
        split_view.set_removed_pages(left=True, right=False)
        button = split_view._unremove_button_rect(SubPage.LEFT_PAGE)
        assert button is not None

        qtbot.mouseMove(split_view.viewport(), button.center().toPoint())

        assert split_view.status_tip_text == "Restore removed page."

    def test_pages_that_were_not_removed_have_no_button(self, split_view):
        assert split_view._unremove_button_rect(SubPage.LEFT_PAGE) is None
        assert split_view._unremove_button_rect(SubPage.RIGHT_PAGE) is None
