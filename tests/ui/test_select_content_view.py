"""The Select Content view: minimum box size in widget pixels, double click."""

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.image_views.base import Edge
from scantailor.app.ui.image_views.select_content import SelectContentImageView
from scantailor.core.content_mask import ContentMask
from scantailor.core.models import Dpi
from scantailor.core.transformation import ImageTransformation

IMAGE_WIDTH = 600
IMAGE_HEIGHT = 800
CONTROL = QtCore.Qt.KeyboardModifier.ControlModifier
SHIFT = QtCore.Qt.KeyboardModifier.ShiftModifier
LEFT = QtCore.Qt.MouseButton.LeftButton
CONTENT = QtCore.QRectF(100, 200, 300, 400)


@pytest.fixture
def view(qtbot: QtBot) -> SelectContentImageView:
    widget = SelectContentImageView()
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
    widget.set_content_rect(CONTENT)
    return widget


def _drag(
    qtbot: QtBot,
    view: SelectContentImageView,
    start: QtCore.QPointF,
    end: QtCore.QPointF,
) -> None:
    viewport = view.viewport()
    assert viewport is not None
    qtbot.mousePress(viewport, LEFT, pos=start.toPoint())
    for step in range(1, 5):
        qtbot.mouseMove(viewport, (start + (end - start) * step / 4).toPoint())
    qtbot.mouseRelease(viewport, LEFT, pos=end.toPoint())


def _hover(
    view: SelectContentImageView,
    position: QtCore.QPointF,
    modifier: QtCore.Qt.KeyboardModifier = QtCore.Qt.KeyboardModifier.NoModifier,
) -> None:
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


def _double_click(
    qtbot: QtBot,
    view: SelectContentImageView,
    virtual: QtCore.QPointF,
    modifier: QtCore.Qt.KeyboardModifier = QtCore.Qt.KeyboardModifier.NoModifier,
) -> None:
    qtbot.mouseDClick(
        view.viewport(),
        LEFT,
        modifier,
        pos=view.virtual_to_widget_point(virtual).toPoint(),
    )


def _mask(*bars: tuple[int, int, int, int]) -> ContentMask:
    """A mask at half the page's resolution with bars given as page pixel boxes."""
    image = np.zeros((IMAGE_HEIGHT // 2, IMAGE_WIDTH // 2), dtype=bool)
    for left, top, right, bottom in bars:
        image[top // 2 : bottom // 2, left // 2 : right // 2] = True
    return ContentMask(image, np.diag([0.5, 0.5, 1.0]))


class TestMinimumBoxSize:
    def test_the_minimum_is_ten_widget_pixels_at_a_zoom_that_shrinks_the_page(
        self, qtbot, view
    ):
        corner = view.virtual_to_widget_point(CONTENT.bottomRight())

        with qtbot.waitSignal(view.manual_content_rect_set) as resized:
            _drag(qtbot, view, corner, corner + QtCore.QPointF(-2000, -2000))

        rect = resized.args[0]
        assert rect.width() * view.zoom_level == pytest.approx(10)
        assert rect.height() * view.zoom_level == pytest.approx(10)

    def test_the_minimum_stays_ten_widget_pixels_when_zoomed_in(self, qtbot, view):
        view.set_zoom(2.0, QtCore.QPointF(150, 200))
        corner = view.virtual_to_widget_point(CONTENT.bottomRight())
        view.set_zoom(2.0, corner)  # Keep the corner on screen.
        corner = view.virtual_to_widget_point(CONTENT.bottomRight())

        with qtbot.waitSignal(view.manual_content_rect_set) as resized:
            _drag(qtbot, view, corner, corner + QtCore.QPointF(-2000, -2000))

        rect = resized.args[0]
        assert rect.width() == pytest.approx(10 / view.zoom_level)
        assert rect.width() < 10

    def test_the_page_box_keeps_the_same_minimum(self, qtbot, view):
        view.set_page_rect_enabled(True)
        view.set_content_rect(QtCore.QRectF())
        view.set_page_rect(QtCore.QRectF(50, 150, 500, 600))
        corner = view.virtual_to_widget_point(QtCore.QPointF(550, 750))

        with qtbot.waitSignal(view.manual_page_rect_set) as resized:
            _drag(qtbot, view, corner, corner + QtCore.QPointF(-2000, -2000))

        assert resized.args[0].width() * view.zoom_level == pytest.approx(10)


class TestDoubleClickOnContent:
    def test_a_click_outside_the_box_grows_it_to_include_the_content(self, qtbot, view):
        view.set_content_mask_provider(lambda: _mask((420, 300, 500, 340)))

        with qtbot.waitSignal(view.manual_content_rect_set) as grown:
            _double_click(qtbot, view, QtCore.QPointF(450, 320))

        rect = grown.args[0]
        assert rect.left() == pytest.approx(CONTENT.left())
        # Only the content within 15 mask pixels of the click is looked at.
        assert rect.right() == pytest.approx(480, abs=4)
        assert rect.top() == pytest.approx(CONTENT.top())

    def test_a_click_inside_the_box_moves_the_nearest_side_to_the_content(
        self, qtbot, view
    ):
        view.set_content_mask_provider(lambda: _mask((120, 230, 360, 560)))

        with qtbot.waitSignal(view.manual_content_rect_set) as adjusted:
            _double_click(qtbot, view, QtCore.QPointF(130, 400))

        rect = adjusted.args[0]
        assert rect.left() == pytest.approx(120, abs=4)
        assert rect.right() == pytest.approx(CONTENT.right())
        assert rect.top() == pytest.approx(CONTENT.top())

    def test_the_vertical_key_moves_only_a_top_or_bottom_side(self, qtbot, view):
        view.set_content_mask_provider(lambda: _mask((120, 230, 360, 560)))

        with qtbot.waitSignal(view.manual_content_rect_set) as adjusted:
            _double_click(qtbot, view, QtCore.QPointF(130, 250), CONTROL)

        rect = adjusted.args[0]
        assert rect.left() == pytest.approx(CONTENT.left())
        assert rect.top() == pytest.approx(230, abs=4)

    def test_the_horizontal_key_moves_only_a_left_or_right_side(self, qtbot, view):
        view.set_content_mask_provider(lambda: _mask((120, 230, 360, 560)))

        with qtbot.waitSignal(view.manual_content_rect_set) as adjusted:
            _double_click(qtbot, view, QtCore.QPointF(130, 240), SHIFT)

        rect = adjusted.args[0]
        assert rect.left() == pytest.approx(120, abs=4)
        assert rect.top() == pytest.approx(CONTENT.top())

    def test_both_keys_move_a_horizontal_and_a_vertical_side_together(
        self, qtbot, view
    ):
        view.set_content_mask_provider(lambda: _mask((120, 230, 360, 560)))

        with qtbot.waitSignal(view.manual_content_rect_set) as adjusted:
            _double_click(qtbot, view, QtCore.QPointF(130, 240), CONTROL | SHIFT)

        rect = adjusted.args[0]
        assert rect.left() == pytest.approx(120, abs=4)
        assert rect.top() == pytest.approx(230, abs=4)
        assert rect.right() == pytest.approx(CONTENT.right())

    def test_a_tie_between_a_vertical_and_a_horizontal_distance_keeps_the_key_axis(
        self, qtbot, view
    ):
        # Fifty from the top and from the left of the 300 by 400 box.
        click = QtCore.QPointF(CONTENT.left() + 50, CONTENT.top() + 50)
        view.set_content_mask_provider(lambda: _mask((120, 230, 360, 560)))

        with qtbot.waitSignal(view.manual_content_rect_set) as vertical:
            _double_click(qtbot, view, click, CONTROL)

        rect = vertical.args[0]
        assert rect.top() == pytest.approx(230, abs=4)
        assert rect.left() == pytest.approx(CONTENT.left())

    def test_a_tie_with_both_keys_moves_the_top_and_the_left_side(self, qtbot, view):
        click = QtCore.QPointF(CONTENT.left() + 50, CONTENT.top() + 50)
        view.set_content_mask_provider(lambda: _mask((120, 230, 360, 560)))

        with qtbot.waitSignal(view.manual_content_rect_set) as both:
            _double_click(qtbot, view, click, CONTROL | SHIFT)

        rect = both.args[0]
        assert rect.top() == pytest.approx(230, abs=4)
        assert rect.left() == pytest.approx(120, abs=4)
        assert rect.right() == pytest.approx(CONTENT.right())
        assert rect.bottom() == pytest.approx(CONTENT.bottom())

    def test_the_nearest_side_without_keys_is_the_smallest_distance(self):
        box = QtCore.QRectF(0, 0, 300, 400)
        nearest = SelectContentImageView._nearest_edges

        no_keys = QtCore.Qt.KeyboardModifier.NoModifier
        assert nearest(box, QtCore.QPointF(50, 50), no_keys) == Edge.TOP
        assert nearest(box, QtCore.QPointF(250, 200), no_keys) == Edge.RIGHT
        assert nearest(box, QtCore.QPointF(150, 390), no_keys) == Edge.BOTTOM
        assert nearest(box, QtCore.QPointF(10, 200), no_keys) == Edge.LEFT

    def test_nothing_happens_when_the_click_is_not_near_content(self, qtbot, view):
        view.set_content_mask_provider(lambda: _mask((120, 230, 160, 260)))
        reported: list[QtCore.QRectF] = []
        view.manual_content_rect_set.connect(reported.append)

        _double_click(qtbot, view, QtCore.QPointF(300, 500))

        assert reported == []
        assert view.content_rect == CONTENT

    def test_nothing_happens_without_a_mask(self, qtbot, view):
        reported: list[QtCore.QRectF] = []
        view.manual_content_rect_set.connect(reported.append)

        _double_click(qtbot, view, QtCore.QPointF(130, 400))

        assert reported == []

    def test_a_provider_without_a_mask_is_tolerated(self, qtbot, view):
        view.set_content_mask_provider(lambda: None)
        reported: list[QtCore.QRectF] = []
        view.manual_content_rect_set.connect(reported.append)

        _double_click(qtbot, view, QtCore.QPointF(130, 400))

        assert reported == []

    def test_a_double_click_does_not_create_a_content_box(self, qtbot, view):
        view.set_content_rect(QtCore.QRectF())
        view.set_content_mask_provider(lambda: _mask((120, 230, 360, 560)))
        reported: list[QtCore.QRectF] = []
        view.manual_content_rect_set.connect(reported.append)

        _double_click(qtbot, view, QtCore.QPointF(300, 400))

        assert reported == []
        assert view.content_rect.isEmpty()

    def test_the_mask_is_asked_for_only_when_a_click_needs_it(self, qtbot, view):
        calls: list[int] = []

        def provider() -> ContentMask:
            calls.append(1)
            return _mask((120, 230, 360, 560))

        view.set_content_mask_provider(provider)
        assert calls == []

        _double_click(qtbot, view, QtCore.QPointF(130, 400))

        assert calls == [1]

    def test_the_page_box_follows_a_content_box_that_grew(self, qtbot, view):
        view.set_page_rect_enabled(True)
        view.set_page_rect(CONTENT)
        view.set_content_mask_provider(lambda: _mask((420, 300, 500, 340)))
        pages: list[QtCore.QRectF] = []
        view.manual_page_rect_set.connect(pages.append)

        _double_click(qtbot, view, QtCore.QPointF(450, 320))

        assert pages and pages[-1].right() >= 478


class TestHints:
    def test_the_default_hint_explains_the_double_click(self, view):
        assert view.default_status_tip.startswith(
            "Use the context menu to enable / disable the content box."
        )
        assert "double-click on content" in view.default_status_tip

    def test_hovering_a_content_box_side_says_how_to_resize_it(self, view):
        _hover(view, view.virtual_to_widget_point(QtCore.QPointF(100, 400)))

        assert (
            view.status_tip_text == "Drag lines or corners to resize the content box."
        )

    def test_hovering_the_body_needs_shift_for_the_move_hint(self, view):
        inside = view.virtual_to_widget_point(QtCore.QPointF(250, 400))

        _hover(view, inside)
        assert view.status_tip_text == view.default_status_tip

        _hover(view, inside, SHIFT)
        assert view.status_tip_text == "Hold left mouse button to drag the content box."

    def test_dragging_the_body_says_how_to_finish(self, qtbot, view):
        inside = view.virtual_to_widget_point(QtCore.QPointF(250, 400))

        qtbot.mousePress(view.viewport(), LEFT, SHIFT, pos=inside.toPoint())
        try:
            assert (
                view.status_tip_text == "Release left mouse button to finish dragging."
            )
        finally:
            qtbot.mouseRelease(view.viewport(), LEFT, SHIFT, pos=inside.toPoint())

    def test_hovering_a_page_box_side_says_how_to_resize_it(self, view):
        view.set_page_rect_enabled(True)
        view.set_page_rect(QtCore.QRectF(50, 150, 500, 600))

        _hover(view, view.virtual_to_widget_point(QtCore.QPointF(50, 450)))

        assert view.status_tip_text == "Drag lines or corners to resize the page box."


def test_nearest_edges_tie_between_left_and_bottom_goes_to_left_without_modifier() -> (
    None
):
    """Equal distances resolve in the order top, left, bottom, right."""
    box = QtCore.QRectF(0, 0, 300, 400)

    edges = SelectContentImageView._nearest_edges(
        box, QtCore.QPointF(50, 350), QtCore.Qt.KeyboardModifier.NoModifier
    )

    assert edges == Edge.LEFT
