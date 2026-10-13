"""Picking a colour from a view: the median of a small circle under the cursor."""

import numpy as np
import pytest
from PySide6 import QtCore, QtGui
from pytestqt.qtbot import QtBot

from scantailor.app.ui.image_views.base import ImageViewBase
from scantailor.app.ui.widgets.color_pickup import (
    PICKUP_RADIUS,
    ColorPickupInteraction,
    bit_mix,
    bit_unmix,
    median_color,
)
from scantailor.core.models import Dpi
from scantailor.core.transformation import ImageTransformation

SIZE = 2 * PICKUP_RADIUS + 1
RED = (200, 30, 30)
BLUE = (30, 30, 200)


def _patch(color: tuple[int, int, int]) -> np.ndarray:
    return np.full((SIZE, SIZE, 3), color, dtype=np.uint8)


class TestBitMixing:
    def test_round_trips_every_channel_value(self):
        rng = np.random.default_rng(2)
        colors = rng.integers(0, 256, size=(200, 3), dtype=np.uint8)

        mixed = bit_mix(colors)

        for color, value in zip(colors, mixed):
            assert bit_unmix(int(value)) == tuple(int(c) for c in color)

    def test_the_top_bits_of_each_channel_lead(self):
        bright_red = bit_mix(np.array([[128, 0, 0]], dtype=np.uint8))[0]
        dim_everything = bit_mix(np.array([[127, 127, 127]], dtype=np.uint8))[0]

        assert bright_red > dim_everything

    def test_black_and_white_are_the_extremes(self):
        values = bit_mix(np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8))

        assert values[0] == 0
        assert values[1] == 0xFFFFFF

    def test_keeps_the_leading_shape(self):
        assert bit_mix(np.zeros((4, 5, 3), dtype=np.uint8)).shape == (4, 5)


class TestMedianColor:
    def test_a_flat_patch_gives_its_colour(self):
        assert median_color(_patch(RED)) == RED

    def test_pixels_outside_the_circle_are_ignored(self):
        patch = _patch(RED)
        for corner in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
            patch[corner] = BLUE

        assert median_color(patch) == RED

    def test_the_majority_colour_wins_over_a_stray_pixel(self):
        patch = _patch(RED)
        patch[SIZE // 2, SIZE // 2] = BLUE

        assert median_color(patch) == RED

    def test_the_result_is_a_colour_that_is_present(self):
        rng = np.random.default_rng(9)
        patch = rng.integers(0, 256, size=(SIZE, SIZE, 3), dtype=np.uint8)

        result = median_color(patch)

        assert result is not None
        assert any(tuple(pixel) == result for row in patch for pixel in row)

    def test_an_empty_patch_gives_nothing(self):
        assert median_color(np.zeros((0, 0, 3), dtype=np.uint8)) is None

    def test_the_patch_is_not_modified(self):
        patch = _patch(RED)
        patch[3, 3] = BLUE
        before = patch.copy()

        median_color(patch)

        assert np.array_equal(patch, before)


@pytest.fixture
def view(qtbot: QtBot) -> ImageViewBase:
    image = np.zeros((800, 600, 3), dtype=np.uint8)
    image[:, :300] = RED
    image[:, 300:] = BLUE
    widget = ImageViewBase()
    qtbot.addWidget(widget)
    widget.resize(QtCore.QSize(300, 400))
    widget.show()
    qtbot.waitExposed(widget)
    widget.set_image(
        image, transformation=ImageTransformation.from_size(600, 800, Dpi.uniform(300))
    )
    return widget


@pytest.fixture
def pickup(view: ImageViewBase) -> ColorPickupInteraction:
    return ColorPickupInteraction(view)


def _mouse(
    kind: QtCore.QEvent.Type,
    point: QtCore.QPoint,
    button: QtCore.Qt.MouseButton = QtCore.Qt.MouseButton.LeftButton,
) -> QtGui.QMouseEvent:
    return QtGui.QMouseEvent(
        kind,
        QtCore.QPointF(point),
        QtCore.QPointF(point),
        button,
        button,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )


class TestInteraction:
    LEFT_HALF = QtCore.QPoint(40, 200)
    RIGHT_HALF = QtCore.QPoint(260, 200)

    def test_is_inactive_until_started(self, pickup: ColorPickupInteraction):
        assert not pickup.is_active()

    def test_starting_shows_a_cross_cursor(
        self, pickup: ColorPickupInteraction, view: ImageViewBase
    ):
        pickup.start(self.LEFT_HALF)

        assert pickup.is_active()
        assert view.viewport().cursor().shape() == QtCore.Qt.CursorShape.CrossCursor

    def test_picking_reports_the_colour_under_the_circle_and_ends(
        self, qtbot: QtBot, pickup: ColorPickupInteraction
    ):
        pickup.start(self.LEFT_HALF)

        with qtbot.waitSignal(pickup.color_picked) as signal:
            color = pickup.pick_at(self.RIGHT_HALF)

        assert color is not None
        assert (color.red(), color.green(), color.blue()) == BLUE
        assert signal.args == [color]
        assert not pickup.is_active()

    def test_a_left_click_picks(self, qtbot: QtBot, pickup: ColorPickupInteraction):
        pickup.start(self.RIGHT_HALF)
        event = _mouse(QtCore.QEvent.Type.MouseButtonPress, self.LEFT_HALF)

        with qtbot.waitSignal(pickup.color_picked) as signal:
            assert pickup.handle_mouse_press(event)

        assert signal.args[0].red() == RED[0]

    def test_a_right_click_is_used_but_picks_nothing(
        self, pickup: ColorPickupInteraction
    ):
        picked: list[QtGui.QColor] = []
        pickup.color_picked.connect(picked.append)
        pickup.start(self.LEFT_HALF)
        event = _mouse(
            QtCore.QEvent.Type.MouseButtonPress,
            self.LEFT_HALF,
            QtCore.Qt.MouseButton.RightButton,
        )

        assert pickup.handle_mouse_press(event)

        assert picked == []
        assert pickup.is_active()

    def test_escape_cancels_without_a_colour(
        self, qtbot: QtBot, pickup: ColorPickupInteraction
    ):
        picked: list[QtGui.QColor] = []
        pickup.color_picked.connect(picked.append)
        pickup.start(self.LEFT_HALF)
        escape = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_Escape,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )

        with qtbot.waitSignal(pickup.cancelled):
            assert pickup.handle_key_press(escape)

        assert not pickup.is_active()
        assert picked == []

    def test_other_keys_are_left_alone(self, pickup: ColorPickupInteraction):
        pickup.start(self.LEFT_HALF)
        key = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_A,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )

        assert not pickup.handle_key_press(key)
        assert pickup.is_active()

    def test_events_are_ignored_while_inactive(self, pickup: ColorPickupInteraction):
        press = _mouse(QtCore.QEvent.Type.MouseButtonPress, self.LEFT_HALF)
        move = _mouse(QtCore.QEvent.Type.MouseMove, self.LEFT_HALF)

        assert not pickup.handle_mouse_press(press)
        assert not pickup.handle_mouse_move(move)

    def test_the_circle_follows_the_mouse(self, pickup: ColorPickupInteraction):
        pickup.start(self.LEFT_HALF)

        moved = pickup.handle_mouse_move(
            _mouse(QtCore.QEvent.Type.MouseMove, self.RIGHT_HALF)
        )

        assert moved
        image = QtGui.QImage(300, 400, QtGui.QImage.Format.Format_RGB32)
        image.fill(QtCore.Qt.GlobalColor.white)
        painter = QtGui.QPainter(image)
        pickup.paint_overlay(painter)
        painter.end()
        assert image.pixelColor(self.RIGHT_HALF.x() + PICKUP_RADIUS, 200).red() > 200
        assert image.pixelColor(self.RIGHT_HALF.x() + PICKUP_RADIUS, 200).green() < 200
        assert image.pixelColor(self.LEFT_HALF.x() + PICKUP_RADIUS, 200) == (
            QtGui.QColor("white")
        )

    def test_nothing_is_drawn_while_inactive(self, pickup: ColorPickupInteraction):
        image = QtGui.QImage(300, 400, QtGui.QImage.Format.Format_RGB32)
        image.fill(QtCore.Qt.GlobalColor.white)
        painter = QtGui.QPainter(image)

        pickup.paint_overlay(painter)

        painter.end()
        assert image == _white(300, 400)


def _white(width: int, height: int) -> QtGui.QImage:
    image = QtGui.QImage(width, height, QtGui.QImage.Format.Format_RGB32)
    image.fill(QtCore.Qt.GlobalColor.white)
    return image
