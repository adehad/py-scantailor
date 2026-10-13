"""Hover hints of the Deskew view, in the original's words."""

import numpy as np
import pytest
from PySide6 import QtCore
from pytestqt.qtbot import QtBot

from scantailor.app.ui.image_views.deskew import DeskewImageView
from scantailor.core.models import Dpi
from scantailor.core.transformation import ImageTransformation


@pytest.fixture
def view(qtbot: QtBot) -> DeskewImageView:
    widget = DeskewImageView()
    qtbot.addWidget(widget)
    widget.resize(300, 400)
    widget.show()
    qtbot.waitExposed(widget)
    widget.set_image(
        np.full((800, 600), 255, dtype=np.uint8),
        transformation=ImageTransformation.from_size(600, 800, Dpi.uniform(300)),
    )
    return widget


def test_the_default_hint_explains_rotating_with_the_wheel(view):
    assert view.status_tip_text == (
        "Use Ctrl+Wheel to rotate or Ctrl+Shift+Wheel for finer rotation."
    )
    assert view.viewport().statusTip() == view.status_tip_text


def test_a_rotation_handle_says_it_rotates_the_image(qtbot, view):
    left, _right = view.rotation_handles()

    qtbot.mouseMove(view.viewport(), left.toPoint())

    assert view.status_tip_text == "Drag this handle to rotate the image."


def test_an_oblique_handle_says_it_adjusts_the_shear(qtbot, view):
    top, _bottom = view.oblique_handles()

    qtbot.mouseMove(view.viewport(), top.toPoint())

    assert view.status_tip_text == "Drag this handle to adjust oblique (shear)."


def test_away_from_the_handles_the_default_returns(qtbot, view):
    left, _right = view.rotation_handles()
    qtbot.mouseMove(view.viewport(), left.toPoint())

    qtbot.mouseMove(view.viewport(), QtCore.QPoint(150, 200))

    assert view.status_tip_text == view.default_status_tip
