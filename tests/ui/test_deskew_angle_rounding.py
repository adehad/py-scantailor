"""The Deskew spin box and the view label round an angle the same way."""

import pytest
from PySide6 import QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.image_views.deskew import format_angle
from tests.ui.locale_numbers import shown_numbers


@pytest.mark.parametrize(
    "angle", [0.125, 0.135, -0.125, 0.005, 2.675, 0.0, 44.999, -0.004]
)
def test_view_label_matches_the_spin_box_text(qtbot: QtBot, angle: float):
    spin_box = QtWidgets.QDoubleSpinBox()
    qtbot.addWidget(spin_box)
    spin_box.setDecimals(2)
    spin_box.setRange(-45.0, 45.0)
    spin_box.setSuffix("°")

    spin_box.setValue(angle)

    assert format_angle(angle) == spin_box.text()


def test_a_tie_rounds_away_from_zero_like_the_spin_box():
    assert shown_numbers(format_angle(0.125)) == pytest.approx([0.13])
