"""The picture zone properties dialog, with upstream's wording and order."""

import pytest
from PySide6 import QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.widgets.picture_zone_dialog import (
    LAYER_CHOICES,
    PictureZoneDialog,
    layer_label,
)
from scantailor.core.zones import PictureLayer

UPSTREAM_TEXTS = [
    "Subtract from all layers",
    "Add to auto layer",
    "Subtract from auto layer",
    "Add to foreground",
    "Add to background",
]


def _buttons(dialog: PictureZoneDialog) -> list[QtWidgets.QRadioButton]:
    return dialog.findChildren(QtWidgets.QRadioButton)


def test_the_title_and_choices_use_the_upstream_strings(qtbot: QtBot):
    dialog = PictureZoneDialog(PictureLayer.PAINTER2)
    qtbot.addWidget(dialog)

    assert dialog.windowTitle() == "Zone Properties"
    assert [button.text() for button in _buttons(dialog)] == UPSTREAM_TEXTS


@pytest.mark.parametrize("layer", [choice[0] for choice in LAYER_CHOICES])
def test_the_current_layer_is_preselected(qtbot: QtBot, layer: PictureLayer):
    dialog = PictureZoneDialog(layer)
    qtbot.addWidget(dialog)

    checked = [button for button in _buttons(dialog) if button.isChecked()]

    assert [button.text() for button in checked] == [layer_label(layer)]
    assert dialog.layer() is layer


def test_a_zone_with_no_layer_starts_with_nothing_chosen(qtbot: QtBot):
    dialog = PictureZoneDialog(PictureLayer.NOOP)
    qtbot.addWidget(dialog)

    assert not any(button.isChecked() for button in _buttons(dialog))
    assert dialog.layer() is PictureLayer.NOOP


def test_choosing_a_layer_reports_it_at_once(qtbot: QtBot):
    dialog = PictureZoneDialog(PictureLayer.PAINTER2)
    qtbot.addWidget(dialog)
    erase = dialog.findChild(QtWidgets.QRadioButton, "zoneeraser1")
    assert erase is not None

    with qtbot.waitSignal(dialog.layer_changed) as signal:
        erase.click()

    assert signal.args == [PictureLayer.ERASER1]
    assert dialog.layer() is PictureLayer.ERASER1


def test_the_ok_and_cancel_buttons_end_the_dialog(qtbot: QtBot):
    dialog = PictureZoneDialog(PictureLayer.PAINTER2)
    qtbot.addWidget(dialog)
    box = dialog.findChild(QtWidgets.QDialogButtonBox)
    assert box is not None

    box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).click()
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted

    box.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel).click()
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Rejected


def test_every_layer_but_noop_has_a_label():
    labels = {layer: layer_label(layer) for layer in PictureLayer}

    assert labels[PictureLayer.NOOP] == ""
    assert all(text for layer, text in labels.items() if layer is not PictureLayer.NOOP)
    assert len({text for text in labels.values() if text}) == 5
