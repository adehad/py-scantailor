"""Dialog that sets which layer a picture zone edits."""

from typing import cast

from PySide6.QtCore import QT_TRANSLATE_NOOP, QCoreApplication, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from scantailor.core.zones import PictureLayer

# In the order and with the names of the upstream dialog.
_DIALOG_CONTEXT = "output::PictureZonePropDialog"
LAYER_CHOICES: tuple[tuple[PictureLayer, str, str], ...] = (
    (
        PictureLayer.ERASER3,
        "zoneeraser3",
        cast(str, QT_TRANSLATE_NOOP(_DIALOG_CONTEXT, "Subtract from all layers")),
    ),
    (
        PictureLayer.PAINTER2,
        "zonepainter2",
        cast(str, QT_TRANSLATE_NOOP(_DIALOG_CONTEXT, "Add to auto layer")),
    ),
    (
        PictureLayer.ERASER1,
        "zoneeraser1",
        cast(str, QT_TRANSLATE_NOOP(_DIALOG_CONTEXT, "Subtract from auto layer")),
    ),
    (
        PictureLayer.FOREGROUND,
        "zoneforeground",
        cast(str, QT_TRANSLATE_NOOP(_DIALOG_CONTEXT, "Add to foreground")),
    ),
    (
        PictureLayer.BACKGROUND,
        "zonebackground",
        cast(str, QT_TRANSLATE_NOOP(_DIALOG_CONTEXT, "Add to background")),
    ),
)


def layer_label(layer: PictureLayer) -> str:
    """Return the dialog's wording for a layer.

    Args:
        layer (PictureLayer): The layer.

    Returns:
        str: What choosing the layer does; empty for ``PictureLayer.NOOP``, which
            the dialog does not offer.
    """
    text = next((text for kind, _name, text in LAYER_CHOICES if kind is layer), "")
    return QCoreApplication.translate(_DIALOG_CONTEXT, text)


class PictureZoneDialog(QDialog):
    """Radio buttons for the layer of a picture zone.

    The choice is reported as it changes, so that the view can show it before the
    dialog is accepted.

    Signals:
        layer_changed: The user chose another layer (PictureLayer).
    """

    layer_changed = Signal(PictureLayer)

    def __init__(self, layer: PictureLayer, parent: QWidget | None = None) -> None:
        """Create the dialog.

        Args:
            layer (PictureLayer): The zone's current layer, preselected.
            parent (QWidget | None): Parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle(
            QCoreApplication.translate(_DIALOG_CONTEXT, "Zone Properties")
        )
        self._layer = layer
        self._buttons: dict[PictureLayer, QRadioButton] = {}
        self._group = QButtonGroup(self)

        layout = QVBoxLayout(self)
        for kind, name, text in LAYER_CHOICES:
            button = QRadioButton(QCoreApplication.translate(_DIALOG_CONTEXT, text))
            button.setObjectName(name)
            button.setChecked(kind is layer)
            button.toggled.connect(
                lambda checked, kind=kind: checked and self._choose(kind)
            )
            self._group.addButton(button)
            self._buttons[kind] = button
            layout.addWidget(button)

        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)

    def layer(self) -> PictureLayer:
        """Return the chosen layer.

        Returns:
            PictureLayer: The selected layer, or the one the dialog started
                with when nothing was chosen.
        """
        return self._layer

    def _choose(self, layer: PictureLayer) -> None:
        self._layer = layer
        self.layer_changed.emit(layer)
