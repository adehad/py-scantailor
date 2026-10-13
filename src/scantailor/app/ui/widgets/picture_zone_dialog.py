"""Dialog that sets which layer a picture zone edits."""

from PySide6.QtCore import Signal
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
LAYER_CHOICES: tuple[tuple[PictureLayer, str, str], ...] = (
    (PictureLayer.ERASER3, "zoneeraser3", "Subtract from all layers"),
    (PictureLayer.PAINTER2, "zonepainter2", "Add to auto layer"),
    (PictureLayer.ERASER1, "zoneeraser1", "Subtract from auto layer"),
    (PictureLayer.FOREGROUND, "zoneforeground", "Add to foreground"),
    (PictureLayer.BACKGROUND, "zonebackground", "Add to background"),
)


def layer_label(layer: PictureLayer) -> str:
    """Return the dialog's wording for a layer.

    Args:
        layer (PictureLayer): The layer.

    Returns:
        str: What choosing the layer does; empty for ``PictureLayer.NOOP``, which
            the dialog does not offer.
    """
    return next((text for kind, _name, text in LAYER_CHOICES if kind is layer), "")


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
        self.setWindowTitle("Zone Properties")
        self._layer = layer
        self._buttons: dict[PictureLayer, QRadioButton] = {}
        self._group = QButtonGroup(self)

        layout = QVBoxLayout(self)
        for kind, name, text in LAYER_CHOICES:
            button = QRadioButton(text)
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
