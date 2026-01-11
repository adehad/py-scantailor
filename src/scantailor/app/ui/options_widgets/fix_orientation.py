"""Options widget for Fix Orientation filter.

Provides UI controls for rotating images by 90-degree increments.
"""

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from scantailor.app.ui.widgets import CollapsibleGroupBox
from scantailor.core.models import ImageId, OrthogonalRotation

if TYPE_CHECKING:
    from scantailor.filters.fix_orientation.filter import Filter


class OptionsWidget(QWidget):
    """Options widget for the Fix Orientation filter.

    Provides buttons to rotate the current image left or right by 90 degrees,
    a visual indicator of the current rotation, and buttons to reset or
    apply the rotation to multiple pages.

    Signals:
        rotation_changed: Emitted when the rotation is changed.
            Args: image_id (ImageId), new_rotation (OrthogonalRotation)
        apply_to_requested: Emitted when user clicks "Apply to..."
    """

    rotation_changed = Signal(ImageId, OrthogonalRotation)
    apply_to_requested = Signal()

    def __init__(
        self,
        filter_: Filter,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the options widget.

        Args:
            filter_: The Fix Orientation filter instance.
            parent: The parent widget.
        """
        super().__init__(parent)

        self._filter = filter_
        self._current_image_id: ImageId | None = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Rotate group box
        self._rotate_group = CollapsibleGroupBox("Rotate", self)
        self._rotate_group.setObjectName("rotateGroupBox")
        rotate_layout = QVBoxLayout(self._rotate_group)

        # Rotation buttons row
        buttons_layout = QHBoxLayout()
        buttons_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )

        self._rotate_left_btn = QToolButton()
        self._rotate_left_btn.setObjectName("rotateLeftBtn")
        self._rotate_left_btn.setToolTip("Rotate 90° counter-clockwise")
        self._rotate_left_btn.setIconSize(self._rotate_left_btn.iconSize() * 1.5)
        # Use text as fallback if icons not available
        self._rotate_left_btn.setText("↶")
        buttons_layout.addWidget(self._rotate_left_btn)

        self._rotate_right_btn = QToolButton()
        self._rotate_right_btn.setObjectName("rotateRightBtn")
        self._rotate_right_btn.setToolTip("Rotate 90° clockwise")
        self._rotate_right_btn.setIconSize(self._rotate_right_btn.iconSize() * 1.5)
        self._rotate_right_btn.setText("↷")
        buttons_layout.addWidget(self._rotate_right_btn)

        buttons_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        rotate_layout.addLayout(buttons_layout)

        # Rotation indicator row
        indicator_layout = QHBoxLayout()
        indicator_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )

        self._rotation_indicator = QLabel()
        self._rotation_indicator.setObjectName("rotationIndicator")
        self._rotation_indicator.setText("↑")  # Up arrow as default
        self._rotation_indicator.setStyleSheet("font-size: 32px; font-weight: bold;")
        indicator_layout.addWidget(self._rotation_indicator)

        indicator_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        rotate_layout.addLayout(indicator_layout)

        # Reset button row
        reset_layout = QHBoxLayout()
        reset_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )

        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setObjectName("resetBtn")
        self._reset_btn.setToolTip("Reset rotation to 0°")
        reset_layout.addWidget(self._reset_btn)

        reset_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        rotate_layout.addLayout(reset_layout)

        layout.addWidget(self._rotate_group)

        # Apply to button
        apply_layout = QHBoxLayout()
        apply_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )

        self._apply_to_btn = QPushButton("Apply to ...")
        self._apply_to_btn.setObjectName("applyToBtn")
        self._apply_to_btn.setToolTip("Apply this rotation to other pages")
        apply_layout.addWidget(self._apply_to_btn)

        apply_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        layout.addLayout(apply_layout)

        # Vertical spacer
        layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        )

    def _connect_signals(self) -> None:
        """Connect widget signals to slots."""
        self._rotate_left_btn.clicked.connect(self._on_rotate_left)
        self._rotate_right_btn.clicked.connect(self._on_rotate_right)
        self._reset_btn.clicked.connect(self._on_reset)
        self._apply_to_btn.clicked.connect(self.apply_to_requested.emit)

    def set_current_image(self, image_id: ImageId) -> None:
        """Set the current image being edited.

        Args:
            image_id: The image ID to edit.
        """
        self._current_image_id = image_id
        self._update_indicator()

    def _update_indicator(self) -> None:
        """Update the rotation indicator to show current rotation."""
        if self._current_image_id is None:
            self._rotation_indicator.setText("↑")
            return

        rotation = self._filter.get_rotation(self._current_image_id)
        arrows = {
            0: "↑",
            90: "→",
            180: "↓",
            270: "←",
        }
        self._rotation_indicator.setText(arrows.get(rotation.degrees, "↑"))

    def _on_rotate_left(self) -> None:
        """Handle rotate left button click."""
        if self._current_image_id is None:
            return

        new_rotation = self._filter.rotate_counter_clockwise(self._current_image_id)
        self._update_indicator()
        self.rotation_changed.emit(self._current_image_id, new_rotation)

    def _on_rotate_right(self) -> None:
        """Handle rotate right button click."""
        if self._current_image_id is None:
            return

        new_rotation = self._filter.rotate_clockwise(self._current_image_id)
        self._update_indicator()
        self.rotation_changed.emit(self._current_image_id, new_rotation)

    def _on_reset(self) -> None:
        """Handle reset button click."""
        if self._current_image_id is None:
            return

        self._filter.reset_rotation(self._current_image_id)
        self._update_indicator()
        self.rotation_changed.emit(
            self._current_image_id, OrthogonalRotation(degrees=0)
        )

    def get_current_rotation(self) -> OrthogonalRotation:
        """Get the current rotation for the active image.

        Returns:
            The current rotation, or 0 degrees if no image is selected.
        """
        if self._current_image_id is None:
            return OrthogonalRotation(degrees=0)
        return self._filter.get_rotation(self._current_image_id)
