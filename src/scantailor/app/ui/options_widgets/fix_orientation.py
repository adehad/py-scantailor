"""Options widget for Fix Orientation filter.

Provides UI controls for rotating images by 90-degree increments and for
trimming the edges of an image by hand.
"""

from PySide6 import QtCore
from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from scantailor.app.ui.icons import IconLabel, icon
from scantailor.app.ui.widgets.collapsible_group_box import CollapsibleGroupBox
from scantailor.core.models import ImageId, OrthogonalRotation
from scantailor.filters.fix_orientation.filter import Filter
from scantailor.filters.fix_orientation.image_trim import (
    MIN_TRIM_INNER_SIDE,
    ImageTrim,
)

_ARROWS = {0: "arrow-up", 90: "arrow-right", 180: "arrow-down", 270: "arrow-left"}
_INDICATOR_SIZE = 32
_TRIM_ICON_SIZE = 16
# Upper limit of a trim spin box while the image's size is not known.
_UNKNOWN_SIZE_MAXIMUM = 99999


class OptionsWidget(QWidget):
    """Options widget for the Fix Orientation filter.

    Provides buttons to rotate the current image left or right by 90 degrees,
    a visual indicator of the current rotation, and buttons to reset or
    apply the rotation to multiple pages. The trim group cuts a number of
    source pixels from each edge of the image.

    Signals:
        rotation_changed: Emitted when the rotation is changed.
            Args: image_id (ImageId), new_rotation (OrthogonalRotation)
        trim_changed: Emitted when the trim of the current image is changed.
            Args: image_id (ImageId)
        apply_to_requested: Emitted when user clicks "Apply to..."
    """

    rotation_changed = Signal(ImageId, OrthogonalRotation)
    trim_changed = Signal(ImageId)
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
        self._image_size: tuple[int, int] | None = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Rotate group box
        self._rotate_group = CollapsibleGroupBox(
            QtCore.QCoreApplication.translate(
                "fix_orientation::OptionsWidget", "Rotate"
            ),
            self,
        )
        self._rotate_group.setObjectName("rotateGroupBox")
        rotate_layout = QVBoxLayout(self._rotate_group)

        # Rotation buttons row
        buttons_layout = QHBoxLayout()
        buttons_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )

        self._rotate_left_btn = QToolButton()
        self._rotate_left_btn.setObjectName("rotateLeftBtn")
        self._rotate_left_btn.setToolTip(
            QtCore.QCoreApplication.translate(
                "fix_orientation::OptionsWidget", "Rotate 90° counter-clockwise"
            )
        )
        self._rotate_left_btn.setIconSize(self._rotate_left_btn.iconSize() * 1.5)
        self._rotate_left_btn.setIcon(icon("rotate-ccw-square"))
        buttons_layout.addWidget(self._rotate_left_btn)

        self._rotate_right_btn = QToolButton()
        self._rotate_right_btn.setObjectName("rotateRightBtn")
        self._rotate_right_btn.setToolTip(
            QtCore.QCoreApplication.translate(
                "fix_orientation::OptionsWidget", "Rotate 90° clockwise"
            )
        )
        self._rotate_right_btn.setIconSize(self._rotate_right_btn.iconSize() * 1.5)
        self._rotate_right_btn.setIcon(icon("rotate-cw-square"))
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

        self._rotation_indicator = IconLabel(_ARROWS[0], _INDICATOR_SIZE)
        self._rotation_indicator.setObjectName("rotationIndicator")
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

        self._reset_btn = QPushButton(
            QtCore.QCoreApplication.translate("fix_orientation::OptionsWidget", "Reset")
        )
        self._reset_btn.setObjectName("resetBtn")
        self._reset_btn.setToolTip(
            QtCore.QCoreApplication.translate(
                "fix_orientation::OptionsWidget", "Reset rotation to 0°"
            )
        )
        reset_layout.addWidget(self._reset_btn)

        reset_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        rotate_layout.addLayout(reset_layout)

        layout.addWidget(self._rotate_group)
        layout.addWidget(self._build_trim_group())

        # Apply to button
        apply_layout = QHBoxLayout()
        apply_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )

        self._apply_to_btn = QPushButton(
            QtCore.QCoreApplication.translate(
                "fix_orientation::OptionsWidget", "Apply to ..."
            )
        )
        self._apply_to_btn.setObjectName("applyToBtn")
        self._apply_to_btn.setToolTip(
            QtCore.QCoreApplication.translate(
                "fix_orientation::OptionsWidget",
                "Copy this image's rotation and trim to other pages you choose.",
            )
        )
        apply_layout.addWidget(self._apply_to_btn)

        apply_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        layout.addLayout(apply_layout)

        # Vertical spacer
        layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        )

    def _build_trim_group(self) -> CollapsibleGroupBox:
        """Build the group box with the manual trim controls."""
        group = CollapsibleGroupBox(
            QtCore.QCoreApplication.translate(
                "fix_orientation::OptionsWidget", "Crop scan (trim)"
            ),
            self,
        )
        group.setObjectName("trimGroupBox")
        group_layout = QVBoxLayout(group)

        enable_layout = QHBoxLayout()
        self._trim_enabled_check = QCheckBox(
            QtCore.QCoreApplication.translate(
                "fix_orientation::OptionsWidget", "Enable manual trim"
            )
        )
        self._trim_enabled_check.setObjectName("trimEnabledCheck")
        self._trim_enabled_check.setToolTip(
            QtCore.QCoreApplication.translate(
                "fix_orientation::OptionsWidget",
                "Cut strips off the edges of the scan before any other stage sees "
                "it. Use it to drop scanner bed, fingers or a neighbouring page "
                "that would confuse page and content detection.",
            )
        )
        enable_layout.addWidget(self._trim_enabled_check)
        enable_layout.addWidget(IconLabel("crop", _TRIM_ICON_SIZE))
        enable_layout.addStretch(1)
        group_layout.addLayout(enable_layout)

        self._trim_top_spin = self._new_trim_spin("trimTopSpin")
        self._trim_left_spin = self._new_trim_spin("trimLeftSpin")
        self._trim_right_spin = self._new_trim_spin("trimRightSpin")
        self._trim_bottom_spin = self._new_trim_spin("trimBottomSpin")
        self._trim_spins = (
            self._trim_left_spin,
            self._trim_right_spin,
            self._trim_top_spin,
            self._trim_bottom_spin,
        )

        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        left = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        center = Qt.AlignmentFlag.AlignCenter
        self._trim_corner_labels = (
            self._new_corner_label("labelTrimTopLeft", "|=", right),
            self._new_corner_label("labelTrimTopRight", "=|", left),
            self._new_corner_label("labelTrimCenter", "-|-", center),
            self._new_corner_label("labelTrimBottomLeft", "|___", right),
            self._new_corner_label("labelTrimBottomRight", "___|", left),
        )
        top_left, top_right, middle, bottom_left, bottom_right = (
            self._trim_corner_labels
        )

        # One help text covers all four sides, so the cross of spin boxes is
        # not broken up by an icon in every cell.
        trim_sides = QWidget()
        trim_sides.setObjectName("trimSides")
        trim_sides.setToolTip(
            QtCore.QCoreApplication.translate(
                "fix_orientation::OptionsWidget",
                "Pixels to cut from the top, left, right and bottom of the scan, "
                "counted on the image as stored, before rotation.",
            )
        )
        grid = QGridLayout(trim_sides)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(top_left, 0, 0)
        grid.addWidget(self._trim_top_spin, 0, 1)
        grid.addWidget(top_right, 0, 2)
        grid.addWidget(self._trim_left_spin, 1, 0)
        grid.addWidget(middle, 1, 1)
        grid.addWidget(self._trim_right_spin, 1, 2)
        grid.addWidget(bottom_left, 2, 0)
        grid.addWidget(self._trim_bottom_spin, 2, 1)
        grid.addWidget(bottom_right, 2, 2)
        group_layout.addWidget(trim_sides)

        self._reset_trim_btn = QPushButton(
            QtCore.QCoreApplication.translate(
                "fix_orientation::OptionsWidget", "Reset trim"
            )
        )
        self._reset_trim_btn.setObjectName("resetTrimBtn")
        self._reset_trim_btn.setToolTip(
            QtCore.QCoreApplication.translate(
                "fix_orientation::OptionsWidget",
                "Set all four trim amounts back to zero, keeping the whole scan.",
            )
        )
        self._reset_trim_btn.setIcon(icon("undo-2"))
        group_layout.addWidget(self._reset_trim_btn)

        self._set_trim_controls_enabled(False)
        return group

    @staticmethod
    def _new_trim_spin(name: str) -> QSpinBox:
        spin = QSpinBox()
        spin.setObjectName(name)
        spin.setMinimum(0)
        spin.setMaximum(_UNKNOWN_SIZE_MAXIMUM)
        return spin

    @staticmethod
    def _new_corner_label(name: str, text: str, alignment: Qt.AlignmentFlag) -> QLabel:
        label = QLabel(text)
        label.setObjectName(name)
        label.setAlignment(alignment)
        return label

    def _connect_signals(self) -> None:
        """Connect widget signals to slots."""
        self._trim_enabled_check.toggled.connect(self._on_trim_enable_toggled)
        for spin in self._trim_spins:
            spin.valueChanged.connect(self._on_trim_margins_changed)
        self._reset_trim_btn.clicked.connect(self._on_reset_trim)
        self._rotate_left_btn.clicked.connect(self._on_rotate_left)
        self._rotate_right_btn.clicked.connect(self._on_rotate_right)
        self._reset_btn.clicked.connect(self._on_reset)
        self._apply_to_btn.clicked.connect(self.apply_to_requested.emit)

    def set_current_image(self, image_id: ImageId) -> None:
        """Set the current image being edited.

        Args:
            image_id: The image ID to edit.
        """
        if image_id != self._current_image_id:
            self._image_size = None
        self._current_image_id = image_id
        self._update_indicator()
        self._pull_trim_to_controls()

    def set_image_size(self, width: int, height: int) -> None:
        """Tell the widget how large the current image is, as stored.

        The trim cannot leave less than a minimum side, so the spin boxes'
        limits follow the image's size.

        Args:
            width (int): Width of the stored image in pixels.
            height (int): Height of the stored image in pixels.
        """
        self._image_size = (width, height)
        self._update_trim_maximums()

    def _update_indicator(self) -> None:
        """Update the rotation indicator to show current rotation."""
        if self._current_image_id is None:
            self._rotation_indicator.set_glyph(_ARROWS[0])
            return

        rotation = self._filter.get_rotation(self._current_image_id)
        self._rotation_indicator.set_glyph(_ARROWS.get(rotation.degrees, _ARROWS[0]))

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

    def get_current_trim(self) -> ImageTrim:
        """Get the trim shown in the controls.

        Returns:
            ImageTrim: Disabled when the check box is off.
        """
        if not self._trim_enabled_check.isChecked():
            return ImageTrim()
        return ImageTrim(
            enabled=True,
            left=self._trim_left_spin.value(),
            top=self._trim_top_spin.value(),
            right=self._trim_right_spin.value(),
            bottom=self._trim_bottom_spin.value(),
        )

    def _set_trim_controls_enabled(self, enabled: bool) -> None:
        for control in (
            *self._trim_spins,
            *self._trim_corner_labels,
            self._reset_trim_btn,
        ):
            control.setEnabled(enabled)

    def _update_trim_maximums(self) -> None:
        """Keep each pair of opposite margins from consuming the whole image."""
        if self._image_size is None:
            return
        width, height = self._image_size
        max_horizontal = max(0, width - MIN_TRIM_INNER_SIDE)
        max_vertical = max(0, height - MIN_TRIM_INNER_SIDE)
        blockers = [QSignalBlocker(spin) for spin in self._trim_spins]
        self._trim_left_spin.setMaximum(
            max(0, max_horizontal - self._trim_right_spin.value())
        )
        self._trim_right_spin.setMaximum(
            max(0, max_horizontal - self._trim_left_spin.value())
        )
        self._trim_top_spin.setMaximum(
            max(0, max_vertical - self._trim_bottom_spin.value())
        )
        self._trim_bottom_spin.setMaximum(
            max(0, max_vertical - self._trim_top_spin.value())
        )
        del blockers

    def _pull_trim_to_controls(self) -> None:
        """Show the stored trim of the current image without storing anything."""
        if self._current_image_id is None:
            return
        trim = self._filter.get_trim(self._current_image_id)
        blockers = [QSignalBlocker(self._trim_enabled_check)]
        blockers += [QSignalBlocker(spin) for spin in self._trim_spins]
        for spin in self._trim_spins:
            spin.setMaximum(_UNKNOWN_SIZE_MAXIMUM)
        self._trim_enabled_check.setChecked(trim.enabled)
        self._trim_left_spin.setValue(trim.left)
        self._trim_right_spin.setValue(trim.right)
        self._trim_top_spin.setValue(trim.top)
        self._trim_bottom_spin.setValue(trim.bottom)
        self._set_trim_controls_enabled(trim.enabled)
        del blockers
        self._update_trim_maximums()

    def _push_trim_from_controls(self) -> None:
        """Store the trim shown in the controls and announce it."""
        if self._current_image_id is None:
            return
        trim = self.get_current_trim()
        if not trim.enabled:
            return
        self._filter.set_trim(self._current_image_id, trim)
        self.trim_changed.emit(self._current_image_id)

    def _on_trim_enable_toggled(self, checked: bool) -> None:
        self._set_trim_controls_enabled(checked)
        if self._current_image_id is None:
            return
        if not checked:
            self._filter.set_trim(self._current_image_id, ImageTrim())
            self.trim_changed.emit(self._current_image_id)
            return
        self._push_trim_from_controls()

    def _on_trim_margins_changed(self, _value: int) -> None:
        if not self._trim_enabled_check.isChecked():
            return
        self._update_trim_maximums()
        self._push_trim_from_controls()

    def _on_reset_trim(self) -> None:
        blockers = [QSignalBlocker(spin) for spin in self._trim_spins]
        for spin in self._trim_spins:
            spin.setValue(0)
        del blockers
        self._update_trim_maximums()
        if self._trim_enabled_check.isChecked():
            self._push_trim_from_controls()
        elif self._current_image_id is not None:
            self._filter.set_trim(self._current_image_id, ImageTrim())
            self.trim_changed.emit(self._current_image_id)

    def get_current_rotation(self) -> OrthogonalRotation:
        """Get the current rotation for the active image.

        Returns:
            The current rotation, or 0 degrees if no image is selected.
        """
        if self._current_image_id is None:
            return OrthogonalRotation(degrees=0)
        return self._filter.get_rotation(self._current_image_id)
