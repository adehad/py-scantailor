"""Options widget for Deskew filter.

Provides UI controls for setting the deskew angle (auto or manual).
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from scantailor.app.ui.widgets import CollapsibleGroupBox
from scantailor.core import PageId
from scantailor.filters.deskew.filter import Filter


class OptionsWidget(QWidget):
    """Options widget for the Deskew filter.

    Provides controls to switch between auto and manual deskew modes,
    a spin box to manually adjust the angle, and an "Apply to..." button.

    Signals:
        angle_changed: Emitted when the deskew angle changes.
            Args: page_id (PageId), angle_deg (float), is_manual (bool)
        apply_to_requested: Emitted when user clicks "Apply to..."
    """

    angle_changed = Signal(PageId, float, bool)
    apply_to_requested = Signal()

    def __init__(
        self,
        filter_: Filter,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the options widget.

        Args:
            filter_: The Deskew filter instance.
            parent: The parent widget.
        """
        super().__init__(parent)

        self._filter = filter_
        self._current_page_id: PageId | None = None
        self._updating = False  # Prevent signal loops

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Deskew group box
        self._deskew_group = CollapsibleGroupBox("Deskew", self)
        self._deskew_group.setObjectName("deskewGroupBox")
        group_layout = QVBoxLayout(self._deskew_group)

        # Auto/Manual buttons
        mode_layout = QHBoxLayout()

        self._auto_btn = QPushButton("Auto")
        self._auto_btn.setObjectName("autoBtn")
        self._auto_btn.setCheckable(True)
        self._auto_btn.setChecked(True)
        self._auto_btn.setAutoExclusive(True)
        self._auto_btn.setToolTip("Automatically detect skew angle")
        mode_layout.addWidget(self._auto_btn)

        self._manual_btn = QPushButton("Manual")
        self._manual_btn.setObjectName("manualBtn")
        self._manual_btn.setCheckable(True)
        self._manual_btn.setAutoExclusive(True)
        self._manual_btn.setToolTip("Manually specify skew angle")
        mode_layout.addWidget(self._manual_btn)

        group_layout.addLayout(mode_layout)

        # Angle spin box row
        angle_layout = QHBoxLayout()
        angle_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )

        self._angle_spinbox = QDoubleSpinBox()
        self._angle_spinbox.setObjectName("angleSpinBox")
        self._angle_spinbox.setRange(-45.0, 45.0)
        self._angle_spinbox.setSingleStep(0.01)
        self._angle_spinbox.setDecimals(2)
        self._angle_spinbox.setSuffix("°")
        self._angle_spinbox.setToolTip("Deskew angle in degrees")
        angle_layout.addWidget(self._angle_spinbox)

        angle_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        group_layout.addLayout(angle_layout)

        # Apply to button row
        apply_layout = QHBoxLayout()
        apply_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )

        self._apply_btn = QPushButton("Apply To ...")
        self._apply_btn.setObjectName("applyDeskewBtn")
        self._apply_btn.setToolTip("Apply this angle to other pages")
        apply_layout.addWidget(self._apply_btn)

        apply_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        group_layout.addLayout(apply_layout)

        layout.addWidget(self._deskew_group)

        # Vertical spacer
        layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        )

    def _connect_signals(self) -> None:
        """Connect widget signals to slots."""
        self._auto_btn.clicked.connect(self._on_auto_clicked)
        self._manual_btn.clicked.connect(self._on_manual_clicked)
        self._angle_spinbox.valueChanged.connect(self._on_angle_changed)
        self._apply_btn.clicked.connect(self.apply_to_requested.emit)

    def set_current_page(self, page_id: PageId) -> None:
        """Set the current page being edited.

        Args:
            page_id: The page ID to edit.
        """
        self._current_page_id = page_id
        self._update_ui()

    def _update_ui(self) -> None:
        """Update the UI to reflect current page settings."""
        if self._current_page_id is None:
            return

        self._updating = True
        try:
            params = self._filter.get_params(self._current_page_id)

            # Update mode buttons
            self._auto_btn.setChecked(params.is_auto())
            self._manual_btn.setChecked(params.is_manual())

            # Update angle spinbox
            self._angle_spinbox.setValue(params.deskew_angle_deg)

            # Enable/disable spinbox based on mode
            self._angle_spinbox.setEnabled(params.is_manual())
        finally:
            self._updating = False

    def _on_auto_clicked(self) -> None:
        """Handle auto mode button click."""
        if self._current_page_id is None or self._updating:
            return

        # Re-detect the skew angle
        # This would typically trigger a re-detection
        params = self._filter.get_params(self._current_page_id)
        new_params = params.with_auto_angle(params.deskew_angle_deg)
        self._filter.settings.set_params(self._current_page_id, new_params)

        self._update_ui()
        self.angle_changed.emit(
            self._current_page_id, new_params.deskew_angle_deg, False
        )

    def _on_manual_clicked(self) -> None:
        """Handle manual mode button click."""
        if self._current_page_id is None or self._updating:
            return

        params = self._filter.get_params(self._current_page_id)
        new_params = params.with_manual_angle(params.deskew_angle_deg)
        self._filter.settings.set_params(self._current_page_id, new_params)

        self._update_ui()
        self.angle_changed.emit(
            self._current_page_id, new_params.deskew_angle_deg, True
        )

    def _on_angle_changed(self, value: float) -> None:
        """Handle angle spinbox value change.

        Args:
            value: The new angle value in degrees.
        """
        if self._current_page_id is None or self._updating:
            return

        params = self._filter.get_params(self._current_page_id)
        # When manually editing, always use manual mode
        new_params = params.with_manual_angle(value)
        self._filter.settings.set_params(self._current_page_id, new_params)

        # Update mode button if needed
        if not self._manual_btn.isChecked():
            self._updating = True
            self._manual_btn.setChecked(True)
            self._updating = False

        self.angle_changed.emit(self._current_page_id, value, True)

    def set_angle(self, angle: float, is_manual: bool = False) -> None:
        """Set the displayed angle (called from external sources like ImageView).

        Args:
            angle: The angle in degrees.
            is_manual: Whether this is a manual setting.
        """
        self._updating = True
        try:
            self._angle_spinbox.setValue(angle)
            self._manual_btn.setChecked(is_manual)
            self._auto_btn.setChecked(not is_manual)
            self._angle_spinbox.setEnabled(is_manual)
        finally:
            self._updating = False
