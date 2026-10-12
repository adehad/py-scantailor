"""Options widget for Deskew filter.

Provides UI controls for setting the deskew angle and the oblique (shear)
correction, each automatic or manual.
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from scantailor.app.ui.widgets.collapsible_group_box import CollapsibleGroupBox
from scantailor.core.models import PageId
from scantailor.filters.deskew.filter import Filter
from scantailor.filters.deskew.params import MAX_OBLIQUE_DEG


class OptionsWidget(QWidget):
    """Options widget for the Deskew filter.

    Provides controls to switch between auto and manual deskew modes,
    a spin box to manually adjust the angle, the same for the oblique
    correction, and an "Apply to..." button.

    Signals:
        angle_changed: Emitted when the deskew angle changes.
            Args: page_id (PageId), angle_deg (float), is_manual (bool)
        oblique_changed: Emitted when the oblique angle or its mode changes.
            Args: page_id (PageId), angle_deg (float), is_manual (bool)
        auto_oblique_default_changed: Emitted when the "Automatic oblique
            correction" check box is toggled. Args: enabled (bool)
        apply_to_requested: Emitted when user clicks "Apply to..."
    """

    angle_changed = Signal(PageId, float, bool)
    oblique_changed = Signal(PageId, float, bool)
    auto_oblique_default_changed = Signal(bool)
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
        self._show_default()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Deskew group box
        self._deskew_group = CollapsibleGroupBox(self.tr("Deskew"), self)
        self._deskew_group.setObjectName("deskewGroupBox")
        group_layout = QVBoxLayout(self._deskew_group)

        # Auto/Manual buttons
        mode_layout = QHBoxLayout()

        self._auto_btn = QPushButton(self.tr("Auto"))
        self._auto_btn.setObjectName("autoBtn")
        self._auto_btn.setCheckable(True)
        self._auto_btn.setChecked(True)
        self._auto_btn.setAutoExclusive(True)
        self._auto_btn.setToolTip("Automatically detect skew angle")
        mode_layout.addWidget(self._auto_btn)

        self._manual_btn = QPushButton(self.tr("Manual"))
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

        self._auto_oblique_check = QCheckBox(self.tr("Automatic oblique correction"))
        self._auto_oblique_check.setObjectName("autoObliqueCheckBox")
        self._auto_oblique_check.setToolTip(
            self.tr(
                "When deskew mode is Auto, run automatic oblique (shear) correction. "
                "Disable for pages that only need rotation."
            )
        )
        group_layout.addWidget(self._auto_oblique_check)

        # Oblique angle row
        oblique_layout = QHBoxLayout()
        self._oblique_label = QLabel(self.tr("Oblique (°):"))
        self._oblique_label.setObjectName("obliqueLabel")
        oblique_layout.addWidget(self._oblique_label)

        self._oblique_spinbox = QDoubleSpinBox()
        self._oblique_spinbox.setObjectName("obliqueSpinBox")
        self._oblique_spinbox.setRange(-MAX_OBLIQUE_DEG, MAX_OBLIQUE_DEG)
        self._oblique_spinbox.setSingleStep(0.1)
        self._oblique_spinbox.setDecimals(2)
        oblique_layout.addWidget(self._oblique_spinbox)
        group_layout.addLayout(oblique_layout)

        # Oblique mode row
        oblique_mode_layout = QHBoxLayout()
        self._oblique_mode_label = QLabel(self.tr("Oblique mode"))
        self._oblique_mode_label.setObjectName("obliqueModeLabel")
        oblique_mode_layout.addWidget(self._oblique_mode_label)

        self._oblique_auto_btn = QPushButton(self.tr("Auto"))
        self._oblique_auto_btn.setObjectName("obliqueAutoBtn")
        self._oblique_auto_btn.setCheckable(True)
        self._oblique_auto_btn.setAutoExclusive(True)
        self._oblique_auto_btn.setToolTip(
            self.tr("Automatically estimate oblique (shear) correction.")
        )
        oblique_mode_layout.addWidget(self._oblique_auto_btn)

        self._oblique_manual_btn = QPushButton(self.tr("Manual"))
        self._oblique_manual_btn.setObjectName("obliqueManualBtn")
        self._oblique_manual_btn.setCheckable(True)
        self._oblique_manual_btn.setChecked(True)
        self._oblique_manual_btn.setAutoExclusive(True)
        self._oblique_manual_btn.setToolTip(
            self.tr("Keep the oblique angle set in the spin box.")
        )
        oblique_mode_layout.addWidget(self._oblique_manual_btn)
        group_layout.addLayout(oblique_mode_layout)

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
        self._auto_oblique_check.toggled.connect(self._on_auto_oblique_toggled)
        self._oblique_auto_btn.clicked.connect(self._on_oblique_auto_clicked)
        self._oblique_manual_btn.clicked.connect(self._on_oblique_manual_clicked)
        self._oblique_spinbox.valueChanged.connect(self._on_oblique_value_changed)
        self._apply_btn.clicked.connect(self.apply_to_requested.emit)

    def _show_default(self) -> None:
        """Show a fresh page: automatic rotation, oblique per the default."""
        self._updating = True
        try:
            self._auto_oblique_check.setChecked(self._filter.auto_oblique_default)
            self._show_oblique_mode(self._filter.auto_oblique_default)
        finally:
            self._updating = False

    def set_current_page(self, page_id: PageId) -> None:
        """Set the current page being edited.

        Args:
            page_id: The page ID to edit.
        """
        self._current_page_id = page_id
        self.refresh()

    def refresh(self) -> None:
        """Show the stored settings of the current page, e.g. a detected angle."""
        if self._current_page_id is None:
            return

        self._updating = True
        try:
            params = self._filter.get_params(self._current_page_id)

            # Update mode buttons
            self._auto_btn.setChecked(params.is_auto())
            self._manual_btn.setChecked(params.is_manual())

            # Setting an equal value would rewrite the text the user is typing.
            if self._angle_spinbox.value() != params.deskew_angle_deg:
                self._angle_spinbox.setValue(params.deskew_angle_deg)

            # Enable/disable spinbox based on mode
            self._angle_spinbox.setEnabled(params.is_manual())

            self._auto_oblique_check.setChecked(self._filter.auto_oblique_default)
            self._show_oblique_mode(params.is_oblique_auto())
            if self._oblique_spinbox.value() != params.oblique_angle_deg:
                self._oblique_spinbox.setValue(params.oblique_angle_deg)
        finally:
            self._updating = False

    def _show_oblique_mode(self, auto: bool) -> None:
        self._oblique_auto_btn.setChecked(auto)
        self._oblique_manual_btn.setChecked(not auto)

    def _on_auto_clicked(self) -> None:
        """Detect the rotation again; the oblique correction is left alone."""
        if self._current_page_id is None or self._updating:
            return

        self._filter.redetect_rotation(self._current_page_id)

        self.refresh()
        self.angle_changed.emit(
            self._current_page_id,
            self._filter.get_deskew_angle(self._current_page_id),
            False,
        )

    def _on_manual_clicked(self) -> None:
        """Handle manual mode button click."""
        if self._current_page_id is None or self._updating:
            return

        params = self._filter.get_params(self._current_page_id)
        new_params = params.with_manual_angle(params.deskew_angle_deg)
        self._filter.settings.set_params(self._current_page_id, new_params)

        self.refresh()
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

    def _on_auto_oblique_toggled(self, enabled: bool) -> None:
        """Change the default for new pages, and the page itself while it is automatic.

        Args:
            enabled: Whether oblique correction is detected by default.
        """
        if self._updating:
            return

        self._filter.auto_oblique_default = enabled
        self.auto_oblique_default_changed.emit(enabled)

        page_id = self._current_page_id
        if page_id is None or not self._filter.get_params(page_id).is_auto():
            return
        if enabled:
            self._filter.redetect_oblique(page_id)
        else:
            self._filter.set_manual_oblique(page_id, 0.0)
        self.refresh()
        self.oblique_changed.emit(
            page_id,
            self._filter.get_params(page_id).oblique_angle_deg,
            not enabled,
        )

    def _on_oblique_auto_clicked(self) -> None:
        """Detect the oblique angle again; the rotation is left alone."""
        if self._current_page_id is None or self._updating:
            return

        self._filter.redetect_oblique(self._current_page_id)

        self.refresh()
        self.oblique_changed.emit(
            self._current_page_id,
            self._filter.get_params(self._current_page_id).oblique_angle_deg,
            False,
        )

    def _on_oblique_manual_clicked(self) -> None:
        """Keep the oblique angle as it is, from now on edited by hand."""
        if self._current_page_id is None or self._updating:
            return

        params = self._filter.get_params(self._current_page_id)
        self._filter.set_manual_oblique(self._current_page_id, params.oblique_angle_deg)

        self.refresh()
        self.oblique_changed.emit(self._current_page_id, params.oblique_angle_deg, True)

    def _on_oblique_value_changed(self, value: float) -> None:
        """Handle the oblique spinbox changing; an edit makes the angle manual.

        Args:
            value: The new oblique angle in degrees.
        """
        if self._current_page_id is None or self._updating:
            return

        self._filter.set_manual_oblique(self._current_page_id, value)

        self._updating = True
        try:
            self._show_oblique_mode(False)
        finally:
            self._updating = False

        self.oblique_changed.emit(self._current_page_id, value, True)

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

    def set_oblique(self, angle: float, is_manual: bool = False) -> None:
        """Set the displayed oblique angle (called from external sources like the view).

        Args:
            angle: The angle in degrees.
            is_manual: Whether this is a manual setting.
        """
        self._updating = True
        try:
            self._oblique_spinbox.setValue(angle)
            self._show_oblique_mode(not is_manual)
        finally:
            self._updating = False
