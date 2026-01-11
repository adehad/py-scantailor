"""Default parameters dialog for setting filter defaults."""

from pathlib import Path

from PySide6 import QtWidgets
from PySide6.QtCore import Signal
from PySide6.QtGui import QPixmap

from scantailor.app.ui import load_ui_widget
from scantailor.app.ui.utils import get_cwidget
from scantailor.core.settings import DefaultParams

_UI_FOLDER = Path(__file__).parent


class DefaultParamsDialog(QtWidgets.QDialog):
    """Dialog for configuring default parameters for all filter stages.

    Provides tabs for each filter stage:
    - Fix Orientation: Default rotation
    - Split Pages: Page layout type and split line mode
    - Deskew: Auto/manual mode and angle
    - Select Content: Content detection settings
    - Margins: Default margins and alignment
    - Output: Color mode, binarization, dewarping, etc.

    Also supports saving/loading parameter profiles.

    Signals:
        params_changed: Emitted when parameters are changed.
    """

    params_changed = Signal()

    def __init__(
        self,
        default_params: DefaultParams | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the default parameters dialog.

        Args:
            default_params: Current default parameters to edit.
            parent: Parent widget.
        """
        super().__init__(parent)
        self._params = default_params
        self._current_rotation = 0  # 0, 90, 180, 270

        # Load UI
        self.ui = load_ui_widget(_UI_FOLDER / "DefaultParamsDialog.ui", self)
        self._setup_widgets()
        self._load_params()
        self._connect_signals()

    def _setup_widgets(self) -> None:
        """Set up widget references."""
        # Profile management
        self._profile_cb = get_cwidget(self.ui, QtWidgets.QComboBox, "profileCB")
        self._profile_save_btn = get_cwidget(
            self.ui, QtWidgets.QPushButton, "profileSaveButton"
        )
        self._profile_delete_btn = get_cwidget(
            self.ui, QtWidgets.QPushButton, "profileDeleteButton"
        )
        self._units_label = get_cwidget(self.ui, QtWidgets.QLabel, "unitsLabel")

        # Tab widget
        self._tab_widget = get_cwidget(self.ui, QtWidgets.QTabWidget, "tabWidget")

        # === Fix Orientation tab ===
        self._rotate_left_btn = get_cwidget(
            self.ui, QtWidgets.QToolButton, "rotateLeftBtn"
        )
        self._rotate_right_btn = get_cwidget(
            self.ui, QtWidgets.QToolButton, "rotateRightBtn"
        )
        self._rotation_indicator = get_cwidget(
            self.ui, QtWidgets.QLabel, "rotationIndicator"
        )
        self._reset_btn = get_cwidget(self.ui, QtWidgets.QPushButton, "resetBtn")

        # === Split Pages tab ===
        self._layout_mode_cb = get_cwidget(self.ui, QtWidgets.QComboBox, "layoutModeCB")
        self._single_page_uncut_btn = get_cwidget(
            self.ui, QtWidgets.QToolButton, "singlePageUncutBtn"
        )
        self._page_plus_offcut_btn = get_cwidget(
            self.ui, QtWidgets.QToolButton, "pagePlusOffcutBtn"
        )
        self._two_pages_btn = get_cwidget(self.ui, QtWidgets.QToolButton, "twoPagesBtn")

        # === Deskew tab ===
        self._deskew_auto_btn = get_cwidget(
            self.ui, QtWidgets.QToolButton, "deskewAutoBtn"
        )
        self._deskew_manual_btn = get_cwidget(
            self.ui, QtWidgets.QToolButton, "deskewManualBtn"
        )
        self._angle_spin_box = get_cwidget(
            self.ui, QtWidgets.QDoubleSpinBox, "angleSpinBox"
        )

        # === Select Content tab ===
        self._content_detect_auto_btn = get_cwidget(
            self.ui, QtWidgets.QToolButton, "contentDetectAutoBtn"
        )
        self._content_detect_disable_btn = get_cwidget(
            self.ui, QtWidgets.QToolButton, "contentDetectDisableBtn"
        )
        self._page_detect_auto_btn = get_cwidget(
            self.ui, QtWidgets.QToolButton, "pageDetectAutoBtn"
        )
        self._page_detect_manual_btn = get_cwidget(
            self.ui, QtWidgets.QToolButton, "pageDetectManualBtn"
        )
        self._page_detect_disable_btn = get_cwidget(
            self.ui, QtWidgets.QToolButton, "pageDetectDisableBtn"
        )
        self._fine_tune_btn = get_cwidget(self.ui, QtWidgets.QCheckBox, "fineTuneBtn")

        # === Margins tab ===
        self._top_margin_sb = get_cwidget(
            self.ui, QtWidgets.QDoubleSpinBox, "topMarginSpinBox"
        )
        self._bottom_margin_sb = get_cwidget(
            self.ui, QtWidgets.QDoubleSpinBox, "bottomMarginSpinBox"
        )
        self._left_margin_sb = get_cwidget(
            self.ui, QtWidgets.QDoubleSpinBox, "leftMarginSpinBox"
        )
        self._right_margin_sb = get_cwidget(
            self.ui, QtWidgets.QDoubleSpinBox, "rightMarginSpinBox"
        )
        self._top_bottom_link = get_cwidget(
            self.ui, QtWidgets.QToolButton, "topBottomLink"
        )
        self._left_right_link = get_cwidget(
            self.ui, QtWidgets.QToolButton, "leftRightLink"
        )
        self._auto_margins_cb = get_cwidget(self.ui, QtWidgets.QCheckBox, "autoMargins")

        # Alignment buttons
        self._align_top_left_btn = get_cwidget(
            self.ui, QtWidgets.QToolButton, "alignTopLeftBtn"
        )
        self._align_top_btn = get_cwidget(self.ui, QtWidgets.QToolButton, "alignTopBtn")
        self._align_top_right_btn = get_cwidget(
            self.ui, QtWidgets.QToolButton, "alignTopRightBtn"
        )
        self._align_left_btn = get_cwidget(
            self.ui, QtWidgets.QToolButton, "alignLeftBtn"
        )
        self._align_center_btn = get_cwidget(
            self.ui, QtWidgets.QToolButton, "alignCenterBtn"
        )
        self._align_right_btn = get_cwidget(
            self.ui, QtWidgets.QToolButton, "alignRightBtn"
        )
        self._align_bottom_left_btn = get_cwidget(
            self.ui, QtWidgets.QToolButton, "alignBottomLeftBtn"
        )
        self._align_bottom_btn = get_cwidget(
            self.ui, QtWidgets.QToolButton, "alignBottomBtn"
        )
        self._align_bottom_right_btn = get_cwidget(
            self.ui, QtWidgets.QToolButton, "alignBottomRightBtn"
        )
        self._align_with_others_cb = get_cwidget(
            self.ui, QtWidgets.QCheckBox, "alignWithOthersCB"
        )
        self._h_alignment_mode_cb = get_cwidget(
            self.ui, QtWidgets.QComboBox, "hAlignmentModeCB"
        )
        self._v_alignment_mode_cb = get_cwidget(
            self.ui, QtWidgets.QComboBox, "vAlignmentModeCB"
        )

        # === Output tab ===
        # Color mode
        self._color_mode_selector = get_cwidget(
            self.ui, QtWidgets.QComboBox, "colorModeSelector"
        )
        self._bw_foreground_rb = get_cwidget(
            self.ui, QtWidgets.QRadioButton, "bwForegroundRB"
        )
        self._color_foreground_rb = get_cwidget(
            self.ui, QtWidgets.QRadioButton, "colorForegroundRB"
        )

        # DPI
        self._dpi_selector = get_cwidget(self.ui, QtWidgets.QComboBox, "dpiSelector")

        # Binarization
        self._threshold_method_box = get_cwidget(
            self.ui, QtWidgets.QComboBox, "thresholdMethodBox"
        )
        self._threshold_slider = get_cwidget(
            self.ui, QtWidgets.QSlider, "thresholdSlider"
        )
        self._neutral_threshold_btn = get_cwidget(
            self.ui, QtWidgets.QPushButton, "neutralThresholdBtn"
        )
        self._sauvola_window_size = get_cwidget(
            self.ui, QtWidgets.QSpinBox, "sauvolaWindowSize"
        )
        self._sauvola_coef = get_cwidget(
            self.ui, QtWidgets.QDoubleSpinBox, "sauvolaCoef"
        )
        self._wolf_window_size = get_cwidget(
            self.ui, QtWidgets.QSpinBox, "wolfWindowSize"
        )
        self._wolf_coef = get_cwidget(self.ui, QtWidgets.QDoubleSpinBox, "wolfCoef")
        self._lower_bound = get_cwidget(self.ui, QtWidgets.QSpinBox, "lowerBound")
        self._upper_bound = get_cwidget(self.ui, QtWidgets.QSpinBox, "upperBound")

        # Picture shape
        self._picture_shape_selector = get_cwidget(
            self.ui, QtWidgets.QComboBox, "pictureShapeSelector"
        )
        self._picture_shape_sensitivity_sb = get_cwidget(
            self.ui, QtWidgets.QSpinBox, "pictureShapeSensitivitySB"
        )
        self._higher_search_sensitivity_cb = get_cwidget(
            self.ui, QtWidgets.QCheckBox, "higherSearchSensitivityCB"
        )

        # Despeckle
        self._despeckle_cb = get_cwidget(self.ui, QtWidgets.QCheckBox, "despeckleCB")
        self._despeckle_slider = get_cwidget(
            self.ui, QtWidgets.QSlider, "despeckleSlider"
        )

        # Smoothing
        self._morphological_smoothing_cb = get_cwidget(
            self.ui, QtWidgets.QCheckBox, "morphologicalSmoothingCB"
        )
        self._savitzky_golay_smoothing_cb = get_cwidget(
            self.ui, QtWidgets.QCheckBox, "savitzkyGolaySmoothingCB"
        )

        # Illumination
        self._equalize_illumination_cb = get_cwidget(
            self.ui, QtWidgets.QCheckBox, "equalizeIlluminationCB"
        )
        self._equalize_illumination_color_cb = get_cwidget(
            self.ui, QtWidgets.QCheckBox, "equalizeIlluminationColorCB"
        )

        # Filling
        self._fill_margins_cb = get_cwidget(
            self.ui, QtWidgets.QCheckBox, "fillMarginsCB"
        )
        self._fill_offcut_cb = get_cwidget(self.ui, QtWidgets.QCheckBox, "fillOffcutCB")
        self._filling_color_box = get_cwidget(
            self.ui, QtWidgets.QComboBox, "fillingColorBox"
        )

        # Splitting
        self._splitting_cb = get_cwidget(self.ui, QtWidgets.QCheckBox, "splittingCB")
        self._original_background_cb = get_cwidget(
            self.ui, QtWidgets.QCheckBox, "originalBackgroundCB"
        )
        self._color_segmentation_cb = get_cwidget(
            self.ui, QtWidgets.QCheckBox, "colorSegmentationCB"
        )
        self._reduce_noise_sb = get_cwidget(
            self.ui, QtWidgets.QSpinBox, "reduceNoiseSB"
        )
        self._red_adjustment_sb = get_cwidget(
            self.ui, QtWidgets.QSpinBox, "redAdjustmentSB"
        )
        self._green_adjustment_sb = get_cwidget(
            self.ui, QtWidgets.QSpinBox, "greenAdjustmentSB"
        )
        self._blue_adjustment_sb = get_cwidget(
            self.ui, QtWidgets.QSpinBox, "blueAdjustmentSB"
        )

        # Posterize
        self._posterize_cb = get_cwidget(self.ui, QtWidgets.QCheckBox, "posterizeCB")
        self._posterize_level_sb = get_cwidget(
            self.ui, QtWidgets.QSpinBox, "posterizeLevelSB"
        )
        self._posterize_normalization_cb = get_cwidget(
            self.ui, QtWidgets.QCheckBox, "posterizeNormalizationCB"
        )
        self._posterize_force_bw_cb = get_cwidget(
            self.ui, QtWidgets.QCheckBox, "posterizeForceBwCB"
        )

        # Dewarping
        self._dewarping_mode_cb = get_cwidget(
            self.ui, QtWidgets.QComboBox, "dewarpingModeCB"
        )
        self._dewarping_post_deskew_cb = get_cwidget(
            self.ui, QtWidgets.QCheckBox, "dewarpingPostDeskewCB"
        )
        self._depth_perception_slider = get_cwidget(
            self.ui, QtWidgets.QSlider, "depthPerceptionSlider"
        )

        # Output dimensions
        self._width_spin_box = get_cwidget(
            self.ui, QtWidgets.QDoubleSpinBox, "widthSpinBox"
        )
        self._height_spin_box = get_cwidget(
            self.ui, QtWidgets.QDoubleSpinBox, "heightSpinBox"
        )

        # Button box
        self._button_box = get_cwidget(self.ui, QtWidgets.QDialogButtonBox, "buttonBox")

        # Set default units
        self._units_label.setText("mm")

        # Populate combo boxes
        self._populate_combo_boxes()

    def _populate_combo_boxes(self) -> None:
        """Populate combo box options."""
        # Profile combo - start with default
        self._profile_cb.clear()
        self._profile_cb.addItem("Default")

        # Layout mode for page split
        self._layout_mode_cb.clear()
        self._layout_mode_cb.addItems(
            [
                "Auto",
                "Manual",
            ]
        )

        # Alignment mode options
        alignment_modes = ["Auto", "Manual", "Original"]
        self._h_alignment_mode_cb.clear()
        self._h_alignment_mode_cb.addItems(alignment_modes)
        self._v_alignment_mode_cb.clear()
        self._v_alignment_mode_cb.addItems(alignment_modes)

        # Color mode
        self._color_mode_selector.clear()
        self._color_mode_selector.addItems(
            [
                "Black and White",
                "Color / Grayscale",
                "Mixed",
            ]
        )

        # DPI options
        self._dpi_selector.clear()
        self._dpi_selector.addItems(
            [
                "300",
                "400",
                "600",
                "1200",
            ]
        )

        # Binarization method
        self._threshold_method_box.clear()
        self._threshold_method_box.addItems(
            [
                "Otsu",
                "Sauvola",
                "Wolf",
            ]
        )

        # Picture shape
        self._picture_shape_selector.clear()
        self._picture_shape_selector.addItems(
            [
                "Off",
                "Free",
                "Rectangular",
            ]
        )

        # Filling color
        self._filling_color_box.clear()
        self._filling_color_box.addItems(
            [
                "White",
                "Black",
            ]
        )

        # Dewarping mode
        self._dewarping_mode_cb.clear()
        self._dewarping_mode_cb.addItems(
            [
                "Off",
                "Auto",
                "Manual",
                "Marginal",
            ]
        )

    def _load_params(self) -> None:
        """Load current parameters into the UI."""
        if not self._params:
            return

        # TODO: Load values from params object based on actual DefaultParams structure

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._button_box.accepted.connect(self._save_and_accept)
        self._button_box.rejected.connect(self.reject)

        # Profile management
        self._profile_save_btn.clicked.connect(self._save_profile)
        self._profile_delete_btn.clicked.connect(self._delete_profile)
        self._profile_cb.currentIndexChanged.connect(self._on_profile_changed)

        # Fix Orientation
        self._rotate_left_btn.clicked.connect(self._rotate_left)
        self._rotate_right_btn.clicked.connect(self._rotate_right)
        self._reset_btn.clicked.connect(self._reset_rotation)

        # Deskew mode
        self._deskew_auto_btn.clicked.connect(
            lambda: self._angle_spin_box.setEnabled(False)
        )
        self._deskew_manual_btn.clicked.connect(
            lambda: self._angle_spin_box.setEnabled(True)
        )

        # Auto margins toggle
        self._auto_margins_cb.toggled.connect(self._on_auto_margins_toggled)

        # Margin linking
        self._top_bottom_link.toggled.connect(self._on_top_bottom_link_toggled)
        self._left_right_link.toggled.connect(self._on_left_right_link_toggled)
        self._top_margin_sb.valueChanged.connect(self._on_top_margin_changed)
        self._left_margin_sb.valueChanged.connect(self._on_left_margin_changed)

        # Threshold method
        self._threshold_method_box.currentIndexChanged.connect(
            self._on_threshold_method_changed
        )
        self._neutral_threshold_btn.clicked.connect(self._reset_threshold)

        # Despeckle
        self._despeckle_cb.toggled.connect(
            lambda checked: self._despeckle_slider.setEnabled(checked)
        )

        # Posterize
        self._posterize_cb.toggled.connect(self._on_posterize_toggled)

        # Splitting
        self._splitting_cb.toggled.connect(self._on_splitting_toggled)

        # Color segmentation
        self._color_segmentation_cb.toggled.connect(self._on_color_segmentation_toggled)

    def _save_and_accept(self) -> None:
        """Save parameters and close dialog."""
        self._save_params()
        self.params_changed.emit()
        self.accept()

    def _save_params(self) -> None:
        """Save UI values to parameters."""
        if not self._params:
            return

        # TODO: Save values to params object based on actual DefaultParams structure

    def _save_profile(self) -> None:
        """Save current settings as a profile."""
        name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Save Profile",
            "Profile name:",
        )
        if ok and name:
            # Add to combo box if new
            if self._profile_cb.findText(name) == -1:
                self._profile_cb.addItem(name)
            self._profile_cb.setCurrentText(name)
            # TODO: Actually save the profile to disk

    def _delete_profile(self) -> None:
        """Delete the current profile."""
        current = self._profile_cb.currentText()
        if current == "Default":
            QtWidgets.QMessageBox.warning(
                self,
                "Cannot Delete",
                "Cannot delete the default profile.",
            )
            return

        reply = QtWidgets.QMessageBox.question(
            self,
            "Delete Profile",
            f"Delete profile '{current}'?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            index = self._profile_cb.currentIndex()
            self._profile_cb.removeItem(index)
            # TODO: Actually delete the profile from disk

    def _on_profile_changed(self, index: int) -> None:
        """Handle profile selection change."""
        # TODO: Load the selected profile
        pass

    def _rotate_left(self) -> None:
        """Rotate default orientation left by 90 degrees."""
        self._current_rotation = (self._current_rotation - 90) % 360
        self._update_rotation_indicator()

    def _rotate_right(self) -> None:
        """Rotate default orientation right by 90 degrees."""
        self._current_rotation = (self._current_rotation + 90) % 360
        self._update_rotation_indicator()

    def _reset_rotation(self) -> None:
        """Reset rotation to 0 degrees."""
        self._current_rotation = 0
        self._update_rotation_indicator()

    def _update_rotation_indicator(self) -> None:
        """Update the rotation indicator image."""
        # The indicator shows an arrow that rotates with the current rotation
        pixmap = QPixmap(":/icons/big-up-arrow.svg")
        if not pixmap.isNull():
            from PySide6.QtGui import QTransform

            transform = QTransform()
            transform.rotate(self._current_rotation)
            rotated = pixmap.transformed(transform)
            self._rotation_indicator.setPixmap(rotated)

    def _on_auto_margins_toggled(self, checked: bool) -> None:
        """Handle auto margins checkbox toggle."""
        # Enable/disable margin spin boxes
        enabled = not checked
        self._top_margin_sb.setEnabled(enabled)
        self._bottom_margin_sb.setEnabled(enabled)
        self._left_margin_sb.setEnabled(enabled)
        self._right_margin_sb.setEnabled(enabled)
        self._top_bottom_link.setEnabled(enabled)
        self._left_right_link.setEnabled(enabled)

    def _on_top_bottom_link_toggled(self, checked: bool) -> None:
        """Handle top/bottom margin link toggle."""
        if checked:
            self._bottom_margin_sb.setValue(self._top_margin_sb.value())

    def _on_left_right_link_toggled(self, checked: bool) -> None:
        """Handle left/right margin link toggle."""
        if checked:
            self._right_margin_sb.setValue(self._left_margin_sb.value())

    def _on_top_margin_changed(self, value: float) -> None:
        """Sync bottom margin if linked."""
        if self._top_bottom_link.isChecked():
            self._bottom_margin_sb.setValue(value)

    def _on_left_margin_changed(self, value: float) -> None:
        """Sync right margin if linked."""
        if self._left_right_link.isChecked():
            self._right_margin_sb.setValue(value)

    def _on_threshold_method_changed(self, index: int) -> None:
        """Handle binarization method change."""
        method = self._threshold_method_box.currentText()

        # Show/hide method-specific options
        is_sauvola = method == "Sauvola"
        is_wolf = method == "Wolf"

        self._sauvola_window_size.setVisible(is_sauvola)
        self._sauvola_coef.setVisible(is_sauvola)
        self._wolf_window_size.setVisible(is_wolf)
        self._wolf_coef.setVisible(is_wolf)

    def _reset_threshold(self) -> None:
        """Reset threshold slider to neutral position."""
        self._threshold_slider.setValue(0)

    def _on_posterize_toggled(self, checked: bool) -> None:
        """Handle posterize checkbox toggle."""
        self._posterize_level_sb.setEnabled(checked)
        self._posterize_normalization_cb.setEnabled(checked)
        self._posterize_force_bw_cb.setEnabled(checked)

    def _on_splitting_toggled(self, checked: bool) -> None:
        """Handle splitting checkbox toggle."""
        self._original_background_cb.setEnabled(checked)

    def _on_color_segmentation_toggled(self, checked: bool) -> None:
        """Handle color segmentation checkbox toggle."""
        self._reduce_noise_sb.setEnabled(checked)
        self._red_adjustment_sb.setEnabled(checked)
        self._green_adjustment_sb.setEnabled(checked)
        self._blue_adjustment_sb.setEnabled(checked)

    def get_params(self) -> DefaultParams | None:
        """Get the (possibly modified) default parameters.

        Returns:
            The default parameters, or None if not set.
        """
        return self._params

    def get_current_rotation(self) -> int:
        """Get the current rotation value.

        Returns:
            Rotation in degrees (0, 90, 180, or 270).
        """
        return self._current_rotation

    def show(self) -> None:
        """Show the dialog."""
        self.ui.show()
