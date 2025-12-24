"""Options widget for Output filter.

Provides UI controls for configuring output settings including color mode,
binarization, despeckling, and dewarping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpacerItem,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from scantailor.app.ui.widgets import CollapsibleGroupBox
from scantailor.core import PageId
from scantailor.filters.output.binarization import BinarizationMethod
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.despeckle import DespeckleLevel
from scantailor.filters.output.dewarping_options import DewarpingMode

if TYPE_CHECKING:
    from scantailor.filters.output.filter import Filter


class OptionsWidget(QWidget):
    """Options widget for the Output filter.

    Provides controls for:
    - Output DPI
    - Color mode (B&W, Grayscale/Color, Mixed)
    - Binarization method and parameters
    - Despeckle level
    - Dewarping options

    Signals:
        params_changed: Emitted when any parameter changes.
            Args: page_id (PageId)
        apply_to_requested: Emitted when user clicks "Apply to..."
    """

    params_changed = Signal(PageId)
    apply_to_requested = Signal()

    def __init__(
        self,
        filter_: Filter,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the options widget.

        Args:
            filter_: The Output filter instance.
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

        # Output DPI group
        self._dpi_group = CollapsibleGroupBox("Output Resolution", self)
        self._dpi_group.setObjectName("dpiGroup")
        dpi_layout = QHBoxLayout(self._dpi_group)
        dpi_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        dpi_layout.addWidget(QLabel("DPI:"))
        self._dpi_spinbox = QSpinBox()
        self._dpi_spinbox.setObjectName("dpiSpinBox")
        self._dpi_spinbox.setRange(50, 1200)
        self._dpi_spinbox.setValue(600)
        self._dpi_spinbox.setSingleStep(50)
        dpi_layout.addWidget(self._dpi_spinbox)
        dpi_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        layout.addWidget(self._dpi_group)

        # Color Mode group
        self._mode_group = CollapsibleGroupBox("Mode", self)
        self._mode_group.setObjectName("modeGroup")
        mode_layout = QVBoxLayout(self._mode_group)

        # Color mode buttons
        mode_btn_layout = QHBoxLayout()
        self._mode_button_group = QButtonGroup(self)

        self._bw_btn = QPushButton("Black and White")
        self._bw_btn.setObjectName("bwBtn")
        self._bw_btn.setCheckable(True)
        self._bw_btn.setChecked(True)
        self._bw_btn.setAutoExclusive(True)
        self._mode_button_group.addButton(self._bw_btn, 0)
        mode_btn_layout.addWidget(self._bw_btn)

        self._gray_btn = QPushButton("Color / Grayscale")
        self._gray_btn.setObjectName("grayBtn")
        self._gray_btn.setCheckable(True)
        self._gray_btn.setAutoExclusive(True)
        self._mode_button_group.addButton(self._gray_btn, 1)
        mode_btn_layout.addWidget(self._gray_btn)

        self._mixed_btn = QPushButton("Mixed")
        self._mixed_btn.setObjectName("mixedBtn")
        self._mixed_btn.setCheckable(True)
        self._mixed_btn.setAutoExclusive(True)
        self._mixed_btn.setToolTip("Black and white with colorful pictures")
        self._mode_button_group.addButton(self._mixed_btn, 2)
        mode_btn_layout.addWidget(self._mixed_btn)

        mode_layout.addLayout(mode_btn_layout)

        # Black on white checkbox
        bow_layout = QHBoxLayout()
        bow_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        self._black_on_white_check = QCheckBox("White margins")
        self._black_on_white_check.setObjectName("blackOnWhiteCB")
        self._black_on_white_check.setChecked(True)
        self._black_on_white_check.setToolTip(
            "Checked: black text on white background\n"
            "Unchecked: white text on black background"
        )
        bow_layout.addWidget(self._black_on_white_check)
        bow_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        mode_layout.addLayout(bow_layout)

        layout.addWidget(self._mode_group)

        # Binarization group (only for B&W and Mixed modes)
        self._binarization_group = CollapsibleGroupBox("Binarization", self)
        self._binarization_group.setObjectName("binarizationGroup")
        bin_layout = QVBoxLayout(self._binarization_group)

        # Method dropdown
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self._method_combo = QComboBox()
        self._method_combo.setObjectName("methodCombo")
        self._method_combo.addItems(["Otsu", "Sauvola", "Wolf", "Bradley", "EdgeDiv"])
        method_layout.addWidget(self._method_combo)
        method_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        bin_layout.addLayout(method_layout)

        # Threshold adjustment slider
        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("Threshold:"))
        self._threshold_slider = QSlider()
        self._threshold_slider.setObjectName("thresholdSlider")
        self._threshold_slider.setOrientation(1)  # Horizontal
        self._threshold_slider.setRange(-100, 100)
        self._threshold_slider.setValue(0)
        thresh_layout.addWidget(self._threshold_slider)
        self._threshold_label = QLabel("0")
        self._threshold_label.setMinimumWidth(30)
        thresh_layout.addWidget(self._threshold_label)
        bin_layout.addLayout(thresh_layout)

        # Advanced options group
        self._bin_advanced_group = QGroupBox("Advanced")
        self._bin_advanced_group.setObjectName("binAdvancedGroup")
        adv_layout = QGridLayout(self._bin_advanced_group)

        adv_layout.addWidget(QLabel("Window size:"), 0, 0)
        self._window_size_spinbox = QSpinBox()
        self._window_size_spinbox.setObjectName("windowSizeSpinBox")
        self._window_size_spinbox.setRange(10, 1000)
        self._window_size_spinbox.setValue(200)
        adv_layout.addWidget(self._window_size_spinbox, 0, 1)

        adv_layout.addWidget(QLabel("Coefficient:"), 1, 0)
        self._coef_spinbox = QDoubleSpinBox()
        self._coef_spinbox.setObjectName("coefSpinBox")
        self._coef_spinbox.setRange(0.0, 1.0)
        self._coef_spinbox.setSingleStep(0.01)
        self._coef_spinbox.setValue(0.34)
        adv_layout.addWidget(self._coef_spinbox, 1, 1)

        # Preprocessing options
        self._normalize_check = QCheckBox("Normalize illumination")
        self._normalize_check.setObjectName("normalizeCB")
        self._normalize_check.setChecked(True)
        adv_layout.addWidget(self._normalize_check, 2, 0, 1, 2)

        self._morphological_check = QCheckBox("Morphological smoothing")
        self._morphological_check.setObjectName("morphologicalCB")
        self._morphological_check.setChecked(True)
        adv_layout.addWidget(self._morphological_check, 3, 0, 1, 2)

        bin_layout.addWidget(self._bin_advanced_group)
        layout.addWidget(self._binarization_group)

        # Despeckle group
        self._despeckle_group = CollapsibleGroupBox("Despeckle", self)
        self._despeckle_group.setObjectName("despeckleGroup")
        despeckle_layout = QVBoxLayout(self._despeckle_group)

        despeckle_btn_layout = QHBoxLayout()
        self._despeckle_button_group = QButtonGroup(self)

        self._despeckle_off_btn = QPushButton("Off")
        self._despeckle_off_btn.setObjectName("despeckleOffBtn")
        self._despeckle_off_btn.setCheckable(True)
        self._despeckle_off_btn.setAutoExclusive(True)
        self._despeckle_button_group.addButton(self._despeckle_off_btn, 0)
        despeckle_btn_layout.addWidget(self._despeckle_off_btn)

        self._despeckle_cautious_btn = QPushButton("Cautious")
        self._despeckle_cautious_btn.setObjectName("despeckleCautiousBtn")
        self._despeckle_cautious_btn.setCheckable(True)
        self._despeckle_cautious_btn.setAutoExclusive(True)
        self._despeckle_button_group.addButton(self._despeckle_cautious_btn, 1)
        despeckle_btn_layout.addWidget(self._despeckle_cautious_btn)

        self._despeckle_normal_btn = QPushButton("Normal")
        self._despeckle_normal_btn.setObjectName("despeckleNormalBtn")
        self._despeckle_normal_btn.setCheckable(True)
        self._despeckle_normal_btn.setChecked(True)
        self._despeckle_normal_btn.setAutoExclusive(True)
        self._despeckle_button_group.addButton(self._despeckle_normal_btn, 2)
        despeckle_btn_layout.addWidget(self._despeckle_normal_btn)

        self._despeckle_aggressive_btn = QPushButton("Aggressive")
        self._despeckle_aggressive_btn.setObjectName("despeckleAggressiveBtn")
        self._despeckle_aggressive_btn.setCheckable(True)
        self._despeckle_aggressive_btn.setAutoExclusive(True)
        self._despeckle_button_group.addButton(self._despeckle_aggressive_btn, 3)
        despeckle_btn_layout.addWidget(self._despeckle_aggressive_btn)

        despeckle_layout.addLayout(despeckle_btn_layout)
        layout.addWidget(self._despeckle_group)

        # Dewarping group
        self._dewarping_group = CollapsibleGroupBox("Dewarping", self)
        self._dewarping_group.setObjectName("dewarpingGroup")
        dewarp_layout = QVBoxLayout(self._dewarping_group)

        dewarp_btn_layout = QHBoxLayout()
        self._dewarp_button_group = QButtonGroup(self)

        self._dewarp_off_btn = QPushButton("Off")
        self._dewarp_off_btn.setObjectName("dewarpOffBtn")
        self._dewarp_off_btn.setCheckable(True)
        self._dewarp_off_btn.setChecked(True)
        self._dewarp_off_btn.setAutoExclusive(True)
        self._dewarp_button_group.addButton(self._dewarp_off_btn, 0)
        dewarp_btn_layout.addWidget(self._dewarp_off_btn)

        self._dewarp_auto_btn = QPushButton("Auto")
        self._dewarp_auto_btn.setObjectName("dewarpAutoBtn")
        self._dewarp_auto_btn.setCheckable(True)
        self._dewarp_auto_btn.setAutoExclusive(True)
        self._dewarp_button_group.addButton(self._dewarp_auto_btn, 1)
        dewarp_btn_layout.addWidget(self._dewarp_auto_btn)

        self._dewarp_manual_btn = QPushButton("Manual")
        self._dewarp_manual_btn.setObjectName("dewarpManualBtn")
        self._dewarp_manual_btn.setCheckable(True)
        self._dewarp_manual_btn.setAutoExclusive(True)
        self._dewarp_button_group.addButton(self._dewarp_manual_btn, 2)
        dewarp_btn_layout.addWidget(self._dewarp_manual_btn)

        self._dewarp_marginal_btn = QPushButton("Marginal")
        self._dewarp_marginal_btn.setObjectName("dewarpMarginalBtn")
        self._dewarp_marginal_btn.setCheckable(True)
        self._dewarp_marginal_btn.setAutoExclusive(True)
        self._dewarp_marginal_btn.setToolTip("Apply dewarping only to page margins")
        self._dewarp_button_group.addButton(self._dewarp_marginal_btn, 3)
        dewarp_btn_layout.addWidget(self._dewarp_marginal_btn)

        dewarp_layout.addLayout(dewarp_btn_layout)

        # Post-deskew options
        self._post_deskew_check = QCheckBox("Post-deskew")
        self._post_deskew_check.setObjectName("postDeskewCB")
        self._post_deskew_check.setChecked(True)
        self._post_deskew_check.setToolTip("Apply additional deskewing after dewarping")
        dewarp_layout.addWidget(self._post_deskew_check)

        layout.addWidget(self._dewarping_group)

        # Apply to button
        apply_layout = QHBoxLayout()
        apply_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        self._apply_btn = QPushButton("Apply to ...")
        self._apply_btn.setObjectName("applyToBtn")
        apply_layout.addWidget(self._apply_btn)
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
        # DPI
        self._dpi_spinbox.valueChanged.connect(self._on_dpi_changed)

        # Color mode
        self._mode_button_group.buttonClicked.connect(self._on_color_mode_changed)
        self._black_on_white_check.toggled.connect(self._on_black_on_white_changed)

        # Binarization
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        self._threshold_slider.valueChanged.connect(self._on_threshold_changed)
        self._window_size_spinbox.valueChanged.connect(self._on_window_size_changed)
        self._coef_spinbox.valueChanged.connect(self._on_coef_changed)
        self._normalize_check.toggled.connect(self._on_normalize_changed)
        self._morphological_check.toggled.connect(self._on_morphological_changed)

        # Despeckle
        self._despeckle_button_group.buttonClicked.connect(self._on_despeckle_changed)

        # Dewarping
        self._dewarp_button_group.buttonClicked.connect(self._on_dewarping_changed)
        self._post_deskew_check.toggled.connect(self._on_post_deskew_changed)

        # Apply
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

            # Update DPI
            self._dpi_spinbox.setValue(params.output_dpi.horizontal)

            # Update color mode
            if params.color_mode == ColorMode.BLACK_AND_WHITE:
                self._bw_btn.setChecked(True)
            elif params.color_mode == ColorMode.COLOR_GRAYSCALE:
                self._gray_btn.setChecked(True)
            else:
                self._mixed_btn.setChecked(True)

            self._black_on_white_check.setChecked(params.black_on_white)

            # Update binarization visibility
            needs_bin = params.needs_binarization()
            self._binarization_group.setVisible(needs_bin)
            self._despeckle_group.setVisible(needs_bin)

            if needs_bin:
                # Update binarization options
                bin_opts = params.binarization
                method_index = list(BinarizationMethod).index(bin_opts.method)
                self._method_combo.setCurrentIndex(method_index)
                self._threshold_slider.setValue(bin_opts.threshold_adjustment)
                self._threshold_label.setText(str(bin_opts.threshold_adjustment))
                self._window_size_spinbox.setValue(bin_opts.window_size)
                self._coef_spinbox.setValue(bin_opts.sauvola_coef)
                self._normalize_check.setChecked(bin_opts.normalize_illumination)
                self._morphological_check.setChecked(bin_opts.morphological_smoothing)

                # Update despeckle
                despeckle_buttons = [
                    self._despeckle_off_btn,
                    self._despeckle_cautious_btn,
                    self._despeckle_normal_btn,
                    self._despeckle_aggressive_btn,
                ]
                despeckle_index = list(DespeckleLevel).index(params.despeckle_level)
                despeckle_buttons[despeckle_index].setChecked(True)

            # Update dewarping
            dewarp_opts = params.dewarping
            dewarp_buttons = [
                self._dewarp_off_btn,
                self._dewarp_auto_btn,
                self._dewarp_manual_btn,
                self._dewarp_marginal_btn,
            ]
            dewarp_index = list(DewarpingMode).index(dewarp_opts.mode)
            dewarp_buttons[dewarp_index].setChecked(True)
            self._post_deskew_check.setChecked(dewarp_opts.post_deskew)
            self._post_deskew_check.setEnabled(dewarp_opts.is_enabled())
        finally:
            self._updating = False

    def _on_dpi_changed(self, value: int) -> None:
        """Handle DPI spinbox change."""
        if self._current_page_id is None or self._updating:
            return

        from scantailor.core import Dpi
        params = self._filter.get_params(self._current_page_id)
        new_params = params.with_output_dpi(Dpi.uniform(value))
        self._filter.set_params(self._current_page_id, new_params)
        self.params_changed.emit(self._current_page_id)

    def _on_color_mode_changed(self, button: QPushButton) -> None:
        """Handle color mode button change."""
        if self._current_page_id is None or self._updating:
            return

        mode_map = {
            self._bw_btn: ColorMode.BLACK_AND_WHITE,
            self._gray_btn: ColorMode.COLOR_GRAYSCALE,
            self._mixed_btn: ColorMode.MIXED,
        }
        mode = mode_map.get(button, ColorMode.BLACK_AND_WHITE)

        params = self._filter.get_params(self._current_page_id)
        new_params = params.with_color_mode(mode)
        self._filter.set_params(self._current_page_id, new_params)
        self._update_ui()
        self.params_changed.emit(self._current_page_id)

    def _on_black_on_white_changed(self, checked: bool) -> None:
        """Handle black on white checkbox change."""
        if self._current_page_id is None or self._updating:
            return

        params = self._filter.get_params(self._current_page_id)
        new_params = params.with_black_on_white(checked)
        self._filter.set_params(self._current_page_id, new_params)
        self.params_changed.emit(self._current_page_id)

    def _on_method_changed(self, index: int) -> None:
        """Handle binarization method change."""
        if self._current_page_id is None or self._updating:
            return

        method = list(BinarizationMethod)[index]
        params = self._filter.get_params(self._current_page_id)
        new_params = params.with_binarization_method(method)
        self._filter.set_params(self._current_page_id, new_params)
        self.params_changed.emit(self._current_page_id)

    def _on_threshold_changed(self, value: int) -> None:
        """Handle threshold slider change."""
        self._threshold_label.setText(str(value))

        if self._current_page_id is None or self._updating:
            return

        params = self._filter.get_params(self._current_page_id)
        new_binarization = params.binarization.with_threshold_adjustment(value)
        new_params = params.with_binarization(new_binarization)
        self._filter.set_params(self._current_page_id, new_params)
        self.params_changed.emit(self._current_page_id)

    def _on_window_size_changed(self, value: int) -> None:
        """Handle window size change."""
        if self._current_page_id is None or self._updating:
            return

        params = self._filter.get_params(self._current_page_id)
        new_binarization = params.binarization.with_window_size(value)
        new_params = params.with_binarization(new_binarization)
        self._filter.set_params(self._current_page_id, new_params)
        self.params_changed.emit(self._current_page_id)

    def _on_coef_changed(self, value: float) -> None:
        """Handle coefficient change."""
        if self._current_page_id is None or self._updating:
            return

        params = self._filter.get_params(self._current_page_id)
        # Update the appropriate coefficient based on method
        method = params.binarization.method
        update = {}
        if method == BinarizationMethod.SAUVOLA:
            update["sauvola_coef"] = value
        elif method == BinarizationMethod.WOLF:
            update["wolf_coef"] = value
        elif method == BinarizationMethod.BRADLEY:
            update["bradley_coef"] = value

        if update:
            new_binarization = params.binarization.model_copy(update=update)
            new_params = params.with_binarization(new_binarization)
            self._filter.set_params(self._current_page_id, new_params)
            self.params_changed.emit(self._current_page_id)

    def _on_normalize_changed(self, checked: bool) -> None:
        """Handle normalize illumination change."""
        if self._current_page_id is None or self._updating:
            return

        params = self._filter.get_params(self._current_page_id)
        new_binarization = params.binarization.model_copy(
            update={"normalize_illumination": checked}
        )
        new_params = params.with_binarization(new_binarization)
        self._filter.set_params(self._current_page_id, new_params)
        self.params_changed.emit(self._current_page_id)

    def _on_morphological_changed(self, checked: bool) -> None:
        """Handle morphological smoothing change."""
        if self._current_page_id is None or self._updating:
            return

        params = self._filter.get_params(self._current_page_id)
        new_binarization = params.binarization.model_copy(
            update={"morphological_smoothing": checked}
        )
        new_params = params.with_binarization(new_binarization)
        self._filter.set_params(self._current_page_id, new_params)
        self.params_changed.emit(self._current_page_id)

    def _on_despeckle_changed(self, button: QPushButton) -> None:
        """Handle despeckle button change."""
        if self._current_page_id is None or self._updating:
            return

        level_map = {
            self._despeckle_off_btn: DespeckleLevel.OFF,
            self._despeckle_cautious_btn: DespeckleLevel.CAUTIOUS,
            self._despeckle_normal_btn: DespeckleLevel.NORMAL,
            self._despeckle_aggressive_btn: DespeckleLevel.AGGRESSIVE,
        }
        level = level_map.get(button, DespeckleLevel.OFF)

        params = self._filter.get_params(self._current_page_id)
        new_params = params.with_despeckle_level(level)
        self._filter.set_params(self._current_page_id, new_params)
        self.params_changed.emit(self._current_page_id)

    def _on_dewarping_changed(self, button: QPushButton) -> None:
        """Handle dewarping button change."""
        if self._current_page_id is None or self._updating:
            return

        mode_map = {
            self._dewarp_off_btn: DewarpingMode.OFF,
            self._dewarp_auto_btn: DewarpingMode.AUTO,
            self._dewarp_manual_btn: DewarpingMode.MANUAL,
            self._dewarp_marginal_btn: DewarpingMode.MARGINAL,
        }
        mode = mode_map.get(button, DewarpingMode.OFF)

        params = self._filter.get_params(self._current_page_id)
        new_dewarping = params.dewarping.with_mode(mode)
        new_params = params.with_dewarping(new_dewarping)
        self._filter.set_params(self._current_page_id, new_params)

        # Update post-deskew checkbox enabled state
        self._post_deskew_check.setEnabled(new_dewarping.is_enabled())

        self.params_changed.emit(self._current_page_id)

    def _on_post_deskew_changed(self, checked: bool) -> None:
        """Handle post-deskew checkbox change."""
        if self._current_page_id is None or self._updating:
            return

        params = self._filter.get_params(self._current_page_id)
        new_dewarping = params.dewarping.with_post_deskew(checked)
        new_params = params.with_dewarping(new_dewarping)
        self._filter.set_params(self._current_page_id, new_params)
        self.params_changed.emit(self._current_page_id)
