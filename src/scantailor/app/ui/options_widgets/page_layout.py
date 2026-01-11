"""Options widget for Page Layout filter.

Provides UI controls for configuring margins and alignment.
"""

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
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
from scantailor.core import PageId
from scantailor.core.models import Margins
from scantailor.filters.page_layout.alignment import (
    Alignment,
    HorizontalAlignment,
    VerticalAlignment,
)

if TYPE_CHECKING:
    from scantailor.filters.page_layout.filter import Filter


class OptionsWidget(QWidget):
    """Options widget for the Page Layout filter.

    Provides controls for:
    - Margins (top, bottom, left, right) with optional auto-margins
    - Alignment (9-point grid for manual, dropdown for mode selection)
    - Match size with other pages option

    Signals:
        margins_changed: Emitted when margin values change.
            Args: page_id (PageId), margins (Margins)
        alignment_changed: Emitted when alignment changes.
            Args: page_id (PageId), alignment (Alignment)
        apply_margins_requested: Emitted when user clicks "Apply To..." for margins
        apply_alignment_requested: Emitted when user clicks "Apply To..." for alignment
    """

    margins_changed = Signal(PageId, Margins)
    alignment_changed = Signal(PageId, Alignment)
    apply_margins_requested = Signal()
    apply_alignment_requested = Signal()

    def __init__(
        self,
        filter_: Filter,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the options widget.

        Args:
            filter_: The Page Layout filter instance.
            parent: The parent widget.
        """
        super().__init__(parent)

        self._filter = filter_
        self._current_page_id: PageId | None = None
        self._updating = False  # Prevent signal loops
        self._link_top_bottom = False
        self._link_left_right = False

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Margins group
        self._margins_group = CollapsibleGroupBox("Margins", self)
        self._margins_group.setObjectName("marginsGroup")
        margins_layout = QVBoxLayout(self._margins_group)

        # Auto margins checkbox row
        auto_layout = QHBoxLayout()
        auto_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        self._auto_margins_check = QCheckBox("Auto Margins")
        self._auto_margins_check.setObjectName("autoMargins")
        auto_layout.addWidget(self._auto_margins_check)
        auto_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        margins_layout.addLayout(auto_layout)

        # Margins spinboxes grid
        spinbox_layout = QHBoxLayout()
        spinbox_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )

        margins_grid = QGridLayout()

        # Top/Bottom spinboxes and link
        margins_grid.addWidget(QLabel("Top"), 0, 0)
        self._top_spinbox = QDoubleSpinBox()
        self._top_spinbox.setObjectName("topMarginSpinBox")
        self._top_spinbox.setDecimals(1)
        self._top_spinbox.setRange(-9999.0, 9999.0)
        margins_grid.addWidget(self._top_spinbox, 0, 1)

        self._top_bottom_link = QToolButton()
        self._top_bottom_link.setObjectName("topBottomLink")
        self._top_bottom_link.setText("⇅")
        self._top_bottom_link.setCheckable(True)
        self._top_bottom_link.setToolTip("Link top and bottom margins")
        self._top_bottom_link.setMinimumSize(20, 36)
        self._top_bottom_link.setMaximumSize(20, 36)
        margins_grid.addWidget(self._top_bottom_link, 0, 2, 2, 1)

        margins_grid.addWidget(QLabel("Bottom"), 1, 0)
        self._bottom_spinbox = QDoubleSpinBox()
        self._bottom_spinbox.setObjectName("bottomMarginSpinBox")
        self._bottom_spinbox.setDecimals(1)
        self._bottom_spinbox.setRange(-9999.0, 9999.0)
        margins_grid.addWidget(self._bottom_spinbox, 1, 1)

        # Left/Right spinboxes and link
        margins_grid.addWidget(QLabel("Left"), 2, 0)
        self._left_spinbox = QDoubleSpinBox()
        self._left_spinbox.setObjectName("leftMarginSpinBox")
        self._left_spinbox.setDecimals(1)
        self._left_spinbox.setRange(-9999.0, 9999.0)
        margins_grid.addWidget(self._left_spinbox, 2, 1)

        self._left_right_link = QToolButton()
        self._left_right_link.setObjectName("leftRightLink")
        self._left_right_link.setText("⇄")
        self._left_right_link.setCheckable(True)
        self._left_right_link.setToolTip("Link left and right margins")
        self._left_right_link.setMinimumSize(20, 36)
        self._left_right_link.setMaximumSize(20, 36)
        margins_grid.addWidget(self._left_right_link, 2, 2, 2, 1)

        margins_grid.addWidget(QLabel("Right"), 3, 0)
        self._right_spinbox = QDoubleSpinBox()
        self._right_spinbox.setObjectName("rightMarginSpinBox")
        self._right_spinbox.setDecimals(1)
        self._right_spinbox.setRange(-9999.0, 9999.0)
        margins_grid.addWidget(self._right_spinbox, 3, 1)

        spinbox_layout.addLayout(margins_grid)
        spinbox_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        margins_layout.addLayout(spinbox_layout)

        # Apply margins button
        apply_margins_layout = QHBoxLayout()
        apply_margins_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        self._apply_margins_btn = QPushButton("Apply To ...")
        self._apply_margins_btn.setObjectName("applyMarginsBtn")
        apply_margins_layout.addWidget(self._apply_margins_btn)
        apply_margins_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        margins_layout.addLayout(apply_margins_layout)

        layout.addWidget(self._margins_group)

        # Alignment group
        self._alignment_group = CollapsibleGroupBox("Alignment", self)
        self._alignment_group.setObjectName("alignmentGroup")
        alignment_layout = QVBoxLayout(self._alignment_group)

        # Match size checkbox
        match_layout = QHBoxLayout()
        match_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        self._match_size_check = QCheckBox("Match size with other pages")
        self._match_size_check.setObjectName("alignWithOthersCB")
        self._match_size_check.setChecked(True)
        match_layout.addWidget(self._match_size_check)
        match_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        alignment_layout.addLayout(match_layout)

        # Alignment mode dropdowns
        mode_layout = QHBoxLayout()
        mode_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )

        mode_grid = QGridLayout()
        mode_grid.addWidget(QLabel("Horizontal mode:"), 0, 0)
        self._h_mode_combo = QComboBox()
        self._h_mode_combo.setObjectName("hAlignmentModeCB")
        self._h_mode_combo.addItems(["Auto", "Manual", "Original"])
        self._h_mode_combo.setCurrentIndex(1)  # Manual by default
        mode_grid.addWidget(self._h_mode_combo, 0, 1)

        mode_grid.addWidget(QLabel("Vertical mode:"), 1, 0)
        self._v_mode_combo = QComboBox()
        self._v_mode_combo.setObjectName("vAlignmentModeCB")
        self._v_mode_combo.addItems(["Auto", "Manual", "Original"])
        self._v_mode_combo.setCurrentIndex(1)  # Manual by default
        mode_grid.addWidget(self._v_mode_combo, 1, 1)

        mode_layout.addLayout(mode_grid)
        mode_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        alignment_layout.addLayout(mode_layout)

        # 9-point alignment grid
        align_grid_layout = QHBoxLayout()
        align_grid_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )

        align_grid = QGridLayout()
        align_grid.setSpacing(16)

        self._align_button_group = QButtonGroup(self)

        # Create 3x3 alignment button grid
        align_buttons = [
            ("↖", 0, 0, "alignTopLeftBtn"),
            ("↑", 0, 1, "alignTopBtn"),
            ("↗", 0, 2, "alignTopRightBtn"),
            ("←", 1, 0, "alignLeftBtn"),
            ("●", 1, 1, "alignCenterBtn"),
            ("→", 1, 2, "alignRightBtn"),
            ("↙", 2, 0, "alignBottomLeftBtn"),
            ("↓", 2, 1, "alignBottomBtn"),
            ("↘", 2, 2, "alignBottomRightBtn"),
        ]

        self._align_buttons: dict[str, QToolButton] = {}
        for text, row, col, name in align_buttons:
            btn = QToolButton()
            btn.setObjectName(name)
            btn.setText(text)
            btn.setCheckable(True)
            btn.setAutoExclusive(True)
            btn.setMinimumSize(32, 32)
            if name == "alignCenterBtn":
                btn.setChecked(True)
            align_grid.addWidget(btn, row, col)
            self._align_button_group.addButton(btn)
            self._align_buttons[name] = btn

        align_grid_layout.addLayout(align_grid)
        align_grid_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        alignment_layout.addLayout(align_grid_layout)

        # Apply alignment button
        apply_align_layout = QHBoxLayout()
        apply_align_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        self._apply_alignment_btn = QPushButton("Apply To ...")
        self._apply_alignment_btn.setObjectName("applyAlignmentBtn")
        apply_align_layout.addWidget(self._apply_alignment_btn)
        apply_align_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        alignment_layout.addLayout(apply_align_layout)

        layout.addWidget(self._alignment_group)

        # Vertical spacer
        layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        )

    def _connect_signals(self) -> None:
        """Connect widget signals to slots."""
        # Margins
        self._auto_margins_check.toggled.connect(self._on_auto_margins_changed)
        self._top_spinbox.valueChanged.connect(self._on_top_margin_changed)
        self._bottom_spinbox.valueChanged.connect(self._on_bottom_margin_changed)
        self._left_spinbox.valueChanged.connect(self._on_left_margin_changed)
        self._right_spinbox.valueChanged.connect(self._on_right_margin_changed)
        self._top_bottom_link.toggled.connect(self._on_top_bottom_link_changed)
        self._left_right_link.toggled.connect(self._on_left_right_link_changed)
        self._apply_margins_btn.clicked.connect(self.apply_margins_requested.emit)

        # Alignment
        self._match_size_check.toggled.connect(self._on_match_size_changed)
        self._h_mode_combo.currentIndexChanged.connect(self._on_h_mode_changed)
        self._v_mode_combo.currentIndexChanged.connect(self._on_v_mode_changed)
        self._align_button_group.buttonClicked.connect(
            self._on_alignment_button_clicked
        )
        self._apply_alignment_btn.clicked.connect(self.apply_alignment_requested.emit)

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

            # Update margins
            self._auto_margins_check.setChecked(params.auto_margins)
            margins = params.hard_margins_mm
            self._top_spinbox.setValue(margins.top)
            self._bottom_spinbox.setValue(margins.bottom)
            self._left_spinbox.setValue(margins.left)
            self._right_spinbox.setValue(margins.right)

            # Enable/disable spinboxes based on auto margins
            spinboxes_enabled = not params.auto_margins
            self._top_spinbox.setEnabled(spinboxes_enabled)
            self._bottom_spinbox.setEnabled(spinboxes_enabled)
            self._left_spinbox.setEnabled(spinboxes_enabled)
            self._right_spinbox.setEnabled(spinboxes_enabled)

            # Update alignment
            alignment = params.alignment
            self._match_size_check.setChecked(alignment.aligned_with_others())

            # Update mode combos
            h_mode_index = self._get_h_mode_index(alignment.horizontal)
            v_mode_index = self._get_v_mode_index(alignment.vertical)
            self._h_mode_combo.setCurrentIndex(h_mode_index)
            self._v_mode_combo.setCurrentIndex(v_mode_index)

            # Update alignment buttons
            self._update_alignment_buttons(alignment)
        finally:
            self._updating = False

    def _get_h_mode_index(self, mode: HorizontalAlignment) -> int:
        """Get combo box index for horizontal alignment mode."""
        if mode == HorizontalAlignment.HAUTO:
            return 0
        if mode == HorizontalAlignment.HORIGINAL:
            return 2
        return 1  # Manual

    def _get_v_mode_index(self, mode: VerticalAlignment) -> int:
        """Get combo box index for vertical alignment mode."""
        if mode == VerticalAlignment.VAUTO:
            return 0
        if mode == VerticalAlignment.VORIGINAL:
            return 2
        return 1  # Manual

    def _update_alignment_buttons(self, alignment: Alignment) -> None:
        """Update which alignment button is checked."""
        # Map alignment to button name
        h = alignment.horizontal
        v = alignment.vertical

        v_map = {
            VerticalAlignment.TOP: "Top",
            VerticalAlignment.VCENTER: "",
            VerticalAlignment.BOTTOM: "Bottom",
        }
        h_map = {
            HorizontalAlignment.LEFT: "Left",
            HorizontalAlignment.HCENTER: "",
            HorizontalAlignment.RIGHT: "Right",
        }

        v_part = v_map.get(v, "")
        h_part = h_map.get(h, "")

        if not v_part and not h_part:
            btn_name = "alignCenterBtn"
        elif not v_part:
            btn_name = f"align{h_part}Btn"
        elif not h_part:
            btn_name = f"align{v_part}Btn"
        else:
            btn_name = f"align{v_part}{h_part}Btn"

        if btn_name in self._align_buttons:
            self._align_buttons[btn_name].setChecked(True)

    def _on_auto_margins_changed(self, checked: bool) -> None:
        """Handle auto margins checkbox change."""
        if self._current_page_id is None or self._updating:
            return

        params = self._filter.get_params(self._current_page_id)
        new_params = params.with_auto_margins(checked)
        self._filter.set_params(self._current_page_id, new_params)
        self._update_ui()

    def _emit_margins_changed(self) -> None:
        """Emit margins_changed signal with current values."""
        if self._current_page_id is None:
            return

        margins = Margins(
            top=self._top_spinbox.value(),
            bottom=self._bottom_spinbox.value(),
            left=self._left_spinbox.value(),
            right=self._right_spinbox.value(),
        )
        params = self._filter.get_params(self._current_page_id)
        new_params = params.with_margins(margins)
        self._filter.set_params(self._current_page_id, new_params)
        self.margins_changed.emit(self._current_page_id, margins)

    def _on_top_margin_changed(self, value: float) -> None:
        """Handle top margin spinbox change."""
        if self._current_page_id is None or self._updating:
            return

        if self._link_top_bottom:
            self._updating = True
            self._bottom_spinbox.setValue(value)
            self._updating = False

        self._emit_margins_changed()

    def _on_bottom_margin_changed(self, value: float) -> None:
        """Handle bottom margin spinbox change."""
        if self._current_page_id is None or self._updating:
            return

        if self._link_top_bottom:
            self._updating = True
            self._top_spinbox.setValue(value)
            self._updating = False

        self._emit_margins_changed()

    def _on_left_margin_changed(self, value: float) -> None:
        """Handle left margin spinbox change."""
        if self._current_page_id is None or self._updating:
            return

        if self._link_left_right:
            self._updating = True
            self._right_spinbox.setValue(value)
            self._updating = False

        self._emit_margins_changed()

    def _on_right_margin_changed(self, value: float) -> None:
        """Handle right margin spinbox change."""
        if self._current_page_id is None or self._updating:
            return

        if self._link_left_right:
            self._updating = True
            self._left_spinbox.setValue(value)
            self._updating = False

        self._emit_margins_changed()

    def _on_top_bottom_link_changed(self, checked: bool) -> None:
        """Handle top/bottom link button change."""
        self._link_top_bottom = checked

    def _on_left_right_link_changed(self, checked: bool) -> None:
        """Handle left/right link button change."""
        self._link_left_right = checked

    def _on_match_size_changed(self, checked: bool) -> None:
        """Handle match size checkbox change."""
        if self._current_page_id is None or self._updating:
            return

        params = self._filter.get_params(self._current_page_id)
        new_alignment = Alignment(
            vertical=params.alignment.vertical,
            horizontal=params.alignment.horizontal,
            is_null=not checked,
        )
        new_params = params.with_alignment(new_alignment)
        self._filter.set_params(self._current_page_id, new_params)
        self.alignment_changed.emit(self._current_page_id, new_alignment)

    def _on_h_mode_changed(self, index: int) -> None:
        """Handle horizontal mode combo change."""
        if self._current_page_id is None or self._updating:
            return

        h_modes = [
            HorizontalAlignment.HAUTO,
            HorizontalAlignment.HCENTER,
            HorizontalAlignment.HORIGINAL,
        ]
        params = self._filter.get_params(self._current_page_id)
        new_alignment = Alignment(
            vertical=params.alignment.vertical,
            horizontal=h_modes[index],
            is_null=params.alignment.is_null,
        )
        new_params = params.with_alignment(new_alignment)
        self._filter.set_params(self._current_page_id, new_params)
        self.alignment_changed.emit(self._current_page_id, new_alignment)

    def _on_v_mode_changed(self, index: int) -> None:
        """Handle vertical mode combo change."""
        if self._current_page_id is None or self._updating:
            return

        v_modes = [
            VerticalAlignment.VAUTO,
            VerticalAlignment.VCENTER,
            VerticalAlignment.VORIGINAL,
        ]
        params = self._filter.get_params(self._current_page_id)
        new_alignment = Alignment(
            vertical=v_modes[index],
            horizontal=params.alignment.horizontal,
            is_null=params.alignment.is_null,
        )
        new_params = params.with_alignment(new_alignment)
        self._filter.set_params(self._current_page_id, new_params)
        self.alignment_changed.emit(self._current_page_id, new_alignment)

    def _on_alignment_button_clicked(self, button: QToolButton) -> None:
        """Handle alignment button click."""
        if self._current_page_id is None or self._updating:
            return

        # Parse button name to get alignment
        name = button.objectName()
        v_align = VerticalAlignment.VCENTER
        h_align = HorizontalAlignment.HCENTER

        if "Top" in name:
            v_align = VerticalAlignment.TOP
        elif "Bottom" in name:
            v_align = VerticalAlignment.BOTTOM

        if "Left" in name:
            h_align = HorizontalAlignment.LEFT
        elif "Right" in name:
            h_align = HorizontalAlignment.RIGHT

        params = self._filter.get_params(self._current_page_id)
        new_alignment = Alignment(
            vertical=v_align,
            horizontal=h_align,
            is_null=params.alignment.is_null,
        )
        new_params = params.with_alignment(new_alignment)
        self._filter.set_params(self._current_page_id, new_params)

        # Update mode combos to Manual since we manually selected
        self._updating = True
        self._h_mode_combo.setCurrentIndex(1)  # Manual
        self._v_mode_combo.setCurrentIndex(1)  # Manual
        self._updating = False

        self.alignment_changed.emit(self._current_page_id, new_alignment)
