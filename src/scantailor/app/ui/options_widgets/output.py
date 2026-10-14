"""Options widget for Output filter.

Provides UI controls for the output settings of the current page: resolution,
colour mode, splitting, binarization, colour segmentation, posterization,
filling, despeckling and dewarping.
"""

from collections.abc import Callable
from typing import cast

from PySide6 import QtCore
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import (
    QBoxLayout,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QSpacerItem,
    QSpinBox,
    QStackedWidget,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from scantailor.app.ui.icons import icon
from scantailor.app.ui.theme.fonts import CAPTION_PX
from scantailor.app.ui.widgets.collapsible_group_box import CollapsibleGroupBox
from scantailor.app.ui.widgets.ticked_slider import TickedSlider
from scantailor.core.models import Dpi, PageId
from scantailor.dewarping.auto_model import DewarpMetrics
from scantailor.filters.output.binarization import BinarizationMethod
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.color_options import FillingColor
from scantailor.filters.output.despeckle import DespeckleLevel
from scantailor.filters.output.dewarping_options import DewarpingMode
from scantailor.filters.output.filter import Filter
from scantailor.filters.output.params import ApplyGroup, DistortionModelSpec, Params
from scantailor.filters.output.picture_shape import PictureShape

_DEPTH_PERCEPTION_SCALE = 10  # Slider steps per unit of depth perception.
_DESPECKLE_LEVELS = (
    (DespeckleLevel.OFF, QtCore.QT_TRANSLATE_NOOP("output::OptionsWidget", "Off")),
    (
        DespeckleLevel.CAUTIOUS,
        QtCore.QT_TRANSLATE_NOOP("output::OptionsWidget", "Cautious"),
    ),
    (
        DespeckleLevel.NORMAL,
        QtCore.QT_TRANSLATE_NOOP("output::OptionsWidget", "Normal"),
    ),
    (
        DespeckleLevel.AGGRESSIVE,
        QtCore.QT_TRANSLATE_NOOP("output::OptionsWidget", "Aggressive"),
    ),
)
_DEWARPING_MODES = (
    (DewarpingMode.OFF, QtCore.QT_TRANSLATE_NOOP("output::OptionsWidget", "Off")),
    (DewarpingMode.AUTO, QtCore.QT_TRANSLATE_NOOP("output::OptionsWidget", "Auto")),
    (DewarpingMode.MANUAL, QtCore.QT_TRANSLATE_NOOP("output::OptionsWidget", "Manual")),
    (
        DewarpingMode.MARGINAL,
        QtCore.QT_TRANSLATE_NOOP("output::OptionsWidget", "Marginal"),
    ),
)
_METHOD_LABELS = {
    BinarizationMethod.OTSU: QtCore.QT_TRANSLATE_NOOP("output::OptionsWidget", "Otsu"),
    BinarizationMethod.SAUVOLA: QtCore.QT_TRANSLATE_NOOP(
        "output::OptionsWidget", "Sauvola"
    ),
    BinarizationMethod.WOLF: QtCore.QT_TRANSLATE_NOOP("output::OptionsWidget", "Wolf"),
    BinarizationMethod.FOX: QtCore.QT_TRANSLATE_NOOP("output::OptionsWidget", "Fox"),
    BinarizationMethod.WINDOW: QtCore.QT_TRANSLATE_NOOP(
        "output::OptionsWidget", "Window"
    ),
    BinarizationMethod.BRADLEY: QtCore.QT_TRANSLATE_NOOP(
        "output::OptionsWidget", "Bradley"
    ),
    BinarizationMethod.GRAD: QtCore.QT_TRANSLATE_NOOP("output::OptionsWidget", "Grad"),
    BinarizationMethod.EDGEPLUS: QtCore.QT_TRANSLATE_NOOP(
        "output::OptionsWidget", "EdgePlus"
    ),
    BinarizationMethod.BLURDIV: QtCore.QT_TRANSLATE_NOOP(
        "output::OptionsWidget", "BlurDiv"
    ),
    BinarizationMethod.EDGEDIV: QtCore.QT_TRANSLATE_NOOP(
        "output::OptionsWidget", "EdgeDiv"
    ),
}
_PICTURE_SHAPES = (
    (PictureShape.OFF, QtCore.QT_TRANSLATE_NOOP("output::OptionsWidget", "Off")),
    (PictureShape.FREE, QtCore.QT_TRANSLATE_NOOP("output::OptionsWidget", "Free")),
    (
        PictureShape.RECTANGULAR,
        QtCore.QT_TRANSLATE_NOOP("output::OptionsWidget", "Rectangular"),
    ),
)
# Methods that share the controls of another method's page.
_SHARED_METHOD_PAGE = {
    BinarizationMethod.FOX: BinarizationMethod.WOLF,
    BinarizationMethod.WINDOW: BinarizationMethod.WOLF,
    BinarizationMethod.GRAD: BinarizationMethod.WOLF,
}
_FILLING_LABELS = {
    FillingColor.BACKGROUND: QtCore.QT_TRANSLATE_NOOP(
        "output::OptionsWidget", "Background"
    ),
    FillingColor.WHITE: QtCore.QT_TRANSLATE_NOOP("output::OptionsWidget", "White"),
    FillingColor.BLACK: QtCore.QT_TRANSLATE_NOOP("output::OptionsWidget", "Black"),
}


def _spacer() -> QSpacerItem:
    return QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)


def _double_spin_box(
    name: str, low: float, high: float, step: float, decimals: int = 2
) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setObjectName(name)
    box.setDecimals(decimals)
    box.setRange(low, high)
    box.setSingleStep(step)
    return box


def _spin_box(name: str, low: int, high: int) -> QSpinBox:
    box = QSpinBox()
    box.setObjectName(name)
    box.setRange(low, high)
    return box


def _check_box(name: str, text: str, tooltip: str = "") -> QCheckBox:
    box = QCheckBox(text)
    box.setObjectName(name)
    if tooltip:
        box.setToolTip(tooltip)
    return box


def _apply_button(layout: QVBoxLayout | QHBoxLayout, group: ApplyGroup) -> QPushButton:
    """Add the button that copies one option group to other pages."""
    button = QPushButton(
        QtCore.QCoreApplication.translate("output::OptionsWidget", "Apply to ...")
    )
    button.setObjectName(f"apply_{group.value}_btn")
    button.setToolTip(
        QtCore.QCoreApplication.translate(
            "output::OptionsWidget", "Copy these settings to other pages"
        )
    )
    layout.addWidget(button)
    return button


def _despeckle_step(level: DespeckleLevel) -> int:
    return next(i for i, (lvl, _label) in enumerate(_DESPECKLE_LEVELS) if lvl is level)


def _despeckle_tooltip(strength: float) -> str:
    """Name the strength when it lies between the steps the slider offers."""
    if strength in {level.to_factor() for level, _label in _DESPECKLE_LEVELS}:
        return ""
    return f"Strength {strength:g}, from a project that stores values between the steps"


def _exclusive_button(
    group: QButtonGroup, layout: QBoxLayout, name: str, text: str, index: int
) -> QPushButton:
    button = QPushButton(text)
    button.setObjectName(name)
    button.setCheckable(True)
    button.setAutoExclusive(True)
    group.addButton(button, index)
    layout.addWidget(button)
    return button


class _StopLabels(QWidget):
    """A row of labels, each centred under its stop of a slider.

    The outer labels are kept inside the row so they are not clipped.
    """

    def __init__(
        self,
        slider: QSlider,
        labels: list[tuple[str, str]],
        parent: QWidget | None = None,
    ) -> None:
        """Create the row.

        Args:
            slider (QSlider): Slider with one stop per label, from its minimum up.
            labels (list[tuple[str, str]]): Text and object name of each label.
            parent (QWidget | None): Parent widget.
        """
        super().__init__(parent)
        self._slider = slider
        self._labels = []
        for text, name in labels:
            label = QLabel(text, self)
            label.setObjectName(name)
            font = label.font()
            font.setPixelSize(CAPTION_PX)
            label.setFont(font)
            label.adjustSize()
            self._labels.append(label)
        self.setFixedHeight(max(label.height() for label in self._labels))
        self.setMinimumWidth(sum(label.width() for label in self._labels))

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Place the labels under the slider stops.

        Args:
            event (QResizeEvent): The event.
        """
        super().resizeEvent(event)
        handle = self._slider.style().pixelMetric(
            QStyle.PixelMetric.PM_SliderLength, None, self._slider
        )
        span = max(self.width() - handle, 0)
        stops = len(self._labels) - 1
        for step, label in enumerate(self._labels):
            centre = handle // 2 + round(step * span / stops)
            left = min(
                max(centre - label.width() // 2, 0), self.width() - label.width()
            )
            label.move(left, 0)


class OptionsWidget(QWidget):
    """Options widget for the Output filter.

    Every control writes the parameters of the current page and emits
    ``params_changed`` so the page renders again. Controls that take a number
    wait ``RELOAD_DELAY_MS`` after the last change before they emit, so that
    dragging or typing renders once. Which controls show depends on the colour
    mode and on whether the dewarping tab of the image view is active.

    Signals:
        params_changed: Emitted when any parameter changes.
            Args: page_id (PageId)
        apply_group_requested: Emitted when the user clicks the "Apply to..."
            button of an option group.
            Args: group (ApplyGroup)
        despeckle_level_changed: Emitted at once when the user picks another
            despeckle level, so that the despeckling tab can preview it before
            the page renders again.
            Args: level (DespeckleLevel)
    """

    params_changed = Signal(PageId)
    apply_group_requested = Signal(object)
    despeckle_level_changed = Signal(object)

    RELOAD_DELAY_MS = 750

    def __init__(
        self,
        filter_: Filter,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the options widget.

        Args:
            filter_ (Filter): The Output filter instance.
            parent (QWidget | None): The parent widget.
        """
        super().__init__(parent)

        self._filter = filter_
        self._current_page_id: PageId | None = None
        self._updating = False  # Prevent signal loops
        self._dewarping_tab_active = False
        self._despeckle_tab_active = False
        self._dewarp_metrics: DewarpMetrics | None = None
        self._apply_buttons: dict[ApplyGroup, QPushButton] = {}
        self._pending_page_id: PageId | None = None
        self._reload_timer = QTimer(self)
        self._reload_timer.setSingleShot(True)
        self._reload_timer.setInterval(self.RELOAD_DELAY_MS)
        self._reload_timer.timeout.connect(self.flush_pending_reload)

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._build_dpi_group(layout)
        self._build_mode_group(layout)
        self._build_common_group(layout)
        self._build_filling_group(layout)
        self._build_splitting_group(layout)
        self._build_binarization_group(layout)
        self._build_segmentation_group(layout)
        self._build_posterization_group(layout)
        self._build_picture_shape_group(layout)
        self._build_despeckle_group(layout)
        self._build_dewarping_group(layout)
        self._build_depth_group(layout)

        layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        )

    def _build_dpi_group(self, layout: QVBoxLayout) -> None:
        self._dpi_group = CollapsibleGroupBox(
            QtCore.QCoreApplication.translate(
                "output::OptionsWidget", "Output Resolution"
            ),
            self,
        )
        self._dpi_group.setObjectName("dpiGroup")
        dpi_layout = QHBoxLayout(self._dpi_group)
        dpi_layout.addItem(_spacer())
        dpi_layout.addWidget(
            QLabel(QtCore.QCoreApplication.translate("output::OptionsWidget", "DPI:"))
        )
        self._dpi_spinbox = _spin_box("dpiSpinBox", 50, 1200)
        self._dpi_spinbox.setValue(600)
        self._dpi_spinbox.setSingleStep(50)
        dpi_layout.addWidget(self._dpi_spinbox)
        dpi_layout.addItem(_spacer())
        self._apply_buttons[ApplyGroup.DPI] = _apply_button(dpi_layout, ApplyGroup.DPI)
        layout.addWidget(self._dpi_group)

    def _build_mode_group(self, layout: QVBoxLayout) -> None:
        self._mode_group = CollapsibleGroupBox(
            QtCore.QCoreApplication.translate("output::OptionsWidget", "Mode"), self
        )
        self._mode_group.setObjectName("modeGroup")
        mode_layout = QVBoxLayout(self._mode_group)

        mode_btn_layout = QVBoxLayout()
        self._mode_button_group = QButtonGroup(self)
        self._bw_btn = _exclusive_button(
            self._mode_button_group,
            mode_btn_layout,
            "bwBtn",
            QtCore.QCoreApplication.translate(
                "output::OptionsWidget", "Black and White"
            ),
            0,
        )
        self._gray_btn = _exclusive_button(
            self._mode_button_group,
            mode_btn_layout,
            "grayBtn",
            QtCore.QCoreApplication.translate(
                "output::OptionsWidget", "Color / Grayscale"
            ),
            1,
        )
        self._mixed_btn = _exclusive_button(
            self._mode_button_group,
            mode_btn_layout,
            "mixedBtn",
            QtCore.QCoreApplication.translate("output::OptionsWidget", "Mixed"),
            2,
        )
        self._mixed_btn.setToolTip(
            QtCore.QCoreApplication.translate(
                "output::OptionsWidget", "Black and white with colorful pictures"
            )
        )
        self._bw_btn.setIcon(icon("contrast"))
        self._gray_btn.setIcon(icon("palette"))
        self._mixed_btn.setIcon(icon("colour-mode-mixed"))
        self._bw_btn.setChecked(True)
        mode_layout.addLayout(mode_btn_layout)

        bow_layout = QHBoxLayout()
        bow_layout.addItem(_spacer())
        self._black_on_white_check = _check_box(
            "blackOnWhiteCB",
            QtCore.QCoreApplication.translate("output::OptionsWidget", "White margins"),
            QtCore.QCoreApplication.translate(
                "output::OptionsWidget",
                "Checked: black text on white background\n"
                "Unchecked: white text on black background",
            ),
        )
        self._black_on_white_check.setChecked(True)
        bow_layout.addWidget(self._black_on_white_check)
        bow_layout.addItem(_spacer())
        mode_layout.addLayout(bow_layout)

        self._apply_buttons[ApplyGroup.COLOR_MODE] = _apply_button(
            mode_layout, ApplyGroup.COLOR_MODE
        )
        layout.addWidget(self._mode_group)

    def _build_common_group(self, layout: QVBoxLayout) -> None:
        self._common_group = CollapsibleGroupBox(
            QtCore.QCoreApplication.translate("output::OptionsWidget", "Options"), self
        )
        self._common_group.setObjectName("commonGroup")
        common_layout = QVBoxLayout(self._common_group)

        self._fill_margins_check = _check_box(
            "fillMarginsCB",
            QtCore.QCoreApplication.translate("output::OptionsWidget", "Fill margins"),
        )
        self._fill_offcut_check = _check_box(
            "fillOffcutCB",
            QtCore.QCoreApplication.translate("output::OptionsWidget", "Fill offcut"),
        )
        self._fill_outside_page_box_check = _check_box(
            "fillOutsidePageBoxCB",
            QtCore.QCoreApplication.translate(
                "output::OptionsWidget", "Fill outside page box"
            ),
            QtCore.QCoreApplication.translate(
                "output::OptionsWidget",
                "Fill the full output page rectangle with the background color outside "
                "the page content, instead of following offcut geometry.",
            ),
        )
        self._fill_outside_page_box_check.setIcon(icon("squares-exclude"))
        self._equalize_check = _check_box(
            "equalizeIlluminationCB",
            QtCore.QCoreApplication.translate(
                "output::OptionsWidget", "Equalize illumination"
            ),
        )
        self._equalize_color_check = _check_box(
            "equalizeIlluminationColorCB",
            QtCore.QCoreApplication.translate(
                "output::OptionsWidget", "Equalize illumination (color)"
            ),
        )
        self._savitzky_golay_check = _check_box(
            "savitzkyGolaySmoothingCB",
            QtCore.QCoreApplication.translate(
                "output::OptionsWidget", "Savitzky-Golay smoothing"
            ),
        )
        self._morphological_check = _check_box(
            "morphologicalSmoothingCB",
            QtCore.QCoreApplication.translate(
                "output::OptionsWidget", "Morphological smoothing"
            ),
        )
        for widget in (
            self._fill_margins_check,
            self._fill_offcut_check,
            self._fill_outside_page_box_check,
            self._equalize_check,
            self._equalize_color_check,
            self._savitzky_golay_check,
            self._morphological_check,
        ):
            common_layout.addWidget(widget)

        wiener_layout = QGridLayout()
        wiener_layout.addWidget(
            QLabel(
                QtCore.QCoreApplication.translate(
                    "output::OptionsWidget", "Wiener coefficient:"
                )
            ),
            0,
            0,
        )
        self._wiener_coef_spinbox = _double_spin_box(
            "wienerCoefSpinBox", 0.0, 1.0, 0.05
        )
        wiener_layout.addWidget(self._wiener_coef_spinbox, 0, 1)
        wiener_layout.addWidget(
            QLabel(
                QtCore.QCoreApplication.translate(
                    "output::OptionsWidget", "Wiener window:"
                )
            ),
            1,
            0,
        )
        self._wiener_window_spinbox = _spin_box("wienerWindowSpinBox", 3, 99)
        wiener_layout.addWidget(self._wiener_window_spinbox, 1, 1)
        common_layout.addLayout(wiener_layout)

        layout.addWidget(self._common_group)

    def _build_filling_group(self, layout: QVBoxLayout) -> None:
        self._filling_group = CollapsibleGroupBox(
            QtCore.QCoreApplication.translate("output::OptionsWidget", "Filling"), self
        )
        self._filling_group.setObjectName("fillingGroup")
        filling_layout = QHBoxLayout(self._filling_group)
        filling_layout.addWidget(
            QLabel(
                QtCore.QCoreApplication.translate(
                    "output::OptionsWidget", "Filling color:"
                )
            )
        )
        self._filling_color_combo = QComboBox()
        self._filling_color_combo.setObjectName("fillingColorBox")
        for color, label in _FILLING_LABELS.items():
            self._filling_color_combo.addItem(
                QtCore.QCoreApplication.translate(
                    "output::OptionsWidget", cast(str, label)
                ),
                color,
            )
        filling_layout.addWidget(self._filling_color_combo)
        filling_layout.addItem(_spacer())
        layout.addWidget(self._filling_group)

    def _build_splitting_group(self, layout: QVBoxLayout) -> None:
        self._splitting_group = CollapsibleGroupBox(
            QtCore.QCoreApplication.translate("output::OptionsWidget", "Splitting"),
            self,
        )
        self._splitting_group.setObjectName("splittingGroup")
        splitting_layout = QVBoxLayout(self._splitting_group)

        self._split_output_check = _check_box(
            "splittingCB",
            QtCore.QCoreApplication.translate("output::OptionsWidget", "Split output"),
        )
        splitting_layout.addWidget(self._split_output_check)
        self._bw_foreground_radio = QRadioButton(
            QtCore.QCoreApplication.translate(
                "output::OptionsWidget", "Black and white foreground"
            )
        )
        self._bw_foreground_radio.setObjectName("bwForegroundRB")
        self._color_foreground_radio = QRadioButton(
            QtCore.QCoreApplication.translate(
                "output::OptionsWidget", "Color foreground"
            )
        )
        self._color_foreground_radio.setObjectName("colorForegroundRB")
        self._bw_foreground_radio.setChecked(True)
        splitting_layout.addWidget(self._bw_foreground_radio)
        splitting_layout.addWidget(self._color_foreground_radio)
        self._original_background_check = _check_box(
            "originalBackgroundCB",
            QtCore.QCoreApplication.translate(
                "output::OptionsWidget", "Original background"
            ),
        )
        splitting_layout.addWidget(self._original_background_check)

        self._apply_buttons[ApplyGroup.SPLITTING] = _apply_button(
            splitting_layout, ApplyGroup.SPLITTING
        )
        layout.addWidget(self._splitting_group)

    def _build_binarization_group(self, layout: QVBoxLayout) -> None:
        self._binarization_group = CollapsibleGroupBox(
            QtCore.QCoreApplication.translate("output::OptionsWidget", "Binarization"),
            self,
        )
        self._binarization_group.setObjectName("binarizationGroup")
        bin_layout = QVBoxLayout(self._binarization_group)

        method_layout = QHBoxLayout()
        method_layout.addWidget(
            QLabel(
                QtCore.QCoreApplication.translate("output::OptionsWidget", "Method:")
            )
        )
        self._method_combo = QComboBox()
        self._method_combo.setObjectName("methodCombo")
        for method, label in _METHOD_LABELS.items():
            self._method_combo.addItem(
                QtCore.QCoreApplication.translate(
                    "output::OptionsWidget", cast(str, label)
                ),
                method,
            )
        method_layout.addWidget(self._method_combo)
        method_layout.addItem(_spacer())
        bin_layout.addLayout(method_layout)

        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(
            QLabel(
                QtCore.QCoreApplication.translate("output::OptionsWidget", "Threshold:")
            )
        )
        self._threshold_slider = QSlider()
        self._threshold_slider.setObjectName("thresholdSlider")
        self._threshold_slider.setOrientation(Qt.Orientation.Horizontal)
        self._threshold_slider.setRange(-100, 100)
        thresh_layout.addWidget(self._threshold_slider)
        self._threshold_label = QLabel("0")
        self._threshold_label.setMinimumWidth(30)
        thresh_layout.addWidget(self._threshold_label)
        bin_layout.addLayout(thresh_layout)

        self._window_size_row = QWidget()
        window_layout = QHBoxLayout(self._window_size_row)
        window_layout.setContentsMargins(0, 0, 0, 0)
        window_layout.addWidget(
            QLabel(
                QtCore.QCoreApplication.translate(
                    "output::OptionsWidget", "Window size:"
                )
            )
        )
        self._window_size_spinbox = _spin_box("windowSizeSpinBox", 10, 1000)
        self._window_size_spinbox.setValue(200)
        window_layout.addWidget(self._window_size_spinbox)
        window_layout.addItem(_spacer())
        bin_layout.addWidget(self._window_size_row)

        self._method_options = QStackedWidget()
        self._method_options.setObjectName("methodOptions")
        self._build_method_pages()
        bin_layout.addWidget(self._method_options)

        self._apply_buttons[ApplyGroup.BINARIZATION] = _apply_button(
            bin_layout, ApplyGroup.BINARIZATION
        )
        layout.addWidget(self._binarization_group)

    def _build_method_pages(self) -> None:
        """Add one page of method specific controls per binarization method."""
        self._sauvola_coef_spinbox = _double_spin_box(
            "sauvolaCoefSpinBox", 0.0, 1.0, 0.01
        )
        self._wolf_coef_spinbox = _double_spin_box("wolfCoefSpinBox", 0.0, 1.0, 0.01)
        self._wolf_lower_spinbox = _spin_box("wolfLowerBoundSpinBox", 0, 255)
        self._wolf_upper_spinbox = _spin_box("wolfUpperBoundSpinBox", 0, 255)
        self._bradley_coef_spinbox = _double_spin_box(
            "bradleyCoefSpinBox", 0.0, 1.0, 0.01
        )
        self._edge_div_kep_spinbox = _double_spin_box(
            "edgeDivKepSpinBox", 0.0, 1.0, 0.05
        )
        self._edge_div_kbd_spinbox = _double_spin_box(
            "edgeDivKbdSpinBox", 0.0, 1.0, 0.05
        )
        self._edge_plus_coef_spinbox = _double_spin_box(
            "edgePlusCoefSpinBox", 0.0, 1.0, 0.05
        )
        self._blur_div_coef_spinbox = _double_spin_box(
            "blurDivCoefSpinBox", 0.0, 1.0, 0.05
        )

        pages: dict[BinarizationMethod, list[tuple[str, QWidget]]] = {
            BinarizationMethod.OTSU: [],
            BinarizationMethod.SAUVOLA: [
                (
                    QtCore.QCoreApplication.translate(
                        "output::OptionsWidget", "Coefficient:"
                    ),
                    self._sauvola_coef_spinbox,
                )
            ],
            BinarizationMethod.WOLF: [
                (
                    QtCore.QCoreApplication.translate(
                        "output::OptionsWidget", "Coefficient:"
                    ),
                    self._wolf_coef_spinbox,
                ),
                (
                    QtCore.QCoreApplication.translate(
                        "output::OptionsWidget", "Lower bound:"
                    ),
                    self._wolf_lower_spinbox,
                ),
                (
                    QtCore.QCoreApplication.translate(
                        "output::OptionsWidget", "Upper bound:"
                    ),
                    self._wolf_upper_spinbox,
                ),
            ],
            BinarizationMethod.BRADLEY: [
                (
                    QtCore.QCoreApplication.translate(
                        "output::OptionsWidget", "Coefficient:"
                    ),
                    self._bradley_coef_spinbox,
                )
            ],
            BinarizationMethod.EDGEPLUS: [
                (
                    QtCore.QCoreApplication.translate(
                        "output::OptionsWidget", "Coefficient:"
                    ),
                    self._edge_plus_coef_spinbox,
                )
            ],
            BinarizationMethod.BLURDIV: [
                (
                    QtCore.QCoreApplication.translate(
                        "output::OptionsWidget", "Coefficient:"
                    ),
                    self._blur_div_coef_spinbox,
                )
            ],
            BinarizationMethod.EDGEDIV: [
                (
                    QtCore.QCoreApplication.translate(
                        "output::OptionsWidget", "Edge weight:"
                    ),
                    self._edge_div_kep_spinbox,
                ),
                (
                    QtCore.QCoreApplication.translate(
                        "output::OptionsWidget", "Blur weight:"
                    ),
                    self._edge_div_kbd_spinbox,
                ),
            ],
        }
        self._method_page_index: dict[BinarizationMethod, int] = {}
        for method, controls in pages.items():
            page = QWidget()
            grid = QGridLayout(page)
            grid.setContentsMargins(0, 0, 0, 0)
            for row, (label, spinbox) in enumerate(controls):
                grid.addWidget(QLabel(label), row, 0)
                grid.addWidget(spinbox, row, 1)
            self._method_page_index[method] = self._method_options.addWidget(page)
        for method, shared in _SHARED_METHOD_PAGE.items():
            self._method_page_index[method] = self._method_page_index[shared]
        self._method_options.currentChanged.connect(self._fit_method_options)

    def _fit_method_options(self, index: int) -> None:
        """Size the stack to the shown page only, and hide it when that is empty.

        A stacked widget is as tall as its tallest page, which leaves a gap
        under the shorter ones.
        """
        for page_index in range(self._method_options.count()):
            page = self._method_options.widget(page_index)
            if page is None:
                continue
            vertical = (
                QSizePolicy.Policy.Preferred
                if page_index == index
                else QSizePolicy.Policy.Ignored
            )
            page.setSizePolicy(QSizePolicy.Policy.Preferred, vertical)
        current = self._method_options.widget(index)
        page_layout = current.layout() if current is not None else None
        self._method_options.setVisible(
            page_layout is not None and page_layout.count() > 0
        )
        self._method_options.updateGeometry()

    def _build_segmentation_group(self, layout: QVBoxLayout) -> None:
        self._segmentation_group = CollapsibleGroupBox(
            QtCore.QCoreApplication.translate(
                "output::OptionsWidget", "Color segmentation"
            ),
            self,
        )
        self._segmentation_group.setObjectName("segmentationGroup")
        seg_layout = QVBoxLayout(self._segmentation_group)

        self._color_segmentation_check = _check_box(
            "colorSegmentationCB",
            QtCore.QCoreApplication.translate(
                "output::OptionsWidget", "Segment colors"
            ),
        )
        seg_layout.addWidget(self._color_segmentation_check)

        self._segmenter_options = QWidget()
        grid = QGridLayout(self._segmenter_options)
        grid.setContentsMargins(0, 0, 0, 0)
        self._reduce_noise_spinbox = _spin_box("reduceNoiseSB", 0, 999)
        self._red_adjustment_spinbox = _spin_box("redAdjustmentSB", -255, 255)
        self._green_adjustment_spinbox = _spin_box("greenAdjustmentSB", -255, 255)
        self._blue_adjustment_spinbox = _spin_box("blueAdjustmentSB", -255, 255)
        for row, (label, spinbox) in enumerate(
            (
                (
                    QtCore.QCoreApplication.translate(
                        "output::OptionsWidget", "Noise reduction:"
                    ),
                    self._reduce_noise_spinbox,
                ),
                (
                    QtCore.QCoreApplication.translate(
                        "output::OptionsWidget", "Red adjustment:"
                    ),
                    self._red_adjustment_spinbox,
                ),
                (
                    QtCore.QCoreApplication.translate(
                        "output::OptionsWidget", "Green adjustment:"
                    ),
                    self._green_adjustment_spinbox,
                ),
                (
                    QtCore.QCoreApplication.translate(
                        "output::OptionsWidget", "Blue adjustment:"
                    ),
                    self._blue_adjustment_spinbox,
                ),
            )
        ):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(spinbox, row, 1)
        seg_layout.addWidget(self._segmenter_options)

        layout.addWidget(self._segmentation_group)

    def _build_posterization_group(self, layout: QVBoxLayout) -> None:
        self._posterization_group = CollapsibleGroupBox(
            QtCore.QCoreApplication.translate("output::OptionsWidget", "Posterization"),
            self,
        )
        self._posterization_group.setObjectName("posterizationGroup")
        poster_layout = QVBoxLayout(self._posterization_group)

        self._posterize_check = _check_box(
            "posterizeCB",
            QtCore.QCoreApplication.translate("output::OptionsWidget", "Posterize"),
        )
        poster_layout.addWidget(self._posterize_check)

        self._posterize_options = QWidget()
        grid = QGridLayout(self._posterize_options)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(
            QLabel(
                QtCore.QCoreApplication.translate("output::OptionsWidget", "Level:")
            ),
            0,
            0,
        )
        self._posterize_level_spinbox = _spin_box("posterizeLevelSB", 2, 255)
        grid.addWidget(self._posterize_level_spinbox, 0, 1)
        self._posterize_normalize_check = _check_box(
            "posterizeNormalizationCB",
            QtCore.QCoreApplication.translate("output::OptionsWidget", "Normalize"),
        )
        grid.addWidget(self._posterize_normalize_check, 1, 0, 1, 2)
        self._posterize_force_bw_check = _check_box(
            "posterizeForceBwCB",
            QtCore.QCoreApplication.translate(
                "output::OptionsWidget", "Force black and white"
            ),
        )
        grid.addWidget(self._posterize_force_bw_check, 2, 0, 1, 2)
        poster_layout.addWidget(self._posterize_options)

        layout.addWidget(self._posterization_group)

    def _build_picture_shape_group(self, layout: QVBoxLayout) -> None:
        self._picture_shape_group = CollapsibleGroupBox(
            QtCore.QCoreApplication.translate("output::OptionsWidget", "Picture Shape"),
            self,
        )
        self._picture_shape_group.setObjectName("pictureShapeOptions")
        shape_layout = QVBoxLayout(self._picture_shape_group)

        selector_layout = QHBoxLayout()
        selector_layout.addItem(_spacer())
        self._picture_shape_combo = QComboBox()
        self._picture_shape_combo.setObjectName("pictureShapeSelector")
        for shape, label in _PICTURE_SHAPES:
            self._picture_shape_combo.addItem(
                QtCore.QCoreApplication.translate(
                    "output::OptionsWidget", cast(str, label)
                ),
                shape,
            )
        selector_layout.addWidget(self._picture_shape_combo)
        selector_layout.addItem(_spacer())
        shape_layout.addLayout(selector_layout)

        self._picture_sensitivity_row = QWidget()
        self._picture_sensitivity_row.setObjectName("pictureShapeSensitivityOptions")
        sensitivity_layout = QHBoxLayout(self._picture_sensitivity_row)
        sensitivity_layout.setContentsMargins(0, 0, 0, 0)
        sensitivity_layout.addWidget(
            QLabel(
                QtCore.QCoreApplication.translate(
                    "output::OptionsWidget", "Sensitivity (%):"
                )
            )
        )
        self._picture_sensitivity_spinbox = _spin_box(
            "pictureShapeSensitivitySB", 0, 100
        )
        sensitivity_layout.addWidget(self._picture_sensitivity_spinbox)
        sensitivity_layout.addItem(_spacer())
        shape_layout.addWidget(self._picture_sensitivity_row)

        self._higher_search_check = _check_box(
            "higherSearchSensitivityCB",
            QtCore.QCoreApplication.translate(
                "output::OptionsWidget", "Higher search sensitivity"
            ),
        )
        shape_layout.addWidget(self._higher_search_check)
        layout.addWidget(self._picture_shape_group)

    def _build_despeckle_group(self, layout: QVBoxLayout) -> None:
        self._despeckle_group = CollapsibleGroupBox(
            QtCore.QCoreApplication.translate("output::OptionsWidget", "Despeckle"),
            self,
        )
        self._despeckle_group.setObjectName("despeckleGroup")
        despeckle_layout = QVBoxLayout(self._despeckle_group)

        self._despeckle_slider = TickedSlider(Qt.Orientation.Horizontal)
        self._despeckle_slider.setObjectName("despeckleSlider")
        self._despeckle_slider.setRange(0, len(_DESPECKLE_LEVELS) - 1)
        self._despeckle_slider.setPageStep(1)
        self._despeckle_slider.setTickInterval(1)
        self._despeckle_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._despeckle_slider.setValue(_despeckle_step(DespeckleLevel.CAUTIOUS))
        despeckle_layout.addWidget(self._despeckle_slider)

        despeckle_layout.addWidget(
            _StopLabels(
                self._despeckle_slider,
                [
                    (
                        QtCore.QCoreApplication.translate(
                            "output::OptionsWidget", cast(str, label)
                        ),
                        f"despeckle{label}Label",
                    )
                    for _lvl, label in _DESPECKLE_LEVELS
                ],
            )
        )

        self._apply_buttons[ApplyGroup.DESPECKLE] = _apply_button(
            despeckle_layout, ApplyGroup.DESPECKLE
        )
        layout.addWidget(self._despeckle_group)

    def _build_dewarping_group(self, layout: QVBoxLayout) -> None:
        self._dewarping_group = CollapsibleGroupBox(
            QtCore.QCoreApplication.translate("output::OptionsWidget", "Dewarping"),
            self,
        )
        self._dewarping_group.setObjectName("dewarpingGroup")
        dewarp_layout = QVBoxLayout(self._dewarping_group)

        btn_layout = QHBoxLayout()
        self._dewarp_button_group = QButtonGroup(self)
        self._dewarp_buttons = {
            mode: _exclusive_button(
                self._dewarp_button_group,
                btn_layout,
                f"dewarp{label}Btn",
                QtCore.QCoreApplication.translate(
                    "output::OptionsWidget", cast(str, label)
                ),
                index,
            )
            for index, (mode, label) in enumerate(_DEWARPING_MODES)
        }
        self._dewarp_off_btn = self._dewarp_buttons[DewarpingMode.OFF]
        self._dewarp_auto_btn = self._dewarp_buttons[DewarpingMode.AUTO]
        self._dewarp_manual_btn = self._dewarp_buttons[DewarpingMode.MANUAL]
        self._dewarp_marginal_btn = self._dewarp_buttons[DewarpingMode.MARGINAL]
        self._dewarp_marginal_btn.setToolTip(
            QtCore.QCoreApplication.translate(
                "output::OptionsWidget", "Apply dewarping only to page margins"
            )
        )
        self._dewarp_off_btn.setChecked(True)
        dewarp_layout.addLayout(btn_layout)

        self._min_span_row = QWidget()
        self._min_span_row.setObjectName("dewarpMinSpanRow")
        span_layout = QHBoxLayout(self._min_span_row)
        span_layout.setContentsMargins(0, 0, 0, 0)
        span_layout.addWidget(
            QLabel(
                QtCore.QCoreApplication.translate(
                    "output::OptionsWidget", "Minimum span (%):"
                )
            )
        )
        self._min_span_spinbox = _spin_box("dewarpMinSpanSB", 0, 100)
        self._min_span_spinbox.setToolTip(
            QtCore.QCoreApplication.translate(
                "output::OptionsWidget",
                "Leave the page as it is when the curves found lie closer together "
                "than this share of the content height",
            )
        )
        span_layout.addWidget(self._min_span_spinbox)
        span_layout.addItem(_spacer())
        dewarp_layout.addWidget(self._min_span_row)

        self._dewarp_metrics_label = QLabel()
        self._dewarp_metrics_label.setObjectName("dewarpMetricsLabel")
        self._dewarp_metrics_label.setWordWrap(True)
        self._dewarp_metrics_label.setVisible(False)
        dewarp_layout.addWidget(self._dewarp_metrics_label)

        self._post_deskew_check = _check_box(
            "postDeskewCB",
            QtCore.QCoreApplication.translate("output::OptionsWidget", "Post-deskew"),
            QtCore.QCoreApplication.translate(
                "output::OptionsWidget", "Apply additional deskewing after dewarping"
            ),
        )
        self._post_deskew_check.setChecked(True)
        dewarp_layout.addWidget(self._post_deskew_check)

        self._apply_buttons[ApplyGroup.DEWARPING] = _apply_button(
            dewarp_layout, ApplyGroup.DEWARPING
        )
        layout.addWidget(self._dewarping_group)

    def _build_depth_group(self, layout: QVBoxLayout) -> None:
        self._depth_group = CollapsibleGroupBox(
            QtCore.QCoreApplication.translate(
                "output::OptionsWidget", "Depth perception"
            ),
            self,
        )
        self._depth_group.setObjectName("depthPerceptionGroup")
        depth_layout = QHBoxLayout(self._depth_group)
        self._depth_slider = QSlider()
        self._depth_slider.setObjectName("depthPerceptionSlider")
        self._depth_slider.setOrientation(Qt.Orientation.Horizontal)
        self._depth_slider.setRange(
            _DEPTH_PERCEPTION_SCALE, 3 * _DEPTH_PERCEPTION_SCALE
        )
        self._depth_slider.setValue(2 * _DEPTH_PERCEPTION_SCALE)
        depth_layout.addWidget(self._depth_slider)
        self._depth_label = QLabel("2.0")
        self._depth_label.setMinimumWidth(30)
        depth_layout.addWidget(self._depth_label)
        self._depth_group.setVisible(False)
        self._apply_buttons[ApplyGroup.DEPTH_PERCEPTION] = _apply_button(
            depth_layout, ApplyGroup.DEPTH_PERCEPTION
        )
        layout.addWidget(self._depth_group)

    def _connect_signals(self) -> None:
        """Connect widget signals to slots."""
        self._dpi_spinbox.valueChanged.connect(self._on_dpi_changed)

        self._mode_button_group.buttonClicked.connect(self._on_color_mode_changed)
        self._black_on_white_check.toggled.connect(self._on_black_on_white_changed)

        self._fill_margins_check.toggled.connect(
            lambda on: self._edit_color_common(fill_margins=on)
        )
        self._fill_offcut_check.toggled.connect(self._on_fill_offcut_changed)
        self._fill_outside_page_box_check.toggled.connect(
            self._on_fill_outside_page_box_changed
        )
        self._equalize_check.toggled.connect(self._on_equalize_changed)
        self._equalize_color_check.toggled.connect(
            lambda on: self._edit_color_common(normalize_illumination=on)
        )
        self._savitzky_golay_check.toggled.connect(
            lambda on: self._edit_binarization(savitzky_golay_smoothing=on)
        )
        self._morphological_check.toggled.connect(
            lambda on: self._edit_binarization(morphological_smoothing=on)
        )
        self._wiener_coef_spinbox.valueChanged.connect(
            lambda value: self._edit_color_common(delayed=True, wiener_coef=value)
        )
        self._wiener_window_spinbox.valueChanged.connect(
            lambda value: self._edit_color_common(
                delayed=True, wiener_window_size=value
            )
        )
        self._filling_color_combo.currentIndexChanged.connect(self._on_filling_changed)

        self._split_output_check.toggled.connect(
            lambda on: self._edit_splitting(split_output=on)
        )
        self._color_foreground_radio.toggled.connect(
            lambda on: self._edit_splitting(color_foreground=on)
        )
        self._original_background_check.toggled.connect(
            lambda on: self._edit_splitting(original_background=on)
        )

        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        self._threshold_slider.valueChanged.connect(self._on_threshold_changed)
        self._threshold_slider.sliderReleased.connect(self._commit_threshold)
        self._window_size_spinbox.valueChanged.connect(
            lambda value: self._edit_binarization(delayed=True, window_size=value)
        )
        for spinbox, field in (
            (self._edge_plus_coef_spinbox, "edge_div_kep"),
            (self._blur_div_coef_spinbox, "edge_div_kbd"),
            (self._sauvola_coef_spinbox, "sauvola_coef"),
            (self._wolf_coef_spinbox, "wolf_coef"),
            (self._wolf_lower_spinbox, "wolf_lower_bound"),
            (self._wolf_upper_spinbox, "wolf_upper_bound"),
            (self._bradley_coef_spinbox, "bradley_coef"),
            (self._edge_div_kep_spinbox, "edge_div_kep"),
            (self._edge_div_kbd_spinbox, "edge_div_kbd"),
        ):
            spinbox.valueChanged.connect(
                lambda value, field=field: self._edit_binarization(
                    delayed=True, **{field: value}
                )
            )

        self._color_segmentation_check.toggled.connect(
            lambda on: self._edit_segmenter(enabled=on)
        )
        self._reduce_noise_spinbox.valueChanged.connect(
            lambda value: self._edit_segmenter(delayed=True, noise_reduction=value)
        )
        self._red_adjustment_spinbox.valueChanged.connect(
            lambda value: self._edit_segmenter(
                delayed=True, red_threshold_adjustment=value
            )
        )
        self._green_adjustment_spinbox.valueChanged.connect(
            lambda value: self._edit_segmenter(
                delayed=True, green_threshold_adjustment=value
            )
        )
        self._blue_adjustment_spinbox.valueChanged.connect(
            lambda value: self._edit_segmenter(
                delayed=True, blue_threshold_adjustment=value
            )
        )

        self._posterize_check.toggled.connect(
            lambda on: self._edit_posterization(enabled=on)
        )
        self._posterize_level_spinbox.valueChanged.connect(
            lambda value: self._edit_posterization(delayed=True, level=value)
        )
        self._posterize_normalize_check.toggled.connect(
            lambda on: self._edit_posterization(normalization_enabled=on)
        )
        self._posterize_force_bw_check.toggled.connect(
            lambda on: self._edit_posterization(force_black_and_white=on)
        )

        self._picture_shape_combo.currentIndexChanged.connect(
            self._on_picture_shape_changed
        )
        self._picture_sensitivity_spinbox.valueChanged.connect(
            lambda value: self._edit_picture_shape(delayed=True, sensitivity=value)
        )
        self._higher_search_check.toggled.connect(
            lambda on: self._edit_picture_shape(higher_search_sensitivity=on)
        )

        self._despeckle_slider.valueChanged.connect(self._on_despeckle_changed)
        self._despeckle_slider.sliderReleased.connect(self._commit_despeckle)

        self._dewarp_button_group.buttonClicked.connect(self._on_dewarping_changed)
        self._min_span_spinbox.valueChanged.connect(self._on_min_span_changed)
        self._post_deskew_check.toggled.connect(self._on_post_deskew_changed)
        self._depth_slider.valueChanged.connect(self._on_depth_changed)
        self._depth_slider.sliderReleased.connect(self._commit_depth)

        for group, button in self._apply_buttons.items():
            button.clicked.connect(
                lambda _checked=False, group=group: self.apply_group_requested.emit(
                    group
                )
            )

    def set_current_page(self, page_id: PageId) -> None:
        """Set the current page being edited.

        A reload still waiting for the previous page is sent first.

        Args:
            page_id (PageId): The page ID to edit.
        """
        if self._pending_page_id not in (None, page_id):
            self.flush_pending_reload()
        if page_id != self._current_page_id:
            self._dewarp_metrics = None
        self._current_page_id = page_id
        self._update_ui()

    def flush_pending_reload(self) -> None:
        """Emit ``params_changed`` now for edits that are waiting for the delay."""
        self._reload_timer.stop()
        page_id, self._pending_page_id = self._pending_page_id, None
        if page_id is not None:
            self.params_changed.emit(page_id)

    def set_dewarping_tab_active(self, active: bool) -> None:
        """Switch between the layout for the dewarping tab and the other tabs.

        The dewarping tab hides the mode and despeckle controls and shows the
        depth perception slider.

        Args:
            active (bool): True while the dewarping tab of the image view shows.
        """
        self._dewarping_tab_active = active
        self._update_ui()

    def set_despeckle_tab_active(self, active: bool) -> None:
        """Tell the panel whether the despeckling tab of the image view is shown.

        That tab previews every level as it is chosen, so the page then renders
        again only after the delay.

        Args:
            active (bool): True while the despeckling tab shows.
        """
        self._despeckle_tab_active = active

    def set_dewarp_metrics(self, metrics: DewarpMetrics | None) -> None:
        """Show what automatic dewarping found when the current page rendered.

        Args:
            metrics (DewarpMetrics | None): The render's findings; None when the
                page was not dewarped automatically or has not rendered yet.
        """
        self._dewarp_metrics = metrics
        self._show_dewarp_metrics()

    def set_distortion_model(self, model: DistortionModelSpec) -> None:
        """Store a manually edited distortion model for the current page.

        Automatic and marginal dewarping become manual, since the model no
        longer is what those modes would detect.

        Args:
            model (DistortionModelSpec): Top and bottom page edges.
        """
        page_id = self._current_page_id
        if page_id is None:
            return
        params = self._filter.get_params(page_id).with_distortion_model(model)
        if params.dewarping.mode in (DewarpingMode.AUTO, DewarpingMode.MARGINAL):
            params = params.with_dewarping(
                params.dewarping.with_mode(DewarpingMode.MANUAL)
            )
        self._filter.set_params(page_id, params)
        self._update_ui()
        self.params_changed.emit(page_id)

    def _edit(
        self,
        change: Callable[[Params], Params],
        refresh: bool = False,
        delayed: bool = False,
    ) -> None:
        """Apply a change to the current page's parameters and report it.

        Args:
            change (Callable[[Params], Params]): Returns the changed parameters.
            refresh (bool): Update the controls afterwards, for changes that
                alter which controls apply.
            delayed (bool): Report after ``RELOAD_DELAY_MS`` without further
                edits instead of at once. The parameters are stored at once.
        """
        page_id = self._current_page_id
        if page_id is None or self._updating:
            return
        self._filter.set_params(page_id, change(self._filter.get_params(page_id)))
        if refresh:
            self._update_ui()
        if delayed:
            self._pending_page_id = page_id
            self._reload_timer.start()
        else:
            self._reload_timer.stop()
            self._pending_page_id = None
            self.params_changed.emit(page_id)

    def _edit_binarization(self, delayed: bool = False, **fields: object) -> None:
        self._edit(
            lambda p: p.with_binarization(p.binarization.model_copy(update=fields)),
            delayed=delayed,
        )

    def _edit_segmenter(self, delayed: bool = False, **fields: object) -> None:
        def change(params: Params) -> Params:
            segmenter = params.binarization.color_segmenter.model_copy(update=fields)
            return params.with_binarization(
                params.binarization.model_copy(update={"color_segmenter": segmenter})
            )

        self._edit(change, refresh=True, delayed=delayed)

    def _edit_color_common(
        self, refresh: bool = False, delayed: bool = False, **fields: object
    ) -> None:
        self._edit(
            lambda p: p.with_color_common(p.color_common.model_copy(update=fields)),
            refresh=refresh,
            delayed=delayed,
        )

    def _edit_posterization(self, delayed: bool = False, **fields: object) -> None:
        def change(params: Params) -> Params:
            posterization = params.color_common.posterization.model_copy(update=fields)
            return params.with_color_common(
                params.color_common.model_copy(update={"posterization": posterization})
            )

        self._edit(change, refresh=True, delayed=delayed)

    def _edit_picture_shape(
        self, delayed: bool = False, refresh: bool = False, **fields: object
    ) -> None:
        self._edit(
            lambda p: p.with_picture_shape(p.picture_shape.model_copy(update=fields)),
            refresh=refresh,
            delayed=delayed,
        )

    def _edit_splitting(self, **fields: object) -> None:
        self._edit(
            lambda p: p.with_splitting(p.splitting.model_copy(update=fields)),
            refresh=True,
        )

    def _update_ui(self) -> None:
        """Update the UI to reflect current page settings."""
        if self._current_page_id is None:
            return

        self._updating = True
        try:
            params = self._without_orphan_color_equalize(
                self._current_page_id, self._filter.get_params(self._current_page_id)
            )
            self._show_resolution_and_mode(params)
            self._show_common_options(params)
            self._show_splitting(params)
            self._show_binarization(params)
            self._show_segmentation_and_posterization(params)
            self._show_picture_shape(params)
            self._show_despeckle_and_dewarping(params)
        finally:
            self._updating = False

    def _without_orphan_color_equalize(self, page_id: PageId, params: Params) -> Params:
        """Store and return params where mixed mode never colour-equalizes alone."""
        if (
            params.color_mode != ColorMode.MIXED
            or params.binarization.normalize_illumination
            or not params.color_common.normalize_illumination
        ):
            return params
        color = params.color_common.model_copy(update={"normalize_illumination": False})
        params = params.with_color_common(color)
        self._filter.set_params(page_id, params)
        return params

    def _show_resolution_and_mode(self, params: Params) -> None:
        self._dpi_spinbox.setValue(params.output_dpi.horizontal)
        buttons = {
            ColorMode.BLACK_AND_WHITE: self._bw_btn,
            ColorMode.COLOR_GRAYSCALE: self._gray_btn,
            ColorMode.MIXED: self._mixed_btn,
        }
        buttons[params.color_mode].setChecked(True)
        self._black_on_white_check.setChecked(params.black_on_white)
        self._mode_group.setVisible(not self._dewarping_tab_active)

    def _show_common_options(self, params: Params) -> None:
        mode = params.color_mode
        common = params.color_common
        threshold = params.needs_binarization()

        self._fill_margins_check.setChecked(common.fill_margins)
        self._fill_offcut_check.setChecked(common.fill_offcut)
        self._fill_outside_page_box_check.setChecked(common.fill_outside_page_box)
        self._equalize_check.setChecked(params.binarization.normalize_illumination)
        self._equalize_check.setVisible(mode != ColorMode.COLOR_GRAYSCALE)
        self._equalize_color_check.setChecked(common.normalize_illumination)
        self._equalize_color_check.setVisible(mode != ColorMode.BLACK_AND_WHITE)
        self._equalize_color_check.setEnabled(
            mode == ColorMode.COLOR_GRAYSCALE
            or params.binarization.normalize_illumination
        )
        self._savitzky_golay_check.setChecked(
            params.binarization.savitzky_golay_smoothing
        )
        self._savitzky_golay_check.setVisible(threshold)
        self._morphological_check.setChecked(
            params.binarization.morphological_smoothing
        )
        self._morphological_check.setVisible(threshold)
        self._wiener_coef_spinbox.setValue(common.wiener_coef)
        self._wiener_window_spinbox.setValue(common.wiener_window_size)

        self._filling_group.setVisible(mode != ColorMode.BLACK_AND_WHITE)
        self._filling_color_combo.setCurrentIndex(
            self._filling_color_combo.findData(common.filling_color)
        )

    def _show_splitting(self, params: Params) -> None:
        splitting = params.splitting
        self._splitting_group.setVisible(params.color_mode == ColorMode.MIXED)
        self._split_output_check.setChecked(splitting.split_output)
        self._color_foreground_radio.setChecked(splitting.color_foreground)
        self._bw_foreground_radio.setChecked(not splitting.color_foreground)
        self._original_background_check.setChecked(splitting.original_background)
        self._bw_foreground_radio.setEnabled(splitting.split_output)
        self._color_foreground_radio.setEnabled(splitting.split_output)
        self._original_background_check.setEnabled(
            splitting.split_output and not splitting.color_foreground
        )

    def _show_binarization(self, params: Params) -> None:
        options = params.binarization
        self._binarization_group.setVisible(params.needs_binarization())
        self._method_combo.setCurrentIndex(self._method_combo.findData(options.method))
        self._threshold_slider.setValue(options.threshold_adjustment)
        self._threshold_label.setText(str(options.threshold_adjustment))
        self._window_size_spinbox.setValue(options.window_size)
        self._window_size_row.setVisible(options.method != BinarizationMethod.OTSU)
        method_index = self._method_page_index[options.method]
        self._method_options.setCurrentIndex(method_index)
        self._fit_method_options(method_index)
        self._sauvola_coef_spinbox.setValue(options.sauvola_coef)
        self._wolf_coef_spinbox.setValue(options.wolf_coef)
        self._wolf_lower_spinbox.setValue(options.wolf_lower_bound)
        self._wolf_upper_spinbox.setValue(options.wolf_upper_bound)
        self._bradley_coef_spinbox.setValue(options.bradley_coef)
        self._edge_div_kep_spinbox.setValue(options.edge_div_kep)
        self._edge_div_kbd_spinbox.setValue(options.edge_div_kbd)
        self._edge_plus_coef_spinbox.setValue(options.edge_div_kep)
        self._blur_div_coef_spinbox.setValue(options.edge_div_kbd)

    def _show_segmentation_and_posterization(self, params: Params) -> None:
        threshold = params.needs_binarization()
        segmenter = params.binarization.color_segmenter
        poster = params.color_common.posterization

        self._segmentation_group.setVisible(threshold)
        self._color_segmentation_check.setChecked(segmenter.enabled)
        self._segmenter_options.setEnabled(segmenter.enabled)
        self._reduce_noise_spinbox.setValue(segmenter.noise_reduction)
        self._red_adjustment_spinbox.setValue(segmenter.red_threshold_adjustment)
        self._green_adjustment_spinbox.setValue(segmenter.green_threshold_adjustment)
        self._blue_adjustment_spinbox.setValue(segmenter.blue_threshold_adjustment)

        posterize_available = segmenter.enabled or not threshold
        self._posterize_check.setEnabled(posterize_available)
        self._posterize_check.setChecked(poster.enabled)
        self._posterize_options.setEnabled(posterize_available and poster.enabled)
        self._posterize_level_spinbox.setValue(poster.level)
        self._posterize_normalize_check.setChecked(poster.normalization_enabled)
        self._posterize_force_bw_check.setChecked(poster.force_black_and_white)

    def _show_picture_shape(self, params: Params) -> None:
        options = params.picture_shape
        self._picture_shape_group.setVisible(params.color_mode == ColorMode.MIXED)
        self._picture_shape_combo.setCurrentIndex(
            self._picture_shape_combo.findData(options.shape)
        )
        self._picture_sensitivity_spinbox.setValue(options.sensitivity)
        self._picture_sensitivity_row.setVisible(
            options.shape == PictureShape.RECTANGULAR
        )
        self._higher_search_check.setChecked(options.higher_search_sensitivity)
        self._higher_search_check.setVisible(options.shape != PictureShape.OFF)

    def _show_despeckle_and_dewarping(self, params: Params) -> None:
        threshold = params.needs_binarization()
        self._despeckle_group.setVisible(threshold and not self._dewarping_tab_active)
        self._despeckle_slider.setValue(
            _despeckle_step(DespeckleLevel.nearest(params.despeckle_strength))
        )
        self._despeckle_slider.setToolTip(_despeckle_tooltip(params.despeckle_strength))

        dewarping = params.dewarping
        self._dewarp_buttons[dewarping.mode].setChecked(True)
        self._min_span_row.setVisible(dewarping.is_auto())
        self._min_span_spinbox.setValue(params.dewarp_min_span_percent)
        self._show_dewarp_metrics()
        self._post_deskew_check.setChecked(dewarping.post_deskew)
        self._post_deskew_check.setEnabled(dewarping.is_enabled())

        self._depth_group.setVisible(self._dewarping_tab_active)
        self._depth_slider.setValue(
            round(params.depth_perception * _DEPTH_PERCEPTION_SCALE)
        )
        self._depth_label.setText(f"{params.depth_perception:.1f}")

    def _on_dpi_changed(self, value: int) -> None:
        self._edit(lambda p: p.with_output_dpi(Dpi.uniform(value)))

    def _on_color_mode_changed(self, button: QPushButton) -> None:
        mode_map = {
            self._bw_btn: ColorMode.BLACK_AND_WHITE,
            self._gray_btn: ColorMode.COLOR_GRAYSCALE,
            self._mixed_btn: ColorMode.MIXED,
        }
        mode = mode_map.get(button, ColorMode.BLACK_AND_WHITE)
        self._edit(lambda p: p.with_color_mode(mode), refresh=True)

    def _on_picture_shape_changed(self, index: int) -> None:
        # Qt hands a str-valued enum member back as a plain str.
        value = self._picture_shape_combo.itemData(index)
        if value is not None:
            self._edit_picture_shape(refresh=True, shape=PictureShape(value))

    def _on_black_on_white_changed(self, checked: bool) -> None:
        self._edit(lambda p: p.with_black_on_white(checked))

    def _on_equalize_changed(self, checked: bool) -> None:
        def change(params: Params) -> Params:
            binarization = params.binarization.model_copy(
                update={"normalize_illumination": checked}
            )
            params = params.with_binarization(binarization)
            if params.color_mode == ColorMode.MIXED and not checked:
                color = params.color_common.model_copy(
                    update={"normalize_illumination": False}
                )
                params = params.with_color_common(color)
            return params

        self._edit(change, refresh=True)

    def _on_fill_offcut_changed(self, checked: bool) -> None:
        """Filling the offcut and filling outside the page box exclude each other."""
        self._edit_color_common(
            refresh=True,
            fill_offcut=checked,
            **({"fill_outside_page_box": False} if checked else {}),
        )

    def _on_fill_outside_page_box_changed(self, checked: bool) -> None:
        self._edit_color_common(
            refresh=True,
            fill_outside_page_box=checked,
            **({"fill_offcut": False} if checked else {}),
        )

    def _on_filling_changed(self, index: int) -> None:
        color = self._filling_color_combo.itemData(index)
        if color is not None:
            self._edit_color_common(filling_color=color)

    def _on_method_changed(self, index: int) -> None:
        method = self._method_combo.itemData(index)
        if method is None:
            return
        self._edit(lambda p: p.with_binarization_method(method), refresh=True)

    def _on_threshold_changed(self, value: int) -> None:
        self._threshold_label.setText(str(value))
        if not self._threshold_slider.isSliderDown():
            self._commit_threshold()

    def _commit_threshold(self) -> None:
        value = self._threshold_slider.value()
        self._edit_binarization(threshold_adjustment=value)

    def _on_despeckle_changed(self, _step: int) -> None:
        if not self._despeckle_slider.isSliderDown():
            self._commit_despeckle()

    def _commit_despeckle(self) -> None:
        if self._current_page_id is None or self._updating:
            return
        level = _DESPECKLE_LEVELS[self._despeckle_slider.value()][0]
        self.despeckle_level_changed.emit(level)
        self._edit(
            lambda p: p.with_despeckle_level(level), delayed=self._despeckle_tab_active
        )

    def _on_dewarping_changed(self, button: QPushButton) -> None:
        mode = next(
            (m for m, btn in self._dewarp_buttons.items() if btn is button),
            DewarpingMode.OFF,
        )
        self._edit(
            lambda p: p.with_dewarping(p.dewarping.with_mode(mode)), refresh=True
        )

    def _show_dewarp_metrics(self) -> None:
        metrics = self._dewarp_metrics
        label = self._dewarp_metrics_label
        label.setText("" if metrics is None else metrics.describe())
        label.setVisible(metrics is not None and self._dewarp_auto_btn.isChecked())

    def _on_min_span_changed(self, value: int) -> None:
        self._edit(lambda p: p.with_dewarp_min_span_percent(value), delayed=True)

    def _on_post_deskew_changed(self, checked: bool) -> None:
        self._edit(lambda p: p.with_dewarping(p.dewarping.with_post_deskew(checked)))

    def _on_depth_changed(self, value: int) -> None:
        self._depth_label.setText(f"{value / _DEPTH_PERCEPTION_SCALE:.1f}")
        if not self._depth_slider.isSliderDown():
            self._commit_depth()

    def _commit_depth(self) -> None:
        depth = self._depth_slider.value() / _DEPTH_PERCEPTION_SCALE
        self._edit(lambda p: p.with_depth_perception(depth))
