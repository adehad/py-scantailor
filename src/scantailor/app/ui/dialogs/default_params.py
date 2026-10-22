"""Default parameters dialog: the values new projects start from, with profiles."""

from pathlib import Path
from typing import cast

from loguru import logger
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Signal

from scantailor.app.ui import embed_ui, load_ui_widget
from scantailor.app.ui.icons import icon
from scantailor.app.ui.utils import get_cwidget
from scantailor.core.default_params import (
    DefaultParams,
    DeskewDefaults,
    FixOrientationDefaults,
    OutputDefaults,
    PageLayoutDefaults,
    PageSplitDefaults,
    SelectContentDefaults,
)
from scantailor.core.default_params_profiles import (
    CUSTOM_PROFILE,
    DEFAULT_PROFILE,
    SOURCE_PROFILE,
    ProfileError,
    ProfileStore,
    is_reserved_profile_name,
    validate_profile_name,
)
from scantailor.core.models import Dpi, Margins, OrthogonalRotation
from scantailor.core.settings import ApplicationSettings, Units
from scantailor.core.xml_profile_import import read_xml_profile
from scantailor.filters.deskew.params import (
    AutoManualMode,
    shown_deskew_angle,
    stored_deskew_angle,
)
from scantailor.filters.output.binarization import (
    BinarizationMethod,
    ColorSegmenterOptions,
)
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.color_options import (
    FillingColor,
    PosterizationOptions,
)
from scantailor.filters.output.despeckle import DespeckleLevel
from scantailor.filters.output.dewarping_options import DewarpingMode
from scantailor.filters.output.picture_shape import PictureShape, PictureShapeOptions
from scantailor.filters.output.splitting_options import SplittingOptions
from scantailor.filters.page_layout.alignment import (
    Alignment,
    HorizontalAlignment,
    VerticalAlignment,
)
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.select_content.content_box import PhysicalSize
from scantailor.filters.select_content.detection import (
    ContentDetectionMode,
    PageDetectionMode,
)

_UI_FOLDER = Path(__file__).parent
_MM_PER_UNIT = {Units.MILLIMETERS: 1.0, Units.CENTIMETERS: 10.0, Units.INCHES: 25.4}
_UNIT_LABEL = {Units.MILLIMETERS: "mm", Units.CENTIMETERS: "cm", Units.INCHES: "in"}
# Algorithm names are shown in the threshold combo box, but the combo loop only
# sees variables, so lupdate needs each literal spelled out here.
_METHOD_LABELS = {
    "Otsu": cast(str, QtCore.QT_TRANSLATE_NOOP("DefaultParamsDialog", "Otsu")),
    "Sauvola": cast(str, QtCore.QT_TRANSLATE_NOOP("DefaultParamsDialog", "Sauvola")),
    "Wolf": cast(str, QtCore.QT_TRANSLATE_NOOP("DefaultParamsDialog", "Wolf")),
    "Bradley": cast(str, QtCore.QT_TRANSLATE_NOOP("DefaultParamsDialog", "Bradley")),
    "Grad": cast(str, QtCore.QT_TRANSLATE_NOOP("DefaultParamsDialog", "Grad")),
    "EdgePlus": cast(str, QtCore.QT_TRANSLATE_NOOP("DefaultParamsDialog", "EdgePlus")),
    "BlurDiv": cast(str, QtCore.QT_TRANSLATE_NOOP("DefaultParamsDialog", "BlurDiv")),
    "EdgeDiv": cast(str, QtCore.QT_TRANSLATE_NOOP("DefaultParamsDialog", "EdgeDiv")),
}
_ROTATION_INDICATOR_SIZE = 32
_CUSTOM_DPI = 200
_DEPTH_STEPS_PER_UNIT = 10
_DESPECKLE_LEVELS = (
    DespeckleLevel.CAUTIOUS,
    DespeckleLevel.NORMAL,
    DespeckleLevel.AGGRESSIVE,
)
V, H = VerticalAlignment, HorizontalAlignment
_ALIGNMENT_BUTTONS = {
    "alignTopLeftBtn": (V.TOP, H.LEFT),
    "alignTopBtn": (V.TOP, H.HCENTER),
    "alignTopRightBtn": (V.TOP, H.RIGHT),
    "alignLeftBtn": (V.VCENTER, H.LEFT),
    "alignCenterBtn": (V.VCENTER, H.HCENTER),
    "alignRightBtn": (V.VCENTER, H.RIGHT),
    "alignBottomLeftBtn": (V.BOTTOM, H.LEFT),
    "alignBottomBtn": (V.BOTTOM, H.HCENTER),
    "alignBottomRightBtn": (V.BOTTOM, H.RIGHT),
}
# Index of the stacked page each binarization method shows: the threshold slider,
# the window size with a coefficient, and the Wolf bounds.
_METHOD_PAGES = {
    "Otsu": 0,
    "Sauvola": 1,
    "Wolf": 2,
    "Bradley": 1,
    "Grad": 2,
    "EdgePlus": 1,
    "BlurDiv": 1,
    "EdgeDiv": 1,
}
_METHODS = {
    "Otsu": BinarizationMethod.OTSU,
    "Sauvola": BinarizationMethod.SAUVOLA,
    "Wolf": BinarizationMethod.WOLF,
    "Bradley": BinarizationMethod.BRADLEY,
    "Grad": BinarizationMethod.GRAD,
    "EdgePlus": BinarizationMethod.EDGEPLUS,
    "BlurDiv": BinarizationMethod.BLURDIV,
    "EdgeDiv": BinarizationMethod.EDGEDIV,
}


class DefaultParamsDialog(QtWidgets.QDialog):
    """Dialog for the default parameters of every filter stage.

    A profile is Default, Source, a saved profile or Custom (the unsaved working
    values). Default and Source are shown read-only. Accepting stores the selected
    profile name in the application settings and the shown values in the profile
    store, which is what new projects start from.

    Signals:
        params_changed: Emitted after the selection was committed.
    """

    params_changed = Signal()

    def __init__(
        self,
        units: Units = Units.MILLIMETERS,
        settings: ApplicationSettings | None = None,
        store: ProfileStore | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the dialog.

        Args:
            units (Units): Units that margins and page size are shown in.
            settings (ApplicationSettings | None): Settings that remember the
                selected profile; defaults to the application instance.
            store (ProfileStore | None): Where profiles are kept; defaults to the
                application's profile folder.
            parent (QtWidgets.QWidget | None): Parent widget.
        """
        super().__init__(parent)
        self._units = Units.MILLIMETERS if units is Units.PIXELS else units
        self._settings = settings or ApplicationSettings.get_instance()
        self._store = store or ProfileStore()
        self._loaded = DefaultParams(units=units)
        self._rotation = OrthogonalRotation()
        self._loading = False
        self._custom_dpi = _CUSTOM_DPI
        self._custom_dpi_index = -1
        self._custom_profile_index = -1

        self.ui = load_ui_widget(_UI_FOLDER / "DefaultParamsDialog.ui", self)
        embed_ui(self, self.ui)
        self._populate_combo_boxes()
        self._update_rotation_indicator()
        self._connect_signals()
        self._select_initial_profile()

    # ------------------------------------------------------------------ setup

    def _w[T: QtWidgets.QWidget](self, kind: type[T], name: str) -> T:
        return get_cwidget(self.ui, kind, name)

    def _populate_combo_boxes(self) -> None:
        self._w(QtWidgets.QComboBox, "layoutModeCB").addItem(self.tr("Auto"), "auto")
        self._w(QtWidgets.QComboBox, "layoutModeCB").addItem(
            self.tr("Manual"), "manual"
        )
        color_modes = self._w(QtWidgets.QComboBox, "colorModeSelector")
        color_modes.addItem(self.tr("Black and White"), ColorMode.BLACK_AND_WHITE)
        color_modes.addItem(self.tr("Color / Grayscale"), ColorMode.COLOR_GRAYSCALE)
        color_modes.addItem(self.tr("Mixed"), ColorMode.MIXED)
        filling = self._w(QtWidgets.QComboBox, "fillingColorBox")
        filling.addItem(self.tr("Background"), FillingColor.BACKGROUND)
        filling.addItem(self.tr("White"), FillingColor.WHITE)
        filling.addItem(self.tr("Black"), FillingColor.BLACK)

        methods = self._w(QtWidgets.QComboBox, "thresholdMethodBox")
        for name in _METHODS:
            methods.addItem(self.tr(_METHOD_LABELS[name]), name)

        shapes = self._w(QtWidgets.QComboBox, "pictureShapeSelector")
        shapes.addItem(self.tr("Off"), PictureShape.OFF)
        shapes.addItem(self.tr("Free"), PictureShape.FREE)
        shapes.addItem(self.tr("Rectangular"), PictureShape.RECTANGULAR)

        dpi = self._w(QtWidgets.QComboBox, "dpiSelector")
        for value in (300, 400, 600, 1200):
            dpi.addItem(str(value), value)
        self._custom_dpi_index = dpi.count()
        dpi.addItem(self.tr("Custom"), self._custom_dpi)

        dewarping = self._w(QtWidgets.QComboBox, "dewarpingModeCB")
        dewarping.addItem(self.tr("Off"), DewarpingMode.OFF)
        dewarping.addItem(self.tr("Auto"), DewarpingMode.AUTO)
        dewarping.addItem(self.tr("Manual"), DewarpingMode.MANUAL)
        dewarping.addItem(self.tr("Marginal"), DewarpingMode.MARGINAL)

        self._w(QtWidgets.QSlider, "thresholdSlider").setRange(-100, 100)
        self._w(QtWidgets.QSlider, "despeckleSlider").setRange(
            1, len(_DESPECKLE_LEVELS)
        )
        self._w(QtWidgets.QSlider, "depthPerceptionSlider").setRange(
            _DEPTH_STEPS_PER_UNIT, 3 * _DEPTH_STEPS_PER_UNIT
        )
        for name in ("darkerThresholdLink", "lighterThresholdLink"):
            label = self._w(QtWidgets.QLabel, name)
            label.setText(f'<a href="#">{label.text()}</a>')

        self._alignment_group = QtWidgets.QButtonGroup(self)
        for name in _ALIGNMENT_BUTTONS:
            self._alignment_group.addButton(self._w(QtWidgets.QToolButton, name))

    def _connect_signals(self) -> None:
        w = self._w
        button_box = w(QtWidgets.QDialogButtonBox, "buttonBox")
        button_box.accepted.connect(self._commit)
        button_box.rejected.connect(self.reject)

        profile = w(QtWidgets.QComboBox, "profileCB")
        profile.currentIndexChanged.connect(self._on_profile_changed)
        w(QtWidgets.QPushButton, "profileSaveButton").clicked.connect(
            self._save_profile
        )
        w(QtWidgets.QPushButton, "profileDeleteButton").clicked.connect(
            self._delete_profile
        )
        w(QtWidgets.QPushButton, "profileImportButton").clicked.connect(
            self._choose_profile_to_import
        )

        w(QtWidgets.QToolButton, "rotateLeftBtn").clicked.connect(
            lambda: self._set_rotation(self._rotation.rotate_counter_clockwise())
        )
        w(QtWidgets.QToolButton, "rotateRightBtn").clicked.connect(
            lambda: self._set_rotation(self._rotation.rotate_clockwise())
        )
        w(QtWidgets.QPushButton, "resetBtn").clicked.connect(
            lambda: self._set_rotation(OrthogonalRotation())
        )

        w(QtWidgets.QComboBox, "layoutModeCB").currentIndexChanged.connect(
            self._on_layout_mode_changed
        )
        w(QtWidgets.QPushButton, "deskewAutoBtn").toggled.connect(
            lambda auto: w(QtWidgets.QDoubleSpinBox, "angleSpinBox").setEnabled(
                not auto
            )
        )

        for name in (
            "pageDetectAutoBtn",
            "pageDetectManualBtn",
            "pageDetectDisableBtn",
        ):
            w(QtWidgets.QPushButton, name).toggled.connect(self._update_page_detect)

        w(QtWidgets.QCheckBox, "autoMargins").toggled.connect(
            lambda auto: w(QtWidgets.QWidget, "marginsWidget").setEnabled(not auto)
        )
        for name in ("topBottomLink", "leftRightLink"):
            w(QtWidgets.QToolButton, name).clicked.connect(self._on_link_clicked)
        for name in ("topMarginSpinBox", "bottomMarginSpinBox"):
            w(QtWidgets.QDoubleSpinBox, name).valueChanged.connect(
                self._on_vertical_margin_changed
            )
        for name in ("leftMarginSpinBox", "rightMarginSpinBox"):
            w(QtWidgets.QDoubleSpinBox, name).valueChanged.connect(
                self._on_horizontal_margin_changed
            )
        w(QtWidgets.QCheckBox, "alignWithOthersCB").toggled.connect(
            self._update_alignment_enabled
        )
        for name in ("hAlignmentModeCB", "vAlignmentModeCB"):
            w(QtWidgets.QComboBox, name).currentIndexChanged.connect(
                self._on_alignment_mode_changed
            )

        for name, slot in (
            ("colorModeSelector", self._update_output_enabled),
            ("pictureShapeSelector", self._update_output_enabled),
            ("thresholdMethodBox", self._on_method_changed),
        ):
            w(QtWidgets.QComboBox, name).currentIndexChanged.connect(slot)
        for name in (
            "equalizeIlluminationCB",
            "splittingCB",
            "bwForegroundRB",
            "colorSegmentationCB",
            "posterizeCB",
            "despeckleCB",
        ):
            w(QtWidgets.QAbstractButton, name).toggled.connect(
                self._update_output_enabled
            )
        w(QtWidgets.QComboBox, "dpiSelector").activated.connect(self._on_dpi_selected)
        w(QtWidgets.QComboBox, "dpiSelector").editTextChanged.connect(
            self._on_dpi_text_changed
        )
        threshold = w(QtWidgets.QSlider, "thresholdSlider")
        threshold.valueChanged.connect(self._on_threshold_changed)
        w(QtWidgets.QToolButton, "neutralThresholdBtn").clicked.connect(
            lambda: threshold.setValue(0)
        )
        w(QtWidgets.QLabel, "lighterThresholdLink").linkActivated.connect(
            lambda _: threshold.setValue(threshold.value() - 1)
        )
        w(QtWidgets.QLabel, "darkerThresholdLink").linkActivated.connect(
            lambda _: threshold.setValue(threshold.value() + 1)
        )

    # --------------------------------------------------------------- profiles

    def _profile_box(self) -> QtWidgets.QComboBox:
        return self._w(QtWidgets.QComboBox, "profileCB")

    def _select_initial_profile(self) -> None:
        self._fill_profile_box()
        box = self._profile_box()
        index = box.findData(
            self._settings.current_profile,
            QtCore.Qt.ItemDataRole.UserRole,
            QtCore.Qt.MatchFlag.MatchFixedString,
        )
        box.blockSignals(True)
        box.setCurrentIndex(index if index != -1 else 0)
        box.blockSignals(False)
        self._on_profile_changed(box.currentIndex())

    def _fill_profile_box(self) -> None:
        box = self._profile_box()
        box.blockSignals(True)
        box.clear()
        box.addItem(self.tr("Default"), DEFAULT_PROFILE)
        box.addItem(self.tr("Source"), SOURCE_PROFILE)
        for name in self._store.names():
            box.addItem(name, name)
        self._custom_profile_index = box.count()
        box.addItem(self.tr("Custom"), CUSTOM_PROFILE)
        box.blockSignals(False)

    def _set_profile_controls(
        self, *, save: bool, delete: bool, editable: bool
    ) -> None:
        self._w(QtWidgets.QPushButton, "profileSaveButton").setEnabled(save)
        self._w(QtWidgets.QPushButton, "profileDeleteButton").setEnabled(delete)
        tabs = self._w(QtWidgets.QTabWidget, "tabWidget")
        for index in range(tabs.count()):
            if (page := tabs.widget(index)) is not None:
                page.setEnabled(editable)

    def _on_profile_changed(self, index: int) -> None:
        box = self._profile_box()
        name = box.itemData(index)
        is_custom = index == self._custom_profile_index
        box.setEditable(is_custom)
        if is_custom:
            box.setEditText(box.currentText())
            line_edit = box.lineEdit()
            if line_edit is not None:
                line_edit.selectAll()
            self._set_profile_controls(save=True, delete=False, editable=True)
            if self._settings.current_profile == CUSTOM_PROFILE:
                self._load_params(self._store.active_params(CUSTOM_PROFILE))
        elif name == DEFAULT_PROFILE:
            self._set_profile_controls(save=False, delete=False, editable=False)
            self._load_params(DefaultParams())
        elif name == SOURCE_PROFILE:
            self._set_profile_controls(save=False, delete=False, editable=False)
            self._load_params(DefaultParams.source())
        else:
            try:
                params = self._store.read(name)
            except ProfileError:
                logger.exception("Profile {} could not be loaded", name)
                QtWidgets.QMessageBox.critical(
                    self, self.tr("Error"), self.tr("Error loading the profile.")
                )
                box.removeItem(index)
                self._custom_profile_index -= 1
                box.setCurrentIndex(0)
                return
            self._set_profile_controls(save=True, delete=True, editable=True)
            self._load_params(params)

    def _save_profile(self) -> None:
        box = self._profile_box()
        name = box.currentText()
        if is_reserved_profile_name(name):
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Error"),
                self.tr(
                    "The name conflicts with a default profile name. "
                    "Please enter a different name."
                ),
            )
            return
        try:
            name = validate_profile_name(name)
            self._store.write(name, self._build_params())
        except ProfileError as error:
            logger.warning("{}", error)
            QtWidgets.QMessageBox.critical(
                self, self.tr("Error"), self.tr("Error saving the profile.")
            )
            return
        if box.currentIndex() == self._custom_profile_index:
            box.blockSignals(True)
            box.setEditable(False)
            index = box.currentIndex()
            existing = box.findData(
                name,
                QtCore.Qt.ItemDataRole.UserRole,
                QtCore.Qt.MatchFlag.MatchFixedString,
            )
            if existing == -1:
                box.setItemText(index, name)
                box.setItemData(index, name)
                self._custom_profile_index = box.count()
                box.addItem(self.tr("Custom"), CUSTOM_PROFILE)
                box.setCurrentIndex(index)
            else:
                box.setCurrentIndex(existing)
            box.blockSignals(False)
            self._set_profile_controls(save=True, delete=True, editable=True)

    def _delete_profile(self) -> None:
        box = self._profile_box()
        index = box.currentIndex()
        try:
            self._store.delete(box.currentText())
        except ProfileError:
            logger.exception("Profile could not be deleted")
            QtWidgets.QMessageBox.critical(
                self, self.tr("Error"), self.tr("Error deleting the profile.")
            )
            return
        box.blockSignals(True)
        box.removeItem(index)
        self._custom_profile_index -= 1
        box.setCurrentIndex(self._custom_profile_index)
        box.blockSignals(False)
        self._on_profile_changed(self._custom_profile_index)

    def _choose_profile_to_import(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Import Profile"),
            str(Path.home()),
            self.tr("ScanTailor Advanced profiles (*.stp *.xml)"),
        )
        if path:
            self.import_profile(Path(path))

    def import_profile(self, path: Path) -> bool:
        """Save a profile file of the C++ ScanTailor Advanced as a profile and show it.

        The profile is named after the file. A profile of the same name, whatever
        its case, is replaced only once the user agrees.

        Args:
            path (Path): The ``.stp`` or ``.xml`` file.

        Returns:
            bool: True if the profile was saved and is now selected.
        """
        try:
            params = read_xml_profile(path)
            name = validate_profile_name(path.stem)
        except ProfileError as error:
            logger.warning("{}", error)
            QtWidgets.QMessageBox.critical(
                self,
                self.tr("Error"),
                self.tr("Error importing the profile.") + f"\n{error}",
            )
            return False
        existing = {saved.casefold() for saved in self._store.names()}
        if name.casefold() in existing and not self._confirm_replace(name):
            return False
        try:
            self._store.write(name, params)
        except ProfileError as error:
            logger.warning("{}", error)
            QtWidgets.QMessageBox.critical(
                self, self.tr("Error"), self.tr("Error saving the profile.")
            )
            return False
        self._fill_profile_box()
        box = self._profile_box()
        index = box.findData(
            name, QtCore.Qt.ItemDataRole.UserRole, QtCore.Qt.MatchFlag.MatchFixedString
        )
        box.blockSignals(True)
        box.setCurrentIndex(index)
        box.blockSignals(False)
        self._on_profile_changed(index)
        return True

    def _confirm_replace(self, name: str) -> bool:
        answer = QtWidgets.QMessageBox.question(
            self,
            self.tr("Import Profile"),
            self.tr("A profile named {name} exists. Replace it?").format(name=name),
        )
        return answer == QtWidgets.QMessageBox.StandardButton.Yes

    def _commit(self) -> None:
        name = self._profile_box().currentData()
        params = self._build_params()
        if name not in (DEFAULT_PROFILE, SOURCE_PROFILE):
            try:
                self._store.set_active(params, name)
            except ProfileError:
                logger.exception("The default parameters could not be stored")
        self._settings.current_profile = name
        try:
            self._settings.save()
        except OSError:
            logger.exception("The settings could not be saved")
        self.params_changed.emit()
        self.accept()

    def get_params(self) -> DefaultParams:
        """Build the parameters shown in the form.

        Returns:
            DefaultParams: The values of the form, in millimetres.
        """
        return self._build_params()

    # ---------------------------------------------------------------- loading

    def _load_params(self, params: DefaultParams) -> None:
        """Show ``params`` in the form without triggering the edit handlers."""
        self._loaded = params
        self._loading = True
        try:
            self._load_units()
            self._load_fix_orientation(params.fix_orientation)
            self._load_page_split(params.page_split)
            self._load_deskew(params.deskew)
            self._load_select_content(params.select_content)
            self._load_page_layout(params.page_layout)
            self._load_output(params.output)
        finally:
            self._loading = False
        self._update_page_detect()
        self._update_alignment_enabled()
        self._update_output_enabled()
        self._on_method_changed()

    def _spin(self, name: str) -> QtWidgets.QDoubleSpinBox:
        return self._w(QtWidgets.QDoubleSpinBox, name)

    def _load_units(self) -> None:
        self._w(QtWidgets.QLabel, "unitsLabel").setText(_UNIT_LABEL[self._units])
        decimals, step = (1, 1.0) if self._units is Units.MILLIMETERS else (2, 0.01)
        for name in (
            "widthSpinBox",
            "heightSpinBox",
            "topMarginSpinBox",
            "bottomMarginSpinBox",
            "leftMarginSpinBox",
            "rightMarginSpinBox",
        ):
            spin = self._spin(name)
            spin.setDecimals(decimals)
            spin.setSingleStep(step)

    def _to_display(self, mm: float) -> float:
        return mm / _MM_PER_UNIT[self._units]

    def _from_display(
        self, spin: QtWidgets.QDoubleSpinBox, original_mm: float
    ) -> float:
        """Convert a spin box to millimetres, keeping the original if untouched.

        Showing a value rounds it, so converting the rounded number back would
        drift a profile that the user never edited.
        """
        if round(self._to_display(original_mm), spin.decimals()) == round(
            spin.value(), spin.decimals()
        ):
            return original_mm
        return spin.value() * _MM_PER_UNIT[self._units]

    def _check(self, name: str, checked: bool) -> None:
        self._w(QtWidgets.QAbstractButton, name).setChecked(checked)

    def _checked(self, name: str) -> bool:
        return self._w(QtWidgets.QAbstractButton, name).isChecked()

    def _uncheck_exclusive(self, button: QtWidgets.QAbstractButton) -> None:
        """Clear a group of auto-exclusive buttons, which Qt cannot do directly."""
        button.setAutoExclusive(False)
        button.setChecked(False)
        button.setAutoExclusive(True)

    def _load_fix_orientation(self, defaults: FixOrientationDefaults) -> None:
        self._rotation = defaults.image_rotation
        self._update_rotation_indicator()

    def _load_page_split(self, defaults: PageSplitDefaults) -> None:
        layout_type = defaults.layout_type
        auto = layout_type is LayoutType.AUTO_LAYOUT_TYPE
        modes = self._w(QtWidgets.QComboBox, "layoutModeCB")
        modes.setCurrentIndex(modes.findData("auto" if auto else "manual"))
        self._w(QtWidgets.QWidget, "pageLayoutGroup").setEnabled(not auto)
        buttons = {
            LayoutType.SINGLE_PAGE_UNCUT: "singlePageUncutBtn",
            LayoutType.PAGE_PLUS_OFFCUT: "pagePlusOffcutBtn",
            LayoutType.TWO_PAGES: "twoPagesBtn",
        }
        if auto:
            self._uncheck_exclusive(self._w(QtWidgets.QToolButton, "twoPagesBtn"))
        else:
            self._check(buttons[layout_type], True)

    def _load_deskew(self, defaults: DeskewDefaults) -> None:
        auto = defaults.mode is AutoManualMode.AUTO
        self._check("deskewAutoBtn" if auto else "deskewManualBtn", True)
        angle = self._spin("angleSpinBox")
        angle.setEnabled(not auto)
        angle.setValue(shown_deskew_angle(defaults.deskew_angle_deg))
        self._check("deskewAutoObliqueCB", defaults.auto_oblique)

    def _load_select_content(self, defaults: SelectContentDefaults) -> None:
        page_buttons = {
            PageDetectionMode.AUTO: "pageDetectAutoBtn",
            PageDetectionMode.MANUAL: "pageDetectManualBtn",
            PageDetectionMode.DISABLED: "pageDetectDisableBtn",
        }
        self._check(page_buttons[defaults.page_detection_mode], True)
        auto_content = defaults.content_detection_mode is ContentDetectionMode.AUTO
        self._check(
            "contentDetectAutoBtn" if auto_content else "contentDetectDisableBtn", True
        )
        self._spin("widthSpinBox").setValue(
            self._to_display(defaults.page_rect_size_mm.width_mm)
        )
        self._spin("heightSpinBox").setValue(
            self._to_display(defaults.page_rect_size_mm.height_mm)
        )
        self._check("fineTuneBtn", defaults.fine_tune_corners)

    def _load_page_layout(self, defaults: PageLayoutDefaults) -> None:
        self._check("autoMargins", defaults.auto_margins)
        self._w(QtWidgets.QWidget, "marginsWidget").setEnabled(
            not defaults.auto_margins
        )
        margins = defaults.hard_margins_mm
        self._spin("topMarginSpinBox").setValue(self._to_display(margins.top))
        self._spin("rightMarginSpinBox").setValue(self._to_display(margins.right))
        self._spin("bottomMarginSpinBox").setValue(self._to_display(margins.bottom))
        self._spin("leftMarginSpinBox").setValue(self._to_display(margins.left))
        self._check("topBottomLink", margins.top == margins.bottom)
        self._check("leftRightLink", margins.left == margins.right)

        alignment = defaults.alignment
        self._w(QtWidgets.QComboBox, "hAlignmentModeCB").setCurrentIndex(
            {H.HAUTO: 0, H.HORIGINAL: 2}.get(alignment.horizontal, 1)
        )
        self._w(QtWidgets.QComboBox, "vAlignmentModeCB").setCurrentIndex(
            {V.VAUTO: 0, V.VORIGINAL: 2}.get(alignment.vertical, 1)
        )
        self._check("alignWithOthersCB", not alignment.is_null)
        self._check_alignment_button(alignment)

    def _check_alignment_button(self, alignment: Alignment) -> None:
        for name, (vertical, horizontal) in _ALIGNMENT_BUTTONS.items():
            if alignment.is_auto_vertical():
                match = vertical is V.VCENTER and horizontal is alignment.horizontal
            elif alignment.is_auto_horizontal():
                match = horizontal is H.HCENTER and vertical is alignment.vertical
            else:
                match = (vertical, horizontal) == (
                    alignment.vertical,
                    alignment.horizontal,
                )
            if match:
                self._check(name, True)
                return

    def _load_output(self, defaults: OutputDefaults) -> None:
        self._select(
            self._w(QtWidgets.QComboBox, "colorModeSelector"), defaults.color_mode
        )
        common = defaults.color_common
        binarization = defaults.binarization
        self._check("fillMarginsCB", common.fill_margins)
        self._check("fillOffcutCB", common.fill_offcut)
        self._check("equalizeIlluminationCB", binarization.normalize_illumination)
        self._check("equalizeIlluminationColorCB", common.normalize_illumination)
        self._check("savitzkyGolaySmoothingCB", binarization.savitzky_golay_smoothing)
        self._check("morphologicalSmoothingCB", binarization.morphological_smoothing)
        self._select(
            self._w(QtWidgets.QComboBox, "fillingColorBox"), common.filling_color
        )

        segmenter = binarization.color_segmenter
        self._check("colorSegmentationCB", segmenter.enabled)
        self._w(QtWidgets.QSpinBox, "reduceNoiseSB").setValue(segmenter.noise_reduction)
        self._w(QtWidgets.QSpinBox, "redAdjustmentSB").setValue(
            segmenter.red_threshold_adjustment
        )
        self._w(QtWidgets.QSpinBox, "greenAdjustmentSB").setValue(
            segmenter.green_threshold_adjustment
        )
        self._w(QtWidgets.QSpinBox, "blueAdjustmentSB").setValue(
            segmenter.blue_threshold_adjustment
        )
        posterization = common.posterization
        self._check("posterizeCB", posterization.enabled)
        self._w(QtWidgets.QSpinBox, "posterizeLevelSB").setValue(posterization.level)
        self._check("posterizeNormalizationCB", posterization.normalization_enabled)
        self._check("posterizeForceBwCB", posterization.force_black_and_white)

        method_name = next(
            (
                name
                for name, method in _METHODS.items()
                if method is binarization.method
            ),
            "Otsu",
        )
        self._select(self._w(QtWidgets.QComboBox, "thresholdMethodBox"), method_name)
        self._w(QtWidgets.QSlider, "thresholdSlider").setValue(
            binarization.threshold_adjustment
        )
        self._w(QtWidgets.QSpinBox, "sauvolaWindowSize").setValue(
            binarization.window_size
        )
        self._w(QtWidgets.QSpinBox, "wolfWindowSize").setValue(binarization.window_size)
        self._spin("sauvolaCoef").setValue(binarization.sauvola_coef)
        self._spin("wolfCoef").setValue(binarization.wolf_coef)
        self._w(QtWidgets.QSpinBox, "lowerBound").setValue(
            binarization.wolf_lower_bound
        )
        self._w(QtWidgets.QSpinBox, "upperBound").setValue(
            binarization.wolf_upper_bound
        )

        shape = defaults.picture_shape
        self._select(self._w(QtWidgets.QComboBox, "pictureShapeSelector"), shape.shape)
        self._w(QtWidgets.QSpinBox, "pictureShapeSensitivitySB").setValue(
            shape.sensitivity
        )
        self._check("higherSearchSensitivityCB", shape.higher_search_sensitivity)

        dpi_box = self._w(QtWidgets.QComboBox, "dpiSelector")
        dpi = defaults.dpi.vertical
        index = dpi_box.findData(dpi)
        if index == -1 or index == self._custom_dpi_index:
            self._custom_dpi = dpi
            index = self._custom_dpi_index
        dpi_box.setCurrentIndex(index)
        self._on_dpi_selected(index)

        splitting = defaults.splitting
        self._check("splittingCB", splitting.split_output)
        self._check(
            "colorForegroundRB" if splitting.color_foreground else "bwForegroundRB",
            True,
        )
        self._check("originalBackgroundCB", splitting.original_background)

        level = defaults.despeckle_level
        self._check("despeckleCB", level is not DespeckleLevel.OFF)
        if level is not DespeckleLevel.OFF:
            self._w(QtWidgets.QSlider, "despeckleSlider").setValue(
                _DESPECKLE_LEVELS.index(level) + 1
            )

        self._select(
            self._w(QtWidgets.QComboBox, "dewarpingModeCB"), defaults.dewarping.mode
        )
        self._check("dewarpingPostDeskewCB", defaults.dewarping.post_deskew)
        self._w(QtWidgets.QSpinBox, "dewarpingMinSpanSB").setValue(
            defaults.dewarp_min_span_percent
        )
        self._w(QtWidgets.QSlider, "depthPerceptionSlider").setValue(
            round(defaults.depth_perception * _DEPTH_STEPS_PER_UNIT)
        )

    @staticmethod
    def _select(box: QtWidgets.QComboBox, data: object) -> None:
        index = box.findData(data)
        if index != -1:
            box.setCurrentIndex(index)

    # --------------------------------------------------------------- building

    def _build_params(self) -> DefaultParams:
        """Read the form into a profile, keeping what the form cannot show."""
        base = self._loaded
        return DefaultParams(
            fix_orientation=FixOrientationDefaults(image_rotation=self._rotation),
            deskew=DeskewDefaults(
                deskew_angle_deg=stored_deskew_angle(
                    self._spin("angleSpinBox").value()
                ),
                mode=AutoManualMode.AUTO
                if self._checked("deskewAutoBtn")
                else AutoManualMode.MANUAL,
                auto_oblique=self._checked("deskewAutoObliqueCB"),
            ),
            page_split=PageSplitDefaults(layout_type=self._build_layout_type()),
            select_content=self._build_select_content(base.select_content),
            page_layout=self._build_page_layout(base.page_layout),
            output=self._build_output(base.output),
            units=self._units,
        )

    def _build_layout_type(self) -> LayoutType:
        if self._w(QtWidgets.QComboBox, "layoutModeCB").currentData() == "auto":
            return LayoutType.AUTO_LAYOUT_TYPE
        if self._checked("singlePageUncutBtn"):
            return LayoutType.SINGLE_PAGE_UNCUT
        if self._checked("pagePlusOffcutBtn"):
            return LayoutType.PAGE_PLUS_OFFCUT
        return LayoutType.TWO_PAGES

    def _build_select_content(
        self, base: SelectContentDefaults
    ) -> SelectContentDefaults:
        if self._checked("pageDetectDisableBtn"):
            page_mode = PageDetectionMode.DISABLED
        elif self._checked("pageDetectManualBtn"):
            page_mode = PageDetectionMode.MANUAL
        else:
            page_mode = PageDetectionMode.AUTO
        content_mode = (
            ContentDetectionMode.DISABLED
            if self._checked("contentDetectDisableBtn")
            else ContentDetectionMode.AUTO
        )
        return SelectContentDefaults(
            page_rect_size_mm=PhysicalSize(
                width_mm=self._from_display(
                    self._spin("widthSpinBox"), base.page_rect_size_mm.width_mm
                ),
                height_mm=self._from_display(
                    self._spin("heightSpinBox"), base.page_rect_size_mm.height_mm
                ),
            ),
            content_detection_mode=content_mode,
            page_detection_mode=page_mode,
            fine_tune_corners=self._checked("fineTuneBtn"),
        )

    def _checked_alignment(self) -> tuple[VerticalAlignment, HorizontalAlignment]:
        button = self._alignment_group.checkedButton()
        return _ALIGNMENT_BUTTONS.get(
            button.objectName() if button else "", (V.VCENTER, H.HCENTER)
        )

    def _build_alignment(self) -> Alignment:
        vertical, horizontal = self._checked_alignment()
        h_mode = self._w(QtWidgets.QComboBox, "hAlignmentModeCB").currentIndex()
        v_mode = self._w(QtWidgets.QComboBox, "vAlignmentModeCB").currentIndex()
        return Alignment(
            horizontal={0: H.HAUTO, 2: H.HORIGINAL}.get(h_mode, horizontal),
            vertical={0: V.VAUTO, 2: V.VORIGINAL}.get(v_mode, vertical),
            is_null=not self._checked("alignWithOthersCB"),
        )

    def _build_page_layout(self, base: PageLayoutDefaults) -> PageLayoutDefaults:
        old = base.hard_margins_mm
        return PageLayoutDefaults(
            hard_margins_mm=Margins(
                left=self._from_display(self._spin("leftMarginSpinBox"), old.left),
                right=self._from_display(self._spin("rightMarginSpinBox"), old.right),
                top=self._from_display(self._spin("topMarginSpinBox"), old.top),
                bottom=self._from_display(
                    self._spin("bottomMarginSpinBox"), old.bottom
                ),
            ),
            alignment=self._build_alignment(),
            auto_margins=self._checked("autoMargins"),
        )

    def _build_output(self, base: OutputDefaults) -> OutputDefaults:
        method_name = self._w(QtWidgets.QComboBox, "thresholdMethodBox").currentData()
        method = _METHODS[method_name] or base.binarization.method
        window_widget = (
            "wolfWindowSize"
            if method in (BinarizationMethod.WOLF, BinarizationMethod.GRAD)
            else "sauvolaWindowSize"
        )
        binarization = base.binarization.model_copy(
            update={
                "method": method,
                "threshold_adjustment": self._w(
                    QtWidgets.QSlider, "thresholdSlider"
                ).value(),
                "window_size": self._w(QtWidgets.QSpinBox, window_widget).value(),
                "sauvola_coef": self._spin("sauvolaCoef").value(),
                "wolf_coef": self._spin("wolfCoef").value(),
                "wolf_lower_bound": self._w(QtWidgets.QSpinBox, "lowerBound").value(),
                "wolf_upper_bound": self._w(QtWidgets.QSpinBox, "upperBound").value(),
                "normalize_illumination": self._checked("equalizeIlluminationCB"),
                "savitzky_golay_smoothing": self._checked("savitzkyGolaySmoothingCB"),
                "morphological_smoothing": self._checked("morphologicalSmoothingCB"),
                "color_segmenter": ColorSegmenterOptions(
                    enabled=self._checked("colorSegmentationCB"),
                    noise_reduction=self._w(
                        QtWidgets.QSpinBox, "reduceNoiseSB"
                    ).value(),
                    red_threshold_adjustment=self._w(
                        QtWidgets.QSpinBox, "redAdjustmentSB"
                    ).value(),
                    green_threshold_adjustment=self._w(
                        QtWidgets.QSpinBox, "greenAdjustmentSB"
                    ).value(),
                    blue_threshold_adjustment=self._w(
                        QtWidgets.QSpinBox, "blueAdjustmentSB"
                    ).value(),
                ),
            }
        )
        common = base.color_common.model_copy(
            update={
                "fill_offcut": self._checked("fillOffcutCB"),
                "fill_margins": self._checked("fillMarginsCB"),
                "normalize_illumination": self._checked("equalizeIlluminationColorCB"),
                "filling_color": self._w(
                    QtWidgets.QComboBox, "fillingColorBox"
                ).currentData(),
                "posterization": PosterizationOptions(
                    enabled=self._checked("posterizeCB"),
                    level=self._w(QtWidgets.QSpinBox, "posterizeLevelSB").value(),
                    normalization_enabled=self._checked("posterizeNormalizationCB"),
                    force_black_and_white=self._checked("posterizeForceBwCB"),
                ),
            }
        )
        dpi = self._current_dpi()
        despeckle = DespeckleLevel.OFF
        if self._checked("despeckleCB"):
            despeckle = _DESPECKLE_LEVELS[
                self._w(QtWidgets.QSlider, "despeckleSlider").value() - 1
            ]
        return OutputDefaults(
            dpi=Dpi.uniform(dpi),
            color_mode=self._w(QtWidgets.QComboBox, "colorModeSelector").currentData(),
            binarization=binarization,
            despeckle_level=despeckle,
            black_on_white=base.black_on_white,
            color_common=common,
            picture_shape=PictureShapeOptions(
                shape=PictureShape(
                    self._w(QtWidgets.QComboBox, "pictureShapeSelector").currentData()
                ),
                sensitivity=self._w(
                    QtWidgets.QSpinBox, "pictureShapeSensitivitySB"
                ).value(),
                higher_search_sensitivity=self._checked("higherSearchSensitivityCB"),
            ),
            splitting=SplittingOptions(
                split_output=self._checked("splittingCB"),
                original_background=self._checked("originalBackgroundCB"),
                color_foreground=self._checked("colorForegroundRB"),
            ),
            dewarping=base.dewarping.model_copy(
                update={
                    "mode": DewarpingMode(
                        self._w(QtWidgets.QComboBox, "dewarpingModeCB").currentData()
                    ),
                    "post_deskew": self._checked("dewarpingPostDeskewCB"),
                }
            ),
            dewarp_min_span_percent=self._w(
                QtWidgets.QSpinBox, "dewarpingMinSpanSB"
            ).value(),
            depth_perception=self._w(QtWidgets.QSlider, "depthPerceptionSlider").value()
            / _DEPTH_STEPS_PER_UNIT,
        )

    def _current_dpi(self) -> int:
        box = self._w(QtWidgets.QComboBox, "dpiSelector")
        if box.currentIndex() == self._custom_dpi_index:
            return max(1, self._custom_dpi)
        return int(box.currentData())

    # --------------------------------------------------------------- handlers

    def _set_rotation(self, rotation: OrthogonalRotation) -> None:
        self._rotation = rotation
        self._update_rotation_indicator()

    def _update_rotation_indicator(self) -> None:
        """Show an up arrow turned by the chosen rotation."""
        label = self._w(QtWidgets.QLabel, "rotationIndicator")
        ratio = label.devicePixelRatioF()
        size = QtCore.QSize(_ROTATION_INDICATOR_SIZE, _ROTATION_INDICATOR_SIZE)
        pixmap = icon("arrow-up").pixmap(size, ratio)
        transform = QtGui.QTransform().rotate(self._rotation.degrees)
        label.setPixmap(
            pixmap.transformed(
                transform, QtCore.Qt.TransformationMode.SmoothTransformation
            )
        )

    def changeEvent(self, event: QtCore.QEvent) -> None:
        """Redraw the rotation indicator when the palette changes.

        Args:
            event (QtCore.QEvent): The event.
        """
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.Type.PaletteChange and hasattr(
            self, "_rotation"
        ):
            self._update_rotation_indicator()

    def _on_layout_mode_changed(self, index: int) -> None:
        if self._loading:
            return
        manual = (
            self._w(QtWidgets.QComboBox, "layoutModeCB").itemData(index) == "manual"
        )
        if manual:
            self._check("singlePageUncutBtn", True)
        else:
            self._uncheck_exclusive(self._w(QtWidgets.QToolButton, "twoPagesBtn"))
        self._w(QtWidgets.QWidget, "pageLayoutGroup").setEnabled(manual)

    def _update_page_detect(self) -> None:
        disabled = self._checked("pageDetectDisableBtn")
        manual = self._checked("pageDetectManualBtn")
        self._w(QtWidgets.QWidget, "pageDetectOptions").setEnabled(not disabled)
        self._w(QtWidgets.QWidget, "fineTuneBtn").setEnabled(
            not disabled and not manual
        )
        self._w(QtWidgets.QWidget, "dimensionsWidget").setEnabled(manual)

    def _on_link_clicked(self) -> None:
        for link, first, second in (
            ("topBottomLink", "topMarginSpinBox", "bottomMarginSpinBox"),
            ("leftRightLink", "leftMarginSpinBox", "rightMarginSpinBox"),
        ):
            if self.sender() is self._w(QtWidgets.QToolButton, link) and self._checked(
                link
            ):
                smaller = min(self._spin(first).value(), self._spin(second).value())
                self._spin(first).setValue(smaller)
                self._spin(second).setValue(smaller)

    def _mirror_margins(self, link: str, first: str, second: str, value: float) -> None:
        if self._loading or not self._checked(link):
            return
        for name in (first, second):
            spin = self._spin(name)
            if spin.value() != value:
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)

    def _on_vertical_margin_changed(self, value: float) -> None:
        self._mirror_margins(
            "topBottomLink", "topMarginSpinBox", "bottomMarginSpinBox", value
        )

    def _on_horizontal_margin_changed(self, value: float) -> None:
        self._mirror_margins(
            "leftRightLink", "leftMarginSpinBox", "rightMarginSpinBox", value
        )

    def _on_alignment_mode_changed(self, _index: int) -> None:
        if self._loading:
            return
        self._update_alignment_enabled()
        self._keep_auto_axis_button()

    def _keep_auto_axis_button(self) -> None:
        """Move the checked button onto the axis that is still chosen by hand."""
        h_auto = self._w(QtWidgets.QComboBox, "hAlignmentModeCB").currentIndex() != 1
        v_auto = self._w(QtWidgets.QComboBox, "vAlignmentModeCB").currentIndex() != 1
        vertical, horizontal = self._checked_alignment()
        if v_auto and not h_auto:
            target = {H.LEFT: "alignLeftBtn", H.RIGHT: "alignRightBtn"}.get(
                horizontal, "alignCenterBtn"
            )
        elif h_auto and not v_auto:
            target = {V.TOP: "alignTopBtn", V.BOTTOM: "alignBottomBtn"}.get(
                vertical, "alignCenterBtn"
            )
        else:
            return
        self._check(target, True)

    def _update_alignment_enabled(self) -> None:
        aligned = self._checked("alignWithOthersCB")
        self._w(QtWidgets.QComboBox, "hAlignmentModeCB").setEnabled(aligned)
        self._w(QtWidgets.QComboBox, "vAlignmentModeCB").setEnabled(aligned)
        horizontal = (
            aligned
            and self._w(QtWidgets.QComboBox, "hAlignmentModeCB").currentIndex() == 1
        )
        vertical = (
            aligned
            and self._w(QtWidgets.QComboBox, "vAlignmentModeCB").currentIndex() == 1
        )
        enabled = {
            "alignTopLeftBtn": horizontal and vertical,
            "alignTopBtn": vertical,
            "alignTopRightBtn": horizontal and vertical,
            "alignLeftBtn": horizontal,
            "alignCenterBtn": horizontal or vertical,
            "alignRightBtn": horizontal,
            "alignBottomLeftBtn": horizontal and vertical,
            "alignBottomBtn": vertical,
            "alignBottomRightBtn": horizontal and vertical,
        }
        for name, state in enabled.items():
            self._w(QtWidgets.QToolButton, name).setEnabled(state)

    def _on_method_changed(self, _index: int = 0) -> None:
        name = self._w(QtWidgets.QComboBox, "thresholdMethodBox").currentData()
        self._w(QtWidgets.QStackedWidget, "binarizationOptions").setCurrentIndex(
            _METHOD_PAGES.get(name, 0)
        )

    def _on_threshold_changed(self, value: int) -> None:
        self._w(QtWidgets.QLabel, "thresholLabel").setText(str(value))
        self._w(QtWidgets.QSlider, "thresholdSlider").setToolTip(str(value))

    def _on_dpi_selected(self, index: int) -> None:
        box = self._w(QtWidgets.QComboBox, "dpiSelector")
        custom = index == self._custom_dpi_index
        box.setEditable(custom)
        if custom:
            box.setEditText(str(self._custom_dpi))
            box.setValidator(QtGui.QIntValidator(1, 9999, box))
            line_edit = box.lineEdit()
            if line_edit is not None:
                line_edit.selectAll()

    def _on_dpi_text_changed(self, text: str) -> None:
        box = self._w(QtWidgets.QComboBox, "dpiSelector")
        if box.currentIndex() == self._custom_dpi_index and text.isdigit():
            self._custom_dpi = int(text)

    def _update_output_enabled(self, *_ignored: object) -> None:
        """Enable the output controls that apply to the chosen colour mode."""
        w = self._w
        mode = w(QtWidgets.QComboBox, "colorModeSelector").currentData()
        mixed = mode is ColorMode.MIXED
        binary_path = mode is not ColorMode.COLOR_GRAYSCALE
        w(QtWidgets.QWidget, "thresholdOptions").setEnabled(binary_path)
        w(QtWidgets.QWidget, "despecklePanel").setEnabled(binary_path)
        w(QtWidgets.QSlider, "despeckleSlider").setEnabled(
            binary_path and self._checked("despeckleCB")
        )
        w(QtWidgets.QWidget, "pictureShapeOptions").setEnabled(mixed)
        w(QtWidgets.QWidget, "splittingOptions").setEnabled(mixed)
        w(QtWidgets.QWidget, "fillingOptions").setEnabled(
            mode is not ColorMode.BLACK_AND_WHITE
        )

        w(QtWidgets.QWidget, "equalizeIlluminationCB").setEnabled(
            mode is not ColorMode.COLOR_GRAYSCALE
        )
        color_equalize = w(QtWidgets.QWidget, "equalizeIlluminationColorCB")
        color_equalize.setEnabled(mode is not ColorMode.BLACK_AND_WHITE)
        if mixed:
            equalize_on = self._checked("equalizeIlluminationCB")
            if self._checked("equalizeIlluminationColorCB") and not equalize_on:
                self._check("equalizeIlluminationColorCB", False)
            color_equalize.setEnabled(equalize_on)
        for name in ("savitzkyGolaySmoothingCB", "morphologicalSmoothingCB"):
            w(QtWidgets.QWidget, name).setEnabled(mode is not ColorMode.COLOR_GRAYSCALE)

        segmentation = self._checked("colorSegmentationCB")
        w(QtWidgets.QWidget, "colorSegmentationCB").setEnabled(binary_path)
        w(QtWidgets.QWidget, "segmenterOptionsWidget").setEnabled(
            binary_path and segmentation
        )
        if binary_path:
            w(QtWidgets.QWidget, "posterizeCB").setEnabled(segmentation)
            w(QtWidgets.QWidget, "posterizeOptionsWidget").setEnabled(
                segmentation and self._checked("posterizeCB")
            )
        else:
            w(QtWidgets.QWidget, "posterizeCB").setEnabled(True)
            w(QtWidgets.QWidget, "posterizeOptionsWidget").setEnabled(
                self._checked("posterizeCB")
            )

        shape = PictureShape(
            w(QtWidgets.QComboBox, "pictureShapeSelector").currentData()
        )
        w(QtWidgets.QWidget, "pictureShapeSensitivityOptions").setEnabled(
            shape is PictureShape.RECTANGULAR
        )
        w(QtWidgets.QWidget, "higherSearchSensitivityCB").setEnabled(
            shape is not PictureShape.OFF
        )

        splitting = self._checked("splittingCB")
        w(QtWidgets.QWidget, "bwForegroundRB").setEnabled(splitting)
        w(QtWidgets.QWidget, "colorForegroundRB").setEnabled(splitting)
        w(QtWidgets.QWidget, "originalBackgroundCB").setEnabled(
            splitting and self._checked("bwForegroundRB")
        )
