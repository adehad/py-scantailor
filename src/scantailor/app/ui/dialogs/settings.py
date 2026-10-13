"""Settings dialog for application preferences."""

from pathlib import Path

from loguru import logger
from PySide6 import QtCore, QtWidgets

from scantailor.app.ui import embed_ui, load_ui_widget
from scantailor.app.ui.icons import icon
from scantailor.app.ui.theme import AUTO, SETTINGS_KEY, apply_theme
from scantailor.app.ui.units_provider import UnitsProvider
from scantailor.app.ui.utils import get_cwidget
from scantailor.core.settings import (
    ApplicationSettings,
    DeviationSettings,
    TiffCompression,
    Units,
    ZoneCreationMode,
)

_UI_FOLDER = Path(__file__).parent
_TRANSLATIONS_FOLDER = _UI_FOLDER.parents[2] / "translations"
_SOURCE_LANGUAGE = "en"
_OPENGL_UNAVAILABLE = "Your hardware / driver don't provide the necessary features"
# C++ derives the thumbnail height from the width with this aspect ratio.
_THUMBNAIL_ASPECT = 16.0 / 25.0
# Options that nothing in the application reads yet; they stay visible but cannot
# be edited, and keep the stored value.
_UNWIRED_WIDGETS = (
    "languageBox",
    "blackOnWhiteDetectionCB",
    "blackOnWhiteDetectionAtOutputCB",
    "highlightDeviationCB",
    "deskewDeviationCoefSB",
    "deskewDeviationThresholdSB",
    "selectContentDeviationCoefSB",
    "selectContentDeviationThresholdSB",
    "marginsDeviationCoefSB",
    "marginsDeviationThresholdSB",
)


def available_languages() -> list[str]:
    """List the locale codes the application can be shown in.

    Returns:
        list[str]: The source language first, then one code per bundled translation.
    """
    codes = {
        path.stem.removeprefix("scantailor_")
        for pattern in ("scantailor_*.ts", "scantailor_*.qm")
        for path in _TRANSLATIONS_FOLDER.glob(pattern)
    }
    codes.discard("untranslated")
    return [_SOURCE_LANGUAGE, *sorted(codes - {_SOURCE_LANGUAGE})]


class SettingsDialog(QtWidgets.QDialog):
    """Settings dialog for configuring application preferences.

    Choosing a colour scheme shows it at once; accepting the dialog writes the
    values to ``settings``, saves them, stores the appearance under the key the
    theme reads and announces the unit to ``units_provider``. Rejecting the dialog
    brings back the scheme that was in use when it opened.

    Signals:
        settings_changed: Emitted after the changes were committed.
    """

    settings_changed = QtCore.Signal()

    def __init__(
        self,
        settings: ApplicationSettings,
        parent: QtWidgets.QWidget | None = None,
        units_provider: UnitsProvider | None = None,
    ) -> None:
        """Initialize the settings dialog.

        Args:
            settings (ApplicationSettings): Application settings to edit.
            parent (QtWidgets.QWidget | None): Parent widget.
            units_provider (UnitsProvider | None): Told the chosen unit when the
                dialog is accepted; the application-wide one when None.
        """
        super().__init__(parent)
        self._settings = settings
        self._units_provider = units_provider or UnitsProvider.get_instance()

        self.ui = load_ui_widget(_UI_FOLDER / "SettingsDialog.ui", self)
        embed_ui(self, self.ui)
        self._setup_widgets()
        self._populate_combo_boxes()
        self._load_settings()
        self._disable_unwired_options()
        self._connect_signals()

    def _setup_widgets(self) -> None:
        """Look up the widgets of the form."""
        ui = self.ui
        check, combo, spin = (
            QtWidgets.QCheckBox,
            QtWidgets.QComboBox,
            QtWidgets.QSpinBox,
        )
        double = QtWidgets.QDoubleSpinBox

        self._enable_opengl_cb = get_cwidget(ui, check, "enableOpenglCb")
        self._opengl_device_label = get_cwidget(
            ui, QtWidgets.QLabel, "openglDeviceLabel"
        )
        self._auto_save_cb = get_cwidget(ui, check, "autoSaveProjectCB")
        self._color_scheme_box = get_cwidget(ui, combo, "colorSchemeBox")
        self._language_box = get_cwidget(ui, combo, "languageBox")
        self._units_box = get_cwidget(ui, combo, "unitsBox")
        self._zone_mode_box = get_cwidget(ui, combo, "zoneCreationModeBox")

        self._thumbnail_quality_sb = get_cwidget(ui, spin, "thumbnailQualitySB")
        self._thumbnail_size_sb = get_cwidget(ui, spin, "thumbnailSizeSB")
        self._single_column_cb = get_cwidget(ui, check, "singleColumnThumbnailsCB")
        self._cancel_selection_cb = get_cwidget(
            ui, check, "cancelingSelectionQuestionCB"
        )

        self._tiff_bw_box = get_cwidget(ui, combo, "tiffCompressionBWBox")
        self._tiff_color_box = get_cwidget(ui, combo, "tiffCompressionColorBox")

        self._black_on_white_cb = get_cwidget(ui, check, "blackOnWhiteDetectionCB")
        self._black_on_white_output_cb = get_cwidget(
            ui, check, "blackOnWhiteDetectionAtOutputCB"
        )

        self._highlight_deviation_cb = get_cwidget(ui, check, "highlightDeviationCB")
        self._deskew_deviation_coef = get_cwidget(ui, double, "deskewDeviationCoefSB")
        self._deskew_deviation_thresh = get_cwidget(
            ui, double, "deskewDeviationThresholdSB"
        )
        self._select_content_deviation_coef = get_cwidget(
            ui, double, "selectContentDeviationCoefSB"
        )
        self._select_content_deviation_thresh = get_cwidget(
            ui, double, "selectContentDeviationThresholdSB"
        )
        self._margins_deviation_coef = get_cwidget(ui, double, "marginsDeviationCoefSB")
        self._margins_deviation_thresh = get_cwidget(
            ui, double, "marginsDeviationThresholdSB"
        )

        self._button_box = get_cwidget(ui, QtWidgets.QDialogButtonBox, "buttonBox")

    def _populate_combo_boxes(self) -> None:
        """Fill the combo boxes; the item data carries the stored value."""
        for text, mode in (("Auto", AUTO), ("Light", "light"), ("Dark", "dark")):
            self._color_scheme_box.addItem(text, mode)

        for text, units in (
            ("Pixels", Units.PIXELS),
            ("Millimetres", Units.MILLIMETERS),
            ("Centimetres", Units.CENTIMETERS),
            ("Inches", Units.INCHES),
        ):
            self._units_box.addItem(text, units)

        for text, zone_mode, glyph in (
            ("Polygonal", ZoneCreationMode.POLYGONAL, "zone-polygon"),
            ("Lasso", ZoneCreationMode.LASSO, "lasso"),
            ("Rectangular", ZoneCreationMode.RECTANGULAR, "square-dashed"),
        ):
            self._zone_mode_box.addItem(icon(glyph), text, zone_mode)

        for code in available_languages():
            name = QtCore.QLocale.languageToString(QtCore.QLocale(code).language())
            self._language_box.addItem(name, code)
        self._language_box.setEnabled(self._language_box.count() > 1)

        for box, last in (
            (self._tiff_bw_box, ("CCITT G4", TiffCompression.CCITT_FAX4)),
            (self._tiff_color_box, ("JPEG", TiffCompression.JPEG)),
        ):
            for text, method in (
                ("None", TiffCompression.NONE),
                ("LZW", TiffCompression.LZW),
                ("Deflate", TiffCompression.DEFLATE),
                last,
            ):
                box.addItem(text, method)

    def _load_settings(self) -> None:
        """Show the stored values."""
        settings = self._settings

        # The Python port has no OpenGL rendering path to switch on.
        self._enable_opengl_cb.setChecked(False)
        self._enable_opengl_cb.setEnabled(False)
        self._opengl_device_label.setEnabled(False)
        self._opengl_device_label.setText(_OPENGL_UNAVAILABLE)

        self._auto_save_cb.setChecked(settings.auto_save_project)
        self._select_data(self._color_scheme_box, AUTO)
        self._select_data(
            self._color_scheme_box,
            str(QtCore.QSettings().value(SETTINGS_KEY, AUTO)).strip().lower(),
        )
        self._select_data(self._units_box, settings.units)
        self._select_data(self._language_box, settings.language)
        self._select_data(self._zone_mode_box, settings.default_zone_creation_mode)

        self._thumbnail_quality_sb.setValue(settings.thumbnail_quality_width)
        self._thumbnail_size_sb.setValue(int(settings.max_thumbnail_width))
        self._single_column_cb.setChecked(settings.single_column_thumbnail_display)
        self._cancel_selection_cb.setChecked(settings.show_canceling_selection_question)

        self._select_data(self._tiff_bw_box, settings.tiff_bw_compression)
        self._select_data(self._tiff_color_box, settings.tiff_color_compression)

        self._black_on_white_cb.setChecked(settings.black_on_white_detection)
        self._black_on_white_output_cb.setChecked(
            settings.black_on_white_detection_output
        )
        self._black_on_white_output_cb.setEnabled(self._black_on_white_cb.isChecked())

        self._highlight_deviation_cb.setChecked(settings.highlight_deviation)
        for coef, thresh, deviation in self._deviation_rows(settings):
            coef.setValue(deviation.coefficient)
            thresh.setValue(deviation.threshold)

    def _deviation_rows(
        self, settings: ApplicationSettings
    ) -> list[
        tuple[QtWidgets.QDoubleSpinBox, QtWidgets.QDoubleSpinBox, DeviationSettings]
    ]:
        return [
            (
                self._deskew_deviation_coef,
                self._deskew_deviation_thresh,
                settings.deskew_deviation,
            ),
            (
                self._select_content_deviation_coef,
                self._select_content_deviation_thresh,
                settings.select_content_deviation,
            ),
            (
                self._margins_deviation_coef,
                self._margins_deviation_thresh,
                settings.margins_deviation,
            ),
        ]

    def _disable_unwired_options(self) -> None:
        for name in _UNWIRED_WIDGETS:
            get_cwidget(self.ui, QtWidgets.QWidget, name).setEnabled(False)

    @staticmethod
    def _select_data(box: QtWidgets.QComboBox, data: object) -> None:
        """Select the item that carries ``data``, leaving the box as is if none does."""
        index = box.findData(data)
        if index != -1:
            box.setCurrentIndex(index)

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._button_box.accepted.connect(self._save_and_accept)
        self._button_box.rejected.connect(self.reject)
        self._black_on_white_cb.clicked.connect(
            self._black_on_white_output_cb.setEnabled
        )
        self._color_scheme_box.currentIndexChanged.connect(self._preview_appearance)

    def _save_and_accept(self) -> None:
        """Commit the values and close the dialog."""
        self._save_settings()
        self._store_appearance()
        self._units_provider.set_units(self._settings.units)
        self.settings_changed.emit()
        self.accept()

    def reject(self) -> None:
        """Close the dialog and bring back the scheme in use before it opened."""
        self._apply_stored_appearance()
        super().reject()

    def _save_settings(self) -> None:
        """Write the form to the settings object and to disk."""
        settings = self._settings
        settings.auto_save_project = self._auto_save_cb.isChecked()
        settings.units = Units(self._units_box.currentData())
        settings.language = self._language_box.currentData()
        settings.default_zone_creation_mode = ZoneCreationMode(
            self._zone_mode_box.currentData()
        )

        quality = self._thumbnail_quality_sb.value()
        settings.thumbnail_quality_width = quality
        settings.thumbnail_quality_height = quality
        width = float(self._thumbnail_size_sb.value())
        settings.max_thumbnail_width = width
        settings.max_thumbnail_height = round(width * _THUMBNAIL_ASPECT, 2)
        settings.single_column_thumbnail_display = self._single_column_cb.isChecked()
        settings.show_canceling_selection_question = (
            self._cancel_selection_cb.isChecked()
        )

        settings.tiff_bw_compression = self._tiff_bw_box.currentData()
        settings.tiff_color_compression = self._tiff_color_box.currentData()

        settings.black_on_white_detection = self._black_on_white_cb.isChecked()
        settings.black_on_white_detection_output = (
            self._black_on_white_output_cb.isChecked()
        )

        settings.highlight_deviation = self._highlight_deviation_cb.isChecked()
        settings.deskew_deviation = self._deviation(
            self._deskew_deviation_coef, self._deskew_deviation_thresh
        )
        settings.select_content_deviation = self._deviation(
            self._select_content_deviation_coef,
            self._select_content_deviation_thresh,
        )
        settings.margins_deviation = self._deviation(
            self._margins_deviation_coef, self._margins_deviation_thresh
        )

        try:
            settings.save()
        except OSError:
            logger.exception("The settings could not be saved")

    @staticmethod
    def _deviation(
        coef: QtWidgets.QDoubleSpinBox, thresh: QtWidgets.QDoubleSpinBox
    ) -> DeviationSettings:
        return DeviationSettings(coefficient=coef.value(), threshold=thresh.value())

    def _store_appearance(self) -> None:
        """Remember the chosen appearance and show it right away."""
        QtCore.QSettings().setValue(SETTINGS_KEY, self._color_scheme_box.currentData())
        self._apply_stored_appearance()

    def _preview_appearance(self) -> None:
        """Show the chosen scheme without remembering it."""
        app = QtWidgets.QApplication.instance()
        if isinstance(app, QtWidgets.QApplication):
            apply_theme(app, self._color_scheme_box.currentData())

    @staticmethod
    def _apply_stored_appearance() -> None:
        app = QtWidgets.QApplication.instance()
        if isinstance(app, QtWidgets.QApplication):
            apply_theme(app)

    def get_settings(self) -> ApplicationSettings:
        """Get the (possibly modified) settings.

        Returns:
            ApplicationSettings: The application settings.
        """
        return self._settings
