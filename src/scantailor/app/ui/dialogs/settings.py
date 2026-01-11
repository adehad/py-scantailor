"""Settings dialog for application preferences."""

from pathlib import Path

from PySide6 import QtWidgets

from scantailor.app.ui import load_ui_widget
from scantailor.app.ui.utils import get_cwidget
from scantailor.core.settings import ApplicationSettings

_UI_FOLDER = Path(__file__).parent


class SettingsDialog(QtWidgets.QDialog):
    """Settings dialog for configuring application preferences.

    Provides tabs for:
    - General settings (UI, thumbnails, saving)
    - Processing settings (detection, deviation thresholds)
    """

    def __init__(
        self,
        settings: ApplicationSettings,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the settings dialog.

        Args:
            settings: Application settings to edit.
            parent: Parent widget.
        """
        super().__init__(parent)
        self._settings = settings
        self._original_settings = settings.model_copy()

        # Load UI
        self.ui = load_ui_widget(_UI_FOLDER / "SettingsDialog.ui", self)
        self._setup_widgets()
        self._load_settings()
        self._connect_signals()

    def _setup_widgets(self) -> None:
        """Set up widget references."""
        # General tab - User Interface
        self._enable_opengl_cb = get_cwidget(
            self.ui, QtWidgets.QCheckBox, "enableOpenglCb"
        )
        self._opengl_device_label = get_cwidget(
            self.ui, QtWidgets.QLabel, "openglDeviceLabel"
        )
        self._auto_save_cb = get_cwidget(
            self.ui, QtWidgets.QCheckBox, "autoSaveProjectCB"
        )
        self._color_scheme_box = get_cwidget(
            self.ui, QtWidgets.QComboBox, "colorSchemeBox"
        )
        self._language_box = get_cwidget(self.ui, QtWidgets.QComboBox, "languageBox")

        # General tab - Thumbnails
        self._thumbnail_quality_sb = get_cwidget(
            self.ui, QtWidgets.QSpinBox, "thumbnailQualitySB"
        )
        self._thumbnail_size_sb = get_cwidget(
            self.ui, QtWidgets.QSpinBox, "thumbnailSizeSB"
        )
        self._single_column_cb = get_cwidget(
            self.ui, QtWidgets.QCheckBox, "singleColumnThumbnailsCB"
        )
        self._cancel_selection_cb = get_cwidget(
            self.ui, QtWidgets.QCheckBox, "cancelingSelectionQuestionCB"
        )

        # General tab - Saving
        self._tiff_bw_box = get_cwidget(
            self.ui, QtWidgets.QComboBox, "tiffCompressionBWBox"
        )
        self._tiff_color_box = get_cwidget(
            self.ui, QtWidgets.QComboBox, "tiffCompressionColorBox"
        )

        # Processing tab - White on Black detection
        self._black_on_white_cb = get_cwidget(
            self.ui, QtWidgets.QCheckBox, "blackOnWhiteDetectionCB"
        )
        self._black_on_white_output_cb = get_cwidget(
            self.ui, QtWidgets.QCheckBox, "blackOnWhiteDetectionAtOutputCB"
        )

        # Processing tab - Deviation
        self._highlight_deviation_cb = get_cwidget(
            self.ui, QtWidgets.QCheckBox, "highlightDeviationCB"
        )
        self._deskew_deviation_coef = get_cwidget(
            self.ui, QtWidgets.QDoubleSpinBox, "deskewDeviationCoefSB"
        )
        self._deskew_deviation_thresh = get_cwidget(
            self.ui, QtWidgets.QDoubleSpinBox, "deskewDeviationThresholdSB"
        )
        self._select_content_deviation_coef = get_cwidget(
            self.ui, QtWidgets.QDoubleSpinBox, "selectContentDeviationCoefSB"
        )
        self._select_content_deviation_thresh = get_cwidget(
            self.ui, QtWidgets.QDoubleSpinBox, "selectContentDeviationThresholdSB"
        )
        self._margins_deviation_coef = get_cwidget(
            self.ui, QtWidgets.QDoubleSpinBox, "marginsDeviationCoefSB"
        )
        self._margins_deviation_thresh = get_cwidget(
            self.ui, QtWidgets.QDoubleSpinBox, "marginsDeviationThresholdSB"
        )

        # Button box
        self._button_box = get_cwidget(self.ui, QtWidgets.QDialogButtonBox, "buttonBox")

        # Populate combo boxes
        self._populate_combo_boxes()

    def _populate_combo_boxes(self) -> None:
        """Populate combo box options."""
        # Color schemes
        self._color_scheme_box.addItems(["Light", "Dark", "System"])

        # Languages (placeholder - would be populated from available translations)
        self._language_box.addItems(["English", "System"])

        # TIFF compression options
        compression_options = ["None", "LZW", "Deflate", "JPEG"]
        self._tiff_bw_box.addItems(["None", "LZW", "Deflate", "CCITT G4"])
        self._tiff_color_box.addItems(compression_options)

    def _load_settings(self) -> None:
        """Load current settings into the UI."""
        # User Interface
        self._enable_opengl_cb.setChecked(self._settings.opengl_enabled)
        self._auto_save_cb.setChecked(self._settings.auto_save_project)

        # Set color scheme
        scheme_index = {"light": 0, "dark": 1, "system": 2}.get(
            self._settings.color_scheme.value, 2
        )
        self._color_scheme_box.setCurrentIndex(scheme_index)

        # Thumbnails
        self._thumbnail_quality_sb.setValue(self._settings.thumbnail_quality_width)
        self._thumbnail_size_sb.setValue(int(self._settings.max_thumbnail_width))
        self._single_column_cb.setChecked(self._settings.single_column_thumbnail_display)
        self._cancel_selection_cb.setChecked(
            self._settings.show_canceling_selection_question
        )

        # TIFF compression
        bw_map = {1: 0, 5: 1, 8: 2, 4: 3}  # TiffCompression enum to index
        bw_index = bw_map.get(self._settings.tiff_bw_compression.value, 3)
        self._tiff_bw_box.setCurrentIndex(bw_index)

        color_map = {1: 0, 5: 1, 8: 2, 7: 3}  # TiffCompression enum to index
        color_index = color_map.get(self._settings.tiff_color_compression.value, 1)
        self._tiff_color_box.setCurrentIndex(color_index)

        # White on black detection
        self._black_on_white_cb.setChecked(self._settings.black_on_white_detection)
        self._black_on_white_output_cb.setChecked(
            self._settings.black_on_white_detection_output
        )

        # Deviation settings
        self._highlight_deviation_cb.setChecked(self._settings.highlight_deviation)
        self._deskew_deviation_coef.setValue(self._settings.deskew_deviation.coefficient)
        self._deskew_deviation_thresh.setValue(self._settings.deskew_deviation.threshold)
        self._select_content_deviation_coef.setValue(
            self._settings.select_content_deviation.coefficient
        )
        self._select_content_deviation_thresh.setValue(
            self._settings.select_content_deviation.threshold
        )
        self._margins_deviation_coef.setValue(self._settings.margins_deviation.coefficient)
        self._margins_deviation_thresh.setValue(self._settings.margins_deviation.threshold)

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._button_box.accepted.connect(self._save_and_accept)
        self._button_box.rejected.connect(self.reject)

    def _save_and_accept(self) -> None:
        """Save settings and close dialog."""
        self._save_settings()
        self.accept()

    def _save_settings(self) -> None:
        """Save UI values to settings."""
        from scantailor.core.settings import ColorScheme, DeviationSettings, TiffCompression

        # User Interface
        self._settings.opengl_enabled = self._enable_opengl_cb.isChecked()
        self._settings.auto_save_project = self._auto_save_cb.isChecked()

        schemes = [ColorScheme.LIGHT, ColorScheme.DARK, ColorScheme.DARK]  # "system" -> dark
        self._settings.color_scheme = schemes[self._color_scheme_box.currentIndex()]

        # Thumbnails
        self._settings.thumbnail_quality_width = self._thumbnail_quality_sb.value()
        self._settings.thumbnail_quality_height = self._thumbnail_quality_sb.value()
        self._settings.max_thumbnail_width = float(self._thumbnail_size_sb.value())
        self._settings.max_thumbnail_height = float(self._thumbnail_size_sb.value())
        self._settings.single_column_thumbnail_display = self._single_column_cb.isChecked()
        self._settings.show_canceling_selection_question = (
            self._cancel_selection_cb.isChecked()
        )

        # TIFF compression
        bw_options = [TiffCompression.NONE, TiffCompression.LZW, TiffCompression.DEFLATE, TiffCompression.CCITT_FAX4]
        self._settings.tiff_bw_compression = bw_options[
            self._tiff_bw_box.currentIndex()
        ]

        color_options = [TiffCompression.NONE, TiffCompression.LZW, TiffCompression.DEFLATE, TiffCompression.JPEG]
        self._settings.tiff_color_compression = color_options[
            self._tiff_color_box.currentIndex()
        ]

        # White on black detection
        self._settings.black_on_white_detection = self._black_on_white_cb.isChecked()
        self._settings.black_on_white_detection_output = (
            self._black_on_white_output_cb.isChecked()
        )

        # Deviation settings
        self._settings.highlight_deviation = self._highlight_deviation_cb.isChecked()
        self._settings.deskew_deviation = DeviationSettings(
            coefficient=self._deskew_deviation_coef.value(),
            threshold=self._deskew_deviation_thresh.value(),
        )
        self._settings.select_content_deviation = DeviationSettings(
            coefficient=self._select_content_deviation_coef.value(),
            threshold=self._select_content_deviation_thresh.value(),
        )
        self._settings.margins_deviation = DeviationSettings(
            coefficient=self._margins_deviation_coef.value(),
            threshold=self._margins_deviation_thresh.value(),
        )

    def get_settings(self) -> ApplicationSettings:
        """Get the (possibly modified) settings.

        Returns:
            The application settings.
        """
        return self._settings

    def show(self) -> None:
        """Show the dialog."""
        self.ui.show()
