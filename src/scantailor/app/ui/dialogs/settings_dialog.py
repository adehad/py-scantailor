"""Settings dialog for application preferences."""

from __future__ import annotations

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
        self._button_box = get_cwidget(
            self.ui, QtWidgets.QDialogButtonBox, "buttonBox"
        )

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
        self._enable_opengl_cb.setChecked(self._settings.enable_opengl)
        self._auto_save_cb.setChecked(self._settings.auto_save_project)

        # Set color scheme
        scheme_index = {"light": 0, "dark": 1, "system": 2}.get(
            self._settings.color_scheme, 2
        )
        self._color_scheme_box.setCurrentIndex(scheme_index)

        # Thumbnails
        self._thumbnail_quality_sb.setValue(self._settings.thumbnail_quality)
        self._thumbnail_size_sb.setValue(self._settings.thumbnail_size)
        self._single_column_cb.setChecked(self._settings.single_column_thumbnails)
        self._cancel_selection_cb.setChecked(
            self._settings.show_cancel_selection_question
        )

        # TIFF compression
        bw_index = {"none": 0, "lzw": 1, "deflate": 2, "ccitt_g4": 3}.get(
            self._settings.tiff_compression_bw, 0
        )
        self._tiff_bw_box.setCurrentIndex(bw_index)

        color_index = {"none": 0, "lzw": 1, "deflate": 2, "jpeg": 3}.get(
            self._settings.tiff_compression_color, 1
        )
        self._tiff_color_box.setCurrentIndex(color_index)

        # White on black detection
        self._black_on_white_cb.setChecked(self._settings.auto_detect_black_on_white)
        self._black_on_white_output_cb.setChecked(
            self._settings.use_black_on_white_at_output
        )

        # Deviation settings
        self._highlight_deviation_cb.setChecked(self._settings.highlight_deviation)
        deviation = self._settings.deviation_thresholds
        self._deskew_deviation_coef.setValue(deviation.deskew_coefficient)
        self._deskew_deviation_thresh.setValue(deviation.deskew_threshold)
        self._select_content_deviation_coef.setValue(
            deviation.select_content_coefficient
        )
        self._select_content_deviation_thresh.setValue(
            deviation.select_content_threshold
        )
        self._margins_deviation_coef.setValue(deviation.margins_coefficient)
        self._margins_deviation_thresh.setValue(deviation.margins_threshold)

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
        from scantailor.core.settings import DeviationThresholds

        # User Interface
        self._settings.enable_opengl = self._enable_opengl_cb.isChecked()
        self._settings.auto_save_project = self._auto_save_cb.isChecked()

        schemes = ["light", "dark", "system"]
        self._settings.color_scheme = schemes[self._color_scheme_box.currentIndex()]

        # Thumbnails
        self._settings.thumbnail_quality = self._thumbnail_quality_sb.value()
        self._settings.thumbnail_size = self._thumbnail_size_sb.value()
        self._settings.single_column_thumbnails = self._single_column_cb.isChecked()
        self._settings.show_cancel_selection_question = (
            self._cancel_selection_cb.isChecked()
        )

        # TIFF compression
        bw_options = ["none", "lzw", "deflate", "ccitt_g4"]
        self._settings.tiff_compression_bw = bw_options[
            self._tiff_bw_box.currentIndex()
        ]

        color_options = ["none", "lzw", "deflate", "jpeg"]
        self._settings.tiff_compression_color = color_options[
            self._tiff_color_box.currentIndex()
        ]

        # White on black detection
        self._settings.auto_detect_black_on_white = (
            self._black_on_white_cb.isChecked()
        )
        self._settings.use_black_on_white_at_output = (
            self._black_on_white_output_cb.isChecked()
        )

        # Deviation settings
        self._settings.highlight_deviation = self._highlight_deviation_cb.isChecked()
        self._settings.deviation_thresholds = DeviationThresholds(
            deskew_coefficient=self._deskew_deviation_coef.value(),
            deskew_threshold=self._deskew_deviation_thresh.value(),
            select_content_coefficient=self._select_content_deviation_coef.value(),
            select_content_threshold=self._select_content_deviation_thresh.value(),
            margins_coefficient=self._margins_deviation_coef.value(),
            margins_threshold=self._margins_deviation_thresh.value(),
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
