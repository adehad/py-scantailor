"""The Settings, Default Parameters, Project Files and Split Mode dialogs via exec().

Each dialog is built from its real ``.ui`` and driven from a timer while ``exec()``
blocks, so what is checked is what a user would see.
"""

import pathlib
import re
from collections.abc import Callable, Iterator

import pytest
from PySide6 import QtCore, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui import theme
from scantailor.app.ui.dialogs.default_params import DefaultParamsDialog
from scantailor.app.ui.dialogs.project_files import ProjectFilesDialog
from scantailor.app.ui.dialogs.settings import SettingsDialog
from scantailor.app.ui.dialogs.split_mode import SplitModeDialog
from scantailor.app.ui.icons import icon
from scantailor.app.ui.units_provider import UnitsProvider
from scantailor.core.default_params import DefaultParams, apply_default_params
from scantailor.core.default_params_profiles import (
    ProfileStore,
    active_default_params,
)
from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.core.settings import ApplicationSettings, TiffCompression, Units
from scantailor.core.stage_sequence import StageSequence
from scantailor.filters.deskew.params import AutoManualMode
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.page_split.layout_type import LayoutType

DIALOG_FOLDER = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src"
    / "scantailor"
    / "app"
    / "ui"
    / "dialogs"
)
MIN_ROWS = 12


@pytest.fixture
def config(tmp_path: pathlib.Path) -> Iterator[pathlib.Path]:
    """Settings and profiles in a temporary folder, never the user's."""
    ApplicationSettings.set_config_path(tmp_path / "settings.json")
    UnitsProvider.reset_instance()
    yield tmp_path
    ApplicationSettings.reset_instance()
    UnitsProvider.reset_instance()


def find[T: QtWidgets.QWidget](
    dialog: QtWidgets.QDialog, kind: type[T], name: str
) -> T:
    widget = dialog.findChild(kind, name)
    assert widget is not None, f"{name} not found"
    return widget


def exec_while(
    dialog: QtWidgets.QDialog, drive: Callable[[QtWidgets.QDialog], None]
) -> QtWidgets.QDialog.DialogCode:
    """Run ``dialog.exec()`` and call ``drive`` once the dialog is on screen."""
    errors: list[BaseException] = []
    clock = QtCore.QElapsedTimer()
    clock.start()
    timer = QtCore.QTimer()
    timer.setInterval(20)

    def poll() -> None:
        if not dialog.isVisible():
            if clock.hasExpired(10000):
                timer.stop()
                errors.append(AssertionError("dialog never became visible"))
                dialog.reject()
            return
        timer.stop()
        try:
            drive(dialog)
        except BaseException as error:
            errors.append(error)
        if dialog.isVisible():
            dialog.reject()

    timer.timeout.connect(poll)
    timer.start()
    result = dialog.exec()
    timer.stop()
    if errors:
        raise errors[0]
    return QtWidgets.QDialog.DialogCode(result)


def assert_shown(dialog: QtWidgets.QDialog, *names: str) -> None:
    assert dialog.isVisible()
    for name in names:
        widget = find(dialog, QtWidgets.QWidget, name)
        assert widget.isVisible(), f"{name} is not visible"
        assert widget.width() > 0 and widget.height() > 0, f"{name} has no size"


def accept(dialog: QtWidgets.QDialog) -> None:
    find(dialog, QtWidgets.QDialogButtonBox, "buttonBox").button(
        QtWidgets.QDialogButtonBox.StandardButton.Ok
    ).click()


def cancel(dialog: QtWidgets.QDialog) -> None:
    find(dialog, QtWidgets.QDialogButtonBox, "buttonBox").button(
        QtWidgets.QDialogButtonBox.StandardButton.Cancel
    ).click()


def window_lightness(app: QtWidgets.QApplication) -> int:
    return app.palette().window().color().lightness()


UNWIRED = (
    ("highlightDeviationCB", QtWidgets.QCheckBox),
    ("deskewDeviationCoefSB", QtWidgets.QDoubleSpinBox),
    ("deskewDeviationThresholdSB", QtWidgets.QDoubleSpinBox),
    ("selectContentDeviationCoefSB", QtWidgets.QDoubleSpinBox),
    ("selectContentDeviationThresholdSB", QtWidgets.QDoubleSpinBox),
    ("marginsDeviationCoefSB", QtWidgets.QDoubleSpinBox),
    ("marginsDeviationThresholdSB", QtWidgets.QDoubleSpinBox),
)
WIRED = (
    ("languageBox", QtWidgets.QComboBox),
    ("colorSchemeBox", QtWidgets.QComboBox),
    ("hoverHighlightBox", QtWidgets.QComboBox),
    ("unitsBox", QtWidgets.QComboBox),
    ("autoSaveProjectCB", QtWidgets.QCheckBox),
    ("thumbnailQualitySB", QtWidgets.QSpinBox),
    ("thumbnailSizeSB", QtWidgets.QSpinBox),
    ("singleColumnThumbnailsCB", QtWidgets.QCheckBox),
    ("cancelingSelectionQuestionCB", QtWidgets.QCheckBox),
    ("tiffCompressionBWBox", QtWidgets.QComboBox),
    ("tiffCompressionColorBox", QtWidgets.QComboBox),
    ("blackOnWhiteDetectionCB", QtWidgets.QCheckBox),
    ("blackOnWhiteDetectionAtOutputCB", QtWidgets.QCheckBox),
)


class TestSettingsDialog:
    def test_shows_its_controls_and_saves_what_was_edited(
        self, qtbot: QtBot, config: pathlib.Path
    ) -> None:
        settings = ApplicationSettings()
        dialog = SettingsDialog(settings)
        qtbot.addWidget(dialog)

        def drive(dlg: QtWidgets.QDialog) -> None:
            assert_shown(
                dlg,
                "colorSchemeBox",
                "languageBox",
                "unitsBox",
                "thumbnailSizeSB",
                "tiffCompressionBWBox",
                "buttonBox",
            )
            units = find(dlg, QtWidgets.QComboBox, "unitsBox")
            units.setCurrentIndex(units.findData(Units.INCHES))
            find(dlg, QtWidgets.QCheckBox, "autoSaveProjectCB").setChecked(True)
            find(dlg, QtWidgets.QSpinBox, "thumbnailSizeSB").setValue(300)
            tiff = find(dlg, QtWidgets.QComboBox, "tiffCompressionBWBox")
            tiff.setCurrentIndex(tiff.findData(TiffCompression.LZW))
            language = find(dlg, QtWidgets.QComboBox, "languageBox")
            language.setCurrentIndex(language.findData("de"))
            accept(dlg)

        assert exec_while(dialog, drive) == QtWidgets.QDialog.DialogCode.Accepted
        saved = ApplicationSettings.load()
        assert saved.units is Units.INCHES
        assert saved.auto_save_project is True
        assert saved.max_thumbnail_width == 300.0
        assert saved.max_thumbnail_height == 192.0
        assert saved.tiff_bw_compression is TiffCompression.LZW
        assert saved.language == "de"

    def test_options_without_a_backend_are_disabled(
        self, qtbot: QtBot, config: pathlib.Path
    ) -> None:
        dialog = SettingsDialog(ApplicationSettings())
        qtbot.addWidget(dialog)

        def drive(dlg: QtWidgets.QDialog) -> None:
            assert_shown(dlg, "enableOpenglCb")
            assert not find(dlg, QtWidgets.QCheckBox, "enableOpenglCb").isEnabled()
            for name, kind in UNWIRED:
                assert not find(dlg, kind, name).isEnabled(), name
            for name, kind in WIRED:
                assert find(dlg, kind, name).isEnabled(), name

        exec_while(dialog, drive)

    def test_unticking_black_on_white_detection_disables_it_at_output_and_saves_both(
        self, qtbot: QtBot, config: pathlib.Path
    ) -> None:
        settings = ApplicationSettings()
        dialog = SettingsDialog(settings)
        qtbot.addWidget(dialog)

        def drive(dlg: QtWidgets.QDialog) -> None:
            at_output = find(
                dlg, QtWidgets.QCheckBox, "blackOnWhiteDetectionAtOutputCB"
            )
            at_output.click()
            find(dlg, QtWidgets.QCheckBox, "blackOnWhiteDetectionCB").click()
            assert not at_output.isEnabled()
            accept(dlg)

        exec_while(dialog, drive)

        saved = ApplicationSettings.load()
        assert saved.black_on_white_detection is False
        assert saved.black_on_white_detection_output is False
        assert saved.polarity_detection.output is False

    def test_disabled_options_keep_their_stored_values(
        self, qtbot: QtBot, config: pathlib.Path
    ) -> None:
        settings = ApplicationSettings(language="de", highlight_deviation=False)
        dialog = SettingsDialog(settings)
        qtbot.addWidget(dialog)

        exec_while(dialog, accept)

        assert settings.language == "de"
        assert settings.highlight_deviation is False

    def test_units_offer_pixels_and_reach_the_provider_on_accept(
        self, qtbot: QtBot, config: pathlib.Path
    ) -> None:
        provider = UnitsProvider(Units.MILLIMETERS)
        dialog = SettingsDialog(ApplicationSettings(), units_provider=provider)
        qtbot.addWidget(dialog)
        announced: list[Units] = []
        provider.units_changed.connect(announced.append)

        def drive(dlg: QtWidgets.QDialog) -> None:
            units = find(dlg, QtWidgets.QComboBox, "unitsBox")
            assert units.findData(Units.PIXELS) != -1
            units.setCurrentIndex(units.findData(Units.PIXELS))
            assert provider.units is Units.MILLIMETERS
            accept(dlg)

        exec_while(dialog, drive)

        assert provider.units is Units.PIXELS
        assert announced == [Units.PIXELS]

    def test_cancel_does_not_change_the_unit(
        self, qtbot: QtBot, config: pathlib.Path
    ) -> None:
        provider = UnitsProvider(Units.MILLIMETERS)
        dialog = SettingsDialog(ApplicationSettings(), units_provider=provider)
        qtbot.addWidget(dialog)

        def drive(dlg: QtWidgets.QDialog) -> None:
            units = find(dlg, QtWidgets.QComboBox, "unitsBox")
            units.setCurrentIndex(units.findData(Units.INCHES))
            cancel(dlg)

        exec_while(dialog, drive)

        assert provider.units is Units.MILLIMETERS

    def test_choosing_a_scheme_previews_it_without_storing_it(
        self,
        qtbot: QtBot,
        qapp: QtWidgets.QApplication,
        config: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv(theme.THEME_ENV_VAR, raising=False)
        theme.apply_theme(qapp, "light")
        QtCore.QSettings().setValue(theme.SETTINGS_KEY, "light")
        dialog = SettingsDialog(ApplicationSettings())
        qtbot.addWidget(dialog)
        light = window_lightness(qapp)

        def drive(dlg: QtWidgets.QDialog) -> None:
            box = find(dlg, QtWidgets.QComboBox, "colorSchemeBox")
            box.setCurrentIndex(box.findData("dark"))
            assert window_lightness(qapp) < light
            assert QtCore.QSettings().value(theme.SETTINGS_KEY) == "light"
            cancel(dlg)

        exec_while(dialog, drive)
        theme.apply_theme(qapp, theme.AUTO)

    @pytest.mark.parametrize("close", ["cancel button", "escape", "window close"])
    def test_closing_without_accepting_brings_back_the_earlier_scheme(
        self,
        qtbot: QtBot,
        qapp: QtWidgets.QApplication,
        config: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        close: str,
    ) -> None:
        monkeypatch.delenv(theme.THEME_ENV_VAR, raising=False)
        theme.apply_theme(qapp, "light")
        QtCore.QSettings().setValue(theme.SETTINGS_KEY, "light")
        dialog = SettingsDialog(ApplicationSettings())
        qtbot.addWidget(dialog)
        light = window_lightness(qapp)

        def drive(dlg: QtWidgets.QDialog) -> None:
            box = find(dlg, QtWidgets.QComboBox, "colorSchemeBox")
            box.setCurrentIndex(box.findData("dark"))
            assert window_lightness(qapp) < light
            if close == "cancel button":
                cancel(dlg)
            elif close == "escape":
                qtbot.keyClick(dlg, QtCore.Qt.Key.Key_Escape)
            else:
                dlg.close()

        assert exec_while(dialog, drive) == QtWidgets.QDialog.DialogCode.Rejected
        assert window_lightness(qapp) == light
        assert QtCore.QSettings().value(theme.SETTINGS_KEY) == "light"
        theme.apply_theme(qapp, theme.AUTO)

    def test_accepting_keeps_the_previewed_scheme(
        self,
        qtbot: QtBot,
        qapp: QtWidgets.QApplication,
        config: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv(theme.THEME_ENV_VAR, raising=False)
        theme.apply_theme(qapp, "light")
        QtCore.QSettings().setValue(theme.SETTINGS_KEY, "light")
        dialog = SettingsDialog(ApplicationSettings())
        qtbot.addWidget(dialog)
        light = window_lightness(qapp)

        def drive(dlg: QtWidgets.QDialog) -> None:
            box = find(dlg, QtWidgets.QComboBox, "colorSchemeBox")
            box.setCurrentIndex(box.findData("dark"))
            accept(dlg)

        exec_while(dialog, drive)

        assert window_lightness(qapp) < light
        assert QtCore.QSettings().value(theme.SETTINGS_KEY) == "dark"
        theme.apply_theme(qapp, theme.AUTO)

    def test_hover_highlight_is_neutral_until_the_accent_is_chosen(
        self,
        qtbot: QtBot,
        qapp: QtWidgets.QApplication,
        config: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv(theme.THEME_ENV_VAR, raising=False)
        theme.apply_theme(qapp, "light")
        neutral = "QToolButton#groupHeader:pressed { background: rgba(42, 38, 32"
        assert neutral in qapp.styleSheet()
        dialog = SettingsDialog(ApplicationSettings())
        qtbot.addWidget(dialog)

        def drive(dlg: QtWidgets.QDialog) -> None:
            box = find(dlg, QtWidgets.QComboBox, "hoverHighlightBox")
            assert box.currentData() == theme.NEUTRAL
            box.setCurrentIndex(box.findData(theme.ACCENT))
            accept(dlg)

        exec_while(dialog, drive)

        assert QtCore.QSettings().value(theme.HOVER_KEY) == theme.ACCENT
        accent = "QToolButton#groupHeader:pressed { background: rgba(47, 111, 237"
        assert accent in qapp.styleSheet()
        theme.apply_theme(qapp, theme.AUTO)

    def test_cancel_leaves_settings_untouched(
        self, qtbot: QtBot, config: pathlib.Path
    ) -> None:
        settings = ApplicationSettings()
        dialog = SettingsDialog(settings)
        qtbot.addWidget(dialog)

        def drive(dlg: QtWidgets.QDialog) -> None:
            find(dlg, QtWidgets.QCheckBox, "autoSaveProjectCB").setChecked(True)
            find(dlg, QtWidgets.QDialogButtonBox, "buttonBox").button(
                QtWidgets.QDialogButtonBox.StandardButton.Cancel
            ).click()

        assert exec_while(dialog, drive) == QtWidgets.QDialog.DialogCode.Rejected
        assert settings.auto_save_project is False
        assert not ApplicationSettings.get_config_path().exists()

    def test_appearance_switches_the_theme_live(
        self,
        qtbot: QtBot,
        qapp: QtWidgets.QApplication,
        config: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv(theme.THEME_ENV_VAR, raising=False)
        lightness = {}
        for mode in ("light", "dark"):
            dialog = SettingsDialog(ApplicationSettings())
            qtbot.addWidget(dialog)

            def drive(dlg: QtWidgets.QDialog, mode: str = mode) -> None:
                box = find(dlg, QtWidgets.QComboBox, "colorSchemeBox")
                box.setCurrentIndex(box.findData(mode))
                accept(dlg)

            exec_while(dialog, drive)
            assert QtCore.QSettings().value(theme.SETTINGS_KEY) == mode
            lightness[mode] = qapp.palette().window().color().lightness()
        assert lightness["light"] > lightness["dark"]
        theme.apply_theme(qapp, theme.AUTO)


class TestDefaultParamsDialog:
    def make(self, qtbot: QtBot, config: pathlib.Path) -> DefaultParamsDialog:
        dialog = DefaultParamsDialog(
            settings=ApplicationSettings(), store=ProfileStore(config / "profiles")
        )
        qtbot.addWidget(dialog)
        return dialog

    def test_shows_its_controls(self, qtbot: QtBot, config: pathlib.Path) -> None:
        dialog = self.make(qtbot, config)

        def drive(dlg: QtWidgets.QDialog) -> None:
            assert_shown(
                dlg,
                "profileCB",
                "profileSaveButton",
                "tabWidget",
                "rotateLeftBtn",
                "rotationIndicator",
                "buttonBox",
            )
            assert (
                not find(dlg, QtWidgets.QLabel, "rotationIndicator").pixmap().isNull()
            )
            assert not find(dlg, QtWidgets.QPushButton, "profileSaveButton").isEnabled()

        exec_while(dialog, drive)

    def test_source_profile_is_selected_and_read_only(
        self, qtbot: QtBot, config: pathlib.Path
    ) -> None:
        dialog = self.make(qtbot, config)

        def drive(dlg: QtWidgets.QDialog) -> None:
            box = find(dlg, QtWidgets.QComboBox, "profileCB")
            box.setCurrentIndex(box.findData("Source"))
            assert find(dlg, QtWidgets.QDoubleSpinBox, "topMarginSpinBox").value() == 0
            assert (
                not find(dlg, QtWidgets.QTabWidget, "tabWidget")
                .currentWidget()
                .isEnabled()
            )
            accept(dlg)

        exec_while(dialog, drive)
        assert ApplicationSettings.load().current_profile == "Source"
        assert (
            active_default_params(
                ApplicationSettings.load(), ProfileStore(config / "profiles")
            )
            == DefaultParams.source()
        )

    def test_custom_profile_saves_loads_and_reaches_a_new_project(
        self, qtbot: QtBot, config: pathlib.Path
    ) -> None:
        store = ProfileStore(config / "profiles")
        dialog = self.make(qtbot, config)

        def edit(dlg: QtWidgets.QDialog) -> None:
            box = find(dlg, QtWidgets.QComboBox, "profileCB")
            box.setCurrentIndex(box.count() - 1)
            find(dlg, QtWidgets.QPushButton, "deskewManualBtn").click()
            find(dlg, QtWidgets.QDoubleSpinBox, "angleSpinBox").setValue(2.5)
            modes = find(dlg, QtWidgets.QComboBox, "colorModeSelector")
            modes.setCurrentIndex(modes.findData(ColorMode.MIXED))
            find(dlg, QtWidgets.QToolButton, "rotateRightBtn").click()
            box.setEditText("Thick books")
            find(dlg, QtWidgets.QPushButton, "profileSaveButton").click()
            assert box.currentText() == "Thick books"
            accept(dlg)

        exec_while(dialog, edit)

        assert store.names() == ["Thick books"]
        saved = store.read("Thick books")
        assert saved.deskew.mode is AutoManualMode.MANUAL
        assert saved.deskew.deskew_angle_deg == -2.5
        assert saved.output.color_mode is ColorMode.MIXED
        assert saved.fix_orientation.image_rotation.degrees == 90

        settings = ApplicationSettings.load()
        assert settings.current_profile == "Thick books"
        image = ImageId(file_path=pathlib.Path("scan.png"))
        stages = StageSequence()
        apply_default_params(stages, [image], active_default_params(settings, store))
        page = PageId(image_id=image, sub_page=SubPage.SINGLE_PAGE)
        assert (
            stages.deskew_filter.settings.get_params(page).mode is AutoManualMode.MANUAL
        )
        assert (
            stages.output_filter.settings.get_params(page).color_mode is ColorMode.MIXED
        )

        reopened = DefaultParamsDialog(settings=settings, store=store)
        qtbot.addWidget(reopened)

        def check(dlg: QtWidgets.QDialog) -> None:
            box = find(dlg, QtWidgets.QComboBox, "profileCB")
            assert box.currentText() == "Thick books"
            assert find(dlg, QtWidgets.QDoubleSpinBox, "angleSpinBox").value() == 2.5
            assert find(dlg, QtWidgets.QPushButton, "profileDeleteButton").isEnabled()

        exec_while(reopened, check)

    def test_untouched_form_builds_the_loaded_profile(
        self, qtbot: QtBot, config: pathlib.Path
    ) -> None:
        for units in (Units.MILLIMETERS, Units.CENTIMETERS, Units.INCHES):
            dialog = DefaultParamsDialog(
                units=units,
                settings=ApplicationSettings(),
                store=ProfileStore(config / "profiles"),
            )
            qtbot.addWidget(dialog)
            assert dialog.get_params().model_copy(
                update={"units": Units.MILLIMETERS}
            ) == (DefaultParams())

    def test_layout_type_and_alignment_round_trip(
        self, qtbot: QtBot, config: pathlib.Path
    ) -> None:
        store = ProfileStore(config / "profiles")
        params = DefaultParams.source()
        store.write("two", params.model_copy(deep=True))
        settings = ApplicationSettings(current_profile="two")
        dialog = DefaultParamsDialog(settings=settings, store=store)
        qtbot.addWidget(dialog)
        assert (
            dialog.get_params().page_split.layout_type is LayoutType.SINGLE_PAGE_UNCUT
        )
        assert dialog.get_params().page_layout.alignment.is_null
        assert dialog.get_params() == params

    def test_dewarp_minimum_span_round_trips_through_a_saved_profile(
        self, qtbot: QtBot, config: pathlib.Path
    ) -> None:
        store = ProfileStore(config / "profiles")
        dialog = self.make(qtbot, config)

        def edit(dlg: QtWidgets.QDialog) -> None:
            box = find(dlg, QtWidgets.QComboBox, "profileCB")
            box.setCurrentIndex(box.count() - 1)
            spin = find(dlg, QtWidgets.QSpinBox, "dewarpingMinSpanSB")
            assert spin.value() == DefaultParams().output.dewarp_min_span_percent
            assert (spin.minimum(), spin.maximum()) == (0, 100)
            spin.setValue(25)
            box.setEditText("Flat books")
            find(dlg, QtWidgets.QPushButton, "profileSaveButton").click()
            accept(dlg)

        exec_while(dialog, edit)
        assert store.read("Flat books").output.dewarp_min_span_percent == 25

        reopened = DefaultParamsDialog(settings=ApplicationSettings.load(), store=store)
        qtbot.addWidget(reopened)
        assert find(reopened, QtWidgets.QSpinBox, "dewarpingMinSpanSB").value() == 25
        assert reopened.get_params().output.dewarp_min_span_percent == 25

    def test_custom_profile_name_may_not_be_reserved(
        self, qtbot: QtBot, config: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        shown: list[str] = []
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "information",
            lambda *args: shown.append(args[2]),
        )
        dialog = self.make(qtbot, config)

        def drive(dlg: QtWidgets.QDialog) -> None:
            box = find(dlg, QtWidgets.QComboBox, "profileCB")
            box.setCurrentIndex(box.count() - 1)
            box.setEditText("source")
            find(dlg, QtWidgets.QPushButton, "profileSaveButton").click()

        exec_while(dialog, drive)
        assert shown
        assert ProfileStore(config / "profiles").names() == []


class TestProjectFilesDialog:
    def test_both_lists_show_at_least_twelve_rows_at_the_default_size(
        self, qtbot: QtBot, tmp_path: pathlib.Path
    ) -> None:
        dialog = ProjectFilesDialog(input_dir=tmp_path)
        qtbot.addWidget(dialog)

        def drive(dlg: QtWidgets.QDialog) -> None:
            for name in ("offProjectList", "inProjectList"):
                view = find(dlg, QtWidgets.QListView, name)
                row = view.fontMetrics().height()
                assert view.viewport().height() >= MIN_ROWS * row, name

        exec_while(dialog, drive)

    def test_icons_of_the_transfer_buttons_are_set(
        self, qtbot: QtBot, tmp_path: pathlib.Path
    ) -> None:
        dialog = ProjectFilesDialog(input_dir=tmp_path)
        qtbot.addWidget(dialog)
        for name in ("addToProjectBtn", "removeFromProjectBtn"):
            assert not find(dialog, QtWidgets.QToolButton, name).icon().isNull()


class TestSplitModeDialog:
    def test_auto_layout_shows_the_wand(self, qtbot: QtBot) -> None:
        dialog = SplitModeDialog(layout_type=LayoutType.AUTO_LAYOUT_TYPE)
        qtbot.addWidget(dialog)

        def drive(dlg: QtWidgets.QDialog) -> None:
            assert_shown(dlg, "layoutTypeLabel")
            assert not find(dlg, QtWidgets.QLabel, "layoutTypeLabel").pixmap().isNull()

        exec_while(dialog, drive)

    def test_manual_mode_shows_the_auto_detected_layout_glyph(
        self, qtbot: QtBot
    ) -> None:
        dialog = SplitModeDialog(
            layout_type=LayoutType.AUTO_LAYOUT_TYPE,
            auto_detected_layout_type=LayoutType.TWO_PAGES,
        )
        qtbot.addWidget(dialog)

        def drive(dlg: QtWidgets.QDialog) -> None:
            label = find(dlg, QtWidgets.QLabel, "layoutTypeLabel")
            auto_image = label.pixmap().toImage()
            find(dlg, QtWidgets.QRadioButton, "modeManual").setChecked(True)
            assert label.pixmap().toImage() != auto_image

        exec_while(dialog, drive)

    def test_apply_cut_is_only_available_in_manual_mode(self, qtbot: QtBot) -> None:
        dialog = SplitModeDialog(layout_type=LayoutType.AUTO_LAYOUT_TYPE)
        qtbot.addWidget(dialog)

        def drive(dlg: QtWidgets.QDialog) -> None:
            cut = find(dlg, QtWidgets.QCheckBox, "applyCutOption")
            assert not cut.isEnabled()
            find(dlg, QtWidgets.QRadioButton, "modeManual").setChecked(True)
            assert cut.isEnabled()
            cut.setChecked(True)
            find(dlg, QtWidgets.QRadioButton, "modeAuto").setChecked(True)
            assert not cut.isEnabled()
            assert not cut.isChecked()

        exec_while(dialog, drive)


def test_no_dialog_file_refers_to_a_missing_icon(qapp: QtWidgets.QApplication) -> None:
    """Every ``:/icons/`` path and ``iconName`` of the four dialogs resolves."""
    names = ("settings", "default_params", "project_files", "split_mode")
    files = [DIALOG_FOLDER / f"{name}.py" for name in names] + [
        DIALOG_FOLDER / f"{name}.ui"
        for name in (
            "SettingsDialog",
            "DefaultParamsDialog",
            "ProjectFilesDialog",
            "SplitModeDialog",
        )
    ]
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert ":/icons/" not in text, f"{path.name} refers to a resource icon"
        for glyph in re.findall(
            r'name="iconName(?:On)?" stdset="0">\s*<string[^>]*>([\w-]+)<', text
        ):
            assert not icon(glyph).isNull(), f"{path.name}: {glyph}"
