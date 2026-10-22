import pathlib
from collections.abc import Iterator

import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtTest import QTest
from pytestqt.qtbot import QtBot

from scantailor.app.ui import MainWindow
from scantailor.app.ui.dialogs.about import AboutDialog
from scantailor.app.ui.units_provider import UnitsProvider
from scantailor.core import diagnostics
from scantailor.core.models import Dpi, Margins
from scantailor.core.settings import ApplicationSettings, Units
from scantailor.core.stage_sequence import FilterStage


def test_main_window_opens(qtbot: QtBot):
    manager = MainWindow.MainWindowManager()
    window = manager.ui
    qtbot.addWidget(window)
    window.show()
    qtbot.wait_active(window)


def test_the_window_owns_its_settings_and_the_application_owns_the_units(
    qtbot: QtBot, qapp: QtWidgets.QApplication
):
    manager = MainWindow.MainWindowManager()
    qtbot.addWidget(manager.ui)
    assert manager.settings.parent() is manager.ui
    assert UnitsProvider.get_instance().parent() is qapp


def test_show_about(qtbot: QtBot):
    manager = MainWindow.MainWindowManager()
    window = manager.ui
    qtbot.addWidget(window)
    before_qdialog = window.findChild(QtWidgets.QDialog)
    manager.actionAbout.trigger()
    qtbot.wait_until(lambda: window.findChild(QtWidgets.QDialog) is not None)
    after_qdialog = window.findChild(QtWidgets.QDialog)
    assert before_qdialog is None
    assert isinstance(after_qdialog, AboutDialog)
    after_qdialog.close()


def test_user_guide_action_opens_the_online_manual_in_the_browser(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
):
    opened: list[QtCore.QUrl] = []

    def fake_open_url(url: QtCore.QUrl) -> bool:
        opened.append(url)
        return True

    monkeypatch.setattr(QtGui.QDesktopServices, "openUrl", fake_open_url)
    manager = MainWindow.MainWindowManager()
    qtbot.addWidget(manager.ui)
    help_menu = manager.ui.findChild(QtWidgets.QMenu, "menuHelp")
    assert manager.actionUserGuide in help_menu.actions()
    manager.actionUserGuide.trigger()
    assert opened == [QtCore.QUrl(MainWindow.USER_GUIDE_URL)]
    assert opened[0].host() == "py-scantailor.readthedocs.io"


def test_open_log_folder_creates_and_shows_the_log_folder(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
):
    folder = tmp_path / "ScanTailor" / "logs"
    monkeypatch.setattr(diagnostics, "log_folder", lambda: folder)
    opened: list[QtCore.QUrl] = []
    monkeypatch.setattr(
        QtGui.QDesktopServices, "openUrl", lambda url: opened.append(url) or True
    )
    manager = MainWindow.MainWindowManager()
    qtbot.addWidget(manager.ui)
    help_menu = manager.ui.findChild(QtWidgets.QMenu, "menuHelp")
    assert manager.actionOpenLogFolder in help_menu.actions()
    assert manager.actionOpenLogFolder.toolTip()

    manager.actionOpenLogFolder.trigger()

    assert folder.is_dir()
    assert opened == [QtCore.QUrl.fromLocalFile(str(folder))]


def test_navigation_without_a_project_keeps_no_current_page(
    main_window: MainWindow.MainWindowManager,
):
    main_window.goNextPage()
    main_window.goLastPage()

    assert main_window.current_page is None


def test_page_navigation_moves_through_pages_and_clamps_at_the_ends(
    project_window: MainWindow.MainWindowManager,
):
    pages = project_window.pages()

    project_window.goPrevPage()
    assert project_window.current_page == pages[0]
    project_window.goNextPage()
    assert project_window.current_page == pages[1]
    project_window.goLastPage()
    assert project_window.current_page == pages[2]
    project_window.goNextPage()
    assert project_window.current_page == pages[2]
    project_window.goFirstPage()
    assert project_window.current_page == pages[0]


def test_page_navigation_actions_move_the_current_page(
    project_window: MainWindow.MainWindowManager,
):
    pages = project_window.pages()

    project_window.actionNextPage.trigger()
    project_window.actionNextPage.trigger()
    project_window.actionPrevPage.trigger()

    assert project_window.current_page == pages[1]


@pytest.fixture(autouse=True)
def isolated_application_settings(tmp_path: pathlib.Path) -> Iterator[None]:
    """Keep settings the window writes out of the user's configuration."""
    ApplicationSettings.set_config_path(tmp_path / "settings.json")
    UnitsProvider.reset_instance()
    yield
    ApplicationSettings.reset_instance()
    UnitsProvider.reset_instance()


# Shortcuts the C++ main window registers, by action.
CPP_SHORTCUTS = [
    ("actionPrevPage", "PgUp"),
    ("actionNextPage", "PgDown"),
    ("actionPrevPageQ", "Q"),
    ("actionNextPageW", "W"),
    ("actionFirstPage", "Home"),
    ("actionLastPage", "End"),
    ("actionPrevSelectedPage", "Shift+PgUp"),
    ("actionNextSelectedPage", "Shift+PgDown"),
    ("actionPrevSelectedPageQ", "Shift+Q"),
    ("actionNextSelectedPageW", "Shift+W"),
    ("actionGotoPage", "Ctrl+G"),
    ("actionMagnifyThumbnails", "Ctrl++"),
    ("actionDiminishThumbnails", "Ctrl+-"),
    ("actionReloadPage", "F5"),
    ("actionSwitchFilter1", "O"),
    ("actionSwitchFilter2", "E"),
    ("actionSwitchFilter3", "N"),
    ("actionSwitchFilter4", "V"),
    ("actionSwitchFilter5", "K"),
    ("actionSwitchFilter6", "P"),
]


class TestShortcuts:
    """The page, thumbnail and stage shortcuts work anywhere in the window."""

    @pytest.fixture
    def project_window(
        self, qtbot: QtBot, project_window: MainWindow.MainWindowManager
    ) -> MainWindow.MainWindowManager:
        """The project window, active so that its shortcuts fire."""
        project_window.ui.activateWindow()
        qtbot.wait_active(project_window.ui)
        return project_window

    @pytest.mark.parametrize(("action_name", "keys"), CPP_SHORTCUTS)
    def test_pressing_the_shortcut_triggers_its_action(
        self,
        main_window: MainWindow.MainWindowManager,
        action_name: str,
        keys: str,
    ):
        action: QtGui.QAction = getattr(main_window, action_name)
        triggered: list[bool] = []
        action.triggered.connect(lambda: triggered.append(True))
        assert action.shortcut() == QtGui.QKeySequence(keys)

        QTest.keySequence(main_window.ui, QtGui.QKeySequence(keys))

        assert triggered == [True]

    def test_page_keys_move_through_the_pages(
        self, project_window: MainWindow.MainWindowManager
    ):
        pages = project_window.pages()

        QTest.keyClick(project_window.ui, QtCore.Qt.Key.Key_PageDown)
        assert project_window.current_page == pages[1]
        QTest.keyClick(project_window.ui, QtCore.Qt.Key.Key_End)
        assert project_window.current_page == pages[2]
        QTest.keyClick(project_window.ui, QtCore.Qt.Key.Key_Q)
        assert project_window.current_page == pages[1]
        QTest.keyClick(project_window.ui, QtCore.Qt.Key.Key_W)
        assert project_window.current_page == pages[2]
        QTest.keyClick(project_window.ui, QtCore.Qt.Key.Key_Home)
        assert project_window.current_page == pages[0]
        project_window.renderer.wait_for_done()

    def test_f5_reloads_the_current_page(
        self, project_window: MainWindow.MainWindowManager
    ):
        page = project_window.current_page
        assert page is not None
        before = project_window._thumbnail_versions.get(page, 0)

        QTest.keyClick(project_window.ui, QtCore.Qt.Key.Key_F5)

        assert project_window._thumbnail_versions[page] == before + 1
        project_window.renderer.wait_for_done()

    def test_stage_letter_switches_the_stage(
        self, project_window: MainWindow.MainWindowManager
    ):
        QTest.keyClick(project_window.ui, QtCore.Qt.Key.Key_K)

        assert project_window.curFilter == FilterStage.PAGE_LAYOUT
        project_window.renderer.wait_for_done()

    def test_go_to_page_shortcut_asks_for_a_page_number_and_goes_there(
        self,
        project_window: MainWindow.MainWindowManager,
        monkeypatch: pytest.MonkeyPatch,
    ):
        asked: list[tuple] = []

        def answer(*args):
            asked.append(args)
            return 3, True

        monkeypatch.setattr(QtWidgets.QInputDialog, "getInt", answer)

        QTest.keySequence(project_window.ui, QtGui.QKeySequence("Ctrl+G"))

        assert project_window.current_page == project_window.pages()[2]
        title, label, value, minimum, maximum = asked[0][1:6]
        assert (title, label) == ("Go To Page", "Enter the page number:")
        assert (value, minimum, maximum) == (1, 1, 3)
        project_window.renderer.wait_for_done()

    def test_cancelling_go_to_page_keeps_the_current_page(
        self,
        project_window: MainWindow.MainWindowManager,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(QtWidgets.QInputDialog, "getInt", lambda *a: (2, False))
        page = project_window.current_page

        project_window.actionGotoPage.trigger()

        assert project_window.current_page == page
        project_window.renderer.wait_for_done()

    def test_navigation_is_ignored_while_a_batch_runs(
        self,
        project_window: MainWindow.MainWindowManager,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(
            type(project_window.batch_runner),
            "is_running",
            property(lambda _self: True),
        )
        page = project_window.current_page

        project_window.actionNextPage.trigger()
        project_window.actionLastPage.trigger()

        assert project_window.current_page == page
        project_window.renderer.wait_for_done()


class TestThumbnailButtons:
    """The buttons beside the thumbnail list behave like the C++ ones."""

    def test_next_and_previous_buttons_step_through_the_pages(
        self, project_window: MainWindow.MainWindowManager
    ):
        pages = project_window.pages()

        project_window.nextPageBtn.click()
        assert project_window.current_page == pages[1]
        project_window.prevPageBtn.click()
        assert project_window.current_page == pages[0]
        project_window.renderer.wait_for_done()

    def test_buttons_follow_the_selected_pages_when_filtering_is_on(
        self, project_window: MainWindow.MainWindowManager
    ):
        pages = project_window.pages()
        project_window.selectionModeBtn.setChecked(True)
        selection = project_window.thumbnails.view.selectionModel()
        for row in (0, 2):
            selection.select(
                project_window.thumbnails.model.index(row),
                QtCore.QItemSelectionModel.SelectionFlag.Select,
            )
        project_window.filterSelectedBtn.setChecked(True)

        project_window.nextPageBtn.click()

        assert project_window.current_page == pages[2]
        project_window.renderer.wait_for_done()

    def test_going_to_the_next_page_turns_selection_mode_off(
        self, project_window: MainWindow.MainWindowManager
    ):
        project_window.selectionModeBtn.setChecked(True)

        project_window.goNextPage()

        assert not project_window.selectionModeBtn.isChecked()
        assert len(project_window.thumbnails.selected_pages()) == 1
        project_window.renderer.wait_for_done()

    def test_go_to_button_opens_the_go_to_page_dialog(
        self,
        project_window: MainWindow.MainWindowManager,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(QtWidgets.QInputDialog, "getInt", lambda *a: (2, True))

        project_window.gotoPageBtn.click()

        assert project_window.current_page == project_window.pages()[1]
        project_window.renderer.wait_for_done()

    def test_magnify_shortcut_enlarges_and_stores_the_thumbnail_size(
        self, qtbot: QtBot, main_window: MainWindow.MainWindowManager
    ):
        settings = ApplicationSettings.get_instance()
        width, height = settings.max_thumbnail_width, settings.max_thumbnail_height

        QTest.keySequence(main_window.ui, QtGui.QKeySequence("Ctrl++"))

        qtbot.waitUntil(
            lambda: main_window.thumbnails.cell_size.width() == round(width + 25)
        )
        assert main_window.thumbnails.cell_size.height() == round(height + 16)
        assert ApplicationSettings.load().max_thumbnail_width == width + 25

    def test_repeated_steps_are_applied_once(
        self, qtbot: QtBot, main_window: MainWindow.MainWindowManager
    ):
        width = ApplicationSettings.get_instance().max_thumbnail_width

        main_window.magnifyThumbnailsBtn.click()
        main_window.magnifyThumbnailsBtn.click()
        main_window.diminishThumbnailsBtn.click()

        qtbot.waitUntil(
            lambda: main_window.thumbnails.cell_size.width() == round(width + 25)
        )

    def test_diminishing_stops_at_the_smallest_size(
        self, qtbot: QtBot, main_window: MainWindow.MainWindowManager
    ):
        for _ in range(40):
            main_window.diminishThumbnailsBtn.click()

        qtbot.waitUntil(lambda: main_window.thumbnails.cell_size.width() == 100)
        assert main_window.thumbnails.cell_size.height() == 64

    def test_alt_wheel_over_the_thumbnails_zooms_them(
        self, qtbot: QtBot, main_window: MainWindow.MainWindowManager
    ):
        width = ApplicationSettings.get_instance().max_thumbnail_width
        viewport = main_window.thumbnails.view.viewport()
        wheel = QtGui.QWheelEvent(
            QtCore.QPointF(5, 5),
            viewport.mapToGlobal(QtCore.QPointF(5, 5)),
            QtCore.QPoint(),
            QtCore.QPoint(0, 120),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.AltModifier,
            QtCore.Qt.ScrollPhase.NoScrollPhase,
            False,
        )

        QtWidgets.QApplication.sendEvent(viewport, wheel)

        qtbot.waitUntil(
            lambda: main_window.thumbnails.cell_size.width() == round(width + 25)
        )

    def test_column_button_switches_single_column_display_and_remembers_it(
        self, main_window: MainWindow.MainWindowManager
    ):
        main_window.thumbColumnViewBtn.click()

        assert (
            main_window.thumbnails.view.viewMode()
            == QtWidgets.QListView.ViewMode.ListMode
        )
        assert ApplicationSettings.load().single_column_thumbnail_display

        main_window.thumbColumnViewBtn.click()

        assert (
            main_window.thumbnails.view.viewMode()
            == QtWidgets.QListView.ViewMode.IconMode
        )
        assert not ApplicationSettings.load().single_column_thumbnail_display

    def test_column_button_shows_the_stored_mode_at_start(self, qtbot: QtBot):
        settings = ApplicationSettings.get_instance()
        settings.single_column_thumbnail_display = True
        settings.save()

        manager = MainWindow.MainWindowManager()
        qtbot.addWidget(manager.ui)

        assert manager.thumbColumnViewBtn.isChecked()

    def test_deviation_button_stores_the_choice_without_rendering_thumbnails(
        self, project_window: MainWindow.MainWindowManager
    ):
        epoch = project_window._thumbnail_epoch
        assert project_window.deviationHighlightingBtn.isChecked()

        project_window.deviationHighlightingBtn.click()

        assert not ApplicationSettings.load().highlight_deviation
        assert project_window._thumbnail_epoch == epoch
        project_window.renderer.wait_for_done()

    def test_the_settings_dialog_sets_the_deviation_button(
        self, main_window: MainWindow.MainWindowManager
    ):
        settings = ApplicationSettings.get_instance()
        main_window.deviationHighlightingBtn.setChecked(True)
        settings.highlight_deviation = False

        main_window._apply_thumbnail_settings(settings)

        assert not main_window.deviationHighlightingBtn.isChecked()

    def test_scrolling_the_current_page_out_of_view_stops_following_it(
        self, project_window: MainWindow.MainWindowManager, qtbot: QtBot
    ):
        project_window.thumbColumnViewBtn.click()
        project_window.thumbnails.view.setFixedHeight(120)
        project_window.focusButton.setChecked(True)
        scroll_bar = project_window.thumbnails.view.verticalScrollBar()
        qtbot.wait_until(lambda: scroll_bar.maximum() > 0)

        scroll_bar.setValue(scroll_bar.maximum())

        assert not project_window.focusButton.isChecked()
        project_window.renderer.wait_for_done()

    def test_checking_the_focus_button_scrolls_to_the_current_page(
        self,
        project_window: MainWindow.MainWindowManager,
        monkeypatch: pytest.MonkeyPatch,
    ):
        project_window.focusButton.setChecked(False)
        scrolls: list[int] = []
        monkeypatch.setattr(
            project_window.thumbnails, "scroll_to_current", lambda: scrolls.append(1)
        )

        project_window.focusButton.setChecked(True)

        assert scrolls == [1]
        project_window.renderer.wait_for_done()

    def test_keyboard_navigation_leaves_the_focus_button_off(
        self, project_window: MainWindow.MainWindowManager
    ):
        project_window.focusButton.setChecked(False)

        project_window.goNextPage()
        project_window.goLastPage()
        project_window.goFirstPage()

        assert not project_window.focusButton.isChecked()
        project_window.renderer.wait_for_done()

    def test_navigating_with_focus_off_does_not_scroll_the_thumbnail_list(
        self,
        project_window: MainWindow.MainWindowManager,
        monkeypatch: pytest.MonkeyPatch,
    ):
        project_window.focusButton.setChecked(False)
        scrolls: list[int] = []
        monkeypatch.setattr(
            project_window.thumbnails, "scroll_to_current", lambda: scrolls.append(1)
        )

        project_window.goLastPage()

        assert scrolls == []
        project_window.renderer.wait_for_done()

    def test_stepping_through_selected_pages_leaves_the_focus_button_off(
        self, project_window: MainWindow.MainWindowManager
    ):
        project_window.selectionModeBtn.setChecked(True)
        selection = project_window.thumbnails.view.selectionModel()
        for row in (0, 2):
            selection.select(
                project_window.thumbnails.model.index(row),
                QtCore.QItemSelectionModel.SelectionFlag.Select,
            )
        project_window.focusButton.setChecked(False)

        project_window.goNextSelectedPage()

        assert not project_window.focusButton.isChecked()
        project_window.renderer.wait_for_done()


def _stored_units() -> Units:
    return ApplicationSettings.load().units


class TestUnitsMenu:
    """The Units menu sets the unit every length is shown in."""

    def test_menu_lists_the_four_units_and_checks_the_current_one(
        self, main_window: MainWindow.MainWindowManager
    ):
        menu = main_window.ui.findChild(QtWidgets.QMenu, "menuUnits")

        assert [action.text() for action in menu.actions()] == [
            "Pixels",
            "Millimetres",
            "Centimetres",
            "Inches",
        ]
        assert main_window.actionMilimeters.isChecked()

    def test_menu_starts_with_the_stored_unit(self, qtbot: QtBot):
        settings = ApplicationSettings.get_instance()
        settings.units = Units.INCHES
        UnitsProvider.reset_instance()

        manager = MainWindow.MainWindowManager()
        qtbot.addWidget(manager.ui)

        assert manager.actionInches.isChecked()
        assert not manager.actionMilimeters.isChecked()

    @pytest.mark.parametrize(
        ("action_name", "units"),
        [
            ("actionPixels", Units.PIXELS),
            ("actionMilimeters", Units.MILLIMETERS),
            ("actionCentimetres", Units.CENTIMETERS),
            ("actionInches", Units.INCHES),
        ],
    )
    def test_choosing_a_unit_announces_and_stores_it(
        self,
        main_window: MainWindow.MainWindowManager,
        action_name: str,
        units: Units,
    ):
        announced: list[Units] = []
        main_window.units_provider.units_changed.connect(announced.append)
        # Millimetres are already current, so start from another unit.
        other = (
            main_window.actionInches
            if units is Units.MILLIMETERS
            else main_window.actionMilimeters
        )
        other.trigger()
        announced.clear()

        getattr(main_window, action_name).trigger()

        assert main_window.units_provider.units is units
        assert announced == [units]
        assert _stored_units() is units

    def test_a_unit_chosen_elsewhere_is_checked_in_the_menu(
        self, main_window: MainWindow.MainWindowManager
    ):
        main_window.units_provider.set_units(Units.CENTIMETERS)

        assert main_window.actionCentimetres.isChecked()
        assert not main_window.actionMilimeters.isChecked()

    def test_margins_spin_boxes_follow_the_menu_choice(
        self, project_window: MainWindow.MainWindowManager
    ):
        window = project_window
        page = window.current_page
        assert page is not None
        filter_ = window.stages.page_layout_filter
        filter_.set_params(
            page,
            filter_.get_params(page).with_margins(
                Margins(top=25.4, bottom=25.4, left=25.4, right=25.4)
            ),
        )
        window.actionSwitchFilter5.trigger()
        spin = window.ui.findChild(QtWidgets.QDoubleSpinBox, "leftMarginSpinBox")
        assert spin.value() == pytest.approx(25.4)

        window.actionInches.trigger()

        assert spin.value() == pytest.approx(1.0)

        window.actionCentimetres.trigger()

        assert spin.value() == pytest.approx(2.54)
        window.renderer.wait_for_done()


class TestStatusBar:
    """The status bar shows the page, its size and the cursor position."""

    @staticmethod
    def _wait_for_size(qtbot: QtBot, window: MainWindow.MainWindowManager) -> None:
        qtbot.waitUntil(
            lambda: window.status_panel.phys_size_label.text() != "", timeout=60_000
        )

    def test_status_bar_is_empty_without_a_project(
        self, main_window: MainWindow.MainWindowManager
    ):
        panel = main_window.status_panel

        assert not panel.page_no_label.isVisibleTo(main_window.ui)
        assert panel.page_no_label.text() == ""
        assert panel.phys_size_label.text() == ""

    def test_status_bar_shows_the_page_number_name_and_size(
        self, qtbot: QtBot, project_window: MainWindow.MainWindowManager
    ):
        panel = project_window.status_panel

        self._wait_for_size(qtbot, project_window)

        assert panel.page_no_label.text() == "p. 1 / 3"
        assert panel.page_info_label.text() == "page_001"
        # 600 x 800 pixels at 300 dpi.
        assert panel.phys_size_label.text() == "51 x 68 mm"
        project_window.renderer.wait_for_done()

    def test_status_bar_shows_the_source_dpi_of_the_page(
        self, project_window: MainWindow.MainWindowManager
    ):
        assert project_window.status_panel.source_dpi_label.text() == "300 dpi"
        project_window.renderer.wait_for_done()

    def test_status_bar_points_out_a_page_scanned_at_another_dpi(
        self, project_window: MainWindow.MainWindowManager
    ):
        project = project_window.project
        assert project is not None
        odd = project.images[-1]
        odd.metadata = odd.metadata.model_copy(update={"dpi": Dpi.uniform(600)})
        project_window._refresh_dpi_overrides()
        panel = project_window.status_panel

        assert panel.source_dpi_label.text() == "300 dpi"
        project_window.goLastPage()
        assert panel.source_dpi_label.text() == "600 dpi (most pages 300 dpi)"
        project_window.renderer.wait_for_done()

    def test_status_bar_follows_the_current_page(
        self, project_window: MainWindow.MainWindowManager
    ):
        project_window.goLastPage()

        panel = project_window.status_panel
        assert panel.page_no_label.text() == "p. 3 / 3"
        assert panel.page_info_label.text() == "page_003"
        project_window.renderer.wait_for_done()

    def test_status_bar_counts_in_the_displayed_order(
        self, project_window: MainWindow.MainWindowManager
    ):
        project_window.sortOptions.setCurrentIndex(2)
        project_window.goLastPage()

        panel = project_window.status_panel
        displayed = project_window.thumbnails.pages()
        assert panel.page_info_label.text() == displayed[-1].image_id.file_path.stem
        assert panel.page_no_label.text() == "p. 3 / 3"
        project_window.renderer.wait_for_done()

    def test_status_bar_size_follows_the_units_menu(
        self, qtbot: QtBot, project_window: MainWindow.MainWindowManager
    ):
        panel = project_window.status_panel
        self._wait_for_size(qtbot, project_window)

        project_window.actionInches.trigger()
        assert panel.phys_size_label.text() == "2 x 2.7 in"

        project_window.actionCentimetres.trigger()
        assert panel.phys_size_label.text() == "5.1 x 6.8 cm"

        project_window.actionPixels.trigger()
        assert panel.phys_size_label.text() == "600 x 800 px"

        project_window.actionMilimeters.trigger()
        assert panel.phys_size_label.text() == "51 x 68 mm"
        project_window.renderer.wait_for_done()

    def test_status_bar_shows_the_size_again_after_the_stage_changes(
        self, qtbot: QtBot, project_window: MainWindow.MainWindowManager
    ):
        self._wait_for_size(qtbot, project_window)

        project_window.actionSwitchFilter3.trigger()

        self._wait_for_size(qtbot, project_window)
        assert project_window.status_panel.phys_size_label.text() != ""
        project_window.renderer.wait_for_done()

    def test_cursor_over_the_view_shows_its_position_on_the_page(
        self, qtbot: QtBot, project_window: MainWindow.MainWindowManager
    ):
        window = project_window
        self._wait_for_size(qtbot, window)
        panels = window._panels
        assert panels is not None
        viewport = panels.view.viewport()
        centre = QtCore.QPointF(viewport.rect().center())
        move = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseMove,
            centre,
            viewport.mapToGlobal(centre),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )

        QtWidgets.QApplication.sendEvent(viewport, move)

        # The page is centred in the view: 300 x 400 pixels is 25.4 x 33.9 mm.
        label = window.status_panel.mouse_pos_label
        x, y = (float(part) for part in label.text().split(","))
        assert x == pytest.approx(25.4, abs=1.5)
        assert y == pytest.approx(33.9, abs=1.5)

        QtWidgets.QApplication.sendEvent(
            viewport, QtCore.QEvent(QtCore.QEvent.Type.Leave)
        )

        assert label.text() == ""
        window.renderer.wait_for_done()

    def test_closing_the_project_clears_the_status_bar(
        self, qtbot: QtBot, project_window: MainWindow.MainWindowManager
    ):
        self._wait_for_size(qtbot, project_window)
        project_window.modified = False

        project_window.actionCloseProject.trigger()

        panel = project_window.status_panel
        assert panel.page_no_label.text() == ""
        assert panel.phys_size_label.text() == ""


def test_options_panel_scrolls_sideways_when_its_content_is_wider(main_window):
    scroll = main_window.ui.findChild(QtWidgets.QScrollArea, "scrollArea")
    assert scroll is not None
    policy = scroll.horizontalScrollBarPolicy()
    assert policy == QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
