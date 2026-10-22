"""Upstream 1.2.1 main window and thumbnail features, and the icons they use."""

import pathlib
from pathlib import Path

import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from pytestqt.qtbot import QtBot

from scantailor.app.ui import icons
from scantailor.app.ui.dialogs.settings import SettingsDialog
from scantailor.app.ui.MainWindow import MainWindowManager
from scantailor.app.ui.thumbnails import ThumbnailListWidget
from scantailor.core.batch import output_file_names
from scantailor.core.default_params_profiles import (
    CUSTOM_PROFILE,
    ProfileStore,
    active_default_params,
)
from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.core.project import ImageMetadata, Project
from scantailor.core.settings import ApplicationSettings, ZoneCreationMode
from scantailor.core.stage_sequence import FilterStage

NEW_ICONS = (
    "pause",
    "play",
    "crosshair",
    "squares-exclude",
    "crop",
    "undo-2",
    "arrow-up-to-line",
    "arrow-down-to-line",
    "arrow-left-to-line",
    "arrow-right-to-line",
    "square-dashed",
    "lasso",
    "zone-polygon",
    "shear",
    "swap-pages",
)
WIDTHS = [900, 300, 600, 450]


@pytest.mark.parametrize("name", NEW_ICONS)
def test_new_icon_resolves_and_draws(qapp: QtWidgets.QApplication, name: str) -> None:
    assert b"<svg" in icons.svg_source(name)

    pixmap = icons.icon(name).pixmap(QtCore.QSize(24, 24))

    assert not pixmap.isNull()
    assert any(
        pixmap.toImage().pixelColor(x, y).alpha() > 0
        for x in range(24)
        for y in range(24)
    )


def _thumbnail_project() -> Project:
    project = Project()
    for number, width in enumerate(WIDTHS, start=1):
        project.add_image(
            ImageId(file_path=Path(f"page_{number}.png")),
            ImageMetadata(width=width, height=1000 - 100 * number),
        )
    return project


def _blank_thumbnail(_page: PageId, _stage: str) -> QtGui.QImage:
    image = QtGui.QImage(40, 60, QtGui.QImage.Format.Format_RGB32)
    image.fill(QtGui.QColor("steelblue"))
    return image


@pytest.fixture
def thumbnails(qtbot: QtBot) -> ThumbnailListWidget:
    widget = ThumbnailListWidget(
        _blank_thumbnail, _thumbnail_project(), stage="page_split"
    )
    qtbot.addWidget(widget)
    widget.resize(240, 700)
    widget.show()
    qtbot.wait_exposed(widget)
    return widget


class TestPausedSorting:
    def test_paused_list_shows_project_order_and_resumes_the_chosen_order(
        self, thumbnails: ThumbnailListWidget
    ) -> None:
        natural = thumbnails.pages()
        thumbnails.set_order(1)
        by_width = thumbnails.pages()
        assert by_width != natural

        thumbnails.set_sorting_paused(True)
        assert thumbnails.pages() == natural

        thumbnails.set_sorting_paused(False)
        assert thumbnails.pages() == by_width

    def test_choosing_an_order_while_paused_applies_after_resuming(
        self, thumbnails: ThumbnailListWidget
    ) -> None:
        natural = thumbnails.pages()
        thumbnails.set_sorting_paused(True)

        thumbnails.set_order(1)
        assert thumbnails.pages() == natural

        thumbnails.set_sorting_paused(False)
        assert thumbnails.pages() != natural

    def test_pausing_keeps_the_selection_and_current_page(
        self, qtbot: QtBot, thumbnails: ThumbnailListWidget
    ) -> None:
        thumbnails.set_order(1)
        target = thumbnails.pages()[0]
        thumbnails.set_current_page(target)

        thumbnails.set_sorting_paused(True)

        assert thumbnails.current_page() == target
        assert thumbnails.selected_pages() == [target]

    def test_main_window_button_pauses_and_needs_a_project(
        self, project_window: MainWindowManager
    ) -> None:
        window = project_window
        window.sortOptions.setCurrentIndex(1)
        by_width = window.thumbnails.pages()

        window.pauseSortingBtn.setChecked(True)
        assert window.thumbnails.pages() == window.pages()

        window.pauseSortingBtn.setChecked(False)
        assert window.thumbnails.pages() == by_width

        window.pauseSortingBtn.setChecked(True)
        window.modified = False
        window.closeProject()
        assert not window.pauseSortingBtn.isEnabled()
        assert not window.pauseSortingBtn.isChecked()
        window.renderer.wait_for_done()

    def test_button_has_a_pause_and_play_icon_pair(
        self, main_window: MainWindowManager
    ) -> None:
        button = main_window.pauseSortingBtn

        assert button.isCheckable()
        assert icons.is_palette_icon(button.icon())
        off = button.icon().pixmap(QtCore.QSize(16, 16), QtGui.QIcon.Mode.Normal)
        on = button.icon().pixmap(
            QtCore.QSize(16, 16), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.On
        )
        assert off.toImage() != on.toImage()


class TestReverseTwoPageOrder:
    @staticmethod
    def _split_first_image(window: MainWindowManager) -> tuple[PageId, PageId]:
        assert window.project is not None
        first = window.project.images[0].id
        window._on_page_count_requested(first, 2)
        return (
            PageId(image_id=first, sub_page=SubPage.LEFT_PAGE),
            PageId(image_id=first, sub_page=SubPage.RIGHT_PAGE),
        )

    def test_action_has_the_swap_pages_icon_and_follows_the_project(
        self, project_window: MainWindowManager
    ) -> None:
        window = project_window
        assert icons.is_palette_icon(window.actionReverseTwoPageOrder.icon())
        assert window.actionReverseTwoPageOrder.isEnabled()

        window.modified = False
        window.closeProject()

        assert not window.actionReverseTwoPageOrder.isEnabled()
        window.renderer.wait_for_done()

    def test_reversing_swaps_the_halves_in_the_page_list_and_marks_the_project(
        self, project_window: MainWindowManager
    ) -> None:
        window = project_window
        left, right = self._split_first_image(window)
        window.modified = False
        assert window.pages()[:2] == [left, right]

        window.actionReverseTwoPageOrder.trigger()

        assert window.project is not None
        assert window.project.layout_direction == "RTL"
        assert window.pages()[:2] == [right, left]
        assert window.thumbnails.pages() == window.project.get_image_pages()
        assert window.thumbnails.pages()[0] == right
        assert window.modified
        window.renderer.wait_for_done()

    def test_reversing_twice_restores_the_left_to_right_order(
        self, project_window: MainWindowManager
    ) -> None:
        window = project_window
        left, right = self._split_first_image(window)

        window.actionReverseTwoPageOrder.trigger()
        window.actionReverseTwoPageOrder.trigger()

        assert window.project is not None
        assert window.project.layout_direction == "LTR"
        assert window.pages()[:2] == [left, right]
        window.renderer.wait_for_done()

    def test_output_file_names_follow_the_new_direction(
        self, project_window: MainWindowManager
    ) -> None:
        window = project_window
        left, right = self._split_first_image(window)
        assert window.project is not None
        before = output_file_names(window.pages(), window.project.layout_direction)

        window.actionReverseTwoPageOrder.trigger()
        after = output_file_names(window.pages(), window.project.layout_direction)

        assert before[left].endswith("_1L.tif")
        assert before[right].endswith("_2R.tif")
        assert after[left].endswith("_2L.tif")
        assert after[right].endswith("_1R.tif")
        window.renderer.wait_for_done()

    def test_the_current_page_and_selection_survive(
        self, project_window: MainWindowManager
    ) -> None:
        window = project_window
        window.actionSwitchFilter3.trigger()
        left, right = self._split_first_image(window)
        window.selectionModeBtn.setChecked(True)
        window.thumbnails.view.selectionModel().select(
            window.thumbnails.model.index(1),
            QtCore.QItemSelectionModel.SelectionFlag.Select,
        )
        window.current_page = left
        selected = set(window.thumbnails.selected_pages())

        window.actionReverseTwoPageOrder.trigger()

        assert window.current_page == left
        assert set(window.thumbnails.selected_pages()) == selected
        assert right in window.pages()
        window.renderer.wait_for_done()

    def test_the_direction_is_saved_with_the_project(
        self, project_window: MainWindowManager, tmp_path: pathlib.Path
    ) -> None:
        window = project_window
        self._split_first_image(window)
        window.actionReverseTwoPageOrder.trigger()
        target = tmp_path / "rtl.ScanTailor"

        assert window._write_project(target)

        assert Project.load(target).layout_direction == "RTL"
        window.renderer.wait_for_done()


class TestRubberBandSelection:
    @staticmethod
    def _drag(
        qtbot: QtBot,
        widget: ThumbnailListWidget,
        start: QtCore.QPoint,
        end: QtCore.QPoint,
        modifier: Qt.KeyboardModifier = Qt.KeyboardModifier.ControlModifier,
    ) -> None:
        viewport = widget.view.viewport()
        left = Qt.MouseButton.LeftButton
        qtbot.mousePress(viewport, left, modifier, start)
        qtbot.mouseMove(viewport, end)
        qtbot.mouseRelease(viewport, left, modifier, end)

    @staticmethod
    def _gutter(widget: ThumbnailListWidget, page: PageId) -> QtCore.QPoint:
        """A point on the viewport beside a page's cell, where no item is."""
        rect = widget.item_rect(page)
        point = QtCore.QPoint(rect.right() + 8, rect.top() + 10)
        assert not widget.view.indexAt(point).isValid()
        return point

    def test_dragging_over_empty_space_selects_every_touched_page(
        self, qtbot: QtBot, thumbnails: ThumbnailListWidget
    ) -> None:
        pages = thumbnails.pages()
        start = self._gutter(thumbnails, pages[0])
        end = thumbnails.item_rect(pages[2]).center()

        self._drag(qtbot, thumbnails, start, end)

        assert thumbnails.selected_pages() == pages[:3]

    def test_first_touched_page_becomes_current_and_is_announced(
        self, qtbot: QtBot, thumbnails: ThumbnailListWidget
    ) -> None:
        pages = thumbnails.pages()
        start = self._gutter(thumbnails, pages[1])
        end = thumbnails.item_rect(pages[3]).center()

        with qtbot.waitSignal(thumbnails.page_selected) as announced:
            self._drag(qtbot, thumbnails, start, end)

        assert announced.args == [pages[1]]
        assert thumbnails.current_page() == pages[1]

    def test_the_existing_selection_is_extended_not_replaced(
        self, qtbot: QtBot, thumbnails: ThumbnailListWidget
    ) -> None:
        pages = thumbnails.pages()
        thumbnails.set_current_page(pages[3])
        start = self._gutter(thumbnails, pages[0])
        end = thumbnails.item_rect(pages[1]).center()

        self._drag(qtbot, thumbnails, start, end)

        assert set(thumbnails.selected_pages()) == {pages[0], pages[1], pages[3]}

    def test_a_current_page_inside_the_rectangle_stays_current(
        self, qtbot: QtBot, thumbnails: ThumbnailListWidget
    ) -> None:
        pages = thumbnails.pages()
        thumbnails.set_current_page(pages[1])
        start = self._gutter(thumbnails, pages[0])
        end = thumbnails.item_rect(pages[2]).center()

        self._drag(qtbot, thumbnails, start, end)

        assert thumbnails.current_page() == pages[1]
        assert thumbnails.selected_pages() == pages[:3]

    def test_a_tiny_drag_selects_nothing(
        self, qtbot: QtBot, thumbnails: ThumbnailListWidget
    ) -> None:
        pages = thumbnails.pages()
        start = self._gutter(thumbnails, pages[0])

        self._drag(qtbot, thumbnails, start, start + QtCore.QPoint(2, 2))

        assert thumbnails.selected_pages() == []

    def test_a_drag_starting_on_a_thumbnail_is_left_to_ctrl_click_toggling(
        self, qtbot: QtBot, thumbnails: ThumbnailListWidget
    ) -> None:
        pages = thumbnails.pages()
        viewport = thumbnails.view.viewport()

        qtbot.mouseClick(
            viewport,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.ControlModifier,
            thumbnails.item_rect(pages[0]).center(),
        )

        assert thumbnails.selected_pages() == [pages[0]]

    def test_single_selection_mode_ignores_the_rubber_band(
        self, qtbot: QtBot, thumbnails: ThumbnailListWidget
    ) -> None:
        thumbnails.set_multi_selection_enabled(False)
        pages = thumbnails.pages()
        start = self._gutter(thumbnails, pages[0])
        end = thumbnails.item_rect(pages[2]).center()

        self._drag(qtbot, thumbnails, start, end)

        assert thumbnails.selected_pages() == []


class TestDefaultZoneCreationMode:
    @staticmethod
    def _box(dialog: SettingsDialog) -> QtWidgets.QComboBox:
        box = dialog.findChild(QtWidgets.QComboBox, "zoneCreationModeBox")
        assert box is not None
        return box

    def test_the_setting_defaults_to_polygonal(self) -> None:
        assert ApplicationSettings().default_zone_creation_mode is (
            ZoneCreationMode.POLYGONAL
        )

    def test_modes_use_the_upstream_numbers(self) -> None:
        assert [int(mode) for mode in ZoneCreationMode] == [0, 1, 2]
        assert ZoneCreationMode.LASSO.name == "LASSO"

    @pytest.mark.parametrize("stored", [-1, 3, 99, "lasso", None])
    def test_an_unknown_stored_mode_falls_back_to_polygonal(
        self, stored: object
    ) -> None:
        settings = ApplicationSettings.model_validate(
            {"default_zone_creation_mode": stored}
        )

        assert settings.default_zone_creation_mode is ZoneCreationMode.POLYGONAL

    def test_the_mode_round_trips_through_the_settings_file(self) -> None:
        settings = ApplicationSettings(
            default_zone_creation_mode=ZoneCreationMode.RECTANGULAR
        )
        settings.save()

        assert ApplicationSettings.load().default_zone_creation_mode is (
            ZoneCreationMode.RECTANGULAR
        )

    def test_the_dialog_offers_the_three_modes_with_their_glyphs(
        self, qtbot: QtBot
    ) -> None:
        dialog = SettingsDialog(ApplicationSettings())
        qtbot.addWidget(dialog)
        box = self._box(dialog)

        assert [box.itemData(row) for row in range(box.count())] == list(
            ZoneCreationMode
        )
        assert all(not box.itemIcon(row).isNull() for row in range(box.count()))
        assert all(
            icons.is_palette_icon(box.itemIcon(row)) for row in range(box.count())
        )

    def test_the_dialog_shows_the_stored_mode_and_saves_the_choice(
        self, qtbot: QtBot
    ) -> None:
        settings = ApplicationSettings(
            default_zone_creation_mode=ZoneCreationMode.LASSO
        )
        dialog = SettingsDialog(settings)
        qtbot.addWidget(dialog)
        box = self._box(dialog)
        assert box.currentData() is ZoneCreationMode.LASSO

        box.setCurrentIndex(box.findData(ZoneCreationMode.RECTANGULAR))
        ok = dialog.findChild(QtWidgets.QDialogButtonBox, "buttonBox")
        assert ok is not None
        ok.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).click()

        assert settings.default_zone_creation_mode is ZoneCreationMode.RECTANGULAR
        assert ApplicationSettings.load().default_zone_creation_mode is (
            ZoneCreationMode.RECTANGULAR
        )


class TestAutoObliqueDefaultPersistence:
    @staticmethod
    def _toggle(window: MainWindowManager, checked: bool) -> None:
        window.actionSwitchFilter2.trigger()
        window.actionSwitchFilter3.trigger()
        assert window._panels is not None
        assert window._panels.stage == FilterStage.DESKEW
        box = window._panels.options.findChild(
            QtWidgets.QCheckBox, "autoObliqueCheckBox"
        )
        assert box is not None
        box.setChecked(checked)

    def test_profiles_live_beside_the_test_settings_file(
        self, tmp_path: pathlib.Path
    ) -> None:
        assert tmp_path in ProfileStore()._folder.parents

    def test_changing_it_on_the_built_in_profile_moves_to_custom(
        self, project_window: MainWindowManager
    ) -> None:
        settings = ApplicationSettings.get_instance()
        assert not active_default_params(settings).deskew.auto_oblique

        self._toggle(project_window, True)

        assert settings.current_profile == CUSTOM_PROFILE
        assert ApplicationSettings.load().current_profile == CUSTOM_PROFILE
        assert active_default_params().deskew.auto_oblique is True
        project_window.renderer.wait_for_done()

    def test_new_projects_start_from_the_remembered_choice(
        self, project_window: MainWindowManager
    ) -> None:
        self._toggle(project_window, True)
        self._toggle(project_window, False)

        assert active_default_params().deskew.auto_oblique is False
        project_window.renderer.wait_for_done()

    def test_other_defaults_of_the_profile_are_kept(
        self, project_window: MainWindowManager
    ) -> None:
        before = active_default_params()

        self._toggle(project_window, True)

        after = active_default_params()
        assert after.deskew.auto_oblique is True
        assert after.model_dump(exclude={"deskew"}) == before.model_dump(
            exclude={"deskew"}
        )
        assert after.deskew.model_dump(exclude={"auto_oblique"}) == (
            before.deskew.model_dump(exclude={"auto_oblique"})
        )
        project_window.renderer.wait_for_done()

    def test_a_saved_profile_keeps_its_file_and_remembers_the_edit(
        self, project_window: MainWindowManager
    ) -> None:
        store = ProfileStore()
        saved = active_default_params()
        store.write("mine", saved)
        settings = ApplicationSettings.get_instance()
        settings.current_profile = "mine"

        self._toggle(project_window, True)

        assert settings.current_profile == "mine"
        assert store.read("mine").deskew.auto_oblique is False
        assert store.active_params("mine").deskew.auto_oblique is True
        project_window.renderer.wait_for_done()


class TestMissingOutputFolderOnOpen:
    def test_the_relinking_dialog_lists_only_the_output_folder(
        self, tmp_path: pathlib.Path
    ) -> None:
        project = Project(output_directory=tmp_path / "gone")
        project.add_image(ImageId(file_path=tmp_path / "page.png"), ImageMetadata())
        (tmp_path / "page.png").write_bytes(b"x")

        listed = MainWindowManager._relinkable_directories(project, only_missing=True)

        assert listed == [tmp_path / "gone"]

    def test_nothing_is_listed_when_every_folder_exists(
        self, tmp_path: pathlib.Path
    ) -> None:
        (tmp_path / "out").mkdir()
        (tmp_path / "page.png").write_bytes(b"x")
        project = Project(output_directory=tmp_path / "out")
        project.add_image(ImageId(file_path=tmp_path / "page.png"), ImageMetadata())

        assert (
            MainWindowManager._relinkable_directories(project, only_missing=True) == []
        )
