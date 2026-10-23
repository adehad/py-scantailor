"""Batch processing, "Apply to...", propagation, reload and the thumbnail controls."""

import pathlib
import shutil
import threading
from collections.abc import Callable, Sequence
from typing import Any, cast

import pytest
from PySide6 import QtCore, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui import batch_runner, page_renderer
from scantailor.app.ui.dialogs.apply_to import ApplyToDialog
from scantailor.app.ui.dialogs.split_mode import SplitModeDialog
from scantailor.app.ui.image_views.select_content import SelectContentImageView
from scantailor.app.ui.MainWindow import MainWindowManager
from scantailor.core.batch import BatchResult, PageDone
from scantailor.core.models import Dpi, ImageId, Margins, PageId, SubPage
from scantailor.core.project import ImageMetadata
from scantailor.core.propagation import ApplyScope, select_pages
from scantailor.core.settings import ApplicationSettings
from scantailor.core.stage_sequence import FilterStage
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.dewarping_options import DewarpingMode
from scantailor.filters.output.params import ApplyGroup, DistortionModelSpec
from scantailor.filters.page_layout.params import ContentRect, ContentSize
from scantailor.filters.page_split.dependencies import (
    Dependencies as SplitDependencies,
)
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.page_split.page_layout import PageLayout
from scantailor.filters.page_split.params import Params as SplitParams
from scantailor.filters.select_content.content_box import ContentBox, PageBox
from scantailor.filters.select_content.params import Params as SelectContentParams

BATCH_TIMEOUT_MS = 300_000
RENDER_TIMEOUT_MS = 60_000
_BATCH_THREADS_KEY = "settings/batch_processing_threads"


def _switch_stage(window: MainWindowManager, stage: FilterStage) -> None:
    getattr(window, f"actionSwitchFilter{int(stage) + 1}").trigger()


def _answer_apply_dialog(
    scope: ApplyScope | None, options: dict[str, bool] | None = None
) -> None:
    """Answer the next apply dialog: pick ``scope`` and accept, or cancel if None."""

    def poll() -> None:
        modal = QtWidgets.QApplication.activeModalWidget()
        if not isinstance(modal, ApplyToDialog):
            QtCore.QTimer.singleShot(10, poll)
            return
        if scope is None:
            modal.reject()
            return
        modal.scope = scope
        for key, checked in (options or {}).items():
            modal._options[key].setChecked(checked)
        modal.accept()

    QtCore.QTimer.singleShot(10, poll)


def _drag_in_view(
    qtbot: QtBot,
    view: QtWidgets.QAbstractScrollArea,
    start: QtCore.QPointF,
    end: QtCore.QPointF,
) -> None:
    viewport = view.viewport()
    left = QtCore.Qt.MouseButton.LeftButton
    qtbot.mousePress(viewport, left, pos=start.toPoint())
    for step in range(1, 5):
        qtbot.mouseMove(viewport, (start + (end - start) * step / 4).toPoint())
    qtbot.mouseRelease(viewport, left, pos=end.toPoint())


def _measure_every_page(window: MainWindowManager) -> None:
    settings = window.stages.page_layout_filter.settings
    for page in window.pages():
        settings.update_content_size_and_get_aggregate(
            page,
            ContentRect(width=600, height=800),
            ContentRect(width=500, height=700),
            ContentSize(width_mm=100, height_mm=150),
        )


def _count_renders(
    window: MainWindowManager, monkeypatch: pytest.MonkeyPatch
) -> list[PageId]:
    """Record every page render, from the main view and from thumbnails alike."""
    rendered: list[PageId] = []
    real_render = page_renderer.render_page_with_model

    def spy(page_id, sequence, end_stage, dpi_overrides=None, **kwargs):
        rendered.append(page_id)
        return real_render(page_id, sequence, end_stage, **kwargs)

    monkeypatch.setattr(page_renderer, "render_page_with_model", spy)
    return rendered


@pytest.fixture
def settled_window(
    qtbot: QtBot, project_window: MainWindowManager
) -> MainWindowManager:
    """A project window whose first render has been delivered."""
    project_window.renderer.wait_for_done()
    qtbot.wait(50)
    return project_window


class TestGeneratedDistortionModel:
    """A model that a render built is kept in the page's output parameters."""

    @staticmethod
    def _edges() -> DistortionModelSpec:
        return DistortionModelSpec(
            top=((0.0, 10.0), (50.0, 5.0), (100.0, 10.0)),
            bottom=((0.0, 90.0), (50.0, 85.0), (100.0, 90.0)),
        )

    def _set_mode(self, window: MainWindowManager, mode: DewarpingMode) -> PageId:
        page = window.pages()[0]
        output = window.stages.output_filter
        params = output.get_params(page)
        output.set_params(page, params.with_dewarping(params.dewarping.with_mode(mode)))
        return page

    def test_a_model_from_the_renderer_is_stored_and_marks_the_project_modified(
        self, settled_window: MainWindowManager
    ):
        window = settled_window
        page = self._set_mode(window, DewarpingMode.AUTO)
        window.modified = False

        window.renderer.distortion_model_found.emit(page, self._edges())

        stored = window.stages.output_filter.get_params(page)
        assert stored.distortion_model == self._edges()
        assert window.modified

    def test_a_model_for_a_page_switched_to_manual_meanwhile_is_dropped(
        self, settled_window: MainWindowManager
    ):
        window = settled_window
        page = self._set_mode(window, DewarpingMode.MANUAL)
        window.modified = False

        window.renderer.distortion_model_found.emit(page, self._edges())

        stored = window.stages.output_filter.get_params(page)
        assert not stored.distortion_model.is_defined()
        assert not window.modified


class TestBatchScope:
    """A batch covers every page in display order, wherever the current page is."""

    @pytest.mark.parametrize("order_index", [0, 1, 2])
    def test_batch_covers_all_pages_from_the_last_page_in_any_order(
        self,
        settled_window: MainWindowManager,
        monkeypatch: pytest.MonkeyPatch,
        order_index: int,
    ):
        window = settled_window
        window.thumbnails.set_order(order_index)
        window.goLastPage()
        started: list[list] = []
        monkeypatch.setattr(
            window, "_start_batch", lambda pages, _stage: started.append(pages)
        )

        window.startBatchProcessing()

        assert started == [window.thumbnails.pages()]
        assert len(started[0]) == len(window.pages())

    def test_starting_a_batch_turns_the_focus_button_on(
        self, settled_window: MainWindowManager, monkeypatch: pytest.MonkeyPatch
    ):
        window = settled_window
        window.focusButton.setChecked(False)
        monkeypatch.setattr(window, "_start_batch", lambda _pages, _stage: None)

        window.startBatchProcessing()

        assert window.focusButton.isChecked()


class TestBatchProcessing:
    """Batch runs on a worker thread and reports through the window."""

    @pytest.mark.slow
    def test_batch_writes_every_page_and_reports_each_one(
        self, qtbot: QtBot, project_window: MainWindowManager, tmp_path: pathlib.Path
    ):
        project_window.settings.setValue(_BATCH_THREADS_KEY, 1)
        project_window._select_stage_row(int(FilterStage.OUTPUT))
        ticks: list[int] = []
        timer = QtCore.QTimer()
        timer.timeout.connect(lambda: ticks.append(1))
        timer.start(20)

        with qtbot.waitSignal(
            project_window.batch_runner.finished, timeout=BATCH_TIMEOUT_MS
        ):
            project_window.startBatchProcessing()
            assert project_window.batch_runner.is_running
        timer.stop()

        result = project_window.last_batch_result
        assert result is not None
        assert len(result.written) == 3
        assert not result.failed
        assert sorted(path.name for path in (tmp_path / "out").iterdir()) == [
            "page_001.tif",
            "page_002.tif",
            "page_003.tif",
        ]
        dialog = project_window.batch_dialog
        assert dialog is not None
        assert dialog.reported == 3
        assert not dialog.is_running
        assert ticks, "the event loop kept running during the batch"

    @pytest.mark.slow
    def test_ui_is_locked_during_the_batch_and_free_afterwards(
        self, qtbot: QtBot, project_window: MainWindowManager
    ):
        project_window.settings.setValue(_BATCH_THREADS_KEY, 1)

        with qtbot.waitSignal(
            project_window.batch_runner.finished, timeout=BATCH_TIMEOUT_MS
        ):
            project_window.startBatchProcessing()
            assert not project_window.filterList.isEnabled()
            assert not project_window.filterOptions.isEnabled()
            assert not project_window.imageViewFrame.isEnabled()
            assert project_window.processing_indicator.is_active

        assert project_window.filterList.isEnabled()
        assert project_window.filterOptions.isEnabled()
        assert project_window.imageViewFrame.isEnabled()
        assert project_window.modified

    def test_a_box_dragged_in_the_view_during_a_batch_is_not_taken(
        self, qtbot: QtBot, settled_window: MainWindowManager, monkeypatch
    ):
        window = settled_window
        _switch_stage(window, FilterStage.SELECT_CONTENT)
        window.renderer.wait_for_done()
        qtbot.wait(50)
        page = window.current_page
        assert page is not None
        stored_before = window.stages.select_content_filter.get_params(page)
        view = cast(SelectContentImageView, window._panels.view)
        view.set_content_rect(QtCore.QRectF(100, 200, 300, 400))
        edits: list[QtCore.QRectF] = []
        view.manual_content_rect_set.connect(edits.append)
        batch_started = threading.Event()
        release_batch = threading.Event()

        def fake_run_batch(stages, pages, output_dir, **_kwargs):
            batch_started.set()
            release_batch.wait(10)
            return BatchResult()

        monkeypatch.setattr(batch_runner, "run_batch", fake_run_batch)

        with qtbot.waitSignal(window.batch_runner.finished, timeout=20_000):
            window.startBatchProcessing()
            qtbot.waitUntil(batch_started.is_set, timeout=10_000)
            edge = view.virtual_to_widget_point(QtCore.QPointF(100, 400))
            _drag_in_view(qtbot, view, edge, edge + QtCore.QPointF(-12, 7))
            release_batch.set()

        assert edits == []
        assert window.stages.select_content_filter.get_params(page) == stored_before
        assert view.isEnabled()

    def test_batch_covers_every_page_even_when_started_from_the_second(
        self, qtbot: QtBot, project_window: MainWindowManager, monkeypatch
    ):
        seen: list[Sequence[PageId]] = []

        def fake_run_batch(stages, pages, output_dir, **_kwargs):
            seen.append(list(pages))
            return BatchResult()

        monkeypatch.setattr(batch_runner, "run_batch", fake_run_batch)
        project_window.goNextPage()

        with qtbot.waitSignal(project_window.batch_runner.finished, timeout=10_000):
            project_window.startBatchProcessing()

        assert seen == [project_window.pages()]

    def test_thumbnails_are_invalidated_for_each_finished_page(
        self, qtbot: QtBot, project_window: MainWindowManager, monkeypatch
    ):
        pages = project_window.pages()

        def fake_run_batch(stages, pages, output_dir, *, on_progress, **_kwargs):
            result = BatchResult()
            for page in pages:
                result.written[page] = output_dir / "x.tif"
                on_progress(PageDone(page, output_dir / "x.tif"))
            return result

        monkeypatch.setattr(batch_runner, "run_batch", fake_run_batch)
        before = dict(project_window._thumbnail_versions)

        with qtbot.waitSignal(project_window.batch_runner.finished, timeout=10_000):
            project_window.startBatchProcessing()

        for page in pages:
            assert project_window._thumbnail_versions[page] > before.get(page, 0)

    def test_stop_cancels_pages_that_have_not_started(
        self, qtbot: QtBot, project_window: MainWindowManager, monkeypatch
    ):
        first_done = threading.Event()

        def fake_run_batch(stages, pages, output_dir, *, on_progress, cancel, **_kw):
            result = BatchResult()
            for index, page in enumerate(pages):
                if cancel.is_set():
                    result.cancelled.extend(pages[index:])
                    break
                result.written[page] = output_dir / f"{index}.tif"
                on_progress(PageDone(page, result.written[page]))
                first_done.set()
                cancel.wait(10)
            return result

        monkeypatch.setattr(batch_runner, "run_batch", fake_run_batch)

        with qtbot.waitSignal(
            project_window.batch_runner.finished, timeout=20_000
        ) as finished:
            project_window.startBatchProcessing()
            qtbot.waitUntil(first_done.is_set, timeout=10_000)
            project_window.stopBatchProcessing()

        result = finished.args[0]
        assert len(result.written) == 1
        assert len(result.cancelled) == 2
        assert project_window.filterList.isEnabled()
        assert project_window.batch_dialog is not None
        assert not project_window.batch_dialog.is_running

    def test_stop_button_of_the_progress_dialog_stops_the_batch(
        self, qtbot: QtBot, project_window: MainWindowManager, monkeypatch
    ):
        started = threading.Event()

        def fake_run_batch(stages, pages, output_dir, *, cancel, **_kwargs):
            started.set()
            cancel.wait(10)
            return BatchResult(cancelled=list(pages))

        monkeypatch.setattr(batch_runner, "run_batch", fake_run_batch)

        with qtbot.waitSignal(
            project_window.batch_runner.finished, timeout=20_000
        ) as finished:
            project_window.startBatchProcessing()
            qtbot.waitUntil(started.is_set, timeout=10_000)
            dialog = project_window.batch_dialog
            assert dialog is not None
            button = dialog.findChild(QtWidgets.QPushButton)
            assert button is not None
            qtbot.mouseClick(button, QtCore.Qt.MouseButton.LeftButton)

        assert len(finished.args[0].cancelled) == 3

    def test_closing_the_project_stops_a_running_batch(
        self, qtbot: QtBot, project_window: MainWindowManager, monkeypatch
    ):
        started = threading.Event()

        def fake_run_batch(stages, pages, output_dir, *, cancel, **_kwargs):
            started.set()
            cancel.wait(10)
            return BatchResult(cancelled=list(pages))

        monkeypatch.setattr(batch_runner, "run_batch", fake_run_batch)
        project_window.startBatchProcessing()
        qtbot.waitUntil(started.is_set, timeout=10_000)
        project_window.modified = False

        project_window.closeProject()

        assert not project_window.batch_runner.is_running
        assert project_window.project is None

    def test_failure_of_the_whole_batch_is_reported_and_unlocks_the_ui(
        self, qtbot: QtBot, project_window: MainWindowManager, monkeypatch
    ):
        def explode(*_args, **_kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(batch_runner, "run_batch", explode)
        shown: list[str] = []
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "warning",
            lambda *args, **_kwargs: shown.append(args[2]),
        )

        with qtbot.waitSignal(project_window.batch_runner.failed, timeout=10_000):
            project_window.startBatchProcessing()

        assert project_window.filterList.isEnabled()
        assert any("boom" in text for text in shown)

    def test_launch_button_of_the_stage_list_starts_the_batch(
        self, qtbot: QtBot, project_window: MainWindowManager, monkeypatch
    ):
        monkeypatch.setattr(batch_runner, "run_batch", lambda *_a, **_kw: BatchResult())

        with qtbot.waitSignal(project_window.batch_runner.finished, timeout=10_000):
            project_window.filterList.launch_batch_processing.emit()

    def test_batch_without_a_project_does_nothing(self, main_window: MainWindowManager):
        main_window.startBatchProcessing()

        assert not main_window.batch_runner.is_running
        assert main_window.batch_dialog is None


class TestApplyTo:
    """The "Apply to..." flow copies the current page's settings."""

    def test_applying_deskew_to_all_pages_copies_the_angle_to_each(
        self, settled_window: MainWindowManager
    ):
        window = settled_window
        _switch_stage(window, FilterStage.DESKEW)
        current = window.current_page
        assert current is not None
        window.stages.deskew_filter.set_manual_angle(current, 1.5)
        button = window.filterOptions.findChild(QtWidgets.QPushButton, "applyDeskewBtn")
        assert button is not None

        _answer_apply_dialog(ApplyScope.ALL_PAGES)
        button.click()

        for page in window.pages():
            params = window.stages.deskew_filter.get_params(page)
            assert params.deskew_angle_deg == pytest.approx(1.5)
            assert params.is_manual()
        assert window.modified

    def test_cancelling_the_dialog_changes_nothing(
        self, settled_window: MainWindowManager
    ):
        window = settled_window
        _switch_stage(window, FilterStage.DESKEW)
        current = window.current_page
        assert current is not None
        window.stages.deskew_filter.set_manual_angle(current, 1.5)
        window.modified = False

        _answer_apply_dialog(None)
        assert not window.show_apply_dialog(FilterStage.DESKEW)

        other = window.pages()[1]
        assert not window.stages.deskew_filter.get_params(other).is_manual()
        assert not window.modified

    def test_this_page_only_changes_nothing(self, settled_window: MainWindowManager):
        window = settled_window
        _switch_stage(window, FilterStage.DESKEW)

        _answer_apply_dialog(ApplyScope.THIS_PAGE)

        assert not window.show_apply_dialog(FilterStage.DESKEW)

    def test_every_other_page_reaches_pages_of_the_same_parity(
        self, settled_window: MainWindowManager
    ):
        window = settled_window
        _switch_stage(window, FilterStage.DESKEW)
        pages = window.pages()
        current = window.current_page
        assert current is not None
        window.stages.deskew_filter.set_manual_angle(current, -0.5)

        _answer_apply_dialog(ApplyScope.EVERY_OTHER)
        window.show_apply_dialog(FilterStage.DESKEW)

        deskew = window.stages.deskew_filter
        assert deskew.get_params(pages[2]).is_manual()
        assert not deskew.get_params(pages[1]).is_manual()

    def test_selected_pages_scope_uses_the_thumbnail_selection(
        self, settled_window: MainWindowManager
    ):
        window = settled_window
        _switch_stage(window, FilterStage.DESKEW)
        pages = window.pages()
        window.selectionModeBtn.setChecked(True)
        window.thumbnails.view.selectionModel().select(
            window.thumbnails.model.index(1),
            QtCore.QItemSelectionModel.SelectionFlag.Select,
        )
        window.thumbnails.view.selectionModel().select(
            window.thumbnails.model.index(2),
            QtCore.QItemSelectionModel.SelectionFlag.Select,
        )
        current = window.current_page
        assert current is not None
        window.stages.deskew_filter.set_manual_angle(current, 2.0)

        _answer_apply_dialog(ApplyScope.SELECTED_PAGES)
        window.show_apply_dialog(FilterStage.DESKEW)

        settings = window.stages.deskew_filter.settings
        assert settings.is_params_set(pages[1])
        assert settings.is_params_set(pages[2])

    def test_orientation_is_applied_per_image(self, settled_window: MainWindowManager):
        window = settled_window
        pages = window.pages()
        current = window.current_page
        assert current is not None
        window.stages.fix_orientation_filter.rotate_clockwise(current.image_id)

        _answer_apply_dialog(ApplyScope.ALL_PAGES)
        window.show_apply_dialog(FilterStage.FIX_ORIENTATION)

        for page in pages:
            assert (
                window.stages.fix_orientation_filter.get_rotation(page.image_id).degrees
                == 90
            )

    def test_select_content_options_decide_whether_manual_boxes_are_copied(
        self, settled_window: MainWindowManager
    ):
        window = settled_window
        _switch_stage(window, FilterStage.SELECT_CONTENT)
        pages = window.pages()
        current = window.current_page
        assert current is not None
        content_filter = window.stages.select_content_filter
        old_box = ContentBox(x=5, y=5, width=50, height=50)
        content_filter.set_params(pages[1], SelectContentParams(content_box=old_box))
        content_filter.set_params(
            current,
            SelectContentParams(
                page_box=PageBox(x=0, y=0, width=600, height=800)
            ).with_manual_content(ContentBox(x=10, y=20, width=300, height=400)),
        )

        _answer_apply_dialog(ApplyScope.ALL_PAGES, {"content_box": False})
        window.show_apply_dialog(FilterStage.SELECT_CONTENT)

        assert content_filter.get_params(pages[1]).content_box == old_box
        assert content_filter.get_params(pages[2]).is_content_manual()

    def test_margins_and_alignment_have_separate_requests(
        self, settled_window: MainWindowManager
    ):
        window = settled_window
        _switch_stage(window, FilterStage.PAGE_LAYOUT)
        pages = window.pages()
        current = window.current_page
        assert current is not None
        layout = window.stages.page_layout_filter.settings
        layout.set_margins(current, Margins.uniform(12.0))
        button = window.filterOptions.findChild(
            QtWidgets.QPushButton, "applyMarginsBtn"
        )
        assert button is not None

        _answer_apply_dialog(ApplyScope.ALL_PAGES)
        button.click()

        assert layout.get_params(pages[2]).hard_margins_mm == Margins.uniform(12.0)

    def test_output_parameters_are_applied_through_the_button(
        self, settled_window: MainWindowManager
    ):
        window = settled_window
        _switch_stage(window, FilterStage.OUTPUT)
        pages = window.pages()
        current = window.current_page
        assert current is not None
        dpi_params = window.stages.output_filter.get_params(current).with_output_dpi(
            Dpi.uniform(400)
        )
        window.stages.output_filter.set_params(current, dpi_params)

        _answer_apply_dialog(ApplyScope.ALL_PAGES)
        window.show_apply_dialog(FilterStage.OUTPUT, part=ApplyGroup.DPI.value)

        assert window.stages.output_filter.get_params(
            pages[2]
        ).output_dpi == Dpi.uniform(400)

    def test_output_group_button_keeps_target_zones_and_dewarp_model(
        self, settled_window: MainWindowManager
    ):
        window = settled_window
        _switch_stage(window, FilterStage.OUTPUT)
        pages = window.pages()
        current = window.current_page
        assert current is not None
        target = pages[2] if pages[2] != current else pages[3]
        output = window.stages.output_filter
        target_before = output.get_params(target)
        output.set_params(
            current, output.get_params(current).with_output_dpi(Dpi.uniform(400))
        )

        _answer_apply_dialog(ApplyScope.ALL_PAGES)
        window.show_apply_dialog(FilterStage.OUTPUT, part=ApplyGroup.DPI.value)

        after = output.get_params(target)
        assert after.output_dpi == Dpi.uniform(400)
        assert after.with_output_dpi(target_before.output_dpi) == target_before

    @pytest.mark.parametrize(
        "scope", [ApplyScope.ALL_PAGES, ApplyScope.THIS_PAGE_EVERY_OTHER]
    )
    def test_processing_button_copies_only_the_polarity_as_chosen_by_hand(
        self, settled_window: MainWindowManager, scope: ApplyScope, qtbot: QtBot
    ):
        window = settled_window
        _switch_stage(window, FilterStage.OUTPUT)
        pages = window.pages()
        current = window.current_page
        assert current is not None
        output = window.stages.output_filter
        output.set_params(
            current, output.get_params(current).with_color_mode(ColorMode.MIXED)
        )
        polarity = window.filterOptions.findChild(QtWidgets.QCheckBox, "blackOnWhiteCB")
        assert polarity is not None and polarity.isChecked()
        polarity.click()
        # The renders started so far measure the printed pages as black on
        # white; they all end before Apply, as a loaded machine can make them.
        qtbot.waitUntil(lambda: window.renderer.wait_for_done(0), timeout=60_000)
        qtbot.wait(50)
        before = {page: output.get_params(page) for page in pages}
        button = window.filterOptions.findChild(
            QtWidgets.QPushButton, "apply_processing_btn"
        )
        assert button is not None

        _answer_apply_dialog(scope)
        button.click()
        targets = set(select_pages(scope, current, pages)) - {current}
        # A render that ends after Apply measures the targets too.
        for page in targets:
            output.settings.set_detected_black_on_white(page, True)

        assert targets
        assert output.get_params(current).black_on_white is False
        for page in pages:
            after = output.get_params(page)
            if page in targets:
                assert after.black_on_white is False
                assert after.black_on_white_set_manually
                assert after.color_mode == before[page].color_mode
            elif page != current:
                assert after == before[page]

    def test_colour_button_leaves_the_polarity_of_other_pages(
        self, settled_window: MainWindowManager
    ):
        window = settled_window
        _switch_stage(window, FilterStage.OUTPUT)
        current = window.current_page
        assert current is not None
        output = window.stages.output_filter
        output.set_params(
            current,
            output.get_params(current)
            .with_manual_black_on_white(False)
            .with_color_mode(ColorMode.MIXED),
        )

        _answer_apply_dialog(ApplyScope.ALL_PAGES)
        window.show_apply_dialog(FilterStage.OUTPUT, part=ApplyGroup.COLOR_MODE.value)

        for page in window.pages():
            if page != current:
                after = output.get_params(page)
                assert after.color_mode == ColorMode.MIXED
                assert after.black_on_white is True
                assert not after.black_on_white_set_manually

    def test_applying_refreshes_the_thumbnails_of_all_pages(
        self, settled_window: MainWindowManager
    ):
        window = settled_window
        _switch_stage(window, FilterStage.DESKEW)
        epoch = window._thumbnail_epoch

        window.apply_to_pages(FilterStage.DESKEW, window.pages()[1:])

        assert window._thumbnail_epoch == epoch + 1


class TestApplyToDialog:
    """The dialog's own behaviour."""

    def test_selection_scopes_are_disabled_without_a_multi_selection(
        self, project_window: MainWindowManager
    ):
        pages = project_window.pages()

        dialog = ApplyToDialog(pages[0], pages, [pages[0]])

        assert not dialog.findChild(
            QtWidgets.QRadioButton, "selectedPagesRB"
        ).isEnabled()
        assert not dialog.findChild(
            QtWidgets.QRadioButton, "everyOtherSelectedRB"
        ).isEnabled()
        assert dialog.findChild(QtWidgets.QRadioButton, "allPagesRB").isEnabled()

    def test_selection_scopes_are_enabled_with_two_selected_pages(
        self, project_window: MainWindowManager
    ):
        pages = project_window.pages()

        dialog = ApplyToDialog(pages[0], pages, pages[:2])

        assert dialog.findChild(QtWidgets.QRadioButton, "selectedPagesRB").isEnabled()

    def test_default_scope_is_this_page(self, project_window: MainWindowManager):
        pages = project_window.pages()

        dialog = ApplyToDialog(pages[1], pages)

        assert dialog.scope is ApplyScope.THIS_PAGE
        assert dialog.pages() == [pages[1]]

    def test_extra_options_start_checked_and_can_be_cleared(
        self, project_window: MainWindowManager
    ):
        pages = project_window.pages()
        dialog = ApplyToDialog(pages[0], pages, options={"flag": "A flag"})

        assert dialog.option_checked("flag")
        dialog._options["flag"].setChecked(False)
        assert not dialog.option_checked("flag")


class TestImageListStages:
    """Fix Orientation and Page Split list and count images, not pages."""

    @pytest.fixture
    def spread_window(self, settled_window: MainWindowManager) -> MainWindowManager:
        """The three scans, each split into two pages, shown at Fix Orientation."""
        window = settled_window
        window.apply_to_pages(
            FilterStage.PAGE_SPLIT, window.pages(), layout_type=LayoutType.TWO_PAGES
        )
        window.renderer.wait_for_done()
        assert len(window.pages()) == 6
        return window

    @staticmethod
    def _images(window: MainWindowManager) -> list[ImageId]:
        assert window.project is not None
        return [info.id for info in window.project.images]

    def test_each_scan_has_one_thumbnail_until_the_split(
        self, spread_window: MainWindowManager
    ):
        window = spread_window
        assert window.project is not None

        at_fix_orientation = window.thumbnails.pages()
        _switch_stage(window, FilterStage.PAGE_SPLIT)
        at_page_split = window.thumbnails.pages()
        _switch_stage(window, FilterStage.DESKEW)

        assert at_fix_orientation == window.project.get_image_pages()
        assert at_page_split == window.project.get_image_pages()
        assert window.thumbnails.pages() == window.pages()
        window.renderer.wait_for_done()

    def test_the_right_page_stays_current_through_page_split(
        self, spread_window: MainWindowManager
    ):
        window = spread_window
        _switch_stage(window, FilterStage.DESKEW)
        right = window.pages()[3]
        window.page_selected(right)

        _switch_stage(window, FilterStage.PAGE_SPLIT)
        window.page_selected(window.thumbnails.pages()[1])
        at_page_split = window.current_page
        _switch_stage(window, FilterStage.DESKEW)

        assert at_page_split == right
        assert window.current_page == right
        window.renderer.wait_for_done()

    def test_every_other_page_at_fix_orientation_counts_images(
        self, spread_window: MainWindowManager
    ):
        window = spread_window
        orientation = window.stages.fix_orientation_filter
        first, second, third = self._images(window)
        orientation.rotate_clockwise(first)

        _answer_apply_dialog(ApplyScope.EVERY_OTHER)
        window.show_apply_dialog(FilterStage.FIX_ORIENTATION)

        assert orientation.get_rotation(third).degrees == 90
        assert orientation.get_rotation(second).degrees == 0

    def test_every_other_page_at_page_split_counts_images(
        self, spread_window: MainWindowManager
    ):
        window = spread_window
        _switch_stage(window, FilterStage.PAGE_SPLIT)
        split = window.stages.page_split_filter
        current = window.current_page
        assert current is not None
        split.set_params(
            current,
            split.get_params(current).with_manual_layout(
                PageLayout.two_pages(600, 800, 300)
            ),
        )

        def answer() -> None:
            modal = QtWidgets.QApplication.activeModalWidget()
            if not isinstance(modal, SplitModeDialog):
                QtCore.QTimer.singleShot(10, answer)
                return
            modal._mode_auto.setChecked(True)
            modal._every_other_rb.setChecked(True)
            modal._on_submit()

        QtCore.QTimer.singleShot(10, answer)
        window.show_apply_dialog(FilterStage.PAGE_SPLIT)

        layout_types = [
            split.get_layout_type(PageId(image_id=image))
            for image in self._images(window)
        ]
        assert layout_types == [
            LayoutType.AUTO_LAYOUT_TYPE,
            LayoutType.TWO_PAGES,
            LayoutType.AUTO_LAYOUT_TYPE,
        ]
        window.renderer.wait_for_done()

    def test_a_page_split_thumbnail_shows_the_whole_scan_and_detects_its_layout(
        self, spread_window: MainWindowManager
    ):
        window = spread_window
        left = window.pages()[0]
        split = window.stages.page_split_filter
        assert split.get_params(left).page_layout is None

        thumbnail = window._render_page_thumbnail(
            left, str(int(FilterStage.PAGE_SPLIT))
        )

        assert thumbnail.width() / thumbnail.height() == pytest.approx(
            600 / 800, abs=0.02
        )
        assert split.get_params(left).page_layout is not None

    def test_a_batch_at_page_split_runs_every_page(
        self, spread_window: MainWindowManager
    ):
        window = spread_window
        _switch_stage(window, FilterStage.PAGE_SPLIT)

        assert window._batch_pages() == window.pages()
        window.renderer.wait_for_done()


class TestPropagationOnStageChange:
    """Leaving a stage passes its results on, as the C++ main window does."""

    def _make_pages_landscape(self, window: MainWindowManager) -> None:
        window.renderer.wait_for_done()
        assert window.project is not None
        for info in window.project.images:
            info.metadata = ImageMetadata(width=2000, height=1000, dpi=Dpi.uniform(300))

    def test_leaving_fix_orientation_splits_landscape_images_into_two_pages(
        self, settled_window: MainWindowManager
    ):
        window = settled_window
        self._make_pages_landscape(window)
        window.modified = False

        _switch_stage(window, FilterStage.PAGE_SPLIT)

        assert len(window.pages()) == 6
        assert len(window.thumbnails.pages()) == 3
        assert window.modified
        assert window.current_page in window.pages()
        window.renderer.wait_for_done()

    def test_rotating_before_leaving_keeps_single_pages(
        self, settled_window: MainWindowManager
    ):
        window = settled_window
        self._make_pages_landscape(window)
        for page in window.pages():
            window.stages.fix_orientation_filter.rotate_clockwise(page.image_id)

        _switch_stage(window, FilterStage.PAGE_SPLIT)

        assert len(window.pages()) == 3
        window.renderer.wait_for_done()

    def test_moving_between_later_stages_does_not_touch_the_page_list(
        self, settled_window: MainWindowManager
    ):
        window = settled_window
        _switch_stage(window, FilterStage.PAGE_SPLIT)
        self._make_pages_landscape(window)

        _switch_stage(window, FilterStage.DESKEW)

        assert len(window.pages()) == 3
        window.renderer.wait_for_done()

    def test_leaving_select_content_records_measured_pages_for_the_layout(
        self, settled_window: MainWindowManager
    ):
        window = settled_window
        _switch_stage(window, FilterStage.SELECT_CONTENT)
        page = window.current_page
        assert page is not None
        window.stages.select_content_filter.set_params(
            page,
            SelectContentParams(
                content_box=ContentBox(x=50, y=50, width=400, height=600),
                page_box=PageBox(x=0, y=0, width=600, height=800),
            ),
        )

        _switch_stage(window, FilterStage.PAGE_LAYOUT)

        params = window.stages.page_layout_filter.get_params(page)
        assert params.content_rect.width == pytest.approx(400)
        assert not params.content_size_mm.is_empty()
        window.renderer.wait_for_done()


class TestOutputReadiness:
    """Output waits until every page size is known."""

    def test_output_with_undefined_layout_shows_the_not_ready_message(
        self, settled_window: MainWindowManager
    ):
        window = settled_window

        _switch_stage(window, FilterStage.OUTPUT)

        assert window.imageFrameLayout.currentWidget() is window.not_ready_widget
        assert not window._panels.options.isVisibleTo(window.ui)
        assert "not yet possible" in window.not_ready_widget.text()
        assert not window.processing_indicator.is_active

    def test_output_does_not_start_a_render_while_not_ready(
        self, qtbot: QtBot, settled_window: MainWindowManager, monkeypatch
    ):
        window = settled_window
        requests: list[PageId] = []
        monkeypatch.setattr(
            window.renderer, "request", lambda page, *_a: requests.append(page)
        )

        _switch_stage(window, FilterStage.OUTPUT)

        assert requests == []

    @pytest.mark.slow
    def test_output_renders_once_every_page_is_measured(
        self, qtbot: QtBot, settled_window: MainWindowManager
    ):
        window = settled_window
        _switch_stage(window, FilterStage.OUTPUT)
        _measure_every_page(window)

        with qtbot.waitSignal(window.renderer.render_done, timeout=RENDER_TIMEOUT_MS):
            window._request_render()

        assert window.imageFrameLayout.currentWidget() is window._panels.view
        assert window._panels.options.isVisibleTo(window.ui)

    @pytest.mark.slow
    def test_current_page_does_not_need_to_be_measured(
        self, qtbot: QtBot, settled_window: MainWindowManager
    ):
        window = settled_window
        _switch_stage(window, FilterStage.OUTPUT)
        _measure_every_page(window)
        current = window.current_page
        assert current is not None
        window.stages.page_layout_filter.settings.clear_page(current)

        with qtbot.waitSignal(window.renderer.render_done, timeout=RENDER_TIMEOUT_MS):
            window._request_render()

        assert window.imageFrameLayout.currentWidget() is window._panels.view

    def test_leaving_output_restores_the_options_and_view(
        self, settled_window: MainWindowManager
    ):
        window = settled_window
        _switch_stage(window, FilterStage.OUTPUT)

        _switch_stage(window, FilterStage.DESKEW)

        assert window.imageFrameLayout.currentWidget() is window._panels.view
        assert window._panels.options.isVisibleTo(window.ui)
        window.renderer.wait_for_done()


class TestReloadAndIndicator:
    """Reloading renders again; the indicator shows while a render runs."""

    def test_reload_renders_the_current_page_again(
        self,
        qtbot: QtBot,
        settled_window: MainWindowManager,
        monkeypatch: pytest.MonkeyPatch,
        wait_for_thumbnails: Callable[[MainWindowManager], None],
    ):
        window = settled_window
        wait_for_thumbnails(window)
        delivered: list[PageId] = []
        window.renderer.render_done.connect(lambda page, _image: delivered.append(page))
        rendered = _count_renders(window, monkeypatch)

        with qtbot.waitSignal(window.renderer.render_done, timeout=RENDER_TIMEOUT_MS):
            window.actionReloadPage.trigger()

        assert delivered == [window.current_page]
        assert window.current_page in rendered
        wait_for_thumbnails(window)

    def test_reload_refreshes_the_thumbnail_of_the_page(
        self, settled_window: MainWindowManager
    ):
        window = settled_window
        page = window.current_page
        assert page is not None
        before = window._thumbnail_versions.get(page, 0)

        window.reloadCurrentPage()

        assert window._thumbnail_versions[page] == before + 1
        window.renderer.wait_for_done()

    def test_reload_without_a_project_does_nothing(
        self, main_window: MainWindowManager
    ):
        main_window.reloadCurrentPage()

        assert not main_window.processing_indicator.is_active

    def test_indicator_shows_while_a_render_runs(
        self, qtbot: QtBot, settled_window: MainWindowManager
    ):
        window = settled_window
        assert not window.processing_indicator.is_active

        with qtbot.waitSignal(window.renderer.render_done, timeout=RENDER_TIMEOUT_MS):
            window.reloadCurrentPage()
            assert window.processing_indicator.is_active

        assert not window.processing_indicator.is_active

    def test_indicator_hides_when_a_render_fails(
        self, qtbot: QtBot, settled_window: MainWindowManager, monkeypatch
    ):
        window = settled_window

        def explode(*_args, **_kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(page_renderer, "render_page_with_model", explode)

        with qtbot.waitSignal(window.renderer.render_failed, timeout=RENDER_TIMEOUT_MS):
            window.reloadCurrentPage()
            assert window.processing_indicator.is_active

        assert not window.processing_indicator.is_active


class TestThumbnailControls:
    """The focus, selection mode and sort controls drive the thumbnail list."""

    def test_sort_options_mirror_the_thumbnail_orders(
        self, main_window: MainWindowManager
    ):
        combo = main_window.thumbnails.order_combo
        items = [
            main_window.sortOptions.itemText(i)
            for i in range(main_window.sortOptions.count())
        ]

        assert items == [combo.itemText(i) for i in range(combo.count())]
        assert combo.isHidden()

    def test_choosing_a_sort_option_changes_the_thumbnail_order(
        self, main_window: MainWindowManager
    ):
        main_window.sortOptions.setCurrentIndex(2)

        assert main_window.thumbnails.order_combo.currentIndex() == 2

    def test_selection_mode_button_makes_a_plain_click_toggle_pages(
        self, main_window: MainWindowManager
    ):
        view = cast(Any, main_window.thumbnails.view)
        assert not view.toggle_on_click

        main_window.selectionModeBtn.setChecked(True)
        assert view.toggle_on_click

        main_window.selectionModeBtn.setChecked(False)
        assert not view.toggle_on_click

    def test_focus_button_decides_whether_the_list_follows_the_current_page(
        self, settled_window: MainWindowManager, monkeypatch: pytest.MonkeyPatch
    ):
        window = settled_window
        scrolls: list[int] = []
        monkeypatch.setattr(
            window.thumbnails, "scroll_to_current", lambda: scrolls.append(1)
        )
        view = window.thumbnails.view
        model = window.thumbnails.model

        window.focusButton.setChecked(True)
        view.setCurrentIndex(model.index(1))
        followed = len(scrolls)
        window.focusButton.setChecked(False)
        view.setCurrentIndex(model.index(2))

        assert followed >= 1
        assert len(scrolls) == followed
        window.renderer.wait_for_done()

    def test_selected_page_navigation_jumps_between_selected_pages(
        self, settled_window: MainWindowManager
    ):
        window = settled_window
        pages = window.pages()
        window.selectionModeBtn.setChecked(True)
        selection = window.thumbnails.view.selectionModel()
        for row in (0, 2):
            selection.select(
                window.thumbnails.model.index(row),
                QtCore.QItemSelectionModel.SelectionFlag.Select,
            )

        window.goNextSelectedPage()
        assert window.current_page == pages[2]
        window.goNextSelectedPage()
        assert window.current_page == pages[2]
        window.goPrevSelectedPage()
        assert window.current_page == pages[0]
        window.renderer.wait_for_done()

    def test_selected_pages_reports_the_thumbnail_selection(
        self, settled_window: MainWindowManager
    ):
        window = settled_window
        window.selectionModeBtn.setChecked(True)
        selection = window.thumbnails.view.selectionModel()
        selection.select(
            window.thumbnails.model.index(1),
            QtCore.QItemSelectionModel.SelectionFlag.Select,
        )

        assert window.pages()[1] in window.selectedPages()
        window.renderer.wait_for_done()


def _store_detected_spreads(window: MainWindowManager, image_ids) -> None:
    """Store a two-page layout for each image the way Page Split detection does."""
    for image_id in image_ids:
        window.stages.page_split_filter.set_params(
            PageId(image_id=image_id),
            SplitParams(
                page_layout=PageLayout.two_pages(600, 800, 300),
                dependencies=SplitDependencies(image_size=(600, 800)),
            ),
        )


class TestDetectedPageCounts:
    """A detected two-page spread adds a page to the list, as in C++."""

    def test_an_interactive_render_that_detects_a_spread_adds_its_pages(
        self, qtbot: QtBot, settled_window: MainWindowManager, monkeypatch
    ):
        window = settled_window
        real_render = page_renderer.render_page_with_model

        def detecting_render(
            page_id, sequence, end_stage, dpi_overrides=None, **kwargs
        ):
            _store_detected_spreads(window, [page_id.image_id])
            return real_render(page_id, sequence, end_stage, **kwargs)

        monkeypatch.setattr(page_renderer, "render_page_with_model", detecting_render)
        first = window.pages()[0]
        assert len(window.pages()) == 3

        window._request_render()
        qtbot.waitUntil(lambda: len(window.pages()) == 4, timeout=RENDER_TIMEOUT_MS)

        assert [page.image_id for page in window.pages()].count(first.image_id) == 2
        current = window.current_page
        assert current is not None
        assert current.image_id == first.image_id
        assert current in window.pages()
        assert window.modified
        window.renderer.wait_for_done()

    def test_a_batch_that_detects_spreads_adds_their_pages(
        self, qtbot: QtBot, project_window: MainWindowManager, monkeypatch
    ):
        window = project_window
        image_count = len(window.pages())

        def detecting_batch(stages, pages, output_dir, **_kwargs):
            _store_detected_spreads(window, [page.image_id for page in pages])
            result = BatchResult()
            for page in pages:
                result.written[page] = output_dir / "x.tif"
            return result

        monkeypatch.setattr(batch_runner, "run_batch", detecting_batch)

        with qtbot.waitSignal(window.batch_runner.finished, timeout=10_000):
            window.startBatchProcessing()

        assert len(window.pages()) == 2 * image_count
        assert window.current_page in window.pages()

    def test_entering_a_later_stage_adopts_layouts_stored_by_an_earlier_run(
        self, settled_window: MainWindowManager
    ):
        window = settled_window
        assert window.project is not None
        _store_detected_spreads(window, [info.id for info in window.project.images])

        _switch_stage(window, FilterStage.DESKEW)

        assert len(window.pages()) == 6
        window.renderer.wait_for_done()


@pytest.fixture
def auto_save_enabled() -> object:
    settings = ApplicationSettings.get_instance()
    previous = settings.auto_save_project
    settings.auto_save_project = True
    yield settings
    settings.auto_save_project = previous


class TestAutoSave:
    """The project is written a while after the current page changes."""

    @staticmethod
    def _arm(window: MainWindowManager, project_file: pathlib.Path) -> None:
        window.projectFile = str(project_file)
        window._auto_save_timer.stop()
        window._auto_save_timer.setInterval(10)

    def test_changing_page_saves_the_project_when_auto_save_is_on(
        self,
        qtbot: QtBot,
        settled_window: MainWindowManager,
        tmp_path: pathlib.Path,
        auto_save_enabled: object,
    ):
        target = tmp_path / "auto.ScanTailor"
        self._arm(settled_window, target)

        settled_window.goNextPage()

        qtbot.waitUntil(target.is_file, timeout=10_000)
        assert not settled_window.modified
        settled_window.renderer.wait_for_done()

    def test_nothing_is_written_when_auto_save_is_off(
        self,
        qtbot: QtBot,
        settled_window: MainWindowManager,
        tmp_path: pathlib.Path,
    ):
        settings = ApplicationSettings.get_instance()
        previous = settings.auto_save_project
        settings.auto_save_project = False
        try:
            target = tmp_path / "auto.ScanTailor"
            self._arm(settled_window, target)

            settled_window.goNextPage()
            qtbot.waitUntil(lambda: not settled_window._auto_save_timer.isActive())
        finally:
            settings.auto_save_project = previous

        assert not target.exists()
        settled_window.renderer.wait_for_done()

    def test_a_project_without_a_file_is_not_saved_automatically(
        self,
        qtbot: QtBot,
        settled_window: MainWindowManager,
        tmp_path: pathlib.Path,
        auto_save_enabled: object,
    ):
        self._arm(settled_window, tmp_path / "unused.ScanTailor")
        settled_window.projectFile = ""

        settled_window.goNextPage()
        qtbot.waitUntil(lambda: not settled_window._auto_save_timer.isActive())

        assert not list(tmp_path.glob("*.ScanTailor"))
        settled_window.renderer.wait_for_done()

    def test_the_countdown_is_not_restarted_by_further_page_changes(
        self, settled_window: MainWindowManager
    ):
        window = settled_window
        window._auto_save_timer.stop()
        window._auto_save_timer.setInterval(60_000)

        window.goNextPage()
        remaining = window._auto_save_timer.remainingTime()
        window.goNextPage()

        assert window._auto_save_timer.isActive()
        assert window._auto_save_timer.remainingTime() <= remaining
        window.renderer.wait_for_done()


@pytest.fixture
def new_images(page_images, dialog_factory) -> list[pathlib.Path]:
    """Two scans that are not in the project, offered by the files dialog."""
    created = []
    for name in ("new_a.png", "new_b.png"):
        path = page_images.directory / name
        shutil.copy(page_images.paths[0], path)
        created.append(path)
    dialog_factory.names = [path.name for path in created]
    return created


def _image_names(window: MainWindowManager) -> list[str]:
    assert window.project is not None
    return [info.id.file_path.stem for info in window.project.images]


def _listed_names(window: MainWindowManager) -> list[str]:
    return [page.image_id.file_path.stem for page in window.thumbnails.pages()]


class TestInsertImages:
    """Insert before, after and at the end add pages where C++ does."""

    def test_insert_before_puts_the_new_images_ahead_of_the_page_in_order(
        self,
        project_window: MainWindowManager,
        new_images: list[pathlib.Path],
        respond_to_modal,
    ):
        window = project_window
        window.modified = False
        respond_to_modal(accept=True)

        window.insert_images_before(window.pages()[1])

        expected = ["page_001", "new_a", "new_b", "page_002", "page_003"]
        assert _image_names(window) == expected
        assert _listed_names(window) == expected
        assert window.modified
        window.renderer.wait_for_done()

    def test_insert_after_puts_the_new_images_behind_the_page_in_order(
        self,
        project_window: MainWindowManager,
        new_images: list[pathlib.Path],
        respond_to_modal,
    ):
        window = project_window
        respond_to_modal(accept=True)

        window.insert_images_after(window.pages()[0])

        expected = ["page_001", "new_a", "new_b", "page_002", "page_003"]
        assert _image_names(window) == expected
        assert _listed_names(window) == expected
        window.renderer.wait_for_done()

    def test_insert_here_appends_after_the_last_page(
        self,
        project_window: MainWindowManager,
        new_images: list[pathlib.Path],
        respond_to_modal,
    ):
        respond_to_modal(accept=True)

        project_window.insert_images_at_end()

        assert _listed_names(project_window) == [
            "page_001",
            "page_002",
            "page_003",
            "new_a",
            "new_b",
        ]
        project_window.renderer.wait_for_done()

    def test_the_current_page_stays_current_and_the_status_bar_counts_the_new_pages(
        self,
        project_window: MainWindowManager,
        new_images: list[pathlib.Path],
        respond_to_modal,
    ):
        window = project_window
        window.goLastPage()
        current = window.current_page
        respond_to_modal(accept=True)

        window.insert_images_before(window.pages()[0])

        assert window.current_page == current
        assert window.status_panel.page_no_label.text() == "p. 5 / 5"
        window.renderer.wait_for_done()

    def test_cancelling_the_dialog_inserts_nothing(
        self,
        project_window: MainWindowManager,
        new_images: list[pathlib.Path],
        respond_to_modal,
    ):
        window = project_window
        window.modified = False
        respond_to_modal(accept=False)

        window.insert_images_before(window.pages()[0])

        assert len(window.pages()) == 3
        assert not window.modified

    def test_a_page_outside_the_project_inserts_nothing(
        self,
        project_window: MainWindowManager,
        new_images: list[pathlib.Path],
        respond_to_modal,
    ):
        window = project_window
        window.modified = False
        stranger = PageId(image_id=ImageId(file_path=pathlib.Path("elsewhere.png")))
        respond_to_modal(accept=True)

        window.insert_images_after(stranger)

        assert len(window.pages()) == 3
        assert not window.modified

    def test_the_thumbnail_menu_requests_reach_the_insert_slots(
        self,
        project_window: MainWindowManager,
        new_images: list[pathlib.Path],
        respond_to_modal,
    ):
        window = project_window
        respond_to_modal(accept=True)

        window.thumbnails.insert_before_requested.emit(window.pages()[2])

        assert _listed_names(window)[2:4] == ["new_a", "new_b"]
        window.renderer.wait_for_done()


class TestRemovePages:
    """Removing pages updates the project and the list as in C++."""

    @staticmethod
    def _answer(window: MainWindowManager, answer: bool | None) -> list[int]:
        asked: list[int] = []

        def prompt(count: int, _parent: QtWidgets.QWidget) -> bool | None:
            asked.append(count)
            return answer

        window.remove_pages_prompt = prompt
        return asked

    def test_removing_a_page_drops_its_image_and_asks_with_the_page_count(
        self, project_window: MainWindowManager
    ):
        window = project_window
        asked = self._answer(window, False)
        window.modified = False

        window.remove_pages([window.pages()[1], window.pages()[2]])

        assert asked == [2]
        assert _image_names(window) == ["page_001"]
        assert _listed_names(window) == ["page_001"]
        assert window.modified
        window.renderer.wait_for_done()

    def test_cancelling_removes_nothing(self, project_window: MainWindowManager):
        window = project_window
        self._answer(window, None)
        window.modified = False

        window.remove_pages([window.pages()[0]])

        assert len(window.pages()) == 3
        assert not window.modified

    def test_removing_the_first_page_selects_the_first_page_that_stays(
        self, project_window: MainWindowManager
    ):
        window = project_window
        self._answer(window, False)
        pages = window.pages()

        window.remove_pages([pages[0], pages[1]])

        assert window.current_page == pages[2]
        window.renderer.wait_for_done()

    def test_removing_a_later_page_selects_the_page_before_the_first_removed(
        self, project_window: MainWindowManager
    ):
        window = project_window
        self._answer(window, False)
        pages = window.pages()
        window.goLastPage()

        window.remove_pages([pages[1], pages[2]])

        assert window.current_page == pages[0]
        window.renderer.wait_for_done()

    def test_status_bar_counts_the_remaining_pages(
        self, project_window: MainWindowManager
    ):
        window = project_window
        self._answer(window, False)

        window.remove_pages([window.pages()[0]])

        assert window.status_panel.page_no_label.text() == "p. 1 / 2"
        assert window.status_panel.page_info_label.text() == "page_002"
        window.renderer.wait_for_done()

    def test_removing_every_page_leaves_an_empty_project(
        self, project_window: MainWindowManager
    ):
        window = project_window
        self._answer(window, False)

        window.remove_pages(window.pages())

        assert window.pages() == []
        assert window.current_page is None
        assert window.status_panel.page_no_label.text() == ""

    def test_output_files_are_erased_when_the_dialog_says_so(
        self, project_window: MainWindowManager, page_images
    ):
        window = project_window
        assert window.project is not None
        out = window.project.output_directory
        out.mkdir(parents=True, exist_ok=True)
        for name in ("page_001.tif", "page_002.tif"):
            (out / name).write_bytes(b"output")
        self._answer(window, True)

        window.remove_pages([window.pages()[0]])

        assert not (out / "page_001.tif").exists()
        assert (out / "page_002.tif").exists()
        assert page_images.paths[0].exists()
        window.renderer.wait_for_done()

    def test_output_files_stay_when_the_dialog_says_so(
        self, project_window: MainWindowManager
    ):
        window = project_window
        assert window.project is not None
        out = window.project.output_directory
        out.mkdir(parents=True, exist_ok=True)
        (out / "page_001.tif").write_bytes(b"output")
        self._answer(window, False)

        window.remove_pages([window.pages()[0]])

        assert (out / "page_001.tif").exists()
        window.renderer.wait_for_done()

    def test_removing_one_half_of_a_spread_keeps_the_other_half(
        self, project_window: MainWindowManager
    ):
        window = project_window
        assert window.project is not None
        first = window.project.images[0].id
        window._on_page_count_requested(first, 2)
        left = PageId(image_id=first, sub_page=SubPage.LEFT_PAGE)
        right = PageId(image_id=first, sub_page=SubPage.RIGHT_PAGE)
        assert left in window.pages()
        out = window.project.output_directory
        out.mkdir(parents=True, exist_ok=True)
        for name in ("page_001_1L.tif", "page_001_2R.tif"):
            (out / name).write_bytes(b"output")
        self._answer(window, True)

        window.remove_pages([left])

        assert right in window.pages()
        assert left not in window.pages()
        assert not (out / "page_001_1L.tif").exists()
        assert (out / "page_001_2R.tif").exists()
        window.renderer.wait_for_done()

    def test_the_thumbnail_menu_request_reaches_the_remove_slot(
        self, project_window: MainWindowManager
    ):
        window = project_window
        self._answer(window, False)

        window.thumbnails.remove_pages_requested.emit([window.pages()[0]])

        assert _listed_names(window) == ["page_002", "page_003"]
        window.renderer.wait_for_done()

    def test_nothing_is_removed_without_a_project(self, main_window: MainWindowManager):
        asked = self._answer(main_window, True)

        main_window.remove_pages([])

        assert asked == []
