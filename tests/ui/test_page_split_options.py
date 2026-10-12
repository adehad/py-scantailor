"""Page Split options: icons, page count, detected values and apply-to."""

import pathlib

import cv2
import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.dialogs.split_mode import SplitModeDialog
from scantailor.app.ui.MainWindow import MainWindowManager
from scantailor.app.ui.options_widgets import page_layout, page_split
from scantailor.app.ui.panels.system_load import SystemLoadWidget
from scantailor.app.ui.stage_panels import StagePanels
from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.core.pipeline import PipelineOptions, process_page
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.page_split.page_layout import PageLayout
from scantailor.filters.page_split.params import AutoManualMode

SKEW_DEGREES = 3.0
DETECTION_TOLERANCE_DEG = 1.0


def _switch_stage(window: MainWindowManager, stage: FilterStage) -> None:
    getattr(window, f"actionSwitchFilter{int(stage) + 1}").trigger()


@pytest.fixture
def split_window(qtbot: QtBot, project_window: MainWindowManager) -> MainWindowManager:
    """A project window showing Page Split with its first render delivered."""
    _switch_stage(project_window, FilterStage.PAGE_SPLIT)
    project_window.renderer.wait_for_done()
    qtbot.wait(50)
    return project_window


def _options(window: MainWindowManager) -> page_split.OptionsWidget:
    assert window._panels is not None
    options = window._panels.options
    assert isinstance(options, page_split.OptionsWidget)
    return options


class TestIcons:
    def test_page_split_buttons_have_icons(self, qtbot: QtBot):
        widget = page_split.OptionsWidget(StageSequence().page_split_filter)
        qtbot.addWidget(widget)

        for name in ("singlePageUncutBtn", "pagePlusOffcutBtn", "twoPagesBtn"):
            button = widget.findChild(QtWidgets.QToolButton, name)
            assert button is not None
            assert not button.icon().isNull(), name

    def test_page_layout_alignment_buttons_have_icons(self, qtbot: QtBot):
        widget = page_layout.OptionsWidget(StageSequence().page_layout_filter)
        qtbot.addWidget(widget)

        buttons = [
            button
            for button in widget.findChildren(QtWidgets.QToolButton)
            if button.objectName().startswith("align")
        ]
        assert len(buttons) == 9
        assert all(not button.icon().isNull() for button in buttons)

    def test_system_load_buttons_have_icons(self, qtbot: QtBot):
        widget = SystemLoadWidget()
        qtbot.addWidget(widget)

        for name in ("minusBtn", "plusBtn"):
            button = widget.findChild(QtWidgets.QToolButton, name)
            assert button is not None
            assert not button.icon().isNull(), name


class TestLabels:
    def _widget(self, qtbot: QtBot) -> tuple[page_split.OptionsWidget, StageSequence]:
        sequence = StageSequence()
        widget = page_split.OptionsWidget(sequence.page_split_filter)
        qtbot.addWidget(widget)
        return widget, sequence

    def test_automatic_request_without_a_layout_shows_a_question_mark(
        self, qtbot: QtBot
    ):
        widget, _ = self._widget(qtbot)

        widget.set_current_page(PageId(image_id=ImageId(file_path=pathlib.Path("a"))))

        assert widget._scope_label.text() == "?"
        assert not widget._auto_btn.isEnabled()
        assert not widget._manual_btn.isEnabled()
        assert not widget._change_btn.isEnabled()
        assert not any(b.isChecked() for b in widget._layout_button_group.buttons())

    def test_detected_layout_is_labelled_auto_detected(self, qtbot: QtBot):
        widget, sequence = self._widget(qtbot)
        page = PageId(image_id=ImageId(file_path=pathlib.Path("a")))
        split = sequence.page_split_filter
        split.set_params(
            page,
            split.get_params(page).with_auto_layout(
                PageLayout.two_pages(600, 400, 300)
            ),
        )

        widget.set_current_page(page)

        assert widget._scope_label.text() == "Auto detected"
        assert widget._two_pages_btn.isChecked()
        assert widget._auto_btn.isChecked()
        assert widget._auto_btn.isEnabled()
        assert widget._change_btn.isEnabled()

    def test_requested_type_is_labelled_set_manually(self, qtbot: QtBot):
        widget, sequence = self._widget(qtbot)
        page = PageId(image_id=ImageId(file_path=pathlib.Path("a")))
        sequence.page_split_filter.set_layout_type(page, LayoutType.PAGE_PLUS_OFFCUT)

        widget.set_current_page(page)

        assert widget._scope_label.text() == "Set manually"
        assert widget._page_offcut_btn.isChecked()
        assert widget._split_group.isVisibleTo(widget)

    def test_widget_has_no_auto_layout_type_button(self, qtbot: QtBot):
        widget, _ = self._widget(qtbot)

        assert len(widget._layout_button_group.buttons()) == 3
        assert [b.text() for b in widget.findChildren(QtWidgets.QPushButton)] == [
            "Change ...",
            "Auto",
            "Manual",
        ]


class TestPageCount:
    def test_choosing_two_pages_gives_the_image_two_pages(
        self, split_window: MainWindowManager
    ):
        window = split_window
        assert window.project is not None

        _options(window)._two_pages_btn.click()

        assert window.project.images[0].num_sub_pages == 2
        assert len(window.pages()) == 4
        assert window.thumbnails.pages() == window.pages()
        assert window.modified

    def test_current_page_is_kept_on_the_same_image(
        self, split_window: MainWindowManager
    ):
        window = split_window
        assert window.project is not None
        image_id = window.project.images[0].id

        _options(window)._two_pages_btn.click()

        current = window.current_page
        assert current is not None
        assert current.image_id == image_id
        assert current.sub_page is SubPage.LEFT_PAGE
        assert window.thumbnails.current_page() == current

    def test_choosing_a_single_page_again_gives_the_image_one_page(
        self, split_window: MainWindowManager
    ):
        window = split_window
        assert window.project is not None
        _options(window)._two_pages_btn.click()

        _options(window)._single_page_btn.click()

        assert window.project.images[0].num_sub_pages == 1
        assert len(window.pages()) == 3

    def test_the_chosen_type_follows_the_image_to_its_new_pages(
        self, split_window: MainWindowManager
    ):
        window = split_window

        _options(window)._two_pages_btn.click()

        assert _options(window)._two_pages_btn.isChecked()
        right = window.pages()[1]
        assert (
            window.stages.page_split_filter.get_layout_type(right)
            is LayoutType.TWO_PAGES
        )


def _answer_split_dialog(*, all_pages: bool) -> None:
    def poll() -> None:
        modal = QtWidgets.QApplication.activeModalWidget()
        if not isinstance(modal, SplitModeDialog):
            QtCore.QTimer.singleShot(10, poll)
            return
        modal._mode_manual.setChecked(True)
        (modal._all_pages_rb if all_pages else modal._this_page_rb).setChecked(True)
        modal._on_submit()

    QtCore.QTimer.singleShot(10, poll)


class TestApplyTo:
    def test_change_button_applies_the_layout_type_to_the_chosen_pages(
        self, split_window: MainWindowManager
    ):
        window = split_window
        split = window.stages.page_split_filter
        current = window.current_page
        assert current is not None
        split.set_params(
            current,
            split.get_params(current)
            .with_manual_layout(PageLayout.two_pages(600, 800, 250))
            .with_layout_type(LayoutType.TWO_PAGES),
        )

        _answer_split_dialog(all_pages=True)
        _options(window)._change_btn.click()

        assert window.project is not None
        assert [info.num_sub_pages for info in window.project.images] == [2, 2, 2]
        for info in window.project.images:
            page = PageId(image_id=info.id)
            assert split.get_layout_type(page) is LayoutType.TWO_PAGES

    def test_cancelled_dialog_changes_nothing(
        self, split_window: MainWindowManager, respond_to_modal
    ):
        window = split_window
        before = [info.num_sub_pages for info in window.project.images]  # type: ignore[union-attr]
        split = window.stages.page_split_filter
        current = window.current_page
        assert current is not None
        split.set_params(
            current,
            split.get_params(current).with_auto_layout(
                PageLayout.single_page_uncut(600, 800)
            ),
        )

        respond_to_modal(accept=False)
        _options(window)._change_btn.click()

        assert [info.num_sub_pages for info in window.project.images] == before  # type: ignore[union-attr]

    def test_two_pages_applied_to_other_pages_grows_their_page_lists(
        self, split_window: MainWindowManager
    ):
        window = split_window
        targets = window.pages()[1:]

        window.apply_to_pages(
            FilterStage.PAGE_SPLIT,
            targets,
            layout_type=LayoutType.TWO_PAGES,
            apply_cut=False,
        )

        assert len(window.pages()) == 5
        assert window.modified


def _skewed_text_page() -> np.ndarray:
    image = np.full((500, 400), 255, dtype=np.uint8)
    for row in range(60, 440, 24):
        for column in range(40, 360, 30):
            image[row : row + 10, column : column + 22] = 0
    height, width = image.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), SKEW_DEGREES, 1.0)
    return np.asarray(
        cv2.warpAffine(image, matrix, (width, height), borderValue=255),
        dtype=np.uint8,
    )


class TestRefreshAfterRender:
    def test_deskew_spin_box_shows_the_detected_angle_after_the_render(
        self, qtbot: QtBot, tmp_path: pathlib.Path
    ):
        scan = tmp_path / "scan.png"
        image = _skewed_text_page()
        assert cv2.imwrite(str(scan), image)
        page = PageId(image_id=ImageId(file_path=scan))
        sequence = StageSequence()
        panels = StagePanels(FilterStage.DESKEW, sequence)
        qtbot.addWidget(panels.options)
        qtbot.addWidget(panels.view)
        panels.set_page(page)
        spin_box = panels.options.findChild(QtWidgets.QDoubleSpinBox, "angleSpinBox")
        assert spin_box is not None
        assert spin_box.value() == 0.0

        rendered = process_page(
            image,
            page,
            sequence,
            PipelineOptions(end_stage=FilterStage.DESKEW, dpi=300.0),
        )
        panels.show_image(rendered.image)

        assert spin_box.value() == pytest.approx(
            SKEW_DEGREES, abs=DETECTION_TOLERANCE_DEG
        )
        assert not panels.options._manual_btn.isChecked()  # type: ignore[attr-defined]

    def test_page_split_buttons_show_the_detected_layout_after_the_render(
        self, qtbot: QtBot, tmp_path: pathlib.Path
    ):
        scan = tmp_path / "spread.png"
        image = np.full((400, 600), 220, dtype=np.uint8)
        image[:, 296:304] = 40
        assert cv2.imwrite(str(scan), image)
        page = PageId(image_id=ImageId(file_path=scan))
        sequence = StageSequence()
        panels = StagePanels(FilterStage.PAGE_SPLIT, sequence)
        qtbot.addWidget(panels.options)
        qtbot.addWidget(panels.view)
        panels.set_page(page)
        options = panels.options
        assert isinstance(options, page_split.OptionsWidget)
        assert not options._auto_btn.isEnabled()

        sequence.page_split_filter.process(image, page)
        panels.show_image(image)

        assert options._auto_btn.isEnabled()
        assert options._auto_btn.isChecked()
        assert options._scope_label.text() == "Auto detected"
        params = sequence.page_split_filter.get_params(page)
        assert params.split_line_mode is AutoManualMode.AUTO
