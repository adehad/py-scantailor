"""Thumbnail sort orders of each stage, the sort direction and the incomplete marker."""

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QStyleOptionViewItem, QWidget
from pytestqt.qtbot import QtBot

from scantailor.app.ui.thumbnails import ThumbnailListWidget
from scantailor.app.ui.thumbnails.model import IncompleteRole
from scantailor.app.ui.thumbnails.stage_orders import (
    completeness_provider,
    is_page_incomplete,
    stage_page_orders,
)
from scantailor.app.ui.thumbnails.widget import ThumbnailDelegate
from scantailor.core.models import ImageId, PageId
from scantailor.core.project import ImageMetadata, Project
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.deskew.params import Params as DeskewParams
from scantailor.filters.page_layout.params import ContentSize
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.page_split.page_layout import PageLayout
from scantailor.filters.page_split.params import Params as SplitParams
from scantailor.filters.select_content.content_box import PhysicalSize
from scantailor.filters.select_content.params import Params as ContentParams
from tests.ui.hosting import show_in_host

COUNT = 5


def _renderer(page: PageId, stage: str) -> QImage:
    image = QImage(40, 60, QImage.Format.Format_RGB32)
    image.fill(QColor("white"))
    return image


@pytest.fixture
def project() -> Project:
    project = Project()
    for number in range(COUNT):
        project.add_image(
            ImageId(file_path=Path(f"page_{number}.png")),
            ImageMetadata(width=500, height=700),
        )
    return project


@pytest.fixture
def sequence() -> StageSequence:
    return StageSequence()


@pytest.fixture
def pages(project: Project) -> list[PageId]:
    return project.get_pages()


def _names(sequence: StageSequence, stage: FilterStage) -> list[str]:
    return [order.name for order in stage_page_orders(sequence, stage)]


def _widget(
    qtbot: QtBot,
    host: QWidget,
    project: Project,
    sequence: StageSequence,
    stage: FilterStage,
) -> ThumbnailListWidget:
    widget = ThumbnailListWidget(_renderer, project, stage=str(int(stage)), parent=host)
    widget.bind_stage_sequence(sequence)
    show_in_host(qtbot, widget, 240, 700)
    return widget


class TestOptionsOfEachStage:
    def test_fix_orientation_offers_no_order(self, sequence):
        assert _names(sequence, FilterStage.FIX_ORIENTATION) == []

    def test_page_split_orders_by_split_type(self, sequence):
        assert _names(sequence, FilterStage.PAGE_SPLIT) == [
            "Natural order",
            "Order by split type",
        ]

    def test_deskew_orders_by_deviation(self, sequence):
        assert _names(sequence, FilterStage.DESKEW) == [
            "Natural order",
            "Order by decreasing deviation",
        ]

    @pytest.mark.parametrize(
        "stage", [FilterStage.SELECT_CONTENT, FilterStage.PAGE_LAYOUT]
    )
    def test_select_content_and_page_layout_order_by_size_and_deviation(
        self, sequence, stage
    ):
        assert _names(sequence, stage) == [
            "Natural order",
            "Order by increasing width",
            "Order by increasing height",
            "Order by decreasing deviation",
        ]

    def test_output_orders_by_completeness(self, sequence):
        assert _names(sequence, FilterStage.OUTPUT) == [
            "Natural order",
            "Order by completeness",
        ]


class TestDeviationOrder:
    def test_pages_furthest_from_the_median_angle_come_first(
        self, qtbot, host, project, sequence, pages
    ):
        angles = [0.0, 5.0, 0.5, -4.0, 0.0]
        for page, angle in zip(pages, angles):
            sequence.deskew_filter.settings.set_params(
                page, DeskewParams(deskew_angle_deg=angle)
            )
        widget = _widget(qtbot, host, project, sequence, FilterStage.DESKEW)

        widget.set_order(1)

        median = sorted(angles)[len(angles) // 2]
        deviations = {page: abs(a - median) for page, a in zip(pages, angles)}
        expected = sorted(pages, key=lambda page: -deviations[page])
        assert widget.pages() == expected

    def test_pages_without_a_result_go_last(
        self, qtbot, host, project, sequence, pages
    ):
        for page, angle in zip(pages[:3], (0.0, 6.0, 1.0)):
            sequence.deskew_filter.settings.set_params(
                page, DeskewParams(deskew_angle_deg=angle)
            )
        widget = _widget(qtbot, host, project, sequence, FilterStage.DESKEW)

        widget.set_order(1)

        assert widget.pages()[-2:] == pages[3:]
        assert widget.pages()[0] == pages[1]

    def test_ties_keep_the_natural_order(self, qtbot, host, project, sequence, pages):
        for page in pages:
            sequence.deskew_filter.settings.set_params(
                page, DeskewParams(deskew_angle_deg=1.0)
            )
        widget = _widget(qtbot, host, project, sequence, FilterStage.DESKEW)

        widget.set_order(1)

        assert widget.pages() == pages


class TestSplitTypeOrder:
    def test_pages_are_grouped_by_type_with_undecided_ones_after(
        self, qtbot, host, project, sequence, pages
    ):
        types = [
            LayoutType.TWO_PAGES,
            LayoutType.SINGLE_PAGE_UNCUT,
            LayoutType.AUTO_LAYOUT_TYPE,
            LayoutType.PAGE_PLUS_OFFCUT,
            LayoutType.SINGLE_PAGE_UNCUT,
        ]
        for page, layout_type in zip(pages, types):
            sequence.page_split_filter.set_params(
                page, SplitParams(layout_type=layout_type)
            )
        widget = _widget(qtbot, host, project, sequence, FilterStage.PAGE_SPLIT)

        widget.set_order(1)

        assert widget.pages() == [pages[1], pages[4], pages[3], pages[0], pages[2]]

    def test_reversing_a_split_type_order_reverses_pages_of_one_type_too(
        self, qtbot, host, project, sequence, pages
    ):
        for page in pages:
            sequence.page_split_filter.set_params(
                page, SplitParams(layout_type=LayoutType.SINGLE_PAGE_UNCUT)
            )
        widget = _widget(qtbot, host, project, sequence, FilterStage.PAGE_SPLIT)
        widget.set_order(1)

        widget.set_sort_reversed(True)

        assert widget.pages() == list(reversed(pages))

    def test_the_type_of_the_layout_wins_over_the_requested_one(
        self, qtbot, host, project, sequence, pages
    ):
        spread = PageLayout.two_pages(500, 700, 250)
        sequence.page_split_filter.set_params(
            pages[0],
            SplitParams(layout_type=LayoutType.AUTO_LAYOUT_TYPE, page_layout=spread),
        )
        sequence.page_split_filter.set_params(
            pages[1], SplitParams(layout_type=LayoutType.SINGLE_PAGE_UNCUT)
        )
        widget = _widget(qtbot, host, project, sequence, FilterStage.PAGE_SPLIT)

        widget.set_order(1)

        assert widget.pages()[:2] == [pages[1], pages[0]]


class TestSizeOrders:
    def test_select_content_pages_are_ordered_by_content_width_then_height(
        self, qtbot, host, project, sequence, pages
    ):
        sizes = [(120, 90), (80, 200), (100, 50), (80, 150), (90, 10)]
        for page, (width, height) in zip(pages, sizes):
            sequence.select_content_filter.set_params(
                page,
                ContentParams(
                    content_size_mm=PhysicalSize(width_mm=width, height_mm=height)
                ),
            )
        widget = _widget(qtbot, host, project, sequence, FilterStage.SELECT_CONTENT)

        widget.set_order(1)
        by_width = widget.pages()
        widget.set_order(2)
        by_height = widget.pages()

        assert by_width == [pages[1], pages[3], pages[4], pages[2], pages[0]]
        assert by_height == [pages[4], pages[2], pages[0], pages[3], pages[1]]

    def test_a_page_without_a_size_goes_last(
        self, qtbot, host, project, sequence, pages
    ):
        for page, width in zip(pages[1:], (30.0, 20.0, 10.0, 40.0)):
            sequence.select_content_filter.set_params(
                page,
                ContentParams(
                    content_size_mm=PhysicalSize(width_mm=width, height_mm=10)
                ),
            )
        widget = _widget(qtbot, host, project, sequence, FilterStage.SELECT_CONTENT)

        widget.set_order(1)

        assert widget.pages()[-1] == pages[0]

    def test_page_layout_adds_the_hard_margins_to_the_content_size(
        self, qtbot, host, project, sequence, pages
    ):
        from scantailor.core.models import Margins

        widths = [100.0, 100.0, 100.0, 100.0, 100.0]
        lefts = [30, 10, 20, 0, 5]
        for page, width, left in zip(pages, widths, lefts):
            params = sequence.page_layout_filter.get_params(page)
            params = params.model_copy(
                update={
                    "content_size_mm": ContentSize(width_mm=width, height_mm=100),
                    "hard_margins_mm": Margins(left=left, right=0, top=0, bottom=0),
                }
            )
            sequence.page_layout_filter.set_params(page, params)
        widget = _widget(qtbot, host, project, sequence, FilterStage.PAGE_LAYOUT)

        widget.set_order(1)

        assert widget.pages() == [pages[3], pages[4], pages[1], pages[2], pages[0]]


class TestIncompletePages:
    def test_a_page_without_stored_results_is_incomplete(self, sequence, pages):
        sequence.deskew_filter.settings.set_params(pages[0], DeskewParams())

        assert not is_page_incomplete(sequence, FilterStage.DESKEW, pages[0])
        assert is_page_incomplete(sequence, FilterStage.DESKEW, pages[1])

    def test_fix_orientation_pages_are_never_incomplete(self, sequence, pages):
        assert not is_page_incomplete(sequence, FilterStage.FIX_ORIENTATION, pages[0])

    @pytest.mark.parametrize("stage", list(FilterStage))
    def test_the_provider_reads_the_stage_from_its_key(self, sequence, pages, stage):
        check = completeness_provider(sequence)

        assert check(pages[0], str(int(stage))) == is_page_incomplete(
            sequence, stage, pages[0]
        )

    def test_the_model_flags_incomplete_pages(
        self, qtbot, host, project, sequence, pages
    ):
        sequence.deskew_filter.settings.set_params(pages[2], DeskewParams())
        widget = _widget(qtbot, host, project, sequence, FilterStage.DESKEW)
        model = widget.model

        flags = [
            model.data(model.index(model.row_of(page)), IncompleteRole)
            for page in pages
        ]

        assert flags == [True, True, False, True, True]

    def test_a_page_leaves_the_incomplete_state_when_its_result_arrives(
        self, qtbot, host, project, sequence, pages
    ):
        widget = _widget(qtbot, host, project, sequence, FilterStage.DESKEW)
        model = widget.model
        row = model.row_of(pages[1])
        assert model.data(model.index(row), IncompleteRole)

        sequence.deskew_filter.settings.set_params(pages[1], DeskewParams())
        widget.invalidate(pages[1])

        assert not model.data(model.index(model.row_of(pages[1])), IncompleteRole)

    def test_the_output_order_puts_incomplete_pages_last(
        self, qtbot, host, project, sequence, pages
    ):
        for page in pages[2:4]:
            sequence.output_filter.set_params(
                page, sequence.output_filter.get_params(page)
            )
        widget = _widget(qtbot, host, project, sequence, FilterStage.OUTPUT)

        widget.set_order(1)

        assert widget.pages() == [pages[2], pages[3], pages[0], pages[1], pages[4]]

    def test_an_order_that_depends_on_results_follows_them(
        self, qtbot, host, project, sequence, pages
    ):
        widget = _widget(qtbot, host, project, sequence, FilterStage.OUTPUT)
        widget.set_order(1)
        assert widget.pages() == pages

        sequence.output_filter.set_params(
            pages[3], sequence.output_filter.get_params(pages[3])
        )
        widget.invalidate(pages[3])

        assert widget.pages()[0] == pages[3]

    def test_a_row_that_moves_keeps_its_selection(
        self, qtbot, host, project, sequence, pages
    ):
        widget = _widget(qtbot, host, project, sequence, FilterStage.OUTPUT)
        widget.set_order(1)
        widget.set_current_page(pages[3])
        widget.select_pages([pages[3]])

        sequence.output_filter.set_params(
            pages[3], sequence.output_filter.get_params(pages[3])
        )
        widget.invalidate(pages[3])

        assert widget.current_page() == pages[3]
        assert widget.selected_pages() == [pages[3]]
        assert widget.model.row_of(pages[3]) == 0


class TestMarker:
    @staticmethod
    def _cell(widget: ThumbnailListWidget, page: PageId) -> QImage:
        delegate = widget.view.itemDelegate()
        assert isinstance(delegate, ThumbnailDelegate)
        size = delegate.cell_size()
        image = QImage(size, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.white)
        option = QStyleOptionViewItem()
        option.rect = QRect(0, 0, size.width(), size.height())
        option.palette = widget.palette()
        painter = QPainter(image)
        delegate.paint(painter, option, widget.model.index(widget.model.row_of(page)))
        painter.end()
        return image

    @staticmethod
    def _reddish(image: QImage) -> int:
        rgba = image.convertToFormat(QImage.Format.Format_RGBA8888)
        pixels = np.frombuffer(
            rgba.constBits(), dtype=np.uint8, count=rgba.sizeInBytes()
        ).reshape(rgba.height(), rgba.bytesPerLine() // 4, 4)[:, : rgba.width()]
        red = pixels[..., 0].astype(int)
        green = pixels[..., 1].astype(int)
        blue = pixels[..., 2].astype(int)
        return int(((red > green + 20) & (red > blue + 20)).sum())

    def test_an_incomplete_page_is_marked_with_a_question_mark(
        self, qtbot, host, project, sequence, pages
    ):
        sequence.deskew_filter.settings.set_params(pages[0], DeskewParams())
        widget = _widget(qtbot, host, project, sequence, FilterStage.DESKEW)

        complete = self._cell(widget, pages[0])
        incomplete = self._cell(widget, pages[1])

        assert self._reddish(incomplete) > 50
        assert self._reddish(complete) == 0

    def test_the_mark_is_drawn_over_a_placeholder_too(
        self, qtbot, host, project, sequence, pages
    ):
        widget = ThumbnailListWidget(
            _renderer,
            project,
            stage=str(int(FilterStage.DESKEW)),
            parent=host,
        )
        widget.bind_stage_sequence(sequence)

        image = self._cell(widget, pages[1])

        assert self._reddish(image) > 50


class TestDirection:
    def test_reversing_flips_the_natural_order(
        self, qtbot, host, project, sequence, pages
    ):
        widget = _widget(qtbot, host, project, sequence, FilterStage.OUTPUT)

        widget.set_sort_reversed(True)

        assert widget.pages() == list(reversed(pages))
        assert widget.sort_reversed

    def test_reversing_flips_the_chosen_order(
        self, qtbot, host, project, sequence, pages
    ):
        for page, angle in zip(pages, (0.0, 5.0, 0.5, -4.5, 2.0)):
            sequence.deskew_filter.settings.set_params(
                page, DeskewParams(deskew_angle_deg=angle)
            )
        widget = _widget(qtbot, host, project, sequence, FilterStage.DESKEW)
        widget.set_order(1)
        forward = widget.pages()

        widget.set_sort_reversed(True)

        assert widget.pages() == list(reversed(forward))

    def test_ties_stay_in_natural_order_when_reversed(
        self, qtbot, host, project, sequence, pages
    ):
        for page in pages:
            sequence.deskew_filter.settings.set_params(
                page, DeskewParams(deskew_angle_deg=1.0)
            )
        widget = _widget(qtbot, host, project, sequence, FilterStage.DESKEW)
        widget.set_order(1)

        widget.set_sort_reversed(True)

        assert widget.pages() == pages

    def test_reversing_back_restores_the_order(
        self, qtbot, host, project, sequence, pages
    ):
        widget = _widget(qtbot, host, project, sequence, FilterStage.OUTPUT)

        widget.set_sort_reversed(True)
        widget.set_sort_reversed(False)

        assert widget.pages() == pages

    def test_the_selection_survives_reversing(
        self, qtbot, host, project, sequence, pages
    ):
        widget = _widget(qtbot, host, project, sequence, FilterStage.OUTPUT)
        widget.set_current_page(pages[1])
        widget.select_pages([pages[1], pages[2]])

        widget.set_sort_reversed(True)

        assert widget.current_page() == pages[1]
        assert set(widget.selected_pages()) == {pages[1], pages[2]}

    def test_pausing_the_sorting_shows_the_natural_order_whatever_the_direction(
        self, qtbot, host, project, sequence, pages
    ):
        widget = _widget(qtbot, host, project, sequence, FilterStage.OUTPUT)
        widget.set_sort_reversed(True)

        widget.set_sorting_paused(True)

        assert widget.pages() == list(reversed(pages))


class TestStageSwitch:
    def test_switching_the_stage_offers_that_stages_options(
        self, qtbot, host, project, sequence
    ):
        widget = _widget(qtbot, host, project, sequence, FilterStage.OUTPUT)
        assert widget.order_names() == ["Natural order", "Order by completeness"]

        with qtbot.waitSignal(widget.orders_changed):
            widget.set_stage(str(int(FilterStage.PAGE_SPLIT)))

        assert widget.order_names() == ["Natural order", "Order by split type"]
        assert widget.order_combo.count() == 2

    def test_a_stage_without_options_leaves_an_empty_combo(
        self, qtbot, host, project, sequence
    ):
        widget = _widget(qtbot, host, project, sequence, FilterStage.OUTPUT)

        widget.set_stage(str(int(FilterStage.FIX_ORIENTATION)))

        assert widget.order_names() == []
        assert widget.order_combo.count() == 0

    def test_a_new_stage_starts_with_its_first_order(
        self, qtbot, host, project, sequence, pages
    ):
        widget = _widget(qtbot, host, project, sequence, FilterStage.OUTPUT)
        widget.set_order(1)

        widget.set_stage(str(int(FilterStage.DESKEW)))

        assert widget.order_combo.currentIndex() == 0
        assert widget.pages() == pages

    def test_incomplete_pages_follow_the_stage(
        self, qtbot, host, project, sequence, pages
    ):
        sequence.deskew_filter.settings.set_params(pages[0], DeskewParams())
        widget = _widget(qtbot, host, project, sequence, FilterStage.FIX_ORIENTATION)
        model = widget.model
        assert not model.data(model.index(0), IncompleteRole)

        widget.set_stage(str(int(FilterStage.DESKEW)))

        assert not model.data(model.index(model.row_of(pages[0])), IncompleteRole)
        assert model.data(model.index(model.row_of(pages[1])), IncompleteRole)

    def test_without_a_sequence_the_built_in_orders_stay(self, qtbot, host, project):
        widget = ThumbnailListWidget(_renderer, project, stage="1", parent=host)

        assert widget.order_names() == [
            "Natural order",
            "Order by increasing width",
            "Order by increasing height",
        ]
        widget.set_stage("2")
        assert widget.order_names()[0] == "Natural order"
        assert len(widget.order_names()) == 3


class TestOrderRememberedPerStage:
    def test_coming_back_to_a_stage_restores_its_chosen_order(
        self, qtbot, host, project, sequence
    ):
        widget = _widget(qtbot, host, project, sequence, FilterStage.SELECT_CONTENT)
        widget.set_order(2)

        widget.set_stage(str(int(FilterStage.DESKEW)))
        widget.set_stage(str(int(FilterStage.SELECT_CONTENT)))

        assert widget.order_combo.currentIndex() == 2

    def test_each_stage_keeps_its_own_choice(self, qtbot, host, project, sequence):
        widget = _widget(qtbot, host, project, sequence, FilterStage.SELECT_CONTENT)
        widget.set_order(2)

        widget.set_stage(str(int(FilterStage.PAGE_LAYOUT)))
        assert widget.order_combo.currentIndex() == 0
        widget.set_order(1)
        widget.set_stage(str(int(FilterStage.SELECT_CONTENT)))
        assert widget.order_combo.currentIndex() == 2
        widget.set_stage(str(int(FilterStage.PAGE_LAYOUT)))
        assert widget.order_combo.currentIndex() == 1

    def test_the_restored_order_sorts_the_pages(
        self, qtbot, host, project, sequence, pages
    ):
        for page, angle in zip(pages, (0.0, 5.0, 0.5, -4.5, 2.0), strict=True):
            sequence.deskew_filter.settings.set_params(
                page, DeskewParams(deskew_angle_deg=angle)
            )
        widget = _widget(qtbot, host, project, sequence, FilterStage.DESKEW)
        widget.set_order(1)
        sorted_pages = widget.pages()

        widget.set_stage(str(int(FilterStage.OUTPUT)))
        widget.set_stage(str(int(FilterStage.DESKEW)))

        assert widget.pages() == sorted_pages
        assert sorted_pages != pages

    def test_a_new_stage_sequence_starts_every_stage_on_its_first_order(
        self, qtbot, host, project, sequence
    ):
        widget = _widget(qtbot, host, project, sequence, FilterStage.SELECT_CONTENT)
        widget.set_order(2)

        widget.bind_stage_sequence(StageSequence())
        widget.set_stage(str(int(FilterStage.DESKEW)))
        widget.set_stage(str(int(FilterStage.SELECT_CONTENT)))

        assert widget.order_combo.currentIndex() == 0
