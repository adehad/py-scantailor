"""Thumbnails of pages that stand out from the others carry an asterisk."""

import math
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QStyleOptionViewItem
from pytestqt.qtbot import QtBot

from scantailor.app.ui.thumbnails import ThumbnailListWidget
from scantailor.app.ui.thumbnails.deviation import (
    deviant_pages,
    deviation_provider,
    stage_rule,
)
from scantailor.app.ui.thumbnails.model import DeviantRole, PageEntry
from scantailor.app.ui.thumbnails.widget import ThumbnailDelegate
from scantailor.core.models import ImageId, Margins, PageId
from scantailor.core.project import ImageMetadata, Project
from scantailor.core.settings import ApplicationSettings, DeviationSettings
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.deskew.params import Params as DeskewParams
from scantailor.filters.select_content.content_box import PhysicalSize
from scantailor.filters.select_content.params import Params as ContentParams

COUNT = 6
LOOSE = DeviationSettings(coefficient=1.0, threshold=0.0)
DESKEW = str(int(FilterStage.DESKEW))


def _page(number: int) -> PageId:
    return PageId(image_id=ImageId(file_path=Path(f"page_{number}.png")))


def _values(*values: float) -> dict[PageId, float]:
    return {_page(number): value for number, value in enumerate(values)}


class _Renders:
    """Counts the renders the list asks for."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self, page: PageId, stage: str) -> QImage:
        self.count += 1
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
def pages(project: Project) -> list[PageId]:
    return project.get_pages()


@pytest.fixture
def sequence() -> StageSequence:
    return StageSequence()


def _set_angles(sequence: StageSequence, pages: list[PageId], angles: list[float]):
    for page, angle in zip(pages, angles):
        sequence.deskew_filter.settings.set_params(
            page, DeskewParams(deskew_angle_deg=angle)
        )


def _widget(
    qtbot: QtBot, project: Project, sequence: StageSequence, renders: _Renders
) -> ThumbnailListWidget:
    widget = ThumbnailListWidget(renders, project, stage=DESKEW)
    qtbot.addWidget(widget)
    widget.bind_stage_sequence(sequence)
    return widget


def _flags(widget: ThumbnailListWidget, pages: list[PageId]) -> list[bool]:
    model = widget.model
    return [model.data(model.index(model.row_of(page)), DeviantRole) for page in pages]


class TestDeviantPages:
    def test_a_value_far_from_the_others_deviates(self):
        values = _values(0.1, 0.2, 0.0, 0.1, 4.0)

        assert deviant_pages(values, DeviationSettings(coefficient=1.5)) == {_page(4)}

    def test_fewer_than_three_pages_never_deviate(self):
        assert deviant_pages(_values(0.0, 9.0), LOOSE) == frozenset()

    def test_identical_values_never_deviate(self):
        assert deviant_pages(_values(1.0, 1.0, 1.0, 1.0), LOOSE) == frozenset()

    def test_the_threshold_spares_pages_within_that_percentage_of_the_median(self):
        values = _values(100.0, 100.0, 100.0, 103.0)
        spread_only = DeviationSettings(coefficient=1.0, threshold=0.0)
        five_percent = DeviationSettings(coefficient=1.0, threshold=5.0)

        assert deviant_pages(values, spread_only) == {_page(3)}
        assert deviant_pages(values, five_percent) == frozenset()

    def test_one_bad_page_does_not_make_the_good_pages_deviate(self):
        # Upstream's mean and standard deviation mark all five pages here.
        values = _values(249.0, 250.0, 251.0, 250.0, 200.0)

        assert deviant_pages(
            values, DeviationSettings(coefficient=0.35, threshold=1.0)
        ) == {_page(4)}

    def test_the_deskew_threshold_counts_from_the_size_of_a_negative_median(self):
        # The signed mean of these angles is negative, which made upstream's
        # threshold term negative and so without effect.
        angles = _values(-2.0, -2.1, -1.9, -2.0, -2.6)

        assert deviant_pages(
            angles, DeviationSettings(coefficient=1.5, threshold=0.0)
        ) == {_page(4)}
        assert (
            deviant_pages(angles, DeviationSettings(coefficient=1.5, threshold=50.0))
            == frozenset()
        )

    @pytest.mark.parametrize(
        ("coefficient", "outside"), [(1.0, 0.3173), (2.0, 0.0455), (3.0, 0.0027)]
    )
    def test_the_coefficient_counts_standard_deviations_of_normal_values(
        self, coefficient, outside
    ):
        draws = np.random.default_rng(275).normal(10.0, 2.0, 20_000)
        values = {_page(number): float(value) for number, value in enumerate(draws)}

        marked = deviant_pages(
            values, DeviationSettings(coefficient=coefficient, threshold=0.0)
        )

        assert len(marked) / len(values) == pytest.approx(outside, abs=0.01)

    def test_a_page_without_a_value_deviates_only_when_asked(self):
        values = _values(10.0, 10.0, math.nan, 10.0)

        assert deviant_pages(values, LOOSE) == frozenset()
        assert deviant_pages(values, LOOSE, unknown_deviates=True) == {_page(2)}


class TestStageRules:
    @pytest.mark.parametrize(
        ("stage", "field"),
        [
            (FilterStage.DESKEW, "deskew_deviation"),
            (FilterStage.SELECT_CONTENT, "select_content_deviation"),
            (FilterStage.PAGE_LAYOUT, "margins_deviation"),
        ],
    )
    def test_each_stage_uses_its_own_coefficients(self, stage, field):
        settings = ApplicationSettings()

        rule = stage_rule(stage, settings)

        assert rule is not None
        assert rule[1] is getattr(settings, field)

    @pytest.mark.parametrize(
        "stage",
        [FilterStage.FIX_ORIENTATION, FilterStage.PAGE_SPLIT, FilterStage.OUTPUT],
    )
    def test_other_stages_mark_no_page(self, stage):
        assert stage_rule(stage, ApplicationSettings()) is None

    def test_an_empty_content_box_deviates_at_select_content(self, sequence, pages):
        for page in pages:
            sequence.select_content_filter.set_params(
                page,
                ContentParams(
                    content_size_mm=PhysicalSize(width_mm=150, height_mm=200)
                ),
            )
        sequence.select_content_filter.set_params(pages[2], ContentParams())
        entries = [PageEntry(page, 500, 700, row) for row, page in enumerate(pages)]

        found = deviation_provider(sequence)(
            entries, str(int(FilterStage.SELECT_CONTENT))
        )

        assert found == {pages[2]}

    def test_incomplete_pages_are_left_out_of_the_comparison(self, sequence, pages):
        _set_angles(sequence, pages, [0.0] * COUNT)
        entries = [
            PageEntry(page, 500, 700, row, incomplete=row == 3)
            for row, page in enumerate(pages)
        ]
        sequence.deskew_filter.settings.set_params(
            pages[3], DeskewParams(deskew_angle_deg=9.0)
        )

        assert deviation_provider(sequence)(entries, DESKEW) == frozenset()

    def test_switching_highlighting_off_marks_no_page(self, sequence, pages):
        _set_angles(sequence, pages, [0.0, 0.1, 0.0, 0.1, 0.0, 6.0])
        entries = [PageEntry(page, 500, 700, row) for row, page in enumerate(pages)]
        settings = ApplicationSettings()
        provider = deviation_provider(sequence, lambda: settings)
        assert provider(entries, DESKEW) == {pages[5]}

        settings.highlight_deviation = False

        assert provider(entries, DESKEW) == frozenset()


class TestShortBookWithOneOutlier:
    """Five pages, one of them off, default settings."""

    @staticmethod
    def _entries(pages: list[PageId]) -> list[PageEntry]:
        return [PageEntry(page, 500, 700, row) for row, page in enumerate(pages[:5])]

    def test_deskew_marks_only_the_skewed_page(self, sequence, pages):
        _set_angles(sequence, pages, [0.2, 0.25, 0.15, 0.2, 3.0])

        assert deviation_provider(sequence)(self._entries(pages), DESKEW) == {pages[4]}

    def test_select_content_marks_only_the_small_content_box(self, sequence, pages):
        for page, diagonal in zip(pages, (249.0, 250.0, 251.0, 250.0, 200.0)):
            sequence.select_content_filter.set_params(
                page,
                ContentParams(
                    content_size_mm=PhysicalSize(
                        width_mm=0.6 * diagonal, height_mm=0.8 * diagonal
                    )
                ),
            )

        found = deviation_provider(sequence)(
            self._entries(pages), str(int(FilterStage.SELECT_CONTENT))
        )

        assert found == {pages[4]}

    def test_margins_mark_only_the_page_with_wide_margins(self, sequence, pages):
        for page, margin in zip(pages, (10.0, 10.1, 9.9, 10.0, 20.0)):
            sequence.page_layout_filter.set_margins(page, Margins.uniform(margin))

        found = deviation_provider(sequence)(
            self._entries(pages), str(int(FilterStage.PAGE_LAYOUT))
        )

        assert found == {pages[4]}


class TestDeviationOrderAgreesWithTheMarks:
    def test_the_marked_pages_sort_first_in_a_short_book(
        self, qtbot, project, sequence, pages
    ):
        # The mean of these angles is 2.0, so ranking from the mean would put
        # the marked 2.0 page last, below the four level pages.
        _set_angles(sequence, pages, [0.0, 0.0, 0.0, 0.0, 2.0, 10.0])
        widget = _widget(qtbot, project, sequence, _Renders())

        widget.set_order(1)

        assert widget.pages()[:2] == [pages[5], pages[4]]
        assert _flags(widget, widget.pages()) == [True, True] + [False] * 4

    def test_an_empty_content_box_sorts_with_the_marked_pages(
        self, qtbot, project, sequence, pages
    ):
        for page in pages:
            sequence.select_content_filter.set_params(
                page,
                ContentParams(
                    content_size_mm=PhysicalSize(width_mm=150, height_mm=200)
                ),
            )
        sequence.select_content_filter.set_params(pages[3], ContentParams())
        widget = ThumbnailListWidget(
            _Renders(), project, stage=str(int(FilterStage.SELECT_CONTENT))
        )
        qtbot.addWidget(widget)
        widget.bind_stage_sequence(sequence)

        widget.set_order(widget.order_names().index("Order by decreasing deviation"))

        assert widget.pages()[0] == pages[3]


class TestList:
    def test_the_outlying_page_is_flagged(self, qtbot, project, sequence, pages):
        _set_angles(sequence, pages, [0.0, 0.1, 0.0, 0.1, 0.0, 6.0])

        widget = _widget(qtbot, project, sequence, _Renders())

        assert _flags(widget, pages) == [False] * 5 + [True]

    def test_a_changed_page_moves_the_flags_of_the_others(
        self, qtbot, project, sequence, pages
    ):
        _set_angles(sequence, pages, [0.0, 0.1, 0.0, 0.1, 0.0, 6.0])
        widget = _widget(qtbot, project, sequence, _Renders())

        sequence.deskew_filter.settings.set_params(
            pages[5], DeskewParams(deskew_angle_deg=0.1)
        )
        sequence.deskew_filter.settings.set_params(
            pages[0], DeskewParams(deskew_angle_deg=-7.0)
        )
        widget.invalidate(pages[5])

        assert _flags(widget, pages) == [True] + [False] * 5

    def test_refreshing_the_marks_asks_for_no_new_thumbnail(
        self, qtbot, project, sequence, pages
    ):
        _set_angles(sequence, pages, [0.0, 0.1, 0.0, 0.1, 0.0, 6.0])
        widget = _widget(qtbot, project, sequence, _Renders())
        changed_roles: list[int] = []
        widget.model.dataChanged.connect(
            lambda _first, _last, roles: changed_roles.extend(roles)
        )

        ApplicationSettings.get_instance().highlight_deviation = False
        widget.refresh_marks()

        assert _flags(widget, pages) == [False] * COUNT
        assert changed_roles == [DeviantRole]


def _cell(widget: ThumbnailListWidget, page: PageId) -> QImage:
    delegate = widget.view.itemDelegate()
    assert isinstance(delegate, ThumbnailDelegate)
    size = delegate.cell_size()
    image = QImage(size, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.white)
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, size.width(), size.height())  # type: ignore[attr-defined]
    option.palette = widget.palette()  # type: ignore[attr-defined]
    painter = QPainter(image)
    delegate.paint(painter, option, widget.model.index(widget.model.row_of(page)))
    painter.end()
    return image


def _red_pixels(image: QImage) -> int:
    rgba = image.convertToFormat(QImage.Format.Format_RGBA8888)
    pixels = np.frombuffer(
        rgba.constBits(), dtype=np.uint8, count=rgba.sizeInBytes()
    ).reshape(rgba.height(), rgba.bytesPerLine() // 4, 4)[:, : rgba.width()]
    red, green, blue = (pixels[..., channel].astype(int) for channel in range(3))
    return int(((red > green + 60) & (red > blue + 60)).sum())


class TestAsterisk:
    def test_a_deviating_page_carries_a_red_asterisk(
        self, qtbot, project, sequence, pages
    ):
        _set_angles(sequence, pages, [0.0, 0.1, 0.0, 0.1, 0.0, 6.0])
        widget = _widget(qtbot, project, sequence, _Renders())

        assert _red_pixels(_cell(widget, pages[5])) > 50
        assert _red_pixels(_cell(widget, pages[0])) == 0

    def test_no_asterisk_while_highlighting_is_off(
        self, qtbot, project, sequence, pages
    ):
        ApplicationSettings.get_instance().highlight_deviation = False
        _set_angles(sequence, pages, [0.0, 0.1, 0.0, 0.1, 0.0, 6.0])
        widget = _widget(qtbot, project, sequence, _Renders())

        assert _red_pixels(_cell(widget, pages[5])) == 0
