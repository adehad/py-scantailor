"""Automatic dewarping traces the page edges on the source image, as upstream does.

The page as laid out for output shows the text but not the paper's edges, which
lie outside the content and its margins. A page with few full-width text lines
gets its model from the paper edges, so those must be traced on the source.
"""

from pathlib import Path

import numpy as np

from scantailor.core.models import Dpi, ImageId, PageId, SubPage
from scantailor.core.pipeline import PipelineOptions, process_page
from scantailor.core.stage_sequence import StageSequence
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.dewarping_options import DewarpingMode, DewarpingOptions
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.select_content.content_box import ContentBox
from tests.dewarping.book_pages import (
    DEFAULT_PAPER,
    BentPage,
    bend_page,
    justified_text_page,
    sagging_edges,
)

DPI = 200
# Content of the bent page, in its pixels: the text block without the paper edges.
CONTENT = ContentBox(x=160, y=200, width=890, height=1090)
# Share of the text width blanked on each side of a short line.
_SHORT_LINE_TRIM = 0.27


def page_with_two_full_lines() -> BentPage:
    """A bent page whose text lines, bar the first two, are short and centred.

    The tracer drops lines that stop far from the content bounds, so the text
    alone yields a model between the first two lines only.
    """
    flat, baselines = justified_text_page(seed=3)
    margin = 70
    text_width = flat.shape[1] - 2 * margin
    trim = int(_SHORT_LINE_TRIM * text_width)
    for baseline in baselines[2:]:
        rows = slice(baseline - 32, baseline + 12)
        flat[rows, margin : margin + trim] = DEFAULT_PAPER
        flat[rows, margin + text_width - trim : margin + text_width] = DEFAULT_PAPER
    top, bottom = sagging_edges(100.0, 1100.0, 140.0, 1450.0, 45.0, -25.0)
    return bend_page(flat, baselines, margin, top, bottom, (1200, 1600))


def _page_id() -> PageId:
    return PageId(
        image_id=ImageId(file_path=Path("edges.tiff")), sub_page=SubPage.SINGLE_PAGE
    )


def _auto_dewarp_sequence(page_id: PageId) -> StageSequence:
    sequence = StageSequence()
    sequence.page_split_filter.set_layout_type(page_id, LayoutType.SINGLE_PAGE_UNCUT)
    sequence.deskew_filter.set_manual_angle(page_id, 0.0)
    sequence.select_content_filter.set_manual_content(page_id, CONTENT)
    params = sequence.output_filter.get_params(page_id).model_copy(
        update={
            "color_mode": ColorMode.COLOR_GRAYSCALE,
            "output_dpi": Dpi.uniform(DPI),
            "dewarping": DewarpingOptions(mode=DewarpingMode.AUTO, post_deskew=False),
        }
    )
    sequence.output_filter.set_params(page_id, params)
    return sequence


def _distance_to_edge(
    curve: tuple[tuple[float, float], ...], edge: np.ndarray
) -> float:
    points = np.asarray(curve)
    expected = np.interp(points[:, 0], edge[:, 0], edge[:, 1])
    return float(np.abs(points[:, 1] - expected).max())


class TestPageEdgesComeFromTheSourceImage:
    def test_the_model_reaches_the_bottom_edge_of_the_paper(self) -> None:
        page = page_with_two_full_lines()
        page_id = _page_id()

        result = process_page(
            page.image,
            page_id,
            _auto_dewarp_sequence(page_id),
            PipelineOptions(dpi=DPI),
        )

        model = result.distortion_model
        assert model is not None
        assert _distance_to_edge(model.bottom, page.bottom) < 20.0

    def test_the_model_reaches_the_top_edge_of_the_paper(self) -> None:
        page = page_with_two_full_lines()
        page_id = _page_id()

        result = process_page(
            page.image,
            page_id,
            _auto_dewarp_sequence(page_id),
            PipelineOptions(dpi=DPI),
        )

        model = result.distortion_model
        assert model is not None
        assert _distance_to_edge(model.top, page.top) < 20.0
