"""A run splits the sample book's two-page spreads into the pages printed on them.

Each spread in ``tools/sample_book.py`` holds two text pages either side of a
spine shadow, so every half of a correct split is a portrait page carrying the
text of one sheet only.
"""

import shutil
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from scantailor.cli.create import create_project
from scantailor.cli.operations import project_status, run_project
from scantailor.cli.project_file import load_project
from scantailor.core.models import SubPage
from scantailor.core.project import Project

# Each module fixture runs the whole pipeline once, about 20 s; one worker keeps it
# from running again on a second one.
pytestmark = [pytest.mark.slow, pytest.mark.xdist_group("sample-book-spreads")]

SPREADS: Final = ("02_spread_light_gutter", "03_spread_black_gutter")
# The half-width of the black spine shadow that sample_book draws.
SPINE_SHADOW_HALF_WIDTH: Final = 70
# Normalised correlation by which a half must match its own sheet's text block
# better than the facing sheet's. Measured margins are 0.39 to 0.78.
OWN_PAGE_MARGIN: Final = 0.25


def _create_and_run(source: Path, folder: Path) -> Path:
    project_path = folder / "project.json"
    create_project(
        source,
        project_path,
        dpi=None,
        layout_direction="LTR",
        output_directory=folder / "out",
        pages_directory=folder / "pages",
        force=False,
    )
    report = run_project(project_path, max_workers=1)
    assert not report.failed
    return project_path


@pytest.fixture(scope="module")
def pdf_project(sample_book: Any, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A project made from the sample PDF and run through every stage."""
    return _create_and_run(sample_book.pdf, tmp_path_factory.mktemp("pdf_project"))


@pytest.fixture(scope="module")
def spreads_project(sample_book: Any, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A project made from a scans folder of both sample spreads, run once."""
    folder = tmp_path_factory.mktemp("spreads_project")
    scans = folder / "scans"
    scans.mkdir()
    for path in sample_book.pages:
        if path.stem in SPREADS:
            shutil.copy(path, scans)
    return _create_and_run(scans, folder)


def _output_names(project_path: Path) -> list[str]:
    out_dir = Path(project_status(project_path).output_directory)
    return sorted(path.name for path in out_dir.glob("*.tif"))


def _outputs_of(project_path: Path, stem: str) -> list[Path]:
    out_dir = Path(project_status(project_path).output_directory)
    return sorted(out_dir.glob(f"{stem}*.tif"))


def _grey_at_sample_dpi(path: Path, sample_dpi: int) -> NDArray[np.uint8]:
    """Load an output page as greyscale, scaled to the sample book's resolution."""
    with Image.open(path) as image:
        scale = float(sample_dpi / image.info["dpi"][0])
        grey = np.asarray(image.convert("L"))
    scaled = cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return np.asarray(scaled, dtype=np.uint8)


def _text_blocks(tool: ModuleType, stem: str) -> tuple[NDArray[np.uint8], ...]:
    """Crop the left and right sheets' text blocks from the rendered spread.

    The text block sits centred on its sheet, so its right edge mirrors the left
    edge of the sheet's leftmost line.
    """
    page = tool.render_sample_page(stem)
    grey = cv2.cvtColor(page.image, cv2.COLOR_RGB2GRAY)
    blocks = []
    for sheet_left in (tool.BED_MARGIN, tool.BED_MARGIN + tool.PAPER_WIDTH):
        sheet_right = sheet_left + tool.PAPER_WIDTH
        lines = [line for line in page.text_lines if sheet_left <= line.x < sheet_right]
        left = min(line.x for line in lines)
        right = sheet_right - (left - sheet_left)
        top = min(line.baseline for line in lines) - 30
        bottom = max(line.baseline for line in lines) + 12
        blocks.append(grey[top:bottom, left:right])
    return tuple(blocks)


def _best_match(page: NDArray[np.uint8], block: NDArray[np.uint8]) -> float:
    if block.shape[0] > page.shape[0] or block.shape[1] > page.shape[1]:
        return 0.0
    scores = cv2.matchTemplate(page, block, cv2.TM_CCOEFF_NORMED)
    return float(scores.max())


def test_the_sample_pdf_runs_to_one_page_two_halves_and_one_page(
    pdf_project: Path,
) -> None:
    assert _output_names(pdf_project) == [
        "page_0001.tif",
        "page_0002_1L.tif",
        "page_0002_2R.tif",
        "page_0003.tif",
    ]


def test_the_saved_sample_pdf_project_lists_four_pages(pdf_project: Path) -> None:
    pages = Project.load(pdf_project).get_pages()

    assert [page.sub_page for page in pages] == [
        SubPage.SINGLE_PAGE,
        SubPage.LEFT_PAGE,
        SubPage.RIGHT_PAGE,
        SubPage.SINGLE_PAGE,
    ]
    assert project_status(pdf_project).page_count == 4


def test_a_folder_of_two_spreads_runs_to_a_left_and_right_page_each(
    spreads_project: Path,
) -> None:
    assert _output_names(spreads_project) == [
        f"{stem}{suffix}.tif" for stem in SPREADS for suffix in ("_1L", "_2R")
    ]


@pytest.mark.parametrize("stem", SPREADS)
def test_each_output_of_a_spread_is_a_portrait_page_with_ink(
    spreads_project: Path, stem: str
) -> None:
    outputs = _outputs_of(spreads_project, stem)

    for path in outputs:
        with Image.open(path) as image:
            width, height = image.size
            ink = np.asarray(image.convert("L")) < 128
        # An A5 sheet is 0.68 wide per unit of height; the whole spread is 1.35.
        assert width / height < 1.0, f"{path.name} is {width}x{height}"
        assert ink.mean() > 0.01, f"{path.name} has no text"
    assert len(outputs) == 2, [path.name for path in outputs]


@pytest.mark.parametrize("stem", SPREADS)
def test_each_half_of_a_spread_carries_its_own_sheets_text_block(
    spreads_project: Path, sample_book_tool: ModuleType, stem: str
) -> None:
    blocks = dict(
        zip(("_1L", "_2R"), _text_blocks(sample_book_tool, stem), strict=True)
    )

    sides = {}
    for path in _outputs_of(spreads_project, stem):
        page = _grey_at_sample_dpi(path, sample_book_tool.SAMPLE_DPI)
        scores = {side: _best_match(page, block) for side, block in blocks.items()}
        best, other = sorted(scores, key=scores.__getitem__, reverse=True)
        assert scores[best] - scores[other] > OWN_PAGE_MARGIN, (
            f"{path.name} matches both sheets: {scores}"
        )
        sides[path.stem] = best
    assert sides == {f"{stem}_1L": "_1L", f"{stem}_2R": "_2R"}


def test_the_black_gutter_spread_is_cut_inside_its_spine_shadow(
    spreads_project: Path, sample_book_tool: ModuleType
) -> None:
    project, stages = load_project(spreads_project)
    pages = [
        page
        for page in project.get_pages()
        if page.image_id.file_path.stem == "03_spread_black_gutter"
    ]
    layout = stages.page_split_filter.settings.get_params(pages[0]).page_layout
    spine = sample_book_tool.BED_MARGIN + sample_book_tool.PAPER_WIDTH

    assert [page.sub_page for page in pages] == [SubPage.LEFT_PAGE, SubPage.RIGHT_PAGE]
    assert layout is not None and layout.cutter1 is not None
    for x, _ in layout.cutter1:
        assert abs(x - spine) <= SPINE_SHADOW_HALF_WIDTH, f"cut at x={x:.0f}"
