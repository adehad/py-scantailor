"""Sort options, removed halves and picture mask wiring of the main window."""

from pathlib import Path

import pytest
from pytestqt.qtbot import QtBot

from scantailor.app.ui.MainWindow import MainWindowManager
from scantailor.core.models import SubPage
from scantailor.core.stage_sequence import FilterStage


@pytest.fixture
def settled_window(qtbot: QtBot, project_window: MainWindowManager):
    project_window.renderer.wait_for_done()
    qtbot.wait(50)
    return project_window


def _show_stage(window: MainWindowManager, stage: FilterStage) -> None:
    window._select_stage_row(int(stage))
    window.renderer.wait_for_done()


def test_sort_options_list_the_shown_stages_orders(
    settled_window: MainWindowManager,
) -> None:
    window = settled_window
    _show_stage(window, FilterStage.PAGE_SPLIT)

    names = [window.sortOptions.itemText(i) for i in range(window.sortOptions.count())]

    assert names == window.thumbnails.order_names()
    assert names
    assert window.sortOptions.isVisibleTo(window.ui)


def test_the_direction_button_reverses_the_thumbnails(
    settled_window: MainWindowManager,
) -> None:
    window = settled_window
    window.sortingOrderBtn.setChecked(True)

    assert window.thumbnails.sort_reversed
    window.sortingOrderBtn.setChecked(False)
    assert not window.thumbnails.sort_reversed


def test_a_stage_without_sort_options_hides_the_sort_controls(
    settled_window: MainWindowManager,
) -> None:
    window = settled_window
    window.thumbnails.set_orders([])

    assert window.sortOptions.count() == 0
    assert not window.sortOptions.isVisibleTo(window.ui)
    assert not window.sortingOrderBtn.isVisibleTo(window.ui)


def test_restoring_a_removed_half_brings_both_pages_back(
    settled_window: MainWindowManager,
) -> None:
    window = settled_window
    project = window.project
    assert project is not None
    image = project.images[0]
    project.images[0] = image.model_copy(
        update={"num_sub_pages": 1, "left_half_removed": True}
    )
    window._refresh_page_list()
    assert window._removed_halves_of(image.id) == (True, False)

    window._on_unremove_page_requested(
        next(p for p in project.get_pages() if p.image_id == image.id)
    )

    assert window._removed_halves_of(image.id) == (False, False)
    halves = [p.sub_page for p in window.pages() if p.image_id == image.id]
    assert halves == [SubPage.LEFT_PAGE, SubPage.RIGHT_PAGE]
    assert window.modified


def test_an_unknown_image_has_no_removed_halves(
    settled_window: MainWindowManager,
) -> None:
    from scantailor.core.models import ImageId

    assert settled_window._removed_halves_of(ImageId(file_path=Path("x.png"))) == (
        False,
        False,
    )


def test_render_pending_reflects_the_pending_render(
    settled_window: MainWindowManager,
) -> None:
    window = settled_window
    window._render_pending = True
    assert window.render_pending
    window._render_pending = False
    assert not window.render_pending


def test_a_legacy_sidecar_mapping_is_adopted_and_removed_after_saving(
    settled_window: MainWindowManager, tmp_path: Path
) -> None:
    from scantailor.core.pdf_source import SOURCE_FILE_NAME, PdfSource

    window = settled_window
    project = window.project
    assert project is not None
    directory = project.images[0].id.file_path.parent
    sidecar = directory / SOURCE_FILE_NAME
    source = PdfSource(pdf_path=Path("book.pdf"), dpi=300, pages={})
    sidecar.write_text(source.model_dump_json(), encoding="utf-8")
    project.pdf_source = None

    window._install_project(project, window.stages, project_file=None, modified=False)
    assert window.pdf_source == source
    assert window.modified
    assert sidecar.exists()

    assert window._write_project(tmp_path / "p.ScanTailor")
    assert not sidecar.exists()


def test_a_page_a_batch_wrote_is_not_incomplete_at_output(
    settled_window: MainWindowManager,
) -> None:
    window = settled_window
    project = window.project
    assert project is not None
    page = window.pages()[0]
    stage_key = str(int(FilterStage.OUTPUT))
    assert not window.stages.output_filter.is_params_set(page)
    assert window._page_incomplete(page, stage_key)

    project.set_text_geometry(page, {"written": True})

    assert not window._page_incomplete(page, stage_key)
    assert window._page_incomplete(window.pages()[1], stage_key)
