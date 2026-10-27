"""A run splits a two-page spread into a left and a right output, as the GUI does."""

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from scantailor.cli.create import create_project
from scantailor.cli.operations import project_status, run_project, set_option
from scantailor.cli.project_file import (
    load_project,
    output_directory_of,
    save_project,
)
from scantailor.core.batch import run_batch
from scantailor.core.models import SubPage
from scantailor.core.page_source import dpi_overrides_of
from scantailor.core.project import Project

FIXTURE_SCANS = (
    Path(__file__).parents[1] / "fixtures/xml_projects/split_and_mixed/scans"
)


def _create(scans: Path, tmp_path: Path) -> Path:
    path = tmp_path / "proj" / "project.json"
    create_project(
        scans,
        path,
        dpi=None,
        layout_direction="LTR",
        output_directory=None,
        pages_directory=None,
        force=False,
    )
    return path


@pytest.fixture
def spread_project(tmp_path: Path) -> Path:
    """A project made from the fixture spread alone, with no batch run yet."""
    scans = tmp_path / "scans"
    scans.mkdir()
    shutil.copy(FIXTURE_SCANS / "spread.tif", scans)
    return _create(scans, tmp_path)


def _output_names(project_path: Path) -> list[str]:
    status = project_status(project_path)
    return sorted(path.name for path in Path(status.output_directory).iterdir())


def test_run_writes_a_detected_spread_as_left_and_right_pages(
    spread_project: Path,
) -> None:
    report = run_project(spread_project, max_workers=1)

    assert report.written == 2
    assert _output_names(spread_project) == ["spread_1L.tif", "spread_2R.tif"]


def test_the_saved_project_lists_both_halves_of_the_spread(
    spread_project: Path,
) -> None:
    run_project(spread_project, max_workers=1)

    pages = Project.load(spread_project).get_pages()
    assert [page.sub_page for page in pages] == [SubPage.LEFT_PAGE, SubPage.RIGHT_PAGE]
    status = project_status(spread_project)
    assert status.page_count == 2
    assert status.outputs_present == 2


@pytest.mark.slow
def test_a_two_pages_layout_setting_splits_the_spread(spread_project: Path) -> None:
    set_option(spread_project, "page_split_layout_type", "two-pages")

    report = run_project(spread_project, max_workers=1)

    assert report.written == 2
    assert _output_names(spread_project) == ["spread_1L.tif", "spread_2R.tif"]


@pytest.mark.slow
def test_a_second_run_keeps_both_halves_of_the_spread(spread_project: Path) -> None:
    run_project(spread_project, max_workers=1)

    report = run_project(spread_project, max_workers=1)

    assert report.kept == 2
    assert _output_names(spread_project) == ["spread_1L.tif", "spread_2R.tif"]


def test_run_writes_one_output_for_a_single_page_scan(
    tmp_path: Path, page_writer: Callable[[Path, int, tuple[int, int] | None], Path]
) -> None:
    scans = tmp_path / "scans"
    scans.mkdir()
    page_writer(scans / "page_01.png", 0, (300, 300))
    project_path = _create(scans, tmp_path)

    report = run_project(project_path, max_workers=1)

    assert report.written == 1
    assert _output_names(project_path) == ["page_01.tif"]
    assert project_status(project_path).page_count == 1


def _run_as_one_page_per_scan(project_path: Path) -> None:
    """Save the outputs and records of a batch that never split a spread."""
    project, stages = load_project(project_path)
    out_dir = output_directory_of(project, project_path)
    result = run_batch(
        stages,
        project.get_pages(),
        out_dir,
        layout_direction=project.layout_direction,
        max_workers=1,
        dpi_overrides=dpi_overrides_of(project),
        file_labels=project.output_file_label_map(),
    )
    project.set_output_records(result.output_records)
    save_project(project, stages, project_path)


@pytest.fixture
def spread_and_single_page(
    tmp_path: Path, page_writer: Callable[[Path, int, tuple[int, int] | None], Path]
) -> Path:
    """A project of the fixture spread and one single page, run unsplit once."""
    scans = tmp_path / "scans"
    scans.mkdir()
    page_writer(scans / "page_01.png", 0, (300, 300))
    shutil.copy(FIXTURE_SCANS / "spread.tif", scans)
    project_path = _create(scans, tmp_path)
    _run_as_one_page_per_scan(project_path)
    return project_path


@pytest.mark.slow
def test_a_rerun_of_an_unsplit_project_leaves_only_the_two_halves(
    spread_and_single_page: Path,
) -> None:
    assert "spread.tif" in _output_names(spread_and_single_page)

    run_project(spread_and_single_page, max_workers=1)

    assert _output_names(spread_and_single_page) == [
        "page_01.tif",
        "spread_1L.tif",
        "spread_2R.tif",
    ]
    recorded = {
        record.page.sub_page
        for record in Project.load(spread_and_single_page).output_records
        if record.page.image_id.file_path.name == "spread.tif"
    }
    assert recorded == {SubPage.LEFT_PAGE, SubPage.RIGHT_PAGE}


@pytest.mark.slow
def test_a_rerun_keeps_the_file_of_an_unchanged_single_page(
    tmp_path: Path,
    page_writer: Callable[[Path, int, tuple[int, int] | None], Path],
    log_messages: list[str],
) -> None:
    scans = tmp_path / "scans"
    scans.mkdir()
    page_writer(scans / "page_01.png", 0, (300, 300))
    project_path = _create(scans, tmp_path)
    _run_as_one_page_per_scan(project_path)
    single = Path(project_status(project_path).output_directory) / "page_01.tif"
    written_at = single.stat().st_mtime_ns

    report = run_project(project_path, max_workers=1)

    assert report.kept == 1
    assert single.stat().st_mtime_ns == written_at
    assert not [line for line in log_messages if line.startswith("deleted")]
