"""Status, stage options and project round trips, without running the batch."""

import json
from pathlib import Path

import pytest

from scantailor.cli.errors import CliError, ExitCode
from scantailor.cli.operations import project_status, set_option, show_options
from scantailor.cli.project_file import load_project, page_name, save_project
from scantailor.core.models import ImageId, PageId, SubPage


def test_status_before_any_run_lists_both_pages_with_no_output(
    project_path: Path,
) -> None:
    status = project_status(project_path)

    assert status.page_count == 2
    assert status.outputs_present == 0
    assert [page.page for page in status.pages] == ["page_01.png", "page_02.png"]
    assert all(not page.output_exists for page in status.pages)
    assert all(page.output and page.output.endswith(".tif") for page in status.pages)


def test_status_json_has_the_documented_top_level_keys(project_path: Path) -> None:
    document = json.loads(project_status(project_path).model_dump_json())

    assert set(document) == {
        "project",
        "output_directory",
        "layout_direction",
        "pdf_sources",
        "page_count",
        "outputs_present",
        "stage_params_set",
        "pages",
    }
    assert document["layout_direction"] == "LTR"
    assert document["pdf_sources"] == []


def test_options_show_reports_stage_defaults_by_name(project_path: Path) -> None:
    defaults = show_options(project_path).defaults

    assert defaults["output_color_mode"] == "bw"
    assert "output_dpi" in defaults


def test_set_option_saves_a_valid_value(project_path: Path) -> None:
    report = set_option(project_path, "output_color_mode", "colorOrGray")

    assert report.defaults["output_color_mode"] == "colorOrGray"
    assert show_options(project_path).defaults["output_color_mode"] == "colorOrGray"


def test_set_option_reads_json_values(project_path: Path) -> None:
    report = set_option(project_path, "page_detection_tolerance", "0.25")

    assert report.defaults["page_detection_tolerance"] == pytest.approx(0.25)


def test_set_option_rejects_a_bad_value_and_leaves_the_file_untouched(
    project_path: Path,
) -> None:
    before = project_path.read_bytes()

    with pytest.raises(CliError) as refused:
        set_option(project_path, "output_color_mode", "not-a-mode")

    assert refused.value.code == ExitCode.OPTION
    assert project_path.read_bytes() == before


def test_set_option_rejects_an_unknown_name_and_lists_the_known_ones(
    project_path: Path,
) -> None:
    with pytest.raises(CliError) as refused:
        set_option(project_path, "no_such_option", "1")

    assert refused.value.code == ExitCode.OPTION
    assert "output_color_mode" in refused.value.message


def test_saved_project_reloads_to_the_same_document(
    tmp_path: Path, project_path: Path
) -> None:
    set_option(project_path, "output_color_mode", "colorOrGray")
    project, stages = load_project(project_path)
    copy = tmp_path / "proj" / "copy.json"

    save_project(project, stages, copy)

    original = json.loads(project_path.read_text(encoding="utf-8"))
    reloaded = json.loads(copy.read_text(encoding="utf-8"))
    assert reloaded == original


def test_load_refuses_a_missing_project_file(tmp_path: Path) -> None:
    with pytest.raises(CliError) as refused:
        load_project(tmp_path / "absent.json")

    assert refused.value.code == ExitCode.PROJECT


def test_load_refuses_a_file_that_is_not_a_project(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{ not json", encoding="utf-8")

    with pytest.raises(CliError) as refused:
        load_project(broken)

    assert refused.value.code == ExitCode.PROJECT


def test_page_name_marks_the_half_of_a_split_page() -> None:
    image = ImageId(file_path=Path("scan.tif"), page=2)

    assert page_name(PageId(image_id=image)) == "scan.tif#2"
    assert page_name(PageId(image_id=image, sub_page=SubPage.LEFT_PAGE)) == (
        "scan.tif#2/left_page"
    )
