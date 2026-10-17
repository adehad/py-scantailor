"""Output settings added in 1.2.1: Fox, Window, Grad, fill outside page box."""

import shutil
from pathlib import Path

import pytest

from scantailor.core.models import PageId
from scantailor.core.project import Project
from scantailor.core.xml_compat import import_xml_project
from scantailor.filters.output.binarization import BinarizationMethod
from scantailor.filters.output.params import Params as OutputParams

FIXTURES = Path(__file__).parents[1] / "fixtures" / "xml_projects"


def _output_params(project: Project) -> dict[PageId, OutputParams]:
    return {
        entry.page: OutputParams.model_validate(entry.params)
        for entry in project.stage_params["output"]
    }


def _import(tmp_path: Path, name: str, replacements: dict[str, str]) -> Project:
    target = tmp_path / name
    shutil.copytree(FIXTURES / name, target)
    project_file = target / f"{name}.ScanTailor"
    text = project_file.read_text(encoding="utf-8")
    for old, new in replacements.items():
        assert old in text, old
        text = text.replace(old, new)
    return import_xml_project(project_file, text).project


@pytest.mark.parametrize(
    ("name", "method"),
    [
        ("fox", BinarizationMethod.FOX),
        ("window", BinarizationMethod.WINDOW),
        ("grad", BinarizationMethod.GRAD),
    ],
)
def test_wolf_family_method_names_are_read_with_the_wolf_options(
    tmp_path: Path, name: str, method: BinarizationMethod
):
    project = _import(
        tmp_path,
        "split_and_mixed",
        {'binarizationMethod="edgeplus"': f'binarizationMethod="{name}"'},
    )

    binarization = _output_params(project)[project.get_pages()[1]].binarization

    assert binarization.method is method
    assert binarization.threshold_adjustment == -10


def test_fill_outside_page_box_is_read_from_the_colour_options(tmp_path: Path):
    project = _import(tmp_path, "version4", {})

    assert all(
        params.color_common.fill_outside_page_box
        for params in _output_params(project).values()
    )


def test_pages_of_a_version_3_project_do_not_fill_outside_the_page_box(
    tmp_path: Path,
):
    project = _import(tmp_path, "simple", {})

    assert not any(
        params.color_common.fill_outside_page_box
        for params in _output_params(project).values()
    )


def test_importing_fill_outside_page_box_logs_no_dropped_setting(
    tmp_path: Path, log_messages: list[str]
):
    _import(tmp_path, "version4", {})

    assert not any("page box" in message for message in log_messages)
