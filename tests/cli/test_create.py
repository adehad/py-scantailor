"""Creating projects from a folder of scans or from a PDF."""

from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image

from scantailor.cli.create import create_project
from scantailor.cli.errors import CliError, ExitCode
from scantailor.core.project import LayoutDirection, Project


def _create(
    source: Path,
    project: Path,
    *,
    dpi: int | None = None,
    layout_direction: LayoutDirection = "LTR",
    force: bool = False,
) -> None:
    create_project(
        source,
        project,
        dpi=dpi,
        layout_direction=layout_direction,
        output_directory=None,
        pages_directory=None,
        force=force,
    )


def test_folder_project_lists_pages_in_filename_order(project_path: Path) -> None:
    project = Project.load(project_path)

    names = [page.image_id.file_path.name for page in project.get_pages()]
    assert names == ["page_01.png", "page_02.png"]


def test_folder_project_puts_output_in_out_folder_beside_scans(
    project_path: Path, sample_scans: Path
) -> None:
    project = Project.load(project_path)

    assert project.output_directory.resolve() == (sample_scans / "out").resolve()
    assert project.output_directory.is_dir()


def test_dpi_option_overrides_the_resolution_of_every_image(
    tmp_path: Path, sample_scans: Path
) -> None:
    path = tmp_path / "dpi.json"

    _create(sample_scans, path, dpi=600)

    dpis = {image.metadata.dpi.horizontal for image in Project.load(path).images}
    assert dpis == {600}


def test_folder_without_dpi_is_refused_until_dpi_is_given(
    tmp_path: Path, page_writer: Callable[..., Path]
) -> None:
    folder = tmp_path / "no_dpi"
    folder.mkdir()
    page_writer(folder / "a.png", 0, None)

    with pytest.raises(CliError) as refused:
        _create(folder, tmp_path / "p.json")

    assert refused.value.code == ExitCode.INPUT
    _create(folder, tmp_path / "p.json", dpi=300)
    assert Project.load(tmp_path / "p.json").images[0].metadata.dpi.horizontal == 300


def test_existing_project_is_kept_unless_force_is_given(
    project_path: Path, sample_scans: Path
) -> None:
    with pytest.raises(CliError) as refused:
        _create(sample_scans, project_path)

    assert refused.value.code == ExitCode.PROJECT
    _create(sample_scans, project_path, force=True)
    assert len(Project.load(project_path).images) == 2


def test_empty_folder_is_an_input_error(tmp_path: Path) -> None:
    folder = tmp_path / "empty"
    folder.mkdir()

    with pytest.raises(CliError) as refused:
        _create(folder, tmp_path / "p.json")

    assert refused.value.code == ExitCode.INPUT
    assert "no image files" in refused.value.message


def test_missing_source_is_an_input_error(tmp_path: Path) -> None:
    with pytest.raises(CliError) as refused:
        _create(tmp_path / "absent", tmp_path / "p.json")

    assert refused.value.code == ExitCode.INPUT


def test_non_positive_dpi_is_an_option_error(
    tmp_path: Path, sample_scans: Path
) -> None:
    with pytest.raises(CliError) as refused:
        _create(sample_scans, tmp_path / "p.json", dpi=0)

    assert refused.value.code == ExitCode.OPTION


def test_pdf_source_renders_pages_beside_the_pdf_and_records_the_source(
    tmp_path: Path, sample_scans: Path
) -> None:
    pages = [sample_scans / "page_01.png", sample_scans / "page_02.png"]
    pdf = tmp_path / "book.pdf"
    Image.init()  # Pillow's PDF writer needs the JPEG saver, which loads lazily
    first, *rest = [Image.open(page).convert("RGB") for page in pages]
    first.save(pdf, save_all=True, append_images=rest, resolution=300)
    path = tmp_path / "from_pdf.json"

    _create(pdf, path)

    project = Project.load(path)
    assert len(project.get_pages()) == 2
    assert [source.pdf_path.name for source in project.pdf_sources] == ["book.pdf"]
    rendered = sorted((tmp_path / "book_pages").glob("*.png"))
    assert len(rendered) == 2


def test_unreadable_pdf_is_an_input_error(tmp_path: Path) -> None:
    pdf = tmp_path / "broken.pdf"
    pdf.write_bytes(b"not a pdf at all")

    with pytest.raises(CliError) as refused:
        _create(pdf, tmp_path / "p.json")

    assert refused.value.code == ExitCode.INPUT
