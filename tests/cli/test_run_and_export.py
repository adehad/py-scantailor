"""A full run on two generated pages, then the PDF and TIFF exports of its output."""

from pathlib import Path

import pikepdf
import pytest

from scantailor.cli.errors import CliError, ExitCode
from scantailor.cli.operations import (
    export_pdf_file,
    export_tiff_files,
    project_status,
    run_project,
)


def test_export_before_run_has_nothing_to_write(
    project_path: Path, tmp_path: Path
) -> None:
    with pytest.raises(CliError) as refused:
        export_pdf_file(project_path, tmp_path / "book.pdf", "jbig2")

    assert refused.value.code == ExitCode.NOTHING_TO_EXPORT
    assert not (tmp_path / "book.pdf").exists()


def test_tiff_export_before_run_has_nothing_to_copy(
    project_path: Path, tmp_path: Path
) -> None:
    with pytest.raises(CliError) as refused:
        export_tiff_files(project_path, tmp_path / "tiffs")

    assert refused.value.code == ExitCode.NOTHING_TO_EXPORT


@pytest.mark.slow
def test_run_writes_one_tiff_per_page_and_status_reports_them(
    project_path: Path,
) -> None:
    report = run_project(project_path, max_workers=1)

    assert report.written == 2
    assert report.failed == {}
    assert report.cancelled == 0
    status = project_status(project_path)
    assert status.outputs_present == 2
    assert all(Path(page.output or "").is_file() for page in status.pages)


@pytest.mark.slow
def test_pdf_export_after_run_has_one_page_per_scan(
    project_path: Path, tmp_path: Path
) -> None:
    run_project(project_path, max_workers=1)
    destination = tmp_path / "exports" / "book.pdf"

    report = export_pdf_file(project_path, destination, "jbig2")

    assert report.exported == 2
    assert report.skipped == []
    with pikepdf.open(destination) as pdf:
        assert len(pdf.pages) == 2


@pytest.mark.slow
def test_tiff_export_after_run_copies_the_processed_files(
    project_path: Path, tmp_path: Path
) -> None:
    run_project(project_path, max_workers=1)
    destination = tmp_path / "tiffs"

    report = export_tiff_files(project_path, destination)

    assert report.exported == 2
    assert sorted(path.name for path in destination.iterdir()) == [
        "page_01.tif",
        "page_02.tif",
    ]
