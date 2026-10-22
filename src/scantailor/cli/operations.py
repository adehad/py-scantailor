"""The project commands: status, stage options, batch run and exports."""

import json
import shutil
from pathlib import Path
from typing import Any

from scantailor.cli.errors import CliError, ExitCode
from scantailor.cli.project_file import (
    load_project,
    output_directory_of,
    save_project,
)
from scantailor.cli.reports import (
    BatchReport,
    ExportReport,
    OptionsReport,
    PageStatus,
    ProjectStatus,
    absolute,
)
from scantailor.core.batch import output_file_names, run_batch
from scantailor.core.image_io import list_image_pages
from scantailor.core.models import PageId, page_name
from scantailor.core.page_source import dpi_overrides_of
from scantailor.core.pdf_export import BilevelCodec, export_pdf
from scantailor.core.project import Project
from scantailor.core.stage_params import apply_stage_defaults, collect_stage_defaults
from scantailor.core.stage_sequence import StageSequence

FALLBACK_DPI = 300.0


def project_status(project_path: Path) -> ProjectStatus:
    """Summarise a project and which of its pages have processed output.

    Args:
        project_path (pathlib.Path): The project JSON file.

    Returns:
        ProjectStatus: Counts and one entry per page.
    """
    project, _ = load_project(project_path)
    out_dir = output_directory_of(project, project_path)
    names = _output_names(project)
    pages = [_page_status(page, names, out_dir) for page in project.get_pages()]
    return ProjectStatus(
        project=absolute(project_path),
        output_directory=absolute(out_dir),
        layout_direction=project.layout_direction,
        pdf_sources=[absolute(source.pdf_path) for source in project.pdf_sources],
        page_count=len(pages),
        outputs_present=sum(page.output_exists for page in pages),
        stage_params_set={
            key: len(entries) for key, entries in project.stage_params.items()
        },
        pages=pages,
    )


def show_options(project_path: Path) -> OptionsReport:
    """Read the stage-wide defaults of a project.

    Args:
        project_path (pathlib.Path): The project JSON file.

    Returns:
        OptionsReport: Current value of every option name.
    """
    _, stages = load_project(project_path)
    return OptionsReport(defaults=collect_stage_defaults(stages))


def set_option(project_path: Path, name: str, raw_value: str) -> OptionsReport:
    """Change one stage-wide default and save the project.

    The value is read as JSON when it parses, otherwise taken as text, so
    "grayscale" and "[0.1, 0.2]" both work. The stage validates it.

    Args:
        project_path (pathlib.Path): The project JSON file.
        name (str): An option name from show_options.
        raw_value (str): The value as typed on the command line.

    Returns:
        OptionsReport: The defaults after the change.

    Raises:
        CliError: With ExitCode.OPTION for an unknown name or rejected value.
    """
    known = collect_stage_defaults(StageSequence())
    if name not in known:
        msg = f"unknown option {name!r}; known options: {', '.join(sorted(known))}"
        raise CliError(ExitCode.OPTION, msg)
    value = _parse_value(raw_value)
    project, stages = load_project(project_path)
    try:
        apply_stage_defaults(stages, {name: value})
    except ValueError as error:
        msg = f"{name} cannot be {raw_value!r}: {error}"
        raise CliError(ExitCode.OPTION, msg) from error
    save_project(project, stages, project_path)
    return OptionsReport(defaults=collect_stage_defaults(stages))


def run_project(project_path: Path, max_workers: int | None) -> BatchReport:
    """Process every page of a project through all stages and save the result.

    Pages that fail are reported and left out of the output. A page whose
    settings, source file and output files are unchanged since the last run
    keeps its output file. The project is saved either way, so detected
    parameters and the record of what was written are kept for the next run.

    Args:
        project_path (pathlib.Path): The project JSON file.
        max_workers (int | None): Worker processes; the CPU count when None.

    Returns:
        BatchReport: Counts of written, failed and cancelled pages.
    """
    project, stages = load_project(project_path)
    out_dir = output_directory_of(project, project_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_batch(
        stages,
        project.get_pages(),
        out_dir,
        layout_direction=project.layout_direction,
        max_workers=max_workers,
        dpi_overrides=dpi_overrides_of(project),
        file_labels=project.output_file_label_map(),
        output_records=project.output_record_map(),
    )
    project.set_output_records(result.output_records)
    for image_id, (width, height) in result.source_sizes.items():
        project.store_source_size(image_id, width, height)
    save_project(project, stages, project_path)
    return BatchReport(
        output_directory=absolute(out_dir),
        written=len(result.written),
        kept=len(result.reused),
        failed={page_name(page): error for page, error in result.failed.items()},
        cancelled=len(result.cancelled),
    )


def export_pdf_file(
    project_path: Path, destination: Path, bilevel_codec: BilevelCodec
) -> ExportReport:
    """Write the processed pages of a project into one PDF.

    Only the processed files are used, so the batch does not run again.
    Pages with no processed file are listed as skipped.

    Args:
        project_path (pathlib.Path): The project JSON file.
        destination (pathlib.Path): The PDF to write.
        bilevel_codec (str): "jbig2" or "g4" for black and white pages.

    Returns:
        ExportReport: The PDF path, page count and skipped pages.

    Raises:
        CliError: With ExitCode.NOTHING_TO_EXPORT when no page has output.
    """
    project, _ = load_project(project_path)
    found = _processed_files(project, project_path)
    included = [
        (file, _output_dpi(file)) for file in found.values() if file is not None
    ]
    if not included:
        raise CliError(ExitCode.NOTHING_TO_EXPORT, "no processed pages to export")
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    export_pdf(
        included,
        destination,
        bilevel_codec=bilevel_codec,
        outline_source=_source_pdf_of(project),
    )
    return ExportReport(
        destination=absolute(destination),
        exported=len(included),
        skipped=_skipped(found),
    )


def export_tiff_files(project_path: Path, destination: Path) -> ExportReport:
    """Copy the processed TIFF files of a project into a folder.

    Args:
        project_path (pathlib.Path): The project JSON file.
        destination (pathlib.Path): Folder to copy into; created when missing.

    Returns:
        ExportReport: The folder, file count and skipped pages.

    Raises:
        CliError: With ExitCode.NOTHING_TO_EXPORT when no page has output.
    """
    project, _ = load_project(project_path)
    found = _processed_files(project, project_path)
    files = [file for file in found.values() if file is not None]
    if not files:
        raise CliError(ExitCode.NOTHING_TO_EXPORT, "no processed pages to export")
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for file in files:
        shutil.copy2(file, destination / file.name)
    return ExportReport(
        destination=absolute(destination),
        exported=len(files),
        skipped=_skipped(found),
    )


def _parse_value(raw_value: str) -> Any:
    """Read a command-line value as JSON, falling back to the bare text."""
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


def _output_names(project: Project) -> dict[PageId, str]:
    """Name the processed file of every page, as the batch names them."""
    return output_file_names(
        project.get_pages(), project.layout_direction, project.output_file_label_map()
    )


def _processed_files(project: Project, project_path: Path) -> dict[PageId, Path | None]:
    """Map every page to its processed file, or None when that file is absent."""
    out_dir = output_directory_of(project, project_path)
    names = _output_names(project)
    found: dict[PageId, Path | None] = {}
    for page in project.get_pages():
        name = names.get(page)
        file = None if name is None else out_dir / name
        found[page] = file if file is not None and file.is_file() else None
    return found


def _skipped(found: dict[PageId, Path | None]) -> list[str]:
    """List the pages that have no processed file, by page name."""
    return [page_name(page) for page, file in found.items() if file is None]


def _page_status(page: PageId, names: dict[PageId, str], out_dir: Path) -> PageStatus:
    """Describe the processing state of one page."""
    name = names.get(page)
    output = None if name is None else out_dir / name
    return PageStatus(
        page=page_name(page),
        source=absolute(page.image_id.file_path),
        output=None if output is None else absolute(output),
        output_exists=output is not None and output.is_file(),
    )


def _output_dpi(file: Path) -> tuple[float, float]:
    """Read the resolution stored in a processed file, or the fallback when absent."""
    _, metadata = list_image_pages(file)[0]
    horizontal, vertical = metadata.dpi.horizontal, metadata.dpi.vertical
    return (horizontal or FALLBACK_DPI, vertical or FALLBACK_DPI)


def _source_pdf_of(project: Project) -> Path | None:
    """Return the first source PDF of the project that is still on disk."""
    for source in project.pdf_sources:
        if source.pdf_path.is_file():
            return source.pdf_path
    return None
