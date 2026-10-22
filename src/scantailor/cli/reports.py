"""JSON documents the commands write to standard output."""

from pathlib import Path

from pydantic import BaseModel, Field


class PageStatus(BaseModel):
    """Processing state of one page.

    Args:
        page (str): Source file name, with the half for split pages.
        source (str): Absolute path of the source image.
        output (str | None): Absolute path of the processed file, when named.
        output_exists (bool): Whether that file is on disk.
    """

    page: str
    source: str
    output: str | None
    output_exists: bool


class ProjectStatus(BaseModel):
    """Summary of a project file and its output folder.

    Args:
        project (str): Absolute path of the project file.
        output_directory (str): Absolute path of the output folder.
        layout_direction (str): LTR or RTL.
        pdf_sources (list[str]): The PDFs the pages were rendered from.
        page_count (int): Pages in the project.
        outputs_present (int): Pages whose processed file exists.
        stage_params_set (dict[str, int]): Explicitly set page parameters per stage.
        pages (list[PageStatus]): One entry per page, in project order.
    """

    project: str
    output_directory: str
    layout_direction: str
    pdf_sources: list[str] = Field(default_factory=list)
    page_count: int
    outputs_present: int
    stage_params_set: dict[str, int] = Field(default_factory=dict)
    pages: list[PageStatus] = Field(default_factory=list)


class BatchReport(BaseModel):
    """Outcome of ``run``.

    Args:
        output_directory (str): Folder the processed files were written to.
        written (int): Pages that finished.
        kept (int): Of those, pages whose output files were already up to date
            and so were not rendered again.
        failed (dict[str, str]): Error text per page that failed.
        cancelled (int): Pages never started.
    """

    output_directory: str
    written: int
    kept: int = 0
    failed: dict[str, str] = Field(default_factory=dict)
    cancelled: int = 0


class ExportReport(BaseModel):
    """Outcome of ``export pdf`` or ``export tiff``.

    Args:
        destination (str): The PDF file, or the folder the images were copied to.
        exported (int): Pages written to the destination.
        skipped (list[str]): Pages with no processed file, by page name.
    """

    destination: str
    exported: int
    skipped: list[str] = Field(default_factory=list)


class CreateReport(BaseModel):
    """Outcome of ``create``.

    Args:
        project (str): Absolute path of the new project file.
        pages (int): Pages added.
        source_kind (str): ``folder`` or ``pdf``.
        output_directory (str): Folder the processed files will go to.
    """

    project: str
    pages: int
    source_kind: str
    output_directory: str


class OptionsReport(BaseModel):
    """Stage-wide defaults of a project.

    Args:
        defaults (dict[str, object]): Current value per option name.
    """

    defaults: dict[str, object]


def absolute(path: Path) -> str:
    """Return the resolved absolute path as text.

    Args:
        path (pathlib.Path): Any path, relative or absolute.

    Returns:
        str: The path with symlinks and ``..`` resolved.
    """
    return str(path.resolve())
