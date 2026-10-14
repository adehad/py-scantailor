"""Create a project from a folder of scans or from the pages of a PDF."""

from pathlib import Path

from loguru import logger

from scantailor.cli.errors import CliError, ExitCode
from scantailor.cli.project_file import save_project
from scantailor.cli.reports import CreateReport, absolute
from scantailor.core.default_params import apply_default_params
from scantailor.core.default_params_profiles import active_default_params
from scantailor.core.filename_ordering import sort_smart
from scantailor.core.image_io import ImageLoadError, list_image_pages
from scantailor.core.models import Dpi, ImageId
from scantailor.core.pdf_import import import_pdf
from scantailor.core.pdf_source import PdfSource
from scantailor.core.project import ImageMetadata, LayoutDirection, Project
from scantailor.core.stage_sequence import StageSequence

IMAGE_SUFFIXES = frozenset({".tif", ".tiff", ".png", ".jpg", ".jpeg"})
PDF_SUFFIX = ".pdf"
DEFAULT_PDF_DPI = 300
OUTPUT_FOLDER_NAME = "out"


def create_project(
    source: Path,
    project_path: Path,
    *,
    dpi: int | None,
    layout_direction: LayoutDirection,
    output_directory: Path | None,
    pages_directory: Path | None,
    force: bool,
) -> CreateReport:
    """Write a new project file for a folder of scans or a PDF.

    A PDF is rendered to PNG pages in ``pages_directory`` first, as the GUI does,
    so the processed files are never mixed with the originals.

    Args:
        source (pathlib.Path): A folder of images, or a PDF file.
        project_path (pathlib.Path): Where to write the project JSON.
        dpi (int | None): Resolution to give every image, or the PDF render
            resolution. Required for a folder whose images carry no usable DPI.
        layout_direction (str): LTR or RTL reading order.
        output_directory (pathlib.Path | None): Folder for processed files;
            defaults to ``out`` beside the images.
        pages_directory (pathlib.Path | None): Folder for rendered PDF pages;
            defaults to ``<pdf stem>_pages`` beside the PDF.
        force (bool): Overwrite an existing project file.

    Returns:
        CreateReport: What was created.

    Raises:
        CliError: With ExitCode.PROJECT if the file exists and ``force`` is
            false, INPUT if the source has no usable pages, OPTION if ``dpi``
            is not positive.
    """
    if project_path.exists() and not force:
        msg = f"{project_path} already exists; pass --force to replace it"
        raise CliError(ExitCode.PROJECT, msg)
    if dpi is not None and dpi <= 0:
        raise CliError(ExitCode.OPTION, f"dpi must be positive, got {dpi}")

    if source.is_file() and source.suffix.lower() == PDF_SUFFIX:
        return _create_from_pdf(
            source,
            project_path,
            dpi=dpi,
            layout_direction=layout_direction,
            output_directory=output_directory,
            pages_directory=pages_directory,
        )
    if source.is_dir():
        return _create_from_folder(
            source,
            project_path,
            dpi=dpi,
            layout_direction=layout_direction,
            output_directory=output_directory,
        )
    msg = f"source is neither a folder nor a PDF: {source}"
    raise CliError(ExitCode.INPUT, msg)


def _create_from_folder(
    folder: Path,
    project_path: Path,
    *,
    dpi: int | None,
    layout_direction: LayoutDirection,
    output_directory: Path | None,
) -> CreateReport:
    """Build a project from the image files of a folder, in smart filename order."""
    files = sort_smart(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        msg = f"no image files ({', '.join(sorted(IMAGE_SUFFIXES))}) in {folder}"
        raise CliError(ExitCode.INPUT, msg)
    pages = _read_pages(files)
    if dpi is not None:
        pages = [(image, _with_dpi(meta, dpi)) for image, meta in pages]
    elif any(not meta.is_dpi_ok() for _, meta in pages):
        msg = "some images have no usable DPI; pass --dpi to set one for all"
        raise CliError(ExitCode.INPUT, msg)
    out_dir = output_directory or folder / OUTPUT_FOLDER_NAME
    return _write_project(
        pages,
        project_path,
        output_directory=out_dir,
        layout_direction=layout_direction,
        pdf_sources=[],
        source_kind="folder",
    )


def _create_from_pdf(
    pdf_path: Path,
    project_path: Path,
    *,
    dpi: int | None,
    layout_direction: LayoutDirection,
    output_directory: Path | None,
    pages_directory: Path | None,
) -> CreateReport:
    """Render the PDF's pages to PNG and build a project from them."""
    render_dpi = dpi or DEFAULT_PDF_DPI
    pages_dir = pages_directory or pdf_path.with_name(f"{pdf_path.stem}_pages")
    try:
        images = import_pdf(pdf_path, pages_dir, render_dpi)
    except (OSError, ValueError, RuntimeError) as error:
        msg = f"cannot render {pdf_path}: {error}"
        raise CliError(ExitCode.INPUT, msg) from error
    if not images:
        raise CliError(ExitCode.INPUT, f"{pdf_path} has no pages")
    pdf_source = PdfSource(
        pdf_path=pdf_path.resolve(),
        dpi=render_dpi,
        pages={image.name: index for index, image in enumerate(images)},
    )
    pages = _read_pages(images)
    out_dir = output_directory or pages_dir / OUTPUT_FOLDER_NAME
    return _write_project(
        pages,
        project_path,
        output_directory=out_dir,
        layout_direction=layout_direction,
        pdf_sources=[pdf_source],
        source_kind="pdf",
    )


def _read_pages(files: list[Path]) -> list[tuple[ImageId, ImageMetadata]]:
    """Read the page list and metadata of each file without decoding pixels."""
    pages: list[tuple[ImageId, ImageMetadata]] = []
    for path in files:
        try:
            pages.extend(list_image_pages(path))
        except (ImageLoadError, OSError) as error:
            msg = f"cannot read image {path}: {error}"
            raise CliError(ExitCode.INPUT, msg) from error
    return pages


def _with_dpi(metadata: ImageMetadata, dpi: int) -> ImageMetadata:
    """Return a copy of ``metadata`` with the given uniform resolution."""
    return metadata.model_copy(update={"dpi": Dpi.uniform(dpi)})


def _write_project(
    pages: list[tuple[ImageId, ImageMetadata]],
    project_path: Path,
    *,
    output_directory: Path,
    layout_direction: LayoutDirection,
    pdf_sources: list[PdfSource],
    source_kind: str,
) -> CreateReport:
    """Save a project holding ``pages`` and create its output folder."""
    out_dir = output_directory.resolve()
    project = Project(
        output_directory=out_dir,
        layout_direction=layout_direction,
        pdf_sources=pdf_sources,
    )
    for image_id, metadata in pages:
        project.add_image(image_id, metadata)
    stages = StageSequence()
    apply_default_params(
        stages, [image_id for image_id, _ in pages], active_default_params()
    )
    project_path = project_path.resolve()
    project_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_project(project, stages, project_path)
    logger.info("Created project {} with {} pages", project_path, len(pages))
    return CreateReport(
        project=absolute(project_path),
        pages=len(project.get_pages()),
        source_kind=source_kind,
        output_directory=absolute(out_dir),
    )
