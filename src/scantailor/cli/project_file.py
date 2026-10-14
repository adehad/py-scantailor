"""Opening and saving project files with their stage settings applied.

A project keeps the explicit per-page parameters and the stage-wide defaults as
plain JSON. The stages are rebuilt from those on every open and written back on
every save, the same way the GUI does it.
"""

from pathlib import Path

from scantailor.cli.errors import CliError, ExitCode
from scantailor.core.models import PageId
from scantailor.core.project import Project
from scantailor.core.stage_params import (
    apply_stage_defaults,
    apply_stage_params,
    collect_stage_defaults,
    collect_stage_params,
)
from scantailor.core.stage_sequence import StageSequence


def load_project(project_path: Path) -> tuple[Project, StageSequence]:
    """Open a project and rebuild its stage settings.

    Args:
        project_path (pathlib.Path): The project JSON file.

    Returns:
        tuple[Project, StageSequence]: The project and its stages, with the
            saved parameters applied.

    Raises:
        CliError: With ExitCode.PROJECT when the file is missing, unreadable or
            holds values the stages reject.
    """
    if not project_path.is_file():
        msg = f"no project file at {project_path}"
        raise CliError(ExitCode.PROJECT, msg)
    try:
        project = Project.load(project_path)
    except (OSError, ValueError) as error:
        msg = f"cannot read project {project_path}: {error}"
        raise CliError(ExitCode.PROJECT, msg) from error
    stages = StageSequence()
    try:
        apply_stage_params(stages, project.stage_params)
        apply_stage_defaults(stages, project.stage_defaults)
    except ValueError as error:
        msg = f"project {project_path} holds invalid settings: {error}"
        raise CliError(ExitCode.PROJECT, msg) from error
    return project, stages


def save_project(project: Project, stages: StageSequence, project_path: Path) -> None:
    """Write the project with the current stage settings folded into it.

    Args:
        project (Project): The project to write; its stage fields are replaced.
        stages (StageSequence): The settings to store.
        project_path (pathlib.Path): Destination JSON file.
    """
    project.stage_params = collect_stage_params(stages, project.get_pages())
    project.stage_defaults = collect_stage_defaults(stages)
    project.save(project_path)


def output_directory_of(project: Project, project_path: Path) -> Path:
    """Resolve the output folder, which may be stored relative to the project file.

    Args:
        project (Project): The project whose output folder is wanted.
        project_path (pathlib.Path): The project file it was opened from.

    Returns:
        pathlib.Path: An absolute output folder.
    """
    return (project_path.resolve().parent / project.output_directory).resolve()


def page_name(page: PageId) -> str:
    """Name a page for humans and JSON, such as ``scan01.tif`` or ``scan01.tif/left``.

    Args:
        page (PageId): The page to name.

    Returns:
        str: File name, then the sub page when the image holds two pages.
    """
    name = page.image_id.file_path.name
    if page.image_id.page:
        name = f"{name}#{page.image_id.page}"
    if page.sub_page.name == "SINGLE_PAGE":
        return name
    return f"{name}/{page.sub_page.name.lower()}"
