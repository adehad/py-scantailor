"""Where a project's page images came from, so its source PDF's text can be reused."""

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, Field

SOURCE_FILE_NAME = "pdf_source.json"
"""Name of the file earlier versions wrote beside the page images."""


class PdfSource(BaseModel):
    """A PDF whose pages were rendered into a project.

    A project can hold several sources. The same image file name recurs in every
    extraction, so a page is matched by its folder as well as its name.

    Attributes:
        pdf_path (pathlib.Path): The PDF the pages were rendered from.
        dpi (int): Resolution the pages were rendered at.
        pages_dir (pathlib.Path | None): Folder holding the rendered images. None for a
            source stored by an earlier version, which matches by name alone.
        pages (dict[str, int]): Zero-based PDF page index per image file name.
    """

    pdf_path: Path
    dpi: int = Field(gt=0)
    pages_dir: Path | None = None
    pages: dict[str, int] = Field(default_factory=dict)

    @classmethod
    def _read_sidecar(cls, directory: Path) -> "PdfSource | None":
        try:
            source = cls.model_validate_json(
                (directory / SOURCE_FILE_NAME).read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return None
        return source.model_copy(update={"pages_dir": directory})

    @classmethod
    def find(cls, image_files: Iterable[Path]) -> list["PdfSource"]:
        """Return the mappings stored beside the image files, one per folder.

        Args:
            image_files (Iterable[pathlib.Path]): Image files of the project.

        Returns:
            list[PdfSource]: The readable mappings, in the order of their folders.
        """
        found: list[PdfSource] = []
        for directory in dict.fromkeys(file.parent for file in image_files):
            if (source := cls._read_sidecar(directory)) is not None:
                found.append(source)
        return found

    @classmethod
    def discard_sidecars(cls, image_files: Iterable[Path]) -> None:
        """Delete the mapping files an earlier version left beside the page images.

        Call once the project file that holds the mapping is saved.

        Args:
            image_files (Iterable[pathlib.Path]): Image files of the project.
        """
        for directory in dict.fromkeys(file.parent for file in image_files):
            try:
                (directory / SOURCE_FILE_NAME).unlink(missing_ok=True)
            except OSError:
                # A read-only folder keeps its file; the project's mapping wins.
                pass

    def page_index(self, image_file: Path) -> int | None:
        """Return the zero-based PDF page an image was rendered from, if it was.

        Args:
            image_file (pathlib.Path): Image file of the project, with its folder.

        Returns:
            int | None: The page index, or None when this source did not render it.
        """
        if self.pages_dir is not None and image_file.parent != self.pages_dir:
            return None
        return self.pages.get(image_file.name)

    @property
    def is_available(self) -> bool:
        """Whether the PDF still exists, so its text can be read."""
        return self.pdf_path.is_file()
