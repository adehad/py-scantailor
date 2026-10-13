"""Where a project's page images came from, so its source PDF's text can be reused."""

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, Field

SOURCE_FILE_NAME = "pdf_source.json"
"""Name of the file earlier versions wrote beside the page images."""


class PdfSource(BaseModel):
    """The PDF a project's pages were rendered from.

    The project stores this model in its own file.

    Attributes:
        pdf_path: The PDF the pages were rendered from.
        dpi: Resolution the pages were rendered at.
        pages: Zero-based PDF page index per image file name.
    """

    pdf_path: Path
    dpi: int = Field(gt=0)
    pages: dict[str, int] = Field(default_factory=dict)

    @classmethod
    def _read_sidecar(cls, directory: Path) -> "PdfSource | None":
        try:
            return cls.model_validate_json(
                (directory / SOURCE_FILE_NAME).read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return None

    @classmethod
    def find(cls, image_files: Iterable[Path]) -> "PdfSource | None":
        """Return the first mapping found beside any of ``image_files``, keeping it.

        Args:
            image_files: Image files of the project.

        Returns:
            PdfSource | None: The mapping, or None when no readable file was found.
        """
        for directory in dict.fromkeys(file.parent for file in image_files):
            if (source := cls._read_sidecar(directory)) is not None:
                return source
        return None

    @classmethod
    def discard_sidecars(cls, image_files: Iterable[Path]) -> None:
        """Delete the mapping files an earlier version left beside the page images.

        Call once the project file that holds the mapping is saved.

        Args:
            image_files: Image files of the project.
        """
        for directory in dict.fromkeys(file.parent for file in image_files):
            try:
                (directory / SOURCE_FILE_NAME).unlink(missing_ok=True)
            except OSError:
                # A read-only folder keeps its file; the project's mapping wins.
                pass

    def page_index(self, image_file: Path) -> int | None:
        """Return the PDF page an image file was rendered from, if it was."""
        return self.pages.get(image_file.name)

    @property
    def is_available(self) -> bool:
        """Whether the PDF still exists, so its text can be read."""
        return self.pdf_path.is_file()
