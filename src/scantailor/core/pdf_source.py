"""Where a project's page images came from, so its source PDF's text can be reused."""

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, Field

SOURCE_FILE_NAME = "pdf_source.json"


class PdfSource(BaseModel):
    """The PDF a project's pages were rendered from.

    A project stores this model in its own file. The mapping is also written
    beside the page images, which is where a project saved without it looks.

    Attributes:
        pdf_path: The PDF the pages were rendered from.
        dpi: Resolution the pages were rendered at.
        pages: Zero-based PDF page index per image file name.
    """

    pdf_path: Path
    dpi: int = Field(gt=0)
    pages: dict[str, int] = Field(default_factory=dict)

    def save(self, directory: Path) -> None:
        """Write the mapping beside the page images in ``directory``."""
        (directory / SOURCE_FILE_NAME).write_text(
            self.model_dump_json(indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> "PdfSource | None":
        """Read the mapping stored in ``directory``; None when absent or unreadable."""
        path = directory / SOURCE_FILE_NAME
        try:
            return cls.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    @classmethod
    def find(cls, image_files: Iterable[Path]) -> "PdfSource | None":
        """Return the first mapping stored beside any of ``image_files``."""
        seen: set[Path] = set()
        for file in image_files:
            if file.parent in seen:
                continue
            seen.add(file.parent)
            if (source := cls.load(file.parent)) is not None:
                return source
        return None

    def page_index(self, image_file: Path) -> int | None:
        """Return the PDF page an image file was rendered from, if it was."""
        return self.pages.get(image_file.name)

    @property
    def is_available(self) -> bool:
        """Whether the PDF still exists, so its text can be read."""
        return self.pdf_path.is_file()
