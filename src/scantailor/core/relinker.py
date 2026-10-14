"""Remapping of directory prefixes when a project's images have moved."""

from collections.abc import Iterable
from pathlib import Path


class Relinker:
    """Maps directory prefixes of image paths to new locations.

    A path is rewritten by the mapping of its nearest mapped ancestor, so moving
    a folder relinks every file below it at once.
    """

    def __init__(self) -> None:
        """Create a relinker that leaves every path unchanged."""
        self._mapping: dict[Path, Path] = {}

    def relink(self, old: Path, new: Path) -> None:
        """Send everything below ``old`` to ``new``.

        Args:
            old (pathlib.Path): Directory (or file) as stored in the project.
            new (pathlib.Path):
                Where it is now. Mapping a path to itself removes the entry.
        """
        if old == new:
            self._mapping.pop(old, None)
        else:
            self._mapping[old] = new

    def unlink(self, old: Path) -> None:
        """Forget the mapping of ``old``, if any.

        Args:
            old (pathlib.Path): A path previously passed to ``relink``.
        """
        self._mapping.pop(old, None)

    def target_of(self, old: Path) -> Path | None:
        """Return what ``old`` itself was relinked to, ignoring its ancestors.

        Args:
            old (pathlib.Path): A path previously passed to ``relink``.

        Returns:
            pathlib.Path | None: The new location, or None when ``old`` is not mapped.
        """
        return self._mapping.get(old)

    def is_empty(self) -> bool:
        """Return True when no path would change."""
        return not self._mapping

    def substitute(self, path: Path) -> Path:
        """Return ``path`` rewritten by its nearest mapped ancestor.

        Args:
            path (pathlib.Path): A stored file or directory path.

        Returns:
            pathlib.Path:
                The relinked path, or ``path`` itself when no ancestor is mapped.
        """
        for ancestor in (path, *path.parents):
            new = self._mapping.get(ancestor)
            if new is not None:
                return new / path.relative_to(ancestor)
        return path


def image_directories(files: Iterable[Path], *, only_missing: bool) -> list[Path]:
    """List the distinct folders that hold the given image files.

    Args:
        files (Iterable[pathlib.Path]): Image file paths.
        only_missing (bool): Keep only folders of files that do not exist.

    Returns:
        list[pathlib.Path]: Folders in sorted order.
    """
    return sorted(
        {file.parent for file in files if not only_missing or not file.exists()}
    )
