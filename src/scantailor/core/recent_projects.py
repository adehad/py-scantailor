"""The list of recently opened project files, kept in QSettings."""

from pathlib import Path

from PySide6.QtCore import QSettings

DEFAULT_MAX_ITEMS = 7
_SETTINGS_ARRAY = "project/recent"
_PATH_KEY = "path"


class RecentProjects:
    """Most-recently-used project file paths, newest first."""

    def __init__(self, paths: list[str] | None = None) -> None:
        """Create the list.

        Args:
            paths (list[str] | None): Initial paths, newest first.
        """
        self._paths: list[str] = list(paths or [])

    @classmethod
    def read(cls, settings: QSettings | None = None) -> RecentProjects:
        """Load the list from settings.

        Args:
            settings (QSettings | None): Where to read; the application's default
                settings when omitted.

        Returns:
            RecentProjects: The stored list.
        """
        settings = settings or QSettings()
        count = settings.beginReadArray(_SETTINGS_ARRAY)
        paths = []
        for index in range(count):
            settings.setArrayIndex(index)
            path = str(settings.value(_PATH_KEY, "", type=str))
            if path:
                paths.append(path)
        settings.endArray()
        return cls(paths)

    def write(
        self, settings: QSettings | None = None, max_items: int = DEFAULT_MAX_ITEMS
    ) -> None:
        """Store the newest ``max_items`` paths, replacing what was stored.

        Args:
            settings (QSettings | None): Where to write; the application's default
                settings when omitted.
            max_items (int): How many paths to keep.
        """
        settings = settings or QSettings()
        # Entries beyond the new length would otherwise survive as stale array rows.
        settings.remove(_SETTINGS_ARRAY)
        settings.beginWriteArray(_SETTINGS_ARRAY)
        for index, path in enumerate(self.enumerate(max_items)):
            settings.setArrayIndex(index)
            settings.setValue(_PATH_KEY, path)
        settings.endArray()

    def set_most_recent(self, file_path: str | Path) -> None:
        """Move ``file_path`` to the front, adding it if new.

        Args:
            file_path (str | Path): A project file.
        """
        key = str(file_path)
        if key in self._paths:
            self._paths.remove(key)
        self._paths.insert(0, key)

    def validate(self) -> bool:
        """Drop paths whose file no longer exists.

        Returns:
            bool: True if every path still existed.
        """
        kept = [path for path in self._paths if Path(path).exists()]
        all_ok = len(kept) == len(self._paths)
        self._paths = kept
        return all_ok

    def enumerate(self, max_items: int = DEFAULT_MAX_ITEMS) -> list[str]:
        """Return the newest ``max_items`` paths.

        Args:
            max_items (int): How many paths to return at most.

        Returns:
            list[str]: Paths, newest first.
        """
        return self._paths[:max_items]
