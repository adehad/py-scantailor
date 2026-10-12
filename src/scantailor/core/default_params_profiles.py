"""Saved default-parameter profiles, kept as JSON in the application config folder.

``Default`` and ``Source`` are built in; every other name is a user profile stored
as ``<config folder>/profiles/saved/<name>.json``. The parameters that new
projects start from are kept next to them in ``active.json`` so that edits made to
a profile in the dialog apply without overwriting the saved profile.
"""

import os
import re
from pathlib import Path

from loguru import logger
from pydantic import ValidationError

from scantailor.core.default_params import DefaultParams
from scantailor.core.settings import ApplicationSettings

DEFAULT_PROFILE = "Default"
SOURCE_PROFILE = "Source"
CUSTOM_PROFILE = "Custom"
RESERVED_PROFILE_NAMES = (DEFAULT_PROFILE, SOURCE_PROFILE, CUSTOM_PROFILE)

_MAX_NAME_LENGTH = 64
# Characters that Windows rejects in file names; rejecting them everywhere keeps a
# profile folder usable when it is copied between systems.
_FORBIDDEN_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class ProfileError(Exception):
    """A profile could not be named, read, written or deleted."""


def is_reserved_profile_name(name: str) -> bool:
    """Tell whether a name belongs to a built-in profile.

    Args:
        name (str): Candidate profile name.

    Returns:
        bool: True for Default, Source and Custom, ignoring case and spaces at
            either end.
    """
    folded = name.strip().casefold()
    return any(folded == reserved.casefold() for reserved in RESERVED_PROFILE_NAMES)


def validate_profile_name(name: str) -> str:
    """Check that a name can be a user profile and a file name.

    Args:
        name (str): Candidate profile name.

    Returns:
        str: The name without surrounding spaces.

    Raises:
        ProfileError: If the name is empty, reserved, too long, or contains a
            character that cannot be part of a file name.
    """
    stripped = name.strip()
    if not stripped:
        raise ProfileError("The profile name is empty.")
    if is_reserved_profile_name(stripped):
        raise ProfileError(f"'{stripped}' is the name of a built-in profile.")
    if len(stripped) > _MAX_NAME_LENGTH:
        raise ProfileError(f"The name is longer than {_MAX_NAME_LENGTH} characters.")
    if _FORBIDDEN_NAME_CHARS.search(stripped) or stripped.endswith("."):
        raise ProfileError(f"'{stripped}' cannot be used as a file name.")
    return stripped


class ProfileStore:
    """Reads and writes profiles in one folder.

    Args:
        folder (Path | None): Folder that holds the profiles; defaults to
            ``profiles`` beside the application settings file.
    """

    def __init__(self, folder: Path | None = None) -> None:
        """Create the store; nothing is written until a profile is saved."""
        self._folder = folder or ApplicationSettings.get_config_path().parent / (
            "profiles"
        )

    @property
    def _saved_folder(self) -> Path:
        return self._folder / "saved"

    @property
    def _active_file(self) -> Path:
        return self._folder / "active.json"

    def names(self) -> list[str]:
        """List the saved user profiles.

        Returns:
            list[str]: Profile names in alphabetical order, without the built-in ones.
        """
        if not self._saved_folder.is_dir():
            return []
        found = []
        for path in self._saved_folder.glob("*.json"):
            try:
                found.append(validate_profile_name(path.stem))
            except ProfileError:
                logger.warning("Ignoring profile file with unusable name: {}", path)
        return sorted(found, key=str.casefold)

    def read(self, name: str) -> DefaultParams:
        """Load a saved profile.

        Args:
            name (str): Name of a user profile.

        Returns:
            DefaultParams: The stored parameters.

        Raises:
            ProfileError: If the file is missing, unreadable or not a valid profile.
        """
        path = self._profile_path(name)
        return self._read_file(path)

    def write(self, name: str, params: DefaultParams) -> None:
        """Save a profile, replacing one of the same name.

        Args:
            name (str): Name of the user profile.
            params (DefaultParams): Parameters to store.

        Raises:
            ProfileError: If the name is not acceptable or the file cannot be written.
        """
        self._write_file(self._profile_path(name), params)

    def delete(self, name: str) -> None:
        """Remove a saved profile.

        Args:
            name (str): Name of a user profile.

        Raises:
            ProfileError: If the file cannot be removed.
        """
        path = self._profile_path(name)
        try:
            path.unlink()
        except OSError as error:
            raise ProfileError(f"Cannot delete profile '{name}': {error}") from error

    def set_active(self, params: DefaultParams) -> None:
        """Remember the parameters new projects start from.

        Args:
            params (DefaultParams): The parameters of the selected profile, with any
                edits made in the dialog.

        Raises:
            ProfileError: If the file cannot be written.
        """
        self._write_file(self._active_file, params)

    def active_params(self, profile_name: str) -> DefaultParams:
        """Resolve the parameters that new projects start from.

        Args:
            profile_name (str): The selected profile, as kept in the application
                settings.

        Returns:
            DefaultParams: The built-in parameters for Default and Source. For any
                other name, the remembered parameters, else the saved profile of that
                name, else the built-in defaults.
        """
        if profile_name == SOURCE_PROFILE:
            return DefaultParams.source()
        if profile_name == DEFAULT_PROFILE:
            return DefaultParams()
        for path in (self._active_file, self._saved_path_or_none(profile_name)):
            if path is None or not path.is_file():
                continue
            try:
                return self._read_file(path)
            except ProfileError as error:
                logger.warning("{}", error)
        return DefaultParams()

    def _saved_path_or_none(self, name: str) -> Path | None:
        try:
            return self._profile_path(name)
        except ProfileError:
            return None

    def _profile_path(self, name: str) -> Path:
        return self._saved_folder / f"{validate_profile_name(name)}.json"

    @staticmethod
    def _read_file(path: Path) -> DefaultParams:
        try:
            return DefaultParams.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, ValueError) as error:
            raise ProfileError(f"Cannot read profile {path}: {error}") from error

    @staticmethod
    def _write_file(path: Path, params: DefaultParams) -> None:
        temporary = path.with_name(path.name + ".tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(params.model_dump_json(indent=2), encoding="utf-8")
            os.replace(temporary, path)
        except OSError as error:
            raise ProfileError(f"Cannot write profile {path}: {error}") from error


def active_default_params(
    settings: ApplicationSettings | None = None, store: ProfileStore | None = None
) -> DefaultParams:
    """Return the parameters a new project should start from.

    Args:
        settings (ApplicationSettings | None): Settings that name the selected
            profile; defaults to the application instance.
        store (ProfileStore | None): Where profiles are kept; defaults to the
            application's profile folder.

    Returns:
        DefaultParams: The selected profile's parameters.
    """
    settings = settings or ApplicationSettings.get_instance()
    return (store or ProfileStore()).active_params(settings.current_profile)
