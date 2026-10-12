"""Saved default-parameter profiles, kept as JSON in the application config folder.

``Default`` and ``Source`` are built in; every other name is a user profile stored
as ``<config folder>/profiles/saved/<name>.json``. Names are unique ignoring case,
so a folder behaves the same on file systems that fold case and on those that
do not. The parameters that new projects start from are kept next to them in
``active.json``, together with the name of the profile they belong to, so that
edits made to a profile in the dialog apply without overwriting the saved
profile and never outlive a change to that profile.
"""

import os
import re
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, ValidationError

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
# Windows treats these as devices whatever the extension, so "con.json" is not a file.
_DEVICE_NAME = re.compile(r"(con|prn|aux|nul|com[0-9]|lpt[0-9])(\..*)?", re.IGNORECASE)


class ProfileError(Exception):
    """A profile could not be named, read, written or deleted."""


class _ActiveProfile(BaseModel):
    """The edited parameters new projects start from and the profile they edit."""

    profile: str
    params: DefaultParams


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
    if (
        _FORBIDDEN_NAME_CHARS.search(stripped)
        or stripped.endswith(".")
        or _DEVICE_NAME.fullmatch(stripped)
    ):
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
            list[str]: Profile names in alphabetical order, without the built-in
                ones; of files whose names differ only in case, the first.
        """
        found: dict[str, str] = {}
        for path in self._saved_files():
            try:
                name = validate_profile_name(path.stem)
            except ProfileError:
                logger.warning("Ignoring profile file with unusable name: {}", path)
                continue
            if found.setdefault(name.casefold(), name) != name:
                logger.warning("Ignoring profile file named like another: {}", path)
        return sorted(found.values(), key=str.casefold)

    def read(self, name: str) -> DefaultParams:
        """Load a saved profile.

        Args:
            name (str): Name of a user profile; case does not matter.

        Returns:
            DefaultParams: The stored parameters.

        Raises:
            ProfileError: If the file is missing, unreadable or not a valid profile.
        """
        path = self._profile_path(name)
        return self._read_file(path)

    def write(self, name: str, params: DefaultParams) -> None:
        """Save a profile, replacing one of the same name, whatever its case.

        The profile takes the spelling of ``name``. Remembered edits of the profile
        are dropped, since the saved profile now holds them.

        Args:
            name (str): Name of the user profile.
            params (DefaultParams): Parameters to store.

        Raises:
            ProfileError: If the name is not acceptable or the file cannot be written.
        """
        stripped = validate_profile_name(name)
        target = self._saved_folder / f"{stripped}.json"
        existing = self._existing_path(stripped)
        self._write_file(target, params, replacing=existing)
        self._forget_active_of(stripped)

    def delete(self, name: str) -> None:
        """Remove a saved profile, and the remembered edits of it.

        Args:
            name (str): Name of a user profile; case does not matter.

        Raises:
            ProfileError: If the file cannot be removed.
        """
        path = self._profile_path(name)
        try:
            path.unlink()
        except OSError as error:
            raise ProfileError(f"Cannot delete profile '{name}': {error}") from error
        self._forget_active_of(name)

    def set_active(
        self, params: DefaultParams, profile_name: str | None = None
    ) -> None:
        """Remember the parameters new projects start from.

        Args:
            params (DefaultParams): The parameters of the selected profile, with any
                edits made in the dialog.
            profile_name (str | None): The profile the parameters belong to. Without
                it the parameters cannot be attributed, so any remembered ones are
                dropped instead.

        Raises:
            ProfileError: If the file cannot be written.
        """
        if profile_name is None:
            self._forget_active()
            return
        record = _ActiveProfile(profile=profile_name, params=params)
        self._write_text(self._active_file, record.model_dump_json(indent=2))

    def active_params(self, profile_name: str) -> DefaultParams:
        """Resolve the parameters that new projects start from.

        Args:
            profile_name (str): The selected profile, as kept in the application
                settings; case does not matter.

        Returns:
            DefaultParams: The built-in parameters for Default and Source. For any
                other name, the remembered parameters if they belong to that
                profile, else the saved profile of that name, else the built-in
                defaults.
        """
        folded = profile_name.strip().casefold()
        if folded == SOURCE_PROFILE.casefold():
            return DefaultParams.source()
        if folded == DEFAULT_PROFILE.casefold():
            return DefaultParams()
        active = self._read_active()
        if active is not None and active.profile.strip().casefold() == folded:
            return active.params
        path = self._saved_path_or_none(profile_name)
        if path is not None and path.is_file():
            try:
                return self._read_file(path)
            except ProfileError as error:
                logger.warning("{}", error)
        return DefaultParams()

    def _saved_files(self) -> list[Path]:
        if not self._saved_folder.is_dir():
            return []
        return sorted(self._saved_folder.glob("*.json"))

    def _existing_path(self, name: str) -> Path | None:
        """Find the saved file of a profile, ignoring the case of its name."""
        folded = name.strip().casefold()
        for path in self._saved_files():
            if path.stem.casefold() == folded:
                return path
        return None

    def _saved_path_or_none(self, name: str) -> Path | None:
        try:
            return self._profile_path(name)
        except ProfileError:
            return None

    def _profile_path(self, name: str) -> Path:
        stripped = validate_profile_name(name)
        return self._existing_path(stripped) or self._saved_folder / f"{stripped}.json"

    def _read_active(self) -> _ActiveProfile | None:
        if not self._active_file.is_file():
            return None
        try:
            return _ActiveProfile.model_validate_json(
                self._active_file.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError, ValueError) as error:
            logger.warning(
                "Ignoring remembered parameters {}: {}", self._active_file, error
            )
            return None

    def _forget_active(self) -> None:
        try:
            self._active_file.unlink(missing_ok=True)
        except OSError as error:
            raise ProfileError(f"Cannot remove {self._active_file}: {error}") from error

    def _forget_active_of(self, name: str) -> None:
        active = self._read_active()
        if active is not None and active.profile.strip().casefold() == (
            name.strip().casefold()
        ):
            self._forget_active()

    @staticmethod
    def _read_file(path: Path) -> DefaultParams:
        try:
            return DefaultParams.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, ValueError) as error:
            raise ProfileError(f"Cannot read profile {path}: {error}") from error

    @classmethod
    def _write_file(
        cls, path: Path, params: DefaultParams, replacing: Path | None = None
    ) -> None:
        cls._write_text(path, params.model_dump_json(indent=2), replacing)

    @staticmethod
    def _write_text(path: Path, text: str, replacing: Path | None = None) -> None:
        """Write through a temporary file; ``replacing`` is removed once it is in place.

        A file that differs from ``path`` only in case would otherwise keep its old
        spelling on a file system that folds case.
        """
        temporary = path.with_name(path.name + ".tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(text, encoding="utf-8")
            if replacing is not None and replacing.name != path.name:
                replacing.unlink(missing_ok=True)
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
