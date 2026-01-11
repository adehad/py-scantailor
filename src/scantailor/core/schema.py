"""JSON Schema generation and validation for ScanTailor projects.

This module provides functions to generate and validate JSON schemas
for project files, ensuring compatibility and data integrity.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from scantailor.core.project import Project

# Schema version - increment when making breaking changes
SCHEMA_VERSION = "1.0.0"


def generate_project_schema() -> dict[str, Any]:
    """Generate a JSON Schema for Project files.

    Returns:
        JSON Schema dictionary for the Project model.

    Example:
        >>> schema = generate_project_schema()
        >>> print(schema["title"])
        'Project'
    """
    schema = Project.model_json_schema()

    # Add metadata
    schema["$schema"] = "http://json-schema.org/draft-07/schema#"
    schema["$id"] = f"https://scantailor.org/schema/project/v{SCHEMA_VERSION}"

    return schema


def save_schema(path: Path) -> None:
    """Save the project schema to a JSON file.

    Args:
        path: Path to write the schema file.

    Example:
        >>> from pathlib import Path
        >>> save_schema(Path("project.schema.json"))
    """
    schema = generate_project_schema()
    path.write_text(
        json.dumps(schema, indent=2),
        encoding="utf-8",
    )


def validate_project_json(content: str) -> tuple[bool, list[str]]:
    """Validate JSON content against the project schema.

    Uses Pydantic validation for comprehensive type checking.

    Args:
        content: JSON string to validate.

    Returns:
        Tuple of (is_valid, error_messages).
        If valid, error_messages is empty.

    Example:
        >>> json_str = '{"version": 1, "images": []}'
        >>> is_valid, errors = validate_project_json(json_str)
        >>> print(is_valid)
        True
    """
    errors: list[str] = []

    try:
        Project.model_validate_json(content)
        return True, errors
    except ValidationError as e:
        for error in e.errors():
            loc = ".".join(str(x) for x in error["loc"])
            msg = error["msg"]
            errors.append(f"{loc}: {msg}")
        return False, errors
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON: {e.msg} at line {e.lineno}")
        return False, errors


def validate_project_file(path: Path) -> tuple[bool, list[str]]:
    """Validate a project file against the schema.

    Args:
        path: Path to the project file.

    Returns:
        Tuple of (is_valid, error_messages).

    Example:
        >>> from pathlib import Path
        >>> is_valid, errors = validate_project_file(Path("project.scantailor"))
    """
    if not path.exists():
        return False, [f"File not found: {path}"]

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        return False, [f"Failed to read file: {e}"]

    return validate_project_json(content)


def get_schema_version_from_project(content: str) -> int | None:
    """Extract the version number from project JSON.

    Args:
        content: JSON string of the project.

    Returns:
        Version number if found, None otherwise.
    """
    try:
        data = json.loads(content)
        return data.get("version")
    except json.JSONDecodeError:
        return None


def migrate_project(content: str, target_version: int = 1) -> str:
    """Migrate a project to a target version.

    Currently only version 1 is supported, so this is a no-op.
    Future versions can add migration logic here.

    Args:
        content: JSON string of the project.
        target_version: Target version number.

    Returns:
        Migrated project JSON string.

    Raises:
        ValueError: If migration is not possible.
    """
    current_version = get_schema_version_from_project(content)

    if current_version is None:
        # Try to parse and re-serialize to add defaults
        try:
            project = Project.model_validate_json(content)
            return project.model_dump_json(indent=2)
        except ValidationError as e:
            msg = f"Cannot migrate invalid project: {e}"
            raise ValueError(msg) from e

    # Validate version is an integer
    if not isinstance(current_version, int):
        version_type = type(current_version).__name__
        msg = f"Cannot migrate: version must be an integer, got {version_type}"
        raise ValueError(msg)

    if current_version == target_version:
        # Re-serialize to add any missing default values
        try:
            project = Project.model_validate_json(content)
            return project.model_dump_json(indent=2)
        except ValidationError:
            # If validation fails, return as-is
            return content

    if current_version > target_version:
        msg = f"Cannot downgrade from version {current_version} to {target_version}"
        raise ValueError(msg)

    # Future: Add migration logic for version upgrades
    # For now, just re-serialize with the target version
    try:
        data = json.loads(content)
        data["version"] = target_version
        return json.dumps(data, indent=2)
    except json.JSONDecodeError as e:
        msg = f"Invalid JSON: {e}"
        raise ValueError(msg) from e
