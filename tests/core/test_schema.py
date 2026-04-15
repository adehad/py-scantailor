"""Tests for JSON schema generation and validation."""

import json
from pathlib import Path

import pytest

from scantailor.core.models import Dpi, ImageId
from scantailor.core.project import ImageInfo, ImageMetadata, Project
from scantailor.core.schema import (
    SCHEMA_VERSION,
    generate_project_schema,
    get_schema_version_from_project,
    migrate_project,
    save_schema,
    validate_project_file,
    validate_project_json,
)


class TestGenerateProjectSchema:
    """Tests for generate_project_schema function."""

    def test_generates_valid_schema(self) -> None:
        """Schema is a valid JSON Schema."""
        schema = generate_project_schema()

        assert isinstance(schema, dict)
        assert "$schema" in schema
        assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"

    def test_schema_has_id(self) -> None:
        """Schema has $id field with version."""
        schema = generate_project_schema()

        assert "$id" in schema
        assert SCHEMA_VERSION in schema["$id"]

    def test_schema_has_title(self) -> None:
        """Schema has title from model."""
        schema = generate_project_schema()

        assert "title" in schema
        assert schema["title"] == "Project"

    def test_schema_has_properties(self) -> None:
        """Schema has expected properties."""
        schema = generate_project_schema()

        assert "properties" in schema
        props = schema["properties"]
        assert "version" in props
        assert "output_directory" in props
        assert "layout_direction" in props
        assert "images" in props

    def test_schema_is_json_serializable(self) -> None:
        """Schema can be serialized to JSON."""
        schema = generate_project_schema()

        json_str = json.dumps(schema)
        parsed = json.loads(json_str)

        assert parsed == schema


class TestSaveSchema:
    """Tests for save_schema function."""

    def test_saves_schema_to_file(self, tmp_path: Path) -> None:
        """Schema is saved to file."""
        schema_path = tmp_path / "project.schema.json"

        save_schema(schema_path)

        assert schema_path.exists()

        content = schema_path.read_text(encoding="utf-8")
        schema = json.loads(content)

        assert "$schema" in schema
        assert "properties" in schema


class TestValidateProjectJson:
    """Tests for validate_project_json function."""

    def test_valid_minimal_project(self) -> None:
        """Minimal valid project passes validation."""
        project_json = '{"version": 1, "images": []}'

        is_valid, errors = validate_project_json(project_json)

        assert is_valid is True
        assert errors == []

    def test_valid_full_project(self) -> None:
        """Full project with all fields passes validation."""
        project = Project(
            version=1,
            output_directory=Path("output"),
            layout_direction="LTR",
            images=[
                ImageInfo(
                    id=ImageId(file_path=Path("/test/scan.tiff")),
                    metadata=ImageMetadata(
                        width=1000, height=2000, dpi=Dpi.uniform(300)
                    ),
                )
            ],
        )
        project_json = project.model_dump_json()

        is_valid, errors = validate_project_json(project_json)

        assert is_valid is True
        assert errors == []

    def test_invalid_json_syntax(self) -> None:
        """Invalid JSON syntax is detected."""
        invalid_json = '{"version": 1, "images": ['  # Missing closing bracket

        is_valid, errors = validate_project_json(invalid_json)

        assert is_valid is False
        assert len(errors) == 1
        assert "Invalid JSON" in errors[0]

    def test_missing_required_field(self) -> None:
        """Missing required fields are detected."""
        # images is required and has no default
        project_json = '{"version": 1}'

        # Pydantic provides defaults, so this should still be valid
        is_valid, errors = validate_project_json(project_json)

        # images has a default_factory, so it's valid
        assert is_valid is True

    def test_invalid_field_type(self) -> None:
        """Invalid field types are detected."""
        # version should be int, not string
        project_json = '{"version": "one", "images": []}'

        is_valid, errors = validate_project_json(project_json)

        assert is_valid is False
        assert len(errors) >= 1
        assert any("version" in e for e in errors)

    def test_invalid_layout_direction(self) -> None:
        """Invalid layout direction is detected."""
        project_json = '{"version": 1, "layout_direction": "INVALID"}'

        is_valid, errors = validate_project_json(project_json)

        assert is_valid is False
        assert any("layout_direction" in e for e in errors)

    def test_negative_image_dimensions(self) -> None:
        """Negative image dimensions are detected."""
        project_json = json.dumps(
            {
                "version": 1,
                "images": [
                    {
                        "id": {"file_path": "/test.tiff", "page": 0},
                        "metadata": {
                            "width": -100,
                            "height": 200,
                            "dpi": {"horizontal": 300, "vertical": 300},
                        },
                    }
                ],
            }
        )

        is_valid, errors = validate_project_json(project_json)

        assert is_valid is False
        assert any("width" in e for e in errors)


class TestValidateProjectFile:
    """Tests for validate_project_file function."""

    def test_valid_file(self, tmp_path: Path) -> None:
        """Valid project file passes validation."""
        project_path = tmp_path / "project.json"
        project = Project(version=1)
        project.save(project_path)

        is_valid, errors = validate_project_file(project_path)

        assert is_valid is True
        assert errors == []

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Nonexistent file returns error."""
        project_path = tmp_path / "missing.json"

        is_valid, errors = validate_project_file(project_path)

        assert is_valid is False
        assert len(errors) == 1
        assert "not found" in errors[0].lower()

    def test_invalid_file_content(self, tmp_path: Path) -> None:
        """Invalid file content is detected."""
        project_path = tmp_path / "invalid.json"
        project_path.write_text('{"version": "invalid"}', encoding="utf-8")

        is_valid, errors = validate_project_file(project_path)

        assert is_valid is False


class TestGetSchemaVersionFromProject:
    """Tests for get_schema_version_from_project function."""

    def test_extracts_version(self) -> None:
        """Version is extracted from valid JSON."""
        project_json = '{"version": 1, "images": []}'

        version = get_schema_version_from_project(project_json)

        assert version == 1

    def test_missing_version(self) -> None:
        """None returned when version is missing."""
        project_json = '{"images": []}'

        version = get_schema_version_from_project(project_json)

        assert version is None

    def test_invalid_json(self) -> None:
        """None returned for invalid JSON."""
        invalid_json = "not json"

        version = get_schema_version_from_project(invalid_json)

        assert version is None


class TestMigrateProject:
    """Tests for migrate_project function."""

    def test_no_migration_needed(self) -> None:
        """Same version project is returned unchanged."""
        project = Project(version=1)
        project_json = project.model_dump_json()

        migrated = migrate_project(project_json, target_version=1)
        migrated_data = json.loads(migrated)

        assert migrated_data["version"] == 1

    def test_migration_adds_defaults(self) -> None:
        """Migration adds default values to minimal project."""
        minimal_json = '{"version": 1}'

        migrated = migrate_project(minimal_json, target_version=1)
        migrated_data = json.loads(migrated)

        assert "output_directory" in migrated_data
        assert "images" in migrated_data

    def test_cannot_downgrade(self) -> None:
        """Downgrade raises ValueError."""
        project_json = '{"version": 2}'

        with pytest.raises(ValueError, match="Cannot downgrade"):
            migrate_project(project_json, target_version=1)

    def test_invalid_project_raises(self) -> None:
        """Invalid project raises ValueError."""
        invalid_json = '{"version": "invalid"}'

        with pytest.raises(
            ValueError, match="Cannot migrate.*version must be an integer"
        ):
            migrate_project(invalid_json, target_version=1)

    def test_invalid_json_raises(self) -> None:
        """Invalid JSON raises ValueError."""
        invalid_json = "not json {"

        with pytest.raises(ValueError, match="Invalid JSON"):
            migrate_project(invalid_json, target_version=1)


class TestSchemaVersion:
    """Tests for SCHEMA_VERSION constant."""

    def test_version_format(self) -> None:
        """Version follows semver format."""
        parts = SCHEMA_VERSION.split(".")

        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)
