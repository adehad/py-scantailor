"""Tests for the recent projects list."""

from pathlib import Path

import pytest
from PySide6.QtCore import QSettings

from scantailor.core.recent_projects import DEFAULT_MAX_ITEMS, RecentProjects


@pytest.fixture
def settings(tmp_path: Path) -> QSettings:
    """Settings stored in a file of their own, never in the user's configuration."""
    return QSettings(str(tmp_path / "recent.ini"), QSettings.Format.IniFormat)


def test_newest_project_comes_first():
    recent = RecentProjects()
    recent.set_most_recent("a.ScanTailor")
    recent.set_most_recent("b.ScanTailor")

    assert recent.enumerate() == ["b.ScanTailor", "a.ScanTailor"]


def test_reopening_a_project_moves_it_to_the_front_without_duplicating():
    recent = RecentProjects(["c", "b", "a"])

    recent.set_most_recent("a")

    assert recent.enumerate() == ["a", "c", "b"]


def test_enumerate_and_write_keep_only_the_default_number_of_newest_items(
    settings: QSettings,
):
    recent = RecentProjects()
    for number in range(DEFAULT_MAX_ITEMS + 3):
        recent.set_most_recent(f"p{number}")

    expected = [f"p{number}" for number in reversed(range(3, DEFAULT_MAX_ITEMS + 3))]
    assert DEFAULT_MAX_ITEMS == 7
    assert recent.enumerate() == expected
    recent.write(settings)
    assert RecentProjects.read(settings).enumerate(100) == expected


def test_write_replaces_a_longer_stored_list(settings: QSettings):
    RecentProjects(["a", "b", "c"]).write(settings)

    RecentProjects(["z"]).write(settings)

    assert RecentProjects.read(settings).enumerate() == ["z"]


def test_read_from_empty_settings_gives_an_empty_list(settings: QSettings):
    assert RecentProjects.read(settings).enumerate() == []


def test_validate_drops_missing_files_and_reports_it(tmp_path: Path):
    present = tmp_path / "here.ScanTailor"
    present.write_text("{}", encoding="utf-8")
    recent = RecentProjects([str(tmp_path / "gone.ScanTailor"), str(present)])

    assert recent.validate() is False
    assert recent.enumerate() == [str(present)]
    assert recent.validate() is True
