"""The macOS Info.plist that app.spec gives PyInstaller's BUNDLE."""

import ast
import plistlib
from pathlib import Path
from typing import Any

import pytest

from scantailor.app.ui.MainWindow import PROJECT_SUFFIX

_SPEC = Path(__file__).resolve().parents[2] / "app.spec"
_PLIST_NAMES = {"BUNDLE_IDENTIFIER", "PROJECT_UTI", "MACOS_INFO_PLIST"}


@pytest.fixture(scope="module")
def info_plist() -> dict[str, Any]:
    """Evaluate only the plist assignments: the rest of the spec needs PyInstaller."""
    tree = ast.parse(_SPEC.read_text(encoding="utf-8"))
    assignments: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in _PLIST_NAMES
            for target in node.targets
        )
    ]
    namespace: dict[str, Any] = {}
    exec(compile(ast.Module(assignments, []), str(_SPEC), "exec"), namespace)
    return namespace["MACOS_INFO_PLIST"]


def test_the_plist_serialises_as_an_info_plist(info_plist: dict[str, Any]) -> None:
    assert plistlib.loads(plistlib.dumps(info_plist)) == info_plist


def test_the_window_app_keeps_its_dock_icon(info_plist: dict[str, Any]) -> None:
    assert info_plist["LSBackgroundOnly"] is False


def test_the_app_owns_the_project_document_type(info_plist: dict[str, Any]) -> None:
    (document_type,) = info_plist["CFBundleDocumentTypes"]
    (exported_type,) = info_plist["UTExportedTypeDeclarations"]

    assert document_type["LSItemContentTypes"] == [exported_type["UTTypeIdentifier"]]
    assert document_type["CFBundleTypeRole"] == "Editor"


def test_the_exported_type_claims_the_project_file_extension(
    info_plist: dict[str, Any],
) -> None:
    (exported_type,) = info_plist["UTExportedTypeDeclarations"]
    tags = exported_type["UTTypeTagSpecification"]

    assert tags["public.filename-extension"] == [PROJECT_SUFFIX.removeprefix(".")]
    assert tags["public.mime-type"] == ["application/x-scantailor-project"]
