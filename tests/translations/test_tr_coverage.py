"""Lint test: UI string literals must be wrapped in tr() or translate()."""

import ast
import pathlib
from dataclasses import dataclass

import pytest

_UI_ROOT = (
    pathlib.Path(__file__).resolve().parents[2] / "src" / "scantailor" / "app" / "ui"
)
_WIDGET_CONSTRUCTORS = frozenset(
    {
        "QLabel",
        "QPushButton",
        "QCheckBox",
        "QRadioButton",
        "QGroupBox",
        "QAction",
        "QMessageBox",
    }
)
_TEXT_SETTERS = frozenset({"setWindowTitle", "setText", "setToolTip", "setStatusTip"})
_TRANSLATION_WRAPPERS = frozenset({"tr", "translate"})
_CONSTRUCTOR_TEXT_KEYWORDS = frozenset({"text", "title"})

# Keyed by (path relative to the UI root, literal text), not line number, so edits
# elsewhere in a file do not break the list. Each entry is a known gap in the
# translation wrapping. Remove the entry once the literal is wrapped in tr().
_ALLOWED_LITERALS: dict[tuple[str, str], str] = {}


@dataclass(frozen=True)
class LiteralFinding:
    """A string literal passed directly to a UI call without translation.

    Attributes:
        path (str): Path relative to the UI root, with forward slashes.
        line (int): Line number of the call.
        call (str): Name of the constructor or setter that received the literal.
        text (str): The literal text.
    """

    path: str
    line: int
    call: str
    text: str


def _callee_name(call: ast.Call) -> str | None:
    """Return the bare name of a call, whether it is a plain name or an attribute.

    Args:
        call (ast.Call): The call node.

    Returns:
        str | None: The function or method name, or None for other callables.
    """
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _is_translated(node: ast.expr) -> bool:
    """Whether an argument is a tr() or translate() call.

    Args:
        node (ast.expr): The argument expression.

    Returns:
        bool: True when the argument is a translation call.
    """
    return isinstance(node, ast.Call) and _callee_name(node) in _TRANSLATION_WRAPPERS


def _candidate_arguments(call: ast.Call) -> list[ast.expr]:
    """Return the arguments of a call that carry user-visible text.

    Args:
        call (ast.Call): The call node.

    Returns:
        list[ast.expr]: The first positional argument and text keywords for widget
            constructors, the first positional argument for setters, and nothing
            for any other call.
    """
    name = _callee_name(call)
    if name in _WIDGET_CONSTRUCTORS:
        keyword_values = [
            keyword.value
            for keyword in call.keywords
            if keyword.arg in _CONSTRUCTOR_TEXT_KEYWORDS
        ]
        return [*call.args[:1], *keyword_values]
    if name in _TEXT_SETTERS:
        return call.args[:1]
    return []


def find_untranslated_literals(source: str, path: str) -> list[LiteralFinding]:
    """Find literal text passed to UI calls without tr() or translate().

    Strings with no letters, such as "0", "%" or the empty string, are not user
    text and are skipped.

    Args:
        source (str): Python source code to inspect.
        path (str): Path label recorded on each finding.

    Returns:
        list[LiteralFinding]: Every offending literal, in source order.
    """
    findings: list[LiteralFinding] = []
    for node in ast.walk(ast.parse(source, filename=path)):
        if not isinstance(node, ast.Call):
            continue
        for argument in _candidate_arguments(node):
            if _is_translated(argument):
                continue
            if not (
                isinstance(argument, ast.Constant) and isinstance(argument.value, str)
            ):
                continue
            if not any(character.isalpha() for character in argument.value):
                continue
            findings.append(
                LiteralFinding(
                    path=path,
                    line=node.lineno,
                    call=_callee_name(node) or "",
                    text=argument.value,
                )
            )
    return sorted(findings, key=lambda finding: finding.line)


def scan_ui_tree(root: pathlib.Path) -> list[LiteralFinding]:
    """Scan every Python file under a UI folder for untranslated literals.

    Args:
        root (pathlib.Path): Folder to scan recursively.

    Returns:
        list[LiteralFinding]: Findings across all files, sorted by path then line.
    """
    findings: list[LiteralFinding] = []
    for file_path in sorted(root.rglob("*.py")):
        relative = file_path.relative_to(root).as_posix()
        source = file_path.read_text(encoding="utf-8")
        findings.extend(find_untranslated_literals(source, relative))
    return findings


def test_flags_literal_passed_to_label_constructor() -> None:
    """A plain string given to QLabel is reported with its line number."""
    source = "label = QLabel('Save project')\n"

    findings = find_untranslated_literals(source, "example.py")

    assert [(finding.line, finding.text) for finding in findings] == [
        (1, "Save project")
    ]


def test_flags_literal_in_text_keyword_of_button_constructor() -> None:
    """A text= keyword on QPushButton is checked the same as a positional argument."""
    source = "button = QPushButton(text='Cancel')\n"

    findings = find_untranslated_literals(source, "example.py")

    assert [finding.text for finding in findings] == ["Cancel"]


def test_flags_literal_passed_to_setter_method() -> None:
    """setToolTip with a bare string is reported under the setter's name."""
    source = "widget.setToolTip('Rotate the page')\n"

    findings = find_untranslated_literals(source, "example.py")

    assert [finding.call for finding in findings] == ["setToolTip"]


def test_accepts_literal_wrapped_in_tr_and_translate() -> None:
    """Both self.tr() and QCoreApplication.translate() count as translated."""
    source = (
        "label = QLabel(self.tr('Save project'))\n"
        "widget.setWindowTitle(QtCore.QCoreApplication.translate('X', 'Open'))\n"
    )

    assert find_untranslated_literals(source, "example.py") == []


def test_ignores_strings_without_letters_and_variables() -> None:
    """Numbers, symbols, empty strings and variables are not user text."""
    source = (
        "label = QLabel('0')\n"
        "other = QLabel('%')\n"
        "blank = QLabel('')\n"
        "dynamic = QLabel(message)\n"
    )

    assert find_untranslated_literals(source, "example.py") == []


def test_ui_sources_have_no_untranslated_literals_outside_known_gaps() -> None:
    """Every UI string literal is translated, or is a listed known gap with a reason."""
    findings = scan_ui_tree(_UI_ROOT)

    unexpected = [
        f"{finding.path}:{finding.line} {finding.call}({finding.text!r})"
        for finding in findings
        if (finding.path, finding.text) not in _ALLOWED_LITERALS
    ]

    assert unexpected == []


def test_allow_list_entries_still_match_a_literal() -> None:
    """A known gap that has been wrapped in tr() must be removed from the allow-list."""
    found = {(finding.path, finding.text) for finding in scan_ui_tree(_UI_ROOT)}

    stale = sorted(key for key in _ALLOWED_LITERALS if key not in found)

    assert stale == []


@pytest.mark.parametrize("reason", sorted(set(_ALLOWED_LITERALS.values())))
def test_allow_list_entries_state_a_reason(reason: str) -> None:
    """Every allow-list entry carries a non-empty reason."""
    assert reason.strip()
