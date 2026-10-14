"""Each stage apply dialog translates its labels under its own class name."""

import ast
import importlib.util
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from PySide6 import QtCore, QtWidgets

from scantailor.app.ui.dialogs.apply_to import (
    DESKEW_OPTIONS,
    MARGIN_SIDE_ICONS,
    MARGIN_SIDE_OPTIONS,
    MARGINS_OPTIONS_TITLE,
    ApplyChangeDewarpingDialog,
    ApplyChangeDpiDialog,
    ApplyColorsDialog,
    ApplyDeskewDialog,
    ApplyFixOrientationDialog,
    ApplyPageLayoutDialog,
    ApplySelectContentDialog,
    ApplyToDialog,
)
from scantailor.app.ui.MainWindow import MainWindowManager
from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.core.stage_sequence import FilterStage
from scantailor.filters.output.params import ApplyGroup
from scantailor.translations import TRANSLATIONS_DIR, load_translation

_REPO_ROOT = Path(__file__).resolve().parents[2]
_APPLY_TO_SOURCE = _REPO_ROOT / "src/scantailor/app/ui/dialogs/apply_to.py"
_SELECT_CONTENT_OPTIONS = {
    "content_box": "Apply content box",
    "page_box": "Apply page box",
}
_SUBCLASS_CASES = [
    pytest.param(ApplyDeskewDialog, {"options": DESKEW_OPTIONS}, id="deskew"),
    pytest.param(ApplyFixOrientationDialog, {}, id="fix_orientation"),
    pytest.param(
        ApplyPageLayoutDialog,
        {
            "options": MARGIN_SIDE_OPTIONS,
            "option_icons": MARGIN_SIDE_ICONS,
            "options_title": MARGINS_OPTIONS_TITLE,
        },
        id="page_layout",
    ),
    pytest.param(
        ApplySelectContentDialog,
        {"options": _SELECT_CONTENT_OPTIONS},
        id="select_content",
    ),
    pytest.param(ApplyColorsDialog, {}, id="output_colors"),
    pytest.param(ApplyChangeDpiDialog, {}, id="output_dpi"),
    pytest.param(ApplyChangeDewarpingDialog, {}, id="output_dewarping"),
]


def _page(name: str) -> PageId:
    return PageId(
        image_id=ImageId(file_path=Path(f"{name}.png")),
        sub_page=SubPage.SINGLE_PAGE,
    )


PAGE = _page("a")
OTHER = _page("b")


def _shown_labels(dialog: QtWidgets.QDialog) -> set[str]:
    """Return every label text a dialog shows, excluding its window title.

    Args:
        dialog (QtWidgets.QDialog): The dialog to read.

    Returns:
        set[str]: Radio button, check box, hint and group box texts.
    """
    labels: set[str] = set()
    for widget_type in (QtWidgets.QRadioButton, QtWidgets.QCheckBox, QtWidgets.QLabel):
        labels.update(widget.text() for widget in dialog.findChildren(widget_type))
    labels.update(group.title() for group in dialog.findChildren(QtWidgets.QGroupBox))
    return labels


def _marker_pairs(class_name: str) -> list[tuple[str, str]]:
    """Read the QT_TRANSLATE_NOOP calls inside a class body from the source.

    lupdate reads these calls literally, so the source is the oracle here.

    Args:
        class_name (str): Name of the class in apply_to.py.

    Returns:
        list[tuple[str, str]]: (context, source text) for each marker call.
    """
    tree = ast.parse(_APPLY_TO_SOURCE.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    pairs: list[tuple[str, str]] = []
    for node in ast.walk(class_node):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == (
            "QT_TRANSLATE_NOOP"
        ):
            context, text = node.args
            assert isinstance(context, ast.Constant)
            assert isinstance(text, ast.Constant)
            pairs.append((str(context.value), str(text.value)))
    return pairs


@pytest.fixture(scope="module")
def german_qm_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile the bundled .ts catalogues into a temporary folder.

    Args:
        tmp_path_factory (pytest.TempPathFactory): Pytest temporary folder factory.

    Returns:
        Path: Folder holding the compiled scantailor_*.qm files.
    """
    spec = importlib.util.spec_from_file_location(
        "build_translations", _REPO_ROOT / "tools" / "build_translations.py"
    )
    assert spec is not None and spec.loader is not None
    build_script: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_script)
    out_dir = tmp_path_factory.mktemp("qm")
    build_script.compile_all(TRANSLATIONS_DIR, out_dir, build_script.find_lrelease())
    return out_dir


@pytest.fixture
def german_ui(qapp: QtWidgets.QApplication, german_qm_dir: Path) -> Iterator[None]:
    """Install the German translators for the duration of one test.

    Args:
        qapp (QtWidgets.QApplication): The shared application.
        german_qm_dir (Path): Folder holding the compiled catalogues.

    Yields:
        None: Control returns to the test while the translators are installed.
    """
    translators = load_translation(qapp, QtCore.QLocale("de"), german_qm_dir)
    yield
    for translator in translators:
        qapp.removeTranslator(translator)


@pytest.mark.parametrize(("dialog_class", "kwargs"), _SUBCLASS_CASES)
def test_declared_labels_are_exactly_the_text_the_dialog_shows(
    qapp: QtWidgets.QApplication,
    dialog_class: type[ApplyToDialog],
    kwargs: dict[str, Any],
) -> None:
    dialog = dialog_class(PAGE, [PAGE, OTHER], [PAGE, OTHER], **kwargs)

    assert set(dialog_class.translatable_labels) == _shown_labels(dialog)


@pytest.mark.parametrize(("dialog_class", "kwargs"), _SUBCLASS_CASES)
def test_markers_are_literal_calls_under_the_class_name(
    dialog_class: type[ApplyToDialog], kwargs: dict[str, Any]
) -> None:
    markers = _marker_pairs(dialog_class.__name__)

    assert markers, "lupdate needs literal QT_TRANSLATE_NOOP calls in the class body"
    assert {context for context, _ in markers} == {dialog_class.__name__}
    assert {text for _, text in markers} == set(dialog_class.translatable_labels)


@pytest.mark.parametrize(("dialog_class", "kwargs"), _SUBCLASS_CASES)
def test_german_catalogue_translates_all_pages_under_the_stage_class(
    qapp: QtWidgets.QApplication,
    german_ui: None,
    dialog_class: type[ApplyToDialog],
    kwargs: dict[str, Any],
) -> None:
    dialog = dialog_class(PAGE, [PAGE, OTHER], [PAGE, OTHER], **kwargs)
    all_pages = dialog.findChild(QtWidgets.QRadioButton, "allPagesRB")

    assert all_pages is not None
    assert all_pages.text() == "Alle Seiten"  # codespell:ignore alle


def test_german_catalogue_translates_deskew_option_under_deskew_class(
    qapp: QtWidgets.QApplication, german_ui: None
) -> None:
    dialog = ApplyDeskewDialog(
        PAGE, [PAGE, OTHER], [PAGE, OTHER], options=DESKEW_OPTIONS
    )
    deskew_box = dialog.findChild(QtWidgets.QCheckBox, "deskewOption")

    assert deskew_box is not None
    assert deskew_box.text() == "Winkel und Modus beim Geraderichten"


@pytest.mark.parametrize(
    ("stage", "part", "expected"),
    [
        pytest.param(
            FilterStage.FIX_ORIENTATION, "", ApplyFixOrientationDialog, id="fix"
        ),
        pytest.param(FilterStage.DESKEW, "", ApplyDeskewDialog, id="deskew"),
        pytest.param(
            FilterStage.SELECT_CONTENT, "", ApplySelectContentDialog, id="select"
        ),
        pytest.param(
            FilterStage.PAGE_LAYOUT, "margins", ApplyPageLayoutDialog, id="margins"
        ),
        pytest.param(
            FilterStage.PAGE_LAYOUT, "alignment", ApplyPageLayoutDialog, id="alignment"
        ),
        pytest.param(
            FilterStage.OUTPUT, ApplyGroup.DPI.value, ApplyChangeDpiDialog, id="dpi"
        ),
        pytest.param(
            FilterStage.OUTPUT,
            ApplyGroup.DEWARPING.value,
            ApplyChangeDewarpingDialog,
            id="dewarping",
        ),
        pytest.param(
            FilterStage.OUTPUT,
            ApplyGroup.COLOR_MODE.value,
            ApplyColorsDialog,
            id="colour",
        ),
    ],
)
def test_main_window_picks_the_dialog_class_for_each_stage_and_group(
    stage: FilterStage, part: str, expected: type[ApplyToDialog]
) -> None:
    assert MainWindowManager._apply_dialog_class(stage, part) is expected
