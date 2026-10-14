"""Give a Qt Designer form its text again in the current language.

QUiLoader translates a form's strings once, while loading it, and keeps no record
of the source text. A form that stays on screen across a language change is
therefore given its text again from the ``.ui`` file.
"""

import dataclasses
import enum
import functools
import pathlib
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator

from PySide6 import QtCore, QtWidgets

# Containers whose pages carry their label as an <attribute name="title">.
_PAGE_TITLE_CONTAINERS = frozenset({"QTabWidget", "QToolBox"})


class TextSlot(enum.Enum):
    """Where a translated string goes on the object it belongs to."""

    PROPERTY = enum.auto()
    PAGE_TITLE = enum.auto()
    ITEM = enum.auto()


@dataclasses.dataclass(frozen=True)
class UiText:
    """One translatable string of a form and where it is shown.

    Attributes:
        object_name (str): Object the string is set on: the widget or action
            owning a property, or the container of a page title or an item.
        slot (TextSlot): How the string is set on that object.
        key (str): Property name for a property, the page's object name for a
            page title; empty for an item.
        index (int): Row of an item in its combo box or list widget; -1 otherwise.
        source (str): Untranslated text as written in the form.
        disambiguation (str | None): The string's ``comment`` attribute, which
            lupdate stores as the disambiguation.
    """

    object_name: str
    slot: TextSlot
    key: str
    index: int
    source: str
    disambiguation: str | None


@dataclasses.dataclass(frozen=True)
class UiTexts:
    """The translatable strings of a form.

    Attributes:
        context (str): The form's class name, which lupdate and QUiLoader use as
            the translation context.
        texts (tuple[UiText, ...]): Every translatable string of the form.
    """

    context: str
    texts: tuple[UiText, ...]


def _translatable(string: ET.Element | None) -> ET.Element | None:
    """Return ``string`` when it holds text marked for translation."""
    if string is None or string.get("notr") == "true" or not string.text:
        return None
    return string


def _child_objects(element: ET.Element) -> Iterator[ET.Element]:
    """Yield the widgets and actions directly inside ``element``, through layouts."""
    for child in element:
        if child.tag in ("widget", "action"):
            yield child
        elif child.tag == "layout":
            yield from _layout_widgets(child)


def _layout_widgets(layout: ET.Element) -> Iterator[ET.Element]:
    """Yield the widgets a layout holds, including those of nested layouts."""
    for item in layout.findall("item"):
        for inner in item:
            if inner.tag == "widget":
                yield inner
            elif inner.tag == "layout":
                yield from _layout_widgets(inner)


def _texts_of(element: ET.Element, container: ET.Element | None) -> Iterator[UiText]:
    """Yield the strings of one widget or action and of everything inside it.

    Args:
        element (ET.Element): A ``<widget>`` or ``<action>`` element.
        container (ET.Element | None): The widget enclosing ``element``, which
            shows a page title when it is a tab widget or toolbox.

    Yields:
        UiText: The strings found.
    """
    name = element.get("name", "")
    row = 0
    for child in element:
        if child.tag == "property":
            string = _translatable(child.find("string"))
            if string is not None:
                yield _record(string, name, TextSlot.PROPERTY, child.get("name", ""))
        elif child.tag == "attribute" and child.get("name") == "title":
            string = _translatable(child.find("string"))
            if (
                string is not None
                and container is not None
                and container.get("class") in _PAGE_TITLE_CONTAINERS
            ):
                yield _record(
                    string, container.get("name", ""), TextSlot.PAGE_TITLE, name
                )
        elif child.tag == "item":
            string = _translatable(child.find("property[@name='text']/string"))
            if string is not None:
                yield _record(string, name, TextSlot.ITEM, "", row)
            row += 1
    for child in _child_objects(element):
        yield from _texts_of(child, element)


def _record(
    string: ET.Element, object_name: str, slot: TextSlot, key: str, index: int = -1
) -> UiText:
    """Build the record of one ``<string>`` element."""
    return UiText(
        object_name, slot, key, index, string.text or "", string.get("comment")
    )


@functools.cache
def ui_texts(ui_filename: pathlib.Path) -> UiTexts:
    """Read the translatable strings of a ``.ui`` file.

    Args:
        ui_filename (pathlib.Path): The Qt Designer file.

    Returns:
        UiTexts: The form's translation context and strings.
    """
    root = ET.parse(ui_filename).getroot()
    form = root.find("widget")
    texts = () if form is None else tuple(_texts_of(form, None))
    return UiTexts(root.findtext("class", ""), texts)


def _find(root: QtWidgets.QWidget, name: str) -> QtCore.QObject | None:
    """Find the object called ``name`` in the form, the form itself included."""
    if root.objectName() == name:
        return root
    return root.findChild(QtCore.QObject, name)


def _apply(root: QtWidgets.QWidget, text: UiText, translated: str) -> None:
    """Show ``translated`` where ``text`` belongs, when that object still exists."""
    target = _find(root, text.object_name)
    if target is None:
        return
    match text.slot:
        case TextSlot.PROPERTY:
            target.setProperty(text.key, translated)
        case TextSlot.PAGE_TITLE:
            page = root.findChild(QtWidgets.QWidget, text.key)
            if page is None:
                return
            if isinstance(target, QtWidgets.QTabWidget):
                index = target.indexOf(page)
                if index != -1:
                    target.setTabText(index, translated)
            elif isinstance(target, QtWidgets.QToolBox):
                index = target.indexOf(page)
                if index != -1:
                    target.setItemText(index, translated)
        case TextSlot.ITEM:
            if isinstance(target, QtWidgets.QComboBox) and text.index < target.count():
                target.setItemText(text.index, translated)
            elif isinstance(target, QtWidgets.QListWidget):
                item = target.item(text.index)
                if item is not None:
                    item.setText(translated)


def retranslate_ui(root: QtWidgets.QWidget, ui_filename: pathlib.Path) -> None:
    """Set every translatable string of a loaded form in the current language.

    Text the code changed after loading is reset to the form's, so callers set
    their own text again afterwards.

    Args:
        root (QtWidgets.QWidget): The widget ``load_ui_widget`` returned.
        ui_filename (pathlib.Path): The ``.ui`` file it was loaded from.
    """
    form = ui_texts(ui_filename)
    for text in form.texts:
        translated = QtCore.QCoreApplication.translate(
            form.context, text.source, text.disambiguation
        )
        _apply(root, text, translated)


class LanguageChangeWatcher(QtCore.QObject):
    """Event filter that calls back when the widget it watches changes language."""

    def __init__(
        self, on_change: Callable[[], None], parent: QtCore.QObject | None = None
    ) -> None:
        """Create the filter.

        Args:
            on_change (Callable[[], None]): Called on every LanguageChange event.
            parent (QtCore.QObject | None): Owner of the filter.
        """
        super().__init__(parent)
        self._on_change = on_change

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Call back on a language change and let every event through."""
        if event.type() == QtCore.QEvent.Type.LanguageChange:
            self._on_change()
        return super().eventFilter(watched, event)
