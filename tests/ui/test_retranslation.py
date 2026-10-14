"""Tests that a loaded Qt Designer form can be given its text in a new language."""

import pathlib
from collections.abc import Callable, Iterator

import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui import UI_FOLDER, load_ui_widget
from scantailor.app.ui import MainWindow as main_window_module
from scantailor.app.ui.MainWindow import APP_TITLE, MainWindowManager
from scantailor.app.ui.retranslation import (
    LanguageChangeWatcher,
    TextSlot,
    retranslate_ui,
    ui_texts,
)

Catalogue = dict[tuple[str, str], str]

_PROBE_UI = """<?xml version="1.0" encoding="UTF-8"?>
<ui version="4.0">
 <class>Probe</class>
 <widget class="QWidget" name="Probe">
  <property name="windowTitle">
   <string>Probe window</string>
  </property>
  <layout class="QVBoxLayout" name="outer">
   <item>
    <widget class="QLabel" name="greeting">
     <property name="text">
      <string>Hello</string>
     </property>
     <property name="toolTip">
      <string>Says hello</string>
     </property>
    </widget>
   </item>
   <item>
    <widget class="QLabel" name="brand">
     <property name="text">
      <string notr="true">ScanTailor</string>
     </property>
    </widget>
   </item>
   <item>
    <layout class="QHBoxLayout" name="inner">
     <item>
      <widget class="QGroupBox" name="nestedGroup">
       <property name="title">
        <string>Nested</string>
       </property>
       <layout class="QVBoxLayout" name="groupLayout">
        <item>
         <widget class="QPushButton" name="openButton">
          <property name="text">
           <string comment="verb">Open</string>
          </property>
         </widget>
        </item>
       </layout>
      </widget>
     </item>
    </layout>
   </item>
   <item>
    <widget class="QComboBox" name="choices">
     <item>
      <property name="text">
       <string>One</string>
      </property>
     </item>
     <item>
      <property name="text">
       <string>Two</string>
      </property>
     </item>
    </widget>
   </item>
   <item>
    <widget class="QTabWidget" name="tabs">
     <widget class="QWidget" name="firstPage">
      <attribute name="title">
       <string>First</string>
      </attribute>
     </widget>
     <widget class="QWidget" name="secondPage">
      <attribute name="title">
       <string>Second</string>
      </attribute>
     </widget>
    </widget>
   </item>
  </layout>
  <action name="actionQuit">
   <property name="text">
    <string>Quit</string>
   </property>
  </action>
 </widget>
</ui>
"""

_GERMAN: Catalogue = {
    ("Probe", "Probe window"): "Probefenster",
    ("Probe", "Hello"): "Hallo",
    ("Probe", "Says hello"): "Sagt hallo",
    ("Probe", "Nested"): "Verschachtelt",
    ("Probe", "Open"): "Öffnen",
    ("Probe", "One"): "Eins",
    ("Probe", "Two"): "Zwei",
    ("Probe", "First"): "Erste",
    ("Probe", "Second"): "Zweite",
    ("Probe", "Quit"): "Beenden",
}


class CatalogueTranslator(QtCore.QTranslator):
    """Translator answering from a dictionary, so no compiled .qm is needed."""

    def __init__(self, catalogue: Catalogue, parent: QtCore.QObject) -> None:
        """Create the translator.

        Args:
            catalogue (Catalogue): Translations keyed by (context, source text).
            parent (QtCore.QObject): Owner of the translator.
        """
        super().__init__(parent)
        self._catalogue = catalogue

    def translate(
        self,
        context: str,
        source_text: str,
        disambiguation: str | None = None,
        n: int = -1,
    ) -> str | None:
        """Return the catalogue's text, or None so Qt falls back to the source."""
        return self._catalogue.get((context, source_text))

    def isEmpty(self) -> bool:
        """Report that the translator has messages, so Qt consults it."""
        return False


@pytest.fixture
def install_catalogue(
    qapp: QtWidgets.QApplication,
) -> Iterator[Callable[[Catalogue], CatalogueTranslator]]:
    """Install dictionary translators on the application; remove them afterwards."""
    installed: list[CatalogueTranslator] = []

    def install(catalogue: Catalogue) -> CatalogueTranslator:
        translator = CatalogueTranslator(catalogue, qapp)
        qapp.installTranslator(translator)
        installed.append(translator)
        return translator

    yield install
    for translator in installed:
        qapp.removeTranslator(translator)
        translator.setParent(None)


@pytest.fixture
def probe_ui(tmp_path: pathlib.Path) -> pathlib.Path:
    """Write the probe form to a file, as it would ship beside its module."""
    path = tmp_path / "Probe.ui"
    path.write_text(_PROBE_UI, encoding="utf-8")
    return path


@pytest.fixture
def probe(probe_ui: pathlib.Path) -> Iterator[QtWidgets.QWidget]:
    """Load the probe form in English."""
    form = load_ui_widget(probe_ui)
    yield form
    form.deleteLater()


def child(form: QtWidgets.QWidget, kind: type, name: str) -> object:
    """Find a named child of the form, failing the test when it is missing."""
    found = form.findChild(kind, name)
    assert found is not None, name
    return found


def test_every_kind_of_form_text_is_shown_in_the_installed_language(
    probe: QtWidgets.QWidget,
    probe_ui: pathlib.Path,
    install_catalogue: Callable[[Catalogue], CatalogueTranslator],
) -> None:
    """Properties, nested widgets, combo items, tab titles and actions all change."""
    install_catalogue(_GERMAN)

    retranslate_ui(probe, probe_ui)

    greeting = child(probe, QtWidgets.QLabel, "greeting")
    choices = child(probe, QtWidgets.QComboBox, "choices")
    tabs = child(probe, QtWidgets.QTabWidget, "tabs")
    assert probe.windowTitle() == "Probefenster"
    assert greeting.text() == "Hallo"
    assert greeting.toolTip() == "Sagt hallo"
    assert child(probe, QtWidgets.QGroupBox, "nestedGroup").title() == "Verschachtelt"
    assert child(probe, QtWidgets.QPushButton, "openButton").text() == "Öffnen"
    assert [choices.itemText(i) for i in range(choices.count())] == ["Eins", "Zwei"]
    assert [tabs.tabText(i) for i in range(tabs.count())] == ["Erste", "Zweite"]
    assert child(probe, QtGui.QAction, "actionQuit").text() == "Beenden"


def test_text_marked_notr_is_left_as_written(
    probe: QtWidgets.QWidget,
    probe_ui: pathlib.Path,
    install_catalogue: Callable[[Catalogue], CatalogueTranslator],
) -> None:
    """A string Designer marks as not translatable is never looked up or reset."""
    install_catalogue({("Probe", "ScanTailor"): "Wrong"})
    brand = child(probe, QtWidgets.QLabel, "brand")

    retranslate_ui(probe, probe_ui)

    assert brand.text() == "ScanTailor"
    assert all(text.source != "ScanTailor" for text in ui_texts(probe_ui).texts)


def test_removing_the_catalogue_returns_the_form_to_english(
    probe: QtWidgets.QWidget,
    probe_ui: pathlib.Path,
    qapp: QtWidgets.QApplication,
    install_catalogue: Callable[[Catalogue], CatalogueTranslator],
) -> None:
    """Translating again after the translator is removed shows the source text."""
    translator = install_catalogue(_GERMAN)
    retranslate_ui(probe, probe_ui)

    qapp.removeTranslator(translator)
    retranslate_ui(probe, probe_ui)

    assert child(probe, QtWidgets.QLabel, "greeting").text() == "Hello"
    assert child(probe, QtWidgets.QTabWidget, "tabs").tabText(1) == "Second"


def test_disambiguation_is_passed_to_the_translator(probe_ui: pathlib.Path) -> None:
    """The ``comment`` attribute is read as the disambiguation lupdate files."""
    open_text = next(text for text in ui_texts(probe_ui).texts if text.source == "Open")

    assert open_text.disambiguation == "verb"
    assert open_text.slot is TextSlot.PROPERTY


def test_a_widget_removed_after_loading_is_skipped(
    probe: QtWidgets.QWidget,
    probe_ui: pathlib.Path,
    install_catalogue: Callable[[Catalogue], CatalogueTranslator],
) -> None:
    """Text for a widget the code deleted does not stop the rest being translated."""
    greeting = child(probe, QtWidgets.QLabel, "greeting")
    greeting.setParent(None)
    greeting.deleteLater()
    install_catalogue(_GERMAN)

    retranslate_ui(probe, probe_ui)

    assert child(probe, QtGui.QAction, "actionQuit").text() == "Beenden"


def test_the_main_window_form_is_read_in_its_own_context() -> None:
    """MainWindow.ui yields its menus and actions under the MainWindow context."""
    form = ui_texts(UI_FOLDER / "MainWindow.ui")

    pairs = {(text.key, text.source) for text in form.texts}
    assert form.context == "MainWindow"
    assert ("title", "File") in pairs
    assert ("windowTitle", "Thumbnails") in pairs


def test_watcher_calls_back_on_a_language_change_only(
    qapp: QtWidgets.QApplication,
) -> None:
    """Other events pass through without a callback; none are swallowed."""
    calls: list[None] = []
    watched = QtCore.QObject()
    watcher = LanguageChangeWatcher(lambda: calls.append(None), watched)
    watched.installEventFilter(watcher)

    QtCore.QCoreApplication.sendEvent(watched, QtCore.QEvent(QtCore.QEvent.Type.Show))
    QtCore.QCoreApplication.sendEvent(
        watched, QtCore.QEvent(QtCore.QEvent.Type.LanguageChange)
    )

    assert len(calls) == 1


_NOT_READY = (
    "Output is not yet possible, as the final size of pages is not yet known.\n"
    'To determine it, run batch processing at "Select Content" or "Margins".'
)
_MAIN_WINDOW_GERMAN: Catalogue = {
    ("MainWindow", "File"): "Datei",
    ("MainWindow", "Thumbnails"): "Miniaturansichten",
    ("MainWindow", "MainWindow"): "Hauptfenster",
    ("MainWindowManager", "Add Files..."): "Dateien hinzufügen...",
    ("MainWindowManager", "Open Recent"): "Zuletzt geöffnet",
    ("MainWindowManager", _NOT_READY): "Noch keine Ausgabe möglich.",
    ("MainWindowManager", "Unnamed"): "Unbenannt",
    ("NewOpenProjectPanel", "New Project ..."): "Neues Projekt ...",
}


def menu_titles(window: MainWindowManager) -> list[str]:
    """Return the titles of the main window's menu bar."""
    return [action.text() for action in window.ui.menuBar().actions()]


def dock_titles(window: MainWindowManager) -> list[str]:
    """Return the titles of the main window's docks."""
    return [
        dock.windowTitle() for dock in window.ui.findChildren(QtWidgets.QDockWidget)
    ]


def welcome_link_texts(window: MainWindowManager) -> list[str]:
    """Return the New Project link text of every welcome panel still alive."""
    return [
        label.text()
        for label in window.ui.findChildren(QtWidgets.QLabel, "newProjectLabel")
    ]


def test_open_main_window_follows_a_language_change(
    main_window: MainWindowManager,
    qtbot: QtBot,
    install_catalogue: Callable[[Catalogue], CatalogueTranslator],
) -> None:
    """Menus, docks and the text the window builds in code change straight away."""
    install_catalogue(_MAIN_WINDOW_GERMAN)

    qtbot.waitUntil(lambda: "Datei" in menu_titles(main_window))

    assert "Miniaturansichten" in dock_titles(main_window)
    assert main_window.actionAddFiles.text() == "Dateien hinzufügen..."
    assert main_window.menuOpenRecent.title() == "Zuletzt geöffnet"
    assert main_window.not_ready_widget.text() == "Noch keine Ausgabe möglich."


def test_window_title_is_kept_when_the_form_is_retranslated(
    main_window: MainWindowManager,
    qtbot: QtBot,
    install_catalogue: Callable[[Catalogue], CatalogueTranslator],
) -> None:
    """The form's placeholder title is not shown in place of the application name."""
    install_catalogue(_MAIN_WINDOW_GERMAN)

    qtbot.waitUntil(lambda: "Datei" in menu_titles(main_window))

    assert main_window.ui.windowTitle() == APP_TITLE


def test_welcome_panel_is_rebuilt_in_the_new_language(
    main_window: MainWindowManager,
    qtbot: QtBot,
    install_catalogue: Callable[[Catalogue], CatalogueTranslator],
) -> None:
    """The New/Open panel shown with no project open changes language too."""
    assert welcome_link_texts(main_window) == ["New Project ..."]
    install_catalogue(_MAIN_WINDOW_GERMAN)

    qtbot.waitUntil(lambda: "Neues Projekt ..." in welcome_link_texts(main_window))


def test_unnamed_project_title_follows_the_language(
    project_window: MainWindowManager,
    qtbot: QtBot,
    install_catalogue: Callable[[Catalogue], CatalogueTranslator],
) -> None:
    """An unsaved project's title uses the translated "Unnamed" after a change."""
    assert project_window.ui.windowTitle().startswith("Unnamed[*]")
    install_catalogue(_MAIN_WINDOW_GERMAN)

    qtbot.waitUntil(lambda: project_window.ui.windowTitle().startswith("Unbenannt[*]"))


def test_returning_to_english_restores_the_source_text(
    main_window: MainWindowManager,
    qtbot: QtBot,
    qapp: QtWidgets.QApplication,
    install_catalogue: Callable[[Catalogue], CatalogueTranslator],
) -> None:
    """Removing the translator shows the window's English text again."""
    translator = install_catalogue(_MAIN_WINDOW_GERMAN)
    qtbot.waitUntil(lambda: "Datei" in menu_titles(main_window))

    qapp.removeTranslator(translator)

    qtbot.waitUntil(lambda: "File" in menu_titles(main_window))
    assert main_window.actionAddFiles.text() == "Add Files..."


def test_closing_the_project_releases_thumbnail_memory(
    project_window: MainWindowManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kept thumbnails and shrunk sources are dropped once no project is open."""
    released: list[None] = []
    monkeypatch.setattr(
        main_window_module, "release_thumbnail_memory", lambda: released.append(None)
    )
    project_window.modified = False

    project_window.actionCloseProject.trigger()

    assert project_window.project is None
    assert released == [None]
