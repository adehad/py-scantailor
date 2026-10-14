"""Qt setup for the documentation screenshot tests."""

import os
import pathlib
from collections.abc import Iterator

# Must be set before the first QApplication is created.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore

from scantailor.app._ui import resources_rc
from scantailor.main import setup_qcore


@pytest.fixture(scope="session", autouse=True)
def _qt_core() -> Iterator[None]:
    """Initialise the Qt core and bundled resources once per worker."""
    setup_qcore()
    resources_rc.qInitResources()
    yield


@pytest.fixture(autouse=True)
def _isolated_qsettings(tmp_path: pathlib.Path) -> Iterator[None]:
    """Keep the screenshot run's QSettings out of the user's profile."""
    settings_dir = tmp_path / "qsettings"
    settings_dir.mkdir()
    previous_format = QtCore.QSettings.defaultFormat()
    previous_org = QtCore.QCoreApplication.organizationName()
    previous_app = QtCore.QCoreApplication.applicationName()
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
    for scope in (
        QtCore.QSettings.Scope.UserScope,
        QtCore.QSettings.Scope.SystemScope,
    ):
        QtCore.QSettings.setPath(
            QtCore.QSettings.Format.IniFormat, scope, str(settings_dir)
        )
    QtCore.QCoreApplication.setOrganizationName("scantailor-docs")
    QtCore.QCoreApplication.setApplicationName("scantailor-docs")
    yield
    QtCore.QSettings.setDefaultFormat(previous_format)
    QtCore.QCoreApplication.setOrganizationName(previous_org)
    QtCore.QCoreApplication.setApplicationName(previous_app)
