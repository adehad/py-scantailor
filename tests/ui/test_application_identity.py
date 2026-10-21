"""The window app names itself so QSettings can store preferences."""

from collections.abc import Iterator

import pytest
from PySide6 import QtCore

from scantailor.__about__ import APP_NAME, __version__
from scantailor.main import ORGANIZATION_NAME, set_application_identity


@pytest.fixture
def restore_identity() -> Iterator[None]:
    """Put back the names the test fixtures set after the test changes them."""
    app = QtCore.QCoreApplication
    previous = (app.organizationName(), app.applicationName(), app.applicationVersion())
    yield
    app.setOrganizationName(previous[0])
    app.setApplicationName(previous[1])
    app.setApplicationVersion(previous[2])


@pytest.mark.usefixtures("restore_identity")
def test_identity_names_the_organisation_application_and_version() -> None:
    set_application_identity()

    assert QtCore.QCoreApplication.organizationName() == ORGANIZATION_NAME
    assert QtCore.QCoreApplication.applicationName() == APP_NAME
    assert QtCore.QCoreApplication.applicationVersion() == __version__


@pytest.mark.usefixtures("restore_identity")
def test_settings_without_an_identity_cannot_store_values() -> None:
    QtCore.QCoreApplication.setOrganizationName("")
    QtCore.QCoreApplication.setApplicationName("")

    settings = QtCore.QSettings()
    settings.setValue("probe", 1)
    settings.sync()

    assert settings.status() == QtCore.QSettings.Status.AccessError
