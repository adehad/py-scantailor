import pytest

from scantailor.app._ui import resources_rc
from scantailor.main import setup_qcore


@pytest.fixture(scope="session", autouse=True)
def _conf_qt():
    """Session-wide Qt core and resource initialisation."""
    setup_qcore()
    resources_rc.qInitResources()
    yield
    # Explicitly release resources to keep valgrind / sanitizers happy
    if hasattr(resources_rc, "qCleanupResources"):
        resources_rc.qCleanupResources()
