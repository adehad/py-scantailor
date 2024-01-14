from __future__ import annotations

import scantailor


def test_version():
    assert scantailor.__version__ != "unknown"
    assert len(scantailor.__version__.split(".")) > 1
