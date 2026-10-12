"""Sphinx extension embedding generated application screenshots.

``.. screenshot:: stage3_deskew`` shows ``light/stage3_deskew.png`` or
``dark/stage3_deskew.png`` according to the reader's colour scheme (furo hides the
other one). Screenshots missing on disk fall back to a placeholder image.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.application import Sphinx
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective

logger = logging.getLogger(__name__)

SKIP_ENV_VAR = "SCANTAILOR_DOCS_SKIP_SCREENSHOTS"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "tools" / "docs_screenshots.py"
_SCREENSHOTS = "_generated/screenshots"
_PLACEHOLDER = "/_static/images/screenshot-missing.svg"
_GENERATION_TIMEOUT_S = 900


def _expected_shots(out_dir: Path) -> list[Path]:
    spec = importlib.util.spec_from_file_location("docs_screenshots", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [
        out_dir / scheme / f"{name}.png"
        for scheme in module.SCHEMES
        for name in module.SHOT_NAMES
    ]


def generate_missing_screenshots(app: Sphinx) -> None:
    """Run the screenshot script when any expected image is absent.

    A failure is logged and the build carries on with placeholders, so a machine
    that cannot start the application can still build the manual.

    Args:
        app (Sphinx): The running application.
    """
    out_dir = Path(app.srcdir) / _SCREENSHOTS
    if os.environ.get(SKIP_ENV_VAR):
        return
    if all(path.is_file() for path in _expected_shots(out_dir)):
        return
    logger.info("Generating documentation screenshots (about a minute)")
    try:
        subprocess.run(
            [sys.executable, str(_SCRIPT), "--out", str(out_dir)],
            check=True,
            timeout=_GENERATION_TIMEOUT_S,
        )
    except (subprocess.SubprocessError, OSError) as error:
        logger.info("Screenshot generation failed, using placeholders: %s", error)


class Screenshot(SphinxDirective):
    """Light and dark variants of one screenshot."""

    required_arguments = 1
    option_spec: ClassVar = {"alt": directives.unchanged}

    def run(self) -> list[nodes.Node]:
        """Build the pair of image nodes.

        Returns:
            list[nodes.Node]: One paragraph holding the light and the dark image.
        """
        name = self.arguments[0]
        alt = self.options.get("alt", name.replace("_", " "))
        images = []
        for scheme in ("light", "dark"):
            relative = f"{_SCREENSHOTS}/{scheme}/{name}.png"
            exists = (Path(self.env.srcdir) / relative).is_file()
            images.append(
                nodes.image(
                    uri=f"/{relative}" if exists else _PLACEHOLDER,
                    alt=alt,
                    classes=["screenshot", f"only-{scheme}"],
                )
            )
        return [nodes.paragraph("", "", *images)]


def setup(app: Sphinx) -> dict[str, bool]:
    """Register the directive and the generation hook.

    Args:
        app (Sphinx): The running application.

    Returns:
        dict[str, bool]: Extension metadata.
    """
    app.add_directive("screenshot", Screenshot)
    app.connect("builder-inited", generate_missing_screenshots)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
