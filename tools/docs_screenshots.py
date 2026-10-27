"""Entry point for the documentation screenshots.

The screenshots are taken by the pytest tests marked ``docs_screenshot`` in
``tests/docs``. The Sphinx build reads ``SCHEMES`` and ``SHOT_NAMES`` from this file
to decide which images are missing. ``main`` runs the tests and writes the PNGs to
``--out``.

Usage:
    python tools/docs_screenshots.py [--out DIR]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "docs" / "_generated" / "screenshots"
OUT_ENV_VAR = "SCANTAILOR_DOCS_SCREENSHOTS"
SCHEMES = ("light", "dark")

SHOT_NAMES = (
    "welcome",
    "new_project",
    "stage1_fix_orientation",
    "stage2_split_pages",
    "stage3_deskew",
    "stage4_select_content",
    "stage5_margins",
    "stage6_output",
    "stage1_fix_orientation_trim",
    "stage3_deskew_oblique",
    "stage5_margins_apply",
    "stage2_split_pages_tilted",
    "stage2_split_pages_removed_half",
    "stage5_margins_guides",
    "thumbnails_incomplete",
    "stage6_tab_picture_mask",
    "status_tip",
    "stage6_output_options",
    "stage6_output_guides",
    "stage6_output_edge_methods",
    "stage6_output_picture_shape",
    "stage6_dewarping_metrics",
    "stage6_tab_picture_zones",
    "stage6_tab_fill_zones",
    "stage6_tab_dewarping",
    "stage6_tab_despeckling",
    "stage6_zone_menu",
    "stage6_zone_properties",
    "thumbnails_sorting",
    "tools_menu",
    "settings",
    "default_parameters",
    "batch_progress",
    "export_pdf",
    "fix_dpi",
    "relinking",
    "remove_pages",
    "open_pdf",
    "thumbnail_menu",
    "stage_play_button",
    "stage4_select_content_page_box_auto",
    "stage6_output_despeckle_levels",
)


def main(argv: list[str] | None = None) -> int:
    """Run the screenshot tests and report where the PNGs went.

    Args:
        argv (list[str] | None): Command line without the program name.

    Returns:
        int: The exit code of pytest; zero when every screenshot was written.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    environment = {**os.environ, OUT_ENV_VAR: str(args.out.resolve())}
    result = subprocess.run(
        # The screenshot tests form two xdist groups, so more workers than two
        # only add start-up time; -n auto would start one per core.
        [
            sys.executable,
            "-m",
            "pytest",
            "-m",
            "docs_screenshot",
            "-n",
            "2",
            "tests/docs",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
