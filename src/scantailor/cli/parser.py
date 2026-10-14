"""Command-line grammar for scantailor-cli."""

import argparse
from pathlib import Path

from scantailor.cli.create import DEFAULT_PDF_DPI

PROGRAM = "scantailor-cli"


def build_parser() -> argparse.ArgumentParser:
    """Build the parser with one subcommand per action.

    Returns:
        argparse.ArgumentParser: The configured parser. Parsing a command line
            yields a namespace whose ``command`` attribute names the action.
    """
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description=(
            "Headless ScanTailor: create projects, set options, run the batch and "
            "export. Results are written to stdout as JSON; diagnostics go to stderr."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="log progress to stderr")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    create = commands.add_parser(
        "create", help="create a project from a folder of images or a PDF"
    )
    create.add_argument("source", type=Path, help="image folder or PDF file")
    create.add_argument("project", type=Path, help="project JSON file to write")
    create.add_argument(
        "--dpi",
        type=int,
        help=f"resolution for every image; the PDF render resolution "
        f"(default {DEFAULT_PDF_DPI} for PDFs)",
    )
    create.add_argument(
        "--layout-direction",
        choices=["LTR", "RTL"],
        default="LTR",
        help="reading order of split pages (default LTR)",
    )
    create.add_argument(
        "--output-dir", type=Path, help="folder for processed files (default out/)"
    )
    create.add_argument(
        "--pages-dir", type=Path, help="folder for rendered PDF pages (PDF only)"
    )
    create.add_argument(
        "--force", action="store_true", help="replace an existing project file"
    )

    status = commands.add_parser("status", help="print project state as JSON")
    status.add_argument("project", type=Path)

    options = commands.add_parser("options", help="read or change stage-wide options")
    option_commands = options.add_subparsers(
        dest="options_command", required=True, metavar="ACTION"
    )
    show = option_commands.add_parser("show", help="print every option as JSON")
    show.add_argument("project", type=Path)
    set_ = option_commands.add_parser("set", help="change one option and save")
    set_.add_argument("project", type=Path)
    set_.add_argument("name", help="option name from 'options show'")
    set_.add_argument("value", help="new value; JSON, or plain text")

    run = commands.add_parser(
        "run", help="process every page through all stages and save the project"
    )
    run.add_argument("project", type=Path)
    run.add_argument(
        "--workers", type=int, help="worker processes (default: CPU count)"
    )

    export = commands.add_parser("export", help="write processed output elsewhere")
    export_kinds = export.add_subparsers(
        dest="export_kind", required=True, metavar="FORMAT"
    )
    pdf = export_kinds.add_parser("pdf", help="write the processed pages as one PDF")
    pdf.add_argument("project", type=Path)
    pdf.add_argument("destination", type=Path, help="PDF file to write")
    pdf.add_argument(
        "--bilevel-codec",
        choices=["jbig2", "g4"],
        default="jbig2",
        help="codec for black and white pages (default jbig2)",
    )
    tiff = export_kinds.add_parser("tiff", help="copy the processed TIFF files")
    tiff.add_argument("project", type=Path)
    tiff.add_argument("destination", type=Path, help="folder to copy into")
    return parser
