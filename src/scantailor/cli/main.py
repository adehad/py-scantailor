"""Dispatch parsed commands to their operations and map failures to exit codes."""

import sys
from argparse import Namespace
from collections.abc import Sequence
from typing import TextIO

from loguru import logger
from pydantic import BaseModel

from scantailor.cli.create import create_project
from scantailor.cli.errors import CliError, ExitCode
from scantailor.cli.operations import (
    export_pdf_file,
    export_tiff_files,
    project_status,
    run_project,
    set_option,
    show_options,
)
from scantailor.cli.parser import build_parser
from scantailor.cli.reports import BatchReport


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and return its exit code.

    Args:
        argv (Sequence[str] | None): Arguments without the program name; the
            process arguments when None.

    Returns:
        int: An ExitCode value.
    """
    args = build_parser().parse_args(argv)
    _configure_logging(verbose=args.verbose)
    try:
        report = _dispatch(args)
    except CliError as error:
        sys.stderr.write(f"error: {error.message}\n")
        return int(error.code)
    except Exception:
        logger.exception("Unexpected failure")
        sys.stderr.write("error: unexpected failure, see the log on stderr\n")
        return int(ExitCode.FAILED)
    _write_json(sys.stdout, report)
    if isinstance(report, BatchReport) and report.failed:
        return int(ExitCode.FAILED)
    return int(ExitCode.OK)


def _dispatch(args: Namespace) -> BaseModel:
    """Call the operation named by the parsed arguments.

    Args:
        args (argparse.Namespace): The parsed command line.

    Returns:
        BaseModel: The report of the operation, written as JSON by the caller.
    """
    command = args.command
    if command == "create":
        return create_project(
            args.source,
            args.project,
            dpi=args.dpi,
            layout_direction=args.layout_direction,
            output_directory=args.output_dir,
            pages_directory=args.pages_dir,
            force=args.force,
        )
    if command == "status":
        return project_status(args.project)
    if command == "options":
        if args.options_command == "show":
            return show_options(args.project)
        return set_option(args.project, args.name, args.value)
    if command == "run":
        return run_project(args.project, args.workers)
    if args.export_kind == "pdf":
        return export_pdf_file(args.project, args.destination, args.bilevel_codec)
    return export_tiff_files(args.project, args.destination)


def _write_json(stream: TextIO, report: BaseModel) -> None:
    """Write a report as indented JSON followed by a newline."""
    stream.write(report.model_dump_json(indent=2) + "\n")


def _configure_logging(*, verbose: bool) -> None:
    """Send log records to stderr so stdout carries only the JSON result."""
    logger.remove()
    logger.add(sys.stderr, level="INFO" if verbose else "WARNING")


if __name__ == "__main__":
    raise SystemExit(main())
