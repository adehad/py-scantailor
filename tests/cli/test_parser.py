"""Argument parsing of scantailor-cli: every command, its options and rejections."""

from argparse import Namespace
from pathlib import Path

import pytest

from scantailor.cli.parser import build_parser


def parse(*argv: str) -> Namespace:
    """Parse a command line with the real parser."""
    return build_parser().parse_args(list(argv))


def test_create_takes_source_and_project_paths_with_defaults() -> None:
    args = parse("create", "scans", "proj/project.json")

    assert args.command == "create"
    assert args.source == Path("scans")
    assert args.project == Path("proj/project.json")
    assert args.dpi is None
    assert args.layout_direction == "LTR"
    assert args.force is False


def test_create_reads_dpi_layout_direction_and_force() -> None:
    args = parse(
        "create",
        "scans",
        "p.json",
        "--dpi",
        "600",
        "--layout-direction",
        "RTL",
        "--force",
    )

    assert args.dpi == 600
    assert args.layout_direction == "RTL"
    assert args.force is True


def test_create_rejects_unknown_layout_direction(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        parse("create", "scans", "p.json", "--layout-direction", "TTB")

    assert raised.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_run_reads_worker_count_and_defaults_to_none() -> None:
    assert parse("run", "p.json").workers is None
    assert parse("run", "p.json", "--workers", "3").workers == 3


def test_options_show_takes_only_the_project() -> None:
    args = parse("options", "show", "p.json")

    assert args.command == "options"
    assert args.options_command == "show"
    assert args.project == Path("p.json")


def test_options_set_takes_name_and_raw_value() -> None:
    args = parse("options", "set", "p.json", "output_dpi", '{"horizontal": 600}')

    assert args.options_command == "set"
    assert args.name == "output_dpi"
    assert args.value == '{"horizontal": 600}'


def test_export_pdf_defaults_to_jbig2_codec() -> None:
    args = parse("export", "pdf", "p.json", "out.pdf")

    assert args.export_kind == "pdf"
    assert args.destination == Path("out.pdf")
    assert args.bilevel_codec == "jbig2"


def test_export_pdf_rejects_unknown_codec(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        parse("export", "pdf", "p.json", "out.pdf", "--bilevel-codec", "zip")

    assert raised.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_export_tiff_takes_destination_folder() -> None:
    args = parse("export", "tiff", "p.json", "tiffs")

    assert args.export_kind == "tiff"
    assert args.destination == Path("tiffs")


def test_verbose_flag_is_accepted_before_the_command() -> None:
    assert parse("--verbose", "status", "p.json").verbose is True
    assert parse("status", "p.json").verbose is False


def test_missing_command_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        parse()

    assert raised.value.code == 2
    assert "COMMAND" in capsys.readouterr().err
