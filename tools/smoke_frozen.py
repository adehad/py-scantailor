"""End-to-end smoke test of a frozen scantailor-cli.

Renders a few sample pages, then drives the frozen executable through
``create``, ``run`` and ``export pdf`` and checks each JSON result, the page
count of the PDF and the log file ``run --log-file`` wrote. Exits non-zero with
a message on the first failure.

Usage:
    python tools/smoke_frozen.py dist/ScanTailor/scantailor-cli
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Final

import pikepdf
from PIL import Image
from sample_book import SAMPLE_DPI, render_sample_page

# Single pages only, so page split leaves the count unchanged.
SMOKE_PAGES: Final = ("01_single_page", "05_skewed", "08_pictures")
FROZEN_BANNER: Final = "frozen (PyInstaller)"
SESSION_ENDED: Final = "| session ended"
_COMMAND_TIMEOUT_SECONDS: Final = 600


class SmokeFailure(Exception):
    """A step of the smoke test produced the wrong result."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line.

    Args:
        argv (list[str] | None): Command line without the program name; the
            process arguments when ``None``.

    Returns:
        argparse.Namespace: ``cli`` (the executable) and ``workers``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cli", type=Path, help="path to the frozen scantailor-cli")
    parser.add_argument(
        "--workers", type=int, default=2, help="worker processes for run (default 2)"
    )
    return parser.parse_args(argv)


def write_pages(directory: Path) -> list[Path]:
    """Render the smoke pages as PNG files.

    Args:
        directory (Path): Folder to write the images into; created when missing.

    Returns:
        list[Path]: The written images, in project order.
    """
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in SMOKE_PAGES:
        path = directory / f"{name}.png"
        Image.fromarray(render_sample_page(name).image).save(
            path, dpi=(SAMPLE_DPI, SAMPLE_DPI)
        )
        paths.append(path)
    return paths


def run_cli(cli: Path, *args: str | Path) -> dict[str, Any]:
    """Run one CLI command and parse the JSON it prints.

    Args:
        cli (Path): The frozen executable.
        *args (str | Path): The command and its arguments.

    Returns:
        dict[str, Any]: The command's JSON result.

    Raises:
        SmokeFailure: The command exited non-zero or printed something other
            than JSON.
    """
    command = [str(cli), *map(str, args)]
    sys.stderr.write(f"$ {' '.join(command)}\n")
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=_COMMAND_TIMEOUT_SECONDS,
        check=False,
    )
    sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        raise SmokeFailure(
            f"{args[0]} exited with {completed.returncode}:\n{completed.stdout}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise SmokeFailure(
            f"{args[0]} printed no JSON ({error}):\n{completed.stdout}"
        ) from error
    sys.stderr.write(json.dumps(result, indent=2) + "\n")
    return result


def expect(condition: bool, message: str) -> None:
    """Raise when a check fails.

    Args:
        condition (bool): The check's outcome.
        message (str): What went wrong, shown when ``condition`` is false.

    Raises:
        SmokeFailure: ``condition`` is false.
    """
    if not condition:
        raise SmokeFailure(message)


def check_log_file(path: Path) -> None:
    """Check that a frozen CLI session wrote its banner and ended cleanly.

    Args:
        path (Path): The file passed to ``--log-file``.

    Raises:
        SmokeFailure: The file is missing, lacks the frozen-build banner line, or
            does not end with the clean-exit marker.
    """
    expect(path.is_file(), f"no log file at {path}")
    lines = path.read_text(encoding="utf-8").rstrip().splitlines()
    expect(
        any(line.endswith(FROZEN_BANNER) for line in lines),
        f"{path} has no '{FROZEN_BANNER}' banner line",
    )
    expect(
        bool(lines) and lines[-1].endswith(SESSION_ENDED),
        f"{path} does not end with '{SESSION_ENDED}'",
    )


def smoke(cli: Path, workers: int, work_dir: Path) -> None:
    """Drive the frozen CLI through create, run and export pdf.

    Args:
        cli (Path): The frozen executable.
        workers (int): Worker processes for ``run``.
        work_dir (Path): Empty scratch folder.

    Raises:
        SmokeFailure: A step failed or reported the wrong counts.
    """
    pages = write_pages(work_dir / "pages")
    project = work_dir / "smoke.json"
    pdf = work_dir / "smoke.pdf"
    log_file = work_dir / "run.log"

    created = run_cli(cli, "create", pages[0].parent, project)
    expect(
        created["pages"] == len(pages),
        f"create added {created['pages']} pages, expected {len(pages)}",
    )

    batch = run_cli(
        cli, "--log-file", log_file, "run", project, "--workers", str(workers)
    )
    expect(not batch["failed"], f"run failed pages: {batch['failed']}")
    expect(not batch["cancelled"], f"run cancelled {batch['cancelled']} pages")
    expect(
        batch["written"] == len(pages),
        f"run wrote {batch['written']} pages, expected {len(pages)}",
    )
    check_log_file(log_file)

    exported = run_cli(cli, "export", "pdf", project, pdf)
    expect(not exported["skipped"], f"export skipped pages: {exported['skipped']}")
    expect(
        exported["exported"] == len(pages),
        f"export wrote {exported['exported']} pages, expected {len(pages)}",
    )
    with pikepdf.open(pdf) as document:
        pdf_pages = len(document.pages)
    expect(pdf_pages == len(pages), f"{pdf} has {pdf_pages} pages, not {len(pages)}")


def main(argv: list[str] | None = None) -> int:
    """Run the smoke test from the command line.

    Args:
        argv (list[str] | None): Command line without the program name.

    Returns:
        int: 0 when every check passed, 1 otherwise.
    """
    args = parse_args(argv)
    if not args.cli.is_file():
        sys.stderr.write(f"smoke test FAILED: no executable at {args.cli}\n")
        return 1
    with tempfile.TemporaryDirectory(prefix="scantailor-smoke-") as scratch:
        try:
            # Not resolve(): an AppImage picks the CLI by the symlink name it
            # was started through.
            smoke(args.cli.absolute(), args.workers, Path(scratch))
        except (SmokeFailure, subprocess.TimeoutExpired) as error:
            sys.stderr.write(f"smoke test FAILED: {error}\n")
            return 1
    sys.stderr.write(f"smoke test passed: {len(SMOKE_PAGES)} pages end to end\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
