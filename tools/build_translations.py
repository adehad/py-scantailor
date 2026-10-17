"""Compile the bundled .ts translation sources into the .qm files the app loads.

The .qm files are build output and stay gitignored. The release build runs this
before PyInstaller; a source checkout does not need it, because the application
compiles missing catalogues into its cache on first use.
"""

import argparse
import pathlib

from loguru import logger

from scantailor.translations import TRANSLATIONS_DIR
from scantailor.translations.catalogues import compile_all, find_lrelease


def main(argv: list[str] | None = None) -> int:
    """Compile the translations from the command line.

    Args:
        argv (list[str] | None): Arguments without the program name; None uses
            sys.argv.

    Returns:
        int: Process exit code, 0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-dir", type=pathlib.Path, default=TRANSLATIONS_DIR)
    parser.add_argument("--out-dir", type=pathlib.Path, default=TRANSLATIONS_DIR)
    args = parser.parse_args(argv)
    written = compile_all(args.source_dir, args.out_dir, find_lrelease())
    logger.info(f"Wrote {len(written)} translation files to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
