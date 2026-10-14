"""Frozen entry point for scantailor-cli."""

import multiprocessing

from scantailor.cli.main import main

if __name__ == "__main__":
    # "run" spawns workers by relaunching this executable; freeze_support turns
    # that relaunch into a worker instead of a second command.
    multiprocessing.freeze_support()
    raise SystemExit(main())
