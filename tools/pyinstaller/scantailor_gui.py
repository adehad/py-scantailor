"""Frozen entry point for the ScanTailor window."""

import multiprocessing

from scantailor.main import main

if __name__ == "__main__":
    # Batch and PDF export spawn workers by relaunching this executable;
    # freeze_support turns that relaunch into a worker instead of a second window.
    multiprocessing.freeze_support()
    raise SystemExit(main())
