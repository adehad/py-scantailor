"""Module entry point, so ``python -m scantailor.cli`` works without installing."""

from scantailor.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
