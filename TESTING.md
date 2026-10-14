# Testing

The suite is pytest with pytest-qt and runs in parallel through pytest-xdist.

```bash
pdm run pytest                               # default run: skips tests marked slow
pdm run pytest -m ""                         # full run, including slow tests; use before merging
pdm run pytest -n0 tests/file.py::test_name  # one test, serially, for debugging
pdm run cov                                  # coverage report in dist/cov/
```

Qt tests need no display: CI sets `QT_QPA_PLATFORM=offscreen`, and the same works locally.

## Markers

| Marker            | Meaning                                                                      |
| ----------------- | ---------------------------------------------------------------------------- |
| `slow`            | Takes more than about 2 s. Excluded from the default run.                    |
| `docs_screenshot` | Drives the window to save the manual's PNGs. Run with `-m docs_screenshot`. |

## Layout

`tests/` mirrors `src/scantailor/`: `tests/filters/deskew/` tests `scantailor.filters.deskew`, and so on. `tests/ui/` covers the windows and dialogs, `tests/cli/` the command line and `tests/docs/` the manual's screenshots. Shared test data lives in `tests/fixtures/`.

`packages/jbig2-numba/` has its own suite under `packages/jbig2-numba/tests/`.
