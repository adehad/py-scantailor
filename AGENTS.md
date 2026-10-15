# AGENTS.md - AI Agent Instructions

## Essential Commands

```bash
pdm run pytest                            # Fast default run: parallel (xdist), skips slow tests
pdm run pytest -m ""                      # Full gate including slow tests (run before merging)
pdm run pytest -n0 tests/file.py::test_name  # Run specific test serially (for debugging)
pdm run cov                               # Run tests with coverage
pdm run typecheck                         # Run type checking
pre-commit run --all-files                # Run all linters/formatters
pdm run docs                              # Build documentation
pdm run exe                               # Build the executables with PyInstaller
```

**Formatting**: Do NOT manually fix formatting/linting errors. Write code first, then run `pre-commit run --all-files` at the end to auto-fix.

## Project Context

**Application**: ScanTailor - document scanning post-processor, written in Python with PySide6
**Origin**: a port of [ScanTailor Advanced](https://github.com/ScanTailor-Advanced/scantailor-advanced) (C++/Qt). The C++ code is not in this repository; read it upstream when you need to compare behaviour.
**Tracking**: work and known defects are GitHub issues on `adehad/py-scantailor`

## Source Code Locations

```text
src/scantailor/              - Main package
src/scantailor/app/ui/       - Qt windows, dialogs, panels and their .ui forms
src/scantailor/core/         - Project model, settings, pipeline
src/scantailor/filters/      - The six processing stages
src/scantailor/imageproc/    - Image algorithms
src/scantailor/dewarping/    - Page dewarping
src/scantailor/math/         - Splines and numeric helpers
src/scantailor/cli/          - Headless command line
src/scantailor/resources/    - Qt resources (icons, fonts)
src/scantailor/translations/ - .ts catalogues
packages/jbig2/              - JBIG2 encoder used by PDF export
tools/                       - Translation, screenshot and PyInstaller helpers
tests/                       - pytest suite, mirroring the package layout
docs/                        - Sphinx user guide and API reference
```

## Design Philosophy

**Do NOT blindly replicate upstream C++ patterns.** Upstream is a behavioural reference, not a template.
- Critically evaluate each upstream design - many are workarounds for C++ limitations
- Use Pythonic idioms (iterators, context managers, dataclasses, etc.)
- Simplify where possible - Python has richer built-ins
- Question the need for each class/abstraction
- Bundle related changes in single commits: code + tests + docs

## Code Rules

### MUST DO
- Type hints on all functions
- Google-style docstrings with type info (Args must include types, not just param names)
- Pydantic for data models
- JSON for persistence (NOT XML)
- loguru for logging
- Tests for every module
- Self-describing code that reads like English
- Keep upstream class names as module names where sensible, use PEP8 variable naming
- Comments explain WHY, not what
- If a comment breaks up a function, extract to a named function instead

### MUST NOT
- Wrapper classes or pass-through functions around OpenCV/NumPy (call cv2/NumPy directly)
- XML for any data storage
- `print()` statements
- Bare `except:` clauses
- Mutable default arguments
- `from x import *`
- Keep deprecated/dead code (rely on git for rollback)
- Add numba, SciPy or scikit-image to the application: it ships NumPy, OpenCV, Pillow and the stdlib only, so there is one OpenBLAS and no compiler in the build

## Library Choices

| Upstream C++  | Python                                       |
| ------------- | -------------------------------------------- |
| BinaryImage   | `np.ndarray` (uint8/bool)                    |
| GrayImage     | `np.ndarray` (uint8)                         |
| LinearSolver  | `np.linalg.solve()`                          |
| Morphology    | `cv2.morphologyEx()`                         |
| GaussBlur     | `cv2.GaussianBlur()`                         |
| Binarize      | `cv2.threshold()`                            |
| Transform     | `cv2.warpAffine()` / `cv2.warpPerspective()` |
| XmlMarshaller | Pydantic + JSON                              |

## Custom Code

These have no library equivalent and live in this repository:
- X-splines (`src/scantailor/math/spline.py`)
- Cylindrical dewarping (`src/scantailor/dewarping/`)
- Text line tracing (`src/scantailor/dewarping/text_line_tracer.py`)
- Mokji binarization (`src/scantailor/imageproc/binarize.py`)

## Testing Requirements

- Verbose, descriptive tests
- Cover: normal, edge cases, errors, round-trips
- Test data in `tests/fixtures/`
- pytest + pytest-qt

## Commit Style

Brief messages grouping related changes: code, tests and docs for one change in one commit.

Examples:
- `Add Mokji binarization`
- `Remember the split line across pages`
- `Add project JSON schema`

## Code Review Focus

CI/pre-commit already enforces: formatting, imports, type hint presence, docstring presence.

**Reviewers should focus on what linters can't catch:**

### Architecture & Design
- Is this the right abstraction? Question over-engineering
- Is this a C++ pattern that should be more Pythonic?
- Filter classes must be stateless; Settings classes must be thread-safe
- Are we using Python stdlib where upstream had manual implementations?

### Thread Safety (Critical for filters/)
- Settings with mutable state must use `threading.Lock`
- Lock should protect minimal critical section only
- Watch for: nested locks (deadlock), lock held during I/O or image processing

### Image Processing (imageproc/)
- Check dtype handling (uint8, float32, bool)
- Avoid mutating input arrays - make copies when needed
- Memory usage for large images
- Document expected input/output dtypes in docstrings

### Quality Checks
- Docstring **content** helpful? (not just present)
- Args descriptions meaningful, not restating parameter names?
- Complex algorithms have WHY comments?
- Magic numbers explained?

### Test Quality
- Test names describe scenario AND expected outcome?
- Edge cases covered (empty, boundary, error)?
- Assertions checking the right thing?
- Avoid: no-assertion tests, exact float comparisons, duplicated tests
