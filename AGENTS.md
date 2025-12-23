# AGENTS.md - AI Agent Instructions

## Essential Commands

```bash
pdm run pytest                            # Run all tests
pdm run pytest tests/file.py::test_name   # Run specific test
pdm run cov                               # Run tests with coverage
pdm run typecheck                         # Run type checking
pre-commit run --all-files                # Run all linters/formatters
pdm run docs                              # Build documentation
```

**Formatting**: Do NOT manually fix formatting/linting errors. Write code first, then run `pre-commit run --all-files` at the end to auto-fix.

## Project Context

**Application**: ScanTailor Advanced - document scanning post-processor
**Migration**: C++ → Python
**Status**: In progress (UI shell complete, backend not started)

## Source Code Locations

### C++ Source (Reference Only)

```text
src/app/           - UI dialogs (37 files)
src/core/          - Core engine (205 files)
src/core/filters/  - 6 filter stages (240 files)
src/imageproc/     - Image algorithms (102 files)
src/math/          - Math utilities (24 files)
src/dewarping/     - Page dewarping (22 files)
src/foundation/    - Utilities (43 files)
```

### Python Source (Active Development)

```text
src/scantailor/           - Main package
src/scantailor/app/ui/    - UI implementations
src/scantailor/resources/ - Qt resources
src/scantailor/translations/
tests/
```

## Migration Philosophy

**Do NOT blindly replicate C++ patterns.** The C++ code is reference material, not a template.
- Critically evaluate each C++ design - many are workarounds for C++ limitations
- Use Pythonic idioms (iterators, context managers, dataclasses, etc.)
- Simplify where possible - Python has richer built-ins
- Question the need for each class/abstraction
- Bundle related changes in single commits: new Python code + tests + removal of corresponding C++ code

## Code Rules

### MUST DO
- `from __future__ import annotations` in every file
- Type hints on all functions
- Google-style docstrings with type info (Args must include types, not just param names)
- Pydantic for data models
- JSON for persistence (NOT XML)
- loguru for logging
- Tests for every module
- Self-describing code that reads like English
- Preserve C++ filenames where sensible, use PEP8 variable naming
- Comments explain WHY, not what
- If a comment breaks up a function, extract to a named function instead

### MUST NOT
- Wrapper classes around OpenCV/NumPy (use directly)
- XML for any data storage
- `print()` statements
- Bare `except:` clauses
- Mutable default arguments
- `from x import *`
- Keep deprecated/dead code (rely on git for rollback)

## Package Mappings

| C++           | Python                                       |
| ------------- | -------------------------------------------- |
| BinaryImage   | `np.ndarray` (uint8/bool)                    |
| GrayImage     | `np.ndarray` (uint8)                         |
| LinearSolver  | `np.linalg.solve()`                          |
| Morphology    | `cv2.morphologyEx()`                         |
| GaussBlur     | `cv2.GaussianBlur()`                         |
| Binarize      | `cv2.threshold()` / `skimage.filters`        |
| Transform     | `cv2.warpAffine()` / `cv2.warpPerspective()` |
| XmlMarshaller | Pydantic + JSON                              |

## Custom Code Required

These have no library equivalent:
- X-splines (`src/math/XSpline.cpp`)
- Cylindrical dewarping (`src/dewarping/`)
- Text line tracing
- Mokji binarization

## Testing Requirements

- Verbose, descriptive tests
- Cover: normal, edge cases, errors, round-trips
- Test data in `tests/fixtures/`
- pytest + pytest-qt

## File References

For task tracking: `plan/MIGRATION_CHECKLIST.md`
For detailed strategy: `plan/MIGRATION_PLAN.md`

## Commit Style

Brief messages grouping related changes. For migration work, one commit should include:
- New Python code + tests
- Removal of corresponding C++ code (if applicable)

Examples:
- `Migrate binarization algorithms`
- `Port page split filter`
- `Add project JSON schema`

## Code Review Focus

CI/pre-commit already enforces: formatting, imports, type hint presence, docstring presence.

**Reviewers should focus on what linters can't catch:**

### Architecture & Design
- Is this the right abstraction? Question over-engineering
- Is this a C++ pattern that should be more Pythonic?
- Filter classes must be stateless; Settings classes must be thread-safe
- Are we using Python stdlib where C++ had manual implementations?

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
