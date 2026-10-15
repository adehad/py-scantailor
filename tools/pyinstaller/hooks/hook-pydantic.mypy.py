"""pydantic.mypy is pydantic's mypy plugin, never loaded by the running app.

Its tomli fallback would otherwise pull in setuptools' vendored copy.
"""

excludedimports = ["mypy", "tomli"]
