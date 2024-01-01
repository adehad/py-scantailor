#!/usr/bin/env bash
pipx install pre-commit
pipx install ruff
pip install pdm

pdm install
