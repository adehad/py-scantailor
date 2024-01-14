#!/bin/bash

 git config --global --add safe.directory /workspaces/py-scantailor-advanced

# Install pdm and pre-commit
pipx install pre-commit
pipx install pdm

pdm install
