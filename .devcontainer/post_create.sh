#!/bin/bash

git config --global --add safe.directory "$(pwd)"

# Install pipx
python -m pip install pipx
pipx ensurepath

# Allow path changes
. ~/.bashrc

# Install pdm and pre-commit
pipx install pre-commit
pipx install pdm

pdm install

# The compiled .qm files are gitignored, so a fresh container has none until this runs.
pdm run translations

RED='\033[0;31m'
NC='\033[0m' # No Color
echo -e "${RED}You may need to 'Reload Window' in VSCode for it to pick up the updated interpreter${NC}"
