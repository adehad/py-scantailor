# See here for image contents: https://github.com/devcontainers/images/blob/main/src/base-ubuntu/.devcontainer/Dockerfile

# [Choice] Ubuntu version: https://github.com/devcontainers/images/tree/main/src/base-ubuntu
ARG VARIANT="ubuntu-22.04"
FROM mcr.microsoft.com/vscode/devcontainers/base:${VARIANT}

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends  \
    qt6-base-dev \
    qt6-tools-dev \
    qt6-tools-dev-tools \
    qt6-wayland \
    qt6-l10n-tools

RUN qtchooser -install qt6 $(which qmake6)

ENV QT_SELECT=qt6
