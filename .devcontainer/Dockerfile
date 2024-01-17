# See here for image contents: https://github.com/devcontainers/images/blob/main/src/base-ubuntu/.devcontainer/Dockerfile

# https://hub.docker.com/_/python
ARG VARIANT="bookworm"
ARG PYTHON="3.12"
FROM python:${PYTHON}-${VARIANT}

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends  \
    qt6-base-dev \
    qt6-tools-dev \
    qt6-tools-dev-tools \
    qt6-wayland \
    qt6-l10n-tools \
    qtchooser

RUN qtchooser -install qt6 $(which qmake6)

ENV QT_SELECT=qt6
