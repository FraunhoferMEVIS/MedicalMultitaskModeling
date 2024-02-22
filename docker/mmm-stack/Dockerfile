# Kaniko does not have caches
# Using incremental images base->stack->user-image is something like a cache
# This is the stack image building upon base
ARG PKGVERSION=latest
FROM hub.cc-asp.fraunhofer.de/medicalmultitaskmodeling/mmm-base:$PKGVERSION

# Install python dependencies from pyproject.toml
COPY . /installation-medicalmultitaskmodeling
RUN poetry config virtualenvs.create false
RUN poetry install --no-root --without=dev --no-interaction --no-ansi --directory=/installation-medicalmultitaskmodeling
RUN rm -rf /installation-medicalmultitaskmodeling
