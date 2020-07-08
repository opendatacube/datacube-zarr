#!/usr/bin/env bash
# Convenience script for running Travis-like checks.

set -eu
set -x

flake8 .

# static type checker
mypy datacube_zarr

black --skip-string-normalization --line-length 90 --check .

# Run tests, taking coverage.
# Users can specify extra folders as arguments.
pytest -r a --cov-report term-missing --cov-report html:coverage/htmlcov \
  --cov-report xml:coverage/cov.xml --cov datacube_zarr --doctest-ignore-import-errors \
  --durations=5 datacube_zarr tests $@
