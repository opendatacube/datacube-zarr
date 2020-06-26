#!/usr/bin/env bash
# Convenience script for running Travis-like checks.

set -eu
set -x

if [ "${1:-}" == "--with-docker" ]; then
    shift
    exec docker run -ti \
         -v $(pwd):/src/datacube-core \
         opendatacube/datacube-tests:latest \
         $0 $@
fi

pycodestyle zarr_io tests integration_tests --max-line-length 120

pylint -j 2 --reports no zarr_io utils

# static type checker
mypy zarr_io tools

# Check imports are sorted
isort --check-only --recursive .

# Run tests, taking coverage.
# Users can specify extra folders as arguments.
ls -al
pytest -r a --cov-report html:coverage/htmlcov --cov-report xml:coverage/cov.xml \
  --cov zarr_io --doctest-ignore-import-errors --durations=5 zarr_io tests/test_convert.py

# pytest -r a --cov-report html:coverage/htmlcov --cov-report xml:coverage/cov.xml \
#   --cov zarr_io --doctest-ignore-import-errors --durations=5 zarr_io tests $@

set +x

# Optinally validate example yaml docs.
if which yamllint;
then
    set -x
    yamllint $(find . \( -iname '*.yaml' -o -iname '*.yml' \) )
    set +x
fi
