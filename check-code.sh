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

flake8 .

# static type checker
mypy zarr_io

# Check imports are sorted
isort --check-only --recursive .

black --skip-string-normalization --line-length 90 --check .

# Run tests, taking coverage.
# Users can specify extra folders as arguments.
pytest -r a --cov-report html:coverage/htmlcov --cov-report xml:coverage/cov.xml \
  --cov zarr_io --doctest-ignore-import-errors --durations=5 zarr_io tests $@

set +x

# Optinally validate example yaml docs.
if which yamllint;
then
    set -x
    yamllint $(find . \( -iname '*.yaml' -o -iname '*.yml' \) )
    set +x
fi
