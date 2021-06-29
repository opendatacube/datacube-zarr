# Open Data Cube Zarr Driver

## Overview

This provides a Zarr [Data Read Plug-in](https://datacube-core.readthedocs.io/en/latest/architecture/driver.html) driver.
for the [Open Data Cube (ODC) project](https://github.com/opendatacube/datacube-core/).

For ODC documentation, please see [ODC documentation](http://datacube-core.readthedocs.io/en/latest/)

See [https://zarr.readthedocs.io/](https://zarr.readthedocs.io/) for Zarr storage specification and Python library.

## User Guides
- [zarrify command line tool](/docs/zarrify.md)
- [OpenDataCube Index](/docs/odc_examples.md#convert-to-zarr-and-index)
- [ZarrIO usage](/docs/zarr_io.md)


## Drivers provided

- Zarr 2D/3D driver
  - supports (s3, file) protocols

## Requirements

### System:
- [ODC 1.8.4+](https://github.com/opendatacube/datacube-core)
- PostgreSQL 9.5+
- Python 3.8+

### Optional:
- [EO Datasets](https://github.com/GeoscienceAustralia/eo-datasets)
  - Used in prepare scripts for generating EO3 compliant metadata.
    ([Documentation](https://github.com/GeoscienceAustralia/eo-datasets/blob/eodatasets3/docs/index.rst))
- [odc-tools](https://github.com/opendatacube/odc-tools)
  - Used for indexing datasets on S3
- [index_from_s3_bucket](https://raw.githubusercontent.com/opendatacube/datacube-dataset-config/master/scripts/index_from_s3_bucket.py)
  - Used for indexing datasets on S3.
    ([Documentation](https://datacube-core.readthedocs.io/en/latest/ops/indexing.html#download-indexing-scripts))


## Developer setup

1. Install ODC with python=3.8 (see [ODC developer setup](https://github.com/opendatacube/datacube-core#developer-setup))
1. Clone:
   ```
   git clone https://csiro-easi@dev.azure.com/csiro-easi/easi-hub-public/_git/datacube-zarr
   ```
1. Activate the conda environment you created when installing ODC
   ```
   conda activate odc
   ```
1. Install new drivers from this repository.
   ```
   cd datacube-drivers
   pip install --upgrade -e  ".[test]"
   ```
1. Run (flake8, isort, mypy, black) + unit tests
   ```
   ./check-code.sh
   ```
1. **(or)** Run all tests, including integration tests.
   ```
   ./check-code.sh integration_tests
   ```
   - Assumes a password-less Postgres database running on localhost called `agdcintegration`
   - Otherwise copy `integration_tests/agdcintegration.conf` to `~/.datacube_integration.conf` and edit to customise.
 1. **(or)** Run all tests, including integration tests with docker
    ```
    docker build -t datacube-zarr-test -f docker/Dockerfile .
    docker run --rm datacube-zarr-test ./check-code.sh integration_tests/
    ```
    - This includes a database server pre-configured for running integration tests.