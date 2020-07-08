# Open Data Cube Zarr Driver

## Overview

This provides a Zarr [Plug-in driver](https://datacube-core.readthedocs.io/en/latest/architecture/driver.html)
for the [Open Data Cube (ODC) project](https://github.com/opendatacube/datacube-core/).

For ODC documentation, please see [ODC documentation](http://datacube-core.readthedocs.io/en/latest/)

## User Guides
- Index and Ingestion
  - [Convert to Zarr and Index (Recommended)](/docs/odc_examples.md#convert-to-zarr-and-index-(recommended))
  - [Index and Ingest](/docs/odc_examples.md#index-and-ingest)
- [ZarrIO usage](/docs/zarr_io.md)


## Drivers provided

- Zarr 2D driver
  - supports (s3, file) protocols

See [https://zarr.readthedocs.io/](https://zarr.readthedocs.io/) for Zarr storage specification and Python library.

## Requirements

### System:
- [ODC 1.8+](https://github.com/opendatacube/datacube-core)
- PostgreSQL 9.5+
- Python 3.6+

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

1. Install ODC (see [ODC developer setup](https://github.com/opendatacube/datacube-core#developer-setup))
1. Clone:
   ```
   git clone https://csiro-easi@dev.azure.com/csiro-easi/easi-hub/_git/datacube-drivers
   ```
1. Activate the conda environment you created when installing ODC
   ```
   conda activate odc
   ```
1. Install new drivers from this repository.
   ```
   cd datacube-drivers
   pip install --upgrade -e .[test,tools]
   ```
1. Run (flake8-isort, mypy, black) + unit tests
   ```
   ./check-code.sh
   ```
1. **(or)** Run all tests, including integration tests.
   ```
   ./check-code.sh integration_tests
   ```
   - Assumes a password-less Postgres database running on localhost called `agdcintegration`
   - Otherwise copy ``integration_tests/agdcintegration.conf`` to `~/.datacube_integration.conf` and edit to customise.
