# Open Data Cube Drivers

## Overview

This provides additional drivers for the Open Data Cube (ODC) project.

For ODC documentation and repository, please see [ODC documentation](http://datacube-core.readthedocs.io/en/latest/) and [ODC repository](https://github.com/opendatacube/datacube-core/)

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
- ODC 1.8+
- PostgreSQL 9.5+
- Python 3.6+

## Developer setup

1. Install ODC (see [ODC developer setup](https://github.com/opendatacube/datacube-core#developer-setup))
2. Clone:
```
git clone https://csiro-easi@dev.azure.com/csiro-easi/easi-hub/_git/datacube-drivers
```
3. Activate the conda environment you created when installing ODC
```
conda activate odc
```
4. Install new drivers from this repository.
```
cd datacube-drivers
pip install --upgrade -e .[test,tools]
```
5. Run unit tests + PyLint
```
./check-code.sh
```
6. **(or)** Run all tests, including integration tests.
```
./check-code.sh integration_tests
```
- Assumes a password-less Postgres database running on localhost called `agdcintegration`
- Otherwise copy ``integration_tests/agdcintegration.conf`` to `~/.datacube_integration.conf` and edit to customise.

Alternatively one can use ``opendatacube/datacube-tests`` docker image to run tests.
This docker includes database server pre-configured for running integration tests.
Add ``--with-docker`` command line option as a first argument to ``./check-code.sh`` script.
```
./check-code.sh --with-docker integration_tests
```
