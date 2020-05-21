# Open Data Cube Drivers

## Overview

This provides additional drivers for the Open Data Cube (ODC) project.

For ODC documentation and repository, please see [ODC documentation](http://datacube-core.readthedocs.io/en/latest/) and [ODC repository](https://github.com/opendatacube/datacube-core/)

## Drivers provided

- Zarr 2D s3 storage driver
- Zarr 2D file storage driver

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
python setup.py install
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

## Convert to Zarr and Index (Recommended)
1. Convert to Zarr
```
python utils/zarrify.py --outpath <zarr output dir> --chunk x:2000 --chunk y:2000 <path to ls5 scenes>
```
2.  Generate `agdc_metadata` file
```
python examples/prepare_zarr_ls5.py <zarr output dir>
```
3. Initialise ODC DB
```
datacube -v system init
```
4. Adding product definition
```
datacube product add docs/config_samples/dataset_types/ls5_scenes_zarr.yaml
```
5. Index scenes
```
datacube dataset add <zarr output dir>
```

## Index and Ingest (Not Recommended)
1. Generate `agdc_metadata` file
See: [Product definitions and prepare scripts](https://github.com/opendatacube/datacube-dataset-config)
```
python galsprepare.py <path to ls5 scenes>/*
```
2. Initialise ODC DB
```
datacube -v system init
```
3. Adding product definition
See: [Product definitions and prepare scripts](https://github.com/opendatacube/datacube-dataset-config)
```
datacube product add docs/config_samples/dataset_types/ls5_scenes.yaml
```
4. Index scenes
```
datacube dataset add <path to ls5 scenes>/*
```
5. Ingest scenes to Zarr format
```
datacube -v ingest -c ls5_nbar_albers_zarr.yaml
```
You can specify `--executor multiproc <num_processes>` to enable multi-processing.
```
datacube -v ingest -c ls5_nbar_albers.yaml --executor multiproc <num_processes>
```