#! /usr/bin/env bash

# Example commands to:
# 1) Prepare and index LS8 SR geotif
# 2) Convert geotifs to zarr
# 3) Prepare and index LS8 SR zarr
# 4) Compare with original geotiff data

set -x

# create tmp dir
TMP_DIR=/root/data/tmp/eo3_test
[ -d $TMP_DIR ] && rm -r $TMP_DIR
mkdir -p $TMP_DIR

# Clean db
python3 /usr/local/bin/db_initialiser.py delete
python3 /usr/local/bin/db_initialiser.py create
datacube system init
datacube system check

# Repos
BASEDIR=/root/okteto
DATACUBE_DIR=$BASEDIR/datacube-core
DATACUBE_DRIVER_DIR=$BASEDIR/datacube-zarr
EO3_ASSEMBLE_DIR=$BASEDIR/eo3-assemble
DATA_PIPELINE_DIR=$BASEDIR/data-pipeline

# Geotiff test data
LS8_SR_TEST_DATA=$DATACUBE_DRIVER_DIR/tests/data/espa/ls8_sr
cp -r $LS8_SR_TEST_DATA $TMP_DIR
LS8_SR_TEST_DATA_TMP=$TMP_DIR/$(basename $LS8_SR_TEST_DATA)
LS8_SR_GTIF_PRODUCT=$DATACUBE_DRIVER_DIR/docs/config_samples/dataset_types/usgs_espa_ls8c1_sr.yaml

# Add/index the original LS5 GeoTiff product
datacube product add $LS8_SR_GTIF_PRODUCT
python $EO3_ASSEMBLE_DIR/eo3prepare_usgs_espa_ls8c1_l2.py -p $LS8_SR_GTIF_PRODUCT $LS8_SR_TEST_DATA_TMP
datacube dataset add $LS8_SR_TEST_DATA_TMP/odc-metadata.yaml

# Convert GeoTiff dataset to zarr
zarrify --outpath $TMP_DIR/zarr --chunk x:500 --chunk y:500 $LS8_SR_TEST_DATA_TMP
LS8_SR_TEST_DATA_ZARR=$TMP_DIR/zarr/$(basename $LS8_SR_TEST_DATA_TMP)
tree $LS8_SR_TEST_DATA_ZARR

# Add/index the LS5 zarr product
LS8_SR_ZARR_PRODUCT=$DATACUBE_DRIVER_DIR/docs/config_samples/dataset_types/usgs_espa_ls8c1_sr_zarr.yaml
datacube product add $LS8_SR_ZARR_PRODUCT
python $DATACUBE_DRIVER_DIR/examples/eo3prepare_usgs_espa_ls8c1_l2_zarr.py \
    -p $LS8_SR_ZARR_PRODUCT $LS8_SR_TEST_DATA_ZARR
cat $LS8_SR_TEST_DATA_ZARR/odc-metadata.yaml
datacube dataset add $LS8_SR_TEST_DATA_ZARR/odc-metadata.yaml

# Load both datasets and compare
$DATACUBE_DRIVER_DIR/integration_tests/indexing/load_and_compare_zarr_ls8_sr.py