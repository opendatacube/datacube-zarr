#! /usr/bin/env bash

# Example commands to:
# 1) Convert an LS5 GeoTiff dataset to zarr
# 2) Run the prepare script to generate `agdc-metadata.xml`
# 3) Index the data into ODC
# 4) Compare with original geotiff data

set -x

# Clean db
python3 /usr/local/bin/db_initialiser.py delete
python3 /usr/local/bin/db_initialiser.py create
datacube system init
datacube system check

# Repos
BASEDIR=/root/okteto
DATACUBE_DIR=$BASEDIR/datacube-core
DATACUBE_DRIVER_DIR=$BASEDIR/datacube-zarr

# Geotiff test data
DATACUBE_TEST_DIR=$DATACUBE_DIR/tests/data/lbg
LBG_NBAR="LS5_TM_NBAR_P54_GANBAR01-002_090_084_19920323"
LBG_PQ="LS5_TM_PQ_P55_GAPQ01-002_090_084_19920323"

# Index the original LS5 GeoTiff product
datacube product add $DATACUBE_DIR/docs/config_samples/dataset_types/ls5_scenes.yaml
datacube dataset add $DATACUBE_TEST_DIR/$LBG_NBAR
datacube dataset add $DATACUBE_TEST_DIR/$LBG_PQ

# Convert GeoTiff dataset to zarr
LOCAL_DATA_DIR=/root/data/zarr/index_test
[ -d $LOCAL_DATA_DIR ] && rm -r $LOCAL_DATA_DIR
mkdir -p $LOCAL_DATA_DIR

$DATACUBE_DRIVER_DIR/tools/zarrify.py \
    --outpath $LOCAL_DATA_DIR \
    --chunk x:500 --chunk y:500 \
    $DATACUBE_TEST_DIR

# Convert GeoTiff dataset to zarr inplace
LOCAL_DATA_DIR2=/root/data/zarr/index_test_inplace
[ -d $LOCAL_DATA_DIR2 ] && rm -r $LOCAL_DATA_DIR2
cp -r $DATACUBE_TEST_DIR $LOCAL_DATA_DIR2

$DATACUBE_DRIVER_DIR/tools/zarrify.py \
    --inplace \
    --chunk x:500 --chunk y:500 \
    $LOCAL_DATA_DIR2


# Prepare zarr dataset metadata
ZARR_DATASET_DIR=$LOCAL_DATA_DIR/$(basename $DATACUBE_TEST_DIR)
$DATACUBE_DRIVER_DIR/examples/prepare_zarr_ls5.py $ZARR_DATASET_DIR/$LBG_NBAR
cat $ZARR_DATASET_DIR/$LBG_NBAR/agdc-metadata.yaml
tree $ZARR_DATASET_DIR

# Add to datacube
datacube product add $DATACUBE_DRIVER_DIR/docs/config_samples/dataset_types/ls5_scenes_zarr.yaml
datacube dataset add $ZARR_DATASET_DIR/$LBG_NBAR
datacube dataset add $ZARR_DATASET_DIR/$LBG_PQ

# Load both datasets and compare
$DATACUBE_DRIVER_DIR/integration_tests/indexing/load_and_compare_zarr.py