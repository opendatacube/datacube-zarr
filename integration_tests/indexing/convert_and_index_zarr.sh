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

BASEDIR=/home/odc/okteto

# Repos
DATACUBE_DIR=$BASEDIR/datacube-core
DATACUBE_DRIVER_DIR=$BASEDIR/datacube-drivers

# Clean local data dir to store zarr
LOCAL_DATA_DIR=$BASEDIR/data/zarr/index_test
[ -d $LOCAL_DATA_DIR ] && rm -r $LOCAL_DATA_DIR
mkdir -p $LOCAL_DATA_DIR

# Geotiff test data
DATACUBE_TEST_DIR=$DATACUBE_DIR/tests/data/lbg
DATASET_NAME="LS5_TM_NBAR_P54_GANBAR01-002_090_084_19920323"

# Index the original LS5 GeoTiff product
datacube product add $DATACUBE_DIR/docs/config_samples/dataset_types/ls5_scenes.yaml
datacube dataset add $DATACUBE_TEST_DIR/$DATASET_NAME/agdc-metadata.yaml

# Convert GeoTiff dataset to zarr
$DATACUBE_DRIVER_DIR/utils/convert.py \
    --outdir $LOCAL_DATA_DIR \
    --chunks x:500 --chunks y:500 \
    $DATACUBE_TEST_DIR

# Prepare zarr dataset metadata
$DATACUBE_DRIVER_DIR/examples/prepare_zarr_ls5.py $LOCAL_DATA_DIR/$DATASET_NAME
tree $LOCAL_DATA_DIR/$DATASET_NAME

# Add to datacube
datacube product add $DATACUBE_DRIVER_DIR/examples/ls5_scenes_zarr.yaml
datacube dataset add $LOCAL_DATA_DIR/$DATASET_NAME

# Load both datasets and compare
$DATACUBE_DRIVER_DIR/integration_tests/indexing/load_and_compare_zarr.py