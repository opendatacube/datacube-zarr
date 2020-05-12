# Indexing a zarr dataset

In order to index a Zarr dataset, Datacube requires:
1) a product description file, and
2) a metadata file describing the dataset

In the example below we start with an existing GeoTiff dataset, convert it to Zarr and then add it to Datacube.


Set up some directies for repository locations and a place to store the generated example dataset:
```
BASEDIR=/home/odc/okteto
DATACUBE_DIR=$BASEDIR/datacube-core
DATACUBE_DRIVER_DIR=$BASEDIR/datacube-drivers
LOCAL_DATA_DIR=$BASEDIR/index_zarr_test
mkdir -p $LOCAL_DATA_DIR
```

This example uses the test Landsat 5 dataset used by Datacube:
```
DATACUBE_TEST_DIR=$DATACUBE_DIR/tests/data/lbg
DATASET_NAME="LS5_TM_NBAR_P54_GANBAR01-002_090_084_19920323"
```

First we can add the GeoTiff LS5 product description and dataset to Datacube using the existing product description and dataset metadata files:
```
LS5_PROD_DESC=$DATACUBE_DIR/docs/config_samples/dataset_types/ls5_scenes.yaml
LS5_AGDC_META=$DATACUBE_TEST_DIR/$DATASET_NAME/agdc-metadata.yaml
datacube product add $LS5_PROD_DESC
datacube dataset add $LS5_AGDC_META
```

The `convert_prod_desc_to_zarr.py` script converts an existing product description file to one with the Zarr format specified. We can then add this product to Datacube.
```
$DATACUBE_DRIVER_DIR/examples/convert_prod_desc_to_zarr.py $LS5_PROD_DESC $LOCAL_DATA_DIR
LS5_PROD_DESC_ZARR=$LOCAL_DATA_DIR/ls5_scenes_zarr.yaml
datacube product add $LS5_PROD_DESC_ZARR
```

The `prepare_zarr_from_metadata.py` scripts takes an existing GeoTiff dataset with a Datacube compatible `agdc-metadata.yaml` file and converts it to Zarr format and creates an new metadata file for the Zarr dataset:
```
Usage: prepare_zarr_from_metadata.py [OPTIONS] METADATA OUTPUT_DIR GROUP_NAME
```
We pass in the LS5 GeoTiff dataset `agdc-metadata.yaml` file and specify the dataset name as the Zarr group name:
```
$DATACUBE_DRIVER_DIR/examples/prepare_zarr_from_metadata.py $LS5_AGDC_META $LOCAL_DATA_DIR $DATASET_NAME
```
The Zarr data directoy (`$LOCAL_DATA_DIR`) now looks like this:
```
.
├── LS5_TM_NBAR_P54_GANBAR01-002_090_084_19920323
│   └── LS5_TM_NBAR_P54_GANBAR01-002_090_084_19920323
│       ├── 1
│       │   ├── 0.0.0
│       │   └── 0.1.0
│       ├── 2
│       │   ├── 0.0.0
│       │   └── 0.1.0
│       ├── 3
│       │   ├── 0.0.0
│       │   └── 0.1.0
│       ├── 4
│       │   ├── 0.0.0
│       │   └── 0.1.0
│       ├── 5
│       │   ├── 0.0.0
│       │   └── 0.1.0
│       ├── 7
│       │   ├── 0.0.0
│       │   └── 0.1.0
│       ├── time
│       │   └── 0
│       ├── x
│       │   └── 0
│       └── y
│           └── 0
├── agdc-metadata.yaml
└── ls5_scenes_zarr.yaml
```


We can then index the Zarr dataset directly:
```
datacube dataset add $LOCAL_DATA_DIR
```

Within python we can load a scene from both datasets and check that they are the same:
```
#! /usr/bin/env python3

import datacube

# LS5 NBAR scene params
crs = "EPSG:28355"
res = (25, -25)

# Load data
dc = datacube.Datacube()
data_tiff = dc.load(product='ls5_nbar_scene', output_crs=crs, resolution=res)
data_zarr = dc.load(product='ls5_nbar_scene_zarr', output_crs=crs, resolution=res)

assert data_zarr.equals(data_tiff)
```