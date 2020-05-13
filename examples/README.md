# Indexing a zarr dataset

In order to index a Zarr dataset, ODC requires:
1) a product description file, and
2) a metadata file describing the dataset

In the example below we first convert a Landsat 5 GeoTiff dataset to Zarr and then run a prepare script to generate the metadata file.

The product description (`./ls5_scenes_zarr.yaml`) for the zarr formatted LS5 data is identical to the GeoTiff product description with the "format" fields changed.

Set up some directies for repository locations and a place to store the generated example dataset:
```
BASEDIR=/home/odc/okteto
DATACUBE_DIR=$BASEDIR/datacube-core
DATACUBE_DRIVER_DIR=$BASEDIR/datacube-drivers
LOCAL_DATA_DIR=$BASEDIR/index_zarr_test
mkdir -p $LOCAL_DATA_DIR
```

This example uses the test Landsat 5 dataset used by ODC:
```
DATACUBE_TEST_DIR=$DATACUBE_DIR/tests/data/lbg
DATASET_NAME="LS5_TM_NBAR_P54_GANBAR01-002_090_084_19920323"
```

Use `./utils/convert.py` tool to convert test LS5 GeoTiffs to Zarr:
```
$DATACUBE_DRIVER_DIR/utils/convert.py \
    --outdir $LOCAL_DATA_DIR \
    --chunks x:500 --chunks y:500 \
    $DATACUBE_TEST_DIR
```

Run the example prepare script to create the `agdc-metadata.xml` file required to index the dataset into ODC:
```
$DATACUBE_DRIVER_DIR/examples/prepare_zarr_ls5.py $LOCAL_DATA_DIR/$DATASET_NAME
```

The zarr directory now looks like this:
```
├── LS5_TM_NBAR_P54_GANBAR01-002_090_084_19920323
│   └── LS5_TM_NBAR_P54_GANBAR01-002_090_084_19920323
│       ├── 1
│       │   └── 0.0.0
│       ├── 2
│       │   └── 0.0.0
│       ├── 3
│       │   └── 0.0.0
│       ├── 4
│       │   └── 0.0.0
│       ├── 5
│       │   └── 0.0.0
│       ├── 7
│       │   └── 0.0.0
│       ├── time
│       │   └── 0
│       ├── x
│       │   └── 0
│       └── y
│           └── 0
├── agdc-metadata.yaml
└── metadata.xml
```

Note: The `convert.py` and `prepare_zarr_ls5.py` tools also work with S3 uris to load and/or store data/metadata on S3.

To index the Zarr dataset add the product description and dataset to ODC:
```
datacube -v system init
datacube product add $DATACUBE_DRIVER_DIR/examples/ls5_scenes_zarr.yaml
datacube dataset add $LOCAL_DATA_DIR/$DATASET_NAME
```

We can also add the original GeoTiff LS5 product description and dataset to ODC:
```
datacube product add $DATACUBE_DIR/docs/config_samples/dataset_types/ls5_scenes.yaml
datacube dataset add $DATACUBE_TEST_DIR/$DATASET_NAME/agdc-metadata.yaml
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