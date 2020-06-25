# ODC examples

## Convert to Zarr and Index (Recommended)
   1. Convert to Zarr (`zarrify.py --help` for usage instructions)
      ```
      zarrify --outpath <zarr output dir> --chunk x:2000 --chunk y:2000 <path to ls5 scenes>
      ```
   1. Generate `agdc_metadata` file
      ```
      python examples/prepare_zarr_ls5.py <zarr output dir>
      ```
   1. Initialise ODC DB
      ```
      datacube -v system init
      ```
   1. Adding product definition
      ```
      datacube product add docs/config_samples/dataset_types/ls5_scenes_zarr.yaml
      ```
   1. Index scenes
      1. Option 1: Index scenes on Disk
         ```
         datacube dataset add <zarr output dir>
         ```
      1. Index scenes on S3 using [odc-tools](https://github.com/opendatacube/odc-tools)
         ```
         s3-find --skip-check 's3://<bucket>/<path to scenes>/*/agdc-metadata.yaml' | s3-to-tar | dc-index-from-tar --ignore-lineage
         ```

## Index and Ingest
   1. Generate `agdc_metadata` file
      See: [Product definitions and prepare scripts](https://github.com/opendatacube/datacube-dataset-config)
      ```
      python galsprepare.py <path to ls5 scenes>/*
      ```
   1. Initialise ODC DB
      ```
      datacube -v system init
      ```
   1. Adding product definition
      See: [Product definitions and prepare scripts](https://github.com/opendatacube/datacube-dataset-config)
      ```
      datacube product add docs/config_samples/dataset_types/ls5_scenes.yaml
      ```
   1. Index scenes
      ```
      datacube dataset add <path to ls5 scenes>/*
      ```
   1. Ingest scenes to Zarr format
      ```
      datacube -v ingest -c ls5_nbar_albers_zarr.yaml
      ```
      You can specify `--executor multiproc <num_processes>` to enable multi-processing.
      ```
      datacube -v ingest -c ls5_nbar_albers.yaml --executor multiproc <num_processes>
      ```