# ODC examples

## Convert to Zarr and Index (Recommended)
   1. Convert to Zarr (see [`zarrify`](zarrify.md) for usage instructions)
      ```
      zarrify --outpath <zarr_output_dir> --chunk x:2000 --chunk y:2000 <path_to_ls5_scenes>
      ```
   1. Generate `agdc_metadata` file (see [Zarr prepare scripts](zarr_prepare_scripts.md))
      1. Option 1: EO LS5 example
         ```
         python examples/prepare_zarr_ls5.py <zarr_output_dir>
         ```
      1. Option 2: EO3 LS8 example (Recommended)
         ```
         python examples/eo3/eo3prepare_usgs_espa_ls8c1_l2_zarr.py -p docs/config_samples/dataset_types/usgs_espa_ls8c1_sr_zarr.yaml <zarr_output_dir>
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
      1. Option 1: Index scenes on Disk (Recommended)
         ```
         datacube dataset add <zarr_output_dir>
         ```
      1. Option 2: Index scenes on S3 using [odc-tools](https://github.com/opendatacube/odc-tools) (Recommended)
         ```
         s3-find --skip-check 's3://<bucket>/<path_to_scenes>/*/agdc-metadata.yaml' | s3-to-tar | dc-index-from-tar --ignore-lineage
         ```
      1. Option 3: Index scenes on S3 using [index_from_s3_bucket.py](https://raw.githubusercontent.com/opendatacube/datacube-dataset-config/master/scripts/index_from_s3_bucket.py)
         ```
         python index_from_s3_bucket.py <bucket> -p <prefix> --suffix="agdc-metadata.yaml"
         ```

## Index and Ingest
   1. Generate `agdc_metadata` file
      See: [Product definitions and prepare scripts](https://github.com/opendatacube/datacube-dataset-config)
      ```
      python galsprepare.py <path_to_ls5_scenes>/*
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
      datacube dataset add <path_to_ls5_scenes>/*
      ```
   1. Ingest scenes to Zarr format
      ```
      datacube -v ingest -c ls5_nbar_albers_zarr.yaml
      ```
      You can specify `--executor multiproc <num_processes>` to enable multi-processing.
      ```
      datacube -v ingest -c ls5_nbar_albers.yaml --executor multiproc <num_processes>
      ```