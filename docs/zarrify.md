# zarrify command line tool
The `zarrify` tool converts existing raster datasets to [Zarr format](https://zarr.readthedocs.io/en/stable/spec/v2.html). It is installed as a command line tool with `datacube-zarr`.

See usage below for details.
## Usage
    $ zarrify --help

    Usage:  [OPTIONS] DATASET

      Convert datasets to Zarr format.

      If DATASET argument is a directory all supported datasets found
      recursively within are converted. Otherwise DATASET must point to a
      supported dataset file.

      Paths can be either local files/directories or 's3://' URIs.

      For `--inplace` conversion, original datafiles are deleted and metadata
      files left inplace. Where `--outpath` is specified metadata files are
      copied to the new directory structure unless explicitly `--ignore`-ed.

      Zarr format:

      By default each raster dataset is converted to a zarr dataset with root
      directory `<raster_name>.zarr`.

      E.g., for raster(s) with 2 bands and shape (200, 300), and ommiting the
      `--outpath` option for simplicity:

      $ zarrify raster.tif

          results in `raster.zarr` with the following structure, where each
          band is a separate dataset named "band#" under the root group "/":

                  /
                  ├── band1 (200, 300) float32
                  ├── band2 (200, 300) float32
                  ├── x (300,) float64
                  └── y (200,) float64

      $ zarrify --multi-dim raster.tif

          results in `raster.zarr` with bands collected into a single dataset
          called "array" and "band" number is an additional dimension:

                  /
                  ├── array (2, 200, 300) float32
                  ├── band (2,) int64
                  ├── x (300,) float64
                  └── y (200,) float64

      $ zarrify --merge-datasets-per-dir path/to/rasters/

          for a directory containing N rasters (e.g raster1.tif,...) results
          in `raster.zarr` with a group per image:

                  /
                  ├── raster0
                  │   ├── band1 (200, 300) float32
                  │   ├── band2 (200, 300) float32
                  │   ├── x (300,) float64
                  │   └── y (200,) float64
                  ...
                  └── rasterN
                      ├── band1 (200, 300) float32
                      ├── band2 (200, 300) float32
                      ├── x (300,) float64
                      └── y (200,) float64

          Note: converting existing heirarchical datasets (e.g. NetCDF) will
          result in a similar grouped zarr structure.

      Chunking:

      Default behaviour is to not chunk the zarr dataset. Chunk sizes for each
      dimension may be set with `--chunk <dim>:<size>`. The chunk `<size>` may
      be specified as any one of:

          - the integer -1, for no chunking (i.e. <dim> length) [default]
          - an integer N, for a fixed chunk size
          - the string 'auto', for automatically determined chunksize

      Automatically determined chunk sizes are based on `--chunk-target-mb`, the
      dtype of the data, and `--approx-compression-ratio`. The flag `--auto-
      chunk` can be used as shorthand for setting chunk size to 'auto' on the
      last two dimensions and -1 on all other dimensions.

      Reprojection:

      The output projection can be specified via `--crs` and/or `--resolution`.

      Supported datasets:

      ENVI, GeoTiff, HDF, JPEG2000.

      Note: Only gridded HDF datasets are supported. s3:// paths are not
      supported for HDF4 datasets.

    Options:
      --outpath FILEORS3PATH          Path to save the converted dataset
                                      directory.

      --inplace                       Convert inplace (deletes original data
                                      files).

      --ignore TEXT                   Comma separated list of file patterns to
                                      ignore.

      --crs CRS                       Output CRS (EPSG code or proj4 string).
      --resolution FLOAT...           Ouput resolution '<xres> <yres>'.
      --chunk KEY:VALUE               Zarr chunk option '<dim>:<size>'.
      --chunk-target-mb FLOAT RANGE   Target chunk size (MB) used for 'auto'
                                      chunking.

      --approx-compression-ratio FLOAT RANGE
                                      Compression ratio used for 'auto' chunking.
      --auto-chunk                    Chunk on last two dimensions only.
      --merge-datasets-per-dir        Create single zarr for all datasets in a
                                      directory.

      --multi-dim                     Keep multi-banded tifs as 3-dimensional
                                      arrays.

      --preload-data                  Load dataset into memory before conversion.
      -v, --verbose                   Enables verbose mode.
      --version                       Show the version and exit.
      --help                          Show this message and exit.
