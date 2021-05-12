# What's New
------------




## v0.2.2 (xx Month 2020)
-------------------------

- Update to python3.8
- Update to moto version 2
- Update to s3fs>5 using aiobotocore
- Change default compression back to blosc
- Updated ls8 product definitions with default load parameters.
- Update to 'black>=20.8b1' and 'isort>=5.1.0'.
- Remove deprecated "+init" from "<auth>:<auth_code>" CRS definitions loaded via xarray
- Update `rasterio` dependency to version >=1.1.5
- Suppress expected warnings in unit tests
- CLI script `set_zarr_product_extra_dim.py` for changing a zarr dataset's extra dimensions.
  - Extra dimensions are defined in a product definition yaml under "extra_dimensions".
- Change `time_idx` to `band_idx` in Zarr driver.
- Support for list of nodata vals required for 3D zarrs.
- Add `transform` attribute from rasters to zarr dataset within zarrify.
- Revert to universal `"band1"` 2D zarr array naming (not `"band01"`, `"band001"`, etc.).
- Update `xarray` dependency to version >=0.16.2
- Remove Zarr writer driver.
  - Creating Zarr datasets via Datacube ingest is no longer supported. The `zarrify` CLI
    tool can be used to convert from other formats to Zarr prior to indexing.
- Zarr dimension rename script supports multiple extra dimensions.


## v0.1.1 (02 September 2020)
-----------------------------

- Function to rename/replace a zarr dataset dimension.
- Workaround for pytest fails with moto and s3fs.
- Support for latest version of moto before it is released.
- Support ERS datasets / fix logging / optional progress bar.
- Zarrify: option to preload data and to determine ideal chunk size.
- Update prepare scripts for eo-datasets PR #88
- ci_cd setup for public release.
- Small script to generate [zarrify command line tool](../zarrify.md)
- Rename import from zarr_io to datacube_zarr to match distribution name.
- Documentation for creating zarr prepare scripts.
- Added support for EO3 with sample product definition and prepare script.
- Publish test coverage results in Azure.
- Disable S3FileSystem file list caching to fix a race condition between moto and boto.
  - Only happens during tests which use moto.
- Additional unit tests for zarrify.
- Added integration test to index zarr dataset located on S3.
- Bring ZarrWriterDriver up to date with datacube-core PR #960
- ZarrIO now using #group notation in the URI.
  - Removed ZarrIO relative flag.
- Zarrify CLI tool for converting raster datasets into zarr format.
  - See [zarrify command line tool](../zarrify.md) for more information.
- Added Apache 2.0 license and setuptools-scm for versioning.
- Change Zarr URI format to <protocol>://<zarr_path>#<group>
- Zarrify Zarr conversion tool and sample product definition and prepare script.
- Fix possible race condition in unit tests with boto and moto.
- Integration tests based on ODC integration tests using the zarr driver.
- save_dataarray and save_dataset: new write mode parameter.
- Slice on time when no window is provided in the read driver.
- Fix selecting zarr group when it already exists.
- Fixes a bug where ZarrIO would create a directory if it didnt exist when attempting a read.
- Unit test for Zarr driver and storage modules.
- Merged the two Zarr reader endpoints into one. Implement remaining ODC connection points.


### v0.1.0 (28 May 2020)
------------------------

- Initial version.
