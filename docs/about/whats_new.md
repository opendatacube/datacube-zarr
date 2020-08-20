# What's New
------------


## v0.1.1 (xx August 2020)
--------------------------

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