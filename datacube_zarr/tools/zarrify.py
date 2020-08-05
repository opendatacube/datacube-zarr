#! /usr/bin/env python

"""
Command line tool for converting dataset to Zarr format.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import click
from rasterio.crs import CRS
from rasterio.errors import CRSError
from s3path import S3Path

from datacube_zarr._version import version
from datacube_zarr.utils.convert import (
    convert_dir,
    convert_to_zarr,
    get_datasets,
    ignore_file,
)

logger = logging.getLogger("zarrify")
handler = logging.StreamHandler()


class KeyValue(click.ParamType):
    """A click param for any key/value pairs."""

    name = "Key:Value"

    def __init__(
        self,
        key: Callable[[str], Any] = str,
        value: Callable[[str], Any] = str,
        sep: str = ":",
    ):
        self.key_fn = key
        self.value_fn = value
        self.sep = sep

    def convert(
        self, value: str, param: Optional[click.Parameter], ctx: Optional[click.Context]
    ) -> Tuple:
        """Convert key:valueto tuple."""
        k, v = value.split(self.sep, 1)
        return self.key_fn(k), self.value_fn(v)


def _chunk_size_value(value: str) -> Union[int, str]:
    """Validate/cast chunk size."""
    if value == "auto":
        return value
    else:
        try:
            size = int(value)
            if size > 0 or size == -1:
                return size
        except ValueError:
            pass

    raise ValueError(f"Invalid chunk size: {value}")


class FileOrS3Path(click.ParamType):
    """A click param for any file or s3 path."""

    name = "FileOrS3Path"

    def __init__(self, exists: bool = False):
        self.exists = exists

    def convert(
        self, value: str, param: Optional[click.Parameter], ctx: Optional[click.Context]
    ) -> Path:
        """Convert file/s3 path str to pathlib Path."""
        if value.startswith("s3:/"):
            path: Path = S3Path(value[4:])
        else:
            if value.startswith("file://"):
                value = value[7:]
            path = Path(value).resolve()

        if self.exists and not path.exists():
            self.fail(f"{path.as_uri()} does not exist.")

        return path


class ClickCRS(click.ParamType):
    """A Click.ParamType for Coordinate Reference Systems (CRS).

    Converts a CLI parameter into a rasterio CRS object.
    The parameter can be either a string or an integer representing an EPSG code.
    """

    name = "CRS"

    def convert(
        self, value: str, param: Optional[click.Parameter], ctx: Optional[click.Context]
    ) -> CRS:
        """Convert value to rasterio CRS object if valid."""
        try:
            try:
                p = CRS.from_epsg(int(value))
            except (ValueError, CRSError):
                p = CRS.from_string(value)
        except CRSError:
            self.fail(f"{value} is not a valid CRS", param, ctx)

        return p


def _check_path_options(outpath: Optional[Path], inplace: bool) -> None:
    """Some checks on command inputs."""
    if not outpath and not inplace:
        raise click.UsageError("--inplace flag is required if --outpath is not set.")

    if outpath and inplace:
        raise click.UsageError("Can not set both --outpath and --inplace options.")


def _check_chunk_options(
    chunks: Optional[Dict[str, Union[str, int]]],
    auto_chunk: bool,
    chunk_target_mb: Optional[float],
    approx_compression_ratio: Optional[float],
) -> None:
    """Some checks on chunk options."""
    if auto_chunk:
        if chunks:
            raise ValueError("Cannot use both `--auto-chunk` and `--chunk` options.")
    elif (chunks is None or "auto" not in chunks) and (
        chunk_target_mb is not None or approx_compression_ratio is not None
    ):
        logger.warning(
            "No 'auto' chunk sizes specified. Options `--chunk_target_mb` and "
            "`--approx_compression_ratio` will be ignored."
        )


def absolute_ignores(ignore: List[str], abs_path: Path) -> List[str]:
    """Prepend absolute ignore patterns with path."""
    return [str(abs_path / i) if i[0] == "/" else i for i in ignore]


def setup_logging(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    """Setup application logging."""
    if value:
        formatter = logging.Formatter(
            "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
        )
        log_level = logging.DEBUG
    else:
        formatter = logging.Formatter("%(levelname)-8s %(message)s")
        log_level = logging.INFO

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(log_level)

    for logger_name in ("boto3", "botocore", "fsspec", "rasterio", "s3fs", "urllib3"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


@click.command()
@click.argument("dataset", type=FileOrS3Path(exists=True), required=True)
@click.option(
    "--outpath",
    type=FileOrS3Path(),
    required=False,
    help="Path to save the converted dataset directory.",
)
@click.option(
    "--inplace", is_flag=True, help="Convert inplace (deletes original data files)."
)
@click.option(
    "--ignore",
    type=str,
    help="Comma separated list of file patterns to ignore.",
    callback=lambda ctx, param, value: value.split(",") if value else [],
)
@click.option("--crs", type=ClickCRS(), help="Output CRS (EPSG code or proj4 string).")
@click.option(
    "--resolution",
    type=float,
    nargs=2,
    help="Ouput resolution '<xres> <yres>'.",
    callback=lambda ctx, param, value: value if value else None,
)
@click.option(
    "--chunk",
    type=KeyValue(value=_chunk_size_value),
    multiple=True,
    help="Zarr chunk option '<dim>:<size>'.",
)
@click.option(
    "--chunk-target-mb",
    type=click.FloatRange(min=0),
    help="Target chunk size (MB) used for 'auto' chunking.",
)
@click.option(
    "--approx-compression-ratio",
    type=click.FloatRange(min=0),
    help="Compression ratio used for 'auto' chunking.",
)
@click.option(
    "--auto-chunk", is_flag=True, help="Chunk on last two dimensions only.",
)
@click.option(
    "--merge-datasets-per-dir",
    is_flag=True,
    help="Create single zarr for all datasets in a directory.",
)
@click.option(
    "--multi-dim", is_flag=True, help="Keep multi-banded tifs as 3-dimensional arrays."
)
@click.option(
    "--preload-data", is_flag=True, help="Load dataset into memory before conversion."
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    callback=setup_logging,
    expose_value=False,
    is_eager=True,
    help="Enables verbose mode.",
)
@click.version_option(version=version)
def main(
    dataset: Path,
    outpath: Optional[Path],
    inplace: bool,
    crs: Optional[CRS],
    resolution: Optional[Tuple[float, float]],
    chunk: Optional[List[Tuple[str, int]]],
    chunk_target_mb: Optional[float],
    approx_compression_ratio: Optional[float],
    auto_chunk: bool,
    ignore: List[str],
    merge_datasets_per_dir: bool,
    multi_dim: bool,
    preload_data: bool,
) -> None:
    """Convert datasets to Zarr format.

    If DATASET argument is a directory all supported datasets found
    recursively within are converted. Otherwise DATASET must point to
    a supported dataset file.

    Paths can be either local files/directories or 's3://' URIs.

    For `--inplace` conversion, original datafiles are deleted and metadata
    files left inplace. Where `--outpath` is specified metadata files are
    copied to the new directory structure unless explicitly `--ignore`-ed.

    Zarr format:

    By default each raster dataset is converted to a zarr dataset with root
    directory `<raster_name>.zarr`.

    E.g., for raster(s) with 2 bands and shape (200, 300), and ommiting
    the `--outpath` option for simplicity:

    $ zarrify raster.tif

        results in `raster.zarr` with the following structure, where each
        band is a separate dataset named "band#" under the root group "/":

        \b
            /
            ├── band1 (200, 300) float32
            ├── band2 (200, 300) float32
            ├── x (300,) float64
            └── y (200,) float64

    $ zarrify --multi-dim raster.tif

        results in `raster.zarr` with bands collected into a single dataset
        called "array" and "band" number is an additional dimension:

        \b
            /
            ├── array (2, 200, 300) float32
            ├── band (2,) int64
            ├── x (300,) float64
            └── y (200,) float64

    $ zarrify --merge-datasets-per-dir path/to/rasters/

        for a directory containing N rasters (e.g raster1.tif,...) results
        in `raster.zarr` with a group per image:

        \b
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

    Default behaviour is to for no chunking of the zarr dataset. Chunk
    sizes for each dimension may be set with `--chunk <dim>:<size>`.

    The chunk `<size>` may be specified as any one of:
        - The integer -1, for no chunking (i.e. dim length) [default],
        - An integer N, for a fixed chunk size,
        - The string 'auto', for automatically determined chunksize.

    Automatically determined chunk sizes are based on `--chunk-target-mb`,
    the dtype of the data and `--approx-compression-ratio`. The flag
    `--auto-chunk` can be used as shorthand for setting chunk size to
    'auto' on the last two dimensions and -1 on all other dimensions.

    Output projection can be specified via `--crs` and/or `--resolution`.

    Supported datasets: ENVI, GeoTiff, HDF, JPEG2000.

    Note: Only gridded HDF datasets are supported. s3:// paths are not
    supported for HDF4 datasets.
    """
    _check_path_options(outpath, inplace)
    ignore = absolute_ignores(ignore, dataset)
    chunks = dict(chunk) if chunk else None
    _check_chunk_options(chunks, auto_chunk, chunk_target_mb, approx_compression_ratio)

    # kwargs to pass to zarr_io
    zarrgs = {
        "chunks": chunks,
        "target_mb": chunk_target_mb,
        "compression_ratio": approx_compression_ratio,
    }
    zarrgs = {k: v for k, v in zarrgs.items() if v is not None}

    # Recurse into directory an convert supported datasets
    if dataset.is_dir():
        outpath = outpath / dataset.parts[-1] if outpath else None
        convert_dir(
            in_dir=dataset,
            out_dir=outpath,
            ignore=ignore,
            crs=crs,
            resolution=resolution,
            merge_datasets_per_dir=merge_datasets_per_dir,
            multi_dim=multi_dim,
            preload_data=preload_data,
            auto_chunk=auto_chunk,
            **zarrgs,
        )

    # Convert this single supported dataset
    else:
        try:
            fmt, files = next(
                ds for ds in get_datasets(dataset.parent) if ds[1][0] == dataset
            )
            if ignore_file(files[0], ignore):
                logger.warning(f"Ignoring dataset: {dataset}")
            else:
                convert_to_zarr(
                    files=files,
                    out_dir=outpath,
                    crs=crs,
                    resolution=resolution,
                    multi_dim=multi_dim,
                    preload_data=preload_data,
                    auto_chunk=auto_chunk,
                    **zarrgs,
                )
        except StopIteration:
            raise click.BadParameter(f"Unsupported dataset: {dataset}")


if __name__ == "__main__":
    main()  # pragma: no cover
