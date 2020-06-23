#! /usr/bin/env python

"""
Command line tool for converting dataset to Zarr format.
"""

import logging
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import click
from rasterio.crs import CRS
from s3path import S3Path

from zarr_io.utils.convert import convert_dir, convert_to_zarr, get_datasets, ignore_file

logger = logging.getLogger()
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

        if self.exists and not path.exists:
            raise ValueError(f"{path.as_uri()} does not exist.")

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
            except ValueError:
                p = CRS.from_string(value)
        except RuntimeError:
            self.fail(f"{value} is not a valid CRS", param, ctx)

        return p


def check_options(outpath: Optional[Path], inplace: bool) -> None:
    """Some checks on command inputs."""
    if not outpath and not inplace:
        raise click.UsageError("--inplace flag is required if --outpath is not set.")

    if outpath and inplace:
        raise click.UsageError("Can not set both --outpath and --inplace options.")


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

    for logger_name in ("boto3", "botocore", "urllib3", "rasterio"):
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
    type=KeyValue(value=int),
    multiple=True,
    help="Zarr chunk option '<dim>:<size>'.",
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
    "-v",
    "--verbose",
    is_flag=True,
    callback=setup_logging,
    expose_value=False,
    is_eager=True,
    help="Enables verbose mode.",
)
def main(
    dataset: Path,
    outpath: Optional[Path],
    inplace: bool,
    crs: Optional[CRS],
    resolution: Optional[Tuple[float, float]],
    chunk: Optional[List[Tuple[str, int]]],
    ignore: List[str],
    merge_datasets_per_dir: bool,
    multi_dim: bool,
) -> None:
    """Convert datasets to Zarr format.

    If DATASET argument is a directory all supported datasets found
    recursively within are converted. Otherwise DATASET must point to
    a supported dataset file.

    Paths can be either local files/directories or 's3://' URIs.

    Output projection can be specified via `--crs` and/or `--resolution`.

    Chunking options should be set such that the resulting zarr chunks
    are approx 10-20 MB. For 2D arrays, a chunk size of ~2000 is a good
    starting point.

    Supported datasets: ENVI, GeoTiff, HDF, JPEG2000.

    Note: Only gridded HDF datasets are supported. s3:// paths are not
    supported for HDF4 datasets.
    """
    check_options(outpath, inplace)
    ignore = absolute_ignores(ignore, dataset)
    chunks = dict(chunk) if chunk else None

    if not dataset.exists():
        raise click.BadParameter(f"Dataset does not exist: {dataset}")

    # Recurse into directory an convert supported datasets
    if dataset.is_dir():
        outpath = outpath / dataset.parts[-1] if outpath else None
        convert_dir(
            in_dir=dataset,
            out_dir=outpath,
            ignore=ignore,
            crs=crs,
            resolution=resolution,
            chunks=chunks,
            merge_datasets_per_dir=merge_datasets_per_dir,
            multi_dim=multi_dim,
        )

    # Convert this single supported dataset
    else:
        try:
            fmt, files = next(
                ds for ds in get_datasets(dataset.parent) if ds[1][0] == dataset
            )
            if ignore_file(files[0], ignore):
                logger.warn(f"Ignoring dataset: {dataset}")
            else:
                convert_to_zarr(
                    files=files,
                    out_dir=outpath,
                    crs=crs,
                    resolution=resolution,
                    chunks=chunks,
                    multi_dim=multi_dim,
                )
        except StopIteration:
            raise click.BadParameter(f"Unsupported dataset: {dataset}")


if __name__ == "__main__":
    main()
