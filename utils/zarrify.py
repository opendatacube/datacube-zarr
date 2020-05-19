#! /usr/bin/env python

"""
Command line tool for converting dataset to Zarr format.
"""

from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import boto3
import click
import rasterio
import xarray as xr
from s3path import S3Path

from zarr_io import ZarrIO

_SUPPORTED_FORMATS = {
    "GeoTiff": (".tif", ".tiff", ".gtif"),
}

_DATA_FILES = [x for xs in _SUPPORTED_FORMATS.values() for x in xs]

_DEFAULT_ARRAY = "array"
_META_PREFIX = "zmeta"
_GTIFF_BAND_ATTRS = ("scales", "offsets", "units", "descriptions")


def path_as_str(path: Path) -> str:
    """Convert Path to string and S3Path to URI."""
    s = path.as_uri()
    if s.startswith("file://"):
        s = s[7:]
    return s


def save_dataset_to_zarr(ds: xr.Dataset, root: Path, group: str, **kwargs: Any) -> None:
    """Save an xarray dataset to s3 or file in Zarr format."""
    protocol = "s3" if root.as_uri().startswith("s3://") else "file"
    zio = ZarrIO(protocol)
    root_str = path_as_str(root)
    zio.save_dataset(root=root_str, group_name=group, dataset=ds, **kwargs)
    print(f"create: {root_str}")


def convert_dir(
    in_dir: Path,
    out_dir: Optional[Path] = None,
    ignore: Optional[List[str]] = None,
    **zarrgs: Any
) -> None:
    """Recursively convert datasets in a directory to Zarr format."""
    assert in_dir.is_dir()
    sub_paths = [
        p for p in in_dir.iterdir()
        if p.relative_to(in_dir).name and not ignore_file(p, ignore)
    ]
    for p in sub_paths:
        out_p = out_dir / p.name if out_dir else None
        if p.is_dir():
            convert_dir(p, out_p)
        elif p.suffix in _DATA_FILES:
            convert_to_zarr(p, out_dir, **zarrgs)
        elif out_p is not None:
            out_p.write_bytes(p.read_bytes())


def convert_to_zarr(in_file: Path, out_dir: Optional[Path] = None, **zarrgs: Any) -> None:
    """Convert file to Zarr format."""
    inplace = out_dir is None
    if out_dir is None:
        out_dir = in_file.parent

    if in_file.suffix in _SUPPORTED_FORMATS["GeoTiff"]:
        geotiff_to_zarr(in_file, out_dir, **zarrgs)
    else:
        raise ValueError(f"Unsupported data file format: {in_file.suffix}")

    # if converting inplace, remove the original file
    if inplace:
        if in_file.as_uri().startswith("s3://"):
            bucket, key = in_file.as_uri()[5:].split("/", 1)
            boto3.resource("s3").Object(bucket, key).delete()
        else:
            in_file.unlink()
        print(f"delete: {path_as_str(in_file)}")


def geotiff_to_zarr(tiff: Path, out_dir: Path, **zarrgs: Any) -> None:
    """Convert a geotiff file to Zarr."""
    with rasterio.open(tiff.as_uri(), "r") as src:
        da = xr.open_rasterio(src)
        nbands = da.shape[0]

        multi_dim = zarrgs.pop("multi_dim", False)
        dim = None if multi_dim else "band"
        name = _DEFAULT_ARRAY if multi_dim else None
        ds = da.to_dataset(dim=dim, name=name)

        if multi_dim:
            # DataSet attrs already passed to DataArray. Set nodata and tags.
            ds[_DEFAULT_ARRAY].attrs["nodata"] = da.nodatavals
            tag_names = {k for i in range(nbands) for k in src.tags(i)}
            for tag in tag_names:
                tag_list = [src.tags(i).get(tag) for i in nbands]
                ds[_DEFAULT_ARRAY].attrs[f"{_META_PREFIX}_{tag}"] = tag_list
        else:
            # Rename variable keys to strings required by zarr
            ds = ds.rename_vars({k: f"band{k}" for k in ds.data_vars.keys()})

            # Copy DataSet attrs to each DataArray
            for i, arr in enumerate(ds.data_vars.values()):
                arr.attrs["nodata"] = da.nodatavals[i]
                arr.attrs["crs"] = ds.crs
                for k, v in da.attrs.items():
                    if k not in ("nodatavals", "crs"):
                        if k in _GTIFF_BAND_ATTRS:
                            v = [v[i]]
                        arr.attrs[f"{_META_PREFIX}_{k}"] = v

                # Get band-specific tags
                for tag, tval in src.tags(i).items():
                    arr.attrs[f"{_META_PREFIX}_{tag}"] = tval

    group = tiff.stem
    root = out_dir / f"{group}.zarr"
    save_dataset_to_zarr(ds, root, group, **zarrgs)


def ignore_file(path: Path, patterns: Optional[List[str]]) -> bool:
    """Check if path matches ignore patterns."""
    return any(path.match(p) for p in patterns) if patterns else False


# CLI functions

class KeyValue(click.ParamType):
    """A click param for any key/value pairs."""

    name = "Key:Value"

    def __init__(
        self,
        key: Callable[[str], Any] = str,
        value: Callable[[str], Any] = str,
        sep: str = ":"
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


def check_options(outpath: Path, inplace: bool) -> None:
    """Some checks on command inputs."""
    if not outpath and not inplace:
        raise click.UsageError("--inplace flag is required if --outpath is not set.")

    if outpath and inplace:
        raise click.UsageError("Can not set both --outpath and --inplace options.")


def absolute_ignores(ignore: List[str], abs_path: Path) -> List[str]:
    """Prepend absolute ignore patterns with path."""
    return [str(abs_path / i) if i[0] == "/" else i for i in ignore]


@click.command()
@click.argument("dataset", type=FileOrS3Path(exists=True), required=True)
@click.option(
    "--outpath", type=FileOrS3Path(), required=False,
    help="Path to save the converted dataset directory."
)
@click.option(
    "--inplace", is_flag=True, help="Convert inplace (deletes original data files)."
)
@click.option(
    "--ignore", type=str, help="Comma separated list of file patterns to ignore.",
    callback=lambda ctx, param, value: value.split(",") if value else [],
)
@click.option(
    "--chunk", type=KeyValue(value=int), multiple=True,
    help="Zarr chunk option '<dim>:<size>'."
)
@click.option(
    "--multi-dim", is_flag=True, help="Keep multi-banded tifs as 3-dimensional arrays."
)
def main(
    dataset: Path,
    outpath: Path,
    inplace: bool,
    chunk: Optional[List[Tuple[str, int]]],
    ignore: List[str],
    multi_dim: bool,
) -> None:
    """Convert datasets to Zarr format.

    If DATASET argument is a directory all supported datasets found
    recursively within are converted. Otherwise DATASET must point to
    a supported dataset file.

    Paths can be either local files/directories or 's3://' URIs.

    Chunking options should be set such that the resulting zarr chunks
    are approx 10-20 MB. For 2D arrays, a chunk size of ~2000 is a good
    starting point.

    Supported datasets: GeoTiff.
    """
    check_options(outpath, inplace)
    ignore = absolute_ignores(ignore, dataset)
    chunks = dict(chunk) if chunk else None

    if dataset.is_dir():
        if outpath:
            outpath = outpath / dataset.parts[-1]
        convert_dir(
            dataset, outpath, ignore=ignore, chunks=chunks, multi_dim=multi_dim
        )
    elif dataset.suffix in _DATA_FILES:
        if ignore_file(dataset, ignore):
            print(f"ignoring dataset: {dataset}")
        else:
            convert_to_zarr(dataset, outpath, chunks=chunks, multi_dim=multi_dim)
    else:
        raise click.BadParameter(f"Unsupported dataset: {dataset}")


if __name__ == "__main__":
    main()
