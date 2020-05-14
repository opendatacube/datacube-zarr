#! /usr/bin/env python

from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import boto3
import click
import xarray as xr
from s3path import S3Path

from zarr_io import ZarrIO

_GEOTIFFS = (".tif", ".tiff", ".gtif")
_DATA_FILES = [x for t in (_GEOTIFFS,) for x in t]


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
    print(f"create: {root_str}/{group}")


def convert_dir(in_dir: Path, out_dir: Optional[Path] = None, **zarrgs: Any) -> None:
    """Recursively convert datasets in a directory to Zarr format."""
    assert in_dir.is_dir()
    sub_paths = [p for p in in_dir.iterdir() if p.relative_to(in_dir).name]
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

    if in_file.suffix in _GEOTIFFS:
        raster_to_zarr(in_file, out_dir, **zarrgs)
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


def raster_to_zarr(raster: Path, out_dir: Path, **zarrgs: Any) -> None:
    """Convert a raster file to Zarr."""
    da = xr.open_rasterio(raster.as_uri())
    da.attrs["nodata"] = da.nodatavals[0]
    ds = da.to_dataset(name="array")
    save_dataset_to_zarr(ds, out_dir, group=raster.stem, **zarrgs)


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


@click.command()
@click.argument("dataset", type=FileOrS3Path(exists=True), required=True)
@click.option(
    "--outpath", type=FileOrS3Path(), required=False,
    help="Path to save the converted dataset directory."
)
@click.option(
    "--chunk", type=KeyValue(value=int), multiple=True,
    help="Zarr chunk option '<dim>:<size>'."
)
@click.option(
    "--inplace", is_flag=True, help="Convert inplace (deletes original data files)."
)
def main(
    dataset: Path, outpath: Path, inplace: bool, chunk: Optional[List[Tuple[str, int]]]
) -> None:
    """Convert datasets to Zarr format."""
    check_options(outpath, inplace)
    chunks = dict(chunk) if chunk else None
    convert_dir(dataset, outpath, chunks=chunks)


if __name__ == "__main__":
    main()
