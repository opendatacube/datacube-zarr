#! /usr/bin/env python

"""
Command line tool for converting dataset to Zarr format.
"""

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Generator, List, Optional, Tuple, Union

import boto3
import click
import rasterio
import xarray as xr
import zarr
from rasterio.crs import CRS
from rasterio.shutil import copy as rio_copy
from rasterio.warp import calculate_default_transform
from s3path import S3Path

from zarr_io import ZarrIO

_DEFAULT_ARRAY = "array"
_META_PREFIX = "zmeta"

_SUPPORTED_FORMATS = {
    "ENVI": (".img/.hdr",),
    "GeoTiff": (".tif", ".tiff", ".gtif"),
    "JPEG2000": (".jp2",),
}

_RASTERIO_FORMATS = ("GeoTiff", "JPEG2000", "ENVI")
_RASTERIO_BAND_ATTRS = ("scales", "offsets", "units", "descriptions")
_RASTERIO_FILES = [
    x.split("/")[0] for f in _RASTERIO_FORMATS for x in _SUPPORTED_FORMATS[f]
]


def get_datasets(in_dir: Path) -> Generator[Tuple[str, List[Path]], None, None]:
    """Find datasets within a directory."""
    for fmt, filetypes in _SUPPORTED_FORMATS.items():
        for exts in [ft.split("/") for ft in filetypes]:
            data_ext = exts.pop(0)
            for datafile in in_dir.glob(f"*{data_ext}"):
                others = [datafile.with_suffix(e) for e in exts]
                if all(o.exists() for o in others):
                    yield fmt, [datafile] + others


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
    crs: Optional[CRS] = None,
    resolution: Optional[Tuple[float, float]] = None,
    **zarrgs: Any,
) -> None:
    """Recursively convert datasets in a directory to Zarr format."""
    assert in_dir.is_dir()

    # find and convert datasets
    converted_files = []
    for fmt, files in get_datasets(in_dir):
        if not ignore_file(files[0], ignore):
            convert_to_zarr(files, out_dir, crs, resolution, **zarrgs)
        converted_files.extend(files)

    ignore_patterns = (ignore or []) + [str(f) for f in converted_files]

    # recurse into directories (and copy other files)
    for p in in_dir.iterdir():
        if p.relative_to(in_dir).name and not ignore_file(p, ignore_patterns):
            out_p = out_dir / p.name if out_dir else None
            if p.is_dir():
                convert_dir(p, out_p, ignore, crs, resolution)
            elif out_p is not None:
                if out_p.as_uri().startswith("file://") and not out_p.parent.exists():
                    out_p.parent.mkdir(exist_ok=True, parents=True)
                out_p.write_bytes(p.read_bytes())


def convert_to_zarr(
    files: List[Path],
    out_dir: Optional[Path] = None,
    crs: Optional[CRS] = None,
    resolution: Optional[Tuple[float, float]] = None,
    **zarrgs: Any,
) -> None:
    """Convert file to Zarr format."""
    data_file = files[0]
    inplace = out_dir is None
    if out_dir is None:
        out_dir = data_file.parent

    if data_file.suffix in _RASTERIO_FILES:
        raster_to_zarr(data_file, out_dir, crs, resolution, **zarrgs)
    else:
        raise ValueError(f"Unsupported data file format: {data_file.suffix}")

    # if converting inplace, remove the original file
    if inplace:
        for f in files:
            if f.as_uri().startswith("s3://"):
                bucket, key = f.as_uri()[5:].split("/", 1)
                boto3.resource("s3").Object(bucket, key).delete()
            else:
                f.unlink()
            print(f"delete: {path_as_str(f)}")


@contextmanager
def warped_vrt(
    src: rasterio.io.DatasetReader,
    crs: Optional[CRS] = None,
    resolution: Optional[Tuple[float, float]] = None,
) -> Union[rasterio.vrt.WarpedVRT, rasterio.io.DatasetReader]:
    """In-memory warped rasterio dataset."""
    src_crs = src.crs
    src_transform = src.transform
    src_params = {"height": src.height, "width": src.width}
    if src_crs:
        src_params.update(src.bounds._asdict())
    else:
        gcps, src_crs = src.gcps
        src_params["gcps"] = gcps
        src_transform = rasterio.transform.from_gcps(gcps)

    dst_crs = crs or src_crs
    transform, width, height = calculate_default_transform(
        src_crs=src_crs, dst_crs=dst_crs, resolution=resolution, **src_params,
    )
    with rasterio.vrt.WarpedVRT(
        src_dataset=src,
        src_crs=src_crs,
        src_transform=src_transform,
        crs=dst_crs,
        transform=transform,
        height=height,
        width=width,
    ) as vrt:
        if not src.crs:
            # For src with gcps write reprojection to temporary file
            # Required due to bug in xarray.open_rasterio() (see xarray PR #4104)
            with tempfile.NamedTemporaryFile() as tmpfile:
                rio_copy(vrt, tmpfile.name, driver="GTiff")
                with rasterio.open(tmpfile) as tmp_src:
                    yield tmp_src
        else:
            yield vrt


@contextmanager
def rasterio_src(
    uri: str, crs: Optional[CRS] = None, resolution: Optional[Tuple[float, float]] = None,
) -> rasterio.io.DatasetReaderBase:
    """Open a rasterio source and virtually warp if required."""
    with rasterio.open(uri) as src:
        reproject = crs is not None or resolution is not None
        gcps = src.crs is None
        if reproject or gcps:
            with warped_vrt(src, crs=crs, resolution=resolution) as vrt:
                yield vrt
        else:
            yield src


def get_protocol_root(path: Path) -> Tuple[str, str]:
    """Split path into protocol and root."""
    if path.as_uri().startswith("s3://"):
        protocol = "s3"
        root = path.as_uri()
    else:
        protocol = "file"
        root = str(path)
    return protocol, root


def zarr_exists(root: Path, group: Optional[str] = None) -> bool:
    """Return True if root (and group) exists."""
    protocol, root_str = get_protocol_root(root)
    store = ZarrIO(protocol).get_root(root_str)
    exists: bool = zarr.storage.contains_group(store, group)
    return exists


def raster_to_zarr(
    raster: Path,
    out_dir: Path,
    crs: Optional[CRS] = None,
    resolution: Optional[Tuple[float, float]] = None,
    **zarrgs: Any,
) -> None:
    """Convert a raster image file to Zarr via rasterio."""
    group = raster.stem
    root = out_dir / f"{group}.zarr"

    if zarr_exists(root, group):
        raise ValueError(
            f"zarr group already exists (root={root.as_uri()}, group={group})."
        )

    with rasterio_src(raster.as_uri(), crs=crs, resolution=resolution) as src:
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
                        if k in _RASTERIO_BAND_ATTRS:
                            v = [v[i]]
                        arr.attrs[f"{_META_PREFIX}_{k}"] = v

                # Get band-specific tags
                for tag, tval in src.tags(i).items():
                    arr.attrs[f"{_META_PREFIX}_{tag}"] = tval

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
    "--multi-dim", is_flag=True, help="Keep multi-banded tifs as 3-dimensional arrays."
)
def main(
    dataset: Path,
    outpath: Optional[Path],
    inplace: bool,
    crs: Optional[CRS],
    resolution: Optional[Tuple[float, float]],
    chunk: Optional[List[Tuple[str, int]]],
    ignore: List[str],
    multi_dim: bool,
) -> None:
    """Convert datasets to Zarr format.

    If DATASET argument is a directory all supported datasets found
    recursively within are converted. Otherwise DATASET must point to
    a supported dataset file.

    Paths can be either local files/directories or 's3://' URIs.

    Output projection can be specified via `--crs` and/or `--resolution`.
    If input dataset contains ground control points it will automatically
    be reprojected to an affine transform.

    Chunking options should be set such that the resulting zarr chunks
    are approx 10-20 MB. For 2D arrays, a chunk size of ~2000 is a good
    starting point.

    Supported datasets: ENVI, GeoTiff, JPEG2000.
    """
    check_options(outpath, inplace)
    ignore = absolute_ignores(ignore, dataset)
    chunks = dict(chunk) if chunk else None

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
            multi_dim=multi_dim,
        )

    # Convert this single supported dataset
    else:
        try:
            fmt, files = next(
                ds for ds in get_datasets(dataset.parent) if ds[1][0] == dataset
            )
            if ignore_file(files[0], ignore):
                print(f"ignoring dataset: {dataset}")
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
