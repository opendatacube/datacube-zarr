"""Function for converting datasets to zarr format."""

import logging
from os.path import commonprefix
from pathlib import Path
from typing import Any, Generator, List, Optional, Tuple

import boto3
from rasterio.crs import CRS

from zarr_io.utils.raster import raster_to_zarr

_SUPPORTED_FORMATS = {
    "ENVI": (".img/.hdr", ".bip/.hdr", ".bil/.hdr", ".bip/.hdr",),
    "GeoTiff": (".tif", ".tiff", ".gtif"),
    "HDF": (".hdf", ".h5",),
    "JPEG2000": (".jp2",),
    "NetCDF": (".nc",),
}

_RASTERIO_FORMATS = (
    "ENVI",
    "GeoTiff",
    "HDF",
    "JPEG2000",
    "NetCDF",
)
_RASTERIO_FILES = [
    x.split("/")[0] for f in _RASTERIO_FORMATS for x in _SUPPORTED_FORMATS[f]
]

logger = logging.getLogger(__name__)


def root_as_str(path: Path) -> str:
    """uri path to str."""
    return path.as_uri() if path.as_uri().startswith("s3://") else str(path)


def ignore_file(path: Path, patterns: Optional[List[str]]) -> bool:
    """Check if path matches ignore patterns."""
    return any(path.match(p) for p in patterns) if patterns else False


def get_datasets(in_dir: Path) -> Generator[Tuple[str, List[Path]], None, None]:
    """Find supported datasets within a directory."""
    for fmt, filetypes in _SUPPORTED_FORMATS.items():
        for exts in [ft.split("/") for ft in filetypes]:
            data_ext = exts.pop(0)
            for datafile in in_dir.glob(f"*{data_ext}"):
                others = [datafile.with_suffix(e) for e in exts]
                if all(o.exists() for o in others):
                    yield fmt, [datafile] + others


def convert_dir(
    in_dir: Path,
    out_dir: Optional[Path] = None,
    ignore: Optional[List[str]] = None,
    crs: Optional[CRS] = None,
    resolution: Optional[Tuple[float, float]] = None,
    merge_datasets_per_dir: bool = False,
    **zarrgs: Any,
) -> List[str]:
    """Recursively convert datasets in a directory to Zarr format."""
    assert in_dir.is_dir()
    output_zarrs = []

    # find and convert datasets
    datasets = [f for t, f in get_datasets(in_dir) if not ignore_file(f[0], ignore)]
    converted_files = []
    if datasets:
        zarr_name = None
        if merge_datasets_per_dir:
            zarr_name = commonprefix([f[0].stem for f in datasets]) or in_dir.name

        for files in datasets:
            zarrs = convert_to_zarr(
                files, out_dir, zarr_name, crs, resolution, **zarrgs
            )
            output_zarrs.extend(zarrs)
            converted_files.extend(files)

    ignore_patterns = (ignore or []) + [str(f) for f in converted_files]

    # recurse into directories (and copy other files)
    for p in in_dir.iterdir():
        if p.relative_to(in_dir).name and not ignore_file(p, ignore_patterns):
            out_p = out_dir / p.name if out_dir else None
            if p.is_dir():
                zarrs = convert_dir(
                    p, out_p, ignore, crs, resolution, merge_datasets_per_dir, **zarrgs
                )
                output_zarrs.extend(zarrs)
            elif out_p is not None:
                if out_p.as_uri().startswith("file://") and not out_p.parent.exists():
                    out_p.parent.mkdir(exist_ok=True, parents=True)
                out_p.write_bytes(p.read_bytes())

    return output_zarrs


def convert_to_zarr(
    files: List[Path],
    out_dir: Optional[Path] = None,
    zarr_name: Optional[str] = None,
    crs: Optional[CRS] = None,
    resolution: Optional[Tuple[float, float]] = None,
    **zarrgs: Any,
) -> List[str]:
    """Convert a dataset (of potentially multiple files) to Zarr format."""
    data_file = files[0]
    inplace = out_dir is None
    if out_dir is None:
        out_dir = data_file.parent

    if data_file.suffix in _RASTERIO_FILES:
        zarrs = raster_to_zarr(data_file, out_dir, zarr_name, crs, resolution, **zarrgs)
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
            logger.info(f"delete: {root_as_str(f)}")

    return zarrs
