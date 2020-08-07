"""
Functions for converting datasets to zarr format.
Conversions are supported on a local filesystem or S3
"""

import logging
from os.path import commonprefix
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple

import boto3

from datacube_zarr.utils.raster import raster_to_zarr

_SUPPORTED_FORMATS = {
    "ENVI": (".img/.hdr", ".bip/.hdr", ".bil/.hdr", ".bip/.hdr",),
    "ERS": (".ers/.ers.aux.xml/",),
    "GeoTiff": (".tif", ".tiff", ".gtif"),
    "HDF": (".hdf", ".h5",),
    "JPEG2000": (".jp2",),
    "NetCDF": (".nc",),
}

_RASTERIO_FORMATS = (
    "ENVI",
    "ERS",
    "GeoTiff",
    "HDF",
    "JPEG2000",
    "NetCDF",
)
_RASTERIO_FILES = [
    x.split("/")[0] for f in _RASTERIO_FORMATS for x in _SUPPORTED_FORMATS[f]
]

logger = logging.getLogger(__name__)


def _root_as_str(path: Path) -> str:
    """uri path to str."""
    return path.as_uri() if path.as_uri().startswith("s3://") else str(path)


def ignore_file(path: Path, patterns: Optional[List[str]]) -> bool:
    """Check if path matches ignore patterns.

    :param path: path to compar with ignore pattern
    :param patterns: list of glob patterns specifying which paths to ignore
    :return True if path is to be ignored
    """
    return any(path.match(p) for p in patterns) if patterns else False


def get_datasets(in_dir: Path) -> Iterator[Tuple[str, List[Path]]]:
    """
    Find supported datasets within a directory.

    :param in_dir: directory (or S3 path) under-which to look for datasets
    :return: iterator of datasets specified by type and file paths
    """
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
    merge_datasets_per_dir: bool = False,
    **zarrgs: Any,
) -> List[str]:
    """
    Recursively convert datasets in a directory to Zarr format.

    All supported datasets found underneath `in_dir` are (optionally) reprojected and
    converted to zarr format. All other files are copied to the `out_dir` unless ignored.
    If `out_dir` is not specfied the conversion is performed inplace and the original
    raster files are removed.

    :param in_dir: directory (or S3 path) under-which to convert rasters to zarr
    :param out_dir: directory (or S3 path) to save converted datasets
    :param ignore: list of glob patterns specifying files to ignore
    :param merge_datasets_per_dir: option to merge all tifs found at a directory level
    :param zarrgs: keyword arguments to pass to conversion function and zarr_io
    """
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
            zarrs = convert_to_zarr(files, out_dir, zarr_name, **zarrgs)
            output_zarrs.extend(zarrs)
            converted_files.extend(files)

    ignore_patterns = (ignore or []) + [str(f) for f in converted_files]

    # recurse into directories (and copy other files)
    for p in in_dir.iterdir():
        if p.relative_to(in_dir).name and not ignore_file(p, ignore_patterns):
            out_p = out_dir / p.name if out_dir else None
            if p.is_dir():
                zarrs = convert_dir(p, out_p, ignore, merge_datasets_per_dir, **zarrgs)
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
    **zarrgs: Any,
) -> List[str]:
    """
    Convert a supported dataset to Zarr format.

    :param files: list of file making up the dataset (local filesystem or S3)
    :param out_dir: output directory (local filesystem or S3)
    :param zarr_name: name to give the created `.zarr` dataset
    :param zarrgs: keyword arguments to pass to conversion function and zarr_io
    :return: list of generated zarr URIs
    """
    data_file = files[0]
    inplace = out_dir is None
    if out_dir is None:
        out_dir = data_file.parent

    if data_file.suffix in _RASTERIO_FILES:
        zarrs = raster_to_zarr(data_file, out_dir, zarr_name, **zarrgs)
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
            logger.info(f"delete: {_root_as_str(f)}")

    return zarrs
