"""Functions for converting and reprojecting rasters to zarr format."""

import logging
import re
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator, List, Optional, Tuple, Union

import rasterio
import xarray as xr
import zarr
from dask.diagnostics import ProgressBar
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform

from datacube_zarr import ZarrIO
from datacube_zarr.utils.uris import uri_join

_DEFAULT_ARRAY = "array"
_META_PREFIX = "zmeta"

_RASTERIO_BAND_ATTRS = ("scales", "offsets", "units", "descriptions")

logger = logging.getLogger(__name__)


class _SimpleTimer:
    last_duration: Optional[float] = None


@contextmanager
def _simple_timer() -> Iterator[_SimpleTimer]:
    """Simple timer context with `last_duration` to match dask `ProgressBar`."""
    start = perf_counter()
    t = _SimpleTimer()
    yield t
    t.last_duration = perf_counter() - start


def make_zarr_uri(root: Path, group: Optional[str] = None) -> str:
    """
    Compose zarr uri from (S3 or file) path: <protocol>://<root>[#<group>].

    :param root: path (S3 or file) to zarr root
    :param group: name of zarr group
    :return zarr uri
    """
    protocol, root_ = root.as_uri().split("://", 1)
    uri = uri_join(protocol, root_, group)
    return uri


def zarr_exists(root: Path, group: Optional[str] = None) -> bool:
    """
    Test if a zarr root (and group) exists.

    :param root: path (S3 or file) to zarr root
    :param group: name of zarr group
    :return: True if root (and optionally group) exists.
    """
    store = ZarrIO().get_root(root.as_uri())
    exists: bool = zarr.storage.contains_group(store, group)
    return exists


@contextmanager
def _warped_vrt(
    src: rasterio.io.DatasetReader,
    crs: Optional[CRS] = None,
    resolution: Optional[Tuple[float, float]] = None,
) -> Union[rasterio.vrt.WarpedVRT, rasterio.io.DatasetReader]:
    """In-memory warped rasterio dataset."""
    src_crs = src.crs
    src_transform = src.transform
    src_params = {"height": src.height, "width": src.width}
    src_params.update(src.bounds._asdict())
    dst_crs = crs or src_crs
    transform, width, height = calculate_default_transform(
        src_crs=src_crs, dst_crs=dst_crs, resolution=resolution, **src_params
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
        yield vrt


@contextmanager
def rasterio_src(
    uri: str, crs: Optional[CRS] = None, resolution: Optional[Tuple[float, float]] = None
) -> rasterio.io.DatasetReaderBase:
    """Open a rasterio source and virtually reproject if required.

    :param uri: rasterio dataset URI
    :param crs: coordinate system to reproject into to
    :param resolution: resolution to resample to
    :return: a rasterio dataset source
    """
    with rasterio.open(uri) as src:
        if crs is not None or resolution is not None:
            with _warped_vrt(src, crs=crs, resolution=resolution) as vrt:
                yield vrt
        else:
            yield src


def get_rasterio_datasets(raster_file: Path) -> List[str]:
    """
    Return full names of rasterio dataset/subdatasets present in a source file.

    :param raster_file: raster file or S3 path
    :return: list of dataset URIs
    """
    with rasterio.open(raster_file.as_uri(), "r") as src:
        names = [src.name] if src.count > 0 else (src.subdatasets or [])

    if not names:
        raise ValueError(f"No datasets found in {raster_file}.")

    return names


def raster_to_zarr(  # noqa: C901
    raster: Path,
    out_dir: Path,
    zarr_name: Optional[str] = None,
    crs: Optional[CRS] = None,
    resolution: Optional[Tuple[float, float]] = None,
    separate_bands: bool = False,
    preload_data: bool = False,
    auto_chunk: bool = False,
    progress: bool = False,
    **zarrgs: Any,
) -> List[str]:
    """
    Convert a raster image file to Zarr via rasterio.

    :param raster: a path to a raster file (local filesystem or S3)
    :param out_dir: output directory (local filesystem or S3)
    :param zarr_name: name to give the created `.zarr` dataset
    :param crs: output coordinate system to reproject to
    :param resolution: output resolution
    :param separate_bands: Split each band into a separate zarr array
    :param preload_data: Load source into memory
    :param auto_chunk: Automatically chunk x, y dimensions
    :param progress: Display progress bar for zarr creation
    :param zarrgs: keyword arguments to pass to `ZarrIO.save_dataset`
    :return: list of generated zarr URIs
    """
    output_uris = []
    for dataset in get_rasterio_datasets(raster):

        # Generate zarr root and group names for dataset
        base_group = raster.stem if zarr_name else ""
        root = out_dir / f"{zarr_name or raster.stem}.zarr"
        match = re.search(fr"{raster.name}:/*(\S+)", dataset)
        if match is not None:
            subgroup = match.groups()[0]
            group = f"{base_group}/{subgroup}" if base_group else subgroup
        else:
            group = base_group
        group = group.replace(":", "/")

        if zarr_exists(root, group):
            logger.warning(f"zarr already exists: {make_zarr_uri(root, group)}")
            continue

        with rasterio_src(dataset, crs=crs, resolution=resolution) as src:
            da = xr.open_rasterio(src)

            # Reset PROJ string to remove "+init" from "<auth>:<auth_code>" defs
            da.attrs["crs"] = CRS.from_string(da.crs).to_string()

            # Can be useful when converting striped source data
            if preload_data:
                logger.debug(f"Preloading {src.name} into memory.")
                da.load()

            nbands = da.shape[0]
            if nbands == 1:
                # Remove uneccesary dimension
                da = da.sel(band=1, drop=True)

            # Convert to Dataset
            dim = "band" if separate_bands else None
            name = None if separate_bands else _DEFAULT_ARRAY
            ds = da.to_dataset(dim=dim, name=name)

            if auto_chunk:
                zarrgs["chunks"] = {d: "auto" for d in list(ds.dims)[-2:]}
                logger.debug(f"Auto setting chunk options to {zarrgs['chunks']}")

            if separate_bands:
                # Rename variable keys to strings required by zarr
                ds = ds.rename_vars({k: f"band{k:d}" for k in ds.data_vars.keys()})

                # Copy DataSet attrs to each DataArray
                for i, arr in enumerate(ds.data_vars.values()):
                    arr.attrs["nodata"] = da.nodatavals[i]
                    arr.attrs["crs"] = ds.crs
                    arr.attrs["transform"] = ds.transform
                    for k, v in da.attrs.items():
                        if k not in ("nodatavals", "crs", "transform"):
                            if k in _RASTERIO_BAND_ATTRS:
                                v = [v[i]]
                            arr.attrs[f"{_META_PREFIX}_{k}"] = v

                    # Get band-specific tags
                    for tag, tval in src.tags(i).items():
                        arr.attrs[f"{_META_PREFIX}_{tag}"] = tval
            else:
                # DataSet attrs already passed to DataArray. Set nodata and tags.
                ds[_DEFAULT_ARRAY].attrs["nodata"] = da.nodatavals
                tag_names = {k for i in range(nbands) for k in src.tags(i)}
                for tag in tag_names:
                    tag_list = [src.tags(i).get(tag) for i in range(nbands)]
                    ds[_DEFAULT_ARRAY].attrs[f"{_META_PREFIX}_{tag}"] = tag_list

        uri = make_zarr_uri(root, group)

        progress_ctx = ProgressBar if progress else _simple_timer
        with progress_ctx() as p:
            ZarrIO().save_dataset(uri=uri, dataset=ds, **zarrgs)

        logger.info(
            f"Created zarr: {uri[7:] if uri.startswith('file') else uri} "
            f"({p.last_duration:.1f} sec)"
        )

        output_uris.append(uri)

    return output_uris
