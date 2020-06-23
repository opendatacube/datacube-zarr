"""Converting and reprojecting rasters to zarr format."""

import logging
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import rasterio
import xarray as xr
import zarr
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform

from zarr_io import ZarrIO
from zarr_io.utils.uris import uri_join

_DEFAULT_ARRAY = "array"
_META_PREFIX = "zmeta"

_RASTERIO_BAND_ATTRS = ("scales", "offsets", "units", "descriptions")

logger = logging.getLogger(__name__)


def make_zarr_uri(root: Path, group: Optional[str] = None) -> str:
    """Compose zarr uri from path: <protocol>://<root>[#<group>]."""
    protocol, root_ = root.as_uri().split("://", 1)
    uri = uri_join(protocol, root_, group)
    return uri


def zarr_exists(root: Path, group: Optional[str] = None) -> bool:
    """Return True if root (and optionally group) exists."""
    store = ZarrIO().get_root(root.as_uri())
    exists: bool = zarr.storage.contains_group(store, group)
    return exists


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
    src_params.update(src.bounds._asdict())
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
        yield vrt


@contextmanager
def rasterio_src(
    uri: str, crs: Optional[CRS] = None, resolution: Optional[Tuple[float, float]] = None,
) -> rasterio.io.DatasetReaderBase:
    """Open a rasterio source and virtually warp if required."""
    with rasterio.open(uri) as src:
        if crs is not None or resolution is not None:
            with warped_vrt(src, crs=crs, resolution=resolution) as vrt:
                yield vrt
        else:
            yield src


def get_rasterio_datasets(path: Path) -> List[str]:
    """Return full names of rasterio dataset/subdatasets present in a source file."""
    with rasterio.open(path.as_uri(), "r") as src:
        names = [src.name] if src.count > 0 else (src.subdatasets or [])

    if not names:
        raise ValueError(f"No datasets found in {path}.")

    return names


def raster_to_zarr(
    raster: Path,
    out_dir: Path,
    zarr_name: Optional[str] = None,
    crs: Optional[CRS] = None,
    resolution: Optional[Tuple[float, float]] = None,
    **zarrgs: Any,
) -> List[str]:
    """Convert a raster image file to Zarr via rasterio."""
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
            logger.warn(f"zarr already exists: {make_zarr_uri(root, group)}")
            continue

        with rasterio_src(dataset, crs=crs, resolution=resolution) as src:
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
                    tag_list = [src.tags(i).get(tag) for i in range(nbands)]
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

        uri = make_zarr_uri(root, group)
        ZarrIO().save_dataset(uri=uri, dataset=ds, **zarrgs)
        logger.info(f"Created zarr: {uri}")
        output_uris.append(uri)

    return output_uris
