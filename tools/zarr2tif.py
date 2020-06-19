#! /usr/bin/env python

"""Convert a zarr dataset to geotiff(s)."""

from functools import reduce
from pathlib import Path
from typing import Any, Generator, Optional

import click
import rasterio
import xarray as xr
import zarr as z
from rasterio.crs import CRS
from zarr.hierarchy import Group

from zarr_io import ZarrIO
from zarr_io.utils.raster import _META_PREFIX, _RASTERIO_BAND_ATTRS
from zarrify import FileOrS3Path


def get_dot_zarrs(path: Path) -> Generator[Path, None, None]:
    """Recursively find .zarr dirs."""
    if path.suffix == ".zarr":
        yield path
    else:
        subdirs = [p for p in path.iterdir() if p.is_dir()]
        for p in subdirs:
            yield from get_dot_zarrs(p)


def get_groups_with_arrays(group: Group) -> Generator[str, None, None]:
    """Find any arrays within zarr heirarchy."""
    name = group.name
    if list(group.arrays()):
        yield name
    for _, g in group.groups():
        yield from get_groups_with_arrays(g)


def xarray_to_geotiff(ds: xr.Dataset, outpath: Path) -> None:
    """Save xarray dataset as geotiff."""
    count = len(ds)
    band_names = list(ds.data_vars.keys())

    def _get_common_attr(name: str) -> Any:
        vals = set(getattr(ds[b], name) for b in band_names)
        if len(vals) > 1:
            raise ValueError(f"Data arrays have inconsistent '{name}'' attr: {vals}")
        return vals.pop()

    height, width = _get_common_attr("shape")
    dtype = _get_common_attr("dtype")
    nodata = [ds[b].nodata for b in band_names]

    meta = {
        "count": count,
        "width": width,
        "height": height,
        "nodatavals": nodata,
        "crs": CRS.from_string(ds.crs),
        "transform": ds.transform,
        "dtype": dtype,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "lzw",
        "driver": "GTiff",
    }

    with rasterio.open(outpath.as_uri(), "w", **meta) as dst:
        for i, b in enumerate(band_names, 1):
            da = ds[b]
            dst.write(da.values, i)
            for attr in _RASTERIO_BAND_ATTRS:
                if attr in ds.attrs:
                    setattr(dst, attr, ds.attrs[attr])
            da_attrs = [(str(k), v) for k, v in da.attrs.items()]
            band_attrs = {
                k.split(_META_PREFIX + "_")[1]: v
                for k, v in da_attrs
                if k.startswith(_META_PREFIX)
            }
            dst.update_tags(i, **band_attrs)


@click.command()
@click.argument("path", type=FileOrS3Path(exists=True), required=True)
@click.option("--outdir", type=FileOrS3Path(exists=True))
def main(path: Path, outdir: Optional[Path]) -> None:
    """Convert a .zarr dataset to geotif(s).

    Can read but not write to S3.
    """
    for zarr in get_dot_zarrs(path):
        relpath = Path(zarr.name) if zarr == path else zarr.relative_to(path)
        print(f"Converting {relpath}")
        outpath = outdir / relpath.parent if outdir else zarr.parent
        zio = ZarrIO()
        root_group = z.group(zio.get_root(str(zarr)))
        for group in get_groups_with_arrays(root_group):
            ds = zio.open_dataset(f"{zarr.as_uri()}#{group}")
            tif_stem = reduce(lambda s, c: s.replace(c, "_"), "/:", group[1:])
            tif_path = outpath / f"{tif_stem}.tif"
            outpath.mkdir(exist_ok=True, parents=True)
            xarray_to_geotiff(ds, tif_path)
            print(tif_path.relative_to(outpath))


if __name__ == "__main__":
    main()
