from pathlib import Path

import pytest
import xarray as xr

from zarr_io.utils.raster import raster_to_zarr
from zarr_io.zarr_io import ZarrIO


@pytest.mark.parametrize("chunks", [None, {"x": 50, "y": 50}])
def test_raster_to_zarr(tmp_raster, tmpdir, chunks):
    """Convert raster to zarr."""
    outdir = Path(tmpdir)
    uris = raster_to_zarr(tmp_raster, outdir, chunks=chunks)
    assert len(uris) == 1

    zarr_file = outdir / f"{tmp_raster.stem}.zarr"
    assert zarr_file.exists() is True

    da_raster = xr.open_rasterio(tmp_raster)
    da_zarr = (
        ZarrIO()
        .load_dataset(uris[0])["band1"]
        .assign_coords({"band": 1})
        .expand_dims("band")
    )
    assert da_raster.equals(da_zarr)
    assert da_raster.crs == da_zarr.crs


@pytest.mark.parametrize("multi_dim", [True, False])
def test_raster_to_zarr_multi_band(tmp_raster_multiband, tmpdir, multi_dim):
    """Convert multibanded raster to zarr."""
    outdir = Path(tmpdir)
    uris = raster_to_zarr(tmp_raster_multiband, outdir, multi_dim=multi_dim)
    assert len(uris) == 1

    zarr_file = outdir / f"{tmp_raster_multiband.stem}.zarr"
    assert zarr_file.exists() is True

    da_raster = xr.open_rasterio(tmp_raster_multiband)
    ds_zarr = ZarrIO().load_dataset(uris[0])

    if multi_dim is True:
        da_zarr = ds_zarr["array"]
    else:
        da_zarr = xr.concat(ds_zarr.data_vars.values(), dim="band").assign_coords(
            {"band": list(range(1, len(ds_zarr) + 1))}
        )
    assert da_raster.equals(da_zarr)
    assert da_raster.crs == da_zarr.crs
