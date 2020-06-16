from pathlib import Path

import pytest
import xarray as xr

from zarr_io.utils.convert import convert_dir, get_datasets
from zarr_io.utils.raster import raster_to_zarr, zarr_exists
from zarr_io.utils.uris import uri_split
from zarr_io.zarr_io import ZarrIO


def raster_and_zarr_are_equal(raster_file, zarr_uri, multi_dim=False):
    """Compare raster and zarr files."""
    da_raster = xr.open_rasterio(raster_file)
    ds_zarr = ZarrIO().load_dataset(zarr_uri)

    if multi_dim is True:
        da_zarr = ds_zarr["array"]
    else:
        da_zarr = xr.concat(ds_zarr.data_vars.values(), dim="band").assign_coords(
            {"band": list(range(1, len(ds_zarr) + 1))}
        )
    data_coords_dims_equal = da_raster.equals(da_zarr)
    crs_equal = da_raster.crs == da_zarr.crs
    return data_coords_dims_equal and crs_equal


@pytest.mark.parametrize("chunks", [None, {"x": 50, "y": 50}])
def test_raster_to_zarr(tmp_raster, tmpdir, chunks):
    """Convert raster to zarr."""
    outdir = Path(tmpdir)
    uris = raster_to_zarr(tmp_raster, outdir, chunks=chunks)
    assert len(uris) == 1

    zarr_file = outdir / f"{tmp_raster.stem}.zarr"
    assert zarr_exists(zarr_file) is True

    assert raster_and_zarr_are_equal(tmp_raster, uris[0])


@pytest.mark.parametrize("multi_dim", [True, False])
def test_raster_to_zarr_multi_band(tmp_raster_multiband, tmpdir, multi_dim):
    """Convert multibanded raster to zarr."""
    outdir = Path(tmpdir)
    uris = raster_to_zarr(tmp_raster_multiband, outdir, multi_dim=multi_dim)
    assert len(uris) == 1

    zarr_file = outdir / f"{tmp_raster_multiband.stem}.zarr"
    assert zarr_exists(zarr_file) is True

    assert raster_and_zarr_are_equal(tmp_raster_multiband, uris[0], multi_dim=multi_dim)


def test_find_datasets_geotif(tmp_dir_of_rasters):
    """Test finding geotif datasets."""
    data_dir, geotifs = tmp_dir_of_rasters
    found_types, found_datasets = zip(*get_datasets(data_dir))
    found_geotifs = [ds[0] for ds in found_datasets]
    assert all(t == "GeoTiff" for t in found_types)
    assert set(geotifs) == set(found_geotifs)


@pytest.mark.parametrize("merge_datasets_per_dir", [False, True])
def test_convert_dir_geotif(tmp_dir_of_rasters, tmpdir, merge_datasets_per_dir):
    """Test converting a directory of geotifs."""
    data_dir, geotifs = tmp_dir_of_rasters
    outdir = Path(tmpdir)
    zarrs = convert_dir(data_dir, outdir, merge_datasets_per_dir=merge_datasets_per_dir)
    assert len(zarrs) == len(geotifs)
    for z, g in zip(sorted(zarrs), sorted(geotifs)):
        protocol, root, group = uri_split(z)
        root_stem = root.rsplit("/", 1)[1].split(".")[0]
        if merge_datasets_per_dir:
            assert root_stem == "raster"
            assert group == g.stem
        else:
            assert root_stem == g.stem
            assert group == ""

        assert raster_and_zarr_are_equal(g, z)
