from pathlib import Path

import pytest
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject
from s3path import S3Path

from datacube_zarr.utils.raster import get_rasterio_datasets, raster_to_zarr, zarr_exists
from datacube_zarr.utils.uris import uri_split

from .utils import raster_and_zarr_are_equal


@pytest.mark.parametrize("chunks", [None, {"x": 50, "y": 50}])
def test_raster_to_zarr(tmp_raster, tmp_storage_path, chunks):
    """Convert raster to zarr."""
    uris = raster_to_zarr(tmp_raster, tmp_storage_path, chunks=chunks)
    assert len(uris) == 1

    zarr_file = tmp_storage_path / f"{tmp_raster.stem}.zarr"
    assert zarr_exists(zarr_file) is True
    assert raster_and_zarr_are_equal(tmp_raster.as_uri(), uris[0])


@pytest.mark.parametrize("preload_data", [False, True])
@pytest.mark.parametrize("multi_dim", [True, False])
def test_raster_to_zarr_multi_band(
    tmp_raster_multiband, tmp_storage_path, multi_dim, preload_data
):
    """Convert multibanded raster to zarr."""
    uris = raster_to_zarr(
        tmp_raster_multiband,
        tmp_storage_path,
        multi_dim=multi_dim,
        preload_data=preload_data,
    )
    assert len(uris) == 1

    zarr_file = tmp_storage_path / f"{tmp_raster_multiband.stem}.zarr"
    assert zarr_exists(zarr_file) is True

    assert raster_and_zarr_are_equal(
        tmp_raster_multiband.as_uri(), uris[0], multi_dim=multi_dim
    )


@pytest.mark.filterwarnings("ignore:Dataset has no geotransform set.")
def test_hdf4_to_zarr(tmp_hdf4_dataset, tmp_storage_path):
    """Convert raster to zarr."""
    uris = raster_to_zarr(tmp_hdf4_dataset, tmp_storage_path)
    assert len(uris) == 5

    zarr_file = tmp_storage_path / f"{tmp_hdf4_dataset.stem}.zarr"
    assert zarr_exists(zarr_file) is True

    for zarr_uri in uris:
        netcdf_uri = f"netcdf:{tmp_hdf4_dataset}:{zarr_uri.split('#')[1]}"
        assert raster_and_zarr_are_equal(netcdf_uri, zarr_uri)


def test_zarr_already_exists(tmp_raster, tmp_storage_path):
    """zarr not created if already exists."""
    uris = raster_to_zarr(tmp_raster, tmp_storage_path)
    assert len(uris) == 1
    protocol, root, group = uri_split(uris[0])
    r = S3Path(f"/{root}") if protocol == "s3" else Path(root)
    assert zarr_exists(r, group) is True
    uris2 = raster_to_zarr(tmp_raster, tmp_storage_path)
    assert uris2 == []
    assert zarr_exists(r, group) is True


@pytest.mark.filterwarnings("ignore:Dataset has no geotransform set.")
def test_raster_with_no_dataset(tmp_empty_dataset):
    with pytest.raises(ValueError):
        get_rasterio_datasets(tmp_empty_dataset)


def _reproject_tif(tif, out_dir, crs=None, resolution=None):
    """Warp a tif with rasterio."""
    with rasterio.open(tif.as_uri()) as src:
        dst_crs = crs or src.crs
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=resolution,
        )
        meta = src.meta.copy()
        meta.update(
            {"crs": dst_crs, "transform": transform, "width": width, "height": height}
        )

        out = out_dir / f"{tif.stem}_reprojected.tif"
        with rasterio.open(out.as_uri(), "w", **meta) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=crs,
                    resampling=Resampling.nearest,
                )
    return out


_proj_params = [
    {"crs": "epsg:4326"},
    {"resolution": [100, 100]},
    {"crs": "epsg:4326", "resolution": [0.01, 0.01]},
]


@pytest.mark.parametrize("proj_params", _proj_params)
def test_reproject(ls5_dataset_path, tmp_path, proj_params):
    """Test reprojection."""
    ls5_tif = (
        ls5_dataset_path
        / "LS5_TM_NBAR_P54_GANBAR01-002_090_084_19920323"
        / "scene01"
        / "LS5_TM_NBAR_P54_GANBAR01-002_090_084_19920323_B10.tif"
    )
    assert ls5_tif.exists()
    out_dir = tmp_path / "tif"
    out_dir.mkdir()

    tif_reproj = _reproject_tif(ls5_tif, out_dir=out_dir, **proj_params,)

    uris = raster_to_zarr(ls5_tif, tmp_path / "zarr", **proj_params)
    assert len(uris) == 1
    assert raster_and_zarr_are_equal(tif_reproj.as_uri(), uris[0])
